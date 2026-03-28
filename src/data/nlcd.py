"""
nlcd.py — Annual NLCD loading, resampling, and masking.

Uses the USGS Annual NLCD product, which provides a land cover map for every
year from 1985 onward. Because coverage is annual, there is no need for
nearest-year approximation — each Landsat scene year maps directly to an
NLCD file of the same year.

Responsibilities:
  1. Load the Annual NLCD GeoTIFF for a given year.
  2. Reproject and resample to TARGET_CRS / TARGET_RESOLUTION.
  3. Produce two outputs consumed by downstream modules:
       - A boolean exclusion mask (True = exclude pixel) for predominantly
         coniferous pixels (NLCD class 42), which show no fall transition signal.
       - A continuous "deciduous_fraction" channel [0, 1] used as a model
         input, encoding forest composition per pixel.

Expected file layout:
    NLCD_RAW/
        {year}.tif      e.g. 2015.tif, 2016.tif, ...

The simple {year}.tif convention is recommended: download the Annual NLCD
files and rename them on ingest. If you prefer to keep original USGS filenames
(e.g. Annual_NLCD_LndCov_2015_CU_C2V0.tif), set NLCD_FILENAME_PATTERN in
config.py and update _find_nlcd_file accordingly — the class logic is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

from config import (
    NLCD_DECIDUOUS_HIGH,
    NLCD_EVERGREEN_HIGH,
    NLCD_INCLUDE_CLASSES,
    NLCD_MIXED_FOREST,
    NLCD_RAW,
    TARGET_CRS,
    TARGET_RESOLUTION,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mapping from NLCD class → approximate deciduous fraction.
# Used to generate the continuous deciduous_fraction input channel.
# Classes not listed here are treated as 0.0 (non-forest or unknown).
NLCD_DECIDUOUS_FRACTION = {
    NLCD_DECIDUOUS_HIGH: 1.0,   # >80% deciduous
    NLCD_MIXED_FOREST:   0.5,   # 20–80% mixed — midpoint as prior
    NLCD_EVERGREEN_HIGH: 0.0,   # >80% conifer
}


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _find_nlcd_file(year: int, root: Path = NLCD_RAW) -> Path:
    """Locate the Annual NLCD GeoTIFF for a given year.

    Looks for {year}.tif first (recommended simple convention), then falls
    back to the original USGS Annual NLCD filename pattern so that files
    can be used without renaming if preferred.

    Raises FileNotFoundError with a clear message if no file is found.
    """
    candidates = [
        root / f"{year}.tif",
        # Original USGS Annual NLCD filename pattern
        *list(root.glob(f"Annual_NLCD_LndCov_{year}_*.tif")),
    ]
    for path in candidates:
        if isinstance(path, Path) and path.exists():
            return path

    raise FileNotFoundError(
        f"No Annual NLCD file found for year {year} in {root}.\n"
        f"Expected '{year}.tif' or 'Annual_NLCD_LndCov_{year}_*.tif'.\n"
        f"Download from https://www.mrlc.gov/data and rename to '{year}.tif'."
    )


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class NLCDLayer:
    """Wraps a single Annual NLCD GeoTIFF and provides reprojected outputs.

    Since the Annual NLCD product has one file per year, construction is simply
    NLCDLayer(year) — no nearest-year approximation is needed.

    Attributes:
        year:  The land cover year this layer represents.
        path:  Path to the source GeoTIFF.

    Example:
        nlcd = NLCDLayer(2018)
        mask = nlcd.exclusion_mask(tile_transform, (H, W))
        frac = nlcd.deciduous_fraction(tile_transform, (H, W))
    """

    def __init__(self, year: int, path: Optional[Path] = None):
        self.year = year
        self.path = path or _find_nlcd_file(year)

    # ------------------------------------------------------------------
    # Reading and reprojection
    # ------------------------------------------------------------------

    def _read_raw(self) -> tuple[np.ndarray, dict]:
        """Read the NLCD raster at native resolution. Returns (data, profile)."""
        with rasterio.open(self.path) as src:
            data    = src.read(1)        # uint8 class codes
            profile = src.profile.copy()
        return data, profile

    def reproject_to_target(
        self,
        dst_transform=None,
        dst_shape: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """Reproject the NLCD raster to TARGET_CRS at TARGET_RESOLUTION.

        Args:
            dst_transform: If provided, snap the output exactly to this grid
                           (use the Landsat tile's output transform to ensure
                           pixel-perfect alignment between layers).
            dst_shape:     (height, width) of the output grid. Required when
                           dst_transform is provided.

        Returns:
            (H, W) uint8 array of NLCD class codes reprojected to TARGET_CRS.

        Note:
            Nearest-neighbour resampling is used unconditionally — NLCD contains
            categorical class codes and any interpolation would produce nonsense.
        """
        data, profile = self._read_raw()

        src_crs       = profile["crs"]
        src_transform = profile["transform"]
        src_h, src_w  = data.shape

        if dst_transform is None:
            dst_transform, dst_w, dst_h = calculate_default_transform(
                src_crs,
                TARGET_CRS,
                src_w,
                src_h,
                resolution=TARGET_RESOLUTION,
            )
        else:
            if dst_shape is None:
                raise ValueError(
                    "dst_shape must be provided when dst_transform is given."
                )
            dst_h, dst_w = dst_shape

        dst = np.zeros((dst_h, dst_w), dtype=np.uint8)

        reproject(
            source=data,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.nearest,
        )

        return dst

    # ------------------------------------------------------------------
    # Derived products
    # ------------------------------------------------------------------

    def exclusion_mask(
        self,
        dst_transform=None,
        dst_shape: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """Boolean mask (H, W): True = exclude pixel from training/inference.

        Excludes everything except deciduous forest (41) and mixed forest (43).
        Urban, cropland, evergreen forest, water, wetlands etc. are all excluded.
        """
        classes = self.reproject_to_target(dst_transform, dst_shape)
        include = np.zeros(classes.shape, dtype=bool)
        for cls in NLCD_INCLUDE_CLASSES:
            include |= (classes == cls)
        return ~include  # invert: True where we should EXCLUDE

    def deciduous_fraction(
        self,
        dst_transform=None,
        dst_shape: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """Float32 (H, W) array of deciduous fraction in [0, 1].

        Values:
            1.0 — predominantly deciduous (NLCD 41, >80% deciduous)
            0.5 — mixed forest           (NLCD 43, 20–80% deciduous)
            0.0 — predominantly conifer  (NLCD 42) or non-forest

        Passed as an explicit input channel so the model can condition its
        spectral interpretation on forest composition.
        """
        classes = self.reproject_to_target(dst_transform, dst_shape)
        out = np.zeros(classes.shape, dtype=np.float32)
        for cls, frac in NLCD_DECIDUOUS_FRACTION.items():
            out[classes == cls] = frac
        return out

    def __repr__(self) -> str:
        return f"NLCDLayer(year={self.year}, path={self.path.name})"