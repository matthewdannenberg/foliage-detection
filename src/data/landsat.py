"""
landsat.py — Landsat ARD tile processing: cloud masking, reflectance
conversion, spectral index computation, and resolution downsampling.

This module provides the core spectral processing logic used by stac.py
(which handles S3 streaming) and build_patches.py (which reads processed
GeoTIFFs). It does not handle file I/O or tile discovery — those concerns
live in stac.py and the preprocessing scripts respectively.

Key exports used by other modules:
    _bit_mask          — QA bit masking (used by stac.py)
    _reproject_array   — Resolution downsampling (used by stac.py)
    compute_evi2       — Spectral index (used by stac.py)
    compute_ndii       — Spectral index (used by stac.py)
    compute_ndvi       — Spectral index (used by stac.py)
    LandsatTile        — Local file processing (used by build_patches.py
                         for tile ID parsing from processed stack filenames)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

from config import (
    LANDSAT_BANDS,
    QA_MASK_BITS,
    SR_OFFSET,
    SR_SCALE,
    TARGET_CRS,
    TARGET_RESOLUTION,
)


# ---------------------------------------------------------------------------
# QA / cloud masking
# ---------------------------------------------------------------------------

def _bit_mask(qa: np.ndarray, bits: list[int]) -> np.ndarray:
    """Boolean mask: True wherever any of the given QA bit positions are set."""
    combined = 0
    for b in bits:
        combined |= (1 << b)
    return (qa & combined).astype(bool)


# ---------------------------------------------------------------------------
# Reprojection / downsampling
# ---------------------------------------------------------------------------

def _reproject_array(
    data: np.ndarray,
    src_transform,
    src_crs: str,
    target_crs: str = TARGET_CRS,
    target_res: float = TARGET_RESOLUTION,
    resampling: Resampling = Resampling.bilinear,
) -> tuple[np.ndarray, object]:
    """Resample a (C, H, W) float32 array to target_res in target_crs.

    Since TARGET_CRS == ARD source CRS (both EPSG:5070), this performs
    resolution downsampling only (~30m → 250m) with no coordinate
    transformation.

    NaN pixels are handled via a sentinel value for robust behaviour
    across rasterio versions.

    Returns:
        (resampled_array, affine_transform_of_output_grid)
    """
    _SENTINEL = -9999.0
    bands, src_h, src_w = data.shape

    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs,
        target_crs,
        src_w,
        src_h,
        resolution=target_res,
        left=src_transform.c,
        top=src_transform.f,
        right=src_transform.c + src_transform.a * src_w,
        bottom=src_transform.f + src_transform.e * src_h,
    )

    dst = np.full((bands, dst_h, dst_w), np.nan, dtype=np.float32)

    for i in range(bands):
        src_band = np.where(np.isnan(data[i]), _SENTINEL, data[i])
        reproject(
            source=src_band,
            destination=dst[i],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            src_nodata=_SENTINEL,
            dst_nodata=_SENTINEL,
            resampling=resampling,
        )
        dst[i][dst[i] == _SENTINEL] = np.nan

    return dst, dst_transform


# ---------------------------------------------------------------------------
# Spectral index computation
# ---------------------------------------------------------------------------

def compute_evi2(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """EVI2 = 2.5 * (NIR - Red) / (NIR + 2.4 * Red + 1)

    Two-band EVI; more sensitive to canopy change than NDVI at moderate-to-high
    biomass. NaN-safe.
    """
    denom = nir + 2.4 * red + 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        evi2 = 2.5 * (nir - red) / denom
    return np.where(np.abs(denom) < 1e-6, np.nan, evi2).astype(np.float32)


def compute_ndii(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """NDII = (NIR - SWIR1) / (NIR + SWIR1)

    Tracks canopy water content; sensitive to early senescence before the
    visible color change that NDVI/EVI2 detect.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        ndii = (nir - swir1) / (nir + swir1)
    return np.where((nir + swir1) == 0, np.nan, ndii).astype(np.float32)


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red)"""
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red)
    return np.where((nir + red) == 0, np.nan, ndvi).astype(np.float32)


# ---------------------------------------------------------------------------
# LandsatTile — tile ID parsing and local file processing
# ---------------------------------------------------------------------------

@dataclass
class LandsatTile:
    """Represents a Landsat Collection 2 ARD tile, either from a local
    directory or identified by its tile ID string.

    Primary use in the current pipeline: parsing tile IDs from processed
    stack filenames in build_patches.py. The from_dir constructor and
    reflectance/stack methods support local file processing if needed,
    but the main preprocessing path uses stac.py for S3 streaming.

    Attributes:
        tile_id:  Full ARD tile ID string.
        date:     Acquisition date as 'YYYY-MM-DD'.
        year:     Acquisition year (int).
        sensor:   Sensor code: 'LT04', 'LT05', 'LE07', 'LC08', or 'LC09'.
        h_tile:   Horizontal ARD tile index string, e.g. '028'.
        v_tile:   Vertical ARD tile index string, e.g. '004'.
        tile_dir: Path to raw tile directory (optional — only set when
                  constructed via from_dir).
    """

    tile_id:  str
    date:     str
    year:     int
    sensor:   str
    h_tile:   str
    v_tile:   str
    tile_dir: Optional[Path] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_tile_id(cls, tile_id: str) -> "LandsatTile":
        """Construct from a tile ID string alone (no directory needed).

        Useful for parsing metadata from processed stack filenames without
        requiring the raw tile directory to exist.
        """
        sensor, h_tile, v_tile, date_str = cls._parse_tile_id(tile_id)
        return cls(
            tile_id=tile_id,
            date=f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
            year=int(date_str[:4]),
            sensor=sensor,
            h_tile=h_tile,
            v_tile=v_tile,
        )

    @classmethod
    def from_dir(cls, tile_dir: Path | str) -> "LandsatTile":
        """Construct from a raw tile directory."""
        tile_dir = Path(tile_dir)
        if not tile_dir.is_dir():
            raise FileNotFoundError(f"Tile directory not found: {tile_dir}")
        tile = cls.from_tile_id(tile_dir.name)
        tile.tile_dir = tile_dir
        return tile

    @staticmethod
    def _parse_tile_id(tile_id: str) -> tuple[str, str, str, str]:
        """Parse sensor, h_tile, v_tile, date_str from an ARD tile ID.

        Format: {SENSOR}_CU_{HHH}{VVV}_{YYYYMMDD}_{proddate}_{coll}_{cat}
        Returns: (sensor, h_tile, v_tile, date_str)
        """
        m = re.match(r"^(L[TCE][0-9]{2})_CU_(\d{3})(\d{3})_(\d{8})_", tile_id)
        if not m:
            raise ValueError(
                f"Cannot parse ARD tile ID '{tile_id}'. "
                "Expected: SENSOR_CU_HHHVVV_YYYYMMDD_..."
            )
        return m.group(1), m.group(2), m.group(3), m.group(4)

    # ------------------------------------------------------------------
    # Local file processing (requires tile_dir to be set)
    # ------------------------------------------------------------------

    def _require_dir(self) -> Path:
        if self.tile_dir is None:
            raise RuntimeError(
                "tile_dir is not set. Use LandsatTile.from_dir() to construct "
                "a tile with local file access."
            )
        return self.tile_dir

    def _band_path(self, band_number: int) -> Path:
        d = self._require_dir()
        candidates = list(d.glob(f"*_SR_B{band_number}.TIF"))
        if not candidates:
            raise FileNotFoundError(f"SR Band {band_number} not found in {d}")
        return candidates[0]

    def _qa_path(self) -> Path:
        d = self._require_dir()
        candidates = list(d.glob("*_QA_PIXEL.TIF"))
        if not candidates:
            raise FileNotFoundError(f"QA_PIXEL file not found in {d}")
        return candidates[0]

    def cloud_mask(self) -> np.ndarray:
        """Boolean mask (H, W): True = exclude pixel."""
        with rasterio.open(self._qa_path()) as src:
            qa = src.read(1)
        return _bit_mask(qa, QA_MASK_BITS)

    def valid_pixel_fraction(self) -> float:
        """Fraction of pixels not excluded by the cloud mask."""
        return float(1.0 - self.cloud_mask().mean())

    def reflectance(self) -> tuple[np.ndarray, dict]:
        """Load all 6 bands as surface reflectance floats.

        Returns:
            refl:    (6, H, W) float32. Masked pixels → NaN.
                     Band order: blue=0, green=1, red=2, nir=3, swir1=4, swir2=5.
            profile: rasterio profile at native ARD resolution.
        """
        mask    = self.cloud_mask()
        arrays  = []
        profile = None

        for _name, number in LANDSAT_BANDS.items():
            with rasterio.open(self._band_path(number)) as src:
                dn = src.read(1).astype(np.float32)
                if profile is None:
                    profile = src.profile.copy()

            refl = np.clip(dn * SR_SCALE + SR_OFFSET, 0.0, 1.0)
            refl[mask] = np.nan
            arrays.append(refl)

        return np.stack(arrays, axis=0), profile

    def indices(self, refl: np.ndarray) -> np.ndarray:
        """Compute EVI2, NDII, NDVI from a (6, H, W) reflectance array."""
        red, nir, swir1 = refl[2], refl[3], refl[4]
        return np.stack([
            compute_evi2(nir, red),
            compute_ndii(nir, swir1),
            compute_ndvi(nir, red),
        ], axis=0)

    def stack(self) -> tuple[np.ndarray, object]:
        """9-channel spectral stack downsampled to TARGET_RESOLUTION (250m)."""
        refl, profile = self.reflectance()
        combined = np.concatenate([refl, self.indices(refl)], axis=0)
        return _reproject_array(
            data=combined,
            src_transform=profile["transform"],
            src_crs=TARGET_CRS,
        )

    def __repr__(self) -> str:
        return (
            f"LandsatTile(sensor={self.sensor}, "
            f"tile=h{self.h_tile}v{self.v_tile}, date={self.date})"
        )