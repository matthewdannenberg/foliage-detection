"""
build_patches.py — Extract labelled patches from processed Landsat stacks.

Combines two label sources:

  1. OBSERVER-BASED PATCHES
     Iterates over processed Landsat tiles. For each tile, finds all
     observations within ±5 days of its acquisition date, re-consolidates
     them by location (plurality vote within the time window), and extracts
     one patch per consolidated location. This tiles→observations approach
     ensures multiple nearby observations are combined into a single
     high-quality patch rather than producing duplicate noisy patches.

  2. SYNTHETIC PATCHES
     Generates additional no_transition labels from early August tiles
     (Aug 1–20) and late labels from mid-to-late November tiles (Nov 10–30),
     where stage assignment is unambiguous regardless of observer data.
     Synthetic patches are capped to avoid overwhelming the minority classes
     (early, peak) which come only from observer data.

All patches are combined into a single HDF5 archive with per-sample
confidence scores, years, and label sources recorded.

Static input channels (elevation, slope, aspect, deciduous_fraction) are
aligned to each tile's 250m grid via on-the-fly reprojection. The DEM and
NLCD are read once per tile and cached within the processing loop.

Output:
    data/processed/patches/patches.h5
    data/processed/norm_stats.json   (per-channel mean/std, training years only)

Usage:
    python scripts/build_patches.py 
    python scripts/build_patches.py --dem path/to/dem.tif
    python scripts/build_patches.py --dem path/to/dem.tif --max-synthetic 3000
    python scripts/build_patches.py --dem path/to/dem.tif --no-synthetic
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from pyproj import Transformer
from tqdm import tqdm

from config import (
    ALL_CHANNEL_NAMES,
    CONSOLIDATION_COORD_DECIMALS,
    DATA_RAW,
    DATA_PROCESSED,
    MAX_NAN_FRACTION,
    MAX_PATCHES_PER_SITE,
    NORM_STATS_PATH,
    NUM_CHANNELS,
    NUM_CLASSES,
    OBSERVATIONS_DIR,
    PATCH_MIN_DECIDUOUS_FRACTION,
    PATCH_SIZE,
    PATCHES_DIR,
    RANDOM_SEED,
    STAGE_NAMES,
    STAGES,
    SYNTHETIC_LATE_CONFIDENCE,
    SYNTHETIC_LATE_WINDOW,
    SYNTHETIC_NO_TRANSITION_CONFIDENCE,
    SYNTHETIC_NO_TRANSITION_WINDOW,
    TARGET_CRS,
    TARGET_RESOLUTION,
    TILE_OBS_WINDOW_DAYS,
    TRAIN_YEARS,
    VAL_YEARS,
    TEST_YEARS,
)
from data.nlcd import NLCDLayer
from process_observations import consolidate

PROCESSED_LANDSAT = DATA_PROCESSED / "landsat"
DEM_PATH = DATA_RAW / "ned" / "northeast_dem.tif"


# ---------------------------------------------------------------------------
# Static layer helpers
# ---------------------------------------------------------------------------

def _load_dem_for_grid(
    dem_path: Path,
    dst_transform,
    dst_shape: tuple[int, int],
) -> np.ndarray:
    """Reproject DEM to the tile's 250m output grid. Returns (H, W) float32."""
    dst_h, dst_w = dst_shape
    dst = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
    with rasterio.open(dem_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.bilinear,
        )
    return dst


def _slope_aspect(
    elevation: np.ndarray,
    resolution: float = TARGET_RESOLUTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute slope (degrees) and aspect (degrees, 0=N CW) from a DEM."""
    dz_dy, dz_dx = np.gradient(elevation, resolution)
    slope  = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))).astype(np.float32)
    aspect = (np.degrees(np.arctan2(-dz_dx, dz_dy)) % 360.0).astype(np.float32)
    return slope, aspect


def _build_cube(
    spectral: np.ndarray,
    elevation: np.ndarray,
    slope: np.ndarray,
    aspect: np.ndarray,
    deciduous_fraction: np.ndarray,
) -> np.ndarray:
    """Stack all channels into (NUM_CHANNELS, H, W) float32 cube."""
    return np.concatenate([
        spectral,                       # 0–8:  6 bands + evi2/ndii/ndvi
        elevation[np.newaxis],          # 9:    elevation
        slope[np.newaxis],              # 10:   slope
        aspect[np.newaxis],             # 11:   aspect
        deciduous_fraction[np.newaxis], # 12:   deciduous fraction
    ], axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Patch validity
# ---------------------------------------------------------------------------

def _is_valid_patch(
    cube: np.ndarray,
    row: int,
    col: int,
    half: int,
    exclusion_mask: np.ndarray,
) -> bool:
    """Return True if a patch centred at (row, col) is usable."""
    _, H, W = cube.shape
    if row < half or row >= H - half or col < half or col >= W - half:
        return False
    if exclusion_mask[row, col]:
        return False
    # Reject if any spectral channel (0–8) is NaN (cloudy/snow/masked)
    patch_spectral = cube[:9, row - half: row + half, col - half: col + half]
    if np.isnan(patch_spectral).any():
        return False
    return True


def _extract_patch(cube: np.ndarray, row: int, col: int, half: int) -> np.ndarray:
    return cube[:, row - half: row + half, col - half: col + half].copy()


def _fill_nan(patch: np.ndarray,
              max_nan_fraction: float = MAX_NAN_FRACTION) -> np.ndarray | None:
    """Fill NaN values in a (C, H, W) patch with per-channel mean of valid pixels.

    Applied to observer patches where context pixels may be cloud-masked.
    Returns None if any spectral channel (0–8) has more than max_nan_fraction
    of its pixels as NaN — these patches are too corrupted to be useful and
    would otherwise produce misleading fill values or trigger the 'mean of
    empty slice' warning.

    For channels where some but not too many pixels are NaN, fills with the
    per-channel mean of the valid pixels in that patch.
    """
    patch = patch.copy()
    n_pixels = patch.shape[1] * patch.shape[2]

    for c in range(patch.shape[0]):
        nan_mask = np.isnan(patch[c])
        if not nan_mask.any():
            continue
        nan_frac = nan_mask.sum() / n_pixels
        # Reject the whole patch if any spectral channel is too sparse
        if c < 9 and nan_frac > max_nan_fraction:
            return None
        valid_mean = np.nanmean(patch[c])
        patch[c][nan_mask] = valid_mean if np.isfinite(valid_mean) else 0.0

    return patch


# ---------------------------------------------------------------------------
# Module-level coordinate transformers — created once, reused everywhere
# ---------------------------------------------------------------------------
_WGS84_TO_ALBERS = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
_ALBERS_TO_WGS84 = Transformer.from_crs(TARGET_CRS, "EPSG:4326", always_xy=True)


# ---------------------------------------------------------------------------
# Coordinate → pixel conversion
# ---------------------------------------------------------------------------

def _lonlat_to_rowcol(
    lon: float,
    lat: float,
    transform,
) -> tuple[int, int]:
    """Convert WGS84 lon/lat to pixel (row, col) on the 250m grid."""
    x, y = _WGS84_TO_ALBERS.transform(lon, lat)
    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)
    return row, col


def _rowcol_to_lonlat(
    row: int,
    col: int,
    transform,
) -> tuple[float, float]:
    """Convert pixel (row, col) on the 250m grid to WGS84 (lon, lat).

    Inverse of _lonlat_to_rowcol. Returns the center of the pixel.
    """
    x = transform.c + (col + 0.5) * transform.a
    y = transform.f + (row + 0.5) * transform.e
    lon, lat = _ALBERS_TO_WGS84.transform(x, y)
    return lon, lat


# ---------------------------------------------------------------------------
# Tile index
# ---------------------------------------------------------------------------

def _index_processed_tiles() -> dict[str, list[Path]]:
    """Return a dict mapping date string → list of stack GeoTIFF paths."""
    index: dict[str, list[Path]] = defaultdict(list)
    for path in PROCESSED_LANDSAT.rglob("*_stack.tif"):
        # Tile ID: LC08_CU_028004_20190915_...  date at position [3]
        parts = path.stem.replace("_stack", "").split("_")
        if len(parts) >= 4:
            raw = parts[3]   # YYYYMMDD
            if len(raw) == 8:
                date_str = f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
                index[date_str].append(path)
    return index


def _observations_for_tile(
    tile_date: date,
    observations: pd.DataFrame,
    window_days: int = TILE_OBS_WINDOW_DAYS,
) -> pd.DataFrame:
    """Return all observations within window_days of tile_date.

    Also adds a 'stage_int' column (integer stage index) needed by
    consolidate(), and a 'confidence' column if not already present.
    """
    obs_dates = pd.to_datetime(observations["date"]).dt.date
    mask = obs_dates.apply(lambda d: abs((d - tile_date).days) <= window_days)
    subset = observations[mask].copy()
    if subset.empty:
        return subset

    # consolidate() expects stage_int — the observations CSV stores
    # stage as an integer already, so just rename it
    subset["stage_int"] = subset["stage"].astype(int)
    return subset


# ---------------------------------------------------------------------------
# Observer-based patch extraction  (tiles → observations)
# ---------------------------------------------------------------------------

def extract_observer_patches(
    observations: pd.DataFrame,
    tile_index: dict[str, list[Path]],
    dem_path: Path,
    nlcd_cache: dict[int, NLCDLayer],
) -> list[dict]:
    """Extract patches by iterating tiles and matching nearby observations.

    For each processed tile:
      1. Find all observations within ±TILE_OBS_WINDOW_DAYS days.
      2. Re-consolidate by snapped location (plurality vote within window).
      3. Extract one patch per consolidated location.

    Filters applied per patch:
      - center pixel must pass NLCD exclusion mask
      - center pixel spectral channels must not be NaN
      - Patch-level deciduous fraction >= PATCH_MIN_DECIDUOUS_FRACTION
      - Each USA-NPN site (snapped lat/lon) capped at MAX_PATCHES_PER_SITE total
      - PhenoCam sites are uncapped — temporal diversity within a site is the value

    Returns a list of record dicts with keys:
        patch, stage, confidence, year, source, label_source
    """
    records = []
    half = PATCH_SIZE // 2
    skipped = Counter()
    tiles_with_obs = 0

    # Track patches per site to enforce MAX_PATCHES_PER_SITE cap
    # Key: (lat_snapped, lon_snapped) tuple
    site_patch_counts: Counter = Counter()

    # Pre-index observations by year for O(1) year lookup instead of
    # scanning all observations for every tile
    obs_by_year: dict[int, pd.DataFrame] = {}
    for yr, grp in observations.groupby(observations["date"].apply(lambda d: d.year)):
        obs_by_year[yr] = grp.reset_index(drop=True)

    # Cache DEM and NLCD arrays keyed by (tile_grid_id, year) so tiles at the
    # same ARD grid position (same transform/shape) sharing data across dates
    # don't reproject the DEM or reproject NLCD repeatedly.
    # tile_grid_id is derived from the transform tuple — same ARD H/V position
    # always has the same transform.
    dem_grid_cache:   dict[tuple, np.ndarray] = {}  # transform_key → elevation
    nlcd_grid_cache:  dict[tuple, tuple]      = {}  # (transform_key, year) → (excl, decid)

    # Collect all unique tile paths sorted by date for clean progress display
    all_tile_paths = sorted(
        {p for paths in tile_index.values() for p in paths},
        key=lambda p: p.stem,
    )

    for tile_path in tqdm(all_tile_paths, desc="Observer patches", unit="tile"):
        # Parse tile date
        parts = tile_path.stem.replace("_stack", "").split("_")
        if len(parts) < 4 or len(parts[3]) != 8:
            continue
        raw = parts[3]
        try:
            tile_date = date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
        except ValueError:
            continue

        year = tile_date.year

        # Find observations near this tile — use year-indexed subset
        year_obs = obs_by_year.get(year, pd.DataFrame())
        if year_obs.empty:
            continue
        tile_obs = _observations_for_tile(tile_date, year_obs)
        if tile_obs.empty:
            continue

        # Read tile metadata and spectral data in a single open
        with rasterio.open(tile_path) as src:
            bounds    = src.bounds
            transform = src.transform
            H, W      = src.height, src.width
            spectral  = src.read().astype(np.float32)   # (9, H, W)

        # Convert tile bounds to WGS84 using module-level transformer
        left_lon,  bottom_lat = _ALBERS_TO_WGS84.transform(bounds.left,  bounds.bottom)
        right_lon, top_lat    = _ALBERS_TO_WGS84.transform(bounds.right, bounds.top)

        spatial_mask = (
            (tile_obs["latitude"]  >= bottom_lat) &
            (tile_obs["latitude"]  <= top_lat)    &
            (tile_obs["longitude"] >= left_lon)   &
            (tile_obs["longitude"] <= right_lon)
        )
        tile_obs = tile_obs[spatial_mask]
        if tile_obs.empty:
            continue

        # Re-consolidate within this tile's spatiotemporal window
        consolidated = consolidate(tile_obs)
        if consolidated.empty:
            continue

        tiles_with_obs += 1

        # Cache key based on transform (uniquely identifies the tile grid)
        transform_key = (transform.c, transform.f, transform.a, transform.e, H, W)

        # DEM — reproject once per unique grid position
        if transform_key not in dem_grid_cache:
            dem_grid_cache[transform_key] = _load_dem_for_grid(
                dem_path, transform, (H, W)
            )
        elevation = dem_grid_cache[transform_key]
        slope, aspect = _slope_aspect(elevation)

        # NLCD — reproject once per (grid position, year)
        nlcd_key = (transform_key, year)
        if nlcd_key not in nlcd_grid_cache:
            nlcd = nlcd_cache.setdefault(year, NLCDLayer(year))
            nlcd_grid_cache[nlcd_key] = (
                nlcd.exclusion_mask(transform, (H, W)),
                nlcd.deciduous_fraction(transform, (H, W)),
            )
        exclusion_mask, deciduous_fraction = nlcd_grid_cache[nlcd_key]

        cube = _build_cube(spectral, elevation, slope, aspect, deciduous_fraction)

        # Extract one patch per consolidated observation location
        for _, obs_row in consolidated.iterrows():
            stage      = int(obs_row["stage"])
            confidence = float(obs_row["confidence"])
            source     = str(obs_row.get("source", ""))

            px_row, px_col = _lonlat_to_rowcol(
                float(obs_row["longitude"]),
                float(obs_row["latitude"]),
                transform,
            )

            _, H, W = cube.shape
            in_bounds = (px_row >= half and px_row < H - half and
                         px_col >= half and px_col < W - half)

            if not in_bounds:
                skipped["out_of_bounds"] += 1
                continue

            # Patch-level deciduous fraction check. The center-pixel NLCD
            # filter passes sites like isolated urban parks where the
            # surrounding patch is dominated by impervious surfaces. Require
            # that at least PATCH_MIN_DECIDUOUS_FRACTION of the patch is
            # classified as deciduous or mixed forest.
            decid_patch = deciduous_fraction[
                px_row - half: px_row + half,
                px_col - half: px_col + half,
            ]
            if decid_patch.mean() < PATCH_MIN_DECIDUOUS_FRACTION:
                skipped["low_deciduous_fraction"] += 1
                continue

            # Per-site cap — prevent any single NPN monitoring site from
            # contributing more than MAX_PATCHES_PER_SITE patches total.
            # PhenoCam sites are exempt: each day represents a genuinely
            # different spectral state along the seasonal trajectory, so
            # temporal diversity within a site is the point, not a problem.
            site_key = (
                round(float(obs_row["latitude"]),  CONSOLIDATION_COORD_DECIMALS),
                round(float(obs_row["longitude"]), CONSOLIDATION_COORD_DECIMALS),
                source,
            )
            if source == "USA-NPN" and site_patch_counts[site_key] >= MAX_PATCHES_PER_SITE:
                skipped["site_cap_exceeded"] += 1
                continue
            site_patch_counts[site_key] += 1

            patch = _fill_nan(_extract_patch(cube, px_row, px_col, half))
            if patch is None:
                skipped["excessive_nan"] += 1
                site_patch_counts[site_key] -= 1  # undo the increment
                continue
            patch_lon, patch_lat = _rowcol_to_lonlat(px_row, px_col, transform)
            records.append({
                "patch":        patch,
                "stage":        stage,
                "confidence":   confidence,
                "year":         year,
                "date":         tile_date.isoformat(),
                "source":       source,
                "label_source": "observer",
                "latitude":     patch_lat,
                "longitude":    patch_lon,
            })

            # Neighbourhood augmentation — extract one patch per adjacent pixel
            # (N, NE, E, SE, S, SW, W, NW). Each shift is one 250m pixel,
            # so the centre moves to a neighbouring location while the label
            # is assumed to hold. Confidence is multiplied by 0.95 to reflect
            # the small added uncertainty from the spatial offset.
            for dr, dc in [(-1, 0), (-1, 1), (0, 1), (1, 1),
                           (1, 0), (1, -1), (0, -1), (-1, -1)]:
                nr, nc = px_row + dr, px_col + dc
                # Check shifted centre is within tile bounds
                if not (half <= nr < H - half and half <= nc < W - half):
                    continue
                # Check deciduous fraction at shifted location
                decid_shifted = deciduous_fraction[
                    nr - half: nr + half,
                    nc - half: nc + half,
                ]
                if decid_shifted.mean() < PATCH_MIN_DECIDUOUS_FRACTION:
                    continue
                shifted_patch = _fill_nan(_extract_patch(cube, nr, nc, half))
                if shifted_patch is None:
                    continue
                shift_lon, shift_lat = _rowcol_to_lonlat(nr, nc, transform)
                records.append({
                    "patch":        shifted_patch,
                    "stage":        stage,
                    "confidence":   round(confidence * 0.95, 4),
                    "year":         year,
                    "date":         tile_date.isoformat(),
                    "source":       source,
                    "label_source": "observer_shifted",
                    "latitude":     shift_lat,
                    "longitude":    shift_lon,
                })

    print(f"  Tiles with nearby observations: {tiles_with_obs}")
    print(f"  Observer patches extracted: {len(records)}")
    if skipped:
        print(f"  Skipped: {dict(skipped)}")
    return records


# ---------------------------------------------------------------------------
# Synthetic patch generation
# ---------------------------------------------------------------------------

def _tile_date(tile_path: Path) -> date | None:
    """Parse acquisition date from a processed stack filename."""
    parts = tile_path.stem.replace("_stack", "").split("_")
    if len(parts) >= 4:
        raw = parts[3]
        if len(raw) == 8:
            try:
                return date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
            except ValueError:
                pass
    return None


def _in_synthetic_window(
    tile_date: date,
    windows: list[tuple[int, int]],
) -> bool:
    """Return True if tile_date falls within a (month, day) window."""
    start_m, start_d = windows[0]
    end_m,   end_d   = windows[1]
    return (
        (tile_date.month, tile_date.day) >= (start_m, start_d) and
        (tile_date.month, tile_date.day) <= (end_m,   end_d)
    )


def generate_synthetic_patches(
    tile_index: dict[str, list[Path]],
    dem_path: Path,
    nlcd_cache: dict[int, NLCDLayer],
    max_per_class: int,
    years: list[int],
    rng: random.Random,
) -> list[dict]:
    """Generate synthetic no_transition and late patches from date windows.

    Randomly samples valid forest pixels from eligible tiles up to
    max_per_class patches per stage.

    Returns a list of record dicts.
    """
    half = PATCH_SIZE // 2
    records_by_stage: dict[int, list[dict]] = {
        STAGES["no_transition"]: [],
        STAGES["late"]:          [],
    }

    # DEM and NLCD caches keyed by grid position — same as observer loop
    dem_grid_cache:  dict[tuple, np.ndarray] = {}
    nlcd_grid_cache: dict[tuple, tuple]      = {}

    # Collect all eligible tiles per stage
    eligible: dict[int, list[Path]] = {
        STAGES["no_transition"]: [],
        STAGES["late"]:          [],
    }

    for date_str, paths in tile_index.items():
        td = date.fromisoformat(date_str)
        if td.year not in years:
            continue
        for path in paths:
            if _in_synthetic_window(td, SYNTHETIC_NO_TRANSITION_WINDOW):
                eligible[STAGES["no_transition"]].append(path)
            elif _in_synthetic_window(td, SYNTHETIC_LATE_WINDOW):
                eligible[STAGES["late"]].append(path)

    for stage_idx, stage_name in [
        (STAGES["no_transition"], "no_transition"),
        (STAGES["late"],          "late"),
    ]:
        confidence = (
            SYNTHETIC_NO_TRANSITION_CONFIDENCE
            if stage_name == "no_transition"
            else SYNTHETIC_LATE_CONFIDENCE
        )

        tile_paths = eligible[stage_idx]
        rng.shuffle(tile_paths)

        n_collected = 0
        for tile_path in tqdm(tile_paths, desc=f"Synthetic {stage_name}", unit="tile"):
            if n_collected >= max_per_class:
                break

            td = _tile_date(tile_path)
            if td is None:
                continue

            year = td.year

            with rasterio.open(tile_path) as src:
                spectral  = src.read().astype(np.float32)
                transform = src.transform
                H, W      = src.height, src.width

            transform_key = (transform.c, transform.f, transform.a, transform.e, H, W)

            if transform_key not in dem_grid_cache:
                dem_grid_cache[transform_key] = _load_dem_for_grid(
                    dem_path, transform, (H, W)
                )
            elevation = dem_grid_cache[transform_key]
            slope, aspect = _slope_aspect(elevation)

            nlcd_key = (transform_key, year)
            if nlcd_key not in nlcd_grid_cache:
                nlcd = nlcd_cache.setdefault(year, NLCDLayer(year))
                nlcd_grid_cache[nlcd_key] = (
                    nlcd.exclusion_mask(transform, (H, W)),
                    nlcd.deciduous_fraction(transform, (H, W)),
                )
            exclusion_mask, deciduous_fraction = nlcd_grid_cache[nlcd_key]

            cube = _build_cube(spectral, elevation, slope, aspect, deciduous_fraction)

            # Find all valid forest pixels in this tile
            valid_rows, valid_cols = np.where(
                ~exclusion_mask[half:-half, half:-half]
            )
            # Adjust indices for the half-border offset
            valid_rows = valid_rows + half
            valid_cols = valid_cols + half

            if len(valid_rows) == 0:
                continue

            # Shuffle and sample until we hit our per-tile contribution limit
            # (avoid over-representing any single tile)
            per_tile_limit = min(50, max_per_class - n_collected)
            indices = list(range(len(valid_rows)))
            rng.shuffle(indices)

            tile_count = 0
            for idx in indices:
                if tile_count >= per_tile_limit:
                    break
                if n_collected >= max_per_class:
                    break

                r, c = int(valid_rows[idx]), int(valid_cols[idx])
                if not _is_valid_patch(cube, r, c, half, exclusion_mask):
                    continue

                patch = _extract_patch(cube, r, c, half)
                patch_lon, patch_lat = _rowcol_to_lonlat(r, c, transform)
                records_by_stage[stage_idx].append({
                    "patch":        patch,
                    "stage":        stage_idx,
                    "confidence":   confidence,
                    "year":         year,
                    "date":         td.isoformat(),
                    "source":       "synthetic",
                    "label_source": f"synthetic_{stage_name}",
                    "latitude":     patch_lat,
                    "longitude":    patch_lon,
                })
                n_collected += 1
                tile_count  += 1

        print(f"  Synthetic {stage_name}: {n_collected} patches")

    return records_by_stage[STAGES["no_transition"]] + records_by_stage[STAGES["late"]]


# ---------------------------------------------------------------------------
# HDF5 writing
# ---------------------------------------------------------------------------

def write_hdf5(records: list[dict], out_path: Path) -> None:
    """Write all records to an HDF5 archive.

    Schema:
        /patches      (N, C, H, W) float32
        /labels       (N,)         int64
        /confidence   (N,)         float32
        /years        (N,)         int32
        /dates        (N,)         variable-length string  (ISO YYYY-MM-DD)
        /latitudes    (N,)         float32
        /longitudes   (N,)         float32
        /label_source (N,)         variable-length string
        /metadata     JSON string
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    N = len(records)

    patches      = np.stack([r["patch"]      for r in records]).astype(np.float32)
    labels       = np.array([r["stage"]      for r in records], dtype=np.int64)
    confidence   = np.array([r["confidence"] for r in records], dtype=np.float32)
    years        = np.array([r["year"]       for r in records], dtype=np.int32)
    latitudes    = np.array([r["latitude"]   for r in records], dtype=np.float32)
    longitudes   = np.array([r["longitude"]  for r in records], dtype=np.float32)
    dates        = [r["date"]                for r in records]
    label_source = [r["label_source"]        for r in records]

    metadata = json.dumps({
        "channel_names": ALL_CHANNEL_NAMES,
        "stage_names":   STAGE_NAMES,
        "patch_size":    PATCH_SIZE,
        "num_channels":  NUM_CHANNELS,
        "n_samples":     N,
    })

    with h5py.File(out_path, "w") as f:
        f.create_dataset(
            "patches", data=patches,
            chunks=(64, NUM_CHANNELS, PATCH_SIZE, PATCH_SIZE),
            compression="gzip", compression_opts=4,
        )
        f.create_dataset("labels",     data=labels)
        f.create_dataset("confidence", data=confidence)
        f.create_dataset("years",      data=years)
        f.create_dataset("latitudes",  data=latitudes)
        f.create_dataset("longitudes", data=longitudes)

        dt = h5py.special_dtype(vlen=str)

        ds = f.create_dataset("dates", (N,), dtype=dt)
        ds[:] = dates

        ds = f.create_dataset("label_source", (N,), dtype=dt)
        ds[:] = label_source

        f.create_dataset("metadata", data=metadata)

    print(f"[build] Wrote {N} patches → {out_path}")


# ---------------------------------------------------------------------------
# Normalisation statistics
# ---------------------------------------------------------------------------

def compute_norm_stats(hdf5_path: Path, train_years: list[int]) -> None:
    """Compute per-channel mean/std over training patches and save to JSON.

    Statistics computed only on training years to prevent leakage.
    Falls back to all patches if no training-year patches exist (e.g. during
    single-year test runs where the year is in val/test rather than train).
    """
    print("[build] Computing normalisation statistics from training patches ...")
    with h5py.File(hdf5_path, "r") as f:
        all_years     = f["years"][:]
        train_mask    = np.isin(all_years, train_years)
        n_train       = int(train_mask.sum())

        if n_train == 0:
            print(
                "[build] WARNING: No training-year patches found. "
                "Computing stats over all patches instead.\n"
                "  This is expected during single-year test runs.\n"
                "  Re-run with training years for production use."
            )
            train_patches = f["patches"][:]
        else:
            train_patches = f["patches"][train_mask]

    mean = train_patches.mean(axis=(0, 2, 3)).tolist()
    std  = train_patches.std(axis=(0, 2, 3)).tolist()

    NORM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NORM_STATS_PATH, "w") as f:
        json.dump({
            "mean":     mean,
            "std":      std,
            "channels": ALL_CHANNEL_NAMES,
            "n_train":  n_train,
        }, f, indent=2)

    print(f"[build] Norm stats saved → {NORM_STATS_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    all_years = sorted(set(TRAIN_YEARS + VAL_YEARS + TEST_YEARS))
    p = argparse.ArgumentParser(
        description="Build HDF5 patch archive from processed Landsat stacks."
    )
    p.add_argument(
        "--dem", type=Path, default=DEM_PATH,
        help="Path to the source DEM GeoTIFF (any CRS; will be reprojected).",
    )
    p.add_argument(
        "--observations", type=Path,
        default=OBSERVATIONS_DIR / "observations.csv",
        help="Path to the standardised observations CSV.",
    )
    p.add_argument(
        "--years", nargs="+", type=int, default=all_years,
        help="Years to include (default: all split years).",
    )
    p.add_argument(
        "--out", type=Path,
        default=PATCHES_DIR / "patches.h5",
        help="Output HDF5 path.",
    )
    p.add_argument(
        "--max-synthetic", type=int, default=None,
        help=(
            "Maximum synthetic patches per class (no_transition, late). "
            "Defaults to 5x the count of the minority observer class."
        ),
    )
    p.add_argument(
        "--no-synthetic", action="store_true",
        help="Skip synthetic patch generation entirely.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dem.exists():
        print(f"[error] DEM not found: {args.dem}")
        sys.exit(1)
    if not args.observations.exists():
        print(f"[error] Observations CSV not found: {args.observations}")
        print("  Run process_observations.py first.")
        sys.exit(1)

    rng = random.Random(RANDOM_SEED)

    # Load observations filtered to requested years
    print(f"[build] Loading observations from {args.observations} ...")
    obs = pd.read_csv(args.observations)
    obs["date"] = pd.to_datetime(obs["date"]).dt.date
    obs_year = obs["date"].apply(lambda d: d.year)
    obs = obs[obs_year.isin(args.years)].reset_index(drop=True)
    print(f"  {len(obs)} observations after year filter")

    # Index processed tiles
    print("[build] Indexing processed Landsat tiles ...")
    tile_index = _index_processed_tiles()
    print(f"  {sum(len(v) for v in tile_index.values())} tiles across "
          f"{len(tile_index)} dates")

    nlcd_cache: dict[int, NLCDLayer] = {}

    # --- Observer-based patches ---
    print("\n[build] Extracting observer-based patches ...")
    observer_records = extract_observer_patches(
        obs, tile_index, args.dem, nlcd_cache
    )

    # Count minority class size to set synthetic cap
    if not args.no_synthetic:
        observer_stage_counts = Counter(r["stage"] for r in observer_records)
        print("\n[build] Observer patch stage distribution:")
        for idx in range(NUM_CLASSES):
            print(f"  {STAGE_NAMES[idx]:20s}: {observer_stage_counts.get(idx, 0)}")

        minority_count = min(
            observer_stage_counts.get(STAGES["early"], 0),
            observer_stage_counts.get(STAGES["peak"],  0),
        )
        max_synth_class_count = min(
            observer_stage_counts.get(STAGES["no_transition"], 0),
            observer_stage_counts.get(STAGES["late"],  0),
        )

        if args.max_synthetic is not None:
            max_synthetic = args.max_synthetic
        elif minority_count == 0:
            print(
                "\n[build] WARNING: No early or peak observer patches found. "
                "Defaulting to 500 synthetic patches per class.\n"
                "  Consider checking your observations CSV and tile coverage."
            )
            max_synthetic = 500
        else:
            max_synthetic = max(
                                min(max_synth_class_count, minority_count * 5), 
                                200
                                )

        print(f"\n[build] Synthetic cap: {max_synthetic} patches per class "
              f"(minority observer count: {minority_count})")

        # --- Synthetic patches ---
        print("\n[build] Generating synthetic patches ...")
        synthetic_records = generate_synthetic_patches(
            tile_index, args.dem, nlcd_cache,
            max_per_class=max_synthetic,
            years=args.years,
            rng=rng,
        )
    else:
        synthetic_records = []
        print("\n[build] Skipping synthetic patch generation (--no-synthetic).")

    # --- Combine and summarise ---
    all_records = observer_records + synthetic_records
    if not all_records:
        print("[build] No patches extracted. Check your data.")
        sys.exit(1)

    print(f"\n[build] Final patch distribution ({len(all_records)} total):")
    stage_counts = Counter(r["stage"] for r in all_records)
    source_counts = Counter(r["label_source"] for r in all_records)
    for idx in range(NUM_CLASSES):
        print(f"  {STAGE_NAMES[idx]:20s}: {stage_counts.get(idx, 0):6d}")
    print("  By source:")
    for src, count in source_counts.most_common():
        print(f"    {src:30s}: {count:6d}")

    # Shuffle before writing so HDF5 isn't sorted by stage
    rng.shuffle(all_records)

    write_hdf5(all_records, args.out)
    compute_norm_stats(args.out, TRAIN_YEARS)

    print(
        f"\n[build] Complete.\n"
        f"  Patches : {args.out}\n"
        f"  Norm    : {NORM_STATS_PATH}\n"
        f"\nNext step: python scripts/train_spectral.py --hdf5 {args.out}"
    )


if __name__ == "__main__":
    main()