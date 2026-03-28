"""
build_patches.py — Extract labelled patches from processed Landsat stacks.

Combines two label sources:

  1. OBSERVER-BASED PATCHES
     Reads data/processed/observations/observations.csv, matches each
     observation to the nearest processed Landsat stack within ±3 days,
     and extracts a (NUM_CHANNELS, PATCH_SIZE, PATCH_SIZE) patch centred
     on the observation's pixel location.

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
from datetime import date, timedelta
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
    DATA_PROCESSED,
    NLCD_RAW,
    NORM_STATS_PATH,
    NUM_CHANNELS,
    NUM_CLASSES,
    OBSERVATIONS_DIR,
    OBSERVER_SCENE_MAX_DAYS,
    PATCH_SIZE,
    PATCHES_DIR,
    STAGE_NAMES,
    STAGES,
    TARGET_CRS,
    TARGET_RESOLUTION,
    TRAIN_YEARS,
    VAL_YEARS,
    TEST_YEARS,
)
from data.nlcd import NLCDLayer

PROCESSED_LANDSAT = DATA_PROCESSED / "landsat"

# Heuristic confidence values for synthetic labels
SYNTHETIC_NO_TRANSITION_CONFIDENCE = 0.95
SYNTHETIC_LATE_CONFIDENCE          = 0.80

# Date windows for synthetic label generation
# Aug 1–20: virtually no Vermont foliage change → no_transition
SYNTHETIC_NO_TRANSITION_MONTH_DAY = [(8, 1),  (8, 20)]
# Nov 10–30: well past peak, significant leaf drop → late
SYNTHETIC_LATE_MONTH_DAY          = [(11, 10), (11, 30)]

# Random seed for reproducible synthetic patch sampling
RANDOM_SEED = 42


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


# ---------------------------------------------------------------------------
# Coordinate → pixel conversion
# ---------------------------------------------------------------------------

def _lonlat_to_rowcol(
    lon: float,
    lat: float,
    transform,
) -> tuple[int, int]:
    """Convert WGS84 lon/lat to pixel (row, col) on the 250m grid."""
    transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    x, y = transformer.transform(lon, lat)
    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)
    return row, col


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


def _find_nearest_tiles(
    obs_date: date,
    tile_index: dict[str, list[Path]],
    max_days: int = OBSERVER_SCENE_MAX_DAYS,
) -> list[Path]:
    """Return all tile paths within max_days of obs_date, sorted by proximity."""
    candidates = []
    for delta in range(0, max_days + 1):
        for sign in ([0] if delta == 0 else [1, -1]):
            d = obs_date + timedelta(days=delta * sign)
            key = d.isoformat()
            if key in tile_index:
                for path in tile_index[key]:
                    candidates.append((delta, path))
    candidates.sort(key=lambda x: x[0])
    return [p for _, p in candidates]


# ---------------------------------------------------------------------------
# Observer-based patch extraction
# ---------------------------------------------------------------------------

def extract_observer_patches(
    observations: pd.DataFrame,
    tile_index: dict[str, list[Path]],
    dem_path: Path,
    nlcd_cache: dict[int, NLCDLayer],
) -> list[dict]:
    """Extract patches at observer locations matched to nearby tiles.

    Returns a list of record dicts with keys:
        patch, stage, confidence, year, source, label_source
    """
    records = []
    half = PATCH_SIZE // 2
    skipped = Counter()

    for _, row in tqdm(observations.iterrows(), total=len(observations),
                       desc="Observer patches", unit="obs"):
        try:
            obs_date = date.fromisoformat(str(row["date"]))
        except ValueError:
            skipped["bad_date"] += 1
            continue

        tile_paths = _find_nearest_tiles(obs_date, tile_index)
        if not tile_paths:
            skipped["no_nearby_tile"] += 1
            continue

        stage      = int(row["stage"])
        confidence = float(row["confidence"])
        source     = str(row.get("source", ""))

        # Try each nearby tile until we get a valid patch
        matched = False
        for tile_path in tile_paths:
            with rasterio.open(tile_path) as src:
                spectral  = src.read().astype(np.float32)   # (9, H, W)
                transform = src.transform
                H, W      = src.height, src.width

            year = int(tile_path.parts[-2])  # .../landsat/{year}/tile_stack.tif

            # Static layers — build once per tile per NLCD year
            nlcd = nlcd_cache.setdefault(year, NLCDLayer(year))
            exclusion_mask     = nlcd.exclusion_mask(transform, (H, W))
            deciduous_fraction = nlcd.deciduous_fraction(transform, (H, W))
            elevation          = _load_dem_for_grid(dem_path, transform, (H, W))
            slope, aspect      = _slope_aspect(elevation)

            cube = _build_cube(spectral, elevation, slope, aspect, deciduous_fraction)

            px_row, px_col = _lonlat_to_rowcol(
                float(row["longitude"]), float(row["latitude"]), transform
            )

            if not _is_valid_patch(cube, px_row, px_col, half, exclusion_mask):
                skipped["invalid_pixel"] += 1
                continue

            patch = _extract_patch(cube, px_row, px_col, half)
            records.append({
                "patch":        patch,
                "stage":        stage,
                "confidence":   confidence,
                "year":         year,
                "source":       source,
                "label_source": "observer",
            })
            matched = True
            break

        if not matched:
            skipped["no_valid_patch"] += 1

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
            if _in_synthetic_window(td, SYNTHETIC_NO_TRANSITION_MONTH_DAY):
                eligible[STAGES["no_transition"]].append(path)
            elif _in_synthetic_window(td, SYNTHETIC_LATE_MONTH_DAY):
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

            nlcd = nlcd_cache.setdefault(year, NLCDLayer(year))
            exclusion_mask     = nlcd.exclusion_mask(transform, (H, W))
            deciduous_fraction = nlcd.deciduous_fraction(transform, (H, W))
            elevation          = _load_dem_for_grid(dem_path, transform, (H, W))
            slope, aspect      = _slope_aspect(elevation)

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
                records_by_stage[stage_idx].append({
                    "patch":        patch,
                    "stage":        stage_idx,
                    "confidence":   confidence,
                    "year":         year,
                    "source":       "synthetic",
                    "label_source": f"synthetic_{stage_name}",
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
        /label_source (N,)         variable-length string
        /metadata     JSON string
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    N = len(records)

    patches      = np.stack([r["patch"]      for r in records]).astype(np.float32)
    labels       = np.array([r["stage"]      for r in records], dtype=np.int64)
    confidence   = np.array([r["confidence"] for r in records], dtype=np.float32)
    years        = np.array([r["year"]       for r in records], dtype=np.int32)
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

        # Variable-length strings for label_source
        dt = h5py.special_dtype(vlen=str)
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
    """
    print("[build] Computing normalisation statistics from training patches ...")
    with h5py.File(hdf5_path, "r") as f:
        all_years    = f["years"][:]
        train_mask   = np.isin(all_years, train_years)
        train_patches = f["patches"][train_mask]   # (N_train, C, H, W)

    mean = train_patches.mean(axis=(0, 2, 3)).tolist()
    std  = train_patches.std(axis=(0, 2, 3)).tolist()

    NORM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NORM_STATS_PATH, "w") as f:
        json.dump({
            "mean":     mean,
            "std":      std,
            "channels": ALL_CHANNEL_NAMES,
            "n_train":  int(train_mask.sum()),
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
        "--dem", type=Path, required=True,
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
            max_synthetic = max(minority_count * 5, 200)

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