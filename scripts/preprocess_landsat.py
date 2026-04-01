"""
preprocess_landsat.py — Stream Landsat ARD from S3 and write processed stacks.

For each year in the training/val/test range:
  1. Query the USGS LandsatLook STAC API for ARD tiles covering the requested
     tile set (default: Vermont; pass --tile-list for Northeast or any subset)
     in the Aug–Nov window with cloud cover below threshold.
  2. For each item, stream only the QA band first to check cloud coverage.
     Items below ARD_MIN_VALID_FRACTION are skipped without reading spectral data.
  3. Stream the 6 spectral bands directly from S3 COGs, compute indices,
     and reproject to 250m — entirely in memory.
  4. Write the processed 9-channel stack as a compressed GeoTIFF to
     data/processed/landsat/{year}/{tile_id}_stack.tif.

No raw Landsat data is ever written to disk.

AWS credentials are read from the standard boto3 credential chain.
The S3 bucket (s3://usgs-landsat) is requester-pays — charges apply to
your AWS account for data transfer.

The --tile-list argument accepts a text file with one tile ID per line
(format: HHHVVV e.g. 028004), as produced by process_observations.py.
This allows targeted downloads covering only tiles with known observation
sites, minimising S3 transfer costs.

Output layout:
    data/processed/landsat/
        {year}/
            {tile_id}_stack.tif    (9-channel float32 GeoTIFF, LZW compressed)

Usage:
    python scripts/preprocess_landsat.py
    python scripts/preprocess_landsat.py --tile-list data/processed/observations/ard_tile_list.txt
    python scripts/preprocess_landsat.py --years 2019 2020
    python scripts/preprocess_landsat.py --years 2019 --max-cloud 50 --overwrite
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import rasterio
from tqdm import tqdm

from config import (
    ARD_MIN_VALID_FRACTION,
    ARD_VERMONT_TILES,
    DATA_PROCESSED,
    LANDSAT_CHANNEL_NAMES,
    NODATA_VALUE,
    TARGET_CRS,
    TRAIN_YEARS,
    VAL_YEARS,
    TEST_YEARS,
)
from data.stac import ARDItem, _s3_env, query_ard

PROCESSED_LANDSAT = DATA_PROCESSED / "landsat"


# ---------------------------------------------------------------------------
# Output path and writing
# ---------------------------------------------------------------------------

def output_path(item: ARDItem) -> Path:
    return PROCESSED_LANDSAT / str(item.year) / f"{item.tile_id}_stack.tif"


def write_stack(stack, transform, path: Path) -> None:
    """Write a (9, H, W) float32 array to a tiled, LZW-compressed GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_bands, height, width = stack.shape

    profile = {
        "driver":     "GTiff",
        "dtype":      "float32",
        "crs":        TARGET_CRS,
        "transform":  transform,
        "width":      width,
        "height":     height,
        "count":      n_bands,
        "nodata":     NODATA_VALUE,
        "compress":   "lzw",
        "predictor":  2,
        "tiled":      True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(path, "w", **profile) as dst:
        for i, name in enumerate(LANDSAT_CHANNEL_NAMES):
            dst.write(stack[i], i + 1)
            dst.update_tags(i + 1, name=name)


# ---------------------------------------------------------------------------
# Per-item processing
# ---------------------------------------------------------------------------

def process_item(item: ARDItem, overwrite: bool = False) -> bool:
    """Stream, process, and write one ARD item.

    Returns True if the item was written (or already existed), False if skipped.
    """
    out = output_path(item)

    if out.exists() and not overwrite:
        return True

    # Cloud gate — streams only the QA band
    frac = item.valid_pixel_fraction()
    if frac < ARD_MIN_VALID_FRACTION:
        print(
            f"  [skip] {item.tile_id} — "
            f"{frac:.1%} valid pixels (threshold {ARD_MIN_VALID_FRACTION:.0%})"
        )
        return False

    # Stream spectral bands, compute indices, reproject — all in memory
    env = _s3_env()
    stack, transform = item.stack(env)
    write_stack(stack, transform, out)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    all_years = sorted(set(TRAIN_YEARS + VAL_YEARS + TEST_YEARS))
    p = argparse.ArgumentParser(
        description="Stream Landsat ARD from S3 and write processed stacks."
    )
    p.add_argument(
        "--tile-list", type=Path, default=None,
        help=(
            "Text file with one ARD tile ID per line (format: HHHVVV, "
            "e.g. 028004), as produced by process_observations.py. "
            "Defaults to Vermont tiles only if not provided."
        ),
    )
    p.add_argument(
        "--years", nargs="+", type=int, default=all_years,
        help="Years to process (default: all split years).",
    )
    p.add_argument(
        "--months", nargs="+", type=int, default=[8, 9, 10, 11],
        help="Months to query (default: 8 9 10 11).",
    )
    p.add_argument(
        "--max-cloud", type=float, default=60.0,
        help="Maximum cloud cover %% for STAC query (default: 60).",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-process items that already have a processed output.",
    )
    return p.parse_args()


def _load_tile_ids(tile_list_path: Path | None) -> list[tuple[str, str]]:
    """Load tile IDs from a text file, or fall back to Vermont defaults.

    File format: one tile ID per line, e.g. '028004' (HHHVVV).
    Returns a list of (h_tile, v_tile) tuples.
    """
    if tile_list_path is None:
        print(f"[preprocess] No --tile-list provided, using Vermont tiles: "
              f"{ARD_VERMONT_TILES}")
        return list(ARD_VERMONT_TILES)

    if not tile_list_path.exists():
        print(f"[preprocess] ERROR: tile list not found: {tile_list_path}")
        sys.exit(1)

    tile_ids = []
    for line in tile_list_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) != 6 or not line.isdigit():
            print(f"[preprocess] WARNING: skipping malformed tile ID '{line}' "
                  f"(expected 6-digit HHHVVV)")
            continue
        tile_ids.append((line[:3], line[3:]))

    print(f"[preprocess] Loaded {len(tile_ids)} tile IDs from {tile_list_path}")
    return tile_ids


def main() -> None:
    args = parse_args()
    tile_ids = _load_tile_ids(args.tile_list)

    print(
        f"[preprocess] Processing years: {args.years}\n"
        f"             Tiles: {len(tile_ids)}\n"
        f"             Months: {args.months}\n"
        f"             Max cloud cover: {args.max_cloud}%%\n"
        f"             Output: {PROCESSED_LANDSAT}\n"
        f"             Note: S3 requester-pays charges apply."
    )

    n_ok = n_skip = n_fail = 0
    failed_years = []

    for year in args.years:
        try:
            items = query_ard(
                year=year,
                tile_ids=tile_ids,
                months=args.months,
                max_cloud_cover=args.max_cloud,
            )
        except Exception:
            print(f"\n[ERROR] STAC query failed for year {year}:")
            traceback.print_exc()
            n_fail += 1
            failed_years += year
            continue

        if not items:
            print(f"[preprocess] No items found for {year} — skipping.")
            n_skip += 1
            continue

        for item in tqdm(items, desc=str(year), unit="tile"):
            ard = ARDItem(item)
            try:
                done = process_item(ard, overwrite=args.overwrite)
                if done:
                    n_ok += 1
                else:
                    n_skip += 1
            except Exception:
                n_fail += 1
                print(f"\n  [ERROR] Failed on {ard.tile_id}:")
                traceback.print_exc()

    print(
        f"\n[preprocess] Done.  "
        f"Written: {n_ok}  Skipped: {n_skip}  Failed: {n_fail}"
    )
    if failed_years:
        print(f"STAC Query Failed for Years {sorted(failed_years)}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()