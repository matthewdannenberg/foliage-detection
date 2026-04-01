"""
prepare_dem.py — Merge USGS NED tiles into a single Vermont DEM GeoTIFF.

Downloads from the USGS National Map come as individual tiles (typically
1x1 degree). This script merges all tiles in a source directory into one
continuous raster covering Vermont, saved as a compressed GeoTIFF.

The output CRS and resolution are preserved from the source tiles (typically
WGS84, ~10m for 1/3 arc-second NED). build_patches.py reprojects the DEM
on the fly to match each Landsat tile's grid, so no reprojection is needed
here.

Usage:
    python scripts/prepare_dem.py
    python scripts/prepare_dem.py --input-dir data/raw/ned --out data/raw/ned/vermont_dem.tif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rasterio
from rasterio.merge import merge


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge NED tiles into a single Vermont DEM GeoTIFF."
    )
    p.add_argument(
        "--input-dir", type=Path,
        default=Path("data/raw/ned"),
        help="Directory containing NED tile GeoTIFFs (default: data/raw/ned).",
    )
    p.add_argument(
        "--out", type=Path,
        default=Path("data/raw/ned/vermont_dem.tif"),
        help="Output path for merged DEM (default: data/raw/ned/vermont_dem.tif).",
    )
    p.add_argument(
        "--pattern", type=str, default="*.tif",
        help="Glob pattern for tile files (default: *.tif). "
             "Use '*.img' for older NED downloads.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tiles = sorted(args.input_dir.glob(args.pattern))
    if not tiles:
        print(
            f"[dem] No files matching '{args.pattern}' found in {args.input_dir}.\n"
            f"  Check --input-dir and --pattern."
        )
        sys.exit(1)

    print(f"[dem] Found {len(tiles)} tile(s):")
    for t in tiles:
        print(f"  {t.name}")

    print("[dem] Merging ...")
    datasets = [rasterio.open(t) for t in tiles]

    try:
        mosaic, transform = merge(datasets)
    finally:
        for ds in datasets:
            ds.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    profile = datasets[0].profile.copy()
    profile.update({
        "driver":    "GTiff",
        "height":    mosaic.shape[1],
        "width":     mosaic.shape[2],
        "count":     1,
        "transform": transform,
        "compress":  "lzw",
        "predictor": 2,
        "tiled":     True,
        "blockxsize": 256,
        "blockysize": 256,
    })

    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(mosaic)

    size_mb = args.out.stat().st_size / 1e6
    print(f"[dem] Saved → {args.out}  ({size_mb:.1f} MB)")
    print(f"      CRS: {datasets[0].crs}")
    print(f"      Size: {mosaic.shape[2]} × {mosaic.shape[1]} px")


if __name__ == "__main__":
    main()
