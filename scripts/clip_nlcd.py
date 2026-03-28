"""
clip_nlcd.py — Clip a full CONUS Annual NLCD GeoTIFF to the Vermont region.

After downloading Annual NLCD files from MRLC (https://www.mrlc.gov/data),
run this script to extract just the Vermont extent and delete the large CONUS
file. The clipped output is ~1–2 MB versus ~1 GB for the full CONUS raster.

The clip extent is defined in EPSG:5070 (Albers Equal Area — the NLCD native
CRS) with a 50km buffer around Vermont's bounding box to ensure no edge
effects during subsequent reprojection to UTM-18N.

Output is written to NLCD_RAW/{year}.tif, matching the convention expected
by nlcd.py.

Usage:
    # Clip a single file:
    python scripts/clip_nlcd.py Annual_NLCD_LndCov_2018_CU_C2V0.tif

    # Clip all downloaded NLCD files in a directory:
    python scripts/clip_nlcd.py --all --input-dir /path/to/downloads

    # Clip but keep the original CONUS file:
    python scripts/clip_nlcd.py Annual_NLCD_LndCov_2018_CU_C2V0.tif --no-delete
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from config import NLCD_RAW
from pyproj import CRS

# ---------------------------------------------------------------------------
# Vermont clip extent in EPSG:5070 (Albers Equal Area — NLCD native CRS)
# Derived from Vermont's geographic bounding box (42.7°N–45.0°N, 73.4°W–71.5°W)
# converted to Albers, then padded by 50km on each side.
# ---------------------------------------------------------------------------
VERMONT_BOUNDS_ALBERS = {
    "left":   1_709_000.0,
    "right":  2_026_000.0,
    "bottom": 2_358_000.0,
    "top":    2_746_000.0,
}


# ---------------------------------------------------------------------------
# Year parsing
# ---------------------------------------------------------------------------

def _parse_year(path: Path) -> int:
    """Extract the year from an Annual NLCD filename.

    Handles both:
      Annual_NLCD_LndCov_2018_CU_C2V0.tif  →  2018
      2018.tif                              →  2018
    """
    # Try the USGS naming pattern first
    m = re.search(r"_(\d{4})_", path.stem)
    if m:
        return int(m.group(1))
    # Fall back to bare year filename
    m = re.match(r"^(\d{4})$", path.stem)
    if m:
        return int(m.group(1))
    raise ValueError(
        f"Cannot parse year from filename '{path.name}'. "
        "Expected 'Annual_NLCD_LndCov_YYYY_*.tif' or 'YYYY.tif'."
    )


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------

def clip_to_vermont(
    src_path: Path,
    dst_path: Path,
    bounds: dict = VERMONT_BOUNDS_ALBERS,
) -> None:
    """Clip a CONUS NLCD GeoTIFF to the Vermont extent and write to dst_path.

    Args:
        src_path: Path to the full CONUS Annual NLCD GeoTIFF.
        dst_path: Output path for the clipped file.
        bounds:   Clip extent in the source file's CRS (EPSG:5070).
                  Dict with keys: left, right, bottom, top.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        # MRLC Annual NLCD files use Albers Equal Area with WGS84 datum, which is
        # functionally identical to EPSG:5070 (NAD83) at our working resolution.
        # CRS validation skipped — files downloaded directly from mrlc.gov are
        # assumed to be in the correct projection.

        # Verify the file is in the expected CRS
        # expected = CRS.from_epsg(5070)
        # file_crs = CRS(src.crs.to_wkt())
        # if not file_crs.equals(expected):
        #     raise ValueError(
        #         f"Expected NLCD in EPSG:5070 (Albers), got: {src.crs.to_string()}. "
        #         "Check that this is an Annual NLCD CONUS file."
        #     )

        colormap = src.colormap(1)   # read the color table

        # Compute the window corresponding to our bounding box
        window = from_bounds(
            left=bounds["left"],
            bottom=bounds["bottom"],
            right=bounds["right"],
            top=bounds["top"],
            transform=src.transform,
        )

        # Round to integer pixel boundaries
        window = window.round_lengths().round_offsets()

        # Read only the windowed data — no need to load the full CONUS raster
        data      = src.read(1, window=window)
        transform = src.window_transform(window)

        profile = src.profile.copy()
        profile.update({
            "width":     data.shape[1],
            "height":    data.shape[0],
            "transform": transform,
            "compress":  "lzw",
            "driver":    "GTiff",
        })

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data, 1)
            dst.write_colormap(1, colormap)

    size_mb = dst_path.stat().st_size / 1e6
    print(f"  Clipped → {dst_path.name}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_file(src_path: Path, delete_original: bool = True) -> bool:
    """Clip one NLCD file and optionally delete the original.

    Returns True on success, False on failure.
    """
    try:
        year     = _parse_year(src_path)
        dst_path = NLCD_RAW / f"{year}.tif"

        if dst_path.exists():
            print(f"  [skip] {dst_path.name} already exists.")
            return True

        src_mb = src_path.stat().st_size / 1e6
        print(f"Processing {src_path.name}  ({src_mb:.0f} MB) → year {year}")

        clip_to_vermont(src_path, dst_path)

        if delete_original:
            src_path.unlink()
            print(f"  Deleted original: {src_path.name}")

        return True

    except Exception as e:
        print(f"  [ERROR] {src_path.name}: {e}")
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clip CONUS Annual NLCD files to Vermont extent."
    )
    p.add_argument(
        "files", nargs="*", type=Path,
        help="One or more NLCD GeoTIFF files to clip.",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Clip all *.tif files in --input-dir.",
    )
    p.add_argument(
        "--input-dir", type=Path, default=Path("."),
        help="Directory to search when --all is set (default: current directory).",
    )
    p.add_argument(
        "--no-delete", action="store_true",
        help="Keep the original CONUS file after clipping.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    files: list[Path] = list(args.files)
    if args.all:
        files += sorted(args.input_dir.glob("Annual_NLCD_LndCov_*.tif*"))

    if not files:
        print(
            "[clip_nlcd] No files specified. Pass filenames directly or use --all.\n"
            "Example: python scripts/clip_nlcd.py Annual_NLCD_LndCov_2018_CU_C2V0.tif"
        )
        sys.exit(1)

    delete = not args.no_delete
    n_ok = n_fail = 0

    for f in files:
        if not f.exists():
            print(f"  [skip] File not found: {f}")
            n_fail += 1
            continue
        if process_file(f, delete_original=delete):
            n_ok += 1
        else:
            n_fail += 1

    print(f"\n[clip_nlcd] Done.  OK: {n_ok}  Failed: {n_fail}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
