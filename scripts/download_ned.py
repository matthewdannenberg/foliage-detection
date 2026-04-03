"""
download_ned.py — Bulk download USGS NED 1 arc-second elevation tiles
for the Northeast via the TNM Access API.

Queries the USGS TNM API for all available 1 arc-second GeoTIFF tiles
covering the Northeast bounding box, then downloads them in parallel to
data/raw/ned/. Existing files are skipped unless --overwrite is passed.

1 arc-second (~30m) is appropriate for this project since all elevation
data is resampled to 250m during patch extraction — the higher-resolution
1/3 arc-second product offers no practical benefit at that output scale
and is roughly 9x larger per tile.

After downloading, run prepare_dem.py to merge into a single GeoTIFF.

Usage:
    python scripts/download_ned.py
    python scripts/download_ned.py --workers 4
    python scripts/download_ned.py --out-dir data/raw/ned
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# TNM API
# ---------------------------------------------------------------------------
TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"

# Northeast bounding box: Maine through Pennsylvania/New Jersey
# (lon_min, lat_min, lon_max, lat_max)
NORTHEAST_BBOX = "-80.6,38.8,-66.8,47.6"

# Dataset name exactly as the TNM API expects it
DATASET = "National Elevation Dataset (NED) 1 arc-second"

# Only GeoTIFF format
FORMAT = "GeoTIFF"

# Max items per API page
PAGE_SIZE = 100

REQUEST_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 300   # 5 minutes per tile
REQUEST_DELAY    = 0.25  # seconds between API calls


# ---------------------------------------------------------------------------
# API querying
# ---------------------------------------------------------------------------

def _query_all_products(bbox: str) -> list[dict]:
    """Return all NED 1 arc-second GeoTIFF products in the bounding box."""
    products = []
    offset   = 0

    print(f"[ned] Querying TNM API for NED tiles in bbox {bbox} ...")

    while True:
        params = {
            "datasets":   DATASET,
            "bbox":       bbox,
            "prodFormats": FORMAT,
            "max":        PAGE_SIZE,
            "offset":     offset,
        }
        try:
            resp = requests.get(TNM_API, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  [ERROR] API query failed at offset {offset}: {e}")
            break

        items = data.get("items", [])
        total = data.get("total", 0)

        if offset == 0:
            print(f"  Total tiles available: {total}")

        products.extend(items)

        if len(products) >= total or not items:
            break

        offset += PAGE_SIZE
        time.sleep(REQUEST_DELAY)

    return products


def _download_url(product: dict) -> str | None:
    """Extract the GeoTIFF download URL from a TNM product dict."""
    # downloadURL is the direct file link
    url = product.get("downloadURL", "")
    if url and url.endswith(".tif"):
        return url
    # Fall back to urls dict
    urls = product.get("urls", {})
    return urls.get("TIFF") or urls.get("tiff") or None


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def _download_one(url: str, out_dir: Path, overwrite: bool) -> tuple[str, bool, str]:
    """Download one tile. Returns (filename, success, message)."""
    filename = url.split("/")[-1]
    out_path = out_dir / filename

    if out_path.exists() and not overwrite:
        return filename, True, "skipped (exists)"

    try:
        resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
        resp.raise_for_status()

        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

        size_mb = out_path.stat().st_size / 1e6
        return filename, True, f"downloaded ({size_mb:.1f} MB)"

    except Exception as e:
        if out_path.exists():
            out_path.unlink()   # remove partial file
        return filename, False, f"FAILED: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bulk download NED 1 arc-second tiles for the Northeast."
    )
    p.add_argument(
        "--bbox", default=NORTHEAST_BBOX,
        help=f"Bounding box as lon_min,lat_min,lon_max,lat_max "
             f"(default: {NORTHEAST_BBOX}).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("data/raw/ned"),
        help="Output directory for downloaded tiles (default: data/raw/ned).",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel download threads (default: 4).",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-download tiles that already exist.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Query TNM API
    products = _query_all_products(args.bbox)
    if not products:
        print("[ned] No products found. Check bbox and dataset name.")
        sys.exit(1)

    # Extract download URLs, deduplicate by geographic tile ID keeping
    # only the most recent version of each tile. NED filenames follow the
    # pattern USGS_1_{tile}_{date}.tif e.g. USGS_1_n39w076_20250507.tif
    # where the date suffix is the product creation date, not acquisition date.
    import re
    tile_latest: dict[str, tuple[str, str]] = {}  # tile_id → (date_str, url)

    for p in products:
        url = _download_url(p)
        if not url:
            print(f"  [warn] No GeoTIFF URL for: {p.get('title', 'unknown')}")
            continue

        filename = url.split("/")[-1]
        m = re.match(r"USGS_1_([a-z]\d+[a-z]\d+)_(\d+)\.tif", filename, re.IGNORECASE)
        if m:
            tile_id   = m.group(1).lower()
            date_str  = m.group(2)
            existing  = tile_latest.get(tile_id)
            if existing is None or date_str > existing[0]:
                tile_latest[tile_id] = (date_str, url)
        else:
            # Filename doesn't match expected pattern — include it anyway
            if url not in {u for _, u in tile_latest.values()}:
                tile_latest[filename] = ("", url)

    download_jobs = [url for _, url in tile_latest.values()]
    print(f"\n[ned] {len(download_jobs)} unique tiles after deduplication "
          f"({len(products) - len(download_jobs)} older versions skipped)"
          f" → {args.out_dir}")

    # Download in parallel
    n_ok = n_skip = n_fail = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_download_one, url, args.out_dir, args.overwrite): url
            for url in download_jobs
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Downloading", unit="tile"):
            filename, success, msg = future.result()
            if "skipped" in msg:
                n_skip += 1
            elif success:
                n_ok += 1
            else:
                n_fail += 1
                tqdm.write(f"  [ERROR] {filename}: {msg}")

    print(
        f"\n[ned] Done.  "
        f"Downloaded: {n_ok}  Skipped: {n_skip}  Failed: {n_fail}"
    )
    if n_fail:
        print(f"  Re-run with --overwrite to retry failed tiles.")
        sys.exit(1)

    print(
        f"\nNext step: python scripts/prepare_dem.py "
        f"--input-dir {args.out_dir} --out data/raw/ned/northeast_dem.tif"
    )


if __name__ == "__main__":
    main()