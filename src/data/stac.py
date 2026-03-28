"""
stac.py — STAC querying and S3 streaming for Landsat ARD COGs.

Handles two responsibilities:
  1. Querying the USGS LandsatLook STAC API for ARD tiles covering Vermont
     within a given date range and cloud cover threshold.
  2. Opening individual band COGs directly from S3 (requester-pays) via
     rasterio, without writing any raw data to disk.

The STAC collection for ARD surface reflectance is 'landsat-c2ard-sr', hosted
at https://landsatlook.usgs.gov/stac-server. Assets in this collection point
to s3://usgs-landsat (requester-pays, us-west-2).

Typical usage:
    items = query_vermont_ard(year=2019, months=[8, 9, 10, 11])
    for item in items:
        tile = ARDItem(item)
        if tile.valid_pixel_fraction() < 0.4:
            continue
        stack, transform = tile.stack()
        # write stack to processed output...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, Optional

import boto3
import numpy as np
import rasterio
from rasterio.env import Env
from pystac_client import Client

from config import (
    ARD_MIN_VALID_FRACTION,
    ARD_VERMONT_TILES,
    LANDSAT_BANDS,
    QA_MASK_BITS,
    SR_OFFSET,
    SR_SCALE,
    TARGET_CRS,
    TARGET_RESOLUTION,
)
from data.landsat import (
    _bit_mask,
    _reproject_array,
    compute_evi2,
    compute_ndii,
    compute_ndvi,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAC_URL        = "https://landsatlook.usgs.gov/stac-server"
ARD_COLLECTION  = "landsat-c2ard-sr"
S3_BUCKET       = "usgs-landsat"
S3_REGION       = "us-west-2"

# Asset key names within a STAC ARD item for each band and QA layer.
# These are the keys used in item.assets — confirmed from LandsatLook STAC.
BAND_ASSET_KEYS = {
    "blue":  "blue",
    "green": "green",
    "red":   "red",
    "nir08": "nir08",
    "swir16": "swir16",
    "swir22": "swir22",
}
QA_ASSET_KEY = "qa_pixel"

# Map from our semantic band names (matching LANDSAT_BANDS in config) to
# ARD STAC asset keys
BAND_NAME_TO_ASSET = {
    "blue":  "blue",
    "green": "green",
    "red":   "red",
    "nir":   "nir08",
    "swir1": "swir16",
    "swir2": "swir22",
}


# ---------------------------------------------------------------------------
# AWS / rasterio environment
# ---------------------------------------------------------------------------

def _s3_env() -> Env:
    """Return a rasterio Env configured for requester-pays S3 access.

    Sets GDAL environment variables so that rasterio can open
    s3://usgs-landsat COGs directly using the caller's AWS credentials.
    AWS credentials are read from the standard boto3 credential chain
    (environment variables, ~/.aws/credentials, IAM role, etc.).
    """
    return Env(
        AWS_REQUEST_PAYER="requester",
        AWS_REGION="us-west-2",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".TIF,.tif",
        GDAL_HTTP_MAX_RETRY="3",
        GDAL_HTTP_RETRY_DELAY="2",
    )


# ---------------------------------------------------------------------------
# STAC querying
# ---------------------------------------------------------------------------

def query_vermont_ard(
    year: int,
    months: list[int] = [8, 9, 10, 11],
    max_cloud_cover: float = 60.0,
    sensors: list[str] = ["LC08", "LC09"],
) -> list:
    """Query the USGS STAC API for ARD tiles covering Vermont.

    Searches the 'landsat-c2ard-sr' collection for items whose tile ID
    falls within the Vermont tile grid (h028–h029, v004–v005) acquired
    during the specified months.

    Args:
        year:             Acquisition year.
        months:           List of month numbers to include (default: Aug–Nov).
        max_cloud_cover:  Maximum cloud cover percentage (0–100). Items above
                          this threshold are excluded from results.
        sensors:          Sensor codes to include. Defaults to LC08/LC09 only
                          for spectral consistency.

    Returns:
        List of pystac Item objects, sorted by datetime.
    """
    catalog = Client.open(STAC_URL)

    # Build date range covering all requested months in the given year
    date_start = datetime(year, min(months), 1)
    # End of the last requested month
    last_month = max(months)
    last_day   = 31 if last_month in [1,3,5,7,8,10,12] else 30 if last_month != 2 else 28
    date_end   = datetime(year, last_month, last_day, 23, 59, 59)

    datetime_str = f"{date_start.isoformat()}Z/{date_end.isoformat()}Z"

    # Vermont bounding box in WGS84 (STAC queries use geographic coordinates)
    vermont_bbox = [-73.44, 42.72, -71.46, 45.02]

    print(
        f"[stac] Querying {ARD_COLLECTION} for year={year}, "
        f"months={months}, max_cloud={max_cloud_cover}%"
    )

    search = catalog.search(
        collections=[ARD_COLLECTION],
        bbox=vermont_bbox,
        datetime=datetime_str,
        query={"eo:cloud_cover": {"lte": max_cloud_cover}},
        max_items=None,   # retrieve all matching items
    )

    items = list(search.items())

    # Filter to Vermont tile grid and requested sensors
    vermont_tile_ids = {
        f"CU_{h}{v}" for h, v in ARD_VERMONT_TILES
    }

    filtered = []
    for item in items:
        # Check sensor — item ID starts with sensor code e.g. LC08_CU_028004_...
        sensor = item.id[:4]
        if sensor not in sensors:
            continue

        # Check tile — item ID contains CU_HHHVVV
        tile_part = item.id[5:14]   # e.g. 'CU_028004'
        if tile_part not in vermont_tile_ids:
            continue

        filtered.append(item)

    filtered.sort(key=lambda i: i.datetime)
    print(f"[stac] Found {len(filtered)} items after sensor/tile filtering.")
    return filtered


# ---------------------------------------------------------------------------
# Per-item streaming
# ---------------------------------------------------------------------------

@dataclass
class ARDItem:
    """Wraps a single STAC ARD item and streams its bands from S3.

    This class mirrors the interface of LandsatTile but reads data directly
    from S3 COGs rather than local files. No raw data is written to disk.

    Attributes:
        item:      pystac Item object from a STAC query.
        tile_id:   ARD tile ID string (e.g. 'LC08_CU_028004_20190915_...').
        date:      Acquisition date as 'YYYY-MM-DD'.
        year:      Acquisition year (int).
        sensor:    Sensor code, e.g. 'LC08'.
        h_tile:    Horizontal tile index string, e.g. '028'.
        v_tile:    Vertical tile index string, e.g. '004'.
    """

    item:    object   # pystac.Item
    tile_id: str      = field(init=False)
    date:    str      = field(init=False)
    year:    int      = field(init=False)
    sensor:  str      = field(init=False)
    h_tile:  str      = field(init=False)
    v_tile:  str      = field(init=False)

    def __post_init__(self):
        self.tile_id = self.item.id
        self.date    = self.item.datetime.strftime("%Y-%m-%d")
        self.year    = self.item.datetime.year
        self.sensor  = self.tile_id[:4]
        # Parse h/v from tile ID: LC08_CU_028004_...
        #                                  ^^^vvv
        grid_part   = self.tile_id[8:14]   # e.g. '028004'
        self.h_tile = grid_part[:3]
        self.v_tile = grid_part[3:]

    def _asset_url(self, asset_key: str) -> str:
        """Return the S3 URL for a given asset key."""
        asset = self.item.assets.get(asset_key)
        if asset is None:
            raise KeyError(
                f"Asset '{asset_key}' not found in item {self.tile_id}. "
                f"Available: {list(self.item.assets.keys())}"
            )
        try:
            return asset.extra_fields['alternate']['s3']['href']
        except KeyError:
            raise KeyError(
                f"No S3 alternate href found for asset '{asset_key}' "
                f"in item {self.tile_id}. Available extra_fields: "
                f"{asset.extra_fields}"
        )

    def _read_band_from_s3(
        self, asset_key: str, env: Env
    ) -> tuple[np.ndarray, dict]:
        """Stream one band from S3 and return (float32 array, rasterio profile)."""
        url = self._asset_url(asset_key)
        with env:
            with rasterio.open(url) as src:
                data    = src.read(1).astype(np.float32)
                profile = src.profile.copy()
        return data, profile

    def cloud_mask(self, env: Optional[Env] = None) -> np.ndarray:
        """Stream the QA_PIXEL band and return a boolean exclusion mask."""
        env = env or _s3_env()
        url = self._asset_url(QA_ASSET_KEY)
        with env:
            with rasterio.open(url) as src:
                qa = src.read(1)
        return _bit_mask(qa, QA_MASK_BITS)

    def valid_pixel_fraction(self) -> float:
        """Fraction of pixels not excluded by the cloud mask.

        Streams only the QA band — does not load spectral data.
        """
        return float(1.0 - self.cloud_mask().mean())

    def reflectance(
        self, env: Optional[Env] = None
    ) -> tuple[np.ndarray, dict]:
        """Stream all 6 bands and return surface reflectance floats.

        Opens a single S3 environment context shared across all band reads
        to avoid repeated credential refresh overhead.

        Returns:
            refl:    (6, H, W) float32. Masked pixels → NaN.
                     Band order: blue=0, green=1, red=2, nir=3, swir1=4, swir2=5.
            profile: rasterio profile at native ARD resolution (~30m, EPSG:5070).
        """
        env  = env or _s3_env()
        mask = self.cloud_mask(env)

        arrays  = []
        profile = None

        with env:
            for band_name in LANDSAT_BANDS.keys():
                asset_key = BAND_NAME_TO_ASSET[band_name]
                url = self._asset_url(asset_key)
                with rasterio.open(url) as src:
                    dn = src.read(1).astype(np.float32)
                    if profile is None:
                        profile = src.profile.copy()

                refl = np.clip(dn * SR_SCALE + SR_OFFSET, 0.0, 1.0)
                refl[mask] = np.nan
                arrays.append(refl)

        return np.stack(arrays, axis=0), profile   # (6, H, W)

    def indices(self, refl: np.ndarray) -> np.ndarray:
        """Compute EVI2, NDII, NDVI from a (6, H, W) reflectance array."""
        red, nir, swir1 = refl[2], refl[3], refl[4]
        return np.stack([
            compute_evi2(nir, red),
            compute_ndii(nir, swir1),
            compute_ndvi(nir, red),
        ], axis=0)

    def stack(self, env: Optional[Env] = None) -> tuple[np.ndarray, object]:
        """Stream, process, and return the 9-channel spectral stack at 250m.

        All S3 reads share a single rasterio Env to minimise credential
        refresh calls. No data is written to disk.

        Returns:
            data:      (9, H', W') float32 at TARGET_CRS / TARGET_RESOLUTION
            transform: Affine transform of the 250m output grid
        """
        env  = env or _s3_env()
        refl, profile = self.reflectance(env)
        combined = np.concatenate([refl, self.indices(refl)], axis=0)

        return _reproject_array(
            data=combined,
            src_transform=profile["transform"],
            src_crs=TARGET_CRS,
        )

    def __repr__(self) -> str:
        return (
            f"ARDItem(sensor={self.sensor}, "
            f"tile=h{self.h_tile}v{self.v_tile}, date={self.date})"
        )
