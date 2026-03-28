"""
config.py — Central configuration for the foliage prediction project.

All paths, physical constants, and hyperparameters live here.
Every other module imports from this file rather than hardcoding values.

Landsat data source: Collection 2 ARD (Analysis Ready Data).
ARD tiles are pre-clipped to a 5000×5000px Albers Equal Area grid (EPSG:5070),
already terrain-corrected, with consistent tiling across years. 
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — everything is relative to this
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_RAW         = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED   = PROJECT_ROOT / "data" / "processed"

LANDSAT_RAW      = DATA_RAW / "landsat"       # raw ARD tiles (not used in S3 pipeline)
NLCD_RAW         = DATA_RAW / "nlcd"          # NLCD GeoTIFFs by year
OBSERVER_RAW     = DATA_RAW / "observer_reports"

PATCHES_DIR      = DATA_PROCESSED / "patches" # HDF5 patch archives
LABELS_DIR       = DATA_PROCESSED / "labels"  # rasterized label GeoTIFFs
OBSERVATIONS_DIR = DATA_PROCESSED / "observations"  # standardised observation CSVs

# ---------------------------------------------------------------------------
# Coordinate reference systems
# ---------------------------------------------------------------------------
# TARGET_CRS: working CRS for all processed outputs.
TARGET_CRS = "EPSG:5070" # (Albers Equal Area Conic, CONUS)
TARGET_RESOLUTION = 250  # metres per pixel

# ---------------------------------------------------------------------------
# Landsat band definitions (Collection 2 ARD, Landsat 4–9)
# Band numbers match the ARD file suffix convention: _SR_B{n}.TIF
# Landsat 4–7 use the same band numbering for SR bands 1–7.
# Landsat 8/9 OLI uses the same band numbers for equivalent wavelengths.
#   These band specifications ONLY apply to Landsat 8/9 OLI
# ---------------------------------------------------------------------------
LANDSAT_BANDS = {
    "blue":   2,
    "green":  3,
    "red":    4,
    "nir":    5,
    "swir1":  6,
    "swir2":  7,
}

# QA_PIXEL bit positions (Collection 2)
QA_FILL          = 0
QA_DILATED_CLOUD = 1
QA_CIRRUS        = 2
QA_CLOUD         = 3
QA_CLOUD_SHADOW  = 4
QA_SNOW          = 5
QA_CLEAR         = 6  # band pixel values set to 1 when pixel is clear
QA_WATER         = 7

# Bits that, if set, indicate a pixel should be masked out
QA_MASK_BITS = [QA_FILL, QA_DILATED_CLOUD, QA_CLOUD, QA_CLOUD_SHADOW, QA_SNOW]

# Scale factors for Collection 2 Level-2 surface reflectance
# DN * scale + offset = physical reflectance
SR_SCALE  = 2.75e-5
SR_OFFSET = -0.2

# ---------------------------------------------------------------------------
# Derived spectral indices
# All computed after DN → reflectance conversion
# ---------------------------------------------------------------------------
# EVI2 (two-band EVI, no blue band required):
#   2.5 * (NIR - Red) / (NIR + 2.4 * Red + 1)
# NDII (Normalized Difference Infrared Index):
#   (NIR - SWIR1) / (NIR + SWIR1)  — sensitive to canopy water / senescence
# NDVI:
#   (NIR - Red) / (NIR + Red)
SPECTRAL_INDICES = ["evi2", "ndii", "ndvi"]

# ---------------------------------------------------------------------------
# NLCD classes used in this project
# Uses the Annual NLCD product (one file per year, 1985–present)
# ---------------------------------------------------------------------------
NLCD_DECIDUOUS_HIGH  = 41   # Deciduous Forest (>75% deciduous)
NLCD_MIXED_FOREST    = 43   # Mixed Forest     (25–75% deciduous)
NLCD_EVERGREEN_HIGH  = 42   # Evergreen Forest (>75% conifer)

# Classes to INCLUDE — everything else is masked out
NLCD_INCLUDE_CLASSES = [NLCD_DECIDUOUS_HIGH, NLCD_MIXED_FOREST]  # 41, 43

# ---------------------------------------------------------------------------
# Foliage stage labels
# ---------------------------------------------------------------------------
STAGES = {
    "no_transition": 0,  # summer green / no change yet
    "early":         1,  # first visible color change, ~10% canopy turned
    "peak":          2,  # maximum color, ~50–70% canopy turned
    "late":          3,  # past peak, significant leaf drop
}
NUM_CLASSES = len(STAGES)
STAGE_NAMES = {v: k for k, v in STAGES.items()}

# ---------------------------------------------------------------------------
# Training years and splits
# Splitting by year (not by patch) tests true temporal generalization.
# Using only Landsat 8-9 to avoid band issues.
# ---------------------------------------------------------------------------
ARD_SENSORS = ["LC08", "LC09"]
TRAIN_YEARS = list(range(2013, 2021))   
VAL_YEARS   = [2021, 2022]
TEST_YEARS  = [2023, 2024]

# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------
PATCH_SIZE   = 32    # pixels — gives ~15x15 effective receptive field at center
PATCH_STRIDE = 16    # stride when tiling a full scene for dense inference

# ---------------------------------------------------------------------------
# Input channels
# 6 Landsat bands + 3 spectral indices + elevation + slope + aspect
# + 1 NLCD deciduous fraction = 14 total
# ---------------------------------------------------------------------------
LANDSAT_CHANNEL_NAMES = list(LANDSAT_BANDS.keys()) + SPECTRAL_INDICES
STATIC_CHANNEL_NAMES  = ["elevation", "slope", "aspect", "deciduous_fraction"]
ALL_CHANNEL_NAMES     = LANDSAT_CHANNEL_NAMES + STATIC_CHANNEL_NAMES
NUM_CHANNELS          = len(ALL_CHANNEL_NAMES)  # 14

# Per-channel normalization statistics (populated by preprocess_landsat.py,
# stored as a JSON sidecar next to the patch archive).
NORM_STATS_PATH = DATA_PROCESSED / "norm_stats.json"

# ---------------------------------------------------------------------------
# ARD tile processing
# ---------------------------------------------------------------------------
# Vermont falls within ARD CONUS tiles h028–h029, v004–v005
ARD_VERMONT_TILES = [
    ("028", "004"),
    ("029", "004"),
    ("029", "005"),
]

# Minimum fraction of valid (non-cloud) pixels required to keep a tile.
# Tiles below this threshold are skipped during preprocessing.
ARD_MIN_VALID_FRACTION = 0.40

# Sentinel value written to processed GeoTIFFs for no-data pixels.
# Using NaN directly in float32 GeoTIFFs; rasterio handles this cleanly.
NODATA_VALUE = float("nan")

# ---------------------------------------------------------------------------
# Model hyperparameters (Stage 1 spectral CNN)
# ---------------------------------------------------------------------------
MODEL = {
    "in_channels":   NUM_CHANNELS,   # 14
    "num_classes":   NUM_CLASSES,    # 4
    "base_filters":  32,             # first conv block output channels
    "dropout":       0.3,
}

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
TRAIN = {
    "batch_size":        64,
    "num_workers":       4,
    "learning_rate":     1e-3,
    "weight_decay":      1e-4,
    "epochs":            50,
    "patience":          8,          # early stopping patience (val loss)
    "checkpoint_dir":    PROJECT_ROOT / "checkpoints",
    "log_dir":           PROJECT_ROOT / "runs",
}

# ---------------------------------------------------------------------------
# Observation data schema
# ---------------------------------------------------------------------------
# Raw CSVs (data/raw/observer_reports/) preserve source data as-is.
# Standardised CSVs (data/processed/observations/) use this schema:
#
#   date        : YYYY-MM-DD string
#   latitude    : float, WGS84 decimal degrees
#   longitude   : float, WGS84 decimal degrees
#   stage       : int, 0–3 index matching STAGES above
#   confidence  : float, 0–1 heuristic probability that the label is correct
#   source      : string, data source identifier e.g. "USA-NPN"
#   notes       : string, free text (optional)
#
# Confidence encoding conventions:
#   1.00  — directly observed with unambiguous intensity (e.g. NPN "75-94%")
#   0.80  — directly observed with moderate intensity (e.g. NPN "25-49%")
#   0.60  — directly observed, intensity at a stage boundary
#   0.50  — inferred from date heuristic (no direct observation)
#   0.00  — not used; records with no confidence should be excluded
#
# Consolidation: multiple observations at the same location on the same day
# are merged into one record. Stage is the plurality vote; confidence is the
# fraction of observations that agree with the plurality.
OBSERVATION_COLUMNS = [
    "date", "latitude", "longitude", "stage", "confidence", "source", "notes"
]
 
# Coordinate snapping precision for consolidation.
# 4 decimal places ≈ 11m — fine enough to group same-site observations
# without accidentally merging distinct nearby locations.
CONSOLIDATION_COORD_DECIMALS = 4
 
# Raw observer report columns (pre-standardisation)
OBSERVER_COLUMNS = ["date", "latitude", "longitude", "stage", "source", "notes"]
 
# Maximum days between an observation and a matching Landsat tile
OBSERVER_SCENE_MAX_DAYS = 3