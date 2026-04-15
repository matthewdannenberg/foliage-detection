# Vermont Foliage Transition Forecasting

A spatiotemporal machine learning system for predicting when and where fall foliage transitions occur across Vermont and the broader Northeast United States, using Landsat satellite imagery and ground-based phenology observations.

---

## Overview

Fall foliage in New England is ecologically significant, economically important to the tourism industry, and increasingly affected by climate variability. Predicting its timing — and mapping it spatially at fine resolution — is a non-trivial remote sensing problem. The transition from summer green to peak color to bare canopy involves subtle spectral changes that vary with elevation, species composition, local microclimate, and year-to-year weather patterns.

This project builds a two-stage forecasting pipeline. The first stage is a convolutional neural network trained on Landsat Analysis Ready Data (ARD) that classifies each 250m pixel in a satellite scene into one of four foliage stages: pre-transition, early, peak, or late. The second stage (in development) is a spatiotemporal model that ingests spring and summer weather data alongside Stage 1 predictions from earlier in the fall season to forecast when each location will reach each stage.

The system is designed to eventually produce spatially continuous maps over Vermont — and the broader Northeast — predicting peak foliage timing at the pixel level, updated as new satellite acquisitions become available through the season.

---

## System Architecture

```
Raw Data Sources
├── Landsat ARD (USGS, via S3)         ← spectral signal
├── NED 1 arc-second DEM (USGS)        ← terrain context
├── NLCD Annual Land Cover             ← forest type filtering
├── USA-NPN Observer Reports           ← ground truth labels
└── PhenoCam Network Time Series       ← ground truth labels

          │
          ▼

Stage 1: Spectral Classifier (this repository)
    Input:  32×32 pixel patch, 14 channels
            [blue, green, red, NIR, SWIR1, SWIR2,
             NDVI, EVI, NDWI, NBR,
             elevation, slope, aspect, hillshade]
    Output: P(no_transition, early, peak, late)

          │
          ▼

Stage 2: Spatiotemporal Forecaster (planned)
    Input:  Stage 1 classifications + spring/summer weather
    Output: Predicted date of early/peak/late transition
            per location, for the current season
```

The Stage 1 model operates on individual satellite scenes and produces a classified raster at 250m resolution. Stage 2 will treat these classifications as observations in a time series, combining them with antecedent weather covariates (accumulated growing degree days, drought indices, photoperiod) to forecast transition timing before it occurs.

---

## Data

### Landsat ARD
Landsat Collection 2 Analysis Ready Data is accessed from the USGS requester-pays S3 bucket (`s3://usgs-landsat`, us-west-2). ARD tiles are pre-georeferenced, atmospherically corrected to surface reflectance, and provided on a consistent 30m grid in EPSG:5070 (Albers Equal Area). Scenes are filtered to August–November, cloud cover below 60%, and reprojected to 250m for consistency with the label resolution.

The current dataset covers 27 ARD tile positions across the Northeast (Maine through Pennsylvania), representing over 1,800 processed scenes across 12 years (2013–2024).

### Ground Truth Labels

Two sources of observer labels are combined.

**USA National Phenology Network (USA-NPN)** maintains a network of trained volunteers recording phenophase observations at fixed sites. Two phenophases are used:

- **Phenophase 498** (Colored Leaves): intensity percentage mapped to *early* (< 25%) or *peak* (≥ 25%). A "No" observation maps to *no_transition*.
- **Phenophase 499** (Falling Leaves): intensity ≥ 25% mapped to *late*.

The Northeast network provides 44,221 observations across 1,157 unique sites and 27 ARD tiles. Observer patches are extracted by spatially and temporally matching each observation to the nearest qualifying Landsat scene (within ±5 days).

**PhenoCam Network** provides continuous daily time series of canopy color from fixed RGB cameras at forest research sites. GCC (Green Chromatic Coordinate) and RCC (Red Chromatic Coordinate) are extracted from each image. Stage labels are derived using PELT changepoint detection on the 2D (GCC, RCC) signal, with the max-RCC-anchor assignment rule — the segment with the highest mean RCC is assigned *peak*, segments after it are *late*, the preceding segment is *early*, and all earlier segments are *no_transition*. This rule was calibrated against 1,700 human-labeled images across 120 Northeast site-years, achieving a macro-averaged F1 of 0.748. Only deciduous broadleaf (DB) and mixed (MX) ROI types are included.

### Synthetic Labels
The *no_transition* and *late* classes are sparsely represented in observer data — late observations in particular are rarely reported by NPN volunteers. Synthetic patches are generated by sampling pixels from August scenes (no_transition) and post-November-20 scenes (late, after snow masking). These are capped at 5× the rarest observer class to prevent pathological imbalance.

### Terrain
USGS NED 1 arc-second DEM tiles covering the Northeast are merged into a regional mosaic. Elevation, slope, aspect, and hillshade are derived at 250m and appended as additional input channels. The 1 arc-second product (~30m native resolution) is appropriate given that all data is resampled to 250m for patch extraction.

### Land Cover
NLCD Annual Land Cover data (MRLC) is used to filter patches to forested pixels. Only NLCD classes 41 (deciduous forest) and 43 (mixed forest) are retained for the observer label matching step.

---

## Stage 1 Model

### Architecture

A lightweight CNN with three convolutional blocks followed by global average pooling and a two-layer classifier head.

```
Input (14, 32, 32)
  → ConvBlock(14→32) + MaxPool     → (32, 16, 16)
  → ConvBlock(32→64) + MaxPool     → (64, 8, 8)
  → ConvBlock(64→128)              → (128, 8, 8)
  → GlobalAveragePool              → (128,)
  → Linear(128→64) + ReLU + Dropout(0.3)
  → Linear(64→4)                   → logits
```

Batch normalisation after each convolution stabilises training under the aggressive oversampling required by class imbalance. Global average pooling rather than flattening provides spatial invariance and reduces parameter count.

### Training

| Parameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 3 × 10⁻⁴ (cosine decay to 6 × 10⁻⁶) |
| Weight decay | 10⁻⁴ |
| Batch size | 64 |
| Max epochs | 150 |
| Early stopping patience | 20 epochs (val loss) |
| Label smoothing | 0.1 |
| Class balancing | WeightedRandomSampler |

The temporal split holds out 2021–2022 for validation and 2023–2024 for final test evaluation. Since NPN sites are persistent monitoring stations that appear across years, this constitutes a strict temporal holdout rather than a site holdout — the model is evaluated on its ability to classify scenes from years it has never seen.

### Current Results

Results on the validation set after the current round of development (test set reserved for final evaluation):

| Stage | Accuracy |
|---|---|
| no_transition | 78% |
| early | 47% |
| peak | 88% |
| late | 75% |
| **overall** | **74%** |

The early/no_transition boundary is the most challenging classification — at 250m resolution, the spectral difference between a fully green canopy and one with 5–15% color change is subtle and observer-dependent. Peak and late, which involve larger spectral shifts, are more reliably classified. Ongoing work focuses on expanding observer coverage and relaxing the NLCD filter to recover excluded training patches.

---

## Reproducing the Pipeline

Full data download and processing instructions are in [`data/README_data.md`](data/README_data.md). The abbreviated sequence is:

```bash
# 1. Download and clip NLCD land cover
python scripts/clip_nlcd.py --all --input-dir /path/to/nlcd/downloads

# 2. Download and merge NED elevation tiles
python scripts/download_ned.py
python scripts/prepare_dem.py

# 3. Download observer data
python scripts/download_npn.py --request-source "Your Name" \
    --states ME NH VT MA RI CT NY NJ PA
python scripts/download_phenocam.py --verbose

# 4. Process and consolidate observations (also generates ARD tile list)
python scripts/process_observations.py \
    --sources npn_northeast.csv phenocam_northeast.csv

# 5. Download and preprocess Landsat ARD
# Recommended: run on an EC2 instance in us-west-2 (co-located with S3)
python scripts/preprocess_landsat.py \
    --tile-list data/processed/observations/ard_tile_list.txt

# 6. Build patch archive
python scripts/build_patches.py

# 7. Train
python scripts/train_spectral.py --run-name exp01

# 8. Evaluate (run once, when development is complete)
python scripts/evaluate.py
```

---

## Project Structure

```
foliage_detection/
├── data/
│   ├── raw/
│   │   ├── ned/              NED elevation tiles and merged mosaic
│   │   ├── nlcd/             Clipped NLCD annual land cover rasters
│   │   └── observer_reports/ Raw NPN CSV downloads
│   ├── processed/
│   │   ├── landsat/          Processed Landsat stacks (year/tile_stack.tif)
│   │   ├── observations/     Consolidated observations CSV and ARD tile list
│   │   └── patches/          HDF5 patch archive (patches.h5)
|   └── human/                Limited pool of human labeled data
|
├── src/
│   ├── config.py             All paths, constants, and hyperparameters
│   ├── data/
│   │   ├── dataset.py        PyTorch Dataset, DataLoader factory
│   │   ├── landsat.py        LandsatTile, spectral indices, QA masking
│   │   ├── nlcd.py           NLCDLayer for land cover filtering
│   │   └── stac.py           STAC querying and S3 streaming
│   ├── models/
│   │   └── spectral_cnn.py   SpectralCNN architecture and checkpoint I/O
│   └── train/
│       └── trainer.py        Training loop, early stopping, metrics
├── scripts/
│   ├── clip_nlcd.py          Clip CONUS NLCD to Northeast extent
│   ├── download_ned.py       Bulk download NED tiles via TNM API
│   ├── download_npn.py       Download USA-NPN phenology observations
│   ├── download_phenocam.py  Download PhenoCam GCC/RCC and derive stage labels
│   ├── prepare_dem.py        Merge NED tiles into regional mosaic
│   ├── preprocess_landsat.py Stream and process Landsat ARD from S3
│   ├── process_observations.py Consolidate and standardise observer data
│   ├── build_patches.py      Extract HDF5 training patch archive
│   ├── train_spectral.py     Training entry point
│   └── evaluate.py           Final test set evaluation (run once)
├── notebooks/
│   ├── label_inspection.ipynb      Visual inspection of labeled patches
│   ├── observation_map.ipynb       Geographic map of observer coverage
│   └── phenocam_calibration.ipynb  PELT calibration and human labeling tool
├── checkpoints/              Saved model checkpoints
├── runs/                     TensorBoard training logs
└── tests/                    Unit tests
```

---

## Future Work

**PhenoCam label refinement.** The current PELT-based labeling achieves macro-F1 of 0.748 against human labels, with early and peak as the weakest classes (F1 ~0.55). The early/peak boundary is inherently ambiguous at daily resolution — leaves turn gradually rather than overnight. Further calibration with more human-labeled site-years, or tighter changepoint parameters tuned specifically to Northeast broadleaf species, could improve this.

**NLCD filter relaxation.** The current requirement that observer patch centres fall in NLCD deciduous or mixed forest classes excludes ~7,400 patches from NPN sites that monitor deciduous trees at locations NLCD classifies differently. Relaxing this to exclude only clearly non-forest classes (developed, cropland, water) would substantially increase training data.

**Stage 2 spatiotemporal model.** The intended final product is a seasonal forecast — given weather data through, say, September 15, predict when each pixel will reach each foliage stage. This requires accumulating Stage 1 classifications across multiple scenes within a season and regressing against antecedent meteorological covariates. The architecture under consideration is a transformer over the sequence of within-season Stage 1 outputs, conditioned on gridded weather features.

**Operational inference.** The system is designed with operational use in mind. New Landsat scenes are added to the USGS S3 bucket within days of acquisition. A production pipeline would poll for new tiles, run Stage 1 inference, and update a web-served foliage map for Vermont on a near-real-time basis through the fall season.

---

## Requirements

```
python >= 3.11
torch >= 2.0
rasterio
pystac-client
h5py
numpy
pandas
pyproj
scipy
ruptures          # PhenoCam PELT changepoint detection
tqdm
tensorboard
ipywidgets        # for phenocam_calibration.ipynb labeling widget
Pillow            # for notebook image display
contextily        # for observation_map.ipynb
scikit-learn      # for calibration notebooks
```

AWS credentials with access to requester-pays S3 buckets are required for the Landsat download step.