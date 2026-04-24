"""
dataset.py — PyTorch Dataset and DataLoader factories for the spectral classifier.

Reads from the HDF5 patch archive produced by scripts/build_patches.py.

HDF5 schema expected:
    /patches      (N, C, H, W) float32  — input patches
    /labels       (N,)         int64    — stage indices (0–3)
    /confidence   (N,)         float32  — per-sample label confidence [0, 1]
    /years        (N,)         int32    — acquisition year
    /label_source (N,)         str      — 'observer', 'synthetic_no_transition', etc.
    /metadata     str                   — JSON with channel/class names

Channel order (NUM_CHANNELS = 13):
    0–5   Landsat bands (blue, green, red, nir, swir1, swir2)
    6–8   Spectral indices (evi2, ndii, ndvi)
    9     Elevation
    10    Slope
    11    Aspect
    12    Deciduous fraction
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import (
    ALL_CHANNEL_NAMES,
    NORM_STATS_PATH,
    NUM_CHANNELS,
    NUM_CLASSES,
    PATCH_SIZE,
    PATCHES_DIR,
    TRAIN_YEARS,
    VAL_YEARS,
    TEST_YEARS,
)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class ChannelNormalizer:
    """Normalises patch tensors using per-channel mean and std.

    Statistics are computed over the training set by scripts/build_patches.py
    and saved to NORM_STATS_PATH. This class loads them at dataset creation.
    """

    def __init__(self, stats_path: Path = NORM_STATS_PATH):
        with open(stats_path) as f:
            stats = json.load(f)
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1)
        self.std  = torch.tensor(stats["std"],  dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        """Normalise a (C, H, W) patch tensor."""
        return (patch - self.mean) / (self.std + 1e-8)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def random_augment(patch: np.ndarray) -> np.ndarray:
    """Apply random spatial augmentations to a (C, H, W) patch.

    Safe augmentations only:
      - Random horizontal / vertical flip
      - Random 90° / 180° / 270° rotation

    Spectral jitter is intentionally excluded — corrupting band values would
    destroy the spectral relationships the model is learning.
    """
    if np.random.rand() < 0.5:
        patch = patch[:, :, ::-1].copy()
    if np.random.rand() < 0.5:
        patch = patch[:, ::-1, :].copy()
    k = np.random.randint(0, 4)
    if k > 0:
        patch = np.rot90(patch, k=k, axes=(1, 2)).copy()
    return patch


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FoliagePatchDataset(Dataset):
    """PyTorch Dataset backed by an HDF5 patch archive.

    Args:
        hdf5_path:         Path to the HDF5 archive.
        transform:         Optional callable applied to each patch tensor.
                           Typically a ChannelNormalizer.
        augment:           If True, apply random_augment in __getitem__.
        years:             If provided, only include samples from these years.
        min_confidence:    If provided, exclude samples with confidence below
                           this threshold. Useful for filtering out low-quality
                           synthetic or date-heuristic labels.
        label_sources:     If provided, only include samples whose label_source
                           is in this set. E.g. {'observer'} to exclude synthetic
                           patches entirely.
    """

    def __init__(
        self,
        hdf5_path: Path | str,
        transform:       Optional[Callable] = None,
        augment:         bool = False,
        years:           Optional[list[int]] = None,
        min_confidence:  Optional[float] = None,
        label_sources:   Optional[set[str]] = None,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.transform = transform
        self.augment   = augment

        with h5py.File(self.hdf5_path, "r") as f:
            all_years      = f["years"][:]
            all_labels     = f["labels"][:]
            all_confidence = f["confidence"][:]
            all_sources    = [s.decode() if isinstance(s, bytes) else s
                              for s in f["label_source"][:]]

        # Build index mask from all filters
        mask = np.ones(len(all_years), dtype=bool)

        if years is not None:
            mask &= np.isin(all_years, years)

        if min_confidence is not None:
            mask &= all_confidence >= min_confidence

        if label_sources is not None:
            source_mask = np.array([s in label_sources for s in all_sources])
            mask &= source_mask

        self.indices    = np.where(mask)[0]
        self.labels     = all_labels[self.indices]
        self.confidence = all_confidence[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        archive_idx = int(self.indices[idx])
        label       = int(self.labels[idx])
        confidence  = float(self.confidence[idx])

        with h5py.File(self.hdf5_path, "r") as f:
            patch = f["patches"][archive_idx].astype(np.float32)  # (C, H, W)

        if self.augment:
            patch = random_augment(patch)

        patch_t = torch.from_numpy(patch)

        if self.transform is not None:
            patch_t = self.transform(patch_t)

        return (patch_t,
                torch.tensor(label, dtype=torch.long),
                torch.tensor(confidence, dtype=torch.float32))

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for weighted cross-entropy loss.

        w_c = N / (num_classes * count_c)

        Returns a (num_classes,) float32 tensor.
        """
        counts  = np.bincount(self.labels, minlength=NUM_CLASSES).astype(np.float32)
        counts  = np.maximum(counts, 1)
        weights = len(self.labels) / (NUM_CLASSES * counts)
        return torch.tensor(weights, dtype=torch.float32)

    def weighted_sampler(self) -> WeightedRandomSampler:
        """WeightedRandomSampler that oversamples minority classes.

        Use instead of shuffle=True when class imbalance is severe.
        """
        weights        = self.class_weights()
        sample_weights = weights[torch.tensor(self.labels)]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self),
            replacement=True,
        )

    def confidence_weighted_sampler(self) -> WeightedRandomSampler:
        """WeightedRandomSampler that combines class rebalancing with
        confidence weighting.

        Samples are weighted by (class_weight × confidence), so minority
        class samples with high confidence are seen most frequently.
        This is useful when synthetic low-confidence labels are present
        alongside high-confidence observer labels.
        """
        class_weights  = self.class_weights()
        per_sample_cw  = class_weights[torch.tensor(self.labels)]
        conf_t         = torch.tensor(self.confidence, dtype=torch.float32)
        combined       = per_sample_cw * conf_t
        return WeightedRandomSampler(
            weights=combined,
            num_samples=len(self),
            replacement=True,
        )

    def summary(self) -> dict:
        """Return a summary dict of dataset composition."""
        from collections import Counter
        from config import STAGE_NAMES
        stage_counts = Counter(self.labels.tolist())
        return {
            "total":         len(self),
            "stages":        {STAGE_NAMES[i]: stage_counts.get(i, 0)
                              for i in range(NUM_CLASSES)},
            "mean_confidence": float(self.confidence.mean()),
        }


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def make_dataloaders(
    hdf5_path:       Path | str,
    norm_stats_path: Path = NORM_STATS_PATH,
    batch_size:      int = 64,
    num_workers:     int = 4,
    min_confidence:  Optional[float] = None,
    use_confidence_weighting: bool = False,
) -> dict[str, DataLoader]:
    """Build train, val, and test DataLoaders from a single HDF5 archive.

    Args:
        hdf5_path:                Path to the HDF5 patch archive.
        norm_stats_path:          Path to normalisation statistics JSON.
        batch_size:               Samples per batch.
        num_workers:              DataLoader worker processes.
        min_confidence:           Exclude samples below this confidence.
        use_confidence_weighting: If True, use confidence_weighted_sampler
                                  for training instead of plain class weighting.

    Returns:
        Dict with keys 'train', 'val', 'test' mapping to DataLoaders.
    """
    normalizer = ChannelNormalizer(norm_stats_path)

    splits = {
        "train": (TRAIN_YEARS, True),    # (years, augment)
        "val":   (VAL_YEARS,   False),
        "test":  (TEST_YEARS,  False),
    }

    loaders = {}
    for split_name, (years, augment) in splits.items():
        ds = FoliagePatchDataset(
            hdf5_path=hdf5_path,
            transform=normalizer,
            augment=augment,
            years=years,
            min_confidence=min_confidence,
        )

        info = ds.summary()
        print(f"[dataset] {split_name}: {info['total']} patches, "
              f"mean_conf={info['mean_confidence']:.2f}, "
              f"stages={info['stages']}")

        if split_name == "train":
            sampler = (ds.confidence_weighted_sampler()
                       if use_confidence_weighting
                       else ds.weighted_sampler())
            loaders[split_name] = DataLoader(
                ds,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
            )
        else:
            loaders[split_name] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

    return loaders