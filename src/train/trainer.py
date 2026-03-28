"""
trainer.py — Training loop for the Stage 1 spectral classifier.

Handles:
  - Cross-entropy loss with per-class weights (addresses class imbalance)
  - AdamW optimiser with cosine LR schedule
  - Epoch-level validation with early stopping on val loss
  - Per-class accuracy logging
  - Checkpointing best and last models
  - Optional TensorBoard logging (gracefully skipped if not installed)

Typical usage (see also scripts/train_spectral.py):
    loaders = make_dataloaders(hdf5_path)
    model   = build_model()
    trainer = Trainer(model, loaders)
    trainer.fit()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config import NUM_CLASSES, STAGE_NAMES, TRAIN
from models.spectral_cnn import SpectralCNN

# TensorBoard is optional
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def per_class_accuracy(
    preds: np.ndarray, labels: np.ndarray, num_classes: int = NUM_CLASSES
) -> dict[str, float]:
    """Compute per-class accuracy from flat arrays of predictions and labels.

    Returns a dict mapping stage names to accuracy floats, plus "overall".
    """
    result = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            result[STAGE_NAMES[c]] = float("nan")
        else:
            result[STAGE_NAMES[c]] = float((preds[mask] == c).mean())
    result["overall"] = float((preds == labels).mean())
    return result


def format_metrics(metrics: dict[str, float]) -> str:
    """Format a metrics dict as a compact string for console logging."""
    parts = [f"{k}={v:.3f}" for k, v in metrics.items() if not np.isnan(v)]
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when val loss stops improving for `patience` epochs."""

    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter   = 0
        self.triggered = False

    def step(self, val_loss: float) -> bool:
        """Update state. Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Encapsulates the full training and evaluation loop.

    Args:
        model:       SpectralCNN instance.
        loaders:     Dict with keys 'train', 'val', 'test' → DataLoader.
        class_weights: (num_classes,) float32 tensor for weighted CE loss.
                       If None, retrieved from the training dataset.
        cfg:         Training hyperparameter dict (defaults to config.TRAIN).
        device:      'cuda', 'mps', or 'cpu'. Auto-detected if None.
        run_name:    Optional name for the TensorBoard run.
    """

    def __init__(
        self,
        model: SpectralCNN,
        loaders: dict[str, DataLoader],
        class_weights: Optional[torch.Tensor] = None,
        cfg: dict = TRAIN,
        device: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        self.model   = model
        self.loaders = loaders
        self.cfg     = cfg

        # Device selection
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        print(f"[trainer] Using device: {self.device}")
        self.model.to(self.device)

        # Class weights
        if class_weights is None:
            train_ds = loaders["train"].dataset
            class_weights = train_ds.class_weights()
        self.class_weights = class_weights.to(self.device)

        # Loss, optimiser, scheduler
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.optimiser = AdamW(
            self.model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(
            self.optimiser,
            T_max=cfg["epochs"],
            eta_min=cfg["learning_rate"] / 50,
        )

        # Checkpointing
        self.checkpoint_dir = Path(cfg["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = None
        if _TB_AVAILABLE:
            log_dir = Path(cfg["log_dir"])
            if run_name:
                log_dir = log_dir / run_name
            self.writer = SummaryWriter(log_dir=str(log_dir))

        self.early_stopping = EarlyStopping(patience=cfg["patience"])
        self.best_val_loss  = float("inf")
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def fit(self) -> list[dict]:
        """Run the full training loop.

        Returns:
            history: List of per-epoch metric dicts.
        """
        epochs = self.cfg["epochs"]
        print(
            f"[trainer] Starting training: {epochs} epochs, "
            f"{self.model.parameter_count():,} parameters."
        )

        for epoch in range(1, epochs + 1):
            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._eval_epoch("val", epoch)

            self.scheduler.step()

            # Log to TensorBoard
            if self.writer:
                for k, v in train_metrics.items():
                    if not np.isnan(v):
                        self.writer.add_scalar(f"train/{k}", v, epoch)
                for k, v in val_metrics.items():
                    if not np.isnan(v):
                        self.writer.add_scalar(f"val/{k}", v, epoch)
                self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], epoch)

            # Console log
            lr = self.scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train_loss={train_metrics['loss']:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  "
                f"val_acc={val_metrics['overall']:.3f}  "
                f"lr={lr:.2e}"
            )

            # Checkpoint
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.model.save(
                    self.checkpoint_dir / "best.pt",
                    extra={"epoch": epoch, "val_loss": val_metrics["loss"]},
                )
            self.model.save(self.checkpoint_dir / "last.pt", extra={"epoch": epoch})

            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            self.history.append(row)

            # Early stopping
            if self.early_stopping.step(val_metrics["loss"]):
                print(f"[trainer] Early stopping triggered at epoch {epoch}.")
                break

        if self.writer:
            self.writer.close()

        return self.history

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """One training epoch. Returns metrics dict."""
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for patches, labels in self.loaders["train"]:
            patches = patches.to(self.device)
            labels  = labels.to(self.device)

            self.optimiser.zero_grad(set_to_none=True)
            logits = self.model(patches)
            loss   = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping prevents exploding gradients in edge cases
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimiser.step()

            total_loss += loss.item() * len(labels)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        n = len(self.loaders["train"].dataset)
        preds  = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        metrics = per_class_accuracy(preds, labels)
        metrics["loss"] = total_loss / n
        return metrics

    @torch.no_grad()
    def _eval_epoch(self, split: str, epoch: int) -> dict[str, float]:
        """One evaluation pass over split ('val' or 'test'). Returns metrics dict."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for patches, labels in self.loaders[split]:
            patches = patches.to(self.device)
            labels  = labels.to(self.device)

            logits = self.model(patches)
            loss   = self.criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        n = len(self.loaders[split].dataset)
        preds  = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        metrics = per_class_accuracy(preds, labels)
        metrics["loss"] = total_loss / n
        return metrics

    def evaluate_test(self) -> dict[str, float]:
        """Run evaluation on the test split and print a detailed report."""
        metrics = self._eval_epoch("test", epoch=0)
        print("\n[trainer] Test set results:")
        for k, v in metrics.items():
            if not np.isnan(v):
                print(f"  {k:20s}: {v:.4f}")
        return metrics
