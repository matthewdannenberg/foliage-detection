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

        # Class weights — used only if use_weighted_loss=True in cfg.
        # When using WeightedRandomSampler, weighted loss double-counts the
        # imbalance correction and destabilizes training. Default is unweighted.
        if cfg.get("use_weighted_loss", False):
            if class_weights is None:
                train_ds = loaders["train"].dataset
                class_weights = train_ds.class_weights()
            self.class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(
                weight=self.class_weights,
                reduction="none",
                label_smoothing=cfg.get("label_smoothing", 0.1),
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                reduction="none",
                label_smoothing=cfg.get("label_smoothing", 0.1),
            )
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

        # Report final validation performance using the best checkpoint
        print("\n[trainer] Loading best checkpoint for final validation report ...")
        best_path = self.checkpoint_dir / "best.pt"
        if best_path.exists():
            self.model = SpectralCNN.load(best_path, map_location=str(self.device))
            self.model.to(self.device)
        self.evaluate(split="val")

        return self.history

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """One training epoch. Returns metrics dict."""
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for patches, labels, confidence in self.loaders["train"]:
            patches    = patches.to(self.device)
            labels     = labels.to(self.device)
            confidence = confidence.to(self.device)

            self.optimiser.zero_grad(set_to_none=True)
            logits = self.model(patches)

            # Per-sample loss weighted by label confidence.
            # Higher-confidence labels (observer originals) contribute more
            # to the gradient than lower-confidence ones (shifted, synthetic).
            per_sample_loss = self.criterion(logits, labels)   # (B,)
            loss = (per_sample_loss * confidence).mean()
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

        for patches, labels, confidence in self.loaders[split]:
            patches = patches.to(self.device)
            labels  = labels.to(self.device)

            logits = self.model(patches)
            # For evaluation use unweighted mean loss for a stable metric
            loss   = self.criterion(logits, labels).mean()

            total_loss += loss.item() * len(labels)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        n = len(self.loaders[split].dataset)
        preds  = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        metrics = per_class_accuracy(preds, labels)
        metrics["loss"] = total_loss / n
        return metrics

    def evaluate(self, split: str = "val") -> dict[str, float]:
        """Run evaluation on the given split and print a detailed report
        including per-class accuracy and a full confusion matrix.

        Args:
            split: One of 'val' or 'test'. During development, always use
                   'val'. Use 'test' only once, via evaluate.py, when all
                   development decisions are final.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for patches, labels, confidence in self.loaders[split]:
                patches = patches.to(self.device)
                labels  = labels.to(self.device)
                logits  = self.model(patches)
                loss    = self.criterion(logits, labels).mean()
                total_loss += loss.item() * len(labels)
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        n      = len(self.loaders[split].dataset)
        preds  = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)

        metrics = per_class_accuracy(preds, labels)
        metrics["loss"] = total_loss / n

        print(f"\n[trainer] {split} set results:")
        for k, v in metrics.items():
            if not np.isnan(v):
                print(f"  {k:20s}: {v:.4f}")

        # Confusion matrix — rows = true class, columns = predicted class
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for t, p in zip(labels, preds):
            cm[t, p] += 1

        col_w = 14
        print(f"\n[trainer] Confusion matrix (rows=true, cols=predicted):")
        header = f"  {'':20s}" + "".join(
            f"{STAGE_NAMES[j]:>{col_w}}" for j in range(NUM_CLASSES)
        )
        print(header)
        print("  " + "-" * (20 + col_w * NUM_CLASSES))
        for i in range(NUM_CLASSES):
            row_total = cm[i].sum()
            row = f"  {STAGE_NAMES[i]:20s}"
            for j in range(NUM_CLASSES):
                count = cm[i, j]
                pct   = 100 * count / row_total if row_total > 0 else 0
                cell  = f"{count}({pct:.0f}%)"
                row  += f"{cell:>{col_w}}"
            print(row)

        return metrics