"""
spectral_cnn.py — Stage 1 spectral classifier CNN.

Architecture overview:
    - 3 convolutional blocks (Conv → BN → ReLU), each doubling feature maps
    - Global average pooling collapses the spatial dimension
    - 2 fully connected layers with dropout
    - 4-class softmax output

With base_filters=32, the channel progression is:
    Input (14) → Block1 (32) → Block2 (64) → Block3 (128) → GAP → FC(64) → FC(4)

Effective receptive field:
    Each 3×3 conv adds 2px to the receptive field. Three convs → 7×7 effective
    receptive field in the feature map. On a 32×32 input at 250m resolution
    the centre pixel sees a ~1.75km radius neighbourhood, which is appropriate
    for capturing micro-topographic context without absorbing regional climate
    signal that is better encoded as explicit features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MODEL, NUM_CHANNELS, NUM_CLASSES


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU.

    The fundamental building block. Using BN before the activation is the
    standard pre-activation convention; it makes training more stable and
    reduces sensitivity to learning rate choice.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SpectralCNN(nn.Module):
    """Spectral foliage stage classifier.

    Classifies each patch into one of four foliage transition stages:
        0 — no transition (summer green)
        1 — early
        2 — peak
        3 — late

    Args:
        in_channels:  Number of input channels (default: NUM_CHANNELS = 14).
        num_classes:  Number of output classes (default: NUM_CLASSES = 4).
        base_filters: Number of filters in the first conv block. Subsequent
                      blocks double this. (default: 32)
        dropout:      Dropout probability applied before the final FC layer.
    """

    def __init__(
        self,
        in_channels: int = NUM_CHANNELS,
        num_classes:  int = NUM_CLASSES,
        base_filters: int = 32,
        dropout:      float = 0.3,
    ):
        super().__init__()

        f = base_filters  # shorthand

        # --- Convolutional backbone ---
        # Each block: 3×3 conv → BN → ReLU, with 2×2 max-pool after blocks 1 and 2
        # Max-pooling grows the receptive field rapidly without adding parameters.
        self.block1 = ConvBlock(in_channels, f)
        self.pool1  = nn.MaxPool2d(2, 2)          # 32×32 → 16×16

        self.block2 = ConvBlock(f, f * 2)
        self.pool2  = nn.MaxPool2d(2, 2)          # 16×16 → 8×8

        self.block3 = ConvBlock(f * 2, f * 4)
        # No pooling after block3 — GAP handles spatial collapse

        # --- Global average pooling ---
        # Collapses (f*4, H, W) → (f*4,) regardless of input spatial size.
        # This makes the model invariant to small input size changes and
        # regularises against spatial overfitting.
        self.gap = nn.AdaptiveAvgPool2d(1)

        # --- Classifier head ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f * 4, f * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(f * 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming uniform initialisation for conv layers, zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, H, W) input patch tensor, normalised.

        Returns:
            logits: (B, num_classes) unnormalised class scores.
                    Apply softmax for probabilities; cross-entropy loss
                    expects raw logits.
        """
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        x = self.gap(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax class probabilities. Convenience method for inference."""
        with torch.no_grad():
            logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return argmax class predictions (integer stage indices)."""
        return self.predict_proba(x).argmax(dim=-1)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: Path | str, extra: Optional[dict] = None):
        """Save model weights and config to a .pt checkpoint.

        Args:
            path:  Output path for the checkpoint file.
            extra: Optional dict of extra metadata to include (e.g. val accuracy).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "model_config": {
                "in_channels":  self.block1.block[0].in_channels,
                "num_classes":  self.classifier[-1].out_features,
                "base_filters": self.block1.block[0].out_channels,
                "dropout":      self.classifier[-2].p,
            },
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"[model] Checkpoint saved → {path}")

    @classmethod
    def load(cls, path: Path | str, map_location: str = "cpu") -> "SpectralCNN":
        """Load a SpectralCNN from a .pt checkpoint."""
        path = Path(path)
        payload = torch.load(path, map_location=map_location)
        cfg = payload["model_config"]
        model = cls(**cfg)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        print(f"[model] Loaded checkpoint from {path}")
        return model

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict = MODEL) -> SpectralCNN:
    """Construct a SpectralCNN from the config dict."""
    return SpectralCNN(
        in_channels=cfg["in_channels"],
        num_classes=cfg["num_classes"],
        base_filters=cfg["base_filters"],
        dropout=cfg["dropout"],
    )
