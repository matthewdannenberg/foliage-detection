"""
train_spectral.py — Entry point for training the Stage 1 spectral classifier.

Usage:
    python scripts/train_spectral.py
    python scripts/train_spectral.py --hdf5 data/processed/patches/patches.h5
    python scripts/train_spectral.py --epochs 100 --batch-size 128 --run-name exp01
"""

import argparse
import sys
from pathlib import Path

# Make the src package importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import PATCHES_DIR, TRAIN
from data.dataset import make_dataloaders
from models.spectral_cnn import build_model
from train.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Train the Stage 1 foliage spectral classifier.")
    p.add_argument(
        "--hdf5",
        type=Path,
        default=PATCHES_DIR / "patches.h5",
        help="Path to the HDF5 patch archive (default: data/processed/patches/patches.h5)",
    )
    p.add_argument("--epochs",     type=int,   default=TRAIN["epochs"])
    p.add_argument("--batch-size", type=int,   default=TRAIN["batch_size"])
    p.add_argument("--lr",         type=float, default=TRAIN["learning_rate"])
    p.add_argument("--workers",    type=int,   default=TRAIN["num_workers"])
    p.add_argument("--device",     type=str,   default=None,
                   help="'cuda', 'mps', or 'cpu'. Auto-detected if omitted.")
    p.add_argument("--run-name",   type=str,   default=None,
                   help="Optional TensorBoard run name.")
    p.add_argument("--resume",     type=Path,  default=None,
                   help="Path to a checkpoint to resume from.")
    p.add_argument("--confidence-weighting", action="store_true",
                   help="Weight training samples by class balance × confidence score.")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.hdf5.exists():
        print(f"[error] Patch archive not found: {args.hdf5}")
        print("  Run scripts/build_patches.py first.")
        sys.exit(1)

    # Override config with CLI args
    cfg = dict(TRAIN)
    cfg["epochs"]        = args.epochs
    cfg["batch_size"]    = args.batch_size
    cfg["learning_rate"] = args.lr
    cfg["num_workers"]   = args.workers

    print(f"[train] Loading patches from {args.hdf5}")
    loaders = make_dataloaders(
        hdf5_path=args.hdf5,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        use_confidence_weighting=args.confidence_weighting,
    )

    model = build_model()
    print(f"[train] Model: {model.parameter_count():,} trainable parameters")

    if args.resume:
        from models.spectral_cnn import SpectralCNN
        model = SpectralCNN.load(args.resume)
        print(f"[train] Resumed from {args.resume}")

    trainer = Trainer(
        model=model,
        loaders=loaders,
        cfg=cfg,
        device=args.device,
        run_name=args.run_name,
    )

    history = trainer.fit()

    print(
        "\n[train] Done."
        "\n  Best checkpoint : checkpoints/best.pt"
        "\n  To evaluate on the test set run: python scripts/evaluate.py"
    )


if __name__ == "__main__":
    main()