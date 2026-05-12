"""
Train CNN and LSTM estimators on a single T-value dataset and compare
validation loss. Saves best weights for each architecture.

Usage:
    uv run python scripts/train_nn.py --T 1000
    uv run python scripts/train_nn.py --T 500 --epochs 50
    uv run python scripts/train_nn.py --T 1000 --arch cnn   # single arch only

Results are saved to:
    checkpoints/<arch>_T<T>/best.pt          model weights
    checkpoints/<arch>_T<T>/train_result.npz train/val loss curves
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import SVConvNet, SVLSTMNet, SVDataset, TrainConfig, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T",      type=int, default=1000, choices=[500, 1000, 2000])
    parser.add_argument("--arch",   type=str, default="both", choices=["cnn", "lstm", "both"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch",  type=int, default=512)
    parser.add_argument("--lr",     type=float, default=1e-3)
    args = parser.parse_args()

    device = get_device()
    logger.info("Device: %s", device)

    repo = Path(__file__).parent.parent
    train_path = repo / f"data/train_T{args.T}.npz"
    val_path   = repo / f"data/val_T{args.T}.npz"

    logger.info("Loading datasets for T=%d ...", args.T)
    train_ds = SVDataset(str(train_path))
    val_ds   = SVDataset(str(val_path))
    logger.info("Train: %d series  Val: %d series", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=device,
        checkpoint_dir=str(repo / "checkpoints"),
    )

    archs = {"cnn": SVConvNet, "lstm": SVLSTMNet}
    to_run = ["cnn", "lstm"] if args.arch == "both" else [args.arch]
    summary = {}

    for arch_name in to_run:
        logger.info("=" * 50)
        logger.info("Training %s on T=%d", arch_name.upper(), args.T)
        logger.info("=" * 50)

        model = archs[arch_name]()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Parameters: %s  Trainable: %d", arch_name.upper(), n_params)

        t0 = time.time()
        result = train(model, train_loader, val_loader, config,
                       run_name=f"{arch_name}_T{args.T}")
        elapsed = time.time() - t0

        # Save loss curves
        out_dir = Path(config.checkpoint_dir) / f"{arch_name}_T{args.T}"
        np.savez(
            out_dir / "train_result.npz",
            train_losses=np.array(result.train_losses),
            val_losses=np.array(result.val_losses),
            best_epoch=np.array(result.best_epoch),
            best_val_loss=np.array(result.best_val_loss),
        )

        summary[arch_name] = result
        logger.info("%s done in %.1fs  best_val=%.5f at epoch %d",
                    arch_name.upper(), elapsed, result.best_val_loss, result.best_epoch)

    # Comparison summary
    if len(to_run) > 1:
        logger.info("")
        logger.info("=== Architecture comparison (T=%d) ===", args.T)
        for name, res in summary.items():
            logger.info("  %-6s  best_val=%.5f  best_epoch=%d",
                        name.upper(), res.best_val_loss, res.best_epoch)
        winner = min(summary, key=lambda k: summary[k].best_val_loss)
        logger.info("  Winner: %s", winner.upper())


if __name__ == "__main__":
    main()
