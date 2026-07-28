"""
Train the correctly-specified (cell c) TCNs for the misspecification analysis.

Same validated TCN architecture as the base model (n_channels=32, kernel_size=7,
n_blocks=6, dropout=0.0, lr=3e-4, batch_size=256), changed ONLY in the output
layer and the target transform:

  asv   -> 4 outputs [mu, phi, sigma_eta, rho],       config SVLeverageParams
  svt   -> 4 outputs [mu, phi, sigma_eta, nu],        config SVtParams
  asvt  -> 5 outputs [mu, phi, sigma_eta, rho, nu],   config ASVtParams

This keeps the (c)-vs-(a) comparison clean: same network, only the target model
differs. Checkpoints saved to checkpoints/{model}_correct_T1000/best.pt.

Usage:
  uv run python scripts/train_misspec_models.py --model asv
  uv run python scripts/train_misspec_models.py --model svt
  uv run python scripts/train_misspec_models.py --model asvt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.dataset import SVDataset
from src.models.tcn import SVTCNNet
from src.models.train import TrainConfig, train
from src.simulation.sv_params import SVLeverageParams, SVtParams, ASVtParams

# Best base-TCN hyperparameters (reused unchanged).
HPARAMS = dict(n_channels=32, kernel_size=7, n_blocks=6, dropout=0.0)
LR = 3e-4
BATCH_SIZE = 256

MODELS = {
    "asv":  (SVLeverageParams, 4),
    "svt":  (SVtParams,        4),
    "asvt": (ASVtParams,       5),
}


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    config_cls, n_outputs = MODELS[args.model]
    config = config_cls()
    device = get_device()
    repo = Path(__file__).resolve().parent.parent

    print(f"Training correct-model TCN: {args.model}  "
          f"(n_outputs={n_outputs}, device={device})")

    train_ds = SVDataset(str(repo / f"data/train_{args.model}_T1000.npz"), config=config)
    val_ds   = SVDataset(str(repo / f"data/val_{args.model}_T1000.npz"),   config=config)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=512,        shuffle=False, num_workers=0)

    model = SVTCNNet(**HPARAMS, n_outputs=n_outputs)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params}")

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        lr=LR,
        patience=15,
        device=device,
        checkpoint_dir=str(repo / "checkpoints"),
    )
    run_name = f"{args.model}_correct_T1000"
    result = train(model, train_loader, val_loader, cfg, run_name=run_name)
    print(f"Done. Best val loss {result.best_val_loss:.4f} at epoch {result.best_epoch}. "
          f"Checkpoint: checkpoints/{run_name}/best.pt")


if __name__ == "__main__":
    main()
