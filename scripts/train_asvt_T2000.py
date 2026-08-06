"""
Train the ASV-t TCN with sign-preserving input at T=2000, for the empirical
application (Chapter 7), where the in-sample window is ~2000 trading days.

The base-SV TCN already exists at T=2000 (checkpoints/tcn_best_T2000). This adds
the correctly-specified ASV-t network at T=2000 with the two-channel
[log(r^2), sign(r)] input established in Chapter 6. Generates the T=2000 ASV-t
training/validation data first if not already present (fresh seeds 2005/2006).

Checkpoint: checkpoints/asvt_correct_sign_T2000/best.pt

Run:  uv run python scripts/train_asvt_T2000.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.simulation.simulator import simulate_asv_t, save_dataset
from src.models.dataset import SVDataset
from src.models.tcn import SVTCNNet
from src.models.train import TrainConfig, train
from src.simulation.sv_params import ASVtParams

REPO = Path(__file__).resolve().parent.parent
T = 2000
HPARAMS = dict(n_channels=32, kernel_size=7, n_blocks=6, dropout=0.0)
LR, BATCH = 3e-4, 256
RUN = "asvt_correct_sign_T2000"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ensure_data():
    tr = REPO / f"data/train_asvt_T{T}.npz"
    va = REPO / f"data/val_asvt_T{T}.npz"
    if not tr.exists():
        print(f"Generating ASV-t train (N=90000, T={T}, seed=2005)...")
        save_dataset(str(tr), simulate_asv_t(N=90_000, T=T, seed=2005))
    if not va.exists():
        print(f"Generating ASV-t val (N=10000, T={T}, seed=2006)...")
        save_dataset(str(va), simulate_asv_t(N=10_000, T=T, seed=2006))


def main():
    ensure_data()
    device = get_device()
    cfg = ASVtParams()
    print(f"Training {RUN} (device={device})")

    train_ds = SVDataset(str(REPO / f"data/train_asvt_T{T}.npz"), config=cfg)
    val_ds = SVDataset(str(REPO / f"data/val_asvt_T{T}.npz"), config=cfg)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    model = SVTCNNet(**HPARAMS, n_outputs=5, second_channel="sign")
    tc = TrainConfig(epochs=100, batch_size=BATCH, lr=LR, patience=15,
                     device=device, checkpoint_dir=str(REPO / "checkpoints"))
    res = train(model, train_loader, val_loader, tc, run_name=RUN)
    print(f"Done. Best val loss {res.best_val_loss:.4f} @ epoch {res.best_epoch}. "
          f"Checkpoint: checkpoints/{RUN}/best.pt")


if __name__ == "__main__":
    main()
