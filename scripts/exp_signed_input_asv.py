"""
Experiment (supervisor Q iii/a/b): does giving the network the SIGN of returns
let it estimate the leverage parameter rho?

The default input transform log(r^2) is sign-blind, which likely explains why the
correct-model ASV network cannot recover rho (RMSE 0.407, corr 0.067 with truth —
near-constant output). This trains an otherwise-identical ASV TCN with a two-channel
input [log(r^2), sign(r)], restoring the sign, and compares rho recovery.

Same architecture and hyperparameters as the baseline correct-model ASV network;
only the input representation differs, so the comparison is clean.

Trains, then evaluates on the ASV misspecification test set (estimation window).

Run:  uv run python scripts/exp_signed_input_asv.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.dataset import SVDataset
from src.models.tcn import SVTCNNet
from src.models.train import TrainConfig, train
from src.simulation.sv_params import SVLeverageParams

HPARAMS = dict(n_channels=32, kernel_size=7, n_blocks=6, dropout=0.0)
LR, BATCH = 3e-4, 256
RUN = "asv_correct_signed_T1000"
REPO = Path(__file__).resolve().parent.parent


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    device = get_device()
    cfg = SVLeverageParams()
    print(f"Training signed-input ASV TCN (device={device})")

    train_ds = SVDataset(str(REPO / "data/train_asv_T1000.npz"), config=cfg)
    val_ds = SVDataset(str(REPO / "data/val_asv_T1000.npz"), config=cfg)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    model = SVTCNNet(**HPARAMS, n_outputs=4, signed_input=True)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters())}")
    tc = TrainConfig(epochs=100, batch_size=BATCH, lr=LR, patience=15,
                     device=device, checkpoint_dir=str(REPO / "checkpoints"))
    res = train(model, train_loader, val_loader, tc, run_name=RUN)
    print(f"Best val loss {res.best_val_loss:.4f} @ epoch {res.best_epoch}")

    # ---- evaluate rho recovery on the ASV test set ----
    data = np.load("data/test_misspec_asv.npz")
    r = data["returns"][:, :1000].astype(np.float32)
    true = data["params"].astype(np.float32)   # (200,4): mu,phi,sigma,rho
    model.load_state_dict(torch.load(REPO / f"checkpoints/{RUN}/best.pt",
                                     map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        preds_t = model(torch.from_numpy(r)).cpu().numpy()
    est = cfg.inverse_transform(preds_t.astype(np.float64)).astype(np.float32)
    np.savez_compressed("results/misspec/asv_tcn_correct_signed.npz", est=est, true=true)

    names = ["mu", "phi", "sigma_eta", "rho"]
    print("\n" + "=" * 60)
    print("SIGNED-INPUT ASV — rho recovery vs sign-blind baseline")
    print("=" * 60)
    rmse = np.sqrt(((est - true) ** 2).mean(axis=0))
    for i, n in enumerate(names):
        c = np.corrcoef(est[:, i], true[:, i])[0, 1]
        print(f"  {n:>10}: RMSE={rmse[i]:.4f}  corr(est,true)={c:+.3f}")
    print("-" * 60)
    print(f"  baseline (log r^2 only): rho RMSE=0.407  corr=0.067")
    print(f"  MCMC reference:          rho RMSE=0.150  corr=0.933")


if __name__ == "__main__":
    main()
