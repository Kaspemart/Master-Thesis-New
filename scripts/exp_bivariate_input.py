"""
Bivariate-input experiment (supervisor request): retrain the leverage models
(ASV, ASV-t) with a two-channel input that restores the sign of returns, which
the log(r^2) transform discards.

  --channel raw   -> [log(r^2), r]        (supervisor's suggestion: raw returns)
  --channel sign  -> [log(r^2), sign(r)]  (sign only; magnitude is already in log r^2)

Same architecture and hyperparameters as the sign-blind correct-model networks;
only the input representation changes, so the comparison is clean. Trains, then
evaluates parameter recovery (esp. rho) on the misspecification test set.

Usage:
  uv run python scripts/exp_bivariate_input.py --model asv  --channel raw
  uv run python scripts/exp_bivariate_input.py --model asvt --channel raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.dataset import SVDataset
from src.models.tcn import SVTCNNet
from src.models.train import TrainConfig, train
from src.simulation.sv_params import SVLeverageParams, ASVtParams

HPARAMS = dict(n_channels=32, kernel_size=7, n_blocks=6, dropout=0.0)
LR, BATCH = 3e-4, 256
REPO = Path(__file__).resolve().parent.parent

SPEC = {
    "asv":  (SVLeverageParams, 4, ["mu", "phi", "sigma_eta", "rho"]),
    "asvt": (ASVtParams,       5, ["mu", "phi", "sigma_eta", "rho", "nu"]),
}
# sign-blind baseline (correct-model) references for comparison
BASELINE = {
    "asv":  dict(rho=(0.407, 0.067)),
    "asvt": dict(rho=(0.407, None)),
}


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(SPEC), required=True)
    ap.add_argument("--channel", choices=["raw", "sign"], required=True)
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    config_cls, n_outputs, names = SPEC[args.model]
    cfg = config_cls()
    device = get_device()
    run = f"{args.model}_correct_{args.channel}_T1000"
    print(f"Training {args.model} with second_channel={args.channel} (device={device})")

    train_ds = SVDataset(str(REPO / f"data/train_{args.model}_T1000.npz"), config=cfg)
    val_ds = SVDataset(str(REPO / f"data/val_{args.model}_T1000.npz"), config=cfg)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    model = SVTCNNet(**HPARAMS, n_outputs=n_outputs, second_channel=args.channel)
    tc = TrainConfig(epochs=args.epochs, batch_size=BATCH, lr=LR, patience=15,
                     device=device, checkpoint_dir=str(REPO / "checkpoints"))
    res = train(model, train_loader, val_loader, tc, run_name=run)
    print(f"Best val loss {res.best_val_loss:.4f} @ epoch {res.best_epoch}")

    # ---- evaluate on the misspecification test set (CPU) ----
    data = np.load(f"data/test_misspec_{args.model}.npz")
    r = torch.from_numpy(data["returns"][:, :1000].astype(np.float32))
    true = data["params"].astype(np.float32)
    model = SVTCNNet(**HPARAMS, n_outputs=n_outputs, second_channel=args.channel)
    model.load_state_dict(torch.load(REPO / f"checkpoints/{run}/best.pt",
                                     map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        est = cfg.inverse_transform(model(r).numpy().astype(np.float64)).astype(np.float32)
    np.savez_compressed(f"results/misspec/{args.model}_tcn_correct_{args.channel}.npz",
                        est=est, true=true)

    print("=" * 60)
    print(f"{args.model} bivariate ({args.channel}) — parameter recovery")
    print("=" * 60)
    rmse = np.sqrt(((est - true) ** 2).mean(axis=0))
    for i, n in enumerate(names):
        c = np.corrcoef(est[:, i], true[:, i])[0, 1]
        print(f"  {n:>10}: RMSE={rmse[i]:.4f}  corr={c:+.3f}  est_std={est[:, i].std():.3f}")
    print(f"  [sign-blind baseline rho: RMSE=0.407, corr~0.07; MCMC rho: RMSE=0.150]")


if __name__ == "__main__":
    main()
