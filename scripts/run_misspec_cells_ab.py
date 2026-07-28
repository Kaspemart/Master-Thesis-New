"""
Cells (a) and (b) of the misspecification analysis: the MISSPECIFIED estimators.

Both fit the BASE SV model to data from the three misspecified DGPs (ASV, SV-t,
ASV-t), using only the estimation window (first T_estim=1000 observations).

  (a) base-SV TCN  — the trained checkpoints/tcn_best_T1000/best.pt
  (b) base-SV stochvol

Outputs per scenario/method:
  results/misspec/{scenario}_{method}.npz
    est      (200, 3)  float32  — estimated [mu, phi, sigma_eta]
    true     (200, 3)  float32  — true base params of the DGP
    (stochvol also stores rhat)

RMSE and bias vs the true base params are printed. These measure the DISTORTION
the missing feature induces (not error in the usual sense). Predictive
log-likelihood is computed separately (run_misspec_predictive_ll.py).

Usage:
  uv run python scripts/run_misspec_cells_ab.py --cell a      # TCN (fast)
  uv run python scripts/run_misspec_cells_ab.py --cell b      # stochvol (slow)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.models import SVTCNNet
from src.models.train import predict
from src.simulation.sv_params import SVParams
from src.estimation.stochvol_runner import StochvolConfig, run_stochvol_batch
from src.simulation.simulator import SimulationResult

SCENARIOS = {
    "asv":  "data/test_misspec_asv.npz",
    "svt":  "data/test_misspec_svt.npz",
    "asvt": "data/test_misspec_asvt.npz",
}
T_ESTIM = 1000
OUT = Path("results/misspec")
TCN_CKPT = "checkpoints/tcn_best_T1000/best.pt"


def _metrics(est, true):
    """RMSE and bias per base parameter [mu, phi, sigma_eta]."""
    err = est - true
    rmse = np.sqrt((err ** 2).mean(axis=0))
    bias = err.mean(axis=0)
    return rmse, bias


def _print_metrics(tag, est, true):
    rmse, bias = _metrics(est, true)
    names = ["mu", "phi", "sigma_eta"]
    print(f"  {tag}:")
    for i, nm in enumerate(names):
        print(f"    {nm:>10}: RMSE={rmse[i]:.4f}  bias={bias[i]:+.4f}")


def run_cell_a():
    """Base-SV TCN on each misspecified scenario's estimation window."""
    device = torch.device("cpu")
    model = SVTCNNet(n_channels=32, kernel_size=7, n_blocks=6, dropout=0.0)
    model.load_state_dict(torch.load(TCN_CKPT, map_location=device, weights_only=True))
    model.eval()
    cfg = SVParams()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CELL (a) — base-SV TCN on misspecified data")
    print("=" * 60)
    for scen, path in SCENARIOS.items():
        data = np.load(path)
        r = data["returns"][:, :T_ESTIM].astype(np.float32)   # estimation window
        true = data["params"][:, :3].astype(np.float32)       # base params
        with torch.no_grad():
            preds_t = model(torch.from_numpy(r)).cpu().numpy()  # unconstrained
        est = cfg.inverse_transform(preds_t.astype(np.float64)).astype(np.float32)
        np.savez_compressed(OUT / f"{scen}_tcn.npz", est=est, true=true)
        _print_metrics(f"{scen}  (TCN)", est, true)


def run_cell_b():
    """Base-SV stochvol on each misspecified scenario's estimation window."""
    print("=" * 60)
    print("CELL (b) — base-SV stochvol on misspecified data")
    print("=" * 60)
    config = StochvolConfig(draws=1000, burnin=1000, n_jobs=4)
    for scen, path in SCENARIOS.items():
        data = np.load(path)
        r = data["returns"][:, :T_ESTIM].astype(np.float32)
        true = data["params"][:, :3].astype(np.float32)
        ds = SimulationResult(returns=r, params=true,
                              latent_h=np.zeros_like(r))
        out_dir = OUT / f"{scen}_stochvol"
        print(f"  running stochvol on {scen} ({r.shape[0]} series)...")
        run_stochvol_batch(ds, config, out_dir)
        res = np.load(out_dir / "results.npz")
        est = res["means"].astype(np.float32)
        np.savez_compressed(OUT / f"{scen}_stochvol.npz",
                            est=est, true=true, rhat=res["rhats"])
        _print_metrics(f"{scen}  (stochvol)", est, true)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["a", "b"], required=True)
    args = ap.parse_args()
    if args.cell == "a":
        run_cell_a()
    else:
        run_cell_b()
