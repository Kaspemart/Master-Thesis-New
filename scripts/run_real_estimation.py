"""
Empirical application (Chapter 7): estimate SV and ASV-t on each real asset via
both MCMC and the TCN, on the in-sample window, then evaluate out-of-sample
predictive log-likelihood.

For each of the 15 assets:
  - returns are demeaned by the in-sample mean (the SV model is mean-zero, and the
    networks were trained on mean-zero simulated returns);
  - SV model     : estimate via base-SV MCMC (stochvol) and base-SV TCN (T=2000);
  - ASV-t model  : estimate via ASV-t MCMC (stochvol, correct-model runner) and
                   ASV-t sign-input TCN (T=2000);
  - out-of-sample predictive log-likelihood is computed with the particle filter
    over the full series, split at the in/out boundary (no look-ahead: the OOS
    likelihood is conditioned on the in-sample state, using the in-sample mean).

MCMC uses the same simulation-consistent priors as Chapters 5-6, so both methods
are confined to the same parameter ranges (fair comparison). Results saved per
asset (crash-safe) to results/real/{asset}.json; a summary table is printed.

Run:  uv run python scripts/run_real_estimation.py
      uv run python scripts/run_real_estimation.py --summary   # just reprint table
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.data.real_data import ALL_ASSETS, ASSET_GROUPS, LABELS, load_returns, OOS_START
from src.models.tcn import SVTCNNet
from src.simulation.sv_params import SVParams, ASVtParams
from src.evaluation.particle_filter import sv_log_likelihood

REPO = Path(__file__).resolve().parent.parent
OUT = Path("results/real"); OUT.mkdir(parents=True, exist_ok=True)
# Per-model MCMC draws. base SV converges at 1000/1000 (R-hat < 1.04). The ASV-t
# general sampler mixes slowly on weak-leverage assets (rho near 0): 3000/3000 left
# R-hat up to 1.34 on five assets and biased nu. 8000/6000 brings every asset under
# R-hat 1.1. Affordable here -- only 15 assets, one run each.
DRAWS_BURNIN = {"sv": (1000, 1000), "asvt": (8000, 6000)}
# The two most persistent FX vols (phi ~ 0.99, near unit root) mix slowly on phi;
# 8000 draws left R-hat ~1.15/1.11. 20000/10000 brings them under 1.1.
ASVT_DRAWS_OVERRIDE = {"DEXUSEU": (20000, 10000), "DEXSZUS": (20000, 10000)}
N_PARTICLES = 10_000
SEEDS = (0, 1, 2)

# The TCN is length-invariant (global average pooling), so we use the network with
# the best held-out parameter recovery for each model, applied to the ~2000-obs real
# windows. base SV: the T=2000 network (best on sigma, validated in Ch5). ASV-t: the
# T=1000 network -- training the ASV-t network directly at T=2000 suffered a
# multi-task optimisation pathology that collapsed the leverage parameter rho (corr
# 0.47), whereas the T=1000 network recovers rho at corr 0.92 on 2000-obs series and
# matches on all other parameters.
TCN_SPEC = {
    "sv":   dict(ckpt="checkpoints/tcn_best_T2000/best.pt",
                 n_outputs=3, second=None,   cfg=SVParams()),
    "asvt": dict(ckpt="checkpoints/asvt_correct_sign_T1000/best.pt",
                 n_outputs=5, second="sign", cfg=ASVtParams()),
}
R_RUNNER = {
    "sv":   ("src/estimation/stochvol_runner.R", []),
    "asvt": ("src/estimation/stochvol_runner_correct.R", ["asvt"]),
}


def demeaned_series(asset):
    """Full demeaned log-returns (by in-sample mean) and the in-sample length."""
    r = load_returns(asset)
    ins = r[r.index < OOS_START]
    full = (r - ins.mean()).values.astype(np.float64)
    return full, len(ins)


def estimate_tcn(model, in_returns):
    spec = TCN_SPEC[model]
    net = SVTCNNet(n_channels=32, kernel_size=7, n_blocks=6, dropout=0.0,
                   n_outputs=spec["n_outputs"], second_channel=spec["second"])
    net.load_state_dict(torch.load(REPO / spec["ckpt"], map_location="cpu", weights_only=True))
    net.eval()
    x = torch.from_numpy(in_returns.astype(np.float32)).unsqueeze(0)   # (1, T_in)
    with torch.no_grad():
        est = spec["cfg"].inverse_transform(net(x).numpy().astype(np.float64))[0]
    return est


def estimate_mcmc(model, in_returns, asset=None):
    script, extra = R_RUNNER[model]
    draws, burnin = DRAWS_BURNIN[model]
    if model == "asvt" and asset in ASVT_DRAWS_OVERRIDE:
        draws, burnin = ASVT_DRAWS_OVERRIDE[asset]
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        np.savetxt(f.name, in_returns); csv = f.name
    out = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False).name
    try:
        r = subprocess.run(
            ["Rscript", str(REPO / script), csv, out, *extra,
             str(draws), str(burnin), "11", "12"],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-400:])
        res = json.load(open(out))
        return np.array(res["post_mean"]), np.array(res["rhat"])
    finally:
        Path(csv).unlink(missing_ok=True); Path(out).unlink(missing_ok=True)


def predictive_ll(full, len_in, model, params):
    mu, phi, sig = float(params[0]), float(params[1]), float(params[2])
    phi = min(max(phi, -0.999), 0.999)
    if model == "sv":
        nu, rho = np.inf, 0.0
    else:
        rho = min(max(float(params[3]), -0.999), 0.999)
        nu = max(float(params[4]), 2.05)
    vals = [sv_log_likelihood(full, mu, phi, sig, nu=nu, rho=rho,
                              n_particles=N_PARTICLES, seed=s, t_split=len_in)[1]
            for s in SEEDS]
    return float(np.mean(vals))


def run(only=None):
    for asset in ALL_ASSETS:
        if only and asset not in only:
            continue
        fp = OUT / f"{asset}.json"
        if fp.exists():
            continue
        full, len_in = demeaned_series(asset)
        in_returns = full[:len_in]
        print(f"[{asset}] len_in={len_in} len_oos={len(full)-len_in} ...", flush=True)
        res = {}
        for model in ["sv", "asvt"]:
            p_tcn = estimate_tcn(model, in_returns)
            ll_tcn = predictive_ll(full, len_in, model, p_tcn)
            p_mc, rhat = estimate_mcmc(model, in_returns, asset=asset)
            ll_mc = predictive_ll(full, len_in, model, p_mc)
            res[model] = dict(
                tcn=dict(params=p_tcn.tolist(), oos_ll=ll_tcn),
                mcmc=dict(params=p_mc.tolist(), rhat=rhat.tolist(), oos_ll=ll_mc),
            )
            print(f"    {model}: TCN oos_ll={ll_tcn:.1f}  MCMC oos_ll={ll_mc:.1f}", flush=True)
        json.dump(res, open(fp, "w"), indent=2)


def summary():
    print("=" * 88)
    print("Out-of-sample predictive log-likelihood by asset (higher = better)")
    print(f"{'Asset':>18} | {'SV MCMC':>9} {'SV TCN':>9} | {'ASVt MCMC':>10} {'ASVt TCN':>9}")
    print("-" * 88)
    for group, members in ASSET_GROUPS.items():
        print(f"[{group}]")
        for a in members:
            fp = OUT / f"{a}.json"
            if not fp.exists():
                print(f"{LABELS[a]:>18} |  (pending)")
                continue
            d = json.load(open(fp))
            print(f"{LABELS[a]:>18} | {d['sv']['mcmc']['oos_ll']:>9.1f} {d['sv']['tcn']['oos_ll']:>9.1f} "
                  f"| {d['asvt']['mcmc']['oos_ll']:>10.1f} {d['asvt']['tcn']['oos_ll']:>9.1f}")
    print("-" * 88)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--only", nargs="+", default=None, help="restrict to these asset codes")
    args = ap.parse_args()
    if not args.summary:
        run(only=args.only)
    summary()
