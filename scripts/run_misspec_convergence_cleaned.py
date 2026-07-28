"""
Convergence-cleaned comparison for Chapter 6 (supervisor request Q4).

Re-computes the (c) correct NN vs (d) correct MCMC predictive-LL comparison, and
cell (d) parameter RMSE, on the subset of series where the correctly-specified
MCMC converged on EVERY parameter (R-hat <= 1.1). The NN has no convergence
notion and always returns an estimate, so cleaning removes only MCMC failures.

Uses already-saved per-series predictive LL and R-hat — no re-sampling.

Output: experiments/chapter6_convergence_cleaned.json + printed tables.
Run:  uv run python scripts/run_misspec_convergence_cleaned.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R = Path("results/misspec")
NAMES = {"asv": ["mu", "phi", "sigma_eta", "rho"],
         "svt": ["mu", "phi", "sigma_eta", "nu"],
         "asvt": ["mu", "phi", "sigma_eta", "rho", "nu"]}


def tstat(x):
    x = np.asarray(x); se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() / se)


def main():
    out = {"_note": "Convergence-cleaned Ch6 results. Non-converged = any "
                    "parameter R-hat>1.1 in cell (d). NN always converges."}
    for s in NAMES:
        ll = np.load(R / f"{s}_predictive_ll_correct.npz")
        d_est = np.load(R / f"{s}_stochvol_correct.npz")
        c, d = ll["c_oos"], ll["d_oos"]
        rhat, est, true = d_est["rhat"], d_est["est"], d_est["true"]
        conv = (rhat <= 1.1).all(axis=1)

        m_full, t_full = tstat(c - d)
        m_cl, t_cl = tstat((c - d)[conv])
        rmse_full = np.sqrt(((est - true) ** 2).mean(axis=0))
        rmse_cl = np.sqrt(((est[conv] - true[conv]) ** 2).mean(axis=0))

        out[s] = dict(
            n_kept=int(conv.sum()), n_total=int(conv.size),
            cd_oos_full=dict(mean=m_full, t=t_full),
            cd_oos_cleaned=dict(mean=m_cl, t=t_cl),
            d_rmse_full={n: float(rmse_full[i]) for i, n in enumerate(NAMES[s])},
            d_rmse_cleaned={n: float(rmse_cl[i]) for i, n in enumerate(NAMES[s])},
        )

    Path("experiments").mkdir(exist_ok=True)
    with open("experiments/chapter6_convergence_cleaned.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"{'scen':>6} | {'kept':>7} | {'c-d full':>16} {'c-d cleaned':>18}")
    print("-" * 56)
    for s in NAMES:
        o = out[s]
        print(f"{s:>6} | {o['n_kept']:>3}/200 | "
              f"{o['cd_oos_full']['mean']:>+8.2f} (t{o['cd_oos_full']['t']:>5.1f}) "
              f"{o['cd_oos_cleaned']['mean']:>+9.2f} (t{o['cd_oos_cleaned']['t']:>5.1f})")
    print("\nWrote experiments/chapter6_convergence_cleaned.json")


if __name__ == "__main__":
    main()
