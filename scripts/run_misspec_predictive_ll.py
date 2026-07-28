"""
Predictive log-likelihood for cells (a) and (b) — the DECISIVE metric.

For each scenario and method, take the estimated base-SV parameters and evaluate
how well the fitted (misspecified) base-SV model predicts the data, via the
particle filter, split into in-sample (first 1000) and out-of-sample (next 1000)
using t_split. The estimators assume base SV, so the filter uses nu=inf, rho=0
with the estimated [mu, phi, sigma_eta].

Matched filter seeds are used for both methods on each series, so the common
Monte Carlo noise partly cancels in the (a)-(b) difference.

Output: results/misspec/predictive_ll.npz and a printed summary. The
out-of-sample comparison is the headline.

Run:  uv run python scripts/run_misspec_predictive_ll.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.evaluation.particle_filter import sv_log_likelihood

SCENARIOS = {
    "asv":  "data/test_misspec_asv.npz",
    "svt":  "data/test_misspec_svt.npz",
    "asvt": "data/test_misspec_asvt.npz",
}
T_SPLIT = 1000
OUT = Path("results/misspec")
N_PARTICLES = 10_000
SEEDS = (0, 1, 2)   # average over matched seeds


def series_ll(returns_full, params, seeds=SEEDS):
    """Mean (in-sample, oos) base-SV predictive log-lik over matched seeds."""
    mu, phi, sig = float(params[0]), float(params[1]), float(params[2])
    ins, oos = [], []
    for s in seeds:
        a, b = sv_log_likelihood(returns_full, mu, phi, sig,
                                 n_particles=N_PARTICLES, seed=s, t_split=T_SPLIT)
        ins.append(a); oos.append(b)
    return np.mean(ins), np.mean(oos)


def main():
    summary = {}
    for scen, path in SCENARIOS.items():
        returns = np.load(path)["returns"].astype(np.float64)   # (200, 2000)
        est_tcn = np.load(OUT / f"{scen}_tcn.npz")["est"]        # (200, 3)
        est_sv  = np.load(OUT / f"{scen}_stochvol.npz")["est"]   # (200, 3)
        N = returns.shape[0]

        tcn_in = np.empty(N); tcn_oos = np.empty(N)
        sv_in  = np.empty(N); sv_oos  = np.empty(N)
        for i in range(N):
            tcn_in[i], tcn_oos[i] = series_ll(returns[i], est_tcn[i])
            sv_in[i],  sv_oos[i]  = series_ll(returns[i], est_sv[i])

        np.savez_compressed(OUT / f"{scen}_predictive_ll.npz",
                            tcn_in=tcn_in, tcn_oos=tcn_oos,
                            sv_in=sv_in, sv_oos=sv_oos)

        # Per-series differences (matched data & seeds); OOS is the headline.
        d_oos = tcn_oos - sv_oos
        summary[scen] = dict(
            tcn_oos=tcn_oos.mean(), sv_oos=sv_oos.mean(),
            tcn_in=tcn_in.mean(), sv_in=sv_in.mean(),
            mean_diff_oos=d_oos.mean(), frac_tcn_better_oos=(d_oos > 0).mean(),
        )

    print("=" * 72)
    print("PREDICTIVE LOG-LIKELIHOOD — cell (a) TCN vs cell (b) stochvol")
    print("higher is better; OOS (out-of-sample) is the headline metric")
    print("=" * 72)
    print(f"{'scenario':>8} | {'TCN OOS':>10} {'stochvol OOS':>13} "
          f"{'mean diff':>10} {'% TCN better':>13}")
    print("-" * 72)
    for scen, s in summary.items():
        print(f"{scen:>8} | {s['tcn_oos']:>10.2f} {s['sv_oos']:>13.2f} "
              f"{s['mean_diff_oos']:>+10.2f} {100*s['frac_tcn_better_oos']:>12.0f}%")
    print("-" * 72)
    print("(mean diff = per-series TCN OOS − stochvol OOS, matched data & seeds)")


if __name__ == "__main__":
    main()
