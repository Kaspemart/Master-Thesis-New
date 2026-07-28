"""
Correct-model predictive log-likelihood for cells (c) and (d), completing the 2x2.

Each correctly-specified estimate is evaluated under ITS model (with estimated
rho and/or nu) via the particle filter, split in/out-of-sample at t=1000. On the
same OOS window as cells (a)/(b), so all four cells' OOS predictive LL are
directly comparable per scenario.

Prints the full 2x2 OOS predictive-LL table (a,b,c,d) and the key contrasts:
  (a) vs (c): NN cost of misspecification
  (b) vs (d): MCMC cost of misspecification
  (c) vs (d): correctly-specified NN vs MCMC

Run:  uv run python scripts/run_misspec_predictive_ll_correct.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.evaluation.particle_filter import sv_log_likelihood

OUT = Path("results/misspec")
T_SPLIT = 1000
N_PARTICLES = 10_000
SEEDS = (0, 1, 2)

# scenario -> column layout of the correct-model estimates
LAYOUT = {
    "asv":  dict(has_rho=True,  has_nu=False),
    "svt":  dict(has_rho=False, has_nu=True),
    "asvt": dict(has_rho=True,  has_nu=True),
}


def oos_ll(returns_full, est_row, has_rho, has_nu):
    """Correct-model OOS predictive LL (mean over matched seeds)."""
    mu, phi, sig = float(est_row[0]), float(est_row[1]), float(est_row[2])
    idx = 3
    rho = 0.0
    nu = np.inf
    if has_rho:
        rho = float(est_row[idx]); idx += 1
    if has_nu:
        nu = float(est_row[idx]); idx += 1
    # guard against out-of-support estimates
    phi = min(max(phi, -0.999), 0.999)
    rho = min(max(rho, -0.999), 0.999)
    if np.isfinite(nu):
        nu = max(nu, 2.05)
    ins, oos = [], []
    for s in SEEDS:
        a, b = sv_log_likelihood(returns_full, mu, phi, sig, nu=nu, rho=rho,
                                 n_particles=N_PARTICLES, seed=s, t_split=T_SPLIT)
        ins.append(a); oos.append(b)
    return float(np.mean(ins)), float(np.mean(oos))


def compute_cell(scen, path, est_file):
    returns = np.load(path)["returns"].astype(np.float64)
    est = np.load(OUT / est_file)["est"]
    lay = LAYOUT[scen]
    res = [oos_ll(returns[i], est[i], lay["has_rho"], lay["has_nu"])
           for i in range(returns.shape[0])]
    return np.array([r[0] for r in res]), np.array([r[1] for r in res])


def main():
    scen_paths = {
        "asv":  "data/test_misspec_asv.npz",
        "svt":  "data/test_misspec_svt.npz",
        "asvt": "data/test_misspec_asvt.npz",
    }
    rows = {}
    for scen, path in scen_paths.items():
        c_in, c = compute_cell(scen, path, f"{scen}_tcn_correct.npz")
        d_in, d = compute_cell(scen, path, f"{scen}_stochvol_correct.npz")
        np.savez_compressed(OUT / f"{scen}_predictive_ll_correct.npz",
                            c_oos=c, d_oos=d, c_in=c_in, d_in=d_in)
        # misspecified (a,b) OOS from earlier
        ab = np.load(OUT / f"{scen}_predictive_ll.npz")
        a, b = ab["tcn_oos"], ab["sv_oos"]
        rows[scen] = dict(a=a.mean(), b=b.mean(), c=c.mean(), d=d.mean(),
                          ac=(c - a).mean(), bd=(d - b).mean(), cd=(c - d).mean())

    print("=" * 74)
    print("OUT-OF-SAMPLE PREDICTIVE LOG-LIKELIHOOD — full 2x2 (higher = better)")
    print("  a=misspec NN  b=misspec MCMC  c=correct NN  d=correct MCMC")
    print("=" * 74)
    print(f"{'scen':>6} | {'(a) NN':>9} {'(b) MCMC':>9} {'(c) NN*':>9} {'(d) MCMC*':>9}"
          f" | {'c−a':>7} {'d−b':>7} {'c−d':>7}")
    print("-" * 74)
    for scen, r in rows.items():
        print(f"{scen:>6} | {r['a']:>9.2f} {r['b']:>9.2f} {r['c']:>9.2f} {r['d']:>9.2f}"
              f" | {r['ac']:>+7.2f} {r['bd']:>+7.2f} {r['cd']:>+7.2f}")
    print("-" * 74)
    print("c−a = NN gain from correct model; d−b = MCMC gain; c−d = NN vs MCMC (both correct)")


if __name__ == "__main__":
    main()
