"""
Validation of the SV-t and ASV-t simulators.

Two kinds of check:

  A. Distributional match to stochvol svsim (the authoritative convention),
     at fixed parameters:
       - variance ~ exp(mu) regardless of nu (standardised t)
       - excess kurtosis of standardised residuals tracks the t_nu value
       - forward leverage correlation on the normal component ~ rho (ASV-t)

  B. Cross-check with the validated particle filter: the filter (already
     validated against svsim) run on OUR simulator's data should peak at the
     true parameters. If our simulator matches the convention the filter
     expects, the peaks land on truth. This closes the loop:
     filter == svsim  and  filter == our sim  =>  our sim == svsim.

Run:  uv run python scripts/validate_t_simulators.py
"""

from __future__ import annotations

import numpy as np

from src.simulation.simulator import simulate_sv_t, simulate_asv_t
from src.simulation.sv_params import SVtParams, ASVtParams
from src.evaluation.particle_filter import sv_log_likelihood


def check_svt_distribution():
    print("=" * 66)
    print("A1. SV-t distributional checks (fixed params, vs theory)")
    print("-" * 66)
    # mu=0 so exp(mu)=1; near-constant vol via tiny sigma, phi=0.
    for nu in (5.0, 8.0, 15.0):
        cfg = SVtParams(mu_range=(0, 0), phi_range=(0.0, 0.0),
                        sigma_eta_range=(0.001, 0.001), nu_range=(nu, nu))
        res = simulate_sv_t(N=200, T=3000, config=cfg, seed=3)
        r = res.returns.astype(np.float64).ravel()
        var = r.var()
        # standardised residual kurtosis; theoretical excess kurt of t_nu = 6/(nu-4)
        exk = (((r - r.mean()) ** 4).mean() / var ** 2) - 3.0
        theo = 6.0 / (nu - 4.0)
        print(f"  nu={nu:>4}: Var={var:.3f} (expect ~1.0)  "
              f"excess kurt={exk:.2f} (t_nu theory {theo:.2f})")
    print()


def check_asvt_leverage():
    print("=" * 66)
    print("A2. ASV-t forward leverage on the normal component (fixed params)")
    print("-" * 66)
    for rho in (-0.6, -0.3):
        cfg = ASVtParams(mu_range=(-1, -1), phi_range=(0.95, 0.95),
                         sigma_eta_range=(0.2, 0.2), rho_range=(rho, rho),
                         nu_range=(6.0, 6.0))
        res = simulate_asv_t(N=200, T=3000, config=cfg, seed=3)
        r = res.returns.astype(np.float64)
        h = res.latent_h.astype(np.float64)
        dh = h[:, 1:] - h[:, :-1]
        # full-residual forward corr (attenuated by tau, like svsim showed)
        fwd = np.corrcoef(r[:, :-1].ravel(), dh.ravel())[0, 1]
        print(f"  rho={rho:+.1f}: forward corr(r_t, h_(t+1)-h_t) = {fwd:+.3f} "
              f"(attenuated by tau; sign & rough magnitude should track rho)")
    print()


def check_filter_crosscheck():
    print("=" * 66)
    print("B. Particle-filter cross-check on OUR simulator data (peaks at truth)")
    print("-" * 66)

    # SV-t series from our simulator.
    cfg = SVtParams(mu_range=(-1, -1), phi_range=(0.95, 0.95),
                    sigma_eta_range=(0.2, 0.2), nu_range=(6.0, 6.0))
    r = simulate_sv_t(N=1, T=2000, config=cfg, seed=5).returns[0].astype(np.float64)
    nu_grid = [4, 5, 6, 8, 12]
    ll = [sv_log_likelihood(r, -1, 0.95, 0.2, nu=n, n_particles=20_000, seed=0) for n in nu_grid]
    print(f"  SV-t data: nu peak at {nu_grid[int(np.argmax(ll))]} (true 6)")

    # ASV-t series from our simulator.
    cfg2 = ASVtParams(mu_range=(-1, -1), phi_range=(0.95, 0.95),
                      sigma_eta_range=(0.2, 0.2), rho_range=(-0.6, -0.6),
                      nu_range=(6.0, 6.0))
    r2 = simulate_asv_t(N=1, T=3000, config=cfg2, seed=5).returns[0].astype(np.float64)
    rho_grid = [-0.8, -0.7, -0.6, -0.5, -0.3, 0.0]
    ll = [np.mean([sv_log_likelihood(r2, -1, 0.95, 0.2, nu=6.0, rho=rr,
                                     n_particles=20_000, seed=s) for s in range(3)])
          for rr in rho_grid]
    print(f"  ASV-t data: rho peak at {rho_grid[int(np.argmax(ll))]:+.1f} (true -0.6)")
    ll_true = np.mean([sv_log_likelihood(r2, -1, 0.95, 0.2, nu=6.0, rho=-0.6,
                                         n_particles=20_000, seed=s) for s in range(5)])
    ll_zero = np.mean([sv_log_likelihood(r2, -1, 0.95, 0.2, nu=6.0, rho=0.0,
                                         n_particles=20_000, seed=s) for s in range(5)])
    print(f"  ASV-t data: leverage improves logL by {ll_true - ll_zero:.2f}")
    print()


if __name__ == "__main__":
    check_svt_distribution()
    check_asvt_leverage()
    check_filter_crosscheck()
