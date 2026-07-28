"""
Validation of the ASV (forward-leverage, Gaussian) particle filter.

Data is generated with stochvol's svsim using its leverage option (verified to
use the forward convention, corr(eps_t, eta_{t+1}) = rho). If our filter's
leverage propagation matches, its log-likelihood should peak near the true rho.

Checks:
  1. rho profile peaks near the true rho on stochvol-generated ASV data.
  2. phi and sigma_eta profiles peak near truth on the same data.
  3. Ignoring leverage (rho = 0) gives a lower log-likelihood than the true rho,
     i.e. accounting for leverage genuinely improves the fit.
  4. Sign check: the profile prefers the correct sign of rho.

Run:  uv run python scripts/validate_particle_filter_leverage.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from src.evaluation.particle_filter import sv_log_likelihood

R_TEMPLATE = """
suppressPackageStartupMessages(library(stochvol))
set.seed({seed})
sim <- svsim({T}, mu = {mu}, phi = {phi}, sigma = {sigma}, rho = {rho})
write.table(sim$y, file = "{out}", row.names = FALSE, col.names = FALSE)
"""


def simulate_asv(T, mu, phi, sigma, rho, seed):
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        out = f.name
    script = R_TEMPLATE.format(seed=seed, T=T, mu=mu, phi=phi, sigma=sigma, rho=rho, out=out)
    subprocess.run(["Rscript", "-e", script], check=True, capture_output=True, text=True)
    y = np.loadtxt(out)
    Path(out).unlink()
    return y


def main():
    mu, phi, sigma, rho = -1.0, 0.95, 0.20, -0.6
    T = 2000
    print("=" * 66)
    print("ASV FILTER VALIDATION (data from stochvol svsim, forward leverage)")
    print(f"  true: mu={mu}, phi={phi}, sigma_eta={sigma}, rho={rho}, T={T}")
    print("-" * 66)

    r = simulate_asv(T, mu, phi, sigma, rho, seed=30)

    # Check 1 + 4: rho profile.
    rho_grid = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.3, 0.0, 0.3]
    ll = [np.mean([sv_log_likelihood(r, mu, phi, sigma, rho=rr,
                                     n_particles=20_000, seed=s) for s in range(3)])
          for rr in rho_grid]
    rho_hat = rho_grid[int(np.argmax(ll))]
    print("  rho profile (mu, phi, sigma at truth):")
    for rr, l in zip(rho_grid, ll):
        mark = "  <-- peak" if rr == rho_hat else ""
        print(f"    rho={rr:+.1f}: logL={l:10.3f}{mark}")
    print(f"  => rho peak at {rho_hat:+.1f} (true {rho})")
    print()

    # Check 3: leverage vs no-leverage on the same data.
    ll_true = np.mean([sv_log_likelihood(r, mu, phi, sigma, rho=rho,
                                         n_particles=20_000, seed=s) for s in range(5)])
    ll_zero = np.mean([sv_log_likelihood(r, mu, phi, sigma, rho=0.0,
                                         n_particles=20_000, seed=s) for s in range(5)])
    print(f"  logL(true rho={rho}) = {ll_true:.3f}")
    print(f"  logL(rho=0, ignore leverage) = {ll_zero:.3f}")
    print(f"  => accounting for leverage improves logL by {ll_true - ll_zero:.3f}")
    print()

    # Check 2: phi and sigma profiles at true rho.
    phi_grid = np.linspace(0.88, 0.99, 8)
    ll = [sv_log_likelihood(r, mu, p, sigma, rho=rho, n_particles=20_000, seed=0)
          for p in phi_grid]
    print(f"  phi peak at {phi_grid[int(np.argmax(ll))]:.3f} (true {phi})")

    sig_grid = np.linspace(0.05, 0.40, 8)
    ll = [sv_log_likelihood(r, mu, phi, s, rho=rho, n_particles=20_000, seed=0)
          for s in sig_grid]
    print(f"  sigma_eta peak at {sig_grid[int(np.argmax(ll))]:.3f} (true {sigma})")


if __name__ == "__main__":
    main()
