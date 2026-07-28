"""
Validation of the ASV-t particle filter (leverage + standardised-t errors).

Data is generated with stochvol's svsim using BOTH nu and rho. The leverage
couples to the normal component of the scale mixture (verified against svsim),
which is what the filter's tau-posterior propagation reproduces. If the coupling
and densities are correct, the filter's log-likelihood should peak near the true
nu and the true rho, and accounting for leverage should improve the fit.

Checks:
  1. nu profile peaks near true nu (rho at truth).
  2. rho profile peaks near true rho (nu at truth).
  3. Leverage improvement: logL(true rho) > logL(rho=0) on ASV-t data.
  4. phi and sigma_eta profiles peak near truth.

Run:  uv run python scripts/validate_particle_filter_asvt.py
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
sim <- svsim({T}, mu = {mu}, phi = {phi}, sigma = {sigma}, nu = {nu}, rho = {rho})
write.table(sim$y, file = "{out}", row.names = FALSE, col.names = FALSE)
"""


def simulate_asvt(T, mu, phi, sigma, nu, rho, seed):
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        out = f.name
    script = R_TEMPLATE.format(seed=seed, T=T, mu=mu, phi=phi, sigma=sigma,
                               nu=nu, rho=rho, out=out)
    subprocess.run(["Rscript", "-e", script], check=True, capture_output=True, text=True)
    y = np.loadtxt(out)
    Path(out).unlink()
    return y


def avg_ll(r, mu, phi, sigma, nu, rho, seeds=3):
    return np.mean([sv_log_likelihood(r, mu, phi, sigma, nu=nu, rho=rho,
                                      n_particles=20_000, seed=s) for s in range(seeds)])


def main():
    mu, phi, sigma, nu, rho = -1.0, 0.95, 0.20, 6.0, -0.6
    T = 3000
    print("=" * 66)
    print("ASV-t FILTER VALIDATION (svsim data: leverage + standardised t)")
    print(f"  true: mu={mu}, phi={phi}, sigma_eta={sigma}, nu={nu}, rho={rho}, T={T}")
    print("-" * 66)

    r = simulate_asvt(T, mu, phi, sigma, nu, rho, seed=40)

    # Check 1: nu profile at true rho.
    nu_grid = [4, 5, 6, 8, 12, 20]
    ll = [avg_ll(r, mu, phi, sigma, n, rho) for n in nu_grid]
    print(f"  nu profile (rho={rho}): peak at {nu_grid[int(np.argmax(ll))]} (true {nu})")
    for n, l in zip(nu_grid, ll):
        print(f"    nu={n:>3}: {l:10.3f}")

    # Check 2: rho profile at true nu.
    rho_grid = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.3, 0.0]
    ll = [avg_ll(r, mu, phi, sigma, nu, rr) for rr in rho_grid]
    print(f"\n  rho profile (nu={nu}): peak at {rho_grid[int(np.argmax(ll))]:+.1f} (true {rho})")
    for rr, l in zip(rho_grid, ll):
        print(f"    rho={rr:+.1f}: {l:10.3f}")

    # Check 3: leverage improvement.
    ll_true = avg_ll(r, mu, phi, sigma, nu, rho, seeds=5)
    ll_zero = avg_ll(r, mu, phi, sigma, nu, 0.0, seeds=5)
    print(f"\n  logL(true rho={rho}) = {ll_true:.3f}")
    print(f"  logL(rho=0)          = {ll_zero:.3f}")
    print(f"  => leverage improves logL by {ll_true - ll_zero:.3f}")

    # Check 4: phi and sigma profiles.
    phi_grid = np.linspace(0.88, 0.99, 8)
    ll = [sv_log_likelihood(r, mu, p, sigma, nu=nu, rho=rho, n_particles=20_000, seed=0)
          for p in phi_grid]
    print(f"\n  phi peak at {phi_grid[int(np.argmax(ll))]:.3f} (true {phi})")
    sig_grid = np.linspace(0.05, 0.40, 8)
    ll = [sv_log_likelihood(r, mu, phi, s, nu=nu, rho=rho, n_particles=20_000, seed=0)
          for s in sig_grid]
    print(f"  sigma_eta peak at {sig_grid[int(np.argmax(ll))]:.3f} (true {sigma})")


if __name__ == "__main__":
    main()
