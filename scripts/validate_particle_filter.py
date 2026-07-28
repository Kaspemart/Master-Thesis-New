"""
Validation of the base SV bootstrap particle filter.

Two checks, both required before any Chapter 6 result depends on the filter:

  1. Convergence in particle count. The log-likelihood estimate should
     stabilise, and its variability across seeds should shrink, as the number
     of particles grows.

  2. Likelihood surface peaks near the truth. On a single series simulated with
     known parameters, the filter's log-likelihood evaluated on a grid around
     the true parameter should peak at (or very near) the true value. This is
     the check that the filter is actually measuring the right quantity.

Run:  uv run python scripts/validate_particle_filter.py
"""

from __future__ import annotations

import numpy as np

from src.evaluation.particle_filter import sv_log_likelihood

TEST_DATA = "data/test_T1000.npz"


def check_convergence() -> None:
    """Log-likelihood estimate should stabilise as n_particles grows."""
    data = np.load(TEST_DATA)
    returns = data["returns"][0]              # first test series
    mu, phi, sigma = data["params"][0]

    print("=" * 66)
    print("CHECK 1 — convergence in particle count (series 0, true params)")
    print(f"  true: mu={mu:.3f}, phi={phi:.3f}, sigma_eta={sigma:.3f}")
    print("-" * 66)
    print(f"  {'n_particles':>12} {'mean logL':>12} {'SD over seeds':>14}")
    for n in (500, 1_000, 5_000, 10_000, 20_000):
        vals = [sv_log_likelihood(returns, mu, phi, sigma, n_particles=n, seed=s)
                for s in range(8)]
        print(f"  {n:>12} {np.mean(vals):>12.3f} {np.std(vals):>14.4f}")
    print()


def check_peak_near_truth() -> None:
    """Profile log-likelihood over each parameter; peak should sit near truth."""
    data = np.load(TEST_DATA)
    returns = data["returns"][0]
    mu_t, phi_t, sigma_t = data["params"][0]

    print("=" * 66)
    print("CHECK 2 — profile likelihood peaks near the true value")
    print(f"  true: mu={mu_t:.3f}, phi={phi_t:.3f}, sigma_eta={sigma_t:.3f}")
    print("-" * 66)

    # Profile phi, holding mu and sigma at the truth.
    phi_grid = np.linspace(max(0.50, phi_t - 0.25), min(0.99, phi_t + 0.25), 11)
    ll = [sv_log_likelihood(returns, mu_t, p, sigma_t, n_particles=20_000, seed=0)
          for p in phi_grid]
    phi_hat = phi_grid[int(np.argmax(ll))]
    print(f"  phi:       peak at {phi_hat:.3f}  (true {phi_t:.3f})")

    # Profile mu.
    mu_grid = np.linspace(mu_t - 1.5, mu_t + 1.5, 13)
    ll = [sv_log_likelihood(returns, m, phi_t, sigma_t, n_particles=20_000, seed=0)
          for m in mu_grid]
    mu_hat = mu_grid[int(np.argmax(ll))]
    print(f"  mu:        peak at {mu_hat:.3f}  (true {mu_t:.3f})")

    # Profile sigma_eta.
    sig_grid = np.linspace(max(0.05, sigma_t - 0.3), sigma_t + 0.3, 13)
    ll = [sv_log_likelihood(returns, mu_t, phi_t, s, n_particles=20_000, seed=0)
          for s in sig_grid]
    sig_hat = sig_grid[int(np.argmax(ll))]
    print(f"  sigma_eta: peak at {sig_hat:.3f}  (true {sigma_t:.3f})")
    print()


if __name__ == "__main__":
    check_convergence()
    check_peak_near_truth()
