"""
T=2000 MCMC pilot run.

Runs MCMC on 5 series from the test_T2000 dataset to verify that the
1000 draws / 1000 tune settings produce acceptable convergence before
committing to the full 200-series batch.

Reports: wall time per series, R-hat per parameter, posterior mean vs true.

Usage:
    uv run python scripts/mcmc_pilot_T2000.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.estimation.mcmc_config import MCMCConfig
from src.estimation.mcmc_runner import run_mcmc_single
from src.simulation.simulator import load_dataset

PILOT_N = 5
PILOT_INDICES = [0, 40, 80, 120, 160]   # spread across the 200-series test set
T = 2000
DATA_PATH = Path("data/test_T2000.npz")
CONFIG = MCMCConfig(draws=1000, tune=1000, chains=2, n_jobs=1)


def main() -> None:
    print(f"T=2000 MCMC pilot — {PILOT_N} series, {CONFIG.draws} draws / {CONFIG.tune} tune / {CONFIG.chains} chains")
    print(f"Data: {DATA_PATH}")
    print()

    dataset = load_dataset(DATA_PATH)

    header = f"{'idx':>4}  {'true_mu':>8} {'true_phi':>9} {'true_sig':>9}  {'mean_mu':>8} {'mean_phi':>9} {'mean_sig':>9}  {'rhat_mu':>8} {'rhat_phi':>9} {'rhat_sig':>9}  {'time_s':>7}"
    print(header)
    print("-" * len(header))

    times = []
    rhats = []

    for idx in PILOT_INDICES:
        returns = dataset.returns[idx]
        true = dataset.params[idx]

        t0 = time.time()
        result = run_mcmc_single(returns, CONFIG, seed=42 + idx)
        elapsed = time.time() - t0
        times.append(elapsed)
        rhats.append(result.rhat)

        print(
            f"{idx:>4}  "
            f"{true[0]:>8.3f} {true[1]:>9.3f} {true[2]:>9.3f}  "
            f"{result.mean[0]:>8.3f} {result.mean[1]:>9.3f} {result.mean[2]:>9.3f}  "
            f"{result.rhat[0]:>8.3f} {result.rhat[1]:>9.3f} {result.rhat[2]:>9.3f}  "
            f"{elapsed:>7.1f}s"
        )

    rhats = np.array(rhats)
    print()
    print(f"Wall time:  mean={np.mean(times):.1f}s  max={np.max(times):.1f}s  total={np.sum(times):.1f}s")
    print(f"R-hat max:  mu={rhats[:,0].max():.3f}  phi={rhats[:,1].max():.3f}  sigma_eta={rhats[:,2].max():.3f}")
    print(f"R-hat > 1.1: {(rhats > 1.1).sum()} / {rhats.size} parameter-series pairs")
    print()

    max_rhat = rhats.max()
    est_full_time_min = np.mean(times) * 200 / 60
    print(f"Estimated full T=2000 batch time (200 series, sequential): {est_full_time_min:.0f} min")
    print(f"Estimated full T=2000 batch time (4 cores):                {est_full_time_min/4:.0f} min")
    print()

    if max_rhat < 1.1:
        print("VERDICT: Settings look good. Proceed with full T=2000 batch.")
    elif max_rhat < 1.2:
        print("VERDICT: R-hat borderline (< 1.2). Acceptable but monitor carefully in full run.")
    else:
        print(f"VERDICT: R-hat too high ({max_rhat:.3f}). Consider increasing tune steps before full run.")


if __name__ == "__main__":
    main()
