"""
Run MCMC benchmark on all three test sets (T=500, T=1000, T=2000).

Configs (from CLAUDE.md pilot results):
    T=500, T=1000: 1000 draws / 1000 tune / target_accept=0.9
    T=2000:        1000 draws / 2000 tune / target_accept=0.9

Results saved to results/mcmc_T<T>/:
    checkpoints/series_XXXX.npz  — per-series checkpoint (crash-safe)
    results.npz                  — assembled batch results
    summary.json                 — RMSE/MAE/bias per parameter

Usage:
    uv run python scripts/run_mcmc_benchmark.py          # all three T values
    uv run python scripts/run_mcmc_benchmark.py --T 500  # single T value
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.estimation.mcmc_config import MCMCConfig
from src.estimation.mcmc_runner import run_mcmc_batch
from src.simulation.simulator import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PARAM_NAMES = ["mu", "phi", "sigma_eta"]

# T=2000 needs more tuning steps (confirmed by pilot — see CLAUDE.md)
CONFIGS = {
    500:  MCMCConfig(draws=1000, tune=1000, target_accept=0.9, n_jobs=4),
    1000: MCMCConfig(draws=1000, tune=1000, target_accept=0.9, n_jobs=4),
    2000: MCMCConfig(draws=1000, tune=2000, target_accept=0.9, n_jobs=4),
}


def save_summary(results_path: Path, true: np.ndarray, T: int) -> None:
    data = np.load(results_path)
    means = data["means"]        # (N, 3)
    rhats = data["rhats"]        # (N, 3)

    errors  = means - true
    summary = {"method": "mcmc", "T": T, "N": len(true), "params": {}}
    for i, name in enumerate(PARAM_NAMES):
        summary["params"][name] = {
            "rmse":     float(np.sqrt(np.mean(errors[:, i] ** 2))),
            "mae":      float(np.mean(np.abs(errors[:, i]))),
            "bias":     float(np.mean(errors[:, i])),
            "rel_rmse": float(np.sqrt(np.mean(errors[:, i] ** 2)) / true[:, i].std()),
        }

    summary["rhat"] = {
        name: {
            "mean":       float(rhats[:, i].mean()),
            "max":        float(rhats[:, i].max()),
            "frac_gt1p1": float((rhats[:, i] > 1.1).mean()),
        }
        for i, name in enumerate(PARAM_NAMES)
    }

    out_dir = results_path.parent
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print table
    print(f"\nMCMC T={T} — test set (N={len(true)})")
    print(f"\n  {'param':<12} {'RMSE':>8} {'MAE':>8} {'Bias':>8} {'Rel_RMSE':>10}")
    print("  " + "-" * 50)
    for name in PARAM_NAMES:
        s = summary["params"][name]
        print(f"  {name:<12} {s['rmse']:>8.4f} {s['mae']:>8.4f} {s['bias']:>8.4f} {s['rel_rmse']:>10.3f}")

    print(f"\n  R-hat summary:")
    for name in PARAM_NAMES:
        r = summary["rhat"][name]
        print(f"  {name:<12} mean={r['mean']:.3f}  max={r['max']:.3f}  frac>1.1={r['frac_gt1p1']:.1%}")

    print(f"\n  Saved to {out_dir}/")


def run_for_T(T: int, repo: Path) -> None:
    config     = CONFIGS[T]
    test_path  = repo / f"data/test_T{T}.npz"
    out_dir    = repo / f"results/mcmc_T{T}"
    ckpt_dir   = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 55)
    logger.info("MCMC T=%d | draws=%d tune=%d n_jobs=%d",
                T, config.draws, config.tune, config.n_jobs)
    logger.info("=" * 55)

    dataset = load_dataset(test_path)
    t0 = time.time()
    run_mcmc_batch(dataset, config, out_path=ckpt_dir)
    elapsed = time.time() - t0
    logger.info("T=%d done in %.1f min", T, elapsed / 60)

    # Move results.npz up to out_dir and save summary
    assembled = ckpt_dir / "results.npz"
    results_dest = out_dir / "results.npz"
    if assembled.exists():
        import shutil
        shutil.copy(assembled, results_dest)

    save_summary(results_dest, dataset.params, T)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, choices=[500, 1000, 2000],
                        help="Run a single T value. Omit to run all three.")
    args = parser.parse_args()

    repo = Path(__file__).parent.parent
    T_values = [args.T] if args.T else [500, 1000, 2000]

    for T in T_values:
        run_for_T(T, repo)


if __name__ == "__main__":
    main()
