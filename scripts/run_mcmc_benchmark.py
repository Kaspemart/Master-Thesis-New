"""
Run the MCMC benchmark on all three test sets (T=500, T=1000, T=2000).

Supports two backends:
    stochvol (default) — stochvol R package, ASIS sampler (Kastner & Frühwirth-Schnatter 2014)
    nuts               — PyMC NUTS sampler (general-purpose HMC, kept for comparison)

Results saved to results/<backend>_T<T>/:
    checkpoints/series_XXXX.npz  — per-series checkpoint (crash-safe)
    results.npz                  — assembled batch results
    summary.json                 — RMSE/MAE/bias/R-hat per parameter

Usage:
    uv run python scripts/run_mcmc_benchmark.py                      # stochvol, all T
    uv run python scripts/run_mcmc_benchmark.py --T 1000             # stochvol, T=1000
    uv run python scripts/run_mcmc_benchmark.py --backend nuts       # NUTS, all T
    uv run python scripts/run_mcmc_benchmark.py --backend nuts --T 500

NUTS configs (from CLAUDE.md pilot results):
    T=500, T=1000: 1000 draws / 1000 tune / target_accept=0.9
    T=2000:        1000 draws / 2000 tune / target_accept=0.9

stochvol configs:
    All T: 1000 draws / 1000 burnin  (stochvol mixes faster than NUTS on SV models)
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.estimation.mcmc_config    import MCMCConfig
from src.estimation.mcmc_runner    import run_mcmc_batch
from src.estimation.stochvol_runner import StochvolConfig, run_stochvol_batch
from src.simulation.simulator       import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PARAM_NAMES = ["mu", "phi", "sigma_eta"]

# ── Per-T configs ─────────────────────────────────────────────────────────────

NUTS_CONFIGS = {
    500:  MCMCConfig(draws=1000, tune=1000, target_accept=0.9, n_jobs=4),
    1000: MCMCConfig(draws=1000, tune=1000, target_accept=0.9, n_jobs=4),
    2000: MCMCConfig(draws=1000, tune=2000, target_accept=0.9, n_jobs=4),
}

STOCHVOL_CONFIGS = {
    500:  StochvolConfig(draws=1000, burnin=1000, n_jobs=4),
    1000: StochvolConfig(draws=1000, burnin=1000, n_jobs=4),
    2000: StochvolConfig(draws=1000, burnin=1000, n_jobs=4),
}


# ── Summary ───────────────────────────────────────────────────────────────────

def save_summary(results_path: Path, true: np.ndarray, T: int, backend: str) -> None:
    data   = np.load(results_path)
    means  = data["means"]   # (N, 3)
    rhats  = data["rhats"]   # (N, 3)

    errors  = means - true
    summary = {"method": backend, "T": T, "N": len(true), "params": {}}
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

    print(f"\n{backend.upper()} T={T} — test set (N={len(true)})")
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


# ── Per-T runner ──────────────────────────────────────────────────────────────

def run_for_T(T: int, backend: str, repo: Path) -> None:
    test_path = repo / f"data/test_T{T}.npz"
    out_dir   = repo / f"results/{backend}_T{T}"
    ckpt_dir  = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(test_path)

    if backend == "stochvol":
        config = STOCHVOL_CONFIGS[T]
        logger.info("=" * 55)
        logger.info("stochvol T=%d | draws=%d burnin=%d n_jobs=%d",
                    T, config.draws, config.burnin, config.n_jobs)
        logger.info("=" * 55)
        t0 = time.time()
        run_stochvol_batch(dataset, config, out_path=ckpt_dir)

    elif backend == "nuts":
        config = NUTS_CONFIGS[T]
        logger.info("=" * 55)
        logger.info("NUTS T=%d | draws=%d tune=%d n_jobs=%d",
                    T, config.draws, config.tune, config.n_jobs)
        logger.info("=" * 55)
        t0 = time.time()
        run_mcmc_batch(dataset, config, out_path=ckpt_dir)

    elapsed = time.time() - t0
    logger.info("T=%d done in %.1f min", T, elapsed / 60)

    assembled = ckpt_dir / "results.npz"
    results_dest = out_dir / "results.npz"
    if assembled.exists():
        shutil.copy(assembled, results_dest)

    save_summary(results_dest, dataset.params, T, backend)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MCMC benchmark (stochvol or NUTS) on SV test sets."
    )
    parser.add_argument(
        "--backend", choices=["stochvol", "nuts"], default="stochvol",
        help="Sampler backend (default: stochvol).",
    )
    parser.add_argument(
        "--T", type=int, choices=[500, 1000, 2000],
        help="Run a single T value. Omit to run all three.",
    )
    args = parser.parse_args()

    repo     = Path(__file__).parent.parent
    T_values = [args.T] if args.T else [500, 1000, 2000]

    for T in T_values:
        run_for_T(T, args.backend, repo)


if __name__ == "__main__":
    main()
