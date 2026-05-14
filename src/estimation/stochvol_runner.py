"""
Stochvol benchmark runner for the base SV model.

Wraps the stochvol R package via subprocess, implementing the ASIS
(ancillarity-sufficiency interweaving) sampler from Kastner &
Frühwirth-Schnatter (2014).  Output format is identical to mcmc_runner.py
so downstream scripts are backend-agnostic.

Priors are weakly informative and broadly consistent with the simulation
parameter ranges — see stochvol_runner.R for full documentation.

Requirements: R >= 4.0 and the stochvol + jsonlite R packages installed.
Install via:
    brew install r
    Rscript -e "install.packages(c('stochvol','jsonlite'), repos='https://cloud.r-project.org')"
"""

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from .mcmc_runner import MCMCResult   # reuse the same result dataclass

logger = logging.getLogger(__name__)

_R_SCRIPT = Path(__file__).parent / "stochvol_runner.R"


@dataclass
class StochvolConfig:
    """
    Configuration for the stochvol benchmark sampler.

    Attributes:
        draws:       Posterior draws per chain. Two chains are run; R-hat is
                     computed across both, posterior summaries use chain 1.
        burnin:      Burn-in (warm-up) iterations per chain.
        n_jobs:      Parallel worker processes for batch runs.
        random_seed: Base seed. Per-series seeds are (random_seed + i) for
                     chain 1 and (random_seed + 10_000 + i) for chain 2.
    """
    draws:       int = 1000
    burnin:      int = 1000
    n_jobs:      int = 4
    random_seed: int = 42


def run_stochvol_single(
    returns: np.ndarray,
    config:  StochvolConfig,
    seed:    int,
) -> MCMCResult:
    """
    Fit the base SV model to one return series using stochvol (ASIS sampler).

    Runs two chains for Gelman-Rubin R-hat computation. Posterior summaries
    (mean, std, samples) are taken from chain 1.

    Args:
        returns: Observed log-returns, shape (T,).
        config:  StochvolConfig hyperparameters.
        seed:    Base random seed for this series (chain 2 uses seed + 10_000).

    Returns:
        MCMCResult with posterior mean, std, samples (draws, 3), and rhat.
        Column order throughout: [mu, phi, sigma_eta].

    Raises:
        RuntimeError: If the R subprocess fails or times out.
    """
    seed1 = seed
    seed2 = seed + 10_000

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path    = Path(tmp)
        input_csv   = tmp_path / "returns.csv"
        output_json = tmp_path / "result.json"

        # Write returns: one value per line, no header
        np.savetxt(input_csv, returns, delimiter=",", fmt="%.10f")

        cmd = [
            "Rscript", str(_R_SCRIPT),
            str(input_csv),
            str(output_json),
            str(config.draws),
            str(config.burnin),
            str(seed1),
            str(seed2),
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,   # 10-minute hard cap per series
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"stochvol timed out after 600s (seed={seed})"
            )

        if proc.returncode != 0:
            raise RuntimeError(
                f"stochvol R script failed (seed={seed}, returncode={proc.returncode}):\n"
                f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
            )

        with open(output_json) as f:
            result = json.load(f)

    post_mean = np.array(result["post_mean"], dtype=np.float32)  # (3,)
    post_sd   = np.array(result["post_sd"],   dtype=np.float32)  # (3,)
    rhat      = np.array(result["rhat"],      dtype=np.float32)  # (3,)
    samples   = np.array(result["samples"],   dtype=np.float32)  # (draws, 3)

    # Sanity checks
    assert post_mean.shape == (3,), f"Unexpected post_mean shape: {post_mean.shape}"
    assert samples.shape == (config.draws, 3), f"Unexpected samples shape: {samples.shape}"
    if not np.all(np.isfinite(samples)):
        logger.warning("Non-finite values in stochvol samples (seed=%d).", seed)

    if np.any(rhat > 1.1):
        logger.warning(
            "R-hat > 1.1 (seed=%d): mu=%.3f  phi=%.3f  sigma_eta=%.3f",
            seed, rhat[0], rhat[1], rhat[2],
        )

    return MCMCResult(mean=post_mean, std=post_sd, samples=samples, rhat=rhat)


# ---------------------------------------------------------------------------
# Checkpointing helpers — mirror mcmc_runner.py exactly
# ---------------------------------------------------------------------------

def _checkpoint_path(out_path: Path, idx: int) -> Path:
    return out_path / f"series_{idx:04d}.npz"


def _save_checkpoint(result: MCMCResult, idx: int, out_path: Path) -> None:
    p   = _checkpoint_path(out_path, idx)
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        mean=result.mean, std=result.std,
        samples=result.samples, rhat=result.rhat,
    )
    tmp.rename(p)   # atomic rename — avoids corrupted files on crash


def _load_checkpoint(idx: int, out_path: Path) -> MCMCResult:
    data = np.load(_checkpoint_path(out_path, idx))
    return MCMCResult(
        mean=data["mean"], std=data["std"],
        samples=data["samples"], rhat=data["rhat"],
    )


def _get_pending_indices(N: int, out_path: Path) -> list[int]:
    return [i for i in range(N) if not _checkpoint_path(out_path, i).exists()]


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_stochvol_batch(
    dataset,
    config:   StochvolConfig,
    out_path: str | Path,
) -> None:
    """
    Run stochvol on all N series in dataset, with checkpointing and parallelism.

    Output structure mirrors run_mcmc_batch():
        out_path/series_{i:04d}.npz  — per-series checkpoint (crash-safe)
        out_path/results.npz         — assembled batch results

    results.npz arrays:
        means       (N, 3)          float32  — posterior means [mu, phi, sigma_eta]
        stds        (N, 3)          float32  — posterior std devs
        samples     (N, draws, 3)   float32  — posterior draws (chain 1)
        rhats       (N, 3)          float32  — Gelman-Rubin R-hat across 2 chains
        true_params (N, 3)          float32  — ground-truth parameters

    Re-running resumes from the first incomplete series (checkpoint-based).

    Args:
        dataset:  SimulationResult with .returns (N, T) and .params (N, 3).
        config:   StochvolConfig hyperparameters.
        out_path: Directory for per-series checkpoints and final results.npz.
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    N       = dataset.returns.shape[0]
    pending = _get_pending_indices(N, out_path)

    if not pending:
        logger.info("All %d series already completed. Assembling results.", N)
    else:
        logger.info(
            "Running stochvol on %d/%d pending series with %d workers.",
            len(pending), N, config.n_jobs,
        )

        def _run_one(i: int) -> tuple[int, MCMCResult]:
            result = run_stochvol_single(
                dataset.returns[i],
                config,
                seed=config.random_seed + i,
            )
            return i, result

        completed = Parallel(n_jobs=config.n_jobs, backend="multiprocessing")(
            delayed(_run_one)(i) for i in pending
        )

        for i, result in completed:
            _save_checkpoint(result, i, out_path)
            logger.info("Saved series %d/%d", i + 1, N)

    all_results = [_load_checkpoint(i, out_path) for i in range(N)]

    np.savez_compressed(
        out_path / "results.npz",
        means=np.stack([r.mean    for r in all_results]).astype(np.float32),
        stds=np.stack([r.std      for r in all_results]).astype(np.float32),
        samples=np.stack([r.samples for r in all_results]).astype(np.float32),
        rhats=np.stack([r.rhat    for r in all_results]).astype(np.float32),
        true_params=dataset.params[:, :3].astype(np.float32),
    )
    logger.info("Results saved to %s", out_path / "results.npz")
