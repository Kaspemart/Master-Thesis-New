"""
MCMC benchmark runner for the base SV model.

Fits a PyMC 5 NUTS sampler to observed return series and returns posterior
estimates for [mu, phi, sigma_eta].
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
from joblib import Parallel, delayed

from .mcmc_config import MCMCConfig

logger = logging.getLogger(__name__)


@dataclass
class MCMCResult:
    """
    Posterior estimates for a single return series.

    Attributes:
        mean:    Posterior means,  shape (3,) — [mu, phi, sigma_eta].
        std:     Posterior std devs, shape (3,).
        samples: Posterior samples, shape (draws, 3).
        rhat:    Gelman-Rubin R-hat per parameter, shape (3,).
    """
    mean:    np.ndarray
    std:     np.ndarray
    samples: np.ndarray
    rhat:    np.ndarray


def run_mcmc_single(
    returns: np.ndarray,
    config: MCMCConfig,
    seed: int,
) -> MCMCResult:
    """
    Fit the base SV model to one return series using PyMC NUTS.

    The model:
        mu        ~ Uniform(-10, 0)
        phi       ~ Uniform(0.5, 0.999)
        sigma_eta ~ Uniform(0.05, 1.0)
        h         ~ AR(1) with intercept mu*(1-phi), coef phi, noise sigma_eta
                    init h_0 ~ N(mu, sigma_eta / sqrt(1 - phi^2))   [stationary]
        r_t       ~ Normal(0, exp(h_t / 2))                         [observed]

    Args:
        returns: Observed log-returns, shape (T,).
        config:  MCMC hyperparameters.
        seed:    Random seed for reproducibility.

    Returns:
        MCMCResult with posterior mean, std, samples (draws*chains, 3), and rhat.
    """
    T = len(returns)

    # Suppress PyMC / ArviZ / NumPy progress noise
    logging.getLogger("pymc").setLevel(logging.ERROR)
    logging.getLogger("arviz").setLevel(logging.ERROR)
    logging.getLogger("pytensor").setLevel(logging.ERROR)

    with pm.Model():
        mu        = pm.Uniform("mu",        lower=-10.0, upper=0.0)
        phi       = pm.Uniform("phi",       lower=0.5,   upper=0.999)
        sigma_eta = pm.Uniform("sigma_eta", lower=0.05,  upper=1.0)

        # AR(1) latent log-volatility:
        #   h_t = mu*(1-phi) + phi*h_{t-1} + sigma_eta*eta_t
        # which is equivalent to: h_t = mu + phi*(h_{t-1} - mu) + sigma_eta*eta_t
        # init_dist = stationary distribution N(mu, sigma_eta^2 / (1 - phi^2))
        h = pm.AR(
            "h",
            rho=[mu * (1 - phi), phi],
            sigma=sigma_eta,
            constant=False,
            init_dist=pm.Normal.dist(mu, sigma_eta / pm.math.sqrt(1 - phi**2)),
            shape=T,
        )

        pm.Normal("r_obs", mu=0.0, sigma=pm.math.exp(h / 2.0), observed=returns)

        trace = pm.sample(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            target_accept=config.target_accept,
            cores=1,           # joblib handles outer parallelism; avoid nested spawning
            progressbar=False,
            random_seed=seed,
        )

    # Extract posterior samples — shape (chains, draws, 1) each; flatten to (total, 3)
    mu_samples        = trace.posterior["mu"].values.reshape(-1)
    phi_samples       = trace.posterior["phi"].values.reshape(-1)
    sigma_eta_samples = trace.posterior["sigma_eta"].values.reshape(-1)
    samples = np.stack([mu_samples, phi_samples, sigma_eta_samples], axis=1).astype(np.float32)

    mean = samples.mean(axis=0)
    std  = samples.std(axis=0)

    # R-hat diagnostics
    summary = az.summary(trace, var_names=["mu", "phi", "sigma_eta"], kind="diagnostics")
    rhat = summary["r_hat"].values.astype(np.float32)

    if np.any(rhat > 1.01):
        logger.warning(
            "R-hat > 1.01 detected for series (seed=%d): mu=%.4f, phi=%.4f, sigma_eta=%.4f. "
            "Chain may not have converged.",
            seed, rhat[0], rhat[1], rhat[2],
        )

    return MCMCResult(mean=mean, std=std, samples=samples, rhat=rhat)


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def _checkpoint_path(out_path: Path, idx: int) -> Path:
    return out_path / f"series_{idx:04d}.npz"


def _save_checkpoint(result: MCMCResult, idx: int, out_path: Path) -> None:
    p = _checkpoint_path(out_path, idx)
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, mean=result.mean, std=result.std,
                        samples=result.samples, rhat=result.rhat)
    tmp.rename(p)   # atomic rename — avoids partially-written files on crash


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

def run_mcmc_batch(
    dataset,              # SimulationResult: .returns (N,T), .params (N,3)
    config: MCMCConfig,
    out_path: str | Path,
) -> None:
    """
    Run MCMC on all N series in dataset, with checkpointing and parallelism.

    Completed series are saved as out_path/series_{i:04d}.npz immediately.
    When all series are done, assembles out_path/results.npz containing:
        means       (N, 3)         float32
        stds        (N, 3)         float32
        samples     (N, draws, 3)  float32
        rhats       (N, 3)         float32
        true_params (N, 3)         float32

    Re-running resumes from the first incomplete series.

    Args:
        dataset:  SimulationResult with .returns (N,T) and .params (N,3).
        config:   MCMC hyperparameters.
        out_path: Directory for per-series checkpoints and final results.npz.
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    N = dataset.returns.shape[0]
    pending = _get_pending_indices(N, out_path)

    if not pending:
        logger.info("All %d series already completed. Assembling results.", N)
    else:
        logger.info("Running MCMC on %d/%d pending series with %d workers.",
                    len(pending), N, config.n_jobs)

        results = Parallel(n_jobs=config.n_jobs, backend="multiprocessing")(
            delayed(run_mcmc_single)(
                dataset.returns[i],
                config,
                seed=config.random_seed + i,
            )
            for i in pending
        )

        for i, result in zip(pending, results):
            _save_checkpoint(result, i, out_path)
            logger.info("Saved series %d/%d", i + 1, N)

    # Assemble final .npz from all per-series checkpoints
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
