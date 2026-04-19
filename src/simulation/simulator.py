"""
Base SV model simulator.

Simulates batches of return series from the discrete-time stochastic volatility model:

    r_t = exp(h_t / 2) * eps_t,    eps_t ~ N(0, 1)
    h_t = mu + phi * (h_{t-1} - mu) + sigma_eta * eta_t,    eta_t ~ N(0, 1)
    h_0 ~ N(mu, sigma_eta^2 / (1 - phi^2))    [stationary distribution]

The simulation is vectorised over N series; only the time loop (T iterations)
runs sequentially. All noise is pre-drawn for speed and predictable memory usage.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .sv_params import SVParams, draw_parameters


H_CLIP = 50.0  # clip |h_t| above this; exp(25) ≈ 7.2e10, finite in float32


@dataclass
class SimulationResult:
    """
    Output of simulate_sv().

    Attributes:
        returns:  Observed log-returns, shape (N, T), float32.
        params:   True parameter vectors [μ, φ, σ_η], shape (N, 3), float32.
        latent_h: Latent log-volatility paths, shape (N, T), float32.
    """
    returns:  np.ndarray
    params:   np.ndarray
    latent_h: np.ndarray


def simulate_sv(
    N: int,
    T: int,
    config: SVParams | None = None,
    seed: int | None = None,
) -> SimulationResult:
    """
    Simulate N independent return series of length T from the base SV model.

    Args:
        N:      Number of series to simulate. Must be >= 1.
        T:      Length of each series (number of time steps). Must be >= 1.
        config: SVParams instance defining prior ranges. Defaults to SVParams()
                (standard wide ranges). Ranges can be set to fixed values by
                passing equal min/max, e.g. SVParams(phi_range=(0.97, 0.97)).
        seed:   Integer seed for reproducibility. If None, results are random
                and will differ across calls.

    Returns:
        SimulationResult with .returns (N,T), .params (N,3), .latent_h (N,T),
        all float32.

    Raises:
        ValueError: If N <= 0 or T <= 0.
        TypeError:  If seed is provided but is not an integer.
    """
    # --- Input validation ---
    if N <= 0:
        raise ValueError(f"N must be >= 1, got {N}")
    if T <= 0:
        raise ValueError(f"T must be >= 1, got {T}")
    if seed is not None and not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer or None, got {type(seed).__name__}")

    if config is None:
        config = SVParams()

    rng = np.random.default_rng(seed)

    # --- Draw parameters for all N series ---
    params = draw_parameters(N, config, rng)   # (N, 3), float32
    mu        = params[:, 0].astype(np.float64)
    phi       = params[:, 1].astype(np.float64)
    sigma_eta = params[:, 2].astype(np.float64)

    # --- Initialise h_0 from the stationary distribution ---
    # h ~ N(mu, sigma_eta^2 / (1 - phi^2))
    stationary_var = sigma_eta ** 2 / (1.0 - phi ** 2)
    h = rng.normal(mu, np.sqrt(stationary_var))   # (N,), float64

    # --- Pre-draw all noise arrays ---
    # Shape (T, N) so that eps[t] and eta[t] are contiguous slices of length N.
    eps = rng.standard_normal((T, N))   # (T, N), float64
    eta = rng.standard_normal((T, N))   # (T, N), float64

    # --- Allocate output arrays ---
    latent_h_out = np.empty((N, T), dtype=np.float32)
    returns_out  = np.empty((N, T), dtype=np.float32)

    clipped_any = False

    # --- Time loop (vectorised over N) ---
    for t in range(T):
        h = mu + phi * (h - mu) + sigma_eta * eta[t]

        # Clip to prevent overflow in exp(h/2)
        clipped = np.abs(h) > H_CLIP
        if clipped.any():
            clipped_any = True
            h = np.clip(h, -H_CLIP, H_CLIP)

        latent_h_out[:, t] = h
        returns_out[:, t]  = np.exp(h / 2.0) * eps[t]

    if clipped_any:
        warnings.warn(
            f"Some h_t values exceeded H_CLIP={H_CLIP} and were clipped. "
            "This may indicate near-unit-root φ or very large σ_η. "
            "Results are still finite but the clipped series deviate from the true model.",
            UserWarning,
            stacklevel=2,
        )

    return SimulationResult(
        returns=returns_out,
        params=params,
        latent_h=latent_h_out,
    )


def save_dataset(path: str | Path, result: SimulationResult) -> None:
    """
    Save a SimulationResult to a compressed .npz file.

    Creates parent directories if they do not exist.

    Args:
        path:   Destination path (will add .npz extension if not present).
        result: SimulationResult to save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        returns=result.returns,
        params=result.params,
        latent_h=result.latent_h,
    )


def load_dataset(path: str | Path) -> SimulationResult:
    """
    Load a SimulationResult from a .npz file saved by save_dataset().

    Args:
        path: Path to the .npz file.

    Returns:
        SimulationResult with the stored arrays.
    """
    data = np.load(Path(path))
    return SimulationResult(
        returns=data["returns"],
        params=data["params"],
        latent_h=data["latent_h"],
    )
