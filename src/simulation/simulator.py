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

from .sv_params import (
    SVParams,
    SVLeverageParams,
    SVtParams,
    ASVtParams,
    draw_parameters,
    draw_nu,
)


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


def simulate_sv_leverage(
    N: int,
    T: int,
    config: SVLeverageParams | None = None,
    seed: int | None = None,
) -> SimulationResult:
    """
    Simulate N return series from the SV model with leverage effect.

    The leverage effect uses the FORWARD convention, matching stochvol
    (Kastner 2016) and the standard ASV literature (Omori et al. 2007): the
    return shock ε_t is correlated with the NEXT volatility increment η_{t+1},
    i.e. corr(ε_t, η_{t+1}) = ρ. Equivalently, η_t is correlated with the
    previous return shock ε_{t-1}. This is the timing under which ρ < 0 gives
    the leverage effect — a negative return raising subsequent volatility.

    Implemented via Cholesky decomposition of the 2×2 correlation matrix:

        Draw z1_t, z2_t ~ N(0,1) independently. Then:
            ε_t = z1_t                                    (return shock)
            η_t = ρ·z1_{t-1} + sqrt(1−ρ²)·z2_t          (volatility shock)

        so corr(ε_{t-1}, η_t) = ρ, i.e. corr(ε_t, η_{t+1}) = ρ. At t = 0 there
        is no preceding return shock, so η_0 = z2_0 (independent component only).
        η_t retains unit variance for all t: ρ² + (1−ρ²) = 1.

    The Cholesky factor sqrt(1−ρ²) is computed once per series before the
    time loop — not inside it — since ρ is fixed for each series.

    Note for methodology chapter: this forward-convention Cholesky construction
    must be described explicitly, and it matches the stochvol/ASV benchmark.
    A verification against svsim confirms the two use the same timing.

    Args:
        N:      Number of series. Must be >= 1.
        T:      Series length (time steps). Must be >= 1.
        config: SVLeverageParams instance. Defaults to SVLeverageParams().
        seed:   Integer seed for reproducibility.

    Returns:
        SimulationResult with params (N, 4) — columns [μ, φ, σ_η, ρ].

    Raises:
        ValueError: If N <= 0 or T <= 0.
        TypeError:  If seed is not an integer.
    """
    # --- Input validation ---
    if N <= 0:
        raise ValueError(f"N must be >= 1, got {N}")
    if T <= 0:
        raise ValueError(f"T must be >= 1, got {T}")
    if seed is not None and not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer or None, got {type(seed).__name__}")

    if config is None:
        config = SVLeverageParams()

    rng = np.random.default_rng(seed)

    # --- Draw base parameters [μ, φ, σ_η] for all N series ---
    base_params = draw_parameters(N, config, rng)          # (N, 3), float32
    mu        = base_params[:, 0].astype(np.float64)
    phi       = base_params[:, 1].astype(np.float64)
    sigma_eta = base_params[:, 2].astype(np.float64)

    # --- Draw ρ and concatenate to form full (N, 4) params ---
    rho = rng.uniform(*config.rho_range, size=N)           # (N,), float64
    params = np.column_stack([base_params, rho.astype(np.float32)])  # (N, 4)

    # --- Cholesky factor — computed once per series, before the time loop ---
    # chol_factor[i] = sqrt(1 - rho[i]^2): the independent component of eta_t
    chol_factor = np.sqrt(1.0 - rho ** 2)                 # (N,)

    # --- Initialise h_0 from the stationary distribution ---
    stationary_var = sigma_eta ** 2 / (1.0 - phi ** 2)
    h = rng.normal(mu, np.sqrt(stationary_var))            # (N,), float64

    # --- Pre-draw all independent noise ---
    # z1 becomes ε_t directly; z2 is combined with z1 to form η_t
    z1 = rng.standard_normal((T, N))   # (T, N) — return shocks
    z2 = rng.standard_normal((T, N))   # (T, N) — independent component of vol shock

    # --- Allocate output arrays ---
    latent_h_out = np.empty((N, T), dtype=np.float32)
    returns_out  = np.empty((N, T), dtype=np.float32)

    clipped_any = False

    # --- Time loop (vectorised over N) ---
    # Forward leverage: eta_t is correlated with the PREVIOUS return shock
    # eps_{t-1} = z1[t-1], so corr(eps_t, eta_{t+1}) = rho. At t=0 there is no
    # eps_{-1}, so eta_0 = z2[0] (a unit-variance independent shock). This
    # matches the stochvol/Omori forward convention.
    for t in range(T):
        eps_t = z1[t]                                      # (N,) return shock
        if t == 0:
            eta_t = z2[t]                                  # unit-variance; no eps_{-1}
        else:
            eta_t = rho * z1[t - 1] + chol_factor * z2[t]  # corr(eps_{t-1}, eta_t) = rho

        h = mu + phi * (h - mu) + sigma_eta * eta_t

        # Clip to prevent overflow in exp(h/2)
        if np.any(np.abs(h) > H_CLIP):
            clipped_any = True
            h = np.clip(h, -H_CLIP, H_CLIP)

        latent_h_out[:, t] = h
        returns_out[:, t]  = np.exp(h / 2.0) * eps_t

    if clipped_any:
        import warnings
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


def _simulate_t_core(
    mu: np.ndarray,
    phi: np.ndarray,
    sigma_eta: np.ndarray,
    rho: np.ndarray,
    nu: np.ndarray,
    T: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shared construction for the Student-t SV variants (SV-t and ASV-t).

    Matches the conventions verified against stochvol's svsim:
      - Standardised-t errors: eps_t = sqrt(tau_t) * z1_t with
        tau_t ~ InverseGamma(nu/2, (nu-2)/2) so E[tau_t] = 1, giving unit
        variance (exp(h_t) is the conditional variance regardless of nu).
      - Forward leverage on the NORMAL component: eta_t = rho*z1_{t-1} +
        sqrt(1-rho^2)*z2_t, so corr(z1_t, eta_{t+1}) = rho. At t=0, eta_0 = z2_0.
        rho = 0 gives the pure SV-t model.

    All parameter arrays are (N,) float64. Returns (returns, latent_h), both
    (N, T) float32.
    """
    N = mu.shape[0]
    chol_factor = np.sqrt(1.0 - rho ** 2)

    stationary_var = sigma_eta ** 2 / (1.0 - phi ** 2)
    h = rng.normal(mu, np.sqrt(stationary_var))            # h_{-1}

    z1 = rng.standard_normal((T, N))   # normal component of the return shock
    z2 = rng.standard_normal((T, N))   # independent component of the vol shock

    # tau_t ~ InverseGamma(nu/2, (nu-2)/2), per (T, N) with per-series nu.
    nu_row = nu[None, :]
    tau = 1.0 / rng.gamma(shape=nu_row / 2.0, scale=2.0 / (nu_row - 2.0), size=(T, N))

    latent_h_out = np.empty((N, T), dtype=np.float32)
    returns_out  = np.empty((N, T), dtype=np.float32)
    clipped_any = False

    for t in range(T):
        if t == 0:
            eta_t = z2[t]                                  # no eps_{-1}
        else:
            eta_t = rho * z1[t - 1] + chol_factor * z2[t]  # forward leverage on z1_{t-1}
        h = mu + phi * (h - mu) + sigma_eta * eta_t

        if np.any(np.abs(h) > H_CLIP):
            clipped_any = True
            h = np.clip(h, -H_CLIP, H_CLIP)

        eps_t = np.sqrt(tau[t]) * z1[t]                    # standardised-t return shock
        latent_h_out[:, t] = h
        returns_out[:, t]  = np.exp(h / 2.0) * eps_t

    if clipped_any:
        warnings.warn(
            f"Some h_t values exceeded H_CLIP={H_CLIP} and were clipped.",
            UserWarning,
            stacklevel=3,
        )
    return returns_out, latent_h_out


def simulate_sv_t(
    N: int,
    T: int,
    config: SVtParams | None = None,
    seed: int | None = None,
) -> SimulationResult:
    """
    Simulate N return series from the SV model with standardised Student-t errors.

    Args:
        N, T:   Number of series and length. Must be >= 1.
        config: SVtParams instance. Defaults to SVtParams().
        seed:   Integer seed for reproducibility.

    Returns:
        SimulationResult with params (N, 4) — columns [μ, φ, σ_η, ν].
    """
    if N <= 0:
        raise ValueError(f"N must be >= 1, got {N}")
    if T <= 0:
        raise ValueError(f"T must be >= 1, got {T}")
    if seed is not None and not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer or None, got {type(seed).__name__}")
    if config is None:
        config = SVtParams()

    rng = np.random.default_rng(seed)
    base = draw_parameters(N, config, rng)                 # (N, 3)
    mu        = base[:, 0].astype(np.float64)
    phi       = base[:, 1].astype(np.float64)
    sigma_eta = base[:, 2].astype(np.float64)
    nu = draw_nu(N, config.nu_range, rng)                  # (N,)
    rho = np.zeros(N)                                       # no leverage

    returns_out, latent_h_out = _simulate_t_core(mu, phi, sigma_eta, rho, nu, T, rng)
    params = np.column_stack([base, nu.astype(np.float32)])  # (N, 4)
    return SimulationResult(returns=returns_out, params=params, latent_h=latent_h_out)


def simulate_asv_t(
    N: int,
    T: int,
    config: ASVtParams | None = None,
    seed: int | None = None,
) -> SimulationResult:
    """
    Simulate N return series from the SV model with leverage AND Student-t errors.

    Args:
        N, T:   Number of series and length. Must be >= 1.
        config: ASVtParams instance. Defaults to ASVtParams().
        seed:   Integer seed for reproducibility.

    Returns:
        SimulationResult with params (N, 5) — columns [μ, φ, σ_η, ρ, ν].
    """
    if N <= 0:
        raise ValueError(f"N must be >= 1, got {N}")
    if T <= 0:
        raise ValueError(f"T must be >= 1, got {T}")
    if seed is not None and not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer or None, got {type(seed).__name__}")
    if config is None:
        config = ASVtParams()

    rng = np.random.default_rng(seed)
    base = draw_parameters(N, config, rng)                 # (N, 3)
    mu        = base[:, 0].astype(np.float64)
    phi       = base[:, 1].astype(np.float64)
    sigma_eta = base[:, 2].astype(np.float64)
    rho = rng.uniform(*config.rho_range, size=N)           # (N,)
    nu = draw_nu(N, config.nu_range, rng)                  # (N,)

    returns_out, latent_h_out = _simulate_t_core(mu, phi, sigma_eta, rho, nu, T, rng)
    params = np.column_stack([base, rho.astype(np.float32), nu.astype(np.float32)])  # (N, 5)
    return SimulationResult(returns=returns_out, params=params, latent_h=latent_h_out)


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
