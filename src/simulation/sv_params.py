"""
SVParams: configuration dataclass for the base SV model.

Owns parameter ranges (for prior sampling) and transformation functions
(for the neural network training pipeline). The simulator itself does not
apply transformations — it works in the natural parameter space.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SVParams:
    """
    Configuration for the base stochastic volatility model.

    Attributes:
        mu_range:        (min, max) for uniform prior on μ (log-volatility mean).
        phi_range:       (min, max) for uniform prior on φ (AR persistence).
        sigma_eta_range: (min, max) for uniform prior on σ_η (vol-of-vol).

    Default ranges are deliberately wide to generalise across asset classes.
    Change the ranges by constructing a custom instance — the simulation logic
    is unaffected.
    """
    mu_range:        tuple[float, float] = (-10.0, 0.0)
    phi_range:       tuple[float, float] = (0.5, 0.999)
    sigma_eta_range: tuple[float, float] = (0.05, 1.0)

    def transform(self, params: np.ndarray) -> np.ndarray:
        """
        Map an (N, 3) array of [μ, φ, σ_η] to unconstrained space.

        Transformations:
            μ     → identity           (already unconstrained)
            φ     → logit(φ)           = log(φ / (1 − φ))
            σ_η   → log(σ_η)

        Used by the neural network training pipeline so the network can output
        unconstrained values and be trained with standard regression losses.
        NOT applied by the simulator.

        Args:
            params: float array of shape (N, 3) in natural parameter space.

        Returns:
            float array of shape (N, 3) in unconstrained space.
        """
        params = np.asarray(params, dtype=np.float64)
        out = params.copy()
        phi = params[:, 1]
        sigma_eta = params[:, 2]
        out[:, 1] = np.log(phi / (1.0 - phi))      # logit
        out[:, 2] = np.log(sigma_eta)               # log
        return out

    def inverse_transform(self, params_t: np.ndarray) -> np.ndarray:
        """
        Inverse of transform(). Maps unconstrained space back to natural space.

        Transformations:
            μ_t     → identity
            φ_t     → sigmoid(φ_t)     = 1 / (1 + exp(−φ_t))
            σ_η_t   → exp(σ_η_t)

        Args:
            params_t: float array of shape (N, 3) in unconstrained space.

        Returns:
            float array of shape (N, 3) in natural parameter space.
        """
        params_t = np.asarray(params_t, dtype=np.float64)
        out = params_t.copy()
        out[:, 1] = 1.0 / (1.0 + np.exp(-params_t[:, 1]))  # sigmoid
        out[:, 2] = np.exp(params_t[:, 2])                  # exp
        return out


def draw_parameters(N: int, config: SVParams, rng: np.random.Generator) -> np.ndarray:
    """
    Sample N parameter vectors uniformly from the ranges in config.

    Args:
        N:      Number of parameter vectors to sample.
        config: SVParams instance defining the prior ranges.
        rng:    NumPy Generator (created and owned by the caller).

    Returns:
        float32 array of shape (N, 3), columns = [μ, φ, σ_η].
    """
    mu = rng.uniform(*config.mu_range, size=N)
    phi = rng.uniform(*config.phi_range, size=N)
    sigma_eta = rng.uniform(*config.sigma_eta_range, size=N)
    return np.column_stack([mu, phi, sigma_eta]).astype(np.float32)
