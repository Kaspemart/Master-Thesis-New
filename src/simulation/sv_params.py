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


@dataclass
class SVLeverageParams(SVParams):
    """
    Configuration for the SV model with leverage effect.

    Extends SVParams by adding a correlation parameter ρ between the return
    shock ε_t and the volatility shock η_t. Negative ρ produces the leverage
    effect: negative returns tend to increase future volatility.

    Correlated shocks are generated via Cholesky decomposition:
        ε_t = z1_t
        η_t = ρ·z1_t + sqrt(1−ρ²)·z2_t,   z1_t, z2_t ~ N(0,1) independently

    rho_range default (−0.95, 0.5) covers:
        Equities:    ρ ≈ −0.7 to −0.3  (leverage effect)
        FX:          ρ ≈ −0.2 to +0.1
        Commodities: ρ up to ≈ +0.2
    Values near ±1 are excluded — they make the covariance matrix near-singular
    and are empirically implausible for any known asset class.

    Note for methodology chapter: The Cholesky decomposition approach must be
    described explicitly. See CLAUDE.md "Correlated Noise Implementation" section.
    """
    rho_range: tuple[float, float] = (-0.95, 0.5)

    def transform(self, params: np.ndarray) -> np.ndarray:
        """
        Map an (N, 4) array of [μ, φ, σ_η, ρ] to unconstrained space.

        First 3 columns: same as SVParams.transform().
        4th column (ρ): arctanh transformation = arctanh(ρ). Correct for ρ ∈ (−1,1).
        Note: logit would be wrong here — logit requires input in (0,1) but ρ can be negative.

        Args:
            params: float array of shape (N, 4) in natural parameter space.

        Returns:
            float array of shape (N, 4) in unconstrained space.
        """
        params = np.asarray(params, dtype=np.float64)
        out = np.empty_like(params)
        # First 3 columns: delegate to parent logic inline
        out[:, 0] = params[:, 0]                                    # μ: identity
        out[:, 1] = np.log(params[:, 1] / (1.0 - params[:, 1]))    # φ: logit (φ ∈ (0,1))
        out[:, 2] = np.log(params[:, 2])                            # σ_η: log
        out[:, 3] = np.arctanh(params[:, 3])                        # ρ: arctanh (ρ ∈ (−1,1)); NOT logit (logit requires input > 0)
        return out

    def inverse_transform(self, params_t: np.ndarray) -> np.ndarray:
        """
        Inverse of transform(). Maps unconstrained space back to natural space.

        4th column (ρ): tanh = (exp(2ρ_t) − 1) / (exp(2ρ_t) + 1). Inverse of arctanh.

        Args:
            params_t: float array of shape (N, 4) in unconstrained space.

        Returns:
            float array of shape (N, 4) in natural parameter space.
        """
        params_t = np.asarray(params_t, dtype=np.float64)
        out = np.empty_like(params_t)
        out[:, 0] = params_t[:, 0]                                         # μ: identity
        out[:, 1] = 1.0 / (1.0 + np.exp(-params_t[:, 1]))                 # φ: sigmoid
        out[:, 2] = np.exp(params_t[:, 2])                                 # σ_η: exp
        out[:, 3] = np.tanh(params_t[:, 3])                                # ρ: tanh (inverse of arctanh)
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
