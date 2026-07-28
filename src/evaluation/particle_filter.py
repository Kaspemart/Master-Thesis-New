"""
Bootstrap particle filter for the stochastic volatility model.

Provides an unbiased estimate of the log-likelihood log p(r_{1:T} | theta) for
a given return series and parameter vector, by integrating over the latent
log-volatility path. This is the machinery behind the predictive log-likelihood
metric described in the methodology chapter (Section 4.8.2).

This module currently implements the BASE SV model (Gaussian observation errors,
no leverage). It is structured so that the two extensions required later —
Student-t observation errors (SV-t) and leverage (ASV) — can be added without
restructuring:

    Base SV:  h_t = mu + phi (h_{t-1} - mu) + sigma_eta * eta_t,   eta_t ~ N(0,1)
              r_t = exp(h_t / 2) * eps_t,                          eps_t ~ N(0,1)

    h_{-1} ~ N(mu, sigma_eta^2 / (1 - phi^2))    [stationary initialisation]

The model timing matches the simulator in src/simulation/simulator.py exactly:
the stationary draw acts as h_{-1}; the first AR step produces h_0, which is the
volatility state for the first observation r_0.

Estimator: standard bootstrap particle filter with systematic resampling at
every step. Resampling every step is the textbook baseline and is the variant
least prone to subtle bias; with a sufficiently large particle count the
Monte Carlo variance of the log-likelihood estimate is small. The log-likelihood
increment at each step is log( mean_i w_t^i ), where w_t^i = p(r_t | h_t^i) and
the particles h_t^i are propagated from equally-weighted (resampled) particles.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

_LOG_2PI = np.log(2.0 * np.pi)


def _obs_log_density(r_t: float, h: np.ndarray, nu: float) -> np.ndarray:
    """
    Observation log-density log p(r_t | h_t^i) for each particle.

    Base SV (nu = inf):  r_t | h_t ~ N(0, exp(h_t)).
    SV-t (finite nu):    r_t = exp(h_t/2) * eps_t, with eps_t a STANDARDISED
                         Student-t of unit variance, matching stochvol's
                         convention (verified empirically against svsim). The
                         standardisation factor is c = sqrt((nu-2)/nu) applied
                         to a raw t_nu, so exp(h_t) remains the conditional
                         variance regardless of nu.

    As nu -> inf the t-density reduces to the Gaussian case, so the two branches
    are consistent at the boundary.
    """
    if not np.isfinite(nu):
        # Gaussian: -0.5 [ log(2 pi) + h + r^2 exp(-h) ]
        return -0.5 * (_LOG_2PI + h + r_t ** 2 * np.exp(-h))

    if nu <= 2.0:
        raise ValueError(f"nu must be > 2 for finite variance, got {nu}.")

    # Standardised-t: eps = c * X_raw, c = sqrt((nu-2)/nu), so r = exp(h/2) c X.
    # x = r exp(-h/2) / c is the raw-t argument.
    log_c = 0.5 * (np.log(nu - 2.0) - np.log(nu))
    x = r_t * np.exp(-0.5 * h) / np.exp(log_c)
    const = gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * np.log(nu * np.pi)
    return (
        const
        - 0.5 * (nu + 1.0) * np.log1p(x ** 2 / nu)
        - 0.5 * h
        - log_c
    )


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Systematic resampling. Returns an index array of length n_particles.

    Lower Monte Carlo variance than multinomial resampling. `weights` must be
    normalised (sum to 1).
    """
    n = weights.shape[0]
    positions = (rng.random() + np.arange(n)) / n
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # guard against floating-point drift on the final entry
    return np.searchsorted(cumsum, positions)


def sv_log_likelihood(
    returns: np.ndarray,
    mu: float,
    phi: float,
    sigma_eta: float,
    nu: float = np.inf,
    rho: float = 0.0,
    n_particles: int = 10_000,
    seed: int | None = None,
    t_split: int | None = None,
) -> float | tuple[float, float]:
    """
    Estimate log p(r_{1:T} | theta) for the SV / SV-t model.

    Args:
        returns:     1-D array of log-returns, shape (T,).
        mu:          Long-run mean of log-volatility.
        phi:         Persistence, must satisfy |phi| < 1.
        sigma_eta:   Volatility of volatility, must be > 0.
        nu:          Degrees of freedom for standardised-t observation errors.
                     np.inf (default) gives Gaussian observation errors.
        rho:         Forward leverage correlation, corr(z_t, eta_{t+1}) = rho,
                     where z_t is the normal component of the observation shock,
                     matching the simulator and stochvol (leverage couples to the
                     normal component of the t scale-mixture, verified against
                     svsim). rho = 0 (default) is the no-leverage case.
        n_particles: Number of particles. Default 10,000.
        seed:        RNG seed for reproducibility.
        t_split:     If given, split the accumulated log-likelihood at this index
                     and return (ll_before, ll_after) where ll_before covers
                     t in [0, t_split) and ll_after covers [t_split, T). Computed
                     in a single filter pass so ll_after is the predictive
                     log-likelihood of the out-of-sample period conditioned on
                     the in-sample period. If None, returns the scalar total.

    Returns:
        Scalar total log-likelihood, or (ll_before, ll_after) if t_split is set.
        (Unbiased for the likelihood itself; the log is slightly downward-biased,
        shrinking as n_particles grows.)

    Raises:
        ValueError: on invalid parameters or empty return series.
    """
    if not np.isfinite([mu, phi, sigma_eta]).all():
        raise ValueError("Non-finite parameter passed to sv_log_likelihood.")
    if abs(phi) >= 1.0:
        raise ValueError(f"phi must satisfy |phi| < 1, got {phi}.")
    if sigma_eta <= 0.0:
        raise ValueError(f"sigma_eta must be > 0, got {sigma_eta}.")
    if abs(rho) >= 1.0:
        raise ValueError(f"rho must satisfy |rho| < 1, got {rho}.")
    if np.isfinite(nu) and nu <= 2.0:
        raise ValueError(f"nu must be > 2 for finite variance, got {nu}.")

    r = np.asarray(returns, dtype=np.float64).ravel()
    T = r.shape[0]
    if T == 0:
        raise ValueError("Empty return series.")

    rng = np.random.default_rng(seed)
    chol = np.sqrt(1.0 - rho ** 2)
    t_errors = np.isfinite(nu)

    # h_{-1}: stationary initialisation, matching the simulator.
    stationary_sd = sigma_eta / np.sqrt(1.0 - phi ** 2)
    h = rng.normal(mu, stationary_sd, size=n_particles)

    loglik = 0.0
    ll_before = 0.0
    ll_after = 0.0
    for t in range(T):
        # Propagate: h_t = mu + phi (h_{t-1} - mu) + sigma_eta * eta_t.
        # Forward leverage: eta_t depends on the previous NORMAL component z_{t-1}.
        #   For Gaussian errors z_{t-1} = eps_{t-1} = r_{t-1} exp(-h_{t-1}/2).
        #   For t errors eps_{t-1} = sqrt(tau_{t-1}) z_{t-1}, so z_{t-1} is
        #   recovered by drawing tau_{t-1} from its posterior given eps_{t-1}:
        #   tau | eps ~ InvGamma((nu+1)/2, (nu-2)/2 + eps^2/2), then
        #   z_{t-1} = eps_{t-1} / sqrt(tau_{t-1}).
        #   At t=0 there is no eps_{-1}, so eta_0 is an independent unit-variance
        #   shock. rho = 0 recovers the plain (base / SV-t) filter.
        if t == 0 or rho == 0.0:
            eta = rng.standard_normal(n_particles)
        else:
            eps_prev = r[t - 1] * np.exp(-0.5 * h)          # from resampled h_{t-1}
            if t_errors:
                b_post = 0.5 * (nu - 2.0) + 0.5 * eps_prev ** 2
                tau_prev = 1.0 / rng.gamma(0.5 * (nu + 1.0), 1.0 / b_post)
                z_prev = eps_prev / np.sqrt(tau_prev)
            else:
                z_prev = eps_prev
            eta = rho * z_prev + chol * rng.standard_normal(n_particles)
        h = mu + phi * (h - mu) + sigma_eta * eta

        # Observation log-density (Gaussian or standardised-t marginal over tau).
        log_w = _obs_log_density(r[t], h, nu)

        # Log-likelihood increment: log( mean_i exp(log_w_i) ), computed stably.
        max_lw = log_w.max()
        inc = max_lw + np.log(np.mean(np.exp(log_w - max_lw)))
        loglik += inc
        if t_split is not None:
            if t < t_split:
                ll_before += inc
            else:
                ll_after += inc

        # Normalise weights and resample every step (equal-weight bootstrap).
        w = np.exp(log_w - max_lw)
        w /= w.sum()
        idx = _systematic_resample(w, rng)
        h = h[idx]

    if t_split is not None:
        return float(ll_before), float(ll_after)
    return float(loglik)
