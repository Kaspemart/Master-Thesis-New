# Feature Specification: MCMC Benchmark

**Status:** Implemented
**Phase:** Ready for Planning
**Created:** 2026-04-20
**Last Updated:** 2026-04-20
**Author:** Martin Kasperlik

---

## 1. Summary

Implements the classical MCMC benchmark for the simulation study comparison. Given a held-out test set of 200 simulated return series (with known true parameters), runs PyMC NUTS on each series and returns posterior estimates. Results are compared against neural network estimates to answer the core thesis question: under what conditions does the neural network outperform MCMC?

This feature covers the **base SV model only** (`[μ, φ, σ_η]`). The leverage extension is a separate future step.

---

## 2. Context: Where MCMC fits in the thesis workflow

```
Training data (100,000 series) ──► Neural network training ──► Trained NN
                                                                     │
Test data (200 series, T=1000) ──► MCMC benchmark  ──────────────► Comparison
         (held-out, never used       (this feature)                  │
          for NN training)                                           ▼
                                                          NN estimates vs MCMC estimates
                                                          vs known true parameters
```

MCMC is **never** run on training data. It sees only the 200 held-out test series.

---

## 3. The PyMC Model

The base SV model in PyMC notation:

```python
with pm.Model():
    # Priors — Uniform matching simulation training ranges (fair comparison)
    mu        = pm.Uniform("mu",        lower=-10.0, upper=0.0)
    phi_raw   = pm.Uniform("phi_raw",   lower=0.5,   upper=0.999)
    sigma_eta = pm.Uniform("sigma_eta", lower=0.05,  upper=1.0)

    # Latent log-volatility — AR(1) process
    # h_0 ~ N(mu, sigma_eta^2 / (1 - phi^2))  [stationary]
    # h_t = mu + phi*(h_{t-1} - mu) + sigma_eta*eta_t
    h = pm.AR(
        "h",
        rho=phi_raw,
        sigma=sigma_eta,
        constant=True,  # includes mu (intercept)
        init_dist=pm.Normal.dist(mu, sigma_eta / pm.math.sqrt(1 - phi_raw**2)),
        shape=T,
    )

    # Likelihood
    r_obs = pm.Normal("r_obs", mu=0, sigma=pm.math.exp(h / 2), observed=returns)
```

**Prior justification (methodology note):** Priors match the simulation training distribution — both NN and MCMC operate with the same information about parameter space. Limitation: in a real application the researcher would not know the true ranges; this gives MCMC slightly idealised conditions. Must be acknowledged in the results chapter.

---

## 4. MCMC Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| Draws | 1,000 | Standard; provides stable posterior estimates |
| Tuning steps | 1,000 | Standard for NUTS adaptation |
| Chains | 2 | Minimum for R-hat convergence diagnostics |
| Target accept | 0.9 | Recommended for latent variable models |
| Parallelism | 4 CPU cores (multiprocessing) | ~4 hours wall time for 200 series |
| Fallback | 500 draws + 500 tuning | If standard settings are too slow |

---

## 5. Acceptance Criteria

### AC-1: Single series — correct output structure
**Given** `run_mcmc_single(returns, config, seed)` with returns shape `(T,)`
**When** the function returns
**Then** output contains `mean [3]`, `std [3]`, `samples (draws, 3)`, `rhat [3]`, all finite

### AC-2: Parameter order consistent with simulators
**Given** MCMC output
**When** inspecting parameter arrays
**Then** column order is always `[μ, φ, σ_η]` — matching `SimulationResult.params`

### AC-3: Posterior means are in correct domains
**Given** a well-specified series
**When** MCMC converges
**Then** posterior mean of `φ ∈ (0, 1)`, `σ_η > 0`, `μ ∈ (−15, 5)`

### AC-4: R-hat convergence diagnostic
**Given** a completed run
**When** inspecting R-hat values
**Then** R-hat is computed per parameter; values > 1.01 trigger a logged warning

### AC-5: Checkpointing — crash recovery
**Given** `run_mcmc_batch()` running on 200 series
**When** the process is interrupted after series k
**Then** re-running resumes from series k+1 (already-completed results are not recomputed)

### AC-6: Batch output — full dataset saved
**Given** a completed batch run of N series
**When** saved to disk
**Then** `.npz` contains: `means (N,3)`, `stds (N,3)`, `samples (N,draws,3)`, `rhats (N,3)`, `true_params (N,3)`

### AC-7: Reproducibility
**Given** the same series and seed
**When** run twice
**Then** posterior samples are identical

### AC-8: Parallelism — 4 cores used
**Given** `run_mcmc_batch()` with `n_jobs=4`
**When** running
**Then** 4 worker processes run concurrently (verifiable via system monitor or timing)

---

## 6. Scope Boundaries

### In Scope
- `MCMCConfig` dataclass (draws, tune, chains, target_accept, n_jobs)
- `run_mcmc_single(returns, config, seed)` — fits PyMC model on one series
- `run_mcmc_batch(dataset, config, out_path)` — runs all series, checkpoints, parallelises
- Convergence diagnostics (R-hat) saved per series
- Full posterior samples saved (not just summaries)

### Out of Scope
- SV with leverage MCMC (separate feature)
- Evaluation / comparison with NN (separate evaluation module)
- Plotting / visualisation (separate)
- Real data application (separate)

---

## 7. Parameter Configuration

```
MCMCConfig:
    draws:         int   = 1000
    tune:          int   = 1000
    chains:        int   = 2
    target_accept: float = 0.9
    n_jobs:        int   = 4
    random_seed:   int   = 42
```

Per-series seed is derived as `random_seed + series_index` to ensure reproducibility while varying across series.

---

## 8. File Structure

```
src/estimation/
    __init__.py
    mcmc_config.py     — MCMCConfig dataclass
    mcmc_runner.py     — run_mcmc_single(), run_mcmc_batch(), MCMCResult

tests/
    test_mcmc.py       — unit + integration tests
```

---

## 9. Dependencies

| Dependency | Status |
|------------|--------|
| `pymc>=5.0` | Already in pyproject.toml |
| `numpy>=1.26` | Already in pyproject.toml |
| `joblib` | Needed for parallelism — add via `uv add joblib` |

---

## 10. Outstanding Questions

- **PyMC AR model API** — the exact syntax for specifying an AR(1) process with a stationary initialisation in PyMC 5 needs to be verified against the PyMC 5 docs before implementation. The pseudocode above is illustrative.
- **Convergence rate** — actual runtime per series on the target machine is unknown until a pilot run. First test on 1 series before committing to 200.

---

## 11. Test Guidance

### Happy Path Tests
- Single series with known parameters → posterior mean close to true values (within 2 posterior SDs)
- Output shapes: `means (3,)`, `stds (3,)`, `samples (draws, 3)`, `rhat (3,)`
- Batch on 3 series → `.npz` with correct shapes

### Edge Case Tests
- T=50 (very short series) — MCMC should run without crashing, R-hat may warn
- Series with extreme parameters (φ close to 0.999) — should still converge

### Checkpointing Test
- Run batch on 5 series, interrupt after 2, re-run → only 3 remaining series are computed

### Convergence Test
- Simulate series with μ=−1.5, φ=0.97, σ_η=0.15 (well-identified params)
- Run MCMC with standard settings
- Assert posterior means within 0.3 of true values (coarse but catches gross failures)
- Assert all R-hats < 1.05
