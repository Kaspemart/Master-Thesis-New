# Technical Plan: MCMC Benchmark

**Status:** Approved
**Spec:** .specs/feat-mcmc-benchmark/spec.md
**Created:** 2026-04-20
**Last Updated:** 2026-04-20

---

## 1. Overview

Implement a PyMC 5 NUTS sampler pipeline that processes 200 held-out test series through the base SV model and returns posterior estimates for `[μ, φ, σ_η]`. The implementation is three layers: a config dataclass, a single-series runner, and a batch orchestrator with checkpointing and joblib parallelism.

The pilot-run-first principle governs execution: run 1 series to verify PyMC API, runtime, and output shape before committing to the full 200.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Entry Point                              │
│              run_mcmc_batch(dataset, config, out_path)       │
│                         │                                    │
│              ┌──────────┴──────────┐                        │
│              │  Checkpoint Layer   │                        │
│              │  - load existing    │                        │
│              │  - skip completed   │                        │
│              │  - save per-series  │                        │
│              └──────────┬──────────┘                        │
│                         │                                    │
│              ┌──────────┴──────────┐                        │
│              │  joblib Parallel    │                        │
│              │  n_jobs=4           │                        │
│              └──────────┬──────────┘                        │
└─────────────────────────┼────────────────────────────────────┘
                          │  (per worker)
┌─────────────────────────▼────────────────────────────────────┐
│              run_mcmc_single(returns, config, seed)           │
│                         │                                    │
│              ┌──────────▼──────────┐                        │
│              │  PyMC Model         │                        │
│              │  pm.Uniform priors  │                        │
│              │  pm.AR latent h     │                        │
│              │  pm.Normal obs r    │                        │
│              └──────────┬──────────┘                        │
│                         │                                    │
│              ┌──────────▼──────────┐                        │
│              │  NUTS Sampler       │                        │
│              │  draws=1000         │                        │
│              │  tune=1000          │                        │
│              │  chains=2           │                        │
│              │  target_accept=0.9  │                        │
│              └──────────┬──────────┘                        │
│                         │                                    │
│              ┌──────────▼──────────┐                        │
│              │  MCMCResult         │                        │
│              │  mean[3], std[3]    │                        │
│              │  samples(draws,3)   │                        │
│              │  rhat[3]            │                        │
│              └─────────────────────┘                        │
└──────────────────────────────────────────────────────────────┘
```

### 2.1 Component Overview

| Component | File | Role |
|-----------|------|------|
| `MCMCConfig` | `mcmc_config.py` | Typed configuration (draws, tune, chains, target_accept, n_jobs, random_seed) |
| `MCMCResult` | `mcmc_runner.py` | Output dataclass for a single series |
| `run_mcmc_single` | `mcmc_runner.py` | Builds PyMC model, runs NUTS, returns MCMCResult |
| `run_mcmc_batch` | `mcmc_runner.py` | Orchestrates parallel runs, checkpoints, assembles final .npz |

### 2.2 Affected Components

| Component | Change Type | Description |
|-----------|-------------|-------------|
| `src/estimation/__init__.py` | New | Module exports |
| `src/estimation/mcmc_config.py` | New | MCMCConfig dataclass |
| `src/estimation/mcmc_runner.py` | New | Single + batch runner |
| `tests/test_mcmc.py` | New | All tests |
| `pyproject.toml` | Modify | Add joblib dependency |

---

## 3. PyMC Model Specification

The PyMC 5 AR model uses `constant=True` to include an intercept (the mean `μ`). When `constant=True`, the `rho` parameter takes 2 elements: `[intercept, AR_coefficient]`.

```python
with pm.Model() as model:
    mu        = pm.Uniform("mu",        lower=-10.0, upper=0.0)
    phi       = pm.Uniform("phi",       lower=0.5,   upper=0.999)
    sigma_eta = pm.Uniform("sigma_eta", lower=0.05,  upper=1.0)

    # AR(1) with stationary initialisation
    # constant=True: rho=[intercept, AR_coef] = [mu*(1-phi), phi]
    # Equivalent to: h_t = mu*(1-phi) + phi*h_{t-1} + sigma_eta*eta_t
    # which is: h_t = mu + phi*(h_{t-1} - mu) + sigma_eta*eta_t
    h = pm.AR(
        "h",
        rho=[mu * (1 - phi), phi],
        sigma=sigma_eta,
        constant=False,   # intercept already in rho[0]
        init_dist=pm.Normal.dist(mu, sigma_eta / pm.math.sqrt(1 - phi**2)),
        shape=T,
    )

    r_obs = pm.Normal("r_obs", mu=0.0, sigma=pm.math.exp(h / 2.0), observed=returns)
```

**Note on AR API:** `constant=True` adds an intercept term internally; instead we pass the intercept `mu*(1-phi)` as the first element of `rho` with `constant=False`. This is the cleaner approach that avoids confusion about how PyMC 5 handles the `constant=True` reparameterisation. The stationary variance `σ_η²/(1−φ²)` requires `φ < 1`, which the prior guarantees (upper=0.999).

**Verification before batch run:** Run a pilot on 1 series to confirm the AR model produces sensible h paths and NUTS does not diverge immediately.

---

## 4. Data Contracts

### Input

```python
run_mcmc_single(
    returns: np.ndarray,    # shape (T,), float — observed log-returns
    config: MCMCConfig,
    seed: int,
) -> MCMCResult
```

```python
run_mcmc_batch(
    dataset: SimulationResult,  # .returns (N,T), .params (N,3)
    config: MCMCConfig,
    out_path: Path | str,       # directory for checkpoints + final .npz
) -> None                       # saves to disk, returns nothing
```

### MCMCResult

```python
@dataclass
class MCMCResult:
    mean:    np.ndarray   # (3,) — posterior means [mu, phi, sigma_eta]
    std:     np.ndarray   # (3,) — posterior standard deviations
    samples: np.ndarray   # (draws, 3) — thinned posterior samples
    rhat:    np.ndarray   # (3,) — Gelman-Rubin R-hat per parameter
```

Parameter order is always `[μ, φ, σ_η]` — matching `SimulationResult.params`.

### Final .npz Output

```
means       (N, 3)         float32
stds        (N, 3)         float32
samples     (N, draws, 3)  float32
rhats       (N, 3)         float32
true_params (N, 3)         float32
```

---

## 5. Checkpointing Design

Each completed series is saved immediately as `out_path/series_{i:04d}.npz`. On re-run, `run_mcmc_batch` scans for existing per-series files and skips those indices.

```
out_path/
    series_0000.npz   ← MCMCResult arrays for series 0
    series_0001.npz
    ...
    series_0199.npz   ← all done
    results.npz       ← assembled after all series complete
```

Per-series .npz keys: `mean`, `std`, `samples`, `rhat`.

Assembly step: once all series are done, load all per-series files, stack into batch arrays, append `true_params` from the dataset, save `results.npz`.

---

## 6. Parallelism Design

```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=config.n_jobs, backend="multiprocessing")(
    delayed(run_mcmc_single)(
        dataset.returns[i],
        config,
        seed=config.random_seed + i,
    )
    for i in pending_indices
)
```

Each worker is an independent process (no shared memory issues with PyMC's internal state). Seeds are derived as `random_seed + series_index` for reproducibility.

**Checkpoint integration:** After each parallel batch of results returns, iterate through results and save per-series files before proceeding. Since joblib collects all results before returning, checkpointing happens in batches rather than after every single series. This is acceptable — if the job is killed mid-batch, those series will be recomputed.

Alternative: use `joblib` with `return_as="generator"` if available, or switch to sequential with per-series saving if crash recovery needs to be finer-grained.

---

## 7. Dependencies

### New Dependencies

| Package | Version | Justification |
|---------|---------|---------------|
| `joblib` | latest | Parallel batch execution across 4 CPU cores |

Install: `uv add joblib`

### Existing Dependencies (already in pyproject.toml)

| Package | Used for |
|---------|----------|
| `pymc>=5.0` | NUTS sampler and probabilistic model |
| `numpy>=1.26` | Array operations, .npz I/O |

---

## 8. Convergence Diagnostics

R-hat is extracted from PyMC's `az.summary()` output. Any R-hat > 1.01 triggers a `logging.warning`. The warning does not abort the run — all series complete, including non-converged ones, and the R-hat array in the output lets the user filter post-hoc.

---

## 9. Testing Strategy

| Test | Type | Approach |
|------|------|----------|
| `test_mcmc_config_defaults` | Unit | Instantiate MCMCConfig, check field values |
| `test_mcmc_config_custom` | Unit | Override fields, check values |
| `test_single_output_shape` | Integration (fast) | Simulate 1 short series (T=50), run with 50 draws + 50 tune, check MCMCResult shapes |
| `test_single_output_finite` | Integration (fast) | All values in MCMCResult are finite |
| `test_parameter_order` | Integration (fast) | Column 0=mu, 1=phi, 2=sigma_eta |
| `test_rhat_computed` | Integration (fast) | rhat has 3 elements, all positive |
| `test_rhat_warning` | Unit (mock) | Mock a high-rhat result, assert warning logged |
| `test_reproducibility` | Integration (fast) | Same seed → identical samples |
| `test_batch_shapes` | Integration (medium) | 3 series → results.npz with correct shapes |
| `test_checkpointing` | Integration (medium) | Run 5 series, interrupt after 2, re-run → only 3 recomputed |
| `test_convergence` | Integration (slow, optional) | μ=−1.5, φ=0.97, σ_η=0.15 → posterior means within 0.3 of truth |

Tests marked "fast" use reduced MCMC settings (50 draws, 50 tune, 1 chain) to keep CI manageable.

---

## 10. Implementation Notes

1. **PyMC verbosity:** PyMC prints progress bars to stderr by default. Suppress with `pm.sample(..., progressbar=False)` or set `logging.getLogger("pymc").setLevel(logging.WARNING)`.

2. **PyMC + multiprocessing:** PyMC uses its own internal multiprocessing for chains. When running joblib with `backend="multiprocessing"`, each worker then also spawns sub-processes for chains. Set `chains=2, cores=1` inside `pm.sample()` to prevent nested multiprocessing conflicts — joblib handles the outer parallelism, PyMC runs chains sequentially within each worker.

3. **ArviZ dependency:** PyMC 5 uses ArviZ for diagnostics. `az.summary()` returns a DataFrame; extract R-hat as `summary["r_hat"].values`.

4. **init_dist numerical stability:** `sigma_eta / pm.math.sqrt(1 - phi**2)` can become large when `phi` approaches 0.999. The prior upper bound of 0.999 gives `sqrt(1 - 0.999²) ≈ 0.045`, so `sigma_eta=1.0` gives `std ≈ 22`. This is large but finite — no special handling needed.

5. **Pilot run first:** Before any batch invocation, run 1 series with standard settings and verify: (a) no exceptions, (b) R-hat < 1.1, (c) wall time. Abort and diagnose if (a) or (b) fail.

---

## 11. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| PyMC AR API misspecified | Medium | High | Verify with pilot run; unit test checks h shape matches T |
| Nested multiprocessing deadlock | Medium | High | Set `cores=1` in pm.sample; joblib handles outer parallelism |
| Runtime > 4 hours | Medium | Medium | Fallback to 500+500 draws; parallelism already mitigates |
| Non-convergence on extreme φ | Low | Low | Log warning; don't abort; filter by R-hat post-hoc |
| Checkpoint file corruption | Very Low | Medium | Write to temp file then rename (atomic write) |
