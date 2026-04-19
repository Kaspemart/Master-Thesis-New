# Feature Specification: Base SV Model Simulator

**Status:** Approved  
**Phase:** Ready for Planning  
**Created:** 2026-04-19  
**Last Updated:** 2026-04-19  
**Author:** Martin Kasperlik  

---

## 1. Summary

A vectorised NumPy simulator for the discrete-time stochastic volatility (SV) model. Given a set of model parameters (or sampled from a configurable prior), it produces batches of simulated return series alongside their true parameters and latent log-volatility paths. The primary use is generating the training dataset of ≥100,000 (parameter, return sequence) pairs used to train the neural network estimator. The design uses a `SVParams` dataclass to separate parameter configuration from simulation logic, making the leverage extension additive rather than requiring changes to the core simulator.

---

## 2. User Stories

- As a researcher, I want to generate a large batch of simulated SV return series with known parameters, so that I can use them as training data for the neural network estimator.
- As a researcher, I want the simulation to be exactly reproducible given the same seed, so that experiments can be replicated.
- As a researcher, I want to save the full dataset (returns, true parameters, latent paths) to disk in a compact format, so that it can be reused across all experiments without re-simulating.

---

## 3. The Model

**Observation equation:**
```
r_t = exp(h_t / 2) * ε_t,    ε_t ~ N(0, 1)
```

**State equation:**
```
h_t = μ + φ(h_{t-1} − μ) + σ_η * η_t,    η_t ~ N(0, 1)
```

**Initialisation:**
```
h_0 ~ N(μ, σ_η² / (1 − φ²))    (stationary distribution)
```

**Parameters:**
| Parameter | Domain | Description |
|-----------|--------|-------------|
| `μ` | ℝ | Long-run mean of log-volatility |
| `φ` | (−1, 1) | Persistence of log-volatility (AR coefficient) |
| `σ_η` | (0, ∞) | Volatility of log-volatility |

---

## 4. Acceptance Criteria

### AC-1: Correct output shape
**Given** a call to `simulate_sv(N=1000, T=500, params=..., seed=42)`  
**When** the function returns  
**Then** `returns.shape == (1000, 500)`, `true_params.shape == (1000, 3)`, `latent_h.shape == (1000, 500)`, and all values are finite (no NaN, no inf)

### AC-2: Reproducibility
**Given** two calls with the same seed  
**When** both complete  
**Then** the returned arrays are identical element-for-element

### AC-3: Different seeds produce different output
**Given** two calls with different seeds  
**When** both complete  
**Then** the returned arrays differ

### AC-4: Single-series case
**Given** `N=1`  
**When** the function returns  
**Then** output shapes are `(1, T)` — batch-of-one works without special casing

### AC-5: Numerical stability under high persistence
**Given** parameters with `φ` close to 0.999  
**When** the simulator runs  
**Then** `h_t` values are clipped to prevent overflow in `exp(h_t / 2)`, a warning is logged, and the function does not raise an exception

### AC-6: Parameter sampling from prior
**Given** a call to `draw_parameters(N=100000, params_config=..., seed=42)`  
**When** the function returns  
**Then** all sampled `φ` values are within `[φ_min, φ_max]`, all `σ_η` values are positive, and the distribution appears uniform (not degenerate)

### AC-7: Dataset save and load round-trip
**Given** a simulated dataset  
**When** saved to `.npz` and reloaded  
**Then** the reloaded arrays are identical to the originals

---

## 5. Scope Boundaries

### In Scope
- Simulation of the base SV model (3 parameters: `μ`, `φ`, `σ_η`)
- Batched generation: all N series simulated in parallel (vectorised over N, loop over T)
- Stationary initialisation of `h_0`
- Parameter prior sampling (uniform over configurable ranges)
- Saving/loading datasets to/from `.npz`
- Numerical safety: clip `h_t` with logged warning
- Seed-based reproducibility
- `SVParams` dataclass holding parameter ranges and transformation functions
- `draw_parameters()` function separated from simulation logic

### Out of Scope
- SV with leverage (separate feature — this spec covers base model only)
- MCMC estimation (separate feature)
- Neural network training (separate feature)
- Non-uniform priors (future extension if needed)
- Any normalisation or preprocessing of returns (handled in the data pipeline, not here)

---

## 6. Technical Constraints

| Constraint | Requirement |
|------------|-------------|
| Vectorisation | Must loop over time steps T, not over series N — no Python-level loop over series |
| Scale | Must handle N=100,000, T=1,000 without running out of memory on a standard laptop (approx. 800 MB for float64; use float32 if needed) |
| Dependencies | NumPy only — no PyTorch in the simulator itself |
| Storage format | `.npz` (NumPy compressed) — arrays: `returns (N,T)`, `params (N,3)`, `latent_h (N,T)` |
| Python version | 3.11+ |

---

## 7. Parameter Configuration

Default prior ranges (uniform sampling). These are deliberately wide to cover multiple asset classes:

| Parameter | Min | Max | Rationale |
|-----------|-----|-----|-----------|
| `μ` | −10.0 | 0.0 | Covers log-variance from ~0.5% to ~100% annualised vol |
| `φ` | 0.5 | 0.999 | From low to near-unit-root persistence |
| `σ_η` | 0.05 | 1.0 | Covers low to very high vol-of-vol regimes |

**These ranges are stored in the `SVParams` dataclass and can be changed without touching the simulation logic.**

---

## 8. Parameter Transformations

Applied so the network always operates on unconstrained values. Transformations are stored in `SVParams` alongside the ranges — the simulator does not apply them; the neural network training pipeline does.

| Parameter | Transformation | Inverse |
|-----------|---------------|---------|
| `φ` | logit: `log(φ / (1−φ))` | sigmoid |
| `σ_η` | log: `log(σ_η)` | exp |
| `μ` | none (unconstrained) | identity |

---

## 9. File Structure

```
src/
  simulation/
    __init__.py
    sv_params.py      # SVParams dataclass + draw_parameters()
    simulator.py      # simulate_sv() function
data/
  generated/          # .npz datasets saved here (git-ignored)
tests/
  test_simulator.py
```

---

## 10. Dependencies

| Dependency | Type | Status |
|------------|------|--------|
| `numpy>=1.26` | Runtime | Already in pyproject.toml |
| `tqdm>=4.66` | Runtime (progress bar) | Already in pyproject.toml |
| `pytest>=8.0` | Dev/test | Already in pyproject.toml |

No new dependencies required.

---

## 11. Outstanding Questions

None — all decisions resolved during spec interview.

---

## 12. Test Guidance

### Happy Path Tests
- Simulate N=1000, T=1000 → verify shapes `(1000,1000)`, all finite
- Same seed twice → identical arrays
- N=1 (single series) → shapes `(1, T)`, works without error
- Save to `.npz`, reload → arrays identical

### Edge Case Tests
- T=10 (very short series) → correct shape, all finite
- `φ` near 0.999 → clipping triggered, warning logged, no exception

### Error Scenario Tests
- N=0 → raises `ValueError` with clear message
- Invalid seed type (e.g. float) → raises `TypeError` with clear message

### Statistical Sanity Check
- Simulate a large batch with known parameters and verify:
  - Sample mean of `|r_t|` is consistent with the unconditional volatility implied by `(μ, φ, σ_η)`
  - Autocorrelation of `r_t²` is positive (volatility clustering)
  - This confirms the simulator produces the right distribution, not just arrays of the correct shape
