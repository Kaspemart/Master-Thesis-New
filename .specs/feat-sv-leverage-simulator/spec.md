# Feature Specification: SV with Leverage Simulator

**Status:** Approved
**Phase:** Ready for Planning
**Created:** 2026-04-20
**Last Updated:** 2026-04-20
**Author:** Martin Kasperlik

---

## 1. Summary

Extends the base SV simulator to handle the SV with leverage model, which adds a correlation parameter ρ between the return shock and the volatility shock. This is the second of the two primary implementation targets. The extension is additive — existing files are extended, no existing behaviour changes. `SVLeverageParams` inherits from `SVParams` and `simulate_sv_leverage()` is added alongside the existing `simulate_sv()`.

---

## 2. The Model

**Observation equation** (same as base):
```
r_t = exp(h_t / 2) * ε_t
```

**State equation** (same as base):
```
h_t = μ + φ(h_{t-1} − μ) + σ_η * η_t
```

**Correlated noise via Cholesky decomposition:**
```
Draw z1_t, z2_t ~ N(0,1) independently, then:
  ε_t = z1_t
  η_t = ρ · z1_t + sqrt(1 − ρ²) · z2_t
```

This is mathematically equivalent to drawing [ε_t, η_t] jointly from N(0, Σ) where Σ = [[1, ρ], [ρ, 1]].

**Parameters:**
| Parameter | Domain | Description |
|-----------|--------|-------------|
| `μ` | ℝ | Long-run mean of log-volatility |
| `φ` | (−1, 1) | Persistence of log-volatility |
| `σ_η` | (0, ∞) | Volatility of log-volatility |
| `ρ` | (−1, 1) | Correlation between return and volatility shocks (leverage effect) |

**Note for methodology chapter:** The Cholesky decomposition approach must be described explicitly when writing up the leverage model. The factorisation ensures a valid covariance matrix and maps cleanly onto the independent noise structure of the base model.

---

## 3. Parameter Configuration

Training ranges (deliberately chosen based on empirical evidence across asset classes):

| Parameter | Min | Max | Rationale |
|-----------|-----|-----|-----------|
| `μ` | −10.0 | 0.0 | Inherited from base |
| `φ` | 0.5 | 0.999 | Inherited from base |
| `σ_η` | 0.05 | 1.0 | Inherited from base |
| `ρ` | −0.95 | 0.5 | Covers equities (−0.7 to −0.3), FX (near 0), commodities (up to +0.2); avoids near-singular extremes |

The range `(−0.99, 0.99)` was explicitly rejected: values near ±1 make the covariance matrix near-singular and are economically implausible for any known asset class.

---

## 4. Acceptance Criteria

### AC-1: Correct output shape
**Given** `simulate_sv_leverage(N=1000, T=500, seed=42)`
**When** the function returns
**Then** `returns.shape == (1000, 500)`, `params.shape == (1000, 4)`, `latent_h.shape == (1000, 500)`, all float32, all finite

### AC-2: ρ is the 4th parameter column
**Given** a simulation result
**When** inspecting `result.params`
**Then** columns are [μ, φ, σ_η, ρ] — ρ is always the 4th column, within `(−0.95, 0.5)`

### AC-3: Reproducibility
**Given** two calls with the same seed
**When** both complete
**Then** arrays are identical element-for-element

### AC-4: Base model is a special case
**Given** `SVLeverageParams` with `rho_range=(0.0, 0.0)` (fixed ρ=0)
**When** simulating
**Then** results are statistically indistinguishable from the base model with the same μ, φ, σ_η

### AC-5: Leverage effect is present in simulated data
**Given** a large simulation with ρ fixed at −0.7
**When** computing the sample correlation between `r_t` and `h_{t+1} − h_t`
**Then** the mean correlation across series is negative and close to the true ρ

### AC-6: SVLeverageParams inherits base behaviour
**Given** `SVLeverageParams()` with no arguments
**When** inspecting defaults
**Then** `mu_range`, `phi_range`, `sigma_eta_range` match `SVParams()` defaults exactly

### AC-7: Transform/inverse_transform includes ρ
**Given** params array of shape (N, 4)
**When** `transform()` then `inverse_transform()` applied
**Then** original params recovered within float32 tolerance; ρ column uses logit/sigmoid

### AC-8: Separate datasets
**Given** base SV and leverage SV simulations
**When** saved to disk
**Then** they are stored as separate `.npz` files; leverage file has `params` with 4 columns, base has 3

---

## 5. Scope Boundaries

### In Scope
- `SVLeverageParams(SVParams)` dataclass — inherits base ranges, adds `rho_range`
- `simulate_sv_leverage()` function — new function in `simulator.py`
- Updated `transform()` and `inverse_transform()` on `SVLeverageParams` — handles 4-column params
- `draw_parameters()` works unchanged (takes any `SVParams` instance)
- Updated `__init__.py` exports

### Out of Scope
- Modifying `simulate_sv()` or `SVParams` — base simulator is untouched
- Any other leverage variants (e.g. asymmetric leverage, stochastic ρ)
- MCMC estimation
- Neural network training

---

## 6. Technical Constraints

| Constraint | Requirement |
|------------|-------------|
| Vectorisation | Same as base — loop over T, vectorised over N |
| Cholesky factor | Compute `sqrt(1−ρ²)` once per series before the time loop, not inside it |
| Dependencies | NumPy only |
| Storage | Same `.npz` format; leverage datasets stored separately from base datasets |
| Backward compatibility | `simulate_sv()` and `SVParams` must remain unchanged |

---

## 7. File Structure

No new files. Extensions to existing files only:

```
src/simulation/
  sv_params.py     ← add SVLeverageParams(SVParams)
  simulator.py     ← add simulate_sv_leverage()
  __init__.py      ← add new exports

tests/
  test_sv_leverage.py   ← new test file (keeps base tests separate)
```

---

## 8. Dependencies

No new dependencies. All existing.

---

## 9. Outstanding Questions

None — all decisions resolved during spec interview.

---

## 10. Test Guidance

### Happy Path Tests
- `simulate_sv_leverage(N=100, T=50, seed=0)` → correct shapes, all finite, all float32
- Same seed → identical arrays
- `params[:, 3]` all within `(−0.95, 0.5)`

### Edge Case Tests
- N=1 → shapes `(1, T)`, works without error
- ρ fixed at 0 → output statistically indistinguishable from base model

### Statistical Sanity Tests
- Fixed ρ=−0.7: sample correlation between `r_t` and `h_{t+1}−h_t` is negative (leverage present)
- Volatility clustering still present (`r_t²` autocorrelation > 0)

### Regression Tests
- `simulate_sv()` still passes all original tests after the extension is merged — base simulator must not be broken
