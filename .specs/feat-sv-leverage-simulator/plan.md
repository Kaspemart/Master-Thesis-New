# Technical Plan: SV with Leverage Simulator

**Status:** Approved
**Spec:** .specs/feat-sv-leverage-simulator/spec.md
**Created:** 2026-04-20
**Last Updated:** 2026-04-20

---

## 1. Overview

Three targeted changes to existing files. `simulate_sv()`, `SVParams`, and `draw_parameters()` are **not modified**. The extension is strictly additive.

| File | Change |
|------|--------|
| `src/simulation/sv_params.py` | Add `SVLeverageParams(SVParams)` class |
| `src/simulation/simulator.py` | Add `simulate_sv_leverage()` function |
| `src/simulation/__init__.py` | Export new names |
| `tests/test_sv_leverage.py` | New test file |

---

## 2. Architecture

```
SVParams                         SVLeverageParams(SVParams)
─────────────────────            ────────────────────────────────
mu_range                         mu_range       (inherited)
phi_range              ◄─────── phi_range      (inherited)
sigma_eta_range                  sigma_eta_range (inherited)
                                 rho_range       (new)

transform()  (N,3)→(N,3)        transform()     (N,4)→(N,4)  (override)
inverse_transform() ...          inverse_transform() ...       (override)

draw_parameters(N, config, rng)  ← unchanged, works for both
  returns (N,3) for SVParams
  NOT called for rho — rho is drawn separately in simulate_sv_leverage()


simulate_sv(N, T, config, seed)  simulate_sv_leverage(N, T, config, seed)
─────────────────────────────    ────────────────────────────────────────
draw_parameters() → (N,3)        draw_parameters() → (N,3) base params
                                 draw rho separately → (N,)
                                 concatenate → params (N,4)
                                 compute chol = sqrt(1-rho^2) per series
                                 pre-draw z1 (T,N), z2 (T,N)
loop t=1..T:                     loop t=1..T:
  eta[t] used directly             eps_t = z1[t]
                                   eta_t = rho*z1[t] + chol*z2[t]
  h update (same)                  h update (same)
  clip + warn (same)               clip + warn (same)
  store h, r (same)                store h, r (same)

return SimulationResult          return SimulationResult
  params (N,3)                     params (N,4)  ← only difference
```

---

## 3. Detailed Component Design

### 3.1 `SVLeverageParams` in `sv_params.py`

```python
@dataclass
class SVLeverageParams(SVParams):
    """
    Configuration for the SV model with leverage effect.

    Extends SVParams by adding a correlation parameter ρ between the
    return shock ε_t and the volatility shock η_t. Negative ρ produces
    the leverage effect: negative returns tend to increase future volatility.

    Correlated shocks are generated via Cholesky decomposition:
        ε_t = z1_t
        η_t = ρ·z1_t + sqrt(1−ρ²)·z2_t
    where z1_t, z2_t ~ N(0,1) independently.

    rho_range: (−0.95, 0.5) covers equities (−0.7 to −0.3), FX (near 0),
    and commodities (up to +0.2). Values near ±1 are excluded — they make
    the covariance matrix near-singular and are empirically implausible.
    """
    rho_range: tuple[float, float] = (-0.95, 0.5)

    def transform(self, params: np.ndarray) -> np.ndarray:
        """
        Map (N, 4) array of [μ, φ, σ_η, ρ] to unconstrained space.
        First 3 columns: same as SVParams.transform().
        4th column (ρ): logit transformation.
        """

    def inverse_transform(self, params_t: np.ndarray) -> np.ndarray:
        """
        Inverse of transform() for 4-column params.
        4th column: sigmoid.
        """
```

**Why ρ is drawn in `simulate_sv_leverage()`, not in `draw_parameters()`:**
`draw_parameters()` always returns `(N, 3)` — changing its signature would break the base simulator. Instead, `simulate_sv_leverage()` calls `draw_parameters()` for the first 3 params, draws ρ separately, and concatenates. This keeps `draw_parameters()` unchanged and the extension fully additive.

---

### 3.2 `simulate_sv_leverage()` in `simulator.py`

```python
def simulate_sv_leverage(
    N: int,
    T: int,
    config: SVLeverageParams | None = None,
    seed: int | None = None,
) -> SimulationResult:
    """
    Simulate N return series from the SV model with leverage effect.

    The leverage effect is implemented via Cholesky decomposition:
        ε_t = z1_t
        η_t = ρ·z1_t + sqrt(1−ρ²)·z2_t,   z1_t, z2_t ~ N(0,1) independently

    This induces correlation ρ between the return shock and the volatility
    shock, producing the leverage effect when ρ < 0.

    Args:
        N:      Number of series. Must be >= 1.
        T:      Series length. Must be >= 1.
        config: SVLeverageParams. Defaults to SVLeverageParams().
        seed:   Integer seed for reproducibility.

    Returns:
        SimulationResult with params (N, 4) — columns [μ, φ, σ_η, ρ].
    """
```

**Simulation loop (pseudocode):**
```
rng = np.random.default_rng(seed)
base_params = draw_parameters(N, config, rng)      # (N, 3)
rho = rng.uniform(*config.rho_range, size=N)       # (N,)
params = np.column_stack([base_params, rho])        # (N, 4)

mu, phi, sigma_eta = base_params columns (float64)

# Stationary initialisation (same as base)
h = rng.normal(mu, sqrt(sigma_eta^2 / (1-phi^2)))  # (N,)

# Cholesky factor — computed once per series, not per time step
chol = sqrt(1 - rho^2)                             # (N,)

# Pre-draw independent noise
z1 = rng.standard_normal((T, N))   # becomes eps_t
z2 = rng.standard_normal((T, N))   # combined with z1 to form eta_t

for t in range(T):
    eps_t = z1[t]                              # (N,) return shock
    eta_t = rho * z1[t] + chol * z2[t]        # (N,) volatility shock

    h = mu + phi*(h - mu) + sigma_eta * eta_t
    clip h, warn if needed
    store h, r_t = exp(h/2) * eps_t

return SimulationResult(returns, params.astype(float32), latent_h)
```

---

## 4. Testing Strategy

New file `tests/test_sv_leverage.py`. Base tests in `test_simulator.py` must still pass after changes.

| Test | What it checks |
|------|----------------|
| `test_output_shapes` | returns (N,50), params (N,4), latent_h (N,50) |
| `test_all_finite` | No NaN/inf |
| `test_reproducibility` | Same seed → identical arrays |
| `test_different_seeds` | Different seeds → different arrays |
| `test_rho_is_4th_column` | `params[:, 3]` within `(−0.95, 0.5)` |
| `test_single_series` | N=1 works |
| `test_n_zero_raises` | ValueError |
| `test_invalid_seed_raises` | TypeError |
| `test_save_load_roundtrip` | 4-column params preserved through .npz |
| `test_svleverageparams_inherits_defaults` | Base ranges unchanged |
| `test_transform_roundtrip` | 4-column transform/inverse_transform |
| `test_leverage_effect_present` | Fixed ρ=−0.7: corr(r_t, Δh_t) is negative |
| `test_rho_zero_matches_base` | ρ=0 produces same h dynamics as base model |
| `test_base_simulator_regression` | All original test_simulator.py tests still pass |

---

## 5. Implementation Order

1. `SVLeverageParams` in `sv_params.py` (no dependencies on simulator changes)
2. `simulate_sv_leverage()` in `simulator.py`
3. Update `__init__.py` exports
4. `tests/test_sv_leverage.py` — write tests, confirm red, implement, confirm green
5. Confirm `uv run pytest tests/` (both test files) passes in full

---

## 6. Risks

| Risk | Mitigation |
|------|-----------|
| Breaking base simulator | `simulate_sv()` and `SVParams` are not touched; regression test confirms |
| Cholesky factor near 0 (ρ near ±1) | Range cap `(−0.95, 0.5)` prevents this by construction |
| params (N,4) mistaken for (N,3) | Column count assertion in every shape test |
