# Technical Plan: Base SV Model Simulator

**Status:** Draft  
**Spec:** .specs/feat-base-sv-simulator/spec.md  
**Created:** 2026-04-19  
**Last Updated:** 2026-04-19  

---

## 1. Overview

Three new files under `src/simulation/`. No existing code is modified. No new dependencies are required — NumPy and tqdm are already installed.

The design has two layers:
- **Configuration layer** (`sv_params.py`): `SVParams` dataclass owns parameter ranges and transformation functions. Nothing else needs to know the ranges or transformations — they live in one place.
- **Simulation layer** (`simulator.py`): `simulate_sv()` takes a config and a seed, draws parameters, runs the vectorised state-space recursion, and returns a `SimulationResult`. Save/load utilities live here too.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│  Caller (training script, notebook, test)           │
│                                                     │
│  result = simulate_sv(N=100_000, T=1_000, seed=42) │
│  save_dataset("data/generated/train.npz", result)  │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │      simulator.py       │
          │                         │
          │  simulate_sv()          │
          │    1. validate inputs   │
          │    2. draw_parameters() │◄──── sv_params.py
          │    3. init h_0          │      SVParams dataclass
          │    4. loop t=1..T       │      draw_parameters()
          │    5. clip + warn       │
          │    6. return result     │
          │                         │
          │  save_dataset()         │
          │  load_dataset()         │
          └─────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │  SimulationResult       │
          │  .returns  (N, T) f32   │
          │  .params   (N, 3) f32   │
          │  .latent_h (N, T) f32   │
          └─────────────────────────┘
```

---

## 3. File Structure

```
src/
  simulation/
    __init__.py      — public exports
    sv_params.py     — SVParams dataclass, draw_parameters()
    simulator.py     — SimulationResult, simulate_sv(), save_dataset(), load_dataset()
data/
  generated/         — .npz files saved here (must be git-ignored)
tests/
  test_simulator.py  — all tests
```

---

## 4. Detailed Component Design

### 4.1 `sv_params.py`

```python
@dataclass
class SVParams:
    mu_range:        tuple[float, float] = (-10.0, 0.0)
    phi_range:       tuple[float, float] = (0.5, 0.999)
    sigma_eta_range: tuple[float, float] = (0.05, 1.0)

    def transform(self, params: np.ndarray) -> np.ndarray:
        """
        Map (N, 3) array of [μ, φ, σ_η] to unconstrained space.
        Used by the neural network training pipeline — NOT by the simulator.
          μ       → identity  (already unconstrained)
          φ       → logit(φ)  = log(φ / (1 − φ))
          σ_η     → log(σ_η)
        Returns (N, 3) float32.
        """

    def inverse_transform(self, params_t: np.ndarray) -> np.ndarray:
        """Inverse of transform(). Sigmoid for φ column, exp for σ_η column."""


def draw_parameters(N: int, config: SVParams, rng: np.random.Generator) -> np.ndarray:
    """
    Sample N parameter vectors uniformly from config ranges.
    Returns float32 array of shape (N, 3), columns = [μ, φ, σ_η].
    """
```

**Why the RNG is passed in, not created here:** `simulate_sv()` creates one RNG from the user's seed and passes it through. This guarantees the full simulation (parameter draws + noise draws) is reproducible from a single seed, with no hidden state.

---

### 4.2 `simulator.py`

```python
H_CLIP = 50.0  # clip |h_t| above this value; exp(25) ≈ 7.2e10, still finite

@dataclass
class SimulationResult:
    returns:  np.ndarray   # (N, T), float32
    params:   np.ndarray   # (N, 3), float32 — columns: [μ, φ, σ_η]
    latent_h: np.ndarray   # (N, T), float32


def simulate_sv(
    N: int,
    T: int,
    config: SVParams | None = None,   # defaults to SVParams() if None
    seed: int | None = None,
) -> SimulationResult:
    """
    Simulate N independent return series of length T from the base SV model.

    Model:
        r_t   = exp(h_t / 2) * eps_t,    eps_t ~ N(0, 1)
        h_t   = mu + phi * (h_{t-1} - mu) + sigma_eta * eta_t,  eta_t ~ N(0, 1)
        h_0   ~ N(mu, sigma_eta^2 / (1 - phi^2))   [stationary distribution]

    Args:
        N:      Number of series to simulate.
        T:      Length of each series (number of time steps).
        config: SVParams instance. Defaults to SVParams() (standard ranges).
        seed:   Integer seed for reproducibility. If None, results are random.

    Returns:
        SimulationResult with .returns (N,T), .params (N,3), .latent_h (N,T).
    """
```

**Simulation loop (pseudocode):**
```
rng = np.random.default_rng(seed)
params = draw_parameters(N, config, rng)       # (N, 3)
mu, phi, sigma_eta = params columns

# Initialise from stationary distribution
stationary_std = sigma_eta / sqrt(1 - phi**2)
h = rng.normal(mu, stationary_std)             # (N,) — h_0

latent_h = zeros(N, T)
returns  = zeros(N, T)

# Pre-draw all noise — faster than drawing inside the loop
eps = rng.standard_normal((T, N))             # (T, N)
eta = rng.standard_normal((T, N))             # (T, N)

for t in range(T):                             # T iterations, not N
    h = mu + phi * (h - mu) + sigma_eta * eta[t]
    h_clipped, any_clipped = clip_with_warning(h, H_CLIP)
    latent_h[:, t] = h_clipped
    returns[:, t]  = exp(h_clipped / 2) * eps[t]

return SimulationResult(returns.astype(float32), params.astype(float32), latent_h.astype(float32))
```

**Why pre-draw all noise:** Allocating two `(T, N)` arrays upfront (800MB peak for T=1000, N=100000 in float64 — use float32 to halve this to 400MB) and then iterating is faster than calling `rng.standard_normal` inside the loop. It also makes the memory cost predictable.

**Memory budget for N=100,000, T=1,000, float32:**
| Array | Size |
|-------|------|
| `eps` (T, N) | 400 MB |
| `eta` (T, N) | 400 MB |
| `latent_h` (N, T) | 400 MB |
| `returns` (N, T) | 400 MB |
| `params` (N, 3) | ~1 MB |
| **Peak (eps+eta in memory simultaneously)** | **~1.6 GB** |

**This is tight.** A laptop with 16 GB RAM handles it fine, but 8 GB may struggle.
**Mitigation:** Process in chunks if `N * T > threshold`. The function will accept an optional `chunk_size` parameter (default: generate all at once; if memory is tight, caller can pass `chunk_size=10_000` to generate and save in 10 chunks of 10,000 series each).

---

### 4.3 `save_dataset` / `load_dataset`

```python
def save_dataset(path: str | Path, result: SimulationResult) -> None:
    """Save SimulationResult to compressed .npz file."""
    np.savez_compressed(path, returns=result.returns, params=result.params, latent_h=result.latent_h)

def load_dataset(path: str | Path) -> SimulationResult:
    """Load SimulationResult from .npz file."""
    data = np.load(path)
    return SimulationResult(returns=data["returns"], params=data["params"], latent_h=data["latent_h"])
```

---

### 4.4 `__init__.py` exports

```python
from .sv_params import SVParams, draw_parameters
from .simulator import SimulationResult, simulate_sv, save_dataset, load_dataset
```

---

## 5. Testing Strategy

All tests in `tests/test_simulator.py`. Run with `uv run pytest tests/test_simulator.py`.

| Test | Type | What it checks |
|------|------|----------------|
| `test_output_shapes` | Happy path | returns/params/latent_h shapes correct for N=100, T=50 |
| `test_all_finite` | Happy path | No NaN or inf in any output array |
| `test_reproducibility` | Happy path | Same seed → identical arrays |
| `test_different_seeds` | Happy path | Different seeds → different arrays |
| `test_single_series` | Edge case | N=1 works, shapes (1, T) |
| `test_short_series` | Edge case | T=10 works |
| `test_high_persistence_clips` | Edge case | phi=0.998 triggers clip warning, no exception |
| `test_save_load_roundtrip` | Happy path | Saved and reloaded arrays are identical |
| `test_n_zero_raises` | Error | N=0 → ValueError |
| `test_invalid_seed_raises` | Error | seed=1.5 (float) → TypeError |
| `test_statistical_sanity` | Statistical | r_t² autocorrelation > 0 (volatility clustering present) |
| `test_parameter_ranges` | Unit | draw_parameters() stays within configured bounds |
| `test_transform_roundtrip` | Unit | transform then inverse_transform recovers original params |

---

## 6. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Memory overflow at N=100k, T=1000 | Medium | High | `chunk_size` parameter; document memory requirements |
| Stationary variance undefined when `φ=1` | Low | Medium | `phi_range` max is 0.999; validation rejects φ≥1 |
| h_t drift with very large σ_η | Low | Low | H_CLIP=50 handles this; warning logged |
| `.npz` file saved to wrong path | Low | Medium | `save_dataset` creates parent directories automatically |

---

## 7. What is NOT in this plan

- No leverage (ρ) — separate feature
- No MCMC — separate feature
- No normalisation of returns — data pipeline feature
- No neural network — separate feature
- No CLI or script — a notebook or script in `notebooks/` can call `simulate_sv()` directly

---

## 8. Implementation Order

1. `sv_params.py` — `SVParams` dataclass and `draw_parameters()` (no dependencies)
2. `simulator.py` — `SimulationResult`, `simulate_sv()`, `save_dataset()`, `load_dataset()`
3. `src/simulation/__init__.py` — wire up exports
4. `tests/test_simulator.py` — full test suite
5. `data/generated/.gitkeep` — ensure directory exists and is tracked; add `data/generated/*.npz` to `.gitignore`
