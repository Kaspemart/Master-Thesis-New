# Implementation Tasks: Base SV Model Simulator

**Status:** Draft  
**Plan:** .specs/feat-base-sv-simulator/plan.md  
**Created:** 2026-04-19  
**Last Updated:** 2026-04-19  

---

## Task Legend

- `[ ]` — Not started
- `[~]` — In progress
- `[x]` — Complete
- `[B]` — Blocked (reason noted)

---

## Phase 1: Setup

### Task 1.1: Directory structure and .gitignore
- [ ] Create `src/simulation/__init__.py` (empty for now)
- [ ] Create `data/generated/.gitkeep` so the directory is tracked by git
- [ ] Add `data/generated/*.npz` to `.gitignore` (dataset files must not be committed)
- [ ] Verify `src/simulation/` is importable: `python -c "import src.simulation"`

**Acceptance:** Directory structure matches the plan; `.npz` files will not be staged by git.

---

## Phase 2: SVParams — configuration layer (TDD)

### Task 2.1: Write tests for SVParams and draw_parameters
- [ ] Create `tests/test_simulator.py`
- [ ] Write `test_sv_params_defaults` — default ranges match spec values
- [ ] Write `test_draw_parameters_shape` — `draw_parameters(N=500, ...)` returns shape `(500, 3)`
- [ ] Write `test_draw_parameters_within_bounds` — all sampled values within configured ranges
- [ ] Write `test_draw_parameters_reproducible` — same RNG state → same output
- [ ] Write `test_transform_roundtrip` — `inverse_transform(transform(params))` recovers original params within float32 tolerance
- [ ] Run tests: all should **fail** (nothing implemented yet) — confirm red state

**Acceptance:** Tests exist, run, and fail with `ImportError` or `AttributeError`.

---

### Task 2.2: Implement SVParams dataclass
- [ ] Create `src/simulation/sv_params.py`
- [ ] Implement `SVParams` dataclass with `mu_range`, `phi_range`, `sigma_eta_range` fields and defaults
- [ ] Implement `draw_parameters(N, config, rng)` — uniform sampling, returns float32 `(N, 3)`
- [ ] Run `test_sv_params_defaults`, `test_draw_parameters_shape`, `test_draw_parameters_within_bounds`, `test_draw_parameters_reproducible` — all should **pass**

**Acceptance:** 4 of 5 SVParams tests pass (transform test still fails — not implemented yet).

---

### Task 2.3: Implement parameter transformations
- [ ] Implement `SVParams.transform(params)` — logit for φ column, log for σ_η column, identity for μ
- [ ] Implement `SVParams.inverse_transform(params_t)` — sigmoid for φ column, exp for σ_η column
- [ ] Run `test_transform_roundtrip` — should **pass**

**Acceptance:** All 5 SVParams/draw_parameters tests pass.

---

## Phase 3: Simulator — simulation layer (TDD)

### Task 3.1: Write tests for simulate_sv
- [ ] Write `test_output_shapes` — `simulate_sv(N=100, T=50, seed=0)` returns shapes `(100,50)`, `(100,3)`, `(100,50)`
- [ ] Write `test_all_finite` — no NaN or inf in any output array
- [ ] Write `test_reproducibility` — two calls with `seed=42` produce identical arrays
- [ ] Write `test_different_seeds` — `seed=42` vs `seed=99` produce different arrays
- [ ] Write `test_single_series` — `N=1` returns shapes `(1, T)` without error
- [ ] Write `test_short_series` — `T=10` returns correct shapes, all finite
- [ ] Write `test_n_zero_raises` — `N=0` raises `ValueError`
- [ ] Write `test_invalid_seed_raises` — `seed=1.5` raises `TypeError`
- [ ] Run tests: all should **fail**

**Acceptance:** Tests exist and fail cleanly.

---

### Task 3.2: Implement SimulationResult and input validation
- [ ] Create `src/simulation/simulator.py`
- [ ] Implement `SimulationResult` dataclass with `returns`, `params`, `latent_h` fields
- [ ] Implement input validation in `simulate_sv()`: raise `ValueError` for N≤0 or T≤0, `TypeError` for non-integer seed
- [ ] Run `test_n_zero_raises`, `test_invalid_seed_raises` — should **pass**

**Acceptance:** 2 error-case tests pass.

---

### Task 3.3: Implement the core simulation loop
- [ ] Implement `simulate_sv()` body:
  - Create `np.random.default_rng(seed)`
  - Call `draw_parameters(N, config, rng)` to get `(N,3)` params
  - Initialise `h_0 ~ N(μ, σ_η² / (1 − φ²))` (stationary distribution)
  - Pre-draw noise arrays `eps (T, N)` and `eta (T, N)` in float32
  - Loop `for t in range(T)`: update `h`, store in `latent_h`, compute `r_t`
  - Return `SimulationResult` with float32 arrays
- [ ] Run `test_output_shapes`, `test_all_finite`, `test_reproducibility`, `test_different_seeds`, `test_single_series`, `test_short_series` — all should **pass**

**Acceptance:** 6 core simulation tests pass.

---

### Task 3.4: Implement numerical clipping with warning
- [ ] Add `H_CLIP = 50.0` constant
- [ ] Inside the time loop: clip `h` to `[-H_CLIP, H_CLIP]` and emit a `warnings.warn()` if any value was clipped
- [ ] Write `test_high_persistence_clips` — force `φ=0.9985`, verify warning is emitted via `pytest.warns()`, verify output is still finite
- [ ] Run `test_high_persistence_clips` — should **pass**

**Acceptance:** Clipping test passes; all previous tests still pass.

---

### Task 3.5: Implement save_dataset and load_dataset
- [ ] Implement `save_dataset(path, result)` using `np.savez_compressed`; create parent directory if it does not exist
- [ ] Implement `load_dataset(path)` returning a `SimulationResult`
- [ ] Write `test_save_load_roundtrip` — save then load, verify arrays are identical
- [ ] Run `test_save_load_roundtrip` — should **pass**

**Acceptance:** Round-trip test passes; saved file is a valid `.npz`.

---

### Task 3.6: Statistical sanity test
- [ ] Write `test_statistical_sanity`:
  - Simulate N=2000, T=500 with fixed parameters `μ=−1.5, φ=0.97, σ_η=0.15`
  - Assert autocorrelation of `r_t²` at lag 1 is positive across the batch (volatility clustering)
  - Assert sample mean of `r_t` is close to 0 (unbiased returns)
- [ ] Run `test_statistical_sanity` — should **pass**

**Acceptance:** Statistical properties of simulated data match SV model theory.

---

## Phase 4: Integration

### Task 4.1: Wire up public exports
- [ ] Update `src/simulation/__init__.py` to export: `SVParams`, `draw_parameters`, `SimulationResult`, `simulate_sv`, `save_dataset`, `load_dataset`
- [ ] Verify: `from src.simulation import simulate_sv, SVParams` works from project root

**Acceptance:** Public API importable with one import line.

---

### Task 4.2: Full test suite passes
- [ ] Run `uv run pytest tests/test_simulator.py -v`
- [ ] All 13 tests pass
- [ ] No warnings other than the intentional clip warning in `test_high_persistence_clips`

**Acceptance:** `pytest` exits 0, 13 passed.

---

### Task 4.3: Smoke test with realistic scale
- [ ] Run `simulate_sv(N=1_000, T=1_000, seed=0)` and confirm it completes in under 30 seconds
- [ ] Check memory usage is reasonable (no swap, no crash)
- [ ] Save result to `data/generated/smoke_test.npz`, verify file is created, delete it

**Acceptance:** Realistic-scale call completes successfully; file saves and loads correctly.

---

## Commit Convention

All commits for this feature:
```
type(simulation): description

Refs: .specs/feat-base-sv-simulator/spec.md
```

Examples:
```
feat(simulation): add SVParams dataclass and draw_parameters
feat(simulation): implement simulate_sv core loop
feat(simulation): add save/load dataset utilities
test(simulation): add full simulator test suite
```

---

## Notes

*(Space for notes during implementation)*
