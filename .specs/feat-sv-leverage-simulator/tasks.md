# Implementation Tasks: SV with Leverage Simulator

**Status:** Draft
**Plan:** .specs/feat-sv-leverage-simulator/plan.md
**Created:** 2026-04-20
**Last Updated:** 2026-04-20

---

## Task Legend
- `[ ]` — Not started
- `[~]` — In progress
- `[x]` — Complete

---

## Phase 1: Write tests first (red state)

### Task 1.1: Create test file and confirm red
- [ ] Create `tests/test_sv_leverage.py` with all tests importing `SVLeverageParams` and `simulate_sv_leverage`
- [ ] Run `uv run pytest tests/test_sv_leverage.py` — confirm ImportError (red state)

---

## Phase 2: SVLeverageParams (TDD)

### Task 2.1: Implement SVLeverageParams
- [ ] Add `SVLeverageParams(SVParams)` to `sv_params.py` with `rho_range = (-0.95, 0.5)`
- [ ] Override `transform()` to handle (N,4): base 3 columns + logit(ρ) for 4th
- [ ] Override `inverse_transform()` to handle (N,4): base 3 columns + sigmoid for 4th
- [ ] Export from `__init__.py`
- [ ] Run `test_svleverageparams_inherits_defaults`, `test_transform_roundtrip` — must pass

---

## Phase 3: simulate_sv_leverage (TDD)

### Task 3.1: Implement simulate_sv_leverage
- [ ] Add `simulate_sv_leverage()` to `simulator.py` with full docstring
- [ ] Input validation (same as base: N≤0, T≤0, non-integer seed)
- [ ] Draw base params via `draw_parameters()`, draw ρ separately, concatenate to (N,4)
- [ ] Compute Cholesky factor `chol = sqrt(1−ρ²)` once per series before loop
- [ ] Pre-draw `z1 (T,N)` and `z2 (T,N)`
- [ ] Time loop: `eps_t = z1[t]`, `eta_t = rho*z1[t] + chol*z2[t]`, h update, clip+warn, store
- [ ] Export from `__init__.py`
- [ ] Run all Phase 3 tests — must pass

---

## Phase 4: Regression and integration

### Task 4.1: Confirm base simulator is untouched
- [ ] Run `uv run pytest tests/test_simulator.py -v` — all 31 tests must still pass

### Task 4.2: Full suite
- [ ] Run `uv run pytest tests/ -v` — all tests in both files pass
- [ ] Smoke test: `simulate_sv_leverage(N=1000, T=1000, seed=0)` completes in under 30s

---

## Commit Convention
```
feat(simulation): description
Refs: .specs/feat-sv-leverage-simulator/spec.md
```
