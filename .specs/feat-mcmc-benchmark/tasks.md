# Implementation Tasks: MCMC Benchmark

**Status:** Draft
**Plan:** .specs/feat-mcmc-benchmark/plan.md
**Spec:** .specs/feat-mcmc-benchmark/spec.md
**Created:** 2026-04-20
**Last Updated:** 2026-04-20

---

## Task Legend

- `[ ]` — Not started
- `[~]` — In progress
- `[x]` — Complete
- `[P]` — Can be parallelized with adjacent [P] tasks
- `[B]` — Blocked (note reason)

---

## Phase 1: Setup

### Task 1.1: Add joblib dependency
- [ ] Run `uv add joblib`
- [ ] Verify `import joblib` works in the venv
- Acceptance: `uv sync` clean, joblib importable

### Task 1.2: Create estimation module skeleton
- [ ] Create `src/estimation/__init__.py` (empty for now)
- [ ] Create `src/estimation/mcmc_config.py` (empty stub)
- [ ] Create `src/estimation/mcmc_runner.py` (empty stub)
- [ ] Create `tests/test_mcmc.py` (empty stub)
- Acceptance: `python -c "import src.estimation"` succeeds

---

## Phase 2: MCMCConfig (TDD)

### Task 2.1: MCMCConfig dataclass
**Tests first:**
- [ ] Write `test_mcmc_config_defaults` — instantiate MCMCConfig(), assert draws=1000, tune=1000, chains=2, target_accept=0.9, n_jobs=4, random_seed=42
- [ ] Write `test_mcmc_config_custom` — override all fields, assert values propagate

**Implementation:**
- [ ] Implement MCMCConfig in `mcmc_config.py`:
  ```python
  @dataclass
  class MCMCConfig:
      draws: int = 1000
      tune: int = 1000
      chains: int = 2
      target_accept: float = 0.9
      n_jobs: int = 4
      random_seed: int = 42
  ```

**Verify:**
- [ ] `pytest tests/test_mcmc.py::test_mcmc_config_defaults` passes
- [ ] `pytest tests/test_mcmc.py::test_mcmc_config_custom` passes
- Acceptance: Both config tests green

---

## Phase 3: MCMCResult and run_mcmc_single (TDD)

### Task 3.1: MCMCResult dataclass [P]
**Tests first:**
- [ ] Write `test_mcmc_result_fields` — construct MCMCResult with dummy arrays, check .mean .std .samples .rhat accessible

**Implementation:**
- [ ] Implement MCMCResult in `mcmc_runner.py`:
  ```python
  @dataclass
  class MCMCResult:
      mean:    np.ndarray   # (3,)
      std:     np.ndarray   # (3,)
      samples: np.ndarray   # (draws, 3)
      rhat:    np.ndarray   # (3,)
  ```
- Acceptance: test_mcmc_result_fields passes

### Task 3.2: run_mcmc_single — output structure [P]
**Tests first (use reduced settings: draws=50, tune=50, chains=1 for speed):**
- [ ] Write `test_single_output_shape` — simulate 1 series T=100, run with fast config, check MCMCResult shapes: mean(3,), std(3,), samples(50,3), rhat(3,)
- [ ] Write `test_single_output_finite` — all values in MCMCResult are finite (no NaN/inf)
- [ ] Write `test_parameter_order` — column 0 corresponds to mu, 1 to phi, 2 to sigma_eta (check mu posterior mean is negative, phi posterior mean > 0.5, sigma_eta posterior mean > 0)

**Implementation:**
- [ ] Implement `run_mcmc_single(returns, config, seed)` in `mcmc_runner.py`
- [ ] Build PyMC model: Uniform priors, AR(1) latent h, Normal obs
  - `rho=[mu*(1-phi), phi]`, `constant=False`
  - `init_dist=pm.Normal.dist(mu, sigma_eta / pm.math.sqrt(1 - phi**2))`
- [ ] Call `pm.sample(draws, tune, chains, target_accept, cores=1, progressbar=False, random_seed=seed)`
- [ ] Extract posterior samples: stack `[mu, phi, sigma_eta]` in that order → (draws×chains, 3)
- [ ] Extract R-hat from `az.summary()`
- [ ] Suppress PyMC/ArviZ logging noise

**Verify:**
- [ ] `pytest tests/test_mcmc.py -k "single"` passes (may take 1-2 min)
- Acceptance: All three single-series tests green

### Task 3.3: run_mcmc_single — reproducibility
**Tests first:**
- [ ] Write `test_reproducibility` — same series, same seed → identical samples arrays

**Implementation:**
- [ ] Confirm seed is passed through to `pm.sample(random_seed=seed)` correctly

**Verify:**
- [ ] `pytest tests/test_mcmc.py::test_reproducibility` passes
- Acceptance: Identical samples on re-run

### Task 3.4: run_mcmc_single — R-hat warning
**Tests first:**
- [ ] Write `test_rhat_warning_logged` — mock a result where rhat > 1.01, assert `logging.warning` is called
  - (Use `unittest.mock.patch` on logging, or check caplog fixture)

**Implementation:**
- [ ] Add R-hat check in run_mcmc_single: `if any(rhat > 1.01): logging.warning(...)`

**Verify:**
- [ ] `pytest tests/test_mcmc.py::test_rhat_warning_logged` passes
- Acceptance: Warning fires correctly

---

## Phase 4: Pilot Run Verification

### Task 4.1: Manual pilot run (1 series, full settings)
- [ ] Generate 1 test series: `simulate_sv(N=1, T=1000, seed=0)`
- [ ] Run `run_mcmc_single` with standard MCMCConfig (draws=1000, tune=1000, chains=2)
- [ ] Record wall time
- [ ] Verify R-hat < 1.05 for all parameters
- [ ] Verify posterior means are plausible (mu < 0, phi > 0.5, sigma_eta > 0)
- Acceptance: Run completes without error, R-hat clean, timing recorded
- **Gate:** Do not proceed to batch implementation until this passes

---

## Phase 5: run_mcmc_batch (TDD)

### Task 5.1: Checkpointing helpers
**Tests first:**
- [ ] Write `test_checkpoint_save_load` — save a MCMCResult to checkpoint file, reload it, check arrays equal
- [ ] Write `test_pending_indices_skips_existing` — create fake checkpoint files for series 0,1; assert pending_indices = [2,3,4] for N=5

**Implementation:**
- [ ] Implement `_save_checkpoint(result, idx, out_path)` — writes `series_{idx:04d}.npz`
- [ ] Implement `_load_checkpoint(idx, out_path) -> MCMCResult`
- [ ] Implement `_get_pending_indices(N, out_path) -> list[int]` — scans for existing checkpoint files

**Verify:**
- [ ] `pytest tests/test_mcmc.py -k "checkpoint"` passes
- Acceptance: Checkpoint save/load round-trips correctly, pending indices correct

### Task 5.2: run_mcmc_batch — batch execution
**Tests first:**
- [ ] Write `test_batch_output_shapes` — run on 3 series (T=100, fast config), verify results.npz keys and shapes: means(3,3), stds(3,3), samples(3,50,3), rhats(3,3), true_params(3,3)
- [ ] Write `test_batch_true_params_preserved` — true_params in output matches dataset.params

**Implementation:**
- [ ] Implement `run_mcmc_batch(dataset, config, out_path)`:
  1. Determine pending_indices
  2. Run `Parallel(n_jobs=config.n_jobs, backend="multiprocessing")` over pending
  3. Save each result as checkpoint immediately after parallel batch returns
  4. After all series done, assemble and save `results.npz`
- [ ] Populate true_params from `dataset.params`

**Verify:**
- [ ] `pytest tests/test_mcmc.py -k "batch"` passes
- Acceptance: results.npz present with correct shapes and keys

### Task 5.3: Checkpointing crash recovery
**Tests first:**
- [ ] Write `test_crash_recovery` — run batch on 5 series with fast config; after completion, delete results.npz and series_0003.npz and series_0004.npz; re-run; assert only series 3 and 4 were recomputed (verify by checking file mtimes or by counting calls)

**Implementation:**
- [ ] Confirm pending_indices logic correctly excludes all existing checkpoints

**Verify:**
- [ ] `pytest tests/test_mcmc.py::test_crash_recovery` passes
- Acceptance: Crash recovery works

---

## Phase 6: Module Exports and Integration

### Task 6.1: Wire up __init__.py
- [ ] Export from `src/estimation/__init__.py`: MCMCConfig, MCMCResult, run_mcmc_single, run_mcmc_batch
- [ ] Write `test_imports` — `from src.estimation import MCMCConfig, MCMCResult, run_mcmc_single, run_mcmc_batch`
- Acceptance: Imports clean, no circular imports

### Task 6.2: Full test suite passes
- [ ] Run `pytest tests/test_mcmc.py -v`
- [ ] Run `ruff check src/estimation/ tests/test_mcmc.py`
- Acceptance: All tests green, no lint errors

---

## Phase 7: Optional Slow Test

### Task 7.1: Convergence test (slow, skip in CI)
- [ ] Mark with `@pytest.mark.slow`
- [ ] Simulate series with μ=−1.5, φ=0.97, σ_η=0.15 (T=1000)
- [ ] Run with standard MCMCConfig
- [ ] Assert all posterior means within 0.3 of true values
- [ ] Assert all R-hats < 1.05
- Acceptance: MCMC recovers known parameters

---

## Phase 8: Completion

### Task 8.1: Commit
- [ ] Stage: `src/estimation/`, `tests/test_mcmc.py`, `pyproject.toml`
- [ ] Commit on `feat/mcmc-benchmark`
- Acceptance: Clean commit, CI passes

### Task 8.2: Update spec status
- [ ] Set `Status: Implemented` in `.specs/feat-mcmc-benchmark/spec.md`
- [ ] Update `.specs/feat-mcmc-benchmark/.session.md`

---

## Commit Convention

```
feat(mcmc): description

Refs: .specs/feat-mcmc-benchmark/spec.md
```

---

## Notes

- Phase 4 (pilot run) is a gate — do not start batch implementation until the single-series run works with full settings
- Fast config for tests: `MCMCConfig(draws=25, tune=50, chains=2, n_jobs=1)` — chains=2 required for valid R-hat
- Mark slow tests with `@pytest.mark.slow`; configure pytest to skip them by default
- `cores=1` in `pm.sample()` is mandatory when using joblib multiprocessing (prevents nested subprocess spawning)
