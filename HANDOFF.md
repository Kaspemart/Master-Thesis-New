# Session Handoff — Misspecification Analysis

*Written 2026-05-16. Read this before doing anything in a new session.*

---

## What has been completed

### Simulators
- **Base SV model** (`src/simulation/simulator.py`) — fully implemented and tested
- **SV with leverage** (`src/simulation/sv_leverage_simulator.py`) — fully implemented and tested
- 71 tests passing

### Datasets (all generated, locked — do not regenerate)
All 9 datasets exist in `data/`:

| Dataset | N | T | Seed | File |
|---|---|---|---|---|
| Test | 200 | 500/1000/2000 | 42 (all T) | `data/test_T{500,1000,2000}.npz` |
| Train | 90,000 | 500/1000/2000 | 123/456/789 | `data/train_T{500,1000,2000}.npz` |
| Validation | 10,000 | 500/1000/2000 | 321/654/987 | `data/val_T{500,1000,2000}.npz` |

Test sets are **nested**: T=500 and T=1000 are slices of the T=2000 series — same parameter draws, same noise realisations. This makes the sample size comparison a controlled experiment.

### Neural network — simulation study complete
Five architectures compared (MLP, CNN, TCN, Transformer, LSTM) via random hyperparameter search at T=1000. **TCN selected** as best architecture.

- LSTM: included in search but proxy task had to be heavily downgraded (2k series, T truncated to 200) — not comparable to others; ruled out for full training
- Transformer T=2000: skipped (OOM)
- TCN best config: n_channels=32, kernel_size=7, n_blocks=6, dropout=0.0, lr=3e-4, batch_size=256 → **88,771 parameters**
- Same hyperparameters reused for T=500 and T=2000 (isolates effect of series length)
- Trained models saved in `checkpoints/tcn_best_T{500,1000,2000}/best.pt`

**All architectures apply `log(r² + 1e-8)` to raw returns inside `forward()` before any processing.** This is confirmed in the code for all five models.

### MCMC benchmark — complete
- **Primary benchmark: stochvol R package (v3.2.9)**, ASIS sampler — Kastner & Frühwirth-Schnatter (2014)
- Implemented via Python subprocess bridge (`src/estimation/stochvol_runner.py` → `src/estimation/stochvol_runner.R`)
- Run on all 3 test sets (N=200 per T), results in `results/stochvol_T{500,1000,2000}/`
- **NUTS (PyMC)** also complete, results in `results/mcmc_T{500,1000,2000}/` — kept for reference only

### Key results (test set N=200 per T)

| Method | T=500 μ/φ/σ RMSE | T=1000 μ/φ/σ RMSE | T=2000 μ/φ/σ RMSE |
|---|---|---|---|
| TCN | 0.362 / 0.091 / 0.100 | 0.279 / 0.081 / 0.082 | 0.201 / 0.075 / 0.074 |
| stochvol ★ | 0.354 / 0.109 / 0.101 | 0.281 / 0.107 / 0.080 | 0.189 / 0.100 / 0.057 |
| NUTS (ref) | 0.370 / 0.091 / 0.089 | 0.297 / 0.081 / 0.072 | 0.199 / 0.073 / 0.055 |

★ primary benchmark. NUTS used Uniform priors matched exactly to simulation ranges (idealised — a practitioner would not know these). stochvol uses weakly informative priors — realistic operating conditions. TCN vs stochvol is the methodologically fair comparison; NUTS is an idealised upper bound.

### Thesis structure
- Chapter 1–3: Theory — **COMPLETE** (supervisor revisions incorporated)
- Chapter 4: Methodology — **IN PROGRESS** (being written now)
- Chapter 5: Simulation Study — not yet written (all results exist)
- Chapter 6: Misspecification Analysis — not yet written (**results not yet generated — this is next**)
- Chapter 7: Real Data Application — not yet started

### Repository state
- Current branch: `main` (all simulation study work merged)
- Python stack: PyTorch, NumPy, joblib, stochvol R package (v3.2.9)
- Package manager: `uv` — always use `uv run python ...` and `uv add`, never pip

---

## What we are about to do: Misspecification Analysis

This is the **core thesis contribution**. The simulation study showed how TCN and stochvol perform under correct model specification. The misspecification analysis tests what happens when the true data-generating process is more complex than assumed.

### The design

**Generate 200 test series at T=1000 from the SV-with-leverage model.**

Apply two misspecified estimators to them:
1. **Base-SV-trained TCN** (`checkpoints/tcn_best_T1000/best.pt`) — outputs [μ̂, φ̂, σ̂_η], unaware of leverage
2. **Base-SV stochvol** — same priors and config as before, also unaware of leverage

Both estimators assume the base SV model. The true DGP has leverage (ρ ≠ 0). This is the misspecification.

**Measure degradation:**
- Compare misspecified RMSE/bias on [μ, φ, σ_η] against the correctly-specified baselines above
- Degradation = misspecified RMSE − correctly-specified RMSE
- Key question: **does TCN or stochvol degrade more under the same misspecification?**

### Planned analysis
- Overall degradation table (misspecified vs correctly-specified)
- Results binned by ρ value: strong leverage (ρ < −0.5), moderate (−0.5 to −0.1), weak (> −0.1) — shows how degradation scales with the degree of misspecification

---

## Two decisions needed before starting implementation

### Decision 1: ρ range for the leverage test data

**Option A — Full range (−0.95, 0.5):** ρ drawn randomly per series. Covers all asset classes (equities: −0.7 to −0.3, FX: near 0, commodities: up to +0.2). Gives an average picture of misspecification. More general but harder to interpret since mild leverage barely misspecifies.

**Option B — Fixed ρ (e.g. −0.5 or −0.7):** All 200 series have the same leverage. Clean controlled experiment. Loses generality.

**Recommended: Option A (full range) + binned analysis by ρ.** This gives both the average picture and shows how degradation scales with severity — a richer finding for the thesis.

→ **Confirm or override this choice.**

### Decision 2: Correctly-specified leverage estimator

Should we also train a new TCN on leverage model data (4 parameters: μ, φ, σ_η, ρ) to serve as an oracle upper bound?

- **Yes:** Adds context — shows how far each misspecified method is from what's achievable. Requires ~1–2 hours to train.
- **No:** The main question (TCN vs stochvol degradation) is answerable without it. Keeps scope tight.

**Recommended: No, skip for now.** Can always be added later if the supervisor requests it.

→ **Confirm or override this choice.**

---

## Once decisions are confirmed

1. Create branch `feat/misspecification`
2. Write script to generate leverage test data (N=200, T=1000, new seed)
3. Write script to run TCN inference on leverage data
4. Write script to run stochvol on leverage data (reuses existing runner)
5. Write comparison script — compute degradation metrics, bin by ρ, save results
6. Commit and merge

All necessary infrastructure (simulators, stochvol runner, TCN architecture) already exists. This is mostly new scripts, not new model code.

---

## Important implementation notes (do not miss these)

- **SVLeverageParams** is in `src/simulation/sv_params.py` — use this for generating leverage data, not SVParams
- **TCN checkpoint** for T=1000 is at `checkpoints/tcn_best_T1000/best.pt`
- **SVParams.inverse_transform()** must be applied to TCN outputs before computing metrics (network outputs in unconstrained space)
- **stochvol runner** takes plain log-returns and returns [μ, φ, σ_η] — it has no knowledge of ρ and will run exactly as before
- **True params for comparison**: use columns [μ, φ, σ_η] from the leverage dataset (columns 0–2), ignoring ρ (column 3) — we're measuring estimation of the base SV parameters under misspecification
- Results should go in `results/misspec_leverage_T1000/`
- Use a new seed clearly distinct from existing ones (42, 123, 321, 456, 654, 789, 987 are all taken)
