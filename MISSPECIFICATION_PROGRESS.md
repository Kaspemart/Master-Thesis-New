# Misspecification Analysis — Progress Report

*Branch: `feat/misspecification-analysis`. Last updated 2026-07-26. Read this to resume.*

This document records the misspecification-analysis build (Chapter 6). It is the
single place to recover full context. Supervisor answers drive the design; see
also `~/.claude/.../memory/project_misspecification_design.md`.

---

## 1. The design (supervisor-approved)

Data generated from **three** misspecified DGPs, each carrying a full 2×2
comparison. Terminology follows the supervisor: **ASV** = asymmetric SV (leverage).

**Three scenarios (Direction 1 — PRIMARY):** train NN on base SV, apply to
- (i) **ASV** — leverage (μ, φ, σ_η, ρ)
- (ii) **SV-t** — Student-t errors (μ, φ, σ_η, ν)
- (iii) **ASV-t** — leverage + t (μ, φ, σ_η, ρ, ν)

**2×2 per scenario:**
- (a) NN estimating **base SV** params on the complex data — misspecified NN
- (b) MCMC (stochvol) estimating **base SV** params — misspecified MCMC
- (c) NN estimating the **correct** model params — correctly-specified NN
- (d) MCMC estimating the **correct** model params — correctly-specified MCMC

**Direction 2 (STRETCH, "ideal" per supervisor):** train NN on ASV-t, apply to
simpler DGPs. Build infrastructure general enough that this is cheap; do not
promise it until Direction 1 is done.

**Metrics (two tiers):**
- Tier 1 (descriptive): RMSE + bias per parameter. Shows *how* misspecification
  acts. For cells (a)/(b), measured vs the base-SV params of the ASV process —
  this is contamination/distortion, not error in the usual sense.
- Tier 2 (decisive): **predictive log-likelihood via particle filter**, in-sample
  and out-of-sample. Answers whether NN or MCMC is more robust. RMSE describes,
  likelihood decides. (a)vs(b) = headline; (a)vs(c),(b)vs(d) = cost of wrong
  model; (c)vs(d) = baseline capability (separates "more robust" from "just
  better at this model class").

**Supervisor's hypothesis (Chapter 6 hook):** NN may be robust to leverage (ASV)
but *less* robust to fat tails (SV-t), because non-normality biases σ_η and a
heuristic-mapping NN may be more sensitive than likelihood-based MCMC.

**Locked decisions:** T=1000 only (scope choice); out-of-sample window T_oos=1000
(one 2000-length path split 1000/1000, latent state continuous); test seed 999;
Chapter 6 uses **stochvol only** as MCMC (not NUTS); ρ prior for stochvol
leverage = symmetric weakly-informative centred at 0 (check post-hoc); ν
simulation range (3,40) sampled uniformly on 1/ν; ν NN transform log(ν−2);
stochvol ν prior sv_exponential(0.1); report ν accuracy split by low/high ν.

---

## 2. Conventions verified against stochvol (each would have broken the chapter)

All three caught by checking against `svsim`, not assuming:

1. **t-standardisation.** stochvol standardises t-errors to unit variance, so
   exp(h_t) is the conditional variance regardless of ν. Std factor
   c=√((ν−2)/ν) on a raw t_ν. (Empirical Var≈1.0 at ν=4,8,15.)
2. **Leverage timing.** stochvol uses the FORWARD convention:
   corr(ε_t, η_{t+1})=ρ. Our simulator originally used contemporaneous — a
   genuine mismatch — now switched. Sign confirmed identical (negative ρ =
   leverage).
3. **Leverage-t coupling (ASV-t).** Leverage couples to the NORMAL COMPONENT
   z_t of the scale mixture (corr(z_t, ·)=−0.69≈−0.7), not the full residual
   (−0.64). Filter reflects this via a tau-posterior draw.

---

## 3. Particle filter — COMPLETE and validated

`src/evaluation/particle_filter.py` — `sv_log_likelihood(returns, mu, phi,
sigma_eta, nu=inf, rho=0.0, n_particles=10000, seed=None)`. Bootstrap PF,
systematic resampling every step, stable log-mean-exp increments. Use 10k–20k
particles. Handles all four models via (nu, rho):

| Model | (nu, rho) | Validation result |
|---|---|---|
| Base SV | (inf, 0) | logL stabilises w/ particles; φ,σ peaks at truth |
| SV-t | (finite, 0) | **ν peak exact**; normal-limit matches Gaussian |
| ASV | (inf, ≠0) | ρ peak ~truth; leverage improves logL +33.5 |
| ASV-t | (finite, ≠0) | ν peak exact; ρ~truth; leverage +14.8 |

Validation scripts (all pass): `scripts/validate_particle_filter{,_t,_leverage,_asvt}.py`.
Design detail: weight by marginal std-t density; for ASV-t propagation draw
tau|eps ~ InvGamma((ν+1)/2, (ν−2)/2 + eps²/2), z=eps/√tau, η_{t+1}=ρz+√(1−ρ²)ξ.

---

## 4. Simulator changes

`src/simulation/simulator.py`:
- `simulate_sv_leverage` **switched to forward convention** (was contemporaneous).
  Verified: forward corr −0.75 matches svsim; contemporaneous 0; ρ=0 → base.
  η_0 = z2_0 (unit variance, no ε_{−1}). Tests updated; all 100 pass.
- **SV-t / ASV-t simulators DONE and validated** — `simulate_sv_t` (params
  (N,4) [μ,φ,σ_η,ν]), `simulate_asv_t` (params (N,5) [μ,φ,σ_η,ρ,ν]), shared
  `_simulate_t_core`. Standardised-t (tau ~ InvGamma(ν/2,(ν−2)/2), E=1), forward
  leverage on normal component. Validated (`scripts/validate_t_simulators.py`):
  Var≈1 at all ν, kurtosis tracks t_ν theory, and the validated PF run on OUR
  sim data recovers ν and ρ exactly (closes the loop: sim == svsim). ν drawn on
  1/ν over (3,40) via `draw_nu`. New param classes `SVtParams`, `ASVtParams` in
  `sv_params.py` with transforms (ν → log(ν−2), ρ → arctanh).

### Test data — GENERATED (2026-07-26)
`scripts/generate_misspec_test_data.py` → `data/test_misspec_{asv,svt,asvt}.npz`.
N=200, path length 2000 (T_estim=1000 + T_oos=1000), seed 999 for all three so
base params (μ,φ,σ_η) are IDENTICAL draws across scenarios (controlled property —
verified). Each file: returns (200,2000), params (200,4 or 5), latent_h, T_estim, T_oos.

### Correct-model NN architecture — CONFIRMED plan (before training)
Cells (c) use the SAME validated TCN (n_channels=32, kernel_size=7, n_blocks=6,
dropout=0.0, same input transform), changed ONLY in the output layer:
ASV→4 outputs (ρ, arctanh), SV-t→4 (ν, log(ν−2)), ASV-t→5 (both). Keeps (c)vs(a)
clean — same architecture, only the target model differs.

---

## 5. Files created / modified on this branch

Created: `src/evaluation/__init__.py`, `src/evaluation/particle_filter.py`,
`scripts/validate_particle_filter*.py` (4 files), this report.
Modified: `src/simulation/simulator.py` (leverage forward convention),
`tests/test_sv_leverage.py` (forward-convention test), `CLAUDE.md`.
**Not committed yet** (per user's global rule — commit only when asked).

---

## 5b. EARLY RESULTS — Cell (a), base-SV TCN on misspecified data (2026-07-26)

RMSE / bias on base params (estimation window T=1000). Baseline (correctly-
specified base-SV TCN, Ch5): μ 0.279, φ 0.081, σ_η 0.082.

| Scenario | μ RMSE | φ RMSE | σ_η RMSE | σ_η bias |
|---|---|---|---|---|
| ASV (leverage) | 0.309 | 0.079 | 0.085 | −0.031 |
| SV-t (fat tails) | 0.447 | 0.120 | **0.275** | **+0.204** |
| ASV-t (both) | 0.467 | 0.113 | 0.259 | +0.190 |

**Cell (b), base-SV stochvol on misspecified data (RMSE):**

| Scenario | μ / φ / σ_η (TCN) | μ / φ / σ_η (stochvol) |
|---|---|---|
| ASV | 0.309 / 0.079 / 0.085 | 0.340 / 0.109 / 0.071 |
| SV-t | 0.447 / 0.120 / 0.275 | 0.461 / 0.320 / 0.429 |
| ASV-t | 0.467 / 0.113 / 0.259 | 0.464 / 0.281 / 0.397 |

**Tier-1 finding (RMSE/bias):** Both methods robust to leverage; both badly hurt
by fat tails with damage in σ_η. BUT the **TCN is MORE robust to fat tails than
stochvol** (σ_η 0.275 vs 0.429; φ 0.120 vs 0.320 under SV-t). This runs COUNTER
to the supervisor's hypothesis that the NN would be *less* robust to non-normality.
Likely because stochvol's Gaussian likelihood reacts strongly to individual
fat-tail outliers (inflating σ_η), while the TCN's learned mapping is less
outlier-sensitive. Flag to supervisor. NOTE: RMSE describes, predictive LL
DECIDES — do not lock the headline until predictive LL is in.
stochvol also shows elevated R-hat on φ/σ_η under fat tails (more non-convergence
on misspecified data — itself worth noting).
Results: `results/misspec/{scenario}_{tcn,stochvol}.npz`.
Runner: `scripts/run_misspec_cells_ab.py`. Predictive LL: `run_misspec_predictive_ll.py`
(running). Filter has `t_split` for in/out-of-sample LL from one pass.

## 5c. DECISIVE RESULT — Cell (a) vs (b) predictive log-likelihood (2026-07-26)

Out-of-sample predictive LL (base-SV model, estimated params, filter t_split=1000,
10k particles, 3 matched seeds). Higher = better. Paired per-series t-tests.

| Scenario | TCN OOS | stochvol OOS | diff (TCN−sv) | t-stat | % TCN better |
|---|---|---|---|---|---|
| ASV | 1002.02 | 1002.35 | −0.33 | −2.99 | 44% |
| SV-t | 1078.72 | 1080.99 | −2.27 | −6.63 | 34% |
| ASV-t | 1106.38 | 1108.54 | −2.15 | −7.91 | 30% |

**HEADLINE (central chapter question, a-vs-b): stochvol produces the more
predictive model under misspecification — large & significant for fat tails,
negligible (near-tie) for leverage.** This REVERSES the Tier-1 RMSE story (where
TCN looked more robust to fat tails). RMSE and predictive LL diverge; LL decides.

Interpretation (as interpretation, not asserted): under fat tails, inflating σ_η
is the likelihood-optimal Gaussian response to large moves, so stochvol's "worse"
σ_η (RMSE 0.43 vs TCN 0.28) is adaptive for prediction; the TCN, trained to
minimise parameter MSE, is anchored to parameter recovery (wrong objective under
misspecification) so distorts σ_η less but predicts worse. Vindicates supervisor's
predictive-LL metric AND his original hypothesis (MCMC more robust to non-normality),
visible only through the right metric. Results: `results/misspec/{scen}_predictive_ll.npz`.

## 5d. Cell (c) — correctly-specified TCN (2026-07-28)

Correct-model TCNs trained (val loss asv 0.178, svt 0.213, asvt 0.219),
`checkpoints/{model}_correct_T1000/`. Estimates on test data:

| Scenario | μ | φ | σ_η | ρ | ν low<10 / high≥10 |
|---|---|---|---|---|---|
| ASV | 0.341 | 0.079 | 0.078 | 0.407 | — |
| SV-t | 0.363 | 0.090 | 0.128 | — | 2.28 / 10.94 |
| ASV-t | 0.343 | 0.089 | 0.119 | 0.407 | 2.24 / 8.57 |

Findings: (1) correct model roughly halves σ_η error under fat tails (0.128 vs
0.275 misspecified cell a) — the cost of misspecification made concrete;
(2) ν well-estimated when low (fat tails identifiable, RMSE 2.28) but badly when
high (near-Gaussian, weakly identified, RMSE 10.9, bias −7.4 shrinking toward
centre) — exactly the supervisor's weak-identification prediction, split as he
requested; (3) ρ hard for the NN (RMSE 0.41 on range 1.45).
Runners: `run_misspec_cell_c.py`. Results: `results/misspec/{scen}_tcn_correct.npz`.

## 5e. Cell (d) — correctly-specified stochvol (DONE 2026-07-28)

| Scenario | μ | φ | σ_η | ρ | ν | max frac R-hat>1.1 |
|---|---|---|---|---|---|---|
| ASV | 0.359 | 0.121 | 0.075 | 0.150 | — | 0.37 |
| SV-t | 0.362 | 0.165 | 0.183 | — | 6.62 | 0.56 |
| ASV-t | 0.421 | 0.159 | 0.167 | 0.174 | 6.72 | 0.60 |

Findings (c-vs-d, both correct): (1) **stochvol recovers ρ far better** (0.15 vs
NN 0.41) — leverage well-identified for MCMC, poorly for NN; reversal of the
fat-tails story. (2) ν: NN better at low ν, both poor at high ν. (3) σ_η: NN
better under fat tails. (4) **MAJOR: correct-model MCMC has severe convergence
problems — 34-60% frac R-hat>1.1** on the general leverage/t sampler (vs ~10%
base fast-SV). Correct MCMC is slow + unreliable; NN always returns instantly.
Runner: `run_misspec_cell_d.py`, `stochvol_runner_correct.R` (rho Beta(4,4),
nu Exp(0.1)). Results: `results/misspec/{scen}_stochvol_correct.npz`.

## 5f. FULL 2×2 — OOS predictive log-likelihood (DONE 2026-07-28) — EXPERIMENTS COMPLETE

| Scenario | (a) misspec NN | (b) misspec MCMC | (c) correct NN | (d) correct MCMC |
|---|---|---|---|---|
| ASV | 1002.02 | 1002.35 | 1004.71 | 1015.64 |
| SV-t | 1078.72 | 1080.99 | 1082.86 | 1082.74 |
| ASV-t | 1106.38 | 1108.54 | 1111.86 | 1120.57 |

Contrasts (paired t): c−d = ASV −10.9 (t−10), SV-t **+0.12 (t0.6 = TIED)**, ASV-t
−8.7 (t−9). d−b = +13.3 / +1.75 / +12.0. c−a = +2.7 / +4.1 / +5.5 (all sig).

**COMPLETE CHAPTER 6 STORY (3 findings):**
1. Two-metric divergence (a-vs-b, spine): RMSE says NN more robust to fat tails,
   predictive LL says MCMC is. Vindicates supervisor's metric + hypothesis.
2. NN's fat-tails weakness is a MISSPECIFICATION ARTIFACT — correctly specified,
   NN & MCMC TIE on fat-tail prediction (c−d +0.12, ns).
3. NN's FUNDAMENTAL weakness is LEVERAGE — even correct it predicts worse than
   MCMC (c−d −9 to −11) because it can't recover ρ (0.41 vs 0.15). MCMC gains big
   from correct model under leverage (+13), little under fat tails (+1.75).
Counterweight favouring NN: correct-model MCMC 34-60% non-convergence + slow;
NN always returns instantly.
Results: `results/misspec/{scen}_predictive_ll_correct.npz`.
Runner: `run_misspec_predictive_ll_correct.py`.

**ALL CHAPTER 6 EXPERIMENTS COMPLETE. Remaining = writing + optional Direction 2.**

## 5g. VERIFICATION PASS (2026-07-28) — all clean
Full double-check performed, no errors: (1) test data forward-leverage convention
confirmed per-series (corr with true ρ = +0.98; pooled measure is a dilution
artifact, not a bug); (2) all 12 result files present/finite/right shape;
(3) reported RMSE match recomputation exactly; (4) 2×2 predictive-LL matches
source files; (5) 100 tests pass; (6) t_split splits sum to total (machine prec);
(7) all four filter validations reproduce identical validated results (later
edits — t_split/leverage/asvt — caused no regression).
Consolidated results: `experiments/chapter6_consolidated_results.json` +
`experiments/chapter6_results.md` (assembled from source npz by
`scripts/assemble_chapter6_results.py`).

## 6. What remains

1. ~~SV-t / ASV-t simulators~~ **DONE + validated** (§4).
2. ~~Generate test data~~ **DONE** — `data/test_misspec_{asv,svt,asvt}.npz` (§4).
3. ~~Cells (a),(b)~~ **DONE** — see §5b/§5c. Headline secured (a-vs-b).
4. Cells (c),(d) IN PROGRESS:
   - Training data **generated**: `data/{train,val}_{asv,svt,asvt}_T1000.npz` (90k/10k, seeds 1001-1006).
   - Cell (c) correct-model TCNs **TRAINING NOW** (chained bg run, ~3hr):
     `scripts/train_misspec_models.py`, checkpoints `checkpoints/{model}_correct_T1000/`.
     TCN made output-configurable (`n_outputs`); asv/svt=4, asvt=5. Smoke-tested
     all three (transforms finite, shapes right, ~88.8k params).
   - Cell (d) STILL TODO: stochvol runners for t / leverage / leverage+t
     (svtsample/svlsample/svtlsample). Then apply to each scenario.
5. Compute RMSE + bias + in/out predictive LL for cells (c),(d) via filters
   (correct-model filter: use estimated ρ/ν). Gives (a)vs(c), (b)vs(d) cost-of-
   -misspecification and (c)vs(d) baseline.
6. Assemble 2×2 per scenario; write Chapter 6 (spine: RMSE vs predictive-LL divergence).

**Checkpoints:** (a) after simulators — DONE, validated. (b) before the three
~1hr NN training runs (+ their training-data generation) — awaiting user go-ahead.

---

## 7. Methodology text still to fix (thesis document, not in repo)

- §4.2.1 leverage equations: now corrected to forward convention by user. ✅
- §4.8.2: predictive log-likelihood added by user. ✅ ("Section X" PF reference
  still a placeholder — fill once PF methodology subsection written.)
- §4.9 Misspecification Design: **WRITTEN and reviewed sound** — covers all three
  scenarios (ASV/SV-t/ASV-t), the 2×2 design, data/eval, predictive-LL metric,
  Direction 2 noted as unpursued. SV-t standardisation eq verified correct.
  Minor tense fix ("the second providing"). 
- New subsection STILL NEEDED: particle filter methodology (bootstrap algorithm,
  particle count 10k–20k, systematic resampling, and the three verified
  conventions — incl. the ASV-t leverage-to-normal-component coupling, which is
  not stated anywhere in the thesis text yet). §4.8.2 "Section X" points here.
- Chapter 4.7: add stochvol runners for t / leverage priors (ν ~ Exp(0.1);
  ρ ~ symmetric weakly-informative).
