# CLAUDE.md — Master's Thesis Project Context

## What this file is
This file provides essential context about an ongoing master's thesis project. Read it at the start of every session before doing anything else. It covers the academic purpose, research design, implementation plan, and current status.

---

## Thesis Overview

**Title (working):** Estimating Parameters of Stochastic Volatility Models Using Neural Networks

**Degree:** Master's thesis (quantitative finance / financial econometrics)

**Core question:** Under what conditions does neural network-based parameter estimation of stochastic volatility models outperform classical benchmark methods, and how does model misspecification affect this comparison?

**Why this matters:** Classical methods for estimating stochastic volatility (SV) models (e.g. MCMC, particle filters) are statistically rigorous but computationally expensive — each new dataset requires running the full estimation procedure from scratch, which can take minutes to hours. Neural networks, once trained, can produce parameter estimates in milliseconds. The thesis investigates whether this enormous speed advantage comes at an acceptable cost in accuracy and robustness, and specifically under what conditions the trade-off is favourable.

---

## The Core Problem We Are Solving

Stochastic volatility models describe how the volatility of a financial asset evolves over time as an unobserved (latent) process. The key challenge is that the likelihood function of these models cannot be evaluated in closed form — you cannot just write down a formula and optimize it. This forces the use of approximate or simulation-based methods.

Classical approaches (MCMC, particle filters, method of moments) are accurate but slow and must be re-run for every new dataset. Neural networks offer a fundamentally different approach: train once on simulated data where the true parameters are known, then apply the trained network to new data instantly. This is called **amortized inference**.

The thesis tests whether this approach works, when it fails, and why.

---

## The Models

### Base Model — PRIMARY IMPLEMENTATION TARGET
The discrete-time stochastic volatility model with latent log-volatility:

- Observation equation: `r_t = exp(h_t / 2) * ε_t` where `ε_t ~ N(0,1)`
- State equation: `h_t = μ + φ(h_{t-1} - μ) + σ_η * η_t` where `η_t ~ N(0,1)`

Parameters to estimate:
- `μ` — long-run mean of log-volatility
- `φ` — persistence of volatility (values close to 1 = strong clustering)
- `σ_η` — volatility of volatility

### SV with Leverage — PRIMARY IMPLEMENTATION TARGET
Extends the base model by introducing a correlation `ρ` between the return shock `ε_t` and the volatility shock `η_t`. For equities, `ρ` is typically negative (the leverage effect: negative returns tend to increase future volatility). This adds one parameter to estimate: `ρ ∈ (−1, 1)`.

### Misspecification Scenarios (described theoretically, NOT fully implemented)
The following variants are used only as out-of-distribution test cases in the misspecification analysis:
- **SV with jumps** — occasional large discontinuous moves in returns and/or volatility
- **SV with long memory** — slow decay of volatility autocorrelation
- **SV with regime switching** — discrete shifts in the long-run volatility level

**Do not implement these in full.** They exist only to generate misspecified test data.

---

## The Research Contribution

The core contribution is a **systematic misspecification analysis**. Most existing work evaluates neural network estimators only under correct model specification — where training and test data come from the same model. This thesis goes further in two specific ways:

### 1. Misspecification Analysis
- Train a neural network on data from a simpler model (e.g. base SV with Gaussian errors)
- Test it on data from a more complex model (e.g. fat tails, jumps, leverage)
- Measure how badly the parameter estimates deteriorate
- Compare this deterioration against how MCMC handles the same misspecification

Key nuance: a small error in a parameter like `φ` (volatility persistence) may look minor in terms of MSE but can have a large impact on the likelihood of the model. This means the choice of evaluation metric matters and needs careful justification.

### 2. Sample Size Analysis
- Investigate how the length of the return series affects performance
- The expectation is that neural networks perform better with more data, while classical probabilistic methods may hold up better with shorter series
- The length of the time series is itself a variable that determines which method wins — this is an interesting and practically relevant finding

---

## The Approach

### Part 1 — Simulation Study
- Generate at least 100,000 simulated return series from the SV model with known parameters
- Split into training (in-sample) and evaluation (out-of-sample) portions — **never evaluate on training data**
- Compare multiple neural network architectures (CNN, LSTM) to find the best performing one
- Test how the length of the return series affects performance (T = 500, 1,000, 2,000)
- Compare the best neural network against MCMC benchmark
- Once the best architecture is found, stress test it under misspecification

### Part 2 — Application Study
- Take the best architecture from Part 1
- Apply to real financial return data
- Compare parameter estimates against MCMC benchmark
- Assess economic plausibility of estimates
- Draw conclusions about practical reliability

---

## Classical Benchmark

**Bayesian MCMC is the confirmed benchmark.** It builds on two key references:
- **Kim, Shephard & Chib (1998)** — mixture of normals approximation within MCMC framework
- **Kastner & Frühwirth-Schnatter (2014)** — ASIS interweaving strategy for improved sampling efficiency

MCMC has established itself as the dominant and most reliable approach in academic research, offering the best combination of statistical accuracy, flexibility, and ability to handle complex model specifications. It is therefore the natural benchmark for comparison with neural network methods.

---

## Evaluation Metrics

**Not yet finalised.** The main candidates are:
- **Predictive likelihood** (estimated via particle filter) — general, accounts for full distribution, used by Fičura & Witzany — but lower interpretability
- **MSE / RMSE per parameter** — more interpretable but can be misleading: a small parameter error can still have a large impact on the likelihood
- **Bias and variance of parameter estimates** — useful for diagnosing systematic errors

**Key constraint:** MSE alone is not sufficient. The metric choice must explicitly account for the fact that small parameter errors can have large likelihood impacts. The choice needs explicit justification in the thesis.

---

## Neural Network Implementation

### Architecture
- Not yet finalised — finding the best architecture is part of the simulation study
- Leading candidates: **CNN** and **LSTM**
- If estimating parameters only from the full return series: CNN likely better (LSTM may struggle with long-range memory over full series)
- If estimating parameters AND latent states jointly: LSTM may be better (sequential structure fits naturally)
- Architecture comparison is a core deliverable of the simulation study

### Input
- Sequence of observed log-returns `r_1, ..., r_T`

### Output
- Estimated parameters: `μ`, `φ`, `σ_η` (base model) plus `ρ` for the leverage extension

### Training Data
- Target: **at least 100,000 simulated series** (Fičura & Witzany used 50,000 which supervisor considers too small)
- Generate once and save permanently — reused across all experiments (`.npz` format)
- Split into training and held-out test portions
- **Parameter ranges must be deliberately wide** — wider than typical literature values, to generalise across asset classes (currencies have lower volatility than equities; narrow ranges produce less robust estimators)
- **Baseline series length: 1,000 observations** (approximately 4 years of daily data)
- Also test T = 500 (short) and T = 2,000 (long) as part of the sample size analysis

### Parameter Transformations
Apply transformations so the network always outputs unconstrained values; transform back at inference time:
- `φ ∈ (−1, 1)` — apply **logit** transformation
- `σ_η > 0` — apply **log** transformation
- `ρ ∈ (−1, 1)` — apply **arctanh** transformation (not logit — logit requires input in (0,1) but ρ can be negative); inverse is tanh; training range `(−0.95, 0.5)` — covers all realistic asset classes (equities: −0.7 to −0.3, FX: near 0, commodities: up to +0.2) without including near-singular extremes that are economically implausible

### Correlated Noise Implementation (Leverage Model)
The leverage effect is implemented via **Cholesky decomposition** of the 2×2 correlation matrix:
- Draw independent `z1, z2 ~ N(0,1)`
- Set `ε_t = z1` (return shock)
- Set `η_t = ρ·z1 + sqrt(1−ρ²)·z2` (volatility shock, correlated with ε_t)
- This is mathematically equivalent to drawing jointly from N(0, Σ) where Σ = [[1,ρ],[ρ,1]]
- **Note for methodology chapter:** This Cholesky decomposition approach must be described explicitly when writing up the leverage model specification.

### Language / Stack
- **Python with PyTorch** (confirmed — not TensorFlow)
- NumPy / SciPy for simulation
- MCMC benchmark: **PyMC** (pure Python, NUTS sampler) — cite Kim et al. (1998) and Kastner & Frühwirth-Schnatter (2014) as methodological references, PyMC as the implementation tool
- Data storage: NumPy `.npz` files

---

## Thesis Structure

### Part I — Theoretical (COMPLETE — supervisor revisions incorporated)

**Chapter 1: Stochastic Volatility**
- 1.1 SV Models (continuous and discrete time formulations)
- 1.2 Latent Volatility and State-Space Representation
- 1.3 Likelihood Intractability
- 1.4 Extensions of the Base Model (leverage, jumps, long memory, regime switching)
- 1.5 Current Use Practice

**Chapter 2: Classical Estimation Methods**
- 2.1 Likelihood-Based Methods
- 2.2 Bayesian Framework
- 2.3 Simulation-Based Methods (MCMC, PMCMC, SMC², EMM, Nonlinear Filtering)
- 2.4 Practical Limitations (including summary noting MCMC as best benchmark)

**Chapter 3: Neural Networks for SV Estimation**
- 3.1 Neural Networks in Volatility Modelling
- 3.2 Neural Estimation of SV Parameters
- 3.3 Limitations of Existing Approaches

**Remaining citation work:** Citations added in recent revisions have not all been manually verified against the source papers — must verify before final submission.

### Part II — Empirical (NOT YET WRITTEN)
- Chapter 4: Simulation Study
- Chapter 5: Misspecification Analysis
- Chapter 6: Real Data Application

---

## Key Decisions Still Open

1. **Evaluation metric** — predictive likelihood vs MSE vs combined approach — needs explicit justification
2. **Neural network architecture** — to be determined via simulation study (CNN vs LSTM)
3. **Real data source** — not yet specified (likely equity index returns, e.g. S&P 500 or similar)
4. **Whether to estimate parameters only or parameters and latent states jointly** — affects architecture choice

**Resolved decisions (no longer open):**
- Benchmark method: **MCMC confirmed** — implemented via **PyMC** (Python, NUTS sampler); Kim et al. (1998) and Kastner & Frühwirth-Schnatter (2014) cited as methodological references
- Model variants to implement: **base SV and SV with leverage** — all others are misspecification scenarios only
- Training dataset size: **≥ 100,000 series**
- Series lengths to test: **T = 500, 1,000, 2,000**
- Stack: **Python + PyTorch + NumPy + .npz storage**

---

## Important Constraints and Risks

- **Misspecification is the contribution:** The novelty of this thesis is the systematic misspecification analysis. Everything else (simulation study, benchmark comparison) is scaffolding. Do not lose sight of this.
- **Scope creep:** Jumps, long memory, and regime switching are misspecification scenarios only — do not implement them fully.
- **Citation accuracy:** Citations added in recent revisions have not all been manually verified. Must verify before final submission.
- **Parameter identifiability:** Different parameter combinations can produce similar return dynamics. This affects both neural and classical methods and should be acknowledged in the thesis.
- **Metric choice:** MSE alone is insufficient. The metric must account for the fact that small parameter errors can have large likelihood impacts.

---

## Current Status

- Part I theoretical draft: **COMPLETE** — all supervisor revisions incorporated
- Part I citations: need manual verification against source papers
- Supervisor meeting: completed, all notes incorporated
- Implementation: **NOT YET STARTED** — currently in specification/planning phase
- Repository: created, environment set up (Python + PyTorch via uv)

---

## Notes for Claude

**Working principles — apply these in every session:**

- **Question everything.** Do not accept assumptions at face value. If a design choice, parameter value, or architectural decision seems arbitrary or underspecified, raise it explicitly before proceeding.
- **Be precise and thorough.** This is a master's thesis — correctness matters more than speed. Prefer doing fewer things correctly over many things approximately.
- **Go step by step on larger tasks.** Break implementation into small, verifiable increments. Confirm each step works before moving to the next. Do not skip ahead.
- **Update this file as decisions are made.** Every time an open question in "Key Decisions Still Open" is resolved, update that section immediately so future sessions start with accurate context.

---

*Last updated: 2026-04-19 — research contribution clarified, model variants finalised, benchmark confirmed, series lengths specified*
