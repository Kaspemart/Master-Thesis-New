# CLAUDE.md — Master's Thesis Project Context

## What this file is
This file provides essential context about an ongoing master's thesis project. Read it at the start of every session before doing anything else. It covers the academic purpose, research design, implementation plan, and current status.

---

## Thesis Overview

**Title (working):** Estimating Parameters of Stochastic Volatility Models Using Neural Networks

**Degree:** Master's thesis (quantitative finance / financial econometrics)

**Core question:** Under what conditions does neural network-based parameter estimation of stochastic volatility models outperform classical benchmark methods, and how does model misspecification affect this comparison?

**Why this matters:** Classical methods for estimating stochastic volatility (SV) models (e.g. MCMC, particle filters) are statistically rigorous but computationally expensive — each new dataset requires running the full estimation procedure from scratch, which can take minutes to hours. Neural networks, once trained, can produce parameter estimates in milliseconds. The thesis investigates whether this speed advantage comes at an acceptable cost in accuracy and reliability, and specifically what happens when the model assumptions are violated.

---

## The Core Problem We Are Solving

Stochastic volatility models describe how the volatility of a financial asset evolves over time as an unobserved (latent) process. The key challenge is that the likelihood function of these models cannot be evaluated in closed form — you cannot just write down a formula and optimize it. This forces the use of approximate or simulation-based methods.

Classical approaches (MCMC, particle filters, method of moments) are accurate but slow and must be re-run for every new dataset. Neural networks offer a fundamentally different approach: train once on simulated data where the true parameters are known, then apply the trained network to new data instantly. This is called **amortized inference**.

The thesis tests whether this approach works, when it fails, and why.

---

## The Models

### Base Model
The discrete-time stochastic volatility model with latent log-volatility:

- Observation equation: `r_t = exp(h_t / 2) * ε_t` where `ε_t ~ N(0,1)`
- State equation: `h_t = μ + φ(h_{t-1} - μ) + σ_η * η_t` where `η_t ~ N(0,1)`

Parameters to estimate:
- `μ` — long-run mean of log-volatility
- `φ` — persistence of volatility (values close to 1 = strong clustering)
- `σ_η` — volatility of volatility

This is the primary model for training and evaluation.

### Extended Model Variants (to be described in thesis, selectively implemented)
In order of increasing complexity:
1. **Base SV** — as above
2. **SV with leverage** — correlation between return shocks and volatility shocks (important for equities; negative correlation = leverage effect)
3. **SV with jumps** — occasional large discontinuous moves in returns and/or volatility
4. **SV with long memory** — slow decay of volatility autocorrelation
5. **SV with regime switching** — discrete shifts in the long-run volatility level

Not all variants will be fully implemented. The thesis will describe all of them theoretically and implement a subset for the empirical work. Which subset is TBD based on complexity and time constraints.

---

## The Approach

### Step 1 — Simulation Study
- Simulate many time series from the SV model with known parameters
- Train a neural network to map from observed return sequences → model parameters
- Evaluate accuracy: how well does the network recover the true parameters?
- Vary sample size (short vs long time series) to see how data length affects performance
- Compare against classical benchmark method (most likely MCMC)
- Find the best-performing neural network architecture and configuration

### Step 2 — Misspecification Analysis
- Train the network on data from one model (e.g. Gaussian errors)
- Test it on data from a different model (e.g. fat-tailed errors, jumps, different parameter distribution)
- Measure how badly performance deteriorates
- Compare degradation of neural network vs classical methods under the same misspecification
- This is the key research contribution — most existing work does not do this systematically

### Step 3 — Real Data Application
- Apply the trained neural network to real financial return data
- Compare parameter estimates against classical method benchmarks
- Assess whether neural estimates are economically reasonable
- Use this to draw conclusions about practical reliability

---

## Classical Benchmark Methods

The neural network results will be compared against at least one classical method. Candidates:
- **MCMC** (most likely primary benchmark — well-established, well-covered in thesis already)
- **Particle filters / SMC** (secondary reference)
- **Efficient Method of Moments** (to be added to thesis)
- **Discretized nonlinear filters** (to be added to thesis — these discretize the latent state space to approximate the likelihood directly)

The benchmark choice is not yet finalized but MCMC is the leading candidate.

---

## Evaluation Metrics

Not yet finalized. Options under consideration:
- **Predictive likelihood** (estimated via particle filter) — general, accounts for full distribution, used by Fičura & Witzany — but lower interpretability
- **MSE / RMSE per parameter** — interpretable but ignores distributional properties
- **Bias and variance of parameter estimates** — useful for diagnosing systematic errors

Important nuance flagged by supervisor: a small error in a parameter like `φ` (autocorrelation of volatility) may have small MSE but large impact on likelihood — so MSE alone can be misleading. Metric choice needs justification.

---

## Neural Network Implementation

### Architecture
- Not yet finalized — finding the best architecture is part of the simulation study
- Leading candidates: convolutional neural networks (CNNs), recurrent networks (RNNs/LSTMs)
- Supervisor guidance: if estimating parameters AND latent states jointly, LSTM may be best (sequential structure fits); if estimating parameters only from the full return series, CNN may outperform LSTM (LSTM may struggle with long-range memory over full series)
- CNNs have shown strong performance and training stability on financial time series in existing literature
- Architecture comparison is a core deliverable of the simulation study

### Input
- Sequence of observed log-returns `r_1, ..., r_T`

### Output
- Estimated parameters: `μ`, `φ`, `σ_η` (and additional parameters for extended models)

### Training data
- Fully simulated: generate thousands of (parameter, return sequence) pairs from the SV model
- True parameters are known by construction — this is the key advantage of simulation-based training
- Target dataset size: at least 100,000 simulated series (Fičura & Witzany used 50,000 which supervisor considers potentially too small)
- Generate the full dataset once and save it permanently — it will be reused across all experiments
- Split into in-sample (training) and out-of-sample (evaluation) portions — never evaluate on training data
- **Parameter ranges should be deliberately wide** — wider than typical literature values, to ensure the trained network generalizes across asset classes (e.g. currencies have lower volatility than equities; a narrow training range produces a less robust estimator)

### Parameter transformations
- Some parameters have bounded domains (e.g. `φ ∈ (-1,1)`, `σ_η > 0`)
- May apply transformations (e.g. log, logit) so the network outputs unconstrained values
- This is standard practice and needs to be handled carefully

### Language / Stack
- Python
- Likely libraries: PyTorch or TensorFlow for neural networks, NumPy/SciPy for simulation, possibly existing MCMC packages for benchmarks (e.g. `stochvol` R package via rpy2, or a Python MCMC implementation)

---

## Thesis Structure (Current)

### Part I — Theoretical (draft exists)
- Chapter 1: Stochastic Volatility — model formulation, latent state, likelihood intractability
- Chapter 2: Classical Estimation Methods — likelihood-based, Bayesian/MCMC, particle methods, practical limitations
  - **TO ADD:** Efficient Method of Moments section
  - **TO ADD:** Discretized nonlinear filter section
  - **TO ADD:** Summary of which classical method performs best according to literature
- Chapter 3: Neural Networks for SV Estimation — existing literature, amortized inference, limitations of current work
  - **TO REVISE:** Strengthen transition from Ch2 to Ch3 with concrete speed comparison numbers
  - **TO REVISE:** Section 3.3 currently repeats 3.2 — needs to be rewritten as a structured problem statement

### Part I additions needed
- **SV model variants review** — describe the progression from base SV to more complex variants (leverage, jumps, long memory, regime switching), including Eraker et al. (2003) jump model
- **Literature review on SV model evolution** — what models are most commonly used today in financial econometrics, and why

### Part II — Empirical (not yet written)
Two confirmed parts:

**Part 1 — Simulation Study**
- Compare multiple NN architectures (CNN, LSTM, potentially others) on simulated data
- Find which architecture performs best and most robustly
- Test how sample size (length of return series) affects performance
- Compare best NN against classical benchmark
- Investigate misspecification: train on one model, test on another

**Part 2 — Application Study**
- Take the best architecture from Part 1
- Apply to real financial return data
- Compare parameter estimates against classical benchmark
- Assess economic plausibility of estimates

---

## Key Decisions Still Open

1. **Which SV model variants to actually implement** (not just describe) — base SV is confirmed; leverage effect is likely; jumps are possible; long memory and regime switching are stretch goals
2. **Which classical benchmark to use** — MCMC is leading candidate, not yet confirmed
3. **Evaluation metric** — predictive likelihood vs MSE vs combined approach
4. **Neural network architecture** — to be determined via simulation study
5. **Real data source** — not yet specified (likely equity index returns, e.g. S&P 500 or similar)

---

## Important Constraints and Risks

- **Citation accuracy:** Some citations in the current draft have not been manually verified against the source papers. Before finalizing any claim, verify it against the actual paper.
- **Scope creep:** The supervisor suggested many extensions (long memory, regime switching, multiple model variants). Not all of these need to be implemented. Stay focused on doing fewer things well rather than many things superficially.
- **Misspecification is the contribution:** The novelty of this thesis is the systematic misspecification analysis. Everything else (simulation study, benchmark comparison) is scaffolding. Don't lose sight of this.
- **Parameter identifiability:** There is a known issue in SV models where different parameter combinations can produce similar return dynamics. This affects how well any method (neural or classical) can recover parameters. This should be acknowledged and ideally analyzed.

---

## Current Status

- Part I theoretical draft: complete first draft, needs revisions per above
- Supervisor meeting: completed, notes incorporated into this file
- Implementation: not yet started
- Repository: created, empty

---

*Last updated: after first supervisor meeting and first draft review*
