
# Estimating Parameters of Stochastic Volatility Models Using Neural Networks

Master's thesis — quantitative finance / financial econometrics.

## Research Question

Under what conditions does neural network-based parameter estimation of stochastic volatility models outperform classical benchmark methods, and how does model misspecification affect this comparison?

## Setup

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
uv venv
source .venv/bin/activate
uv sync
```

For development dependencies (Jupyter, pytest):

```bash
uv sync --extra dev
```

## Project Structure

```
src/
  simulation/   # SV model simulators (base SV + SV-with-leverage)
  models/       # Neural network architectures (MLP, CNN, TCN, Transformer, LSTM)
  estimation/   # Classical benchmark (MCMC via PyMC NUTS)
data/           # Simulated datasets — train/val/test × T=500/1000/2000 (.npz)
experiments/    # Hparam search logs and comparison summaries
results/        # Test set predictions and evaluation summaries per architecture/T
checkpoints/    # Best model weights (.pt files)
scripts/        # Training, evaluation, and MCMC runner scripts
tests/          # Unit tests (71 passing)
```

## Current Status

**Simulation study: COMPLETE**

| Method | T=500 μ RMSE | T=500 φ RMSE | T=1000 μ RMSE | T=1000 φ RMSE | T=2000 μ RMSE | T=2000 φ RMSE |
|---|---|---|---|---|---|---|
| TCN | 0.362 | 0.091 | 0.279 | 0.081 | 0.201 | 0.075 |
| Transformer | 0.354 | 0.096 | 0.282 | 0.081 | — | — |
| MCMC (NUTS) | 0.370 | 0.091 | 0.297 | 0.081 | 0.199 | 0.073 |

NNs match or beat MCMC on μ and φ; MCMC retains advantage on σ_η. TCN is the best practical architecture (88k params, fast, competitive across all series lengths).

**Pending:** misspecification analysis (core thesis contribution), real data application.

## Thesis Structure

- **Part I (theoretical):** SV model formulation, classical estimation methods, neural networks for SV estimation — COMPLETE
- **Part II (empirical):** Simulation study (COMPLETE), misspecification analysis, real data application
