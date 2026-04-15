
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
  simulation/   # SV model simulation (data generation)
  models/       # Neural network architectures
  estimation/   # Classical benchmark methods (MCMC etc.)
  evaluation/   # Metrics and visualisation
notebooks/      # Exploratory analysis and experiments
data/           # Real financial data (not committed)
tests/          # Unit tests
.claude/
  commands/     # Claude Code slash commands (SDD workflow)
.specs/         # SDD feature specifications
```

## Thesis Structure

- **Part I (theoretical):** SV model formulation, classical estimation methods, neural networks for SV estimation
- **Part II (empirical):** Simulation study, misspecification analysis, real data application
