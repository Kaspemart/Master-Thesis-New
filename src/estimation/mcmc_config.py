from dataclasses import dataclass


@dataclass
class MCMCConfig:
    """
    Configuration for the MCMC benchmark sampler.

    Attributes:
        draws:         Number of posterior samples to draw per chain.
        tune:          Number of tuning (warm-up) steps per chain.
        chains:        Number of MCMC chains (minimum 2 for R-hat diagnostics).
        target_accept: Target acceptance rate for NUTS step-size adaptation.
        n_jobs:        Number of parallel worker processes for batch runs.
        random_seed:   Base seed; per-series seed = random_seed + series_index.
    """
    draws: int = 1000
    tune: int = 1000
    chains: int = 2
    target_accept: float = 0.9
    n_jobs: int = 4
    random_seed: int = 42
