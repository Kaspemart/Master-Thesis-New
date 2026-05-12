"""
PyTorch Dataset for SV model training and evaluation.

Returns (returns_tensor, params_transformed_tensor) pairs.
Parameters are transformed to unconstrained space before being returned
so the network can be trained with a plain MSE loss on unbounded outputs.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.simulation.simulator import load_dataset
from src.simulation.sv_params import SVParams


class SVDataset(Dataset):
    """
    Dataset wrapping a saved .npz simulation file.

    Each item is (returns, params_t) where:
        returns:  float32 tensor of shape (T,)   — observed log-returns
        params_t: float32 tensor of shape (3,)   — [μ, logit(φ), log(σ_η)]

    Parameters are transformed to unconstrained space via SVParams.transform()
    so the network outputs unconstrained values trained with MSE loss.
    Inverse-transform at inference time to recover natural-space estimates.

    Args:
        path:   Path to a .npz file produced by save_dataset().
        config: SVParams instance used to define the transform. Defaults to
                SVParams() (standard wide ranges).
    """

    def __init__(self, path: str, config: SVParams | None = None):
        result = load_dataset(path)
        if config is None:
            config = SVParams()

        self.returns = torch.from_numpy(result.returns)          # (N, T), float32
        params_t = config.transform(result.params.astype(np.float64)).astype(np.float32)
        self.params_t = torch.from_numpy(params_t)               # (N, 3), float32

    def __len__(self) -> int:
        return self.returns.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.returns[idx], self.params_t[idx]
