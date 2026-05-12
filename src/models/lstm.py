"""
LSTM neural network for SV parameter estimation.

Input:  return series of shape (batch, T), transformed internally to log(r² + 1e-8)
Output: unconstrained parameter estimates of shape (batch, 3)
        columns = [μ, logit(φ), log(σ_η)]

Architecture: stacked LSTM layers processing the series sequentially,
final hidden state fed into an MLP head.
"""

import torch
import torch.nn as nn


class SVLSTMNet(nn.Module):
    """
    LSTM for estimating [μ, φ, σ_η] from a return series.

    Applies log(r² + 1e-8) to raw returns before the LSTM, matching the
    CNN preprocessing so both architectures receive the same input signal.

    Args:
        hidden_dim: LSTM hidden size (default 128).
        num_layers: Number of stacked LSTM layers (default 2).
        mlp_dim:    Width of the intermediate MLP layer (default 64).
        dropout:    Dropout between LSTM layers and in MLP (default 0.1).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        mlp_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T)
        x = torch.log(x ** 2 + 1e-8)
        x = x.unsqueeze(2)           # → (batch, T, 1)
        _, (h_n, _) = self.lstm(x)   # h_n: (num_layers, batch, hidden_dim)
        x = h_n[-1]                  # final layer → (batch, hidden_dim)
        return self.head(x)          # → (batch, 3)
