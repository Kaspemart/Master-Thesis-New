"""
1D convolutional neural network for SV parameter estimation.

Input:  return series of shape (batch, T), transformed internally to log(r² + 1e-8)
Output: unconstrained parameter estimates of shape (batch, 3)
        columns = [μ, logit(φ), log(σ_η)]

Architecture: stacked Conv1d blocks with ReLU and dropout,
followed by global average pooling and an MLP head.

Input preprocessing note:
    Raw returns span ~100x in scale across series (μ ∈ (-10,0) means exp(μ/2)
    varies from 0.007 to 1.0). BatchNorm running statistics cannot represent
    this range reliably, causing train/eval divergence. The log(r²) transform
    compresses inputs to a ~25-unit window (log(r²) ≈ h_t + log(eps²), mean
    ≈ μ − 1.27) with similar variance regardless of μ. This is also the
    standard preprocessing in the SV neural network literature.
"""

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SVConvNet(nn.Module):
    """
    1D CNN for estimating [μ, φ, σ_η] from a return series.

    Internally applies log(r² + 1e-8) to the raw returns before the conv
    stack. Global average pooling makes the network applicable to any T.

    Args:
        hidden_dim: Number of channels in the final conv blocks (default 128).
        mlp_dim:    Width of the intermediate MLP layer (default 64).
        dropout:    Dropout probability applied after each conv and in MLP (default 0.1).
    """

    def __init__(self, hidden_dim: int = 128, mlp_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.Sequential(
            _ConvBlock(1,               hidden_dim // 4, kernel_size=7, dropout=dropout),
            _ConvBlock(hidden_dim // 4, hidden_dim // 2, kernel_size=5, dropout=dropout),
            _ConvBlock(hidden_dim // 2, hidden_dim,      kernel_size=3, dropout=dropout),
            _ConvBlock(hidden_dim,      hidden_dim,      kernel_size=3, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T)
        # log(r² + 1e-8) compresses the 100x scale variation across series
        x = torch.log(x ** 2 + 1e-8)
        x = x.unsqueeze(1)         # → (batch, 1, T)
        x = self.convs(x)          # → (batch, hidden_dim, T)
        x = x.mean(dim=2)          # global average pool → (batch, hidden_dim)
        return self.head(x)        # → (batch, 3)
