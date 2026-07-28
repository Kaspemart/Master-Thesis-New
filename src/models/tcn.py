"""
Temporal Convolutional Network (TCN) for SV parameter estimation.

TCNs use dilated causal convolutions with exponentially growing dilation
rates (1, 2, 4, 8, ...) to achieve an exponentially large receptive field
with a linear number of parameters. Residual connections stabilise training.

For a return series of length T=1000 with kernel_size=3 and 8 TCN blocks,
the receptive field covers 2*(3-1)*(1+2+4+...+128) + 1 = 2*2*255+1 = 1021
time steps — effectively the entire series.

Input:  return series of shape (batch, T), transformed to log(r² + 1e-8)
Output: unconstrained parameter estimates of shape (batch, 3)

Reference: Bai, Kolter & Koltun (2018) "An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modelling", arXiv:1803.01271
"""

import torch
import torch.nn as nn


class _TCNBlock(nn.Module):
    """
    Single TCN residual block: two dilated causal conv layers + residual.

    Causal padding ensures the output at time t depends only on inputs ≤ t.
    Note: weight_norm is omitted — it is deprecated in PyTorch 2.x and
    performs poorly on MPS. Standard conv with careful LR tuning is equivalent.
    """

    def __init__(self, n_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation   # causal padding (left only)

        self.conv1 = nn.Conv1d(
            n_channels, n_channels, kernel_size,
            padding=pad, dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            n_channels, n_channels, kernel_size,
            padding=pad, dilation=dilation,
        )
        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, T)
        out = self.relu(self.conv1(x)[..., :x.shape[-1]])   # trim causal padding
        out = self.dropout(out)
        out = self.relu(self.conv2(out)[..., :x.shape[-1]])
        out = self.dropout(out)
        return self.relu(out + x)   # residual — same channel count throughout


class SVTCNNet(nn.Module):
    """
    TCN for estimating [μ, φ, σ_η] from a return series.

    Stacks n_blocks TCN residual blocks with dilation 1, 2, 4, ..., 2^(n_blocks-1).
    Receptive field = 2 * (kernel_size-1) * (2^n_blocks - 1) + 1.

    Global average pooling aggregates the output across the time axis,
    making the network applicable to any series length T.

    Args:
        n_channels:  Number of channels throughout the TCN (default 64).
        kernel_size: Convolutional kernel size (default 3).
        n_blocks:    Number of TCN residual blocks (default 6).
        mlp_dim:     Width of the MLP head hidden layer (default 64).
        dropout:     Dropout probability (default 0.1).
        n_outputs:   Number of output parameters (default 3 = base SV;
                     4 = ASV +ρ or SV-t +ν; 5 = ASV-t +ρ +ν).
    """

    def __init__(
        self,
        n_channels:  int = 64,
        kernel_size: int = 3,
        n_blocks:    int = 6,
        mlp_dim:     int = 64,
        dropout:     float = 0.1,
        n_outputs:   int = 3,
    ):
        super().__init__()

        # Project from 1 input channel to n_channels
        self.input_proj = nn.Conv1d(1, n_channels, kernel_size=1)

        self.blocks = nn.Sequential(*[
            _TCNBlock(n_channels, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(n_blocks)
        ])

        self.head = nn.Sequential(
            nn.Linear(n_channels, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_dim, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T)
        x = torch.log(x ** 2 + 1e-8)
        x = x.unsqueeze(1)              # → (batch, 1, T)
        x = self.input_proj(x)          # → (batch, n_channels, T)
        x = self.blocks(x)              # → (batch, n_channels, T)
        x = x.mean(dim=2)              # global average pool → (batch, n_channels)
        return self.head(x)            # → (batch, 3)
