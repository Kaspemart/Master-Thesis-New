"""
Transformer estimator for SV parameter estimation.

Uses multi-head self-attention over the return series. The series is first
projected to d_model dimensions and augmented with sinusoidal positional
encoding. After N transformer encoder blocks, global average pooling
produces a fixed-size representation fed to an MLP head.

Computational note: self-attention is O(T²) in time and memory. For T=2000
this produces 4M attention values per head per layer — feasible but slow.
Training T=2000 with this architecture takes significantly longer than CNN/TCN.

Input:  return series of shape (batch, T), transformed to log(r² + 1e-8)
Output: unconstrained parameter estimates of shape (batch, 3)
"""

import math

import torch
import torch.nn as nn


class _SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al. 2017).

    Supports sequences up to max_len. The encoding is added to the input
    embeddings and is not learned.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, d_model)
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)


class SVTransformerNet(nn.Module):
    """
    Transformer encoder for estimating [μ, φ, σ_η] from a return series.

    Each return value is projected to d_model dimensions, positionally encoded,
    then passed through n_layers transformer encoder blocks. Global average
    pooling over the time axis gives a fixed-size representation.

    Args:
        d_model:   Embedding dimension (default 64). Must be divisible by n_heads.
        n_heads:   Number of attention heads (default 4).
        n_layers:  Number of transformer encoder blocks (default 3).
        d_ff:      Feed-forward hidden dimension inside each block (default 128).
        mlp_dim:   Width of the final MLP head hidden layer (default 64).
        dropout:   Dropout probability throughout (default 0.1).
    """

    def __init__(
        self,
        d_model:  int = 64,
        n_heads:  int = 4,
        n_layers: int = 3,
        d_ff:     int = 128,
        mlp_dim:  int = 64,
        dropout:  float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc    = _SinusoidalPositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,      # pre-norm (more stable training)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T)
        x = torch.log(x ** 2 + 1e-8)
        x = x.unsqueeze(2)              # → (batch, T, 1)
        x = self.input_proj(x)          # → (batch, T, d_model)
        x = self.pos_enc(x)             # → (batch, T, d_model)
        x = self.encoder(x)             # → (batch, T, d_model)
        x = x.mean(dim=1)              # global average pool → (batch, d_model)
        return self.head(x)            # → (batch, 3)
