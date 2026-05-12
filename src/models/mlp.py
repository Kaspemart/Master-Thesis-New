"""
MLP estimator for SV parameter estimation.

Input: ~14 summary statistics computed from the return series (fixed size,
       independent of T). This makes the MLP a representative of classical
       feature-engineering approaches — it can only use information that is
       explicitly computed and handed to it.

Summary statistics used:
    From log(r²):
        mean, variance, skewness, kurtosis
        autocorrelations at lags 1, 2, 3, 5, 10, 20
    From r²:
        autocorrelations at lags 1, 2, 3, 5

Why these features?
    - Mean of log(r²) ≈ μ − 1.27  →  identifies μ directly
    - Variance of log(r²) = σ_η²/(1−φ²) + π²/2  →  identifies σ_η given φ
    - ACF of log(r²) at lag k ≈ φ^k · correction  →  identifies φ
    - ACF of r² carries the same persistence signal through a different transform
    - Higher moments add robustness signal about the noise distribution

Output: unconstrained parameter estimates of shape (batch, 3)
        columns = [μ, logit(φ), log(σ_η)]
"""

import torch
import torch.nn as nn


def extract_features(x: torch.Tensor) -> torch.Tensor:
    """
    Compute summary statistics from a batch of return series.

    Args:
        x: raw log-returns, shape (batch, T)

    Returns:
        float32 tensor of shape (batch, 14)
    """
    eps = 1e-8
    log_r2 = torch.log(x ** 2 + eps)          # (batch, T)
    r2     = x ** 2                             # (batch, T)
    T      = x.shape[1]

    # ── Moments of log(r²) ────────────────────────────────────────────────
    mean_lr2 = log_r2.mean(dim=1, keepdim=True)             # (batch, 1)
    std_lr2  = log_r2.std(dim=1, keepdim=True).clamp(min=eps)
    centered = log_r2 - mean_lr2

    var_lr2  = (centered ** 2).mean(dim=1)                  # (batch,)
    skew_lr2 = (centered ** 3).mean(dim=1) / std_lr2[:, 0] ** 3
    kurt_lr2 = (centered ** 4).mean(dim=1) / std_lr2[:, 0] ** 4 - 3.0

    # ── ACF of log(r²) at lags 1,2,3,5,10,20 ────────────────────────────
    acf_lags_lr2 = [1, 2, 3, 5, 10, 20]
    acf_lr2 = []
    var_total = var_lr2.clamp(min=eps)
    for lag in acf_lags_lr2:
        if lag >= T:
            acf_lr2.append(torch.zeros(x.shape[0], device=x.device))
        else:
            cov = (centered[:, lag:] * centered[:, :T - lag]).mean(dim=1)
            acf_lr2.append(cov / var_total)

    # ── ACF of r² at lags 1,2,3,5 ────────────────────────────────────────
    acf_lags_r2 = [1, 2, 3, 5]
    mean_r2  = r2.mean(dim=1, keepdim=True)
    var_r2   = ((r2 - mean_r2) ** 2).mean(dim=1).clamp(min=eps)
    cent_r2  = r2 - mean_r2
    acf_r2   = []
    for lag in acf_lags_r2:
        if lag >= T:
            acf_r2.append(torch.zeros(x.shape[0], device=x.device))
        else:
            cov = (cent_r2[:, lag:] * cent_r2[:, :T - lag]).mean(dim=1)
            acf_r2.append(cov / var_r2)

    features = torch.stack(
        [mean_lr2[:, 0], var_lr2, skew_lr2, kurt_lr2]
        + acf_lr2
        + acf_r2,
        dim=1,
    )  # (batch, 14)

    # Clamp extreme values to keep gradients stable
    return features.clamp(-20.0, 20.0)


N_FEATURES = 14  # must match the feature count above


class SVMLP(nn.Module):
    """
    MLP for estimating [μ, φ, σ_η] from hand-crafted summary statistics.

    Internally calls extract_features() to compute a 14-dimensional input
    from the raw return series, then passes through stacked linear layers.

    Args:
        hidden_dim: Width of each hidden layer (default 256).
        n_layers:   Number of hidden layers (default 3).
        dropout:    Dropout probability (default 0.1).
    """

    def __init__(self, hidden_dim: int = 256, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()

        layers: list[nn.Module] = [
            nn.Linear(N_FEATURES, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        ]
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
        layers.append(nn.Linear(hidden_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T) raw returns
        features = extract_features(x)   # (batch, 14)
        return self.net(features)         # (batch, 3)
