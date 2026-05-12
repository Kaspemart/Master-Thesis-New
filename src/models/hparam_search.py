"""
Random hyperparameter search framework for SV neural network estimators.

Each architecture has a defined search space. The search samples random
configs, trains each on a 10k-series subsample for up to 30 epochs, and
logs results to experiments/log.jsonl for later analysis.

The proxy task (10k samples, 30 epochs) gives a fast relative ranking.
The best config per architecture is then retrained on the full 90k dataset.
"""

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .cnn         import SVConvNet
from .lstm        import SVLSTMNet
from .mlp         import SVMLP
from .tcn         import SVTCNNet
from .transformer import SVTransformerNet
from .train       import TrainConfig, train


# ── Search spaces ──────────────────────────────────────────────────────────

SEARCH_SPACES: dict[str, dict[str, list]] = {
    "mlp": {
        "hidden_dim": [64, 128, 256, 512],
        "n_layers":   [2, 3, 4, 5],
        "dropout":    [0.0, 0.05, 0.1, 0.2, 0.3],
        "lr":         [1e-4, 3e-4, 1e-3, 3e-3],
        "batch_size": [128, 256, 512],
    },
    "cnn": {
        "hidden_dim": [64, 128, 256, 512],
        "n_blocks":   [2, 3, 4, 6],
        "dropout":    [0.0, 0.05, 0.1, 0.2, 0.3],
        "lr":         [1e-4, 3e-4, 1e-3, 3e-3],
        "batch_size": [128, 256, 512],
    },
    "lstm": {
        "hidden_dim": [64, 128, 256],
        "num_layers": [1, 2, 3],
        "dropout":    [0.0, 0.1, 0.2],
        "lr":         [1e-4, 3e-4, 1e-3],
        "batch_size": [128, 256, 512],
    },
    "tcn": {
        "n_channels":  [32, 64, 128],
        "kernel_size": [3, 5, 7],
        "n_blocks":    [4, 6, 8],
        "dropout":     [0.0, 0.05, 0.1, 0.2],
        "lr":          [1e-4, 3e-4, 1e-3, 3e-3],
        "batch_size":  [256, 512],
    },
    "transformer": {
        "d_model":  [32, 64, 128, 256],
        "n_heads":  [2, 4, 8],
        "n_layers": [2, 3, 4, 6],
        "d_ff":     [64, 128, 256, 512],
        "dropout":  [0.0, 0.05, 0.1, 0.2],
        "lr":       [1e-4, 3e-4, 1e-3],
        "batch_size": [128, 256],
    },
}

# Number of random configs to try per architecture (LSTM capped — slow)
N_TRIALS: dict[str, int] = {
    "mlp":         20,
    "cnn":         20,
    "lstm":        10,
    "tcn":         20,
    "transformer": 15,
}


def _build_model(arch: str, hparams: dict[str, Any]) -> torch.nn.Module:
    if arch == "mlp":
        return SVMLP(
            hidden_dim=hparams["hidden_dim"],
            n_layers=hparams["n_layers"],
            dropout=hparams["dropout"],
        )
    if arch == "cnn":
        # Build CNN with variable number of blocks
        from .cnn import SVConvNet, _ConvBlock
        import torch.nn as nn

        class _FlexCNN(SVConvNet):
            def __init__(self, hidden_dim, n_blocks, dropout):
                # Bypass SVConvNet.__init__, build directly
                nn.Module.__init__(self)
                blocks = []
                in_ch = 1
                for i in range(n_blocks):
                    out_ch = hidden_dim // max(1, 2 ** (n_blocks - 1 - i))
                    out_ch = max(out_ch, hidden_dim // 4)
                    blocks.append(_ConvBlock(in_ch, out_ch,
                                             kernel_size=7 if i == 0 else (5 if i == 1 else 3),
                                             dropout=dropout))
                    in_ch = out_ch
                self.convs = nn.Sequential(*blocks)
                self.head  = nn.Sequential(
                    nn.Linear(in_ch, max(in_ch // 2, 32)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                    nn.Linear(max(in_ch // 2, 32), 3),
                )

        return _FlexCNN(hparams["hidden_dim"], hparams["n_blocks"], hparams["dropout"])

    if arch == "lstm":
        return SVLSTMNet(
            hidden_dim=hparams["hidden_dim"],
            num_layers=hparams["num_layers"],
            dropout=hparams["dropout"],
        )
    if arch == "tcn":
        return SVTCNNet(
            n_channels=hparams["n_channels"],
            kernel_size=hparams["kernel_size"],
            n_blocks=hparams["n_blocks"],
            dropout=hparams["dropout"],
        )
    if arch == "transformer":
        d_model = hparams["d_model"]
        n_heads = hparams["n_heads"]
        # Ensure d_model divisible by n_heads
        if d_model % n_heads != 0:
            d_model = (d_model // n_heads) * n_heads
            if d_model == 0:
                d_model = n_heads
        return SVTransformerNet(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=hparams["n_layers"],
            d_ff=hparams["d_ff"],
            dropout=hparams["dropout"],
        )
    raise ValueError(f"Unknown architecture: {arch}")


def sample_hparams(arch: str, rng: random.Random) -> dict[str, Any]:
    space = SEARCH_SPACES[arch]
    return {k: rng.choice(v) for k, v in space.items()}


def config_id(arch: str, hparams: dict[str, Any]) -> str:
    """Short hash to uniquely identify a config."""
    key = json.dumps({"arch": arch, **hparams}, sort_keys=True)
    return arch + "_" + hashlib.md5(key.encode()).hexdigest()[:8]


@dataclass
class SearchResult:
    arch:       str
    hparams:    dict[str, Any]
    run_id:     str
    T:          int
    val_loss:   float
    best_epoch: int
    train_time_s: float
    n_params:   int


def run_search(
    arch: str,
    T: int,
    train_dataset,
    val_dataset,
    device: str,
    log_path: Path,
    checkpoint_dir: Path,
    seed: int = 0,
    proxy_n: int = 10_000,
    proxy_epochs: int = 30,
    proxy_T: int | None = None,
) -> list[SearchResult]:
    """
    Run random hyperparameter search for one architecture.

    Training uses a proxy task: proxy_n randomly sampled series from
    train_dataset, trained for at most proxy_epochs epochs. This gives
    a fast relative ranking without burning hours per config.

    Results are appended to log_path (JSONL format) after each trial
    so the search is resumable.

    Args:
        arch:           Architecture name (mlp/cnn/lstm/tcn/transformer).
        T:              Series length (used for labelling only).
        train_dataset:  Full SVDataset for training.
        val_dataset:    SVDataset for validation.
        device:         Torch device string.
        log_path:       Path to JSONL log file.
        checkpoint_dir: Root directory for per-trial weights.
        seed:           Random seed for config sampling.
        proxy_n:        Number of training series to use in the proxy task.
        proxy_epochs:   Maximum epochs per trial.
        proxy_T:        If set, truncate sequences to this length for the proxy
                        task. Useful for slow architectures (e.g. LSTM) where
                        the full T is too expensive for a quick relative ranking.

    Returns:
        List of SearchResult, one per completed trial.
    """
    import time

    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results to allow resuming
    completed_ids: set[str] = set()
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("arch") == arch and rec.get("T") == T:
                    completed_ids.add(rec["run_id"])

    rng = random.Random(seed)
    n_trials = N_TRIALS[arch]
    results: list[SearchResult] = []

    # Proxy training subset
    proxy_indices = list(range(min(proxy_n, len(train_dataset))))
    proxy_subset  = Subset(train_dataset, proxy_indices)

    # Sequence-truncating collate function for slow architectures
    def _truncate_collate(batch, trunc_T: int):
        returns = torch.stack([b[0][:trunc_T] for b in batch])
        params  = torch.stack([b[1] for b in batch])
        return returns, params

    collate_fn = (
        (lambda batch: _truncate_collate(batch, proxy_T))
        if proxy_T is not None else None
    )

    for trial in range(n_trials):
        hparams = sample_hparams(arch, rng)
        run_id  = config_id(arch, hparams)

        if run_id in completed_ids:
            print(f"  [{trial+1}/{n_trials}] {run_id} already done, skipping.")
            continue

        model = _build_model(arch, hparams)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        batch_size = hparams.get("batch_size", 256)
        train_loader = DataLoader(proxy_subset, batch_size=batch_size, shuffle=True,
                                  num_workers=0, collate_fn=collate_fn)
        val_loader   = DataLoader(val_dataset,  batch_size=512,        shuffle=False,
                                  num_workers=0, collate_fn=collate_fn)

        cfg = TrainConfig(
            epochs=proxy_epochs,
            batch_size=batch_size,
            lr=hparams["lr"],
            patience=proxy_epochs,   # no early stopping in proxy — run full epochs
            device=device,
            checkpoint_dir=str(checkpoint_dir / f"{arch}_search"),
            max_time_s=300.0,        # skip any trial that takes > 5 minutes
        )

        print(f"  [{trial+1}/{n_trials}] {arch} T={T} | {hparams} | params={n_params:,}")
        t0 = time.time()
        try:
            train_result = train(model, train_loader, val_loader, cfg, run_name=run_id)
            elapsed = time.time() - t0
        except Exception as e:
            print(f"    FAILED: {e}")
            continue


        sr = SearchResult(
            arch=arch,
            hparams=hparams,
            run_id=run_id,
            T=T,
            val_loss=train_result.best_val_loss,
            best_epoch=train_result.best_epoch,
            train_time_s=elapsed,
            n_params=n_params,
        )
        results.append(sr)

        # Append to log
        record = {
            "arch": arch, "T": T, "run_id": run_id,
            "hparams": hparams, "val_loss": sr.val_loss,
            "best_epoch": sr.best_epoch, "train_time_s": elapsed,
            "n_params": n_params,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"    val_loss={sr.val_loss:.5f}  best_epoch={sr.best_epoch}  time={elapsed:.0f}s")

    return results


def best_config(arch: str, T: int, log_path: Path) -> dict[str, Any] | None:
    """Return the hparams with the lowest val_loss for a given arch and T."""
    if not log_path.exists():
        return None
    records = []
    with open(log_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["arch"] == arch and rec["T"] == T:
                records.append(rec)
    if not records:
        return None
    best = min(records, key=lambda r: r["val_loss"])
    return best
