"""
Training loop and evaluation utilities for SV parameter estimation networks.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """
    Hyperparameters for neural network training.

    Attributes:
        epochs:        Maximum number of training epochs.
        batch_size:    Mini-batch size.
        lr:            Initial learning rate for Adam.
        patience:      Early stopping patience (epochs without val loss improvement).
        lr_factor:     Factor by which LR is reduced on plateau.
        lr_patience:   Epochs without improvement before reducing LR.
        weight_decay:  L2 regularisation coefficient.
        num_workers:   DataLoader worker processes.
        device:        Torch device string (e.g. "cpu", "cuda", "mps").
        checkpoint_dir: Directory to save the best model weights.
    """
    epochs: int = 100
    batch_size: int = 512
    lr: float = 1e-3
    patience: int = 15
    lr_factor: float = 0.5
    lr_patience: int = 7
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"
    max_time_s: float | None = None   # per-trial wall-clock cap (seconds)


@dataclass
class TrainResult:
    """
    Output of train().

    Attributes:
        train_losses: MSE loss on training set per epoch.
        val_losses:   MSE loss on validation set per epoch.
        best_epoch:   Epoch with lowest validation loss (1-indexed).
        best_val_loss: Lowest validation loss achieved.
    """
    train_losses: list[float] = field(default_factory=list)
    val_losses:   list[float] = field(default_factory=list)
    best_epoch:   int = 0
    best_val_loss: float = float("inf")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    run_name: str = "model",
) -> TrainResult:
    """
    Train a model with early stopping and LR reduction on plateau.

    Best model weights (lowest validation MSE) are saved to
    config.checkpoint_dir / run_name / best.pt

    Args:
        model:        The network to train (SVConvNet or SVLSTMNet).
        train_loader: DataLoader over training SVDataset.
        val_loader:   DataLoader over validation SVDataset.
        config:       Training hyperparameters.
        run_name:     Subdirectory name for checkpoints (e.g. "cnn_T1000").

    Returns:
        TrainResult with per-epoch losses and best epoch info.
    """
    device = torch.device(config.device)
    model = model.to(device)

    checkpoint_path = Path(config.checkpoint_dir) / run_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_weights_path = checkpoint_path / "best.pt"

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=config.lr_factor, patience=config.lr_patience,
    )
    criterion = nn.MSELoss()

    result = TrainResult()
    epochs_no_improve = 0
    t_start = time.time()

    for epoch in range(1, config.epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        timed_out = False
        for returns, params_t in train_loader:
            if config.max_time_s is not None and (time.time() - t_start) > config.max_time_s:
                timed_out = True
                break
            returns  = returns.to(device)
            params_t = params_t.to(device)
            optimizer.zero_grad()
            preds = model(returns)
            loss = criterion(preds, params_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(returns)
        if timed_out:
            logger.info("Aborting: trial exceeded %.0fs wall-clock limit at epoch %d", config.max_time_s, epoch)
            break
        train_loss /= len(train_loader.dataset)

        # --- Validation ---
        val_loss = evaluate(model, val_loader, device, criterion)

        result.train_losses.append(train_loss)
        result.val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < result.best_val_loss:
            result.best_val_loss = val_loss
            result.best_epoch = epoch
            torch.save(model.state_dict(), best_weights_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %3d/%d  train=%.5f  val=%.5f  lr=%.2e  best_epoch=%d",
                epoch, config.epochs, train_loss, val_loss, lr_now, result.best_epoch,
            )

        if epochs_no_improve >= config.patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, config.patience)
            break

        if epoch == 1 and (val_loss > 1e6 or not np.isfinite(val_loss)):
            logger.info("Aborting: val loss exploded at epoch 1 (%.3e)", val_loss)
            break


    logger.info(
        "Training complete. Best val loss=%.5f at epoch %d. Weights: %s",
        result.best_val_loss, result.best_epoch, best_weights_path,
    )
    return result


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> float:
    """
    Compute mean MSE loss over a DataLoader. Model is set to eval mode.
    """
    if criterion is None:
        criterion = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for returns, params_t in loader:
            returns  = returns.to(device)
            params_t = params_t.to(device)
            preds = model(returns)
            total_loss += criterion(preds, params_t).item() * len(returns)
    return total_loss / len(loader.dataset)


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Run inference and return predictions in unconstrained space.

    Returns:
        float32 array of shape (N, 3) — [μ, logit(φ), log(σ_η)].
        Apply SVParams.inverse_transform() to convert to natural space.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for returns, _ in loader:
            returns = returns.to(device)
            preds.append(model(returns).cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)
