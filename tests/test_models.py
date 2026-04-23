"""Tests for neural network architecture and training components."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.models import SVConvNet, SVLSTMNet, SVDataset, TrainConfig, train, evaluate, predict
from src.simulation.sv_params import SVParams


# ── Dataset ──────────────────────────────────────────────────────────────────

def test_dataset_loads(tmp_path):
    """SVDataset loads an npz file and returns correct shapes."""
    from src.simulation.simulator import simulate_sv, save_dataset
    result = simulate_sv(N=10, T=500, seed=0)
    path = tmp_path / "test.npz"
    save_dataset(path, result)

    ds = SVDataset(str(path))
    assert len(ds) == 10
    returns, params_t = ds[0]
    assert returns.shape == (500,)
    assert params_t.shape == (3,)
    assert returns.dtype == torch.float32
    assert params_t.dtype == torch.float32


def test_dataset_transform_roundtrip(tmp_path):
    """Parameters stored in dataset are in unconstrained space; inverse_transform recovers originals."""
    from src.simulation.simulator import simulate_sv, save_dataset
    result = simulate_sv(N=20, T=100, seed=1)
    path = tmp_path / "test.npz"
    save_dataset(path, result)

    ds = SVDataset(str(path))
    config = SVParams()

    # Collect all transformed params from dataset
    all_t = torch.stack([ds[i][1] for i in range(len(ds))]).numpy().astype(np.float64)
    recovered = config.inverse_transform(all_t).astype(np.float32)

    np.testing.assert_allclose(recovered, result.params, atol=1e-4)


# ── Architectures ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("T", [500, 1000, 2000])
def test_cnn_output_shape(T):
    model = SVConvNet()
    batch = torch.randn(8, T)
    out = model(batch)
    assert out.shape == (8, 3)


@pytest.mark.parametrize("T", [500, 1000, 2000])
def test_lstm_output_shape(T):
    model = SVLSTMNet()
    batch = torch.randn(8, T)
    out = model(batch)
    assert out.shape == (8, 3)


def test_cnn_different_T_same_weights():
    """CNN with global average pooling handles different T values with the same weights."""
    model = SVConvNet()
    out_500  = model(torch.randn(2, 500))
    out_1000 = model(torch.randn(2, 1000))
    assert out_500.shape == out_1000.shape == (2, 3)


def test_models_are_deterministic():
    """Same input + seed → same output."""
    torch.manual_seed(0)
    x = torch.randn(4, 500)
    for ModelCls in [SVConvNet, SVLSTMNet]:
        model = ModelCls()
        model.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)


# ── Training loop ─────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_loaders(tmp_path):
    """Create tiny train/val DataLoaders from simulated data."""
    from src.simulation.simulator import simulate_sv, save_dataset
    train_result = simulate_sv(N=64, T=200, seed=10)
    val_result   = simulate_sv(N=16, T=200, seed=11)

    train_path = tmp_path / "train.npz"
    val_path   = tmp_path / "val.npz"
    save_dataset(train_path, train_result)
    save_dataset(val_path, val_result)

    train_ds = SVDataset(str(train_path))
    val_ds   = SVDataset(str(val_path))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)
    return train_loader, val_loader


def test_train_cnn_runs(tmp_path, tiny_loaders):
    """Train CNN for 3 epochs without error; val loss decreases from random init."""
    train_loader, val_loader = tiny_loaders
    model = SVConvNet()
    config = TrainConfig(epochs=3, batch_size=16, lr=1e-3,
                         patience=10, device="cpu",
                         checkpoint_dir=str(tmp_path / "ckpt"))
    result = train(model, train_loader, val_loader, config, run_name="cnn_test")

    assert len(result.train_losses) == 3
    assert len(result.val_losses) == 3
    assert result.best_epoch >= 1
    assert (tmp_path / "ckpt" / "cnn_test" / "best.pt").exists()


def test_train_lstm_runs(tmp_path, tiny_loaders):
    """Train LSTM for 3 epochs without error."""
    train_loader, val_loader = tiny_loaders
    model = SVLSTMNet()
    config = TrainConfig(epochs=3, batch_size=16, lr=1e-3,
                         patience=10, device="cpu",
                         checkpoint_dir=str(tmp_path / "ckpt"))
    result = train(model, train_loader, val_loader, config, run_name="lstm_test")
    assert len(result.train_losses) == 3


def test_evaluate_returns_scalar(tiny_loaders):
    _, val_loader = tiny_loaders
    model = SVConvNet()
    loss = evaluate(model, val_loader, torch.device("cpu"))
    assert isinstance(loss, float)
    assert loss > 0


def test_predict_shape(tiny_loaders):
    _, val_loader = tiny_loaders
    model = SVConvNet()
    preds = predict(model, val_loader, torch.device("cpu"))
    assert preds.shape == (16, 3)
    assert preds.dtype == np.float32
