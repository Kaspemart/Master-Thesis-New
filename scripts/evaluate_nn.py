"""
Evaluate a trained neural network on the test set and save results.

Saves to results/<model>_T<T>/:
    predictions.npz   — preds (N,3) and true (N,3) in natural parameter space
    summary.json      — RMSE, MAE, bias per parameter

Usage:
    uv run python scripts/evaluate_nn.py --arch cnn --T 1000
    uv run python scripts/evaluate_nn.py --arch lstm --T 1000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import SVConvNet, SVLSTMNet, SVDataset, predict
from src.simulation.simulator import load_dataset
from src.simulation.sv_params import SVParams

ARCH_MAP = {"cnn": SVConvNet, "lstm": SVLSTMNet}
PARAM_NAMES = ["mu", "phi", "sigma_eta"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=["cnn", "lstm"])
    parser.add_argument("--T",    required=True, type=int, choices=[500, 1000, 2000])
    args = parser.parse_args()

    repo = Path(__file__).parent.parent
    weights_path = repo / f"checkpoints/{args.arch}_T{args.T}/best.pt"
    test_path    = repo / f"data/test_T{args.T}.npz"
    out_dir      = repo / f"results/{args.arch}_T{args.T}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        print(f"No weights found at {weights_path}. Train the model first.")
        sys.exit(1)

    # Load model
    model = ARCH_MAP[args.arch]()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))

    # Predict
    test_ds     = SVDataset(str(test_path))
    test_loader = DataLoader(test_ds, batch_size=200, shuffle=False)
    preds_t = predict(model, test_loader, torch.device("cpu"))  # unconstrained

    config = SVParams()
    preds  = config.inverse_transform(preds_t.astype(np.float64)).astype(np.float32)
    true   = load_dataset(test_path).params  # (N, 3)

    # Save raw predictions
    np.savez(out_dir / "predictions.npz", preds=preds, true=true)

    # Compute and save summary
    errors  = preds - true
    summary = {"arch": args.arch, "T": args.T, "N": len(true), "params": {}}
    for i, name in enumerate(PARAM_NAMES):
        summary["params"][name] = {
            "rmse":     float(np.sqrt(np.mean(errors[:, i] ** 2))),
            "mae":      float(np.mean(np.abs(errors[:, i]))),
            "bias":     float(np.mean(errors[:, i])),
            "rel_rmse": float(np.sqrt(np.mean(errors[:, i] ** 2)) / true[:, i].std()),
        }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print table
    print(f"\n{args.arch.upper()} T={args.T} — test set (N={len(true)})")
    print(f"\n  {'param':<12} {'RMSE':>8} {'MAE':>8} {'Bias':>8} {'Rel_RMSE':>10}")
    print("  " + "-" * 50)
    for name in PARAM_NAMES:
        s = summary["params"][name]
        print(f"  {name:<12} {s['rmse']:>8.4f} {s['mae']:>8.4f} {s['bias']:>8.4f} {s['rel_rmse']:>10.3f}")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
