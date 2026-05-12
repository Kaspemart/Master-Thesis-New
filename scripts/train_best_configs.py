"""
Retrain the best hyperparameter config for each architecture on the full
90k training dataset, then evaluate on the test set.

Run after run_hparam_search.py has completed.

Usage:
    uv run python scripts/train_best_configs.py --T 1000
    uv run python scripts/train_best_configs.py --T 1000 --arch cnn
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dataset       import SVDataset
from src.models.hparam_search import _build_model, best_config
from src.models.train         import TrainConfig, train, predict
from src.simulation.simulator import load_dataset
from src.simulation.sv_params import SVParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PARAM_NAMES = ["mu", "phi", "sigma_eta"]
ALL_ARCHS   = ["mlp", "cnn", "lstm", "tcn", "transformer"]


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T",        type=int, default=1000, choices=[500, 1000, 2000])
    parser.add_argument("--hparam-T", type=int, default=None, choices=[500, 1000, 2000],
                        help="Use hparam log from this T (default: same as --T). "
                             "Set to 1000 to reuse T=1000 configs when training at T=500/2000.")
    parser.add_argument("--arch",     type=str, default="all",
                        choices=["all"] + ALL_ARCHS)
    parser.add_argument("--epochs",   type=int, default=100)
    args = parser.parse_args()

    device    = get_device()
    repo      = Path(__file__).parent.parent
    hparam_T  = args.hparam_T if args.hparam_T is not None else args.T
    log_path  = repo / f"experiments/hparam_log_T{hparam_T}.jsonl"

    archs = ALL_ARCHS if args.arch == "all" else [args.arch]

    train_ds = SVDataset(str(repo / f"data/train_T{args.T}.npz"))
    val_ds   = SVDataset(str(repo / f"data/val_T{args.T}.npz"))
    test_ds  = SVDataset(str(repo / f"data/test_T{args.T}.npz"))

    comparison = []

    for arch in archs:
        best = best_config(arch, hparam_T, log_path)
        if best is None:
            logger.warning("No search results for %s T=%d — skipping.", arch, hparam_T)
            continue

        hparams    = best["hparams"]
        run_name   = f"{arch}_best_T{args.T}"
        ckpt_dir   = repo / "checkpoints"

        logger.info("=" * 55)
        logger.info("Retraining %s T=%d (best hparams from search)", arch.upper(), args.T)
        logger.info("hparams: %s", hparams)
        logger.info("=" * 55)

        model = _build_model(arch, hparams)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Trainable parameters: %d", n_params)

        batch_size = hparams.get("batch_size", 256)
        # Transformer O(T²) attention — scale batch size down with T to avoid MPS OOM
        if arch == "transformer":
            if args.T >= 2000:
                batch_size = min(batch_size, 8)
                transformer_val_batch = 8
            else:
                batch_size = min(batch_size, 32)
                transformer_val_batch = 32
        else:
            transformer_val_batch = None
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=transformer_val_batch if transformer_val_batch else 512, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=transformer_val_batch if transformer_val_batch else 200, shuffle=False, num_workers=0)

        cfg = TrainConfig(
            epochs=args.epochs,
            batch_size=batch_size,
            lr=hparams["lr"],
            patience=15,
            device=device,
            checkpoint_dir=str(ckpt_dir),
        )

        result = train(model, train_loader, val_loader, cfg, run_name=run_name)

        # Load best weights and evaluate on test set
        model.load_state_dict(torch.load(
            ckpt_dir / run_name / "best.pt", map_location="cpu", weights_only=True,
        ))
        model = model.cpu()

        preds_t = predict(model, test_loader, torch.device("cpu"))
        config  = SVParams()
        preds   = config.inverse_transform(preds_t.astype(np.float64)).astype(np.float32)
        true    = load_dataset(repo / f"data/test_T{args.T}.npz").params

        # Save results
        out_dir = repo / f"results/{arch}_best_T{args.T}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "predictions.npz", preds=preds, true=true)

        errors  = preds - true
        summary = {"arch": arch, "T": args.T, "N": len(true),
                   "n_params": n_params, "hparams": hparams,
                   "best_val_loss": result.best_val_loss, "params": {}}
        for i, name in enumerate(PARAM_NAMES):
            summary["params"][name] = {
                "rmse":     float(np.sqrt(np.mean(errors[:, i] ** 2))),
                "mae":      float(np.mean(np.abs(errors[:, i]))),
                "bias":     float(np.mean(errors[:, i])),
                "rel_rmse": float(np.sqrt(np.mean(errors[:, i] ** 2)) / true[:, i].std()),
            }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Print
        print(f"\n{arch.upper()} T={args.T} (best config, full training)")
        print(f"  val_loss={result.best_val_loss:.5f}  n_params={n_params:,}")
        print(f"\n  {'param':<12} {'RMSE':>8} {'MAE':>8} {'Bias':>8} {'Rel_RMSE':>10}")
        print("  " + "-" * 50)
        for name in PARAM_NAMES:
            s = summary["params"][name]
            print(f"  {name:<12} {s['rmse']:>8.4f} {s['mae']:>8.4f} {s['bias']:>8.4f} {s['rel_rmse']:>10.3f}")

        comparison.append(summary)

    # Cross-architecture comparison
    if len(comparison) > 1:
        print(f"\n{'='*65}")
        print(f"Architecture comparison — T={args.T}")
        print(f"{'='*65}")
        print(f"  {'arch':<14} {'mu RMSE':>9} {'phi RMSE':>9} {'sig RMSE':>9} {'val_loss':>10}")
        print("  " + "-" * 55)
        for s in sorted(comparison, key=lambda x: x["best_val_loss"]):
            print(f"  {s['arch']:<14} "
                  f"{s['params']['mu']['rmse']:>9.4f} "
                  f"{s['params']['phi']['rmse']:>9.4f} "
                  f"{s['params']['sigma_eta']['rmse']:>9.4f} "
                  f"{s['best_val_loss']:>10.5f}")


if __name__ == "__main__":
    main()
