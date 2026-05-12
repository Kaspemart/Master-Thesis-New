"""
Run random hyperparameter search for one or all architectures.

Each trial trains on a 10k-series proxy dataset for up to 30 epochs.
Results are logged to experiments/hparam_log_T<T>.jsonl.

After the search, the best config for each architecture is retrained
on the full 90k dataset via scripts/train_best_configs.py.

Usage:
    uv run python scripts/run_hparam_search.py --T 1000
    uv run python scripts/run_hparam_search.py --T 1000 --arch cnn
    uv run python scripts/run_hparam_search.py --T 1000 --arch transformer
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dataset import SVDataset
from src.models.hparam_search import run_search, best_config, N_TRIALS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T",    type=int, default=1000, choices=[500, 1000, 2000])
    parser.add_argument("--arch", type=str, default="all",
                        choices=["all", "mlp", "cnn", "lstm", "tcn", "transformer"])
    parser.add_argument("--proxy-n",      type=int, default=10_000)
    parser.add_argument("--proxy-epochs", type=int, default=30)
    parser.add_argument("--proxy-T",      type=int, default=None,
                        help="Truncate sequences to this length for proxy (useful for LSTM)")
    parser.add_argument("--seed",         type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    repo   = Path(__file__).parent.parent
    log_path = repo / f"experiments/hparam_log_T{args.T}.jsonl"

    archs = ["mlp", "cnn", "lstm", "tcn", "transformer"] if args.arch == "all" else [args.arch]

    print(f"Device: {device}  |  T={args.T}  |  proxy_n={args.proxy_n}  |  proxy_epochs={args.proxy_epochs}  |  proxy_T={args.proxy_T}")
    print(f"Log: {log_path}")
    print(f"Architectures: {archs}")
    print()

    train_ds = SVDataset(str(repo / f"data/train_T{args.T}.npz"))
    val_ds   = SVDataset(str(repo / f"data/val_T{args.T}.npz"))

    for arch in archs:
        print(f"{'='*55}")
        print(f"Architecture: {arch.upper()}  ({N_TRIALS[arch]} trials)")
        print(f"{'='*55}")

        run_search(
            arch=arch,
            T=args.T,
            train_dataset=train_ds,
            val_dataset=val_ds,
            device=device,
            log_path=log_path,
            checkpoint_dir=repo / "experiments" / "checkpoints",
            seed=args.seed,
            proxy_n=args.proxy_n,
            proxy_epochs=args.proxy_epochs,
            proxy_T=args.proxy_T,
        )

        best = best_config(arch, args.T, log_path)
        if best:
            print(f"\nBest {arch.upper()} config:  val_loss={best['val_loss']:.5f}")
            print(f"  hparams: {best['hparams']}")
        print()


if __name__ == "__main__":
    main()
