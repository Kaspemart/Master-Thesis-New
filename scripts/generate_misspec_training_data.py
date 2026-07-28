"""
Generate training/validation data for the correctly-specified (cell c) networks.

Three models — ASV, SV-t, ASV-t — each 90,000 train + 10,000 validation series at
T=1000, mirroring the base SV setup. Used to train the correct-model TCNs.

Fresh seeds, distinct from all existing (42/123/321/456/654/789/987/999):
  ASV   train 1001, val 1002
  SV-t  train 1003, val 1004
  ASV-t train 1005, val 1006

Output: data/train_{asv,svt,asvt}_T1000.npz, data/val_{asv,svt,asvt}_T1000.npz

Run:  uv run python scripts/generate_misspec_training_data.py
"""

from __future__ import annotations

from src.simulation.simulator import (
    simulate_sv_leverage,
    simulate_sv_t,
    simulate_asv_t,
    save_dataset,
)

T = 1000
N_TRAIN = 90_000
N_VAL = 10_000

JOBS = [
    ("asv",  simulate_sv_leverage, 1001, 1002),
    ("svt",  simulate_sv_t,        1003, 1004),
    ("asvt", simulate_asv_t,       1005, 1006),
]


def main():
    for name, fn, seed_tr, seed_val in JOBS:
        print(f"[{name}] train (N={N_TRAIN}, seed={seed_tr})...")
        save_dataset(f"data/train_{name}_T{T}.npz", fn(N=N_TRAIN, T=T, seed=seed_tr))
        print(f"[{name}] val   (N={N_VAL}, seed={seed_val})...")
        save_dataset(f"data/val_{name}_T{T}.npz", fn(N=N_VAL, T=T, seed=seed_val))
        print(f"[{name}] done.")


if __name__ == "__main__":
    main()
