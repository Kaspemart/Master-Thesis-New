"""
Descriptive statistics of daily log-returns for the 15 real-data assets (Chapter 7,
Section 7.2). Reports, per asset: number of observations, mean, standard deviation,
skewness, excess kurtosis, minimum and maximum. Means/SD/min/max are shown in
percent for readability; skewness and excess kurtosis are unitless.

Also reports the in-sample / out-of-sample split sizes.

Run:  uv run python scripts/real_descriptive_stats.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from src.data.real_data import ASSET_GROUPS, LABELS, load_returns, split_returns


def main():
    print("=" * 92)
    print("Descriptive statistics of daily log-returns (mean/SD/min/max in %, full sample)")
    print("=" * 92)
    print(f"{'Asset':>18} {'N':>6} {'Mean':>7} {'SD':>7} {'Skew':>7} "
          f"{'ExKurt':>8} {'Min':>8} {'Max':>8} {'In':>6} {'Out':>6}")
    print("-" * 92)
    for group, members in ASSET_GROUPS.items():
        print(f"[{group}]")
        for a in members:
            r = load_returns(a).values
            ins, oos = split_returns(a)
            print(f"{LABELS[a]:>18} {len(r):>6} "
                  f"{100*r.mean():>7.3f} {100*r.std():>7.3f} "
                  f"{stats.skew(r):>7.2f} {stats.kurtosis(r):>8.2f} "
                  f"{100*r.min():>8.2f} {100*r.max():>8.2f} "
                  f"{len(ins):>6} {len(oos):>6}")
    print("-" * 92)
    print("ExKurt = excess kurtosis (0 for a normal distribution). "
          "In/Out = in-sample (2010-2017) / out-of-sample (2018-2025) observations.")


if __name__ == "__main__":
    main()
