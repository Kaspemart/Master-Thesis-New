"""
Loader for the real-data application (Chapter 7).

Handles both file formats (FRED: observation_date,CODE; Yahoo: date,price),
inverts the two reversed FX series to a common USD-per-foreign basis, computes
decimal log-returns, and splits into in-sample (2010-2017) and out-of-sample
(2018-2025) windows.

Decimal log-returns (log(P_t / P_{t-1})) are on the ~0.01 scale, matching the SV
simulation scale the networks were trained on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REAL_DIR = Path(__file__).resolve().parents[2] / "data" / "real"
OOS_START = "2018-01-01"   # in-sample 2010-2017, out-of-sample 2018-2025

ASSET_GROUPS = {
    "FX":        ["DEXUSEU", "DEXUSUK", "DEXSZUS", "DEXJPUS", "DEXUSAL"],
    "Equity":    ["GSPC", "N225", "AXJO", "STOXX50E", "FTSE"],
    "Commodity": ["GLD", "SLV", "USO", "DBC", "DBA"],
}

LABELS = {
    "DEXUSEU": "EUR/USD", "DEXUSUK": "GBP/USD", "DEXSZUS": "CHF/USD",
    "DEXJPUS": "JPY/USD", "DEXUSAL": "AUD/USD",
    "GSPC": "S&P 500", "N225": "Nikkei 225", "AXJO": "ASX 200",
    "STOXX50E": "EuroStoxx 50", "FTSE": "FTSE 100",
    "GLD": "Gold (GLD)", "SLV": "Silver (SLV)", "USO": "Oil (USO)",
    "DBC": "Commodities (DBC)", "DBA": "Agriculture (DBA)",
}

# FRED series quoted as foreign currency per USD; inverted to USD per foreign.
REVERSED_FX = {"DEXSZUS", "DEXJPUS"}

ALL_ASSETS = [a for group in ASSET_GROUPS.values() for a in group]


def load_prices(name: str) -> pd.Series:
    """Load one asset's price series (dates as index), FX inversion applied."""
    df = pd.read_csv(REAL_DIR / f"{name}.csv")
    df.columns = ["date", "price"]                      # normalise both formats
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["price"]
    s = pd.to_numeric(s, errors="coerce").dropna()      # drop FRED '.' markers
    if name in REVERSED_FX:
        s = 1.0 / s                                     # -> USD per foreign currency
    return s.sort_index()


def load_returns(name: str) -> pd.Series:
    """Decimal daily log-returns for one asset."""
    p = load_prices(name)
    return np.log(p).diff().dropna()


def split_returns(name: str, oos_start: str = OOS_START):
    """(in_sample, out_of_sample) log-return Series for one asset."""
    r = load_returns(name)
    return r[r.index < oos_start], r[r.index >= oos_start]


def group_of(name: str) -> str:
    for g, members in ASSET_GROUPS.items():
        if name in members:
            return g
    raise KeyError(name)
