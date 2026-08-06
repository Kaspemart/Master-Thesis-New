"""
Fetch the equity-index and commodity-ETF daily series from Yahoo Finance for the
empirical application (Chapter 7), saving one CSV per asset to data/real/.

Adjusted prices (auto_adjust=True folds in dividends and splits) are used, which
is the appropriate "total-return" price for return/volatility modelling. Each CSV
has two columns: date and price, matching the FRED files.

Run:  uv run python scripts/fetch_yahoo_data.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = {
    # equity indices
    "GSPC": "^GSPC", "N225": "^N225", "AXJO": "^AXJO",
    "STOXX50E": "^STOXX50E", "FTSE": "^FTSE",
    # commodity ETFs
    "GLD": "GLD", "SLV": "SLV", "USO": "USO", "DBC": "DBC", "DBA": "DBA",
}
START, END = "2010-01-01", "2026-01-01"
OUT = Path("data/real")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print(f"{'asset':>10} {'n':>6}  {'start':>12} {'end':>12}")
    for name, tk in TICKERS.items():
        df = yf.download(tk, start=START, end=END, auto_adjust=True, progress=False)
        if df.empty:
            print(f"{name:>10}  EMPTY — check ticker/network")
            continue
        # single-ticker download may return multi-level columns; flatten
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        s = close.dropna()
        s.index.name = "date"
        s.to_frame("price").to_csv(OUT / f"{name}.csv")
        print(f"{name:>10} {len(s):>6}  {str(s.index.min().date()):>12} {str(s.index.max().date()):>12}")


if __name__ == "__main__":
    main()
