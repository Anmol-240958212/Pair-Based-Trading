"""Create a fast, derived Parquet copy of the unmodified annual workbooks."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.data import PRICE_COLUMNS, _read_capitaline_excel


def main() -> None:
    output_dir = ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in range(2015, 2027):
        path = ROOT / f"data_{year}.xlsx"
        print(f"Reading {path.name}...", flush=True)
        frame = _read_capitaline_excel(path, PRICE_COLUMNS)
        frame["source_year"] = year
        frames.append(frame)

    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    numeric_columns = [
        "capitaline_code", "open", "close", "volume_adjusted", "turnover_rs_thousand"
    ]
    for column in numeric_columns:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    prices = prices.dropna(subset=["capitaline_code", "date", "close"])
    prices = prices.loc[prices["close"] > 0].copy()
    prices["capitaline_code"] = prices["capitaline_code"].astype("int64")
    prices = prices.sort_values(["capitaline_code", "date"]).reset_index(drop=True)

    output_path = output_dir / "prices.parquet"
    prices.to_parquet(output_path, index=False, compression="zstd")
    print(
        f"Saved {len(prices):,} rows and {prices['capitaline_code'].nunique():,} companies "
        f"to {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()

