"""Build 0/1/3-day delayed confirmation panels from Experiment 3 signals."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.data import load_static_data
from pairs_trading.experiment5 import build_delayed_confirmation_rows, prepare_return_context
from pairs_trading.experiment5_config import OBSERVATION_DELAYS


def main() -> None:
    output = ROOT / "data" / "processed"
    panel = pd.read_parquet(output / "experiment3_daily_panel.parquet")
    prices = pd.read_parquet(output / "prices.parquet")
    master, _ = load_static_data(ROOT)
    context, trading_dates = prepare_return_context(prices, master)
    summary = []
    for delay in OBSERVATION_DELAYS:
        rows = build_delayed_confirmation_rows(panel, context, trading_dates, delay)
        rows.to_parquet(output / f"experiment5_delay_{delay}_panel.parquet", index=False)
        summary.append(
            {
                "observation_days": delay,
                "rows": len(rows),
                "confirmation_pass_rows": int(rows["passes_confirmation"].sum()),
            }
        )
        print(f"Delay {delay}: {len(rows):,} rows; {rows.passes_confirmation.sum():,} pass")
    pd.DataFrame(summary).to_csv(output / "experiment5_panel_summary.csv", index=False)


if __name__ == "__main__":
    main()

