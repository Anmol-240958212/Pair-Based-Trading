"""Reconstruct daily return paths for every trade in the locked test sample."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.backtest import backtest_pair
from pairs_trading.data import load_price_data
from pairs_trading.research_config import (
    ENTRY_Z,
    EXIT_Z,
    MAXIMUM_HOLDING_DAYS,
    STOP_Z,
    TRANSACTION_COST_BPS,
)


def main() -> None:
    predictions = pd.read_parquet(
        ROOT / "data" / "processed" / "locked_test_predictions.parquet"
    )
    prices = load_price_data(ROOT, range(2015, 2027))
    path_frames: list[pd.DataFrame] = []
    group_columns = [
        "window_id",
        "left_code",
        "right_code",
        "formation_start",
        "formation_end",
        "trading_start",
        "trading_end",
    ]
    for keys, pair_events in predictions.groupby(group_columns, sort=False):
        values = dict(zip(group_columns, keys))
        _, daily, trades = backtest_pair(
            prices,
            int(values["left_code"]),
            int(values["right_code"]),
            values["formation_start"],
            values["formation_end"],
            values["trading_start"],
            values["trading_end"],
            ENTRY_Z,
            EXIT_Z,
            STOP_Z,
            MAXIMUM_HOLDING_DAYS,
            TRANSACTION_COST_BPS,
        )
        trades["signal_date"] = pd.to_datetime(trades["signal_date"])
        pair_events = pair_events.copy()
        pair_events["signal_date"] = pd.to_datetime(pair_events["signal_date"])
        matched = trades.merge(
            pair_events[["signal_date", "event_id"]], on="signal_date", how="inner"
        )
        for trade in matched.itertuples(index=False):
            path = daily.loc[
                daily["date"].between(trade.entry_date, trade.exit_date),
                ["date", "gross_return", "turnover", "transaction_cost", "net_return"],
            ].copy()
            path.insert(0, "event_id", trade.event_id)
            path_frames.append(path)
    paths = pd.concat(path_frames, ignore_index=True)
    paths.to_parquet(
        ROOT / "data" / "processed" / "locked_test_trade_paths.parquet",
        index=False,
    )
    print(f"Saved {len(paths):,} daily trade-path rows for {paths.event_id.nunique()} events.")


if __name__ == "__main__":
    main()
