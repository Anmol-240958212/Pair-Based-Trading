"""Reconstruct daily paths for Experiment 2 comparison trades."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.backtest import backtest_pair
from pairs_trading.experiment2_config import (
    ENTRY_Z,
    EXIT_Z,
    HEDGE_LOOKBACK_DAYS,
    HEDGE_RATIO_MODE,
    MAXIMUM_HOLDING_DAYS,
    MINIMUM_ENTRY_DAYS_REMAINING,
    STOP_Z,
    TRANSACTION_COST_BPS,
)


def main() -> None:
    output = ROOT / "data" / "processed"
    predictions = pd.read_parquet(output / "experiment2_predictions.parquet")
    prices = pd.read_parquet(output / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    path_frames: list[pd.DataFrame] = []
    group_columns = [
        "window_id", "left_code", "right_code", "formation_start",
        "formation_end", "trading_start", "trading_end",
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
            minimum_entry_days_remaining=MINIMUM_ENTRY_DAYS_REMAINING,
            hedge_ratio_mode=HEDGE_RATIO_MODE,
            hedge_lookback_days=HEDGE_LOOKBACK_DAYS,
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
    paths.to_parquet(output / "experiment2_trade_paths.parquet", index=False)
    print(f"Saved {len(paths)} rows for {paths.event_id.nunique()} events.")


if __name__ == "__main__":
    main()
