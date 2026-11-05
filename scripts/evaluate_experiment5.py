"""Evaluate each delayed-confirmation rule under identical portfolio constraints."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.evaluation import (
    moving_block_bootstrap_mean_test,
    portfolio_from_trade_paths,
    select_portfolio_trades,
)
from pairs_trading.experiment3 import build_daily_trade_path
from pairs_trading.experiment5_config import MAXIMUM_PORTFOLIO_POSITIONS, OBSERVATION_DELAYS


def main() -> None:
    output = ROOT / "data" / "processed"
    prices = pd.read_parquet(output / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    validation = pd.read_csv(output / "experiment5_validation.csv")
    selected_delay = int(
        validation.loc[validation["selected_on_validation"], "observation_days"].iloc[0]
    )
    portfolio_records = []
    significance_records = []
    window_frames = []
    for delay in OBSERVATION_DELAYS:
        predictions = pd.read_parquet(
            output / f"experiment5_delay_{delay}_predictions.parquet"
        )
        candidates = predictions.loc[predictions["approved"]].copy()
        candidates["selection_score"] = candidates["predicted_probability"]
        accepted = select_portfolio_trades(
            candidates,
            maximum_positions=MAXIMUM_PORTFOLIO_POSITIONS,
            avoid_shared_companies=True,
        )
        paths = pd.concat(
            [build_daily_trade_path(prices, row) for row in accepted.to_dict(orient="records")],
            ignore_index=True,
        )
        paths.to_parquet(output / f"experiment5_delay_{delay}_paths.parquet", index=False)
        summary = accepted.groupby("window_id", as_index=False).agg(
            accepted_trades=("event_id", "size"),
            mean_trade_return=("trade_net_return", "mean"),
            profit_rate=("profitable", "mean"),
            take_profit_rate=("target", "mean"),
        )
        summary.insert(0, "observation_days", delay)
        window_frames.append(summary)
        for cost in [5.0, 10.0, 20.0]:
            metrics, daily, selected = portfolio_from_trade_paths(
                accepted,
                paths,
                maximum_positions=MAXIMUM_PORTFOLIO_POSITIONS,
                transaction_cost_bps=cost,
            )
            metrics["observation_days"] = delay
            metrics["selected_on_validation"] = delay == selected_delay
            portfolio_records.append(metrics)
            if cost == 10.0:
                daily.to_csv(output / f"experiment5_delay_{delay}_daily.csv", index=False)
                selected.to_parquet(
                    output / f"experiment5_delay_{delay}_trades.parquet", index=False
                )
                significance = moving_block_bootstrap_mean_test(daily["portfolio_return"])
                significance["observation_days"] = delay
                significance["selected_on_validation"] = delay == selected_delay
                significance_records.append(significance)
    portfolio = pd.DataFrame(portfolio_records)
    significance = pd.DataFrame(significance_records)
    portfolio.to_csv(output / "experiment5_portfolio.csv", index=False)
    significance.to_csv(output / "experiment5_significance.csv", index=False)
    pd.concat(window_frames, ignore_index=True).to_csv(
        output / "experiment5_window_summary.csv", index=False
    )
    print(portfolio.to_string(index=False))
    print("\nBlock-bootstrap mean tests")
    print(significance.to_string(index=False))


if __name__ == "__main__":
    main()

