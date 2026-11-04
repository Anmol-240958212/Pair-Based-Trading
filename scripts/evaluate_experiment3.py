"""Apply non-overlap constraints and evaluate daily-panel strategies."""

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
from pairs_trading.experiment3_config import MAXIMUM_PORTFOLIO_POSITIONS


def main() -> None:
    output = ROOT / "data" / "processed"
    predictions = pd.read_parquet(output / "experiment3_predictions.parquet")
    prices = pd.read_parquet(output / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])

    strategies: dict[str, pd.DataFrame] = {}
    baseline = predictions.copy()
    baseline["selection_score"] = baseline["absolute_entry_z_score"]
    strategies["daily_signal_baseline"] = baseline
    for model in ["logistic_regression", "hist_gradient_boosting"]:
        approved = predictions.loc[predictions[f"{model}_approved"]].copy()
        approved["selection_score"] = approved[f"{model}_probability"]
        strategies[model] = approved

    records: list[dict[str, object]] = []
    significance_records: list[dict[str, object]] = []
    window_frames: list[pd.DataFrame] = []
    for strategy, candidates in strategies.items():
        accepted = select_portfolio_trades(
            candidates,
            maximum_positions=MAXIMUM_PORTFOLIO_POSITIONS,
            avoid_shared_companies=True,
        )
        paths = pd.concat(
            [
                build_daily_trade_path(prices, row)
                for row in accepted.to_dict(orient="records")
            ],
            ignore_index=True,
        )
        paths.to_parquet(output / f"experiment3_{strategy}_paths.parquet", index=False)
        window_summary = accepted.groupby("window_id", as_index=False).agg(
            accepted_trades=("event_id", "size"),
            mean_trade_return=("trade_net_return", "mean"),
            profit_rate=("profitable", "mean"),
            take_profit_rate=("target", "mean"),
        )
        window_summary.insert(0, "strategy", strategy)
        window_frames.append(window_summary)
        for cost in [5.0, 10.0, 20.0]:
            metrics, daily, selected = portfolio_from_trade_paths(
                accepted,
                paths,
                maximum_positions=MAXIMUM_PORTFOLIO_POSITIONS,
                transaction_cost_bps=cost,
            )
            metrics["strategy"] = strategy
            records.append(metrics)
            if cost == 10.0:
                daily.to_csv(output / f"experiment3_{strategy}_daily.csv", index=False)
                selected.to_parquet(
                    output / f"experiment3_{strategy}_trades.parquet", index=False
                )
                significance = moving_block_bootstrap_mean_test(
                    daily["portfolio_return"]
                )
                significance["strategy"] = strategy
                significance_records.append(significance)
    result = pd.DataFrame.from_records(records)
    result.to_csv(output / "experiment3_portfolio.csv", index=False)
    pd.DataFrame.from_records(significance_records).to_csv(
        output / "experiment3_significance.csv", index=False
    )
    pd.concat(window_frames, ignore_index=True).to_csv(
        output / "experiment3_window_summary.csv", index=False
    )
    print(result.to_string(index=False))
    print("\nBlock-bootstrap mean tests")
    print(pd.DataFrame.from_records(significance_records).to_string(index=False))


if __name__ == "__main__":
    main()
