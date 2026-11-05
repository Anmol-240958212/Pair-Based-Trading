"""Apply portfolio constraints and evaluate Experiment 4 models."""

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
from pairs_trading.experiment4 import MODEL_NAMES
from pairs_trading.experiment4_config import MAXIMUM_PORTFOLIO_POSITIONS


def main() -> None:
    output = ROOT / "data" / "processed"
    predictions = pd.read_parquet(output / "experiment4_predictions.parquet")
    prices = pd.read_parquet(output / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    records: list[dict[str, object]] = []
    significance_records: list[dict[str, object]] = []
    window_frames: list[pd.DataFrame] = []
    for model in MODEL_NAMES:
        candidates = predictions.loc[predictions[f"{model}_approved"]].copy()
        candidates["selection_score"] = candidates[f"{model}_probability"]
        accepted = select_portfolio_trades(
            candidates,
            maximum_positions=MAXIMUM_PORTFOLIO_POSITIONS,
            avoid_shared_companies=True,
        )
        if accepted.empty:
            continue
        paths = pd.concat(
            [
                build_daily_trade_path(prices, row)
                for row in accepted.to_dict(orient="records")
            ],
            ignore_index=True,
        )
        paths.to_parquet(output / f"experiment4_{model}_paths.parquet", index=False)
        window_summary = accepted.groupby("window_id", as_index=False).agg(
            accepted_trades=("event_id", "size"),
            mean_trade_return=("trade_net_return", "mean"),
            profit_rate=("profitable", "mean"),
            take_profit_rate=("target", "mean"),
        )
        window_summary.insert(0, "strategy", model)
        window_frames.append(window_summary)
        for cost in [5.0, 10.0, 20.0]:
            metrics, daily, selected = portfolio_from_trade_paths(
                accepted,
                paths,
                maximum_positions=MAXIMUM_PORTFOLIO_POSITIONS,
                transaction_cost_bps=cost,
            )
            metrics["strategy"] = model
            records.append(metrics)
            if cost == 10.0:
                daily.to_csv(output / f"experiment4_{model}_daily.csv", index=False)
                selected.to_parquet(
                    output / f"experiment4_{model}_trades.parquet", index=False
                )
                significance = moving_block_bootstrap_mean_test(daily["portfolio_return"])
                significance["strategy"] = model
                significance_records.append(significance)
    portfolio = pd.DataFrame.from_records(records)
    portfolio.to_csv(output / "experiment4_portfolio.csv", index=False)
    pd.DataFrame.from_records(significance_records).to_csv(
        output / "experiment4_significance.csv", index=False
    )
    pd.concat(window_frames, ignore_index=True).to_csv(
        output / "experiment4_window_summary.csv", index=False
    )
    print(portfolio.to_string(index=False))
    print("\nBlock-bootstrap mean tests")
    print(pd.DataFrame.from_records(significance_records).to_string(index=False))


if __name__ == "__main__":
    main()

