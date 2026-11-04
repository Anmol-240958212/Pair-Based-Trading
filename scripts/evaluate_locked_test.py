"""Evaluate statistical significance, capacity and cost sensitivity."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.evaluation import (
    portfolio_from_trade_paths,
    stratified_filter_permutation_test,
)


def main() -> None:
    output = ROOT / "data" / "processed"
    predictions = pd.read_parquet(output / "locked_test_predictions.parquet")
    paths = pd.read_parquet(output / "locked_test_trade_paths.parquet")
    significance = stratified_filter_permutation_test(predictions)
    pd.DataFrame([significance]).to_csv(
        output / "locked_test_significance.csv", index=False
    )

    records = []
    for strategy, events in {
        "baseline": predictions,
        "ml_filtered": predictions.loc[predictions["approved_by_ml"]],
    }.items():
        for cost in [5.0, 10.0, 20.0]:
            metrics, daily, accepted = portfolio_from_trade_paths(
                events, paths, maximum_positions=10, transaction_cost_bps=cost
            )
            metrics["strategy"] = strategy
            records.append(metrics)
            if cost == 10.0:
                daily.to_csv(output / f"{strategy}_portfolio_daily.csv", index=False)
                accepted.to_parquet(
                    output / f"{strategy}_portfolio_trades.parquet", index=False
                )
    portfolio = pd.DataFrame.from_records(records)
    portfolio.to_csv(output / "locked_test_portfolio_cost_sensitivity.csv", index=False)
    print("Significance")
    print(significance)
    print("\nPortfolio and cost sensitivity")
    print(portfolio.to_string(index=False))


if __name__ == "__main__":
    main()
