"""Evaluate Experiment 2 with costs, capacity and volatility-aware sizing."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.evaluation import (
    stratified_filter_permutation_test,
    volatility_scaled_portfolio_from_trade_paths,
)
from pairs_trading.experiment2_config import (
    MAXIMUM_PORTFOLIO_POSITIONS,
    PORTFOLIO_DAILY_RISK_BUDGET,
)


def main() -> None:
    output = ROOT / "data" / "processed"
    predictions = pd.read_parquet(output / "experiment2_predictions.parquet")
    paths = pd.read_parquet(output / "experiment2_trade_paths.parquet")
    significance_input = predictions.rename(
        columns={"approved_by_experiment2": "approved_by_ml"}
    )
    significance = stratified_filter_permutation_test(significance_input)
    pd.DataFrame([significance]).to_csv(
        output / "experiment2_significance.csv", index=False
    )

    records: list[dict[str, object]] = []
    strategies = {
        "dynamic_boundary_baseline": predictions,
        "two_stage_filtered": predictions.loc[
            predictions["approved_by_experiment2"]
        ],
    }
    for strategy, events in strategies.items():
        for cost in [5.0, 10.0, 20.0]:
            metrics, daily, accepted = volatility_scaled_portfolio_from_trade_paths(
                events,
                paths,
                maximum_positions=MAXIMUM_PORTFOLIO_POSITIONS,
                transaction_cost_bps=cost,
                daily_risk_budget=PORTFOLIO_DAILY_RISK_BUDGET,
            )
            metrics["strategy"] = strategy
            records.append(metrics)
            if cost == 10.0:
                daily.to_csv(output / f"experiment2_{strategy}_daily.csv", index=False)
                accepted.to_parquet(
                    output / f"experiment2_{strategy}_trades.parquet", index=False
                )
    result = pd.DataFrame.from_records(records)
    result.to_csv(output / "experiment2_portfolio.csv", index=False)
    print("Significance")
    print(significance)
    print("\nPortfolio")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
