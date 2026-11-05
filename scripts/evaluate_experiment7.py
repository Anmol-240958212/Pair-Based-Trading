"""Portfolio evaluation for alternative payoff barriers."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.evaluation import moving_block_bootstrap_mean_test, portfolio_from_trade_paths, select_portfolio_trades
from pairs_trading.experiment3 import build_daily_trade_path
from pairs_trading.experiment5_config import MAXIMUM_PORTFOLIO_POSITIONS
from pairs_trading.experiment7 import BARRIER_CONFIGURATIONS


def main() -> None:
    output = ROOT / "data" / "processed"
    prices = pd.read_parquet(output / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    validation = pd.read_csv(output / "experiment7_validation.csv")
    selected_name = str(validation.loc[validation["selected_on_validation"], "configuration"].iloc[0])
    portfolio_records = []
    significance_records = []
    for name in BARRIER_CONFIGURATIONS:
        predictions = pd.read_parquet(output / f"experiment7_{name}_predictions.parquet")
        candidates = predictions.loc[predictions["approved"]].copy()
        candidates["selection_score"] = candidates["predicted_probability"]
        accepted = select_portfolio_trades(candidates, MAXIMUM_PORTFOLIO_POSITIONS, True)
        paths = pd.concat(
            [build_daily_trade_path(prices, row) for row in accepted.to_dict(orient="records")],
            ignore_index=True,
        )
        for cost in [5.0, 10.0, 20.0]:
            metrics, daily, selected = portfolio_from_trade_paths(
                accepted, paths, MAXIMUM_PORTFOLIO_POSITIONS, cost
            )
            metrics["configuration"] = name
            metrics["selected_on_validation"] = name == selected_name
            portfolio_records.append(metrics)
            if cost == 10.0:
                daily.to_csv(output / f"experiment7_{name}_daily.csv", index=False)
                selected.to_parquet(output / f"experiment7_{name}_trades.parquet", index=False)
                significance = moving_block_bootstrap_mean_test(daily["portfolio_return"])
                significance["configuration"] = name
                significance["selected_on_validation"] = name == selected_name
                significance_records.append(significance)
    portfolio = pd.DataFrame(portfolio_records)
    significance = pd.DataFrame(significance_records)
    portfolio.to_csv(output / "experiment7_portfolio.csv", index=False)
    significance.to_csv(output / "experiment7_significance.csv", index=False)
    print(portfolio.to_string(index=False))
    print("\nSignificance")
    print(significance.to_string(index=False))


if __name__ == "__main__":
    main()

