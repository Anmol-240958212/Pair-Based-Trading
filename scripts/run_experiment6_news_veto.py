"""Attach [-3,0] NSE announcements and evaluate a one-sided material-event veto."""

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
from pairs_trading.experiment6 import add_announcement_features
from pairs_trading.experiment5_config import MAXIMUM_PORTFOLIO_POSITIONS


def summary(frame: pd.DataFrame, approved_column: str, sample: str) -> dict[str, object]:
    approved = frame.loc[frame[approved_column]]
    return {
        "sample": sample,
        "strategy": approved_column,
        "covered_rows": len(frame),
        "approved_rows": len(approved),
        "approved_mean_return": float(approved["trade_net_return"].mean()),
        "approved_profit_rate": float(approved["profitable"].mean()),
        "approved_take_profit_rate": float(approved["target"].mean()),
    }


def main() -> None:
    output = ROOT / "data" / "processed"
    announcements = pd.read_parquet(output / "experiment6_nse_announcements.parquet")
    covered_codes = set(announcements["capitaline_code"])
    comparison_records = []
    post_with_news = None
    for sample, filename in [
        ("validation_2020_2021", "experiment5_delay_3_validation_predictions.parquet"),
        ("post_2022", "experiment5_delay_3_predictions.parquet"),
    ]:
        frame = pd.read_parquet(output / filename)
        frame = frame.loc[
            frame["left_code"].isin(covered_codes) & frame["right_code"].isin(covered_codes)
        ].copy()
        frame = add_announcement_features(frame, announcements)
        frame["approved_without_news"] = frame["approved"].astype(bool)
        frame["approved_with_news_veto"] = (
            frame["approved_without_news"] & frame["passes_news_veto"]
        )
        comparison_records.extend(
            [
                summary(frame, "approved_without_news", sample),
                summary(frame, "approved_with_news_veto", sample),
            ]
        )
        frame.to_parquet(output / f"experiment6_{sample}_predictions.parquet", index=False)
        if sample == "post_2022":
            post_with_news = frame
    comparison = pd.DataFrame(comparison_records)
    comparison.to_csv(output / "experiment6_comparison.csv", index=False)

    if post_with_news is None:
        raise RuntimeError("Post-2022 news frame was not created.")
    prices = pd.read_parquet(output / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    portfolio_records = []
    significance_records = []
    for strategy in ["approved_without_news", "approved_with_news_veto"]:
        candidates = post_with_news.loc[post_with_news[strategy]].copy()
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
        for cost in [5.0, 10.0, 20.0]:
            metrics, daily, selected = portfolio_from_trade_paths(
                accepted,
                paths,
                maximum_positions=MAXIMUM_PORTFOLIO_POSITIONS,
                transaction_cost_bps=cost,
            )
            metrics["strategy"] = strategy
            portfolio_records.append(metrics)
            if cost == 10.0:
                daily.to_csv(output / f"experiment6_{strategy}_daily.csv", index=False)
                selected.to_parquet(output / f"experiment6_{strategy}_trades.parquet", index=False)
                significance = moving_block_bootstrap_mean_test(daily["portfolio_return"])
                significance["strategy"] = strategy
                significance_records.append(significance)
    portfolio = pd.DataFrame(portfolio_records)
    significance = pd.DataFrame(significance_records)
    portfolio.to_csv(output / "experiment6_portfolio.csv", index=False)
    significance.to_csv(output / "experiment6_significance.csv", index=False)
    print("Event-level comparison")
    print(comparison.to_string(index=False))
    print("\nPortfolio comparison")
    print(portfolio.to_string(index=False))
    print("\nSignificance")
    print(significance.to_string(index=False))


if __name__ == "__main__":
    main()

