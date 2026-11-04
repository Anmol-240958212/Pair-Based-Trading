"""Run the frozen 2022–2026H1 walk-forward test exactly as pre-specified."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.ml import walkforward_predict_locked_test
from pairs_trading.research_config import (
    CLASSIFIER_NAME,
    CLASSIFIER_TARGET,
    LOCKED_TEST_END,
    LOCKED_TEST_START,
    PAIR_SPECIFICATION,
    PROBABILITY_THRESHOLD,
)
from pairs_trading.selection import apply_pair_specification


def main() -> None:
    events = pd.read_parquet(ROOT / "data" / "processed" / "ml_events.parquet")
    selected = apply_pair_specification(events, PAIR_SPECIFICATION)
    predictions = walkforward_predict_locked_test(
        selected,
        LOCKED_TEST_START,
        LOCKED_TEST_END,
        CLASSIFIER_NAME,
        CLASSIFIER_TARGET,
        PROBABILITY_THRESHOLD,
    )
    output = ROOT / "data" / "processed"
    predictions.to_parquet(output / "locked_test_predictions.parquet", index=False)
    summary = (
        predictions.groupby("window_id", as_index=False)
        .agg(
            baseline_trades=("event_id", "size"),
            ml_approved_trades=("approved_by_ml", "sum"),
            baseline_success_rate=("profitable", "mean"),
            baseline_mean_trade_return=("trade_net_return", "mean"),
        )
    )
    approved = predictions.loc[predictions["approved_by_ml"]]
    ml_summary = approved.groupby("window_id", as_index=False).agg(
        ml_success_rate=("profitable", "mean"),
        ml_mean_trade_return=("trade_net_return", "mean"),
    )
    summary = summary.merge(ml_summary, on="window_id", how="left")
    summary.to_csv(output / "locked_test_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nOverall")
    print(
        {
            "baseline_trades": len(predictions),
            "ml_approved_trades": int(predictions["approved_by_ml"].sum()),
            "baseline_mean_trade_return": float(predictions["trade_net_return"].mean()),
            "ml_mean_trade_return": float(approved["trade_net_return"].mean()),
            "baseline_success_rate": float(predictions["profitable"].mean()),
            "ml_success_rate": float(approved["profitable"].mean()),
        }
    )


if __name__ == "__main__":
    main()
