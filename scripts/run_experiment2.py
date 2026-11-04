"""Validate and run the post-hoc two-stage Experiment 2 comparison."""

from pathlib import Path
import sys
import argparse

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.experiment2 import (
    chronological_validation,
    walkforward_two_stage_predictions,
)
from pairs_trading.experiment2_config import (
    COMPARISON_END,
    COMPARISON_START,
    DEVELOPMENT_END,
    VALIDATION_END,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-prefix", default="experiment2")
    parser.add_argument("--output-prefix", default="experiment2")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    output = ROOT / "data" / "processed"
    events = pd.read_parquet(output / f"{args.input_prefix}_events.parquet")
    validation_metrics, validation_predictions = chronological_validation(
        events, DEVELOPMENT_END, VALIDATION_END
    )
    pd.DataFrame([validation_metrics]).to_csv(
        output / f"{args.output_prefix}_validation.csv", index=False
    )
    validation_predictions.to_parquet(
        output / f"{args.output_prefix}_validation_predictions.parquet", index=False
    )

    print("Validation")
    print(validation_metrics)
    if args.validation_only:
        return

    predictions = walkforward_two_stage_predictions(
        events, COMPARISON_START, COMPARISON_END
    )
    predictions.to_parquet(
        output / f"{args.output_prefix}_predictions.parquet", index=False
    )
    approved = predictions.loc[predictions["approved_by_experiment2"]]
    summary = predictions.groupby("window_id", as_index=False).agg(
        candidate_trades=("event_id", "size"),
        stability_passes=("passes_structural_stability", "sum"),
        approved_trades=("approved_by_experiment2", "sum"),
        all_mean_trade_return=("trade_net_return", "mean"),
        all_profit_rate=("profitable", "mean"),
    )
    approved_summary = approved.groupby("window_id", as_index=False).agg(
        approved_mean_trade_return=("trade_net_return", "mean"),
        approved_profit_rate=("profitable", "mean"),
    )
    summary.merge(approved_summary, on="window_id", how="left").to_csv(
        output / f"{args.output_prefix}_summary.csv", index=False
    )
    print("\n2022+ post-hoc comparison")
    print(
        {
            "candidate_trades": len(predictions),
            "approved_trades": int(predictions["approved_by_experiment2"].sum()),
            "all_mean_return": float(predictions["trade_net_return"].mean()),
            "approved_mean_return": float(approved["trade_net_return"].mean())
            if len(approved)
            else float("nan"),
            "all_profit_rate": float(predictions["profitable"].mean()),
            "approved_profit_rate": float(approved["profitable"].mean())
            if len(approved)
            else float("nan"),
        }
    )


if __name__ == "__main__":
    main()
