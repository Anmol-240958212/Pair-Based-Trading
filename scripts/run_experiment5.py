"""Validate observation delays and produce chronological Experiment 5 predictions."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.experiment5 import validation_result, walkforward_predictions
from pairs_trading.experiment5_config import (
    COMPARISON_END,
    COMPARISON_START,
    DEVELOPMENT_END,
    OBSERVATION_DELAYS,
    VALIDATION_END,
)


def main() -> None:
    output = ROOT / "data" / "processed"
    validation_records = []
    validation_predictions = []
    comparison_records = []
    for delay in OBSERVATION_DELAYS:
        rows = pd.read_parquet(output / f"experiment5_delay_{delay}_panel.parquet")
        validation, predictions = validation_result(rows, DEVELOPMENT_END, VALIDATION_END)
        validation_records.append(validation)
        predictions.to_parquet(
            output / f"experiment5_delay_{delay}_validation_predictions.parquet",
            index=False,
        )
        post = walkforward_predictions(rows, COMPARISON_START, COMPARISON_END)
        post.to_parquet(output / f"experiment5_delay_{delay}_predictions.parquet", index=False)
        approved = post.loc[post["approved"]]
        comparison_records.append(
            {
                "observation_days": delay,
                "comparison_rows": len(post),
                "confirmation_pass_rows": int(post["passes_confirmation"].sum()),
                "approved_rows": len(approved),
                "approved_mean_return": float(approved["trade_net_return"].mean()),
                "approved_profit_rate": float(approved["profitable"].mean()),
                "approved_take_profit_rate": float(approved["target"].mean()),
            }
        )
    validation_frame = pd.DataFrame(validation_records)
    eligible = validation_frame.loc[validation_frame["approved_rows"].ge(30)]
    selected_delay = int(
        eligible.sort_values(
            ["approved_mean_return", "approved_rows"], ascending=[False, False]
        ).iloc[0]["observation_days"]
    )
    validation_frame["selected_on_validation"] = validation_frame[
        "observation_days"
    ].eq(selected_delay)
    validation_frame.to_csv(output / "experiment5_validation.csv", index=False)
    comparison_frame = pd.DataFrame(comparison_records)
    comparison_frame["selected_on_validation"] = comparison_frame[
        "observation_days"
    ].eq(selected_delay)
    comparison_frame.to_csv(output / "experiment5_comparison.csv", index=False)
    print("Validation")
    print(validation_frame.to_string(index=False))
    print(f"\nSelected delay: {selected_delay} trading day(s)")
    print("\nPost-2022 diagnostic")
    print(comparison_frame.to_string(index=False))


if __name__ == "__main__":
    main()

