"""Train and compare alternative take-profit/stop-loss specifications."""

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
    VALIDATION_END,
)
from pairs_trading.experiment7 import BARRIER_CONFIGURATIONS


def main() -> None:
    output = ROOT / "data" / "processed"
    validation_records = []
    comparison_records = []
    for name in BARRIER_CONFIGURATIONS:
        rows = pd.read_parquet(output / f"experiment7_{name}_panel.parquet")
        validation, validation_predictions = validation_result(
            rows, DEVELOPMENT_END, VALIDATION_END
        )
        validation["configuration"] = name
        validation_records.append(validation)
        validation_predictions.to_parquet(
            output / f"experiment7_{name}_validation_predictions.parquet", index=False
        )
        predictions = walkforward_predictions(rows, COMPARISON_START, COMPARISON_END)
        predictions.to_parquet(output / f"experiment7_{name}_predictions.parquet", index=False)
        approved = predictions.loc[predictions["approved"]]
        comparison_records.append(
            {
                "configuration": name,
                "comparison_rows": len(predictions),
                "approved_rows": len(approved),
                "approved_mean_return": float(approved["trade_net_return"].mean()),
                "approved_profit_rate": float(approved["profitable"].mean()),
                "approved_take_profit_rate": float(approved["target"].mean()),
            }
        )
    validation_frame = pd.DataFrame(validation_records)
    eligible = validation_frame.loc[validation_frame["approved_rows"].ge(30)]
    selected = str(
        eligible.sort_values(
            ["approved_mean_return", "approved_rows"], ascending=[False, False]
        ).iloc[0]["configuration"]
    )
    validation_frame["selected_on_validation"] = validation_frame["configuration"].eq(selected)
    validation_frame.to_csv(output / "experiment7_validation.csv", index=False)
    comparison = pd.DataFrame(comparison_records)
    comparison["selected_on_validation"] = comparison["configuration"].eq(selected)
    comparison.to_csv(output / "experiment7_comparison.csv", index=False)
    print("Validation")
    print(validation_frame.to_string(index=False))
    print(f"\nSelected payoff: {selected}")
    print("\nPost-2022 diagnostic")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

