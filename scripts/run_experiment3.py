"""Validate and run daily-panel Logistic and Gradient Boosting models."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.experiment3 import (
    validation_comparison,
    walkforward_daily_predictions,
)
from pairs_trading.experiment3_config import (
    COMPARISON_END,
    COMPARISON_START,
    DEVELOPMENT_END,
    VALIDATION_END,
)


def main() -> None:
    output = ROOT / "data" / "processed"
    events = pd.read_parquet(output / "experiment3_daily_panel.parquet")
    validation, validation_predictions = validation_comparison(
        events, DEVELOPMENT_END, VALIDATION_END
    )
    validation.to_csv(output / "experiment3_validation.csv", index=False)
    validation_predictions.to_parquet(
        output / "experiment3_validation_predictions.parquet", index=False
    )
    predictions = walkforward_daily_predictions(
        events, COMPARISON_START, COMPARISON_END
    )
    predictions.to_parquet(output / "experiment3_predictions.parquet", index=False)
    records: list[dict[str, object]] = []
    for model in ["logistic_regression", "hist_gradient_boosting"]:
        approved = predictions.loc[predictions[f"{model}_approved"]]
        records.append(
            {
                "model": model,
                "comparison_rows": int(len(predictions)),
                "approved_rows": int(len(approved)),
                "approved_mean_return": float(approved["trade_net_return"].mean()),
                "approved_profit_rate": float(approved["profitable"].mean()),
                "approved_take_profit_rate": float(approved["target"].mean()),
            }
        )
    pd.DataFrame.from_records(records).to_csv(
        output / "experiment3_comparison.csv", index=False
    )
    print("Validation")
    print(validation.to_string(index=False))
    print("\nPost-2022 development comparison")
    print(pd.DataFrame.from_records(records).to_string(index=False))


if __name__ == "__main__":
    main()
