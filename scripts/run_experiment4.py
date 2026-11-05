"""Validate and run the mathematical-model comparison."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.experiment4 import MODEL_NAMES, validation_comparison, walkforward_predictions
from pairs_trading.experiment4_config import (
    COMPARISON_END,
    COMPARISON_START,
    DEVELOPMENT_END,
    VALIDATION_END,
)


def main() -> None:
    output = ROOT / "data" / "processed"
    events = pd.read_parquet(output / "experiment3_daily_panel.parquet")
    validation, validation_predictions, thresholds = validation_comparison(
        events, DEVELOPMENT_END, VALIDATION_END
    )
    validation.to_csv(output / "experiment4_validation.csv", index=False)
    validation_predictions.to_parquet(
        output / "experiment4_validation_predictions.parquet", index=False
    )
    pd.DataFrame(
        [{"model": model, "selected_threshold": value} for model, value in thresholds.items()]
    ).to_csv(output / "experiment4_thresholds.csv", index=False)
    predictions = walkforward_predictions(
        events, thresholds, COMPARISON_START, COMPARISON_END
    )
    predictions.to_parquet(output / "experiment4_predictions.parquet", index=False)
    comparison = []
    for model in MODEL_NAMES:
        approved = predictions.loc[predictions[f"{model}_approved"]]
        comparison.append(
            {
                "model": model,
                "comparison_rows": int(len(predictions)),
                "approved_rows": int(len(approved)),
                "approved_mean_return": float(approved["trade_net_return"].mean()),
                "approved_profit_rate": float(approved["profitable"].mean()),
                "approved_take_profit_rate": float(approved["target"].mean()),
            }
        )
    pd.DataFrame(comparison).to_csv(output / "experiment4_comparison.csv", index=False)
    print("Validation and frozen thresholds")
    print(validation.to_string(index=False))
    print("\nPost-2022 development comparison")
    print(pd.DataFrame(comparison).to_string(index=False))


if __name__ == "__main__":
    main()

