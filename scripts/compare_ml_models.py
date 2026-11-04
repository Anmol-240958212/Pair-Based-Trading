"""Reproduce the initial development/validation classifier comparison."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.ml import compare_classifiers_chronologically


def main() -> None:
    event_path = ROOT / "data" / "processed" / "ml_events.parquet"
    if not event_path.exists():
        raise FileNotFoundError("Run scripts/build_ml_events.py first.")
    events = pd.read_parquet(event_path)
    comparison, _, predictions = compare_classifiers_chronologically(
        events,
        training_end="2019-12-31",
        validation_end="2021-12-31",
    )
    output = ROOT / "data" / "processed"
    comparison.to_csv(output / "ml_validation_comparison.csv", index=False)
    predictions.to_parquet(output / "ml_validation_predictions.parquet", index=False)
    print(comparison.to_string(index=False))
    print("\nThe 2022 onward data was not used in this comparison.")


if __name__ == "__main__":
    main()
