import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.ml import (
    FEATURE_COLUMNS,
    build_labeled_events,
    compare_classifiers_chronologically,
)


class EventLabellingTests(unittest.TestCase):
    def test_success_requires_reversion_and_positive_net_return(self):
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        daily = pd.DataFrame(
            {
                "date": dates,
                "left_close": np.linspace(100, 110, 30),
                "right_close": np.linspace(80, 87, 30),
                "spread": np.sin(np.arange(30) / 3),
                "z_score": np.linspace(-2.5, 2.5, 30),
            }
        )
        trades = pd.DataFrame(
            {
                "signal_date": [dates[20], dates[22]],
                "entry_date": [dates[21], dates[23]],
                "exit_date": [dates[24], dates[27]],
                "entry_z_score": [2.1, -2.2],
                "exit_reason": ["mean_reversion", "stop_loss"],
                "net_return": [0.02, 0.01],
            }
        )
        metrics = {
            "cointegration_p_value": 0.01,
            "return_correlation": 0.6,
            "hedge_ratio": 1.2,
            "spread_half_life_days": 12.0,
        }
        events = build_labeled_events(metrics, daily, trades, 1, 2, "IT")
        self.assertEqual(events["target"].tolist(), [1, 0])
        self.assertEqual(events["reverted"].tolist(), [1, 0])
        self.assertEqual(events["profitable"].tolist(), [1, 1])


class ChronologicalComparisonTests(unittest.TestCase):
    def test_training_and_validation_are_separated_by_date(self):
        rng = np.random.default_rng(7)
        dates = pd.date_range("2018-01-01", periods=80, freq="MS")
        events = pd.DataFrame({"signal_date": dates})
        for column in FEATURE_COLUMNS:
            events[column] = rng.normal(size=len(events))
        events["target"] = (events["absolute_entry_z_score"] > 0).astype(int)
        events["label_available_date"] = events["signal_date"] + pd.Timedelta(days=5)

        comparison, fitted, predictions = compare_classifiers_chronologically(
            events,
            training_end="2022-12-31",
            validation_end="2024-08-31",
        )
        self.assertEqual(set(fitted), {
            "logistic_regression", "decision_tree", "bagging_decision_tree"
        })
        self.assertTrue((predictions["signal_date"] > pd.Timestamp("2022-12-31")).all())
        self.assertEqual(comparison["training_events"].unique().tolist(), [60])
        self.assertEqual(comparison["validation_events"].unique().tolist(), [20])

    def test_unfinished_training_label_is_not_used(self):
        rng = np.random.default_rng(9)
        dates = pd.date_range("2020-01-01", periods=30, freq="MS")
        events = pd.DataFrame({"signal_date": dates})
        for column in FEATURE_COLUMNS:
            events[column] = rng.normal(size=len(events))
        events["target"] = np.arange(len(events)) % 2
        events["label_available_date"] = events["signal_date"]
        events.loc[events["signal_date"].eq(pd.Timestamp("2020-12-01")),
                   "label_available_date"] = pd.Timestamp("2021-02-01")

        comparison, _, _ = compare_classifiers_chronologically(
            events,
            training_end="2020-12-31",
            validation_end="2022-06-30",
        )
        self.assertEqual(comparison["training_events"].unique().tolist(), [11])


if __name__ == "__main__":
    unittest.main()
