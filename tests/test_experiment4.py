import numpy as np
import pandas as pd
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.experiment4 import (
    REGIME_FEATURE_COLUMNS,
    REGIME_INPUT_COLUMNS,
    add_markov_regime_features,
    select_validation_threshold,
)


def _regime_frame(start: str, periods: int, offset: float = 0.0) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    values = np.sin(np.arange(periods) / 7.0) + offset
    frame = pd.DataFrame({"signal_date": dates, "target": (values > offset).astype(int)})
    for index, column in enumerate(REGIME_INPUT_COLUMNS):
        frame[column] = values + index * 0.05
    return frame


class Experiment4Tests(unittest.TestCase):
    def test_markov_features_are_probabilities_and_preserve_rows(self):
        training = _regime_frame("2018-01-01", 100)
        prediction = _regime_frame("2019-01-01", 20, offset=0.4)
        augmented_training, augmented_prediction = add_markov_regime_features(
            training, prediction
        )
        self.assertEqual(len(augmented_training), len(training))
        self.assertEqual(len(augmented_prediction), len(prediction))
        for column in REGIME_FEATURE_COLUMNS:
            self.assertTrue(augmented_training[column].between(0, 1).all())
            self.assertTrue(augmented_prediction[column].between(0, 1).all())

    def test_threshold_selection_uses_predeclared_cutoff(self):
        validation = pd.DataFrame(
            {
                "trade_net_return": np.r_[np.repeat(0.02, 35), np.repeat(-0.01, 35)],
            }
        )
        probability = np.r_[np.repeat(0.8, 35), np.repeat(0.5, 35)]
        threshold, mean_return, count = select_validation_threshold(
            validation, probability
        )
        self.assertEqual(threshold, 0.6)
        self.assertEqual(mean_return, 0.02)
        self.assertEqual(count, 35)


if __name__ == "__main__":
    unittest.main()
