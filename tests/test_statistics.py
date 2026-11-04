import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.statistics import analyze_pair


class CointegrationRobustnessTests(unittest.TestCase):
    def test_reports_both_directions_and_johansen(self):
        rng = np.random.default_rng(12)
        dates = pd.bdate_range("2018-01-01", periods=500)
        common = np.cumsum(rng.normal(0, 0.01, len(dates)))
        left = np.exp(4.5 + 1.1 * common + rng.normal(0, 0.01, len(dates)))
        right = np.exp(4.0 + common)
        prices = pd.DataFrame(
            {
                "date": np.tile(dates, 2),
                "capitaline_code": np.repeat([1, 2], len(dates)),
                "close": np.concatenate([left, right]),
            }
        )
        metrics, _ = analyze_pair(prices, 1, 2, dates.min(), dates.max())
        self.assertLess(metrics["cointegration_p_value"], 0.05)
        self.assertLess(metrics["cointegration_p_value_reverse"], 0.05)
        self.assertEqual(
            metrics["cointegration_p_value_conservative"],
            max(
                metrics["cointegration_p_value"],
                metrics["cointegration_p_value_reverse"],
            ),
        )
        self.assertIn("johansen_passes_5pct", metrics)


if __name__ == "__main__":
    unittest.main()
