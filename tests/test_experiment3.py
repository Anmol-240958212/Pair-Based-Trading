import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.experiment3 import build_daily_pair_samples


class DailyPanelTests(unittest.TestCase):
    def test_labels_are_available_after_signal_date(self):
        dates = pd.bdate_range("2020-01-01", periods=110)
        right = 100 * np.exp(np.arange(len(dates)) * 0.001)
        left = right * np.exp(0.15 * np.sin(np.arange(len(dates)) / 5))
        prices = pd.concat(
            [
                pd.DataFrame(
                    {
                        "date": dates,
                        "capitaline_code": code,
                        "close": values,
                        "turnover_rs_thousand": 1_000.0,
                    }
                )
                for code, values in [(1, left), (2, right)]
            ],
            ignore_index=True,
        )
        metrics = {
            "hedge_ratio": 1.0,
            "intercept": 0.0,
            "formation_spread_mean": 0.0,
            "formation_spread_standard_deviation": 0.05,
            "cointegration_p_value": 0.01,
            "cointegration_p_value_reverse": 0.02,
            "return_correlation": 0.8,
            "spread_half_life_days": 10.0,
            "formation_hedge_ratio_drift": 0.1,
            "formation_spread_mean_shift": 0.2,
            "formation_spread_volatility_ratio": 1.0,
            "formation_zero_crossing_rate": 0.1,
            "minimum_pair_median_turnover": 1_000.0,
        }
        samples = build_daily_pair_samples(
            prices,
            metrics,
            1,
            2,
            "Synthetic",
            "2020H1",
            dates[60],
            dates[-1],
            feature_history_start=dates[0],
        )
        self.assertFalse(samples.empty)
        self.assertTrue((samples["label_available_date"] > samples["signal_date"]).all())
        self.assertTrue(samples["holding_days"].between(1, 20).all())


if __name__ == "__main__":
    unittest.main()
