import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.experiment5 import build_delayed_confirmation_rows


class ConfirmationDelayTests(unittest.TestCase):
    def test_confirmation_entry_uses_later_signal_row(self):
        dates = pd.bdate_range("2020-01-01", periods=6)
        panel = pd.DataFrame(
            {
                "window_id": "2020H1",
                "event_id": [f"e{index}" for index in range(6)],
                "left_code": 1,
                "right_code": 2,
                "signal_date": dates,
                "entry_date": dates + pd.offsets.BDay(1),
                "exit_date": dates + pd.offsets.BDay(5),
                "label_available_date": dates + pd.offsets.BDay(5),
                "entry_z_score": np.linspace(2.0, 1.5, 6),
                "left_return_volatility_20d": 0.02,
                "right_return_volatility_20d": 0.02,
            }
        )
        trading_dates = pd.DataFrame(
            {"date": dates, "trading_day_index": np.arange(len(dates))}
        )
        context = pd.DataFrame(
            [
                {
                    "capitaline_code": code,
                    "trading_day_index": index,
                    "stock_return": 0.01,
                    "industry_return": 0.008,
                    "industry_residual": 0.002,
                    "abnormal_turnover_ratio": 1.0,
                }
                for index in range(len(dates))
                for code in [1, 2]
            ]
        )
        delayed = build_delayed_confirmation_rows(panel, context, trading_dates, 3)
        self.assertTrue((delayed["signal_date"] > delayed["trigger_date"]).all())
        self.assertTrue((delayed["entry_date"] > delayed["signal_date"]).all())
        self.assertTrue((delayed["observation_days"] == 3).all())


if __name__ == "__main__":
    unittest.main()
