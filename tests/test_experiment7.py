import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.experiment7 import relabel_barriers


class BarrierRelabellingTests(unittest.TestCase):
    def test_symmetric_barrier_updates_label_and_availability_date(self):
        dates = pd.bdate_range("2024-01-01", periods=6)
        prices = pd.concat(
            [
                pd.DataFrame({"date": dates, "capitaline_code": 1, "close": [100, 100, 102, 102, 102, 102]}),
                pd.DataFrame({"date": dates, "capitaline_code": 2, "close": 100.0}),
            ],
            ignore_index=True,
        )
        rows = pd.DataFrame(
            {
                "window_id": ["2024H1"], "event_id": ["event"], "left_code": [1],
                "right_code": [2], "signal_date": [dates[0]], "entry_date": [dates[1]],
                "exit_date": [dates[-1]], "label_available_date": [dates[-1]],
                "trading_end": [dates[-1]], "absolute_hedge_ratio": [1.0], "direction": [1],
            }
        )
        result = relabel_barriers(rows, prices, 0.005, -0.005).iloc[0]
        self.assertEqual(result["target"], 1)
        self.assertEqual(result["barrier_outcome"], "take_profit")
        self.assertEqual(result["label_available_date"], result["exit_date"])
        self.assertGreater(result["exit_date"], result["signal_date"])


if __name__ == "__main__":
    unittest.main()
