import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.backtest import _position_path


class PositionPathTests(unittest.TestCase):
    def test_entry_decision_is_available_for_next_day_application(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        decisions, _ = _position_path(
            pd.Series([2.5, 2.2, 0.4], index=dates),
            entry_z=2.0,
            exit_z=0.5,
            stop_z=4.0,
            maximum_holding_days=60,
        )
        applied = decisions.shift(1, fill_value=0).astype(int)
        self.assertEqual(applied.tolist(), [0, -1, -1])

    def test_stop_loss_requires_reset_before_reentry(self):
        dates = pd.date_range("2020-01-01", periods=6, freq="D")
        decisions, reasons = _position_path(
            pd.Series([2.5, 4.2, 3.0, 1.0, 2.4, 0.2], index=dates),
            entry_z=2.0,
            exit_z=0.5,
            stop_z=4.0,
            maximum_holding_days=60,
        )
        self.assertEqual(decisions.tolist(), [-1, 0, 0, 0, -1, 0])
        self.assertEqual(reasons[dates[1]], "stop_loss")

    def test_threshold_exit_is_recorded_as_mean_reversion(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        decisions, reasons = _position_path(
            pd.Series([-2.5, -1.0, -0.4], index=dates),
            entry_z=2.0,
            exit_z=0.5,
            stop_z=4.0,
            maximum_holding_days=60,
        )
        self.assertEqual(decisions.tolist(), [1, 1, 0])
        self.assertEqual(reasons[dates[2]], "mean_reversion")


if __name__ == "__main__":
    unittest.main()
