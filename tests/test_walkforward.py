import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.walkforward import semiannual_windows


class WindowTests(unittest.TestCase):
    def test_only_complete_half_years_are_created(self):
        windows = semiannual_windows("2018-01-01", "2026-10-30")
        self.assertEqual(windows.iloc[0]["window_id"], "2018H1")
        self.assertEqual(windows.iloc[-1]["window_id"], "2026H1")
        self.assertEqual(windows.iloc[0]["formation_start"], pd.Timestamp("2015-01-01"))
        self.assertEqual(windows.iloc[0]["formation_end"], pd.Timestamp("2017-12-31"))
        self.assertEqual(windows.iloc[0]["trading_end"], pd.Timestamp("2018-06-30"))


if __name__ == "__main__":
    unittest.main()
