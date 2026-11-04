import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.evaluation import select_portfolio_trades


class PortfolioSelectionTests(unittest.TestCase):
    def test_rejects_overlapping_company(self):
        events = pd.DataFrame(
            {
                "event_id": ["a", "b", "c"],
                "entry_date": pd.to_datetime(["2022-01-03"] * 3),
                "exit_date": pd.to_datetime(["2022-01-10"] * 3),
                "left_code": [1, 1, 4],
                "right_code": [2, 3, 5],
                "predicted_probability": [0.9, 0.8, 0.7],
            }
        )
        selected = select_portfolio_trades(events, maximum_positions=2)
        self.assertEqual(selected["event_id"].tolist(), ["a", "c"])


if __name__ == "__main__":
    unittest.main()
