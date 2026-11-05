import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.experiment6 import add_announcement_features


class AnnouncementFeatureTests(unittest.TestCase):
    def test_only_point_in_time_window_news_is_used(self):
        rows = pd.DataFrame(
            {
                "event_id": ["event"],
                "left_code": [1],
                "right_code": [2],
                "trigger_date": ["2024-01-01"],
                "signal_date": ["2024-01-04"],
            }
        )
        announcements = pd.DataFrame(
            {
                "capitaline_code": [1, 1, 2],
                "seq_id": ["before", "inside", "after"],
                "broadcast_at": ["2023-12-31 10:00", "2024-01-03 16:00", "2024-01-05 09:00"],
                "material_event": [True, True, True],
            }
        )
        result = add_announcement_features(rows, announcements).iloc[0]
        self.assertEqual(result["left_announcement_count"], 1)
        self.assertEqual(result["right_announcement_count"], 0)
        self.assertTrue(result["one_sided_material_event"])
        self.assertFalse(result["passes_news_veto"])


if __name__ == "__main__":
    unittest.main()
