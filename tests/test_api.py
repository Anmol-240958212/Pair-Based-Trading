import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient

from api.main import app


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_overview_and_windows(self):
        overview = self.client.get("/api/overview")
        self.assertEqual(overview.status_code, 200)
        self.assertEqual(overview.json()["companies"], 206)
        windows = self.client.get("/api/windows")
        self.assertEqual(windows.status_code, 200)
        self.assertEqual(len(windows.json()), 17)

    def test_pair_filters_reject_unknown_specification(self):
        response = self.client.get(
            "/api/pairs",
            params={"window_id": "2025H1", "specification": "invalid"},
        )
        self.assertEqual(response.status_code, 422)

    def test_experiment2_is_labelled_post_hoc(self):
        response = self.client.get("/api/results/experiment2")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["interpretation"],
            "post_hoc_development_not_locked_holdout",
        )

    def test_experiment3_is_labelled_post_hoc(self):
        response = self.client.get("/api/results/experiment3")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["interpretation"],
            "post_hoc_daily_panel_development_not_locked_holdout",
        )

    def test_experiment4_is_labelled_post_hoc(self):
        response = self.client.get("/api/results/experiment4")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["interpretation"],
            "post_hoc_mathematical_model_development_not_locked_holdout",
        )

    def test_experiment5_is_labelled_post_hoc(self):
        response = self.client.get("/api/results/experiment5")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["interpretation"],
            "post_hoc_confirmation_study_not_locked_holdout",
        )

    def test_experiment6_is_labelled_post_hoc(self):
        response = self.client.get("/api/results/experiment6")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["interpretation"],
            "post_hoc_nse_announcement_veto_not_locked_holdout",
        )

    def test_experiment7_is_labelled_post_hoc(self):
        response = self.client.get("/api/results/experiment7")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["interpretation"],
            "post_hoc_payoff_barrier_study_not_locked_holdout",
        )


if __name__ == "__main__":
    unittest.main()
