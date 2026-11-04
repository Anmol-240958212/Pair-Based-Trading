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


if __name__ == "__main__":
    unittest.main()
