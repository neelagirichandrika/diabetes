"""
Unit and integration tests for Course Outcomes 1 through 6 (CO1 to CO6).
Verifies that all Course Outcome runners, certified audit reports,
and visual architecture/diagnostic diagrams are present and valid.

Run with:
    py -m unittest tests/test_co_modules.py
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.app import app


class TestCourseOutcomes(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_co_directories_exist(self):
        """Verifies that all 6 CO directories exist."""
        for co_id in ["co1", "co2", "co3", "co4", "co5", "co6"]:
            d_path = os.path.join(PROJECT_ROOT, co_id)
            self.assertTrue(os.path.isdir(d_path), f"Directory {co_id} missing.")

    def test_co_runners_exist(self):
        """Verifies that each Course Outcome has an executable audit runner."""
        runners = {
            "co1": "run_co1_lifecycle.py",
            "co2": "run_co2_linear_models.py",
            "co3": "run_co3_tree_models.py",
            "co4": "run_co4_unsupervised.py",
            "co5": "run_co5_evaluation.py",
            "co6": "run_co6_docker.py"
        }
        for co_id, fname in runners.items():
            path = os.path.join(PROJECT_ROOT, co_id, fname)
            self.assertTrue(os.path.isfile(path), f"Runner {co_id}/{fname} missing.")

    def test_co_reports_exist(self):
        """Verifies that all 6 Course Outcome audit reports exist and are non-empty."""
        reports = {
            "co1": "co1_lifecycle_report.txt",
            "co2": "co2_linear_models_report.txt",
            "co3": "co3_tree_models_report.txt",
            "co4": "co4_unsupervised_report.txt",
            "co5": "co5_evaluation_report.txt",
            "co6": "co6_docker_report.txt"
        }
        for co_id, fname in reports.items():
            path = os.path.join(PROJECT_ROOT, co_id, fname)
            self.assertTrue(os.path.isfile(path), f"Report {co_id}/{fname} missing.")
            self.assertGreater(os.path.getsize(path), 200, f"Report {co_id}/{fname} is too small or empty.")

    def test_co_visuals_exist(self):
        """Verifies that all 6 Course Outcome visual plots exist and are non-empty images."""
        visuals = {
            "co1": "co1_lifecycle_architecture.png",
            "co2": "co2_odds_ratios_regularization.png",
            "co3": "co3_tree_depth_overfitting.png",
            "co4": "patient_clusters_pca.png",
            "co5": "co5_evaluation_calibration.png",
            "co6": "docker_container_architecture.png"
        }
        for co_id, fname in visuals.items():
            path = os.path.join(PROJECT_ROOT, co_id, fname)
            self.assertTrue(os.path.isfile(path), f"Visual {co_id}/{fname} missing.")
            self.assertGreater(os.path.getsize(path), 10000, f"Visual {co_id}/{fname} is truncated or empty.")

    def test_api_co_status_endpoint(self):
        """Verifies that GET /api/co-status returns 200 and certifies all 6 COs."""
        resp = self.client.get("/api/co-status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["total_cos"], 6)
        self.assertEqual(data["certified_count"], 6)


if __name__ == "__main__":
    unittest.main()
