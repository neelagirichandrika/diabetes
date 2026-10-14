"""
Integration tests for Flask web application routes and API endpoints.
Run with:
    py -m unittest tests/test_api.py
"""

import unittest
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.app import app
from src.predict import SAMPLE_PATIENTS


class TestWebApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_get_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Diabetic Inpatient Readmission Prediction', response.data)
        self.assertIn(b'Section 01', response.data)

    def test_post_home_page_prediction(self):
        response = self.app.post('/', data=SAMPLE_PATIENTS['sample_baseline'])
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Readmission Risk', response.data)

    def test_api_predict_valid_json(self):
        response = self.app.post('/api/predict', json=SAMPLE_PATIENTS['sample_elevated'])
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('prediction', data['data'])
        self.assertIn('readmission_probability', data['data'])

    def test_api_model_info(self):
        response = self.app.get('/api/model-info')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('model_name', data)
    def test_all_dashboard_tabs_present(self):
        """Verifies that all 9 main dashboard tabs and all 6 curriculum subpanels are present."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')

        expected_tabs = [
            'tab-predict', 'tab-modules', 'tab-co6', 'tab-dataset',
            'tab-eda', 'tab-preprocessing', 'tab-models', 'tab-diagnostics', 'tab-coefficients'
        ]
        for t in expected_tabs:
            self.assertIn(f'id="{t}"', html, f"Tab {t} missing in rendered dashboard.")

        expected_subpanels = ['subpanel-m1', 'subpanel-m2', 'subpanel-m3', 'subpanel-m4', 'subpanel-m5', 'subpanel-m6']
        for s in expected_subpanels:
            self.assertIn(f'id="{s}"', html, f"Subpanel {s} missing in curriculum tab.")


if __name__ == '__main__':
    unittest.main()
