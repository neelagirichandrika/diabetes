"""
Unit tests for pipeline loading, inference formatting, and sample predictions.
Run with:
    py -m unittest tests/test_pipeline.py
"""

import unittest
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import PIPELINE_PATH
from src.predict import get_pipeline, predict_patient, SAMPLE_PATIENTS, validate_and_format_input


class TestPipelineInference(unittest.TestCase):

    def test_pipeline_file_exists(self):
        self.assertTrue(os.path.exists(PIPELINE_PATH), f"Pipeline not found at {PIPELINE_PATH}")

    def test_pipeline_loading(self):
        pipeline = get_pipeline()
        self.assertIsNotNone(pipeline)
        self.assertTrue(hasattr(pipeline, 'predict'))
        self.assertTrue(hasattr(pipeline, 'predict_proba'))

    def test_validate_and_format_input(self):
        sample = SAMPLE_PATIENTS['sample_baseline']
        df_formatted = validate_and_format_input(sample)
        self.assertEqual(len(df_formatted), 1)
        self.assertEqual(df_formatted.shape[1], 22)

    def test_baseline_patient_prediction(self):
        result = predict_patient(SAMPLE_PATIENTS['sample_baseline'])
        self.assertIn(result['prediction'], [0, 1])
        self.assertGreaterEqual(result['readmission_probability'], 0.0)
        self.assertLessEqual(result['readmission_probability'], 100.0)
        self.assertIn('risk_level', result)
        self.assertIn('disclaimer', result)

    def test_elevated_patient_prediction(self):
        result = predict_patient(SAMPLE_PATIENTS['sample_elevated'])
        self.assertIn(result['prediction'], [0, 1])
        self.assertGreaterEqual(result['readmission_probability'], 0.0)
        self.assertLessEqual(result['readmission_probability'], 100.0)


if __name__ == '__main__':
    unittest.main()
