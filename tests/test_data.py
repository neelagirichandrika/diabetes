"""
Unit tests for data loading, data cleaning, and feature engineering.
Run with:
    py -m unittest tests/test_data.py
"""

import unittest
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import RAW_DATA_PATH, EXPIRED_DISPOSITION_IDS
from src.preprocessing import load_and_clean_data, map_icd9_category


class TestDataProcessing(unittest.TestCase):

    def test_raw_file_exists(self):
        self.assertTrue(os.path.exists(RAW_DATA_PATH), f"Raw dataset not found at {RAW_DATA_PATH}")

    def test_clean_data_loading(self):
        X, y, df = load_and_clean_data()
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(X), 90000)
        self.assertEqual(X.shape[1], 22)

    def test_expired_patients_filtered(self):
        _, _, df = load_and_clean_data()
        for exp_id in EXPIRED_DISPOSITION_IDS:
            self.assertFalse(df['discharge_disposition_id'].isin([exp_id]).any(),
                             f"Found expired disposition {exp_id} in cleaned dataset!")

    def test_icd9_mapping(self):
        self.assertEqual(map_icd9_category('250.01'), 'Diabetes')
        self.assertEqual(map_icd9_category('410'), 'Circulatory')
        self.assertEqual(map_icd9_category('486'), 'Respiratory')
        self.assertEqual(map_icd9_category('540'), 'Digestive')
        self.assertEqual(map_icd9_category('V57'), 'Other')
        self.assertEqual(map_icd9_category('?'), 'Other')

    def test_target_values(self):
        _, y, _ = load_and_clean_data()
        unique_targets = set(y.unique())
        self.assertEqual(unique_targets, {0, 1})


if __name__ == '__main__':
    unittest.main()
