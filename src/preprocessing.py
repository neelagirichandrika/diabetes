"""
Preprocessing and feature engineering module for Diabetic Inpatient Readmission Prediction.
Handles ICD-9 clinical grouping, admission categorization, cleaning, and transformer creation.
"""

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    RAW_DATA_PATH, CLEAN_DATA_PATH, EXPIRED_DISPOSITION_IDS,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN, BINARY_TARGET_COLUMN
)


def map_icd9_category(code):
    """
    Groups raw ICD-9 diagnosis codes into standard clinical categories
    commonly used in medical informatics literature (Strack et al., 2014).
    
    Categories:
      - Circulatory: 390-459, 785
      - Respiratory: 460-519, 786
      - Digestive: 520-579, 787
      - Diabetes: 250.xx
      - Injury: 800-999
      - Musculoskeletal: 710-739
      - Genitourinary: 580-629, 788
      - Neoplasms: 140-239
      - Other: all other valid codes and V/E codes
    """
    if pd.isna(code) or code == '?' or code == '':
        return 'Other'
    code_str = str(code).strip()
    if code_str.startswith('V') or code_str.startswith('E'):
        return 'Other'
    try:
        val = float(code_str)
        if (390 <= val <= 459) or val == 785:
            return 'Circulatory'
        elif (460 <= val <= 519) or val == 786:
            return 'Respiratory'
        elif (520 <= val <= 579) or val == 787:
            return 'Digestive'
        elif 250 <= val < 251:
            return 'Diabetes'
        elif 800 <= val <= 999:
            return 'Injury'
        elif 710 <= val <= 739:
            return 'Musculoskeletal'
        elif (580 <= val <= 629) or val == 788:
            return 'Genitourinary'
        elif 140 <= val <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
    except ValueError:
        return 'Other'


def map_admission_type(val):
    """
    Converts admission_type_id numeric codes into readable categories
    based on IDS_mapping.csv.
    """
    v = str(val).strip()
    if v == '1':
        return 'Emergency'
    elif v == '2':
        return 'Urgent'
    elif v == '3':
        return 'Elective'
    elif v == '7':
        return 'Trauma Center'
    else:
        return 'Other/Not Available'


def map_admission_source(val):
    """
    Converts admission_source_id numeric codes into readable categories
    based on IDS_mapping.csv.
    """
    v = str(val).strip()
    if v == '7':
        return 'Emergency Room'
    elif v == '1':
        return 'Physician Referral'
    elif v == '2':
        return 'Clinic Referral'
    elif v in ['4', '5', '6']:
        return 'Transfer from Facility'
    else:
        return 'Other/Unknown'


def check_diabetes_diagnosed(row):
    """Checks if diabetes (ICD-9 250.xx) is listed across any of diag_1, diag_2, or diag_3."""
    for col in ['diag_1', 'diag_2', 'diag_3']:
        val = str(row.get(col, '')).strip()
        if val.startswith('250'):
            return 'Yes'
    return 'No'


def load_and_clean_data(raw_path=RAW_DATA_PATH):
    """
    Loads raw diabetic_data.csv, removes expired cases, handles missing
    entries, engineers clinical features, and produces X and y.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset file not found at: {raw_path}")

    df = pd.read_csv(raw_path)

    # 1. Filter deceased patients (discharge disposition 11, 19, 20, 21)
    # Readmission is logically impossible for expired patients; retaining them
    # creates artificial negative bias (100% have readmitted='NO').
    df = df[~df['discharge_disposition_id'].isin(EXPIRED_DISPOSITION_IDS)].copy()

    # 2. Filter invalid gender records (only 3 in 100k+ rows)
    df = df[df['gender'].isin(['Male', 'Female'])].copy()

    # 3. Clinical feature engineering
    df['primary_diagnosis'] = df['diag_1'].apply(map_icd9_category)
    df['diabetes_diagnosed'] = df.apply(check_diabetes_diagnosed, axis=1)
    df['admission_type'] = df['admission_type_id'].apply(map_admission_type)
    df['admission_source'] = df['admission_source_id'].apply(map_admission_source)

    # 4. Clean numeric features (fill missing/invalid with 0)
    for num_col in NUMERIC_FEATURES:
        df[num_col] = pd.to_numeric(df[num_col], errors='coerce').fillna(0)

    # 5. Clean categorical features
    for cat_col in CATEGORICAL_FEATURES:
        df[cat_col] = df[cat_col].replace('?', 'Unknown').fillna('Unknown').astype(str)

    # 6. Binary target:
    # 0 = Not Readmitted (raw 'NO')
    # 1 = Readmitted (raw '>30' or '<30')
    df[BINARY_TARGET_COLUMN] = df[TARGET_COLUMN].isin(['>30', '<30']).astype(int)

    # Return clean feature matrix X and target y
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[BINARY_TARGET_COLUMN].copy()

    return X, y, df


def build_preprocessor():
    """
    Constructs a ColumnTransformer that scales numerical features with StandardScaler
    and one-hot encodes categorical features.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor


def create_full_pipeline(estimator):
    """
    Wraps the feature preprocessor and classification estimator into a single Pipeline.
    Ensures identical preprocessing during training and prediction.
    """
    preprocessor = build_preprocessor()
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', estimator)
    ])
    return pipeline


def save_processed_dataset():
    """Utility to process and save clean dataset to data/processed/diabetes_clean.csv."""
    os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
    X, y, df_clean = load_and_clean_data()
    clean_export = X.copy()
    clean_export[BINARY_TARGET_COLUMN] = y
    clean_export.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Cleaned dataset saved to: {CLEAN_DATA_PATH} (Shape: {clean_export.shape})")
    return clean_export


if __name__ == '__main__':
    save_processed_dataset()
