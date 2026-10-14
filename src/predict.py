"""
Prediction inference module for Diabetic Inpatient Readmission Prediction.
Loads saved pipeline from models/diabetes_pipeline.joblib and accepts
user inputs to generate predictions and estimated readmission probabilities.

Runnable directly from PyCharm:
    py src/predict.py
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    PIPELINE_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES
)

_cached_pipeline = None


def get_pipeline():
    """Returns the cached pipeline or loads it from disk if not yet in memory."""
    global _cached_pipeline
    if _cached_pipeline is None:
        if not os.path.exists(PIPELINE_PATH):
            raise FileNotFoundError(
                f"Model pipeline not found at {PIPELINE_PATH}. "
                "Please run 'src/train.py' first to build and save the model."
            )
        _cached_pipeline = joblib.load(PIPELINE_PATH)
    return _cached_pipeline


def validate_and_format_input(input_dict):
    """
    Validates user inputs from the web form and formats them into a single-row
    DataFrame adhering to the exact feature requirements and dtypes of the pipeline.
    """
    row = {}

    # Numeric features with bounds validation
    bounds = {
        'time_in_hospital': (1, 14, 3),
        'num_lab_procedures': (1, 132, 43),
        'num_procedures': (0, 6, 1),
        'num_medications': (1, 81, 15),
        'number_outpatient': (0, 50, 0),
        'number_emergency': (0, 80, 0),
        'number_inpatient': (0, 30, 0),
        'number_diagnoses': (1, 16, 7)
    }

    for feat in NUMERIC_FEATURES:
        min_v, max_v, default_v = bounds.get(feat, (0, 999, 0))
        raw_val = input_dict.get(feat)
        try:
            val = float(raw_val)
            val = max(min_v, min(max_v, val))
        except (ValueError, TypeError):
            val = float(default_v)
        row[feat] = val

    # Categorical features with fallbacks
    for feat in CATEGORICAL_FEATURES:
        raw_val = input_dict.get(feat)
        if raw_val is None or str(raw_val).strip() == '':
            row[feat] = 'None' if feat in ['max_glu_serum', 'A1Cresult'] else 'Unknown'
        else:
            row[feat] = str(raw_val).strip()

    df = pd.DataFrame([row])
    df = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    return df


def predict_patient(input_dict):
    """
    Predicts readmission risk for a single patient record.
    Returns:
      dict with prediction (0/1), label, probabilities, and context.
    """
    pipeline = get_pipeline()
    X_input = validate_and_format_input(input_dict)

    prediction = int(pipeline.predict(X_input)[0])
    probabilities = pipeline.predict_proba(X_input)[0]

    prob_not_readmitted = float(probabilities[0])
    prob_readmitted = float(probabilities[1])

    if prediction == 1:
        risk_level = "Elevated Risk of Readmission"
        risk_class = "warning"
        summary = (
            f"Based on the supplied information, the model predicts an elevated likelihood "
            f"of subsequent readmission ({prob_readmitted * 100:.1f}% estimated probability). "
            f"This patient profile shares clinical characteristics with diabetic inpatients who experienced "
            f"subsequent readmission. Enhanced post-discharge transitional care is recommended."
        )
    else:
        risk_level = "Baseline / Lower Risk of Readmission"
        risk_class = "normal"
        summary = (
            f"Based on the supplied information, the model predicts a lower likelihood "
            f"of subsequent readmission ({prob_readmitted * 100:.1f}% estimated readmission probability; "
            f"{prob_not_readmitted * 100:.1f}% probability of no readmission). "
            f"Standard outpatient follow-up and routine diabetes care recommended."
        )

    return {
        "prediction": prediction,
        "risk_level": risk_level,
        "risk_class": risk_class,
        "readmission_probability": round(prob_readmitted * 100, 1),
        "non_readmission_probability": round(prob_not_readmitted * 100, 1),
        "summary": summary,
        "disclaimer": (
            "This application provides a machine-learning prediction based on the supplied dataset "
            "and is intended for educational purposes. It is not a medical diagnosis and should not "
            "replace advice from a qualified healthcare professional."
        )
    }


# Realistic patient profiles derived directly from diabetic_data.csv patterns
SAMPLE_PATIENTS = {
    "sample_baseline": {
        "gender": "Female",
        "age": "[40-50)",
        "admission_type": "Elective",
        "admission_source": "Physician Referral",
        "time_in_hospital": 2,
        "num_lab_procedures": 25,
        "num_procedures": 1,
        "num_medications": 8,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "number_diagnoses": 4,
        "primary_diagnosis": "Digestive",
        "diabetes_diagnosed": "No",
        "max_glu_serum": "None",
        "A1Cresult": "Norm",
        "metformin": "Steady",
        "glipizide": "No",
        "glyburide": "No",
        "insulin": "No",
        "change": "No",
        "diabetesMed": "Yes"
    },
    "sample_elevated": {
        "gender": "Male",
        "age": "[70-80)",
        "admission_type": "Emergency",
        "admission_source": "Emergency Room",
        "time_in_hospital": 8,
        "num_lab_procedures": 68,
        "num_procedures": 2,
        "num_medications": 26,
        "number_outpatient": 1,
        "number_emergency": 2,
        "number_inpatient": 3,
        "number_diagnoses": 9,
        "primary_diagnosis": "Circulatory",
        "diabetes_diagnosed": "Yes",
        "max_glu_serum": ">200",
        "A1Cresult": ">8",
        "metformin": "No",
        "glipizide": "Steady",
        "glyburide": "No",
        "insulin": "Up",
        "change": "Ch",
        "diabetesMed": "Yes"
    }
}


def run_prediction_demo():
    print("=" * 65)
    print("PREDICTION INFERENCE DEMONSTRATION")
    print("=" * 65)
    get_pipeline()

    sample = SAMPLE_PATIENTS["sample_elevated"]
    print("[PREDICTION]")
    print("Input:")
    for k, v in sample.items():
        print(f"  {k}: {v}")

    res = predict_patient(sample)
    print("\nPrediction:")
    pred_text = "Readmitted (1)" if res['prediction'] == 1 else "Not Readmitted (0)"
    print(f"  {pred_text}")

    print("\nProbability:")
    print(f"  Readmission Risk: {res['readmission_probability']}%")
    print(f"  No Readmission:   {res['non_readmission_probability']}%")

    print(f"\nSummary:\n  {res['summary']}")
    print(f"\nDisclaimer:\n  {res['disclaimer']}\n")


if __name__ == "__main__":
    run_prediction_demo()
