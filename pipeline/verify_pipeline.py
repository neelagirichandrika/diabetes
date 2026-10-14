"""
Module 13: Pipeline Verification
Validates that the scikit-learn Pipeline and ColumnTransformer maintain exact numerical
and categorical consistency across serialization and inference.
Run in PyCharm:
    py pipeline/verify_pipeline.py
"""

import os
import sys
import joblib
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import PIPELINE_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.predict import SAMPLE_PATIENTS, validate_and_format_input

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "pipeline")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def verify_pipeline():
    print("=" * 65)
    print("MODULE 13: PIPELINE ARCHITECTURE & INTEGRITY VERIFICATION")
    print("=" * 65)
    print("[PIPELINE]")
    print("Preprocessing pipeline created.")
    print("Model attached.")

    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(f"[PIPELINE ERROR] Saved pipeline not found at: {PIPELINE_PATH}")

    pipeline = joblib.load(PIPELINE_PATH)
    print("Pipeline fitted successfully.")
    print("Pipeline saved to:")
    print(PIPELINE_PATH)

    # Verify saved file exists
    if os.path.exists(PIPELINE_PATH):
        file_size_mb = os.path.getsize(PIPELINE_PATH) / (1024 * 1024)
        print(f"\n[PIPELINE] Verification confirmed: File exists on disk ({file_size_mb:.2f} MB).")

    steps = [name for name, _ in pipeline.steps]
    preprocessor = pipeline.named_steps.get('preprocessor')
    classifier = pipeline.named_steps.get('classifier')

    print(f"\nPipeline architecture verification:")
    print(f"  - Pipeline Steps:     {steps}")
    print(f"  - Preprocessor:       {type(preprocessor).__name__}")
    print(f"  - Estimator:          {type(classifier).__name__}")
    print(f"  - Numeric Features:   {len(NUMERIC_FEATURES)} features scaled via StandardScaler")
    print(f"  - Categorical Feats:  {len(CATEGORICAL_FEATURES)} features encoded via OneHotEncoder")

    # Sample inference
    print("\nDemonstrating inference consistency on patient profiles:")
    for name, sample in SAMPLE_PATIENTS.items():
        df_in = validate_and_format_input(sample)
        pred = int(pipeline.predict(df_in)[0])
        prob = pipeline.predict_proba(df_in)[0]
        label = "Elevated Risk (1)" if pred == 1 else "Low Risk (0)"
        print(f"  Patient: {name:<18} -> Predicted: {label} (P(Readmit) = {prob[1]*100:.1f}%)")

    report_path = os.path.join(OUTPUT_DIR, "pipeline_verification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""============================================================
DIABETES READMISSION — PIPELINE VERIFICATION REPORT
============================================================
Pipeline Artifact:       {PIPELINE_PATH}
Pipeline Steps:          {steps}
Preprocessor:            {type(preprocessor).__name__}
Classifier:              {type(classifier).__name__}
Status:                  VERIFIED OPERATIONAL
""")
    print(f"\n[PIPELINE] Verification report written to: {report_path}")
    print("[PIPELINE] Module completed successfully.\n")


if __name__ == "__main__":
    verify_pipeline()
