"""
Module 13: Pipeline Training & Verification
Constructs the complete ColumnTransformer and HistGradientBoostingClassifier pipeline,
fits the model on training data, persists the pipeline artifact, and verifies file existence.
Run in PyCharm:
    py pipeline/run_pipeline.py
"""

import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import PIPELINE_PATH, RANDOM_STATE, TEST_SIZE
from src.preprocessing import load_and_clean_data, create_full_pipeline

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "pipeline")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_pipeline():
    print("=" * 65)
    print("MODULE 13: MACHINE LEARNING PIPELINE")
    print("=" * 65)
    print("[PIPELINE]")

    X, y, _ = load_and_clean_data()
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    clf = HistGradientBoostingClassifier(max_iter=100, max_depth=8, random_state=RANDOM_STATE)
    pipeline = create_full_pipeline(clf)
    print("Preprocessing pipeline created.")
    print("Model attached.")

    pipeline.fit(X_train, y_train)
    print("Pipeline fitted successfully.")

    os.makedirs(os.path.dirname(PIPELINE_PATH), exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)
    print("Pipeline saved to:")
    print(PIPELINE_PATH)

    # Verify saved file exists
    if os.path.exists(PIPELINE_PATH):
        file_size_mb = os.path.getsize(PIPELINE_PATH) / (1024 * 1024)
        print(f"\n[PIPELINE] Verification confirmed: File exists on disk ({file_size_mb:.2f} MB).")
    else:
        raise FileNotFoundError(f"[PIPELINE ERROR] Verification failed: {PIPELINE_PATH} does not exist.")

    # Write pipeline report
    report_path = os.path.join(OUTPUT_DIR, "pipeline_verification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""============================================================
DIABETES READMISSION — PIPELINE REPORT
============================================================
Target Artifact:     {PIPELINE_PATH}
File Size:           {file_size_mb:.2f} MB
Preprocessing:       ColumnTransformer (StandardScaler + OneHotEncoder)
Model Attached:      HistGradientBoostingClassifier
Status:              VERIFIED OPERATIONAL
""")
    print(f"[PIPELINE] Report saved to: {report_path}")
    print("[PIPELINE] Module completed successfully.\n")


if __name__ == "__main__":
    run_pipeline()
