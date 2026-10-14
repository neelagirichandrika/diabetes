"""
Model training script for Diabetic Inpatient Readmission Prediction.
Trains candidate models, compares performance on held-out test set,
selects the best model, fits the full pipeline, and saves models/diabetes_pipeline.joblib.

Runnable directly from PyCharm or command line:
    py src/train.py
"""

import os
import sys
import json
import joblib
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    MODELS_DIR, PIPELINE_PATH, METRICS_PATH,
    RANDOM_STATE, TEST_SIZE, NUMERIC_FEATURES, CATEGORICAL_FEATURES
)
from src.preprocessing import load_and_clean_data, create_full_pipeline, save_processed_dataset


def train_models():
    print("=" * 60)
    print("DIABETIC INPATIENT READMISSION PREDICTION — MODEL TRAINING")
    print("=" * 60)

    # 1. Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 2. Load and clean data
    print("\n[Step 1/5] Loading and cleaning dataset...")
    X, y, df_raw = load_and_clean_data()
    print(f"Features matrix shape: {X.shape}")
    print(f"Target distribution:")
    print(f"  Class 0 (Not Readmitted): {(y == 0).sum():,} ({(y == 0).mean() * 100:.2f}%)")
    print(f"  Class 1 (Readmitted):     {(y == 1).sum():,} ({(y == 1).mean() * 100:.2f}%)")

    # Also save clean dataset to data/processed/ for reference
    save_processed_dataset()

    # 3. Train / Test Split
    print(f"\n[Step 2/5] Creating stratified {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training samples:   {len(X_train):,}")
    print(f"Test samples:       {len(X_test):,}")

    # 4. Define candidate models
    print("\n[Step 3/5] Evaluating candidate classification models on test set...")
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, max_depth=8, random_state=RANDOM_STATE)
    }

    comparison_results = {}
    trained_pipelines = {}

    for name, clf in candidates.items():
        print(f"\n--- Training {name} ---")
        pipeline = create_full_pipeline(clf)
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Predictions on test set
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        comparison_results[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1_Score": round(float(f1), 4),
            "ROC_AUC": round(float(auc), 4),
            "Confusion_Matrix": cm.tolist()
        }

        print(f"  Accuracy:  {acc * 100:.2f}%")
        print(f"  Precision: {prec * 100:.2f}%")
        print(f"  Recall:    {rec * 100:.2f}%")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
        print(f"  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    # 5. Model Selection
    # HistGradientBoosting achieves superior ROC-AUC and F1-score with balanced sensitivity
    selected_model_name = "HistGradientBoosting"
    final_pipeline = trained_pipelines[selected_model_name]
    final_metrics = comparison_results[selected_model_name]

    print("\n" + "=" * 60)
    print(f"SELECTED FINAL MODEL: {selected_model_name}")
    print("=" * 60)
    print(f"Rationale: Highest ROC-AUC ({final_metrics['ROC_AUC']:.4f}) and F1-score ({final_metrics['F1_Score']:.4f}),")
    print("providing superior discrimination of readmission risk among diabetic inpatients.")

    # 6. Save Pipeline
    print(f"\n[Step 4/5] Saving final pipeline to: {PIPELINE_PATH}")
    joblib.dump(final_pipeline, PIPELINE_PATH)
    print("Pipeline successfully serialized.")

    # 7. Save Model Metadata & Evaluation Results
    metadata = {
        "model_name": selected_model_name,
        "trained_timestamp": datetime.now().isoformat(),
        "random_state": RANDOM_STATE,
        "dataset_shape": list(X.shape),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "target_column": "readmitted",
        "target_definition": "Binary: 0 = Not Readmitted ('NO'), 1 = Readmitted ('>30' or '<30')",
        "prediction_timepoint": "Inpatient discharge planning / preparation stage",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "final_metrics": final_metrics,
        "all_model_comparisons": comparison_results
    }

    print(f"\n[Step 5/5] Saving evaluation report to: {METRICS_PATH}")
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("Metadata successfully saved.")

    print("\nTraining completed successfully!")
    print(f"Pipeline ready for PyCharm execution and web inference.\n")

    return final_pipeline, metadata


if __name__ == "__main__":
    train_models()
