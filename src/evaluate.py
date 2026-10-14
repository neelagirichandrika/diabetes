"""
Module: Standalone Evaluation
Evaluates models/diabetes_pipeline.joblib on the held-out test partition,
prints classification metrics, and displays the confusion matrix plot.
Run in PyCharm:
    py src/evaluate.py
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import PIPELINE_PATH, RANDOM_STATE, TEST_SIZE, PALETTE, show_or_save_plot
from src.preprocessing import load_and_clean_data

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")


def evaluate_saved_model():
    print("=" * 65)
    print("MODULE: STANDALONE MODEL EVALUATION")
    print("=" * 65)
    print(f"[EVALUATION] Loading pipeline from: {PIPELINE_PATH}")

    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(f"[EVALUATION ERROR] Saved pipeline not found at {PIPELINE_PATH}.")

    pipeline = joblib.load(PIPELINE_PATH)

    print("[EVALUATION] Re-creating identical held-out test partition...")
    X, y, _ = load_and_clean_data()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[EVALUATION] Evaluating on {len(X_test):,} held-out patient encounters...")

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\nActual test evaluation metrics:")
    print(f"- Accuracy:  {acc * 100:.2f}%")
    print(f"- Precision: {prec * 100:.2f}%")
    print(f"- Recall:    {rec * 100:.2f}%")
    print(f"- F1-Score:  {f1:.4f}")
    print(f"- ROC-AUC:   {auc:.4f}")

    print("\n- Confusion Matrix:")
    print("               Predicted Not Readmitted    Predicted Readmitted")
    print(f"Actual Not Readmitted:   {cm[0,0]:>8}                {cm[0,1]:>8}")
    print(f"Actual Readmitted:       {cm[1,0]:>8}                {cm[1,1]:>8}")

    print("\n- Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Readmitted (0)", "Readmitted (1)"]))

    # Visual Confusion Matrix
    print("\n[EVALUATION] Generating Confusion Matrix Visualization...")
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=PALETTE['SOFT_LINEN'])
    ax.set_facecolor("#FFFFFF")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=ax,
                xticklabels=['Pred 0 (No)', 'Pred 1 (Readmit)'],
                yticklabels=['Actual 0 (No)', 'Actual 1 (Readmit)'])
    ax.set_title("Final Model Confusion Matrix (Test Set)", fontsize=12,
                 fontweight='bold', color=PALETTE['SLATE_DARK'], pad=12)
    
    cm_plot_path = os.path.join(OUTPUT_DIR, "final_confusion_matrix.png")
    show_or_save_plot(cm_plot_path, plt)
    print(f"[EVALUATION] Confusion matrix saved to: {cm_plot_path}")
    print("[EVALUATION] Module completed successfully.\n")


if __name__ == "__main__":
    evaluate_saved_model()
