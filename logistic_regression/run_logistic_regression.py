"""
Module 08: Logistic Regression Baseline
Trains standard L2 regularized Logistic Regression, evaluates test metrics,
and renders the Confusion Matrix and ROC Curve figures.
Run in PyCharm:
    py logistic_regression/run_logistic_regression.py
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_and_clean_data, create_full_pipeline
from src.config import RANDOM_STATE, TEST_SIZE, PALETTE, show_or_save_plot

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "logistic_regression")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_logistic_regression():
    print("=" * 65)
    print("MODULE 08: LOGISTIC REGRESSION")
    print("=" * 65)
    print("[LOGISTIC REGRESSION]")
    print("Training model...")
    X, y, _ = load_and_clean_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    pipeline = create_full_pipeline(clf)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\nActual evaluation results:")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nConfusion Matrix:")
    print("               Predicted No (0)    Predicted Readmitted (1)")
    print(f"Actual No (0):         {cm[0,0]:>6}             {cm[0,1]:>6}")
    print(f"Actual Readmitted (1): {cm[1,0]:>6}             {cm[1,1]:>6}")

    # Plot Confusion Matrix
    print("\n[LOGISTIC REGRESSION] Generating Confusion Matrix Visualization...")
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=PALETTE['SOFT_LINEN'])
    ax.set_facecolor("#FFFFFF")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=ax,
                xticklabels=['Pred 0 (No)', 'Pred 1 (Readmit)'],
                yticklabels=['Actual 0 (No)', 'Actual 1 (Readmit)'])
    ax.set_title("Logistic Regression Confusion Matrix", fontsize=12,
                 fontweight='bold', color=PALETTE['SLATE_DARK'], pad=12)
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    show_or_save_plot(cm_path, plt)

    # Plot ROC Curve
    print("\n[LOGISTIC REGRESSION] Generating ROC Curve Visualization...")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6.5, 5), facecolor=PALETTE['SOFT_LINEN'])
    ax.set_facecolor("#FFFFFF")
    ax.plot(fpr, tpr, color=PALETTE['TERRACOTTA'], lw=2, label=f'Logistic Regression (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color=PALETTE['SLATE_DARK'], linestyle='--', lw=1.2, label='Chance (AUC = 0.50)')
    ax.set_xlabel('False Positive Rate', fontsize=10, color=PALETTE['SLATE_DARK'])
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=10, color=PALETTE['SLATE_DARK'])
    ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=12,
                 fontweight='bold', color=PALETTE['SLATE_DARK'], pad=12)
    ax.legend(loc='lower right', frameon=True)
    ax.grid(alpha=0.3)
    roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    show_or_save_plot(roc_path, plt)

    report_path = os.path.join(OUTPUT_DIR, "logistic_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""============================================================
DIABETES READMISSION — LOGISTIC REGRESSION BASELINE
============================================================
Accuracy:  {acc * 100:.2f}%
Precision: {prec * 100:.2f}%
Recall:    {rec * 100:.2f}%
F1-Score:  {f1:.4f}
ROC-AUC:   {auc:.4f}

Confusion Matrix:
TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}
""")
    print(f"\n[LOGISTIC REGRESSION] Report saved to: {report_path}")
    print("[LOGISTIC REGRESSION] Module completed successfully.\n")


if __name__ == "__main__":
    run_logistic_regression()
