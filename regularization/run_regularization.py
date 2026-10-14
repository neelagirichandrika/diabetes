"""
Module 10: Regularization Analysis
Investigates L1 (Lasso) vs L2 (Ridge) penalty across varying C regularization strengths.
Run in PyCharm:
    py regularization/run_regularization.py
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_and_clean_data, create_full_pipeline
from src.config import RANDOM_STATE, TEST_SIZE, PALETTE, show_or_save_plot

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "regularization")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_regularization():
    print("=" * 65)
    print("MODULE 10: REGULARIZATION ANALYSIS (L1 vs L2)")
    print("=" * 65)
    print("[REGULARIZATION]")
    print("Testing L1 (Lasso) vs L2 (Ridge) across inverse regularization strength C = [0.001, 0.01, 0.1, 1.0, 10.0]...")

    X, y, _ = load_and_clean_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    c_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    records = []

    print("\nActual calculated results:\n")
    for c in c_values:
        # L2 (Ridge)
        clf_l2 = LogisticRegression(penalty='l2', C=c, solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE)
        pipe_l2 = create_full_pipeline(clf_l2)
        pipe_l2.fit(X_train, y_train)
        pred_l2 = pipe_l2.predict(X_test)
        auc_l2 = roc_auc_score(y_test, pipe_l2.predict_proba(X_test)[:, 1])
        acc_l2 = accuracy_score(y_test, pred_l2)
        f1_l2 = f1_score(y_test, pred_l2)

        # L1 (Lasso with liblinear)
        clf_l1 = LogisticRegression(penalty='l1', C=c, solver='liblinear', max_iter=500, random_state=RANDOM_STATE)
        pipe_l1 = create_full_pipeline(clf_l1)
        pipe_l1.fit(X_train, y_train)
        pred_l1 = pipe_l1.predict(X_test)
        auc_l1 = roc_auc_score(y_test, pipe_l1.predict_proba(X_test)[:, 1])
        acc_l1 = accuracy_score(y_test, pred_l1)
        f1_l1 = f1_score(y_test, pred_l1)
        non_zero_l1 = int(np.count_nonzero(clf_l1.coef_))

        print(f"C = {c}")
        print(f"L1 -> F1 = {f1_l1:.4f}  (ROC-AUC = {auc_l1:.4f}, Non-Zero Coefs = {non_zero_l1})")
        print(f"L2 -> F1 = {f1_l2:.4f}  (ROC-AUC = {auc_l2:.4f})\n")

        records.append({
            'C': c,
            'L2_Acc': round(acc_l2 * 100, 2),
            'L2_F1': round(f1_l2, 4),
            'L2_AUC': round(auc_l2, 4),
            'L1_Acc': round(acc_l1 * 100, 2),
            'L1_F1': round(f1_l1, 4),
            'L1_AUC': round(auc_l1, 4),
            'L1_NonZero_Coefs': non_zero_l1
        })

    df_reg = pd.DataFrame(records).set_index('C')

    # Plot
    print("[REGULARIZATION] Generating Regularization Comparison Plot...")
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=PALETTE['SOFT_LINEN'])
    ax.set_facecolor("#FFFFFF")
    ax.plot([str(c) for c in c_values], df_reg['L2_AUC'], marker='o',
            color=PALETTE['TERRACOTTA'], lw=2, label='L2 (Ridge) ROC-AUC')
    ax.plot([str(c) for c in c_values], df_reg['L1_AUC'], marker='s',
            color=PALETTE['SLATE_DARK'], lw=2, label='L1 (Lasso) ROC-AUC')
    ax.set_title("Regularization Strength (C) vs ROC-AUC",
                 fontsize=12, fontweight='bold', color=PALETTE['SLATE_DARK'], pad=12)
    ax.set_xlabel("Regularization Parameter C (Lower = Stronger Regularization)", fontsize=10, color=PALETTE['SLATE_DARK'])
    ax.set_ylabel("ROC-AUC Score", fontsize=10, color=PALETTE['SLATE_DARK'])
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)
    
    plot_path = os.path.join(OUTPUT_DIR, "regularization_comparison.png")
    show_or_save_plot(plot_path, plt)

    report_path = os.path.join(OUTPUT_DIR, "regularization_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""============================================================
DIABETES READMISSION — REGULARIZATION COMPARISON REPORT
============================================================
{df_reg.to_string()}
""")
    print(f"\n[REGULARIZATION] Report saved to: {report_path}")
    print("[REGULARIZATION] Module completed successfully.\n")


if __name__ == "__main__":
    run_regularization()
