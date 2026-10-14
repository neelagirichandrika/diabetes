"""
Module 11: Bias-Variance Analysis
Demonstrates underfitting (high bias) vs overfitting (high variance) across model complexity regimes.
Run in PyCharm:
    py bias_variance/run_bias_variance.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_and_clean_data, create_full_pipeline
from src.config import RANDOM_STATE, TEST_SIZE, PALETTE, show_or_save_plot

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "bias_variance")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_bias_variance():
    print("=" * 65)
    print("MODULE 11: BIAS-VARIANCE ANALYSIS")
    print("=" * 65)
    print("[BIAS-VARIANCE]")
    print("Analyzing underfitting vs overfitting across complexity regimes...")

    X, y, _ = load_and_clean_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    regimes = [
        ('Shallow Tree (Depth 2)', 2, 'High Bias (Underfitting)'),
        ('Moderate Tree (Depth 6)', 6, 'Balanced Regime'),
        ('Deep Tree (Depth 12)', 12, 'Moderate Variance'),
        ('Unconstrained Tree (Depth 22)', 22, 'High Variance (Overfitting)')
    ]

    records = []
    print("\nModel complexity                  | Training score | Validation score | Generalization Gap")
    print("-" * 80)
    for label, depth, diagnosis in regimes:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        pipe = create_full_pipeline(clf)
        pipe.fit(X_train, y_train)

        train_score = accuracy_score(y_train, pipe.predict(X_train)) * 100
        test_score = accuracy_score(y_test, pipe.predict(X_test)) * 100
        gap = train_score - test_score

        print(f"{label:<33} | {train_score:>12.2f}% | {test_score:>14.2f}% | {gap:>16.2f}% ({diagnosis})")

        records.append({
            'Model Complexity': label,
            'Diagnosis': diagnosis,
            'Training Score (%)': round(train_score, 2),
            'Validation Score (%)': round(test_score, 2),
            'Generalization Gap (%)': round(gap, 2)
        })

    df_bv = pd.DataFrame(records).set_index('Model Complexity')

    # Plot
    print("\n[BIAS-VARIANCE] Generating Bias-Variance Learning Curves...")
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=PALETTE['SOFT_LINEN'])
    ax.set_facecolor("#FFFFFF")
    x_pos = np.arange(len(records))
    w = 0.35
    ax.bar(x_pos - w/2, [r['Training Score (%)'] for r in records], w,
           label='Training Score', color=PALETTE['TERRACOTTA'], edgecolor=PALETTE['WARM_SAND'])
    ax.bar(x_pos + w/2, [r['Validation Score (%)'] for r in records], w,
           label='Validation Score', color=PALETTE['SLATE_DARK'], edgecolor=PALETTE['WARM_SAND'])
    
    ax.set_title("Bias-Variance Tradeoff Analysis across Model Complexity",
                 fontsize=12, fontweight='bold', color=PALETTE['SLATE_DARK'], pad=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r['Model Complexity'].split('(')[0].strip() for r in records], fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=10, color=PALETTE['SLATE_DARK'])
    ax.set_ylim(45, 95)
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plot_path = os.path.join(OUTPUT_DIR, "learning_curves.png")
    show_or_save_plot(plot_path, plt)

    report_path = os.path.join(OUTPUT_DIR, "bias_variance_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""============================================================
DIABETES READMISSION — BIAS-VARIANCE EXPERIMENT REPORT
============================================================
{df_bv.to_string()}
""")
    print(f"\n[BIAS-VARIANCE] Report saved to: {report_path}")
    print("[BIAS-VARIANCE] Module completed successfully.\n")


if __name__ == "__main__":
    run_bias_variance()
