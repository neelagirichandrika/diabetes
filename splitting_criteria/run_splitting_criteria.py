"""
Module 07: Decision Tree Splitting Criteria
Investigates Gini Impurity vs Entropy (Information Gain) for tree splitting on the diabetes dataset.
Run in PyCharm:
    py splitting_criteria/run_splitting_criteria.py
"""

import os
import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_and_clean_data, create_full_pipeline
from src.config import RANDOM_STATE, TEST_SIZE, PALETTE, show_or_save_plot

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "splitting_criteria")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_splitting_criteria():
    print("=" * 65)
    print("MODULE 07: DECISION TREE SPLITTING CRITERIA EXPERIMENT")
    print("=" * 65)
    print("[SPLITTING CRITERIA] Loading partitioned data...")
    X, y, _ = load_and_clean_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    criteria = ['gini', 'entropy', 'log_loss']
    results = []

    print("[SPLITTING CRITERIA] Comparing Gini Impurity vs Entropy/Log-Loss (Depth=6)...")
    for crit in criteria:
        t0 = time.time()
        clf = DecisionTreeClassifier(criterion=crit, max_depth=6, random_state=RANDOM_STATE)
        pipeline = create_full_pipeline(clf)
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - t0

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results.append({
            'Criterion': crit.upper(),
            'Accuracy': round(acc * 100, 2),
            'F1_Score': round(f1, 4),
            'ROC_AUC': round(auc, 4),
            'Train_Time_s': round(fit_time, 2)
        })

    df_res = pd.DataFrame(results).set_index('Criterion')
    print("\nActual splitting criteria comparison results:")
    print(df_res)

    # Plot
    print("\n[SPLITTING CRITERIA] Generating Splitting Criteria Comparison Plot...")
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=PALETTE['SOFT_LINEN'])
    ax.set_facecolor("#FFFFFF")
    df_res[['Accuracy', 'ROC_AUC']].plot(kind='bar', ax=ax,
                                         color=[PALETTE['TERRACOTTA'], PALETTE['SLATE_DARK']],
                                         edgecolor=PALETTE['WARM_SAND'], width=0.55)
    ax.set_title("Decision Tree Splitting Criteria Comparison (Depth=6)",
                 fontsize=12, fontweight='bold', color=PALETTE['SLATE_DARK'], pad=12)
    ax.set_ylabel("Score", fontsize=10, color=PALETTE['SLATE_DARK'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_ylim(0, 80)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plot_path = os.path.join(OUTPUT_DIR, "gini_vs_entropy_comparison.png")
    show_or_save_plot(plot_path, plt)

    report_path = os.path.join(OUTPUT_DIR, "splitting_criteria_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""============================================================
DIABETES READMISSION — SPLITTING CRITERIA EXPERIMENT
============================================================
{df_res.to_string()}
""")
    print(f"\n[SPLITTING CRITERIA] Report saved to: {report_path}")
    print("[SPLITTING CRITERIA] Module completed successfully.\n")


if __name__ == "__main__":
    run_splitting_criteria()
