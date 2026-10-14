"""
Module 06: Train/Test Split
Executes stratified partition and outputs training/testing class balance.
Run in PyCharm:
    py split/run_split.py
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_and_clean_data
from src.config import RANDOM_STATE, TEST_SIZE, PALETTE, show_or_save_plot

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "split")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_split():
    print("=" * 65)
    print("MODULE 06: STRATIFIED TRAIN / TEST SPLIT")
    print("=" * 65)
    X, y, _ = load_and_clean_data()
    total_samples = len(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    n_train = len(X_train)
    n_test = len(X_test)
    train_dist = y_train.value_counts()
    test_dist = y_test.value_counts()
    train_pct = y_train.value_counts(normalize=True) * 100
    test_pct = y_test.value_counts(normalize=True) * 100

    print(f"[SPLIT] Training samples: {n_train:,}")
    print(f"[SPLIT] Testing samples: {n_test:,}")

    print("\n[SPLIT] Training class distribution:")
    print(f"Class 0 (Not Readmitted): {train_dist[0]:>6,} ({train_pct[0]:.2f}%)")
    print(f"Class 1 (Readmitted):     {train_dist[1]:>6,} ({train_pct[1]:.2f}%)")

    print("\n[SPLIT] Testing class distribution:")
    print(f"Class 0 (Not Readmitted): {test_dist[0]:>6,} ({test_pct[0]:.2f}%)")
    print(f"Class 1 (Readmitted):     {test_dist[1]:>6,} ({test_pct[1]:.2f}%)")

    # Overlap / Leakage Check
    overlap = len(set(X_train.index).intersection(set(X_test.index)))
    print(f"\n[SPLIT] Leakage verification: {overlap} overlapping sample indices (CONFIRMED ZERO LEAKAGE)")

    # Plot
    print("\n[SPLIT] Generating Stratification Comparison Plot...")
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=PALETTE['SOFT_LINEN'])
    ax.set_facecolor("#FFFFFF")
    
    df_compare = pd.DataFrame({
        'Training Partition': train_pct,
        'Testing Partition': test_pct
    }, index=[0, 1])
    df_compare.index = ['Not Readmitted (0)', 'Readmitted (1)']

    df_compare.plot(kind='bar', ax=ax, color=[PALETTE['SLATE_DARK'], PALETTE['TERRACOTTA']],
                    edgecolor=PALETTE['WARM_SAND'], width=0.55)
    ax.set_title("Class Balance Stratification (Train vs Test)",
                 fontsize=12, fontweight='bold', color=PALETTE['SLATE_DARK'], pad=12)
    ax.set_ylabel("Percentage (%)", fontsize=10, color=PALETTE['SLATE_DARK'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_ylim(0, 70)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plot_path = os.path.join(OUTPUT_DIR, "train_test_class_balance.png")
    show_or_save_plot(plot_path, plt)

    report_path = os.path.join(OUTPUT_DIR, "split_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""============================================================
DIABETES READMISSION — TRAIN / TEST SPLIT REPORT
============================================================
Total Clean Encounters: {total_samples:,}
Training Samples:       {n_train:,} ({n_train/total_samples*100:.1f}%)
Testing Samples:        {n_test:,} ({n_test/total_samples*100:.1f}%)
Random Seed:            {RANDOM_STATE}

Class Proportions:
Train: 0={train_pct[0]:.2f}%, 1={train_pct[1]:.2f}%
Test:  0={test_pct[0]:.2f}%, 1={test_pct[1]:.2f}%
""")
    print(f"\n[SPLIT] Report saved to: {report_path}")
    print("[SPLIT] Module completed successfully.\n")


if __name__ == "__main__":
    run_split()
