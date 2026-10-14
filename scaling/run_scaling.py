"""
Module 05: Feature Scaling
Standardizes numerical clinical features and prints before/after comparison table and sample.
Run in PyCharm:
    py scaling/run_scaling.py
"""

import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_and_clean_data
from src.config import NUMERIC_FEATURES, PALETTE, show_or_save_plot

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scaling")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_scaling():
    print("=" * 65)
    print("MODULE 05: NUMERICAL FEATURE SCALING")
    print("=" * 65)
    X, _, _ = load_and_clean_data()
    X_num = X[NUMERIC_FEATURES].copy()

    print("[SCALING] Numerical features:")
    for feat in NUMERIC_FEATURES:
        print(f"- {feat} (min: {X_num[feat].min():.0f}, max: {X_num[feat].max():.0f})")

    print("\n[SCALING] Scaling method:")
    print("StandardScaler: z = (x - mean) / std (zero mean, unit variance)")

    stats_before = X_num.describe().T[['mean', 'std', 'min', 'max']].round(2)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=NUMERIC_FEATURES)
    stats_after = X_scaled.describe().T[['mean', 'std', 'min', 'max']].round(2)

    print("\nStatistical comparison before vs after scaling:")
    comp_df = pd.DataFrame({
        'Mean (Raw)': stats_before['mean'],
        'Std (Raw)': stats_before['std'],
        'Mean (Scaled)': stats_after['mean'],
        'Std (Scaled)': stats_after['std'],
        'Min (Scaled)': stats_after['min'],
        'Max (Scaled)': stats_after['max']
    })
    pd.set_option('display.width', 120)
    print(comp_df)

    print("\nSample of scaled data (first 5 rows):")
    print(X_scaled.head())

    # Visualizations
    print("\n[SCALING] Generating Scaling Distributions Plot...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), facecolor=PALETTE['SOFT_LINEN'])
    features_to_plot = ['num_lab_procedures', 'num_medications']
    
    for i, feat in enumerate(features_to_plot):
        # Raw
        axes[i, 0].set_facecolor("#FFFFFF")
        axes[i, 0].hist(X_num[feat], bins=30, color=PALETTE['TERRACOTTA'], edgecolor=PALETTE['WARM_SAND'], alpha=0.9)
        axes[i, 0].set_title(f"Raw: {feat} (mean={stats_before.loc[feat, 'mean']:.1f}, std={stats_before.loc[feat, 'std']:.1f})",
                             fontsize=9.5, color=PALETTE['SLATE_DARK'], fontweight='bold')
        axes[i, 0].grid(axis='y', linestyle='--', alpha=0.3)

        # Scaled
        axes[i, 1].set_facecolor("#FFFFFF")
        axes[i, 1].hist(X_scaled[feat], bins=30, color=PALETTE['SAGE_GREEN'], edgecolor=PALETTE['WARM_SAND'], alpha=0.9)
        axes[i, 1].set_title(f"Scaled: {feat} (mean=0.00, std=1.00)",
                             fontsize=9.5, color=PALETTE['SLATE_DARK'], fontweight='bold')
        axes[i, 1].grid(axis='y', linestyle='--', alpha=0.3)

    plt.suptitle("Numerical Feature Distributions: Raw vs StandardScaler",
                 fontsize=12, fontweight='bold', color=PALETTE['SLATE_DARK'], y=0.98)
    plot_path = os.path.join(OUTPUT_DIR, "numeric_features_before_after.png")
    show_or_save_plot(plot_path, plt)

    report_path = os.path.join(OUTPUT_DIR, "scaling_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""============================================================
DIABETES READMISSION — FEATURE SCALING REPORT
============================================================
Method: StandardScaler (z = (x - mean) / std)

Comparison Table:
{comp_df.to_string()}

Scaled Preview:
{X_scaled.head().to_string()}
""")
    print(f"\n[SCALING] Report saved to: {report_path}")
    print("[SCALING] Module completed successfully.\n")


if __name__ == "__main__":
    run_scaling()
