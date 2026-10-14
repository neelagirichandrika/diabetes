"""
Module 12: Coefficient Interpretation
Extracts and displays Logistic Regression weights and Odds Ratios to identify clinical risk drivers.
Run in PyCharm:
    py coefficients/run_coefficients.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_and_clean_data, build_preprocessor
from src.config import RANDOM_STATE, TEST_SIZE, NUMERIC_FEATURES, CATEGORICAL_FEATURES, PALETTE, show_or_save_plot

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "coefficients")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_coefficients():
    print("=" * 65)
    print("MODULE 12: FEATURE COEFFICIENT & ODDS RATIO ANALYSIS")
    print("=" * 65)
    print("[COEFFICIENTS]")
    print("Extracting and analyzing feature weights from Logistic Regression...")

    X, y, _ = load_and_clean_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train_trans, y_train)

    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES)
    all_feature_names = NUMERIC_FEATURES + list(cat_names)

    coefs = clf.coef_[0]
    odds_ratios = np.exp(coefs)

    df_coef = pd.DataFrame({
        'Feature': all_feature_names,
        'Coefficient': coefs.round(4),
        'Odds_Ratio': odds_ratios.round(4)
    }).sort_values(by='Coefficient', ascending=False)

    top_risk = df_coef.head(10)
    top_protective = df_coef.tail(10)

    print("\nFeature                                  Coefficient")
    print("-" * 60)
    print("Top 10 Risk-Increasing Features (Higher Readmission Likelihood):")
    for _, r in top_risk.iterrows():
        print(f"{r['Feature']:<40} {r['Coefficient']:>12.4f}")

    print("\nTop 10 Risk-Decreasing Features (Lower Readmission Likelihood):")
    for _, r in top_protective.iterrows():
        print(f"{r['Feature']:<40} {r['Coefficient']:>12.4f}")

    # Plot
    print("\n[COEFFICIENTS] Generating Coefficients Horizontal Bar Chart...")
    top_combined = pd.concat([top_risk, top_protective]).sort_values(by='Coefficient', ascending=True)
    
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=PALETTE['SOFT_LINEN'])
    ax.set_facecolor("#FFFFFF")
    colors = [PALETTE['SAGE_GREEN'] if c < 0 else PALETTE['TERRACOTTA'] for c in top_combined['Coefficient']]
    ax.barh(top_combined['Feature'], top_combined['Coefficient'], color=colors,
            edgecolor=PALETTE['WARM_SAND'], height=0.6)
    
    ax.axvline(0, color=PALETTE['SLATE_DARK'], lw=0.8, linestyle='--')
    ax.set_title("Top Logistic Regression Feature Coefficients (Log-Odds Impact)",
                 fontsize=12, fontweight='bold', color=PALETTE['SLATE_DARK'], pad=12)
    ax.set_xlabel("Log-Odds Coefficient (Positive = Increases Readmission Risk)", fontsize=10, color=PALETTE['SLATE_DARK'])
    
    plot_path = os.path.join(OUTPUT_DIR, "top_coefficients.png")
    show_or_save_plot(plot_path, plt)

    report_path = os.path.join(OUTPUT_DIR, "coefficients_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""============================================================
DIABETES READMISSION — COEFFICIENT REPORT
============================================================
Top 10 Risk-Increasing Factors:
{top_risk.to_string(index=False)}

Top 10 Risk-Decreasing Factors:
{top_protective.to_string(index=False)}
""")
    print(f"\n[COEFFICIENTS] Report saved to: {report_path}")
    print("[COEFFICIENTS] Module completed successfully.\n")


if __name__ == "__main__":
    run_coefficients()
