"""
Module 04: Categorical Encoding & ICD-9 Grouping
Analyzes the ICD-9 clinical categorization, categorical feature encodings,
and demonstrates the resulting one-hot feature representation.
Run in PyCharm:
    py encoding/run_encoding.py
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_and_clean_data, build_preprocessor
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, PALETTE, show_or_save_plot

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "encoding")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_encoding():
    print("=" * 65)
    print("MODULE 04: CATEGORICAL ENCODING & FEATURE REPRESENTATION")
    print("=" * 65)
    X, y, _ = load_and_clean_data()

    print("[ENCODING] Categorical features detected:")
    for cat in CATEGORICAL_FEATURES:
        n_unique = X[cat].nunique()
        samples = list(X[cat].unique()[:3])
        print(f"- {cat} ({n_unique} unique values: {', '.join(str(s) for s in samples)})")

    print("\n[ENCODING] Encoding method:")
    print("- ICD-9 Clinical Grouping: primary_diagnosis codes grouped into 9 clinical categories")
    print("  (Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other)")
    print("- OneHotEncoder(handle_unknown='ignore', sparse_output=False) applied to all 14 categorical features")

    # Plot diagnosis category distribution
    print("\n[ENCODING] Generating Diagnosis Category Distribution Plot...")
    diag_counts = X['primary_diagnosis'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=PALETTE['SOFT_LINEN'])
    ax.set_facecolor("#FFFFFF")
    bars = ax.barh(diag_counts.index, diag_counts.values,
                   color=PALETTE['TERRACOTTA'], edgecolor=PALETTE['WARM_SAND'], height=0.6)
    ax.set_title("Primary Diagnosis Clinical Groupings (ICD-9)",
                 fontsize=12, fontweight='bold', color=PALETTE['SLATE_DARK'], pad=12)
    ax.set_xlabel("Inpatient Encounters", fontsize=10, color=PALETTE['SLATE_DARK'])
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 400, bar.get_y() + bar.get_height()/2.0, f"{w:,} ({w/len(X)*100:.1f}%)",
                va='center', fontsize=8.5, color=PALETTE['SLATE_DARK'])
    
    ax.set_xlim(0, max(diag_counts.values) * 1.18)
    diag_plot_path = os.path.join(OUTPUT_DIR, "diagnosis_category_distribution.png")
    show_or_save_plot(diag_plot_path, plt)

    # Transform demonstration
    preprocessor = build_preprocessor()
    preprocessor.fit(X.head(10000))
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    total_trans_cols = len(NUMERIC_FEATURES) + len(encoded_feature_names)

    print(f"\n[ENCODING] Transformed Dimensions:")
    print(f"  Input Features:            22 (8 numerical + 14 categorical)")
    print(f"  One-Hot Encoded Features:  {len(encoded_feature_names)}")
    print(f"  Total Transformed Matrix:  {total_trans_cols} columns across {len(X):,} records")

    # Sample output
    X_sample_trans = preprocessor.transform(X.head(5))
    df_sample_encoded = pd.DataFrame(X_sample_trans[:, len(NUMERIC_FEATURES):len(NUMERIC_FEATURES)+8],
                                     columns=encoded_feature_names[:8])
    print("\nResulting feature representation (sample output of encoded features, first 5 rows):")
    pd.set_option('display.width', 120)
    print(df_sample_encoded)

    report_path = os.path.join(OUTPUT_DIR, "encoding_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""============================================================
DIABETES READMISSION — CATEGORICAL ENCODING REPORT
============================================================
Total Categorical Features:   {len(CATEGORICAL_FEATURES)}
One-Hot Encoded Dimensions:   {len(encoded_feature_names)}
Total Transformed Dimensions: {total_trans_cols}

Diagnosis Group Distribution:
{diag_counts.to_string()}

All Encoded Features:
{pd.Series(encoded_feature_names).to_string()}
""")
    print(f"\n[ENCODING] Report saved to: {report_path}")
    print("[ENCODING] Module completed successfully.\n")


if __name__ == "__main__":
    run_encoding()
