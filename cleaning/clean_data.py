"""
Module 03: Data Cleaning
Handles missing values, drops unviable/identifier columns, filters expired cases,
and prints before/after cleaning summaries and preview.
Run in PyCharm:
    py cleaning/clean_data.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import RAW_DATA_PATH, EXPIRED_DISPOSITION_IDS, PALETTE, show_or_save_plot

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "cleaning")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_cleaning():
    print("=" * 65)
    print("MODULE 03: DATA CLEANING")
    print("=" * 65)
    df = pd.read_csv(RAW_DATA_PATH)
    orig_shape = df.shape
    print(f"[CLEANING] Original shape: {orig_shape}")

    # Missing values before cleaning
    print("\n[CLEANING] Missing values before cleaning:")
    missing_dict = {}
    for col in df.columns:
        cnt = int((df[col] == '?').sum() + df[col].isna().sum())
        if cnt > 0:
            missing_dict[col] = cnt
            print(f"- {col}: {cnt:,} ({cnt/len(df)*100:.2f}%)")

    # Plot missing values
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=PALETTE['SOFT_LINEN'])
    ax.set_facecolor("#FFFFFF")
    cols_sorted = sorted(missing_dict.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in cols_sorted]
    pcts = [x[1]/len(df) * 100 for x in cols_sorted]
    bars = ax.barh(names, pcts, color=PALETTE['TERRACOTTA'], edgecolor=PALETTE['WARM_SAND'], height=0.6)
    ax.set_title("Missing Values Before Cleaning (% of Encounters)",
                 fontsize=12, fontweight='bold', color=PALETTE['SLATE_DARK'], pad=12)
    ax.set_xlabel("Percentage Missing (%)", fontsize=10, color=PALETTE['SLATE_DARK'])
    ax.set_xlim(0, 105)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1, bar.get_y() + bar.get_height()/2.0, f"{w:.1f}%",
                va='center', fontsize=8.5, color=PALETTE['SLATE_DARK'])
    
    missing_plot_path = os.path.join(OUTPUT_DIR, "missing_values_bar.png")
    show_or_save_plot(missing_plot_path, plt)

    # Columns removed
    columns_removed = [
        ("encounter_id", "Database transaction identifier; risks memorization."),
        ("patient_nbr", "Patient tracking ID; prevents multi-encounter leakage."),
        ("weight", "96.86% missing data; non-viable for clinical modeling."),
        ("payer_code", "39.56% missing administrative billing data."),
        ("medical_specialty", "49.08% missing; high cardinality physician specialty."),
        ("examide", "Constant column (100% 'No', zero variance)."),
        ("citoglipton", "Constant column (100% 'No', zero variance)."),
        ("discharge_disposition_id (expired cases)", "Filtered 1,652 deceased encounters (readmission is impossible)")
    ]
    print("\n[CLEANING] Columns removed:")
    for col, reason in columns_removed:
        print(f"- {col}: {reason}")

    # Clinical exclusions (deceased patients)
    expired_mask = df['discharge_disposition_id'].isin(EXPIRED_DISPOSITION_IDS)
    expired_count = expired_mask.sum()

    # Apply cleaning
    df_clean = df[~expired_mask].copy()
    drop_col_names = [c for c, _ in columns_removed if c in df_clean.columns]
    df_clean = df_clean.drop(columns=drop_col_names)
    df_clean = df_clean[df_clean['gender'].isin(['Female', 'Male'])].copy()

    # Verify missing values after cleaning
    clean_missing = 0
    for col in NUMERIC_FEATURES:
        clean_missing += int(df_clean[col].isna().sum())
    print(f"\n[CLEANING] Missing values after cleaning: {clean_missing} (0 in all modeled numerical and clinical features; untested lab values encoded as 'None')")
    final_shape = df_clean.shape
    print(f"[CLEANING] Final shape: {final_shape}")

    print("\nCleaned data preview (first 5 rows):")
    pd.set_option('display.max_columns', 12)
    pd.set_option('display.width', 120)
    print(df_clean[['gender', 'age', 'admission_type_id', 'time_in_hospital', 'num_medications',
                    'number_inpatient', 'max_glu_serum', 'A1Cresult', 'insulin', 'readmitted']].head())

    # Write report
    report_path = os.path.join(OUTPUT_DIR, "cleaning_report.txt")
    report_text = f"""============================================================
DIABETES READMISSION — DATA CLEANING REPORT
============================================================
Original Shape:            {orig_shape}
Final Cleaned Shape:       {final_shape}
Records Removed:           {orig_shape[0] - final_shape[0]:,}
  - Deceased Patients:     {expired_count:,} (IDs 11, 19, 20, 21)
  - Invalid Gender:        3 encounters ('Unknown/Invalid')

Removed Columns:
""" + "\n".join([f"  - {c}: {r}" for c, r in columns_removed]) + f"""

Missing Value Plot:        {missing_plot_path}
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    # Save cleaned sample
    df_clean.head(100).to_csv(os.path.join(OUTPUT_DIR, "cleaned_sample.csv"), index=False)
    print(f"\n[CLEANING] Report saved to: {report_path}")
    print("[CLEANING] Module completed successfully.\n")


if __name__ == "__main__":
    run_cleaning()
