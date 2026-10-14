"""
Module 01: Data Ingestion
Loads raw dataset files, audits columns and data types, and prints full console inspection.
Run in PyCharm:
    py ingestion/ingest.py
"""

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import RAW_DATA_PATH, IDS_MAPPING_PATH

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ingestion")
os.makedirs(OUTPUT_DIR, exist_ok=True)
REPORT_PATH = os.path.join(OUTPUT_DIR, "ingestion_report.txt")
DICT_PATH = os.path.join(OUTPUT_DIR, "data_dictionary.csv")


def run_ingestion():
    print("=" * 65)
    print("MODULE 01: DATA INGESTION")
    print("=" * 65)
    print("[INGESTION] Loading dataset...")
    
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"[INGESTION ERROR] File not found: {RAW_DATA_PATH}")

    df_raw = pd.read_csv(RAW_DATA_PATH)
    rows, cols = df_raw.shape
    memory_mb = df_raw.memory_usage(deep=True).sum() / (1024 * 1024)

    print("[INGESTION] Dataset loaded successfully.")
    print(f"[INGESTION] Rows: {rows:,}")
    print(f"[INGESTION] Columns: {cols}")
    print(f"[INGESTION] Memory Footprint: {memory_mb:.2f} MB")
    
    print("\n[INGESTION] Columns available:")
    for i, col in enumerate(df_raw.columns, 1):
        print(f"  {i:>2}. {col:<26} (dtype: {df_raw[col].dtype})")

    # Generate data dictionary
    data_dict = []
    for col in df_raw.columns:
        q_count = int((df_raw[col] == '?').sum())
        null_count = int(df_raw[col].isna().sum())
        data_dict.append({
            'column': col,
            'dtype': str(df_raw[col].dtype),
            'unique_count': int(df_raw[col].nunique()),
            'question_marks': q_count,
            'null_count': null_count,
            'sample_values': str(list(df_raw[col].dropna().unique()[:3]))
        })
    df_dict = pd.DataFrame(data_dict)
    df_dict.to_csv(DICT_PATH, index=False)
    print(f"\n[INGESTION] Data dictionary saved to: {DICT_PATH}")

    # First 5 rows preview
    print("\nFirst 5 rows:")
    preview_cols = ['encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'time_in_hospital',
                    'num_medications', 'number_inpatient', 'diag_1', 'readmitted']
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 120)
    print(df_raw[preview_cols].head())

    # Write report
    report = f"""============================================================
DIABETES HOSPITAL READMISSION PREDICTION — INGESTION REPORT
============================================================
Raw Data File:              {RAW_DATA_PATH}
ID Mapping File:            {IDS_MAPPING_PATH}
Total Records (Encounters): {rows:,}
Total Feature Columns:      {cols}
Memory Footprint:           {memory_mb:.2f} MB
Data Dictionary Path:       {DICT_PATH}

Target Column:              readmitted
Raw Value Counts:
{df_raw['readmitted'].value_counts().to_string()}

Ingestion completed successfully.
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[INGESTION] Audit report written to: {REPORT_PATH}")
    print("[INGESTION] Module completed successfully.\n")


if __name__ == "__main__":
    run_ingestion()
