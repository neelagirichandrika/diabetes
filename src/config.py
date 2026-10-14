"""
Central configuration for Diabetic Inpatient Readmission Prediction.
Defines paths, feature definitions, clinical categories, and model parameters.
"""

import os

# Base paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "diabetic_data.csv")
if not os.path.exists(RAW_DATA_PATH):
    RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "diabetic_data.csv")

IDS_MAPPING_PATH = os.path.join(DATA_DIR, "raw", "IDS_mapping.csv")
if not os.path.exists(IDS_MAPPING_PATH):
    IDS_MAPPING_PATH = os.path.join(PROJECT_ROOT, "IDS_mapping.csv")

PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
CLEAN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "diabetes_clean.csv")

# Models path
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PIPELINE_PATH = os.path.join(MODELS_DIR, "diabetes_pipeline.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# Target definition
# 0 = Not Readmitted (raw 'NO')
# 1 = Readmitted (raw '>30' or '<30')
TARGET_COLUMN = "readmitted"
BINARY_TARGET_COLUMN = "target"
TARGET_LABELS = {
    0: "Not Readmitted",
    1: "Readmitted"
}

# Clinical filtering
# Encounters where the patient expired (disposition 11, 19, 20, 21) are excluded
# because subsequent readmission is logically and clinically impossible.
EXPIRED_DISPOSITION_IDS = [11, 19, 20, 21]

# Numerical features
NUMERIC_FEATURES = [
    "time_in_hospital",      # Length of stay in days (1 to 14)
    "num_lab_procedures",    # Number of lab tests performed (1 to 132)
    "num_procedures",        # Number of non-lab medical procedures (0 to 6)
    "num_medications",       # Number of distinct medications administered (1 to 81)
    "number_outpatient",     # Outpatient visits in past 12 months (0 to 42)
    "number_emergency",      # Emergency visits in past 12 months (0 to 76)
    "number_inpatient",      # Inpatient hospitalizations in past 12 months (0 to 21)
    "number_diagnoses"       # Number of diagnoses recorded (1 to 16)
]

# Categorical features
CATEGORICAL_FEATURES = [
    "gender",                # Male, Female
    "age",                   # 10-year brackets: [0-10) ... [90-100)
    "admission_type",        # Emergency, Urgent, Elective, Trauma Center, Other/Not Available
    "admission_source",      # Emergency Room, Physician Referral, Clinic Referral, Transfer from Facility, Other/Unknown
    "primary_diagnosis",     # ICD-9 clinical group: Circulatory, Respiratory, Diabetes, Digestive, etc.
    "diabetes_diagnosed",    # Whether diabetes code (250.xx) was diagnosed during encounter (Yes/No)
    "max_glu_serum",         # Serum glucose test result: None, Norm, >200, >300
    "A1Cresult",             # HbA1c test result: None, Norm, >7, >8
    "metformin",             # No, Steady, Up, Down
    "glipizide",             # No, Steady, Up, Down
    "glyburide",             # No, Steady, Up, Down
    "insulin",               # No, Steady, Up, Down
    "change",                # Ch (dosage changed), No
    "diabetesMed"            # Yes, No
]

ALL_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.20

# UI Theme Palette (Terracotta & Warm Sand)
PALETTE = {
    "TERRACOTTA": "#C85A32",
    "SLATE_DARK": "#261E1A",
    "WARM_SAND": "#E8E2D8",
    "SAGE_GREEN": "#4A7C59",
    "SOFT_LINEN": "#FDFBF7",
    "PURE_WHITE": "#FFFFFF",
    "MUTED_TAUPE": "#786B63",
    "MARIGOLD": "#D9822B"
}


# Standalone execution plot display toggle (default True for PyCharm interactive viewing)
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "1").lower() not in ("0", "false", "no")


def show_or_save_plot(save_path, plt_module=None, block=True):
    """
    Saves plot to save_path and calls plt.show() so visualizations
    render interactively when running individually in PyCharm, while
    allowing headless automated execution when SHOW_PLOTS=0.
    """
    import matplotlib.pyplot as plt
    p = plt_module if plt_module else plt
    p.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"[Plot Saved] {save_path}")

    show_env = os.environ.get("SHOW_PLOTS", "1").lower()
    if show_env not in ("0", "false", "no"):
        try:
            print(f"[Visualization] Displaying plot: {os.path.basename(save_path)} (Close plot window to proceed)")
            p.show(block=block)
        except Exception as e:
            print(f"[Note] GUI display unavailable ({e}). Plot saved to: {save_path}")
    p.close()

