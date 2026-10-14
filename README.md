# Diabetic Inpatient Readmission Prediction

A modular machine-learning project and web application for predicting hospital readmission risk among diabetic inpatients, built using Python, scikit-learn, and Flask on the UCI Diabetes 130-US Hospitals dataset.

---

## 1. Project Title
**Diabetic Inpatient Readmission Prediction — End-to-End Clinical Machine Learning Workflow**

---

## 2. Objective & Clinical Problem Statement
The primary objective of this project is **Diabetic Inpatient Readmission Prediction**: estimating whether a diabetic patient admitted to the hospital will experience a subsequent hospital readmission following the index encounter.

In diabetic inpatient care, hospital readmission represents an adverse patient outcome and significant clinical burden. By assessing readmission likelihood at the discharge preparation stage, healthcare teams can prioritize patients for enhanced transitional care, diabetes self-management education, post-discharge medication reconciliation, and scheduled primary care follow-up.

---

## 3. Dataset
* **Source**: UCI Machine Learning Repository — *Diabetes 130-US Hospitals for Years 1999–2008* (Strack et al., 2014).
* **Clinical Context**: Inpatient admissions across 130 United States hospital facilities.
* **Inclusion Criteria**: Patients diagnosed with diabetes, prescribed diabetes medications, with inpatient stays ranging from 1 to 14 days.
* **Source Files**:
  - `data/raw/diabetic_data.csv`: 101,766 inpatient encounters across 50 raw clinical and administrative fields.
  - `data/raw/IDS_mapping.csv`: Descriptor mappings for admission types, admission sources, and discharge dispositions.

---

## 4. Dataset Dimensions
* **Raw Dimensions**: 101,766 rows × 50 columns
* **Modeled Clean Encounters**: 100,111 records (after filtering 1,652 deceased encounters and 3 invalid records).

---

## 5. Target Variable
* **Column Name**: `readmitted`
* **Formulation**: Binary Classification
* **Class Definitions**:
  - **`0` — Not Readmitted**: Maps from raw value `NO` (53,209 encounters, **53.15%**).
  - **`1` — Readmitted**: Maps from raw values `>30` (readmitted after 30 days) and `<30` (readmitted within 30 days) combined (46,902 encounters, **46.85%**).
* **Target Scope**: Predicts whether a diabetic inpatient will experience *any* subsequent hospital readmission following the current encounter. (It is explicitly **not** described as a "30-day readmission prediction" because both `<30` and `>30` readmissions are grouped into Class 1).

---

## 6. Features (22 Modeled Features)

The final model consumes 22 features realistically available at the discharge planning timepoint:

### Numerical Features (8)
1. `time_in_hospital`: Inpatient length of stay in days (1 to 14).
2. `num_lab_procedures`: Total diagnostic laboratory tests performed during encounter (1 to 132).
3. `num_procedures`: Number of diagnostic or surgical non-lab medical procedures (0 to 6).
4. `num_medications`: Total distinct medications administered during encounter (1 to 81).
5. `number_outpatient`: Outpatient clinic visits in the preceding 12 months (0 to 42).
6. `number_emergency`: Emergency room visits in the preceding 12 months (0 to 76).
7. `number_inpatient`: Inpatient hospital admissions in the preceding 12 months (0 to 21).
8. `number_diagnoses`: Total diagnoses entered on the encounter record (1 to 16).

### Categorical Features (14)
1. `gender`: Patient gender (`Female`, `Male`).
2. `age`: 10-year age brackets (`[0-10)`, `[10-20)`, ..., `[90-100)`).
3. `admission_type`: Emergency, Urgent, Elective, Trauma Center, Other/Not Available.
4. `admission_source`: Emergency Room, Physician Referral, Clinic Referral, Transfer from Facility, Other/Unknown.
5. `primary_diagnosis`: Grouped ICD-9 clinical categories (`Circulatory`, `Respiratory`, `Digestive`, `Diabetes`, `Injury`, `Musculoskeletal`, `Genitourinary`, `Neoplasms`, `Other`).
6. `diabetes_diagnosed`: Whether diabetes (ICD-9 250.xx) was diagnosed during this stay (`Yes`, `No`).
7. `max_glu_serum`: Peak serum glucose test result (`None`, `Norm`, `>200`, `>300`).
8. `A1Cresult`: Glycated hemoglobin HbA1c test result (`None`, `Norm`, `>7`, `>8`).
9. `insulin`: Inpatient insulin prescription status (`No`, `Steady`, `Up`, `Down`).
10. `metformin`: Metformin prescription status (`No`, `Steady`, `Up`, `Down`).
11. `glipizide`: Glipizide prescription status (`No`, `Steady`, `Up`, `Down`).
12. `glyburide`: Glyburide prescription status (`No`, `Steady`, `Up`, `Down`).
13. `change`: Whether any diabetes medication dosage was modified (`Ch`, `No`).
14. `diabetesMed`: Whether any diabetes medication was prescribed (`Yes`, `No`).

---

## 7. Data Preprocessing & Leakage Checks
1. **Clinical Leakage Prevention**:
   - Encounters where the patient expired (`discharge_disposition_id` in 11, 19, 20, 21; 1,652 records) are excluded because deceased patients cannot be readmitted (100% have `readmitted = NO`). Retaining them induces artificial negative correlation.
   - Non-predictive transaction IDs (`encounter_id`, `patient_nbr`) are dropped to prevent ID memorization.
   - High-missingness columns (`weight` with 96.86% missing) and administrative billing codes (`payer_code`) are removed.
   - Constant zero-variance columns (`examide`, `citoglipton`) are dropped.
2. **Clinical ICD-9 Grouping**:
   - Primary diagnoses (`diag_1`) are grouped into 9 clinical categories following Strack et al. (2014) to control cardinality while retaining clinical context.
3. **Imputation & Encoding**:
   - Lab tests (`max_glu_serum`, `A1Cresult`) retain `'None'`, representing "test not ordered", a clinically meaningful negative indicator.
   - Categorical features are encoded with `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
   - Numerical features are standardized with `StandardScaler()`.
4. **Unified Pipeline**:
   - Feature preprocessing and estimator are combined in a single `sklearn.pipeline.Pipeline` serialized as `models/diabetes_pipeline.joblib`.

---

## 8. Model Selection & Candidate Benchmarking
Models were evaluated on an identical stratified 80/20 train/test split (80,088 train / 20,023 test, `random_state = 42`):

| Model Algorithm | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HistGradientBoosting (Selected)** | **63.69%** | **63.62%** | **52.53%** | **0.5755** | **0.6857** |
| Random Forest (100 trees, depth 12) | 63.26% | 64.27% | 48.61% | 0.5535 | 0.6821 |
| Logistic Regression (L2 Regularized)| 62.58% | 64.15% | 45.62% | 0.5332 | 0.6694 |

**Selected Model**: **HistGradientBoostingClassifier** — provides the highest discrimination (ROC-AUC 0.6857) and highest clinical recall (52.53%) for detecting readmission risk among diabetic inpatients.

---

## 9. Modular ML Workflow Structure

The project follows a step-by-step modular progression where each directory contains a standalone script, generated report, and publication-quality visualization:

```text
diabetes_data_project/
│
├── data/
│   ├── raw/
│   │   ├── diabetic_data.csv                    # Raw dataset (101,766 rows)
│   │   └── IDS_mapping.csv                      # ID descriptor mappings
│   └── processed/
│       └── diabetes_clean.csv                   # Cleaned feature matrix (100,111 rows)
│
├── ingestion/
│   ├── ingest.py                                # Ingestion script & schema audit
│   ├── ingestion_report.txt                     # Data audit report
│   └── data_dictionary.csv                      # Column data dictionary
│
├── eda/
│   ├── run_eda.py                               # Exploratory data analysis script
│   ├── eda_summary_report.txt                   # Statistical summary
│   ├── target_distribution.png                  # Target class balance plot
│   └── numerical_correlation_heatmap.png        # Feature correlation matrix
│
├── cleaning/
│   ├── clean_data.py                            # Data hygiene & filter script
│   ├── cleaning_report.txt                      # Detailed exclusion report
│   ├── missing_values_bar.png                   # Missingness visualization
│   └── cleaned_sample.csv                       # Cleaned sample preview
│
├── encoding/
│   ├── run_encoding.py                          # ICD-9 grouping & encoding script
│   ├── encoding_analysis_report.txt             # Dimension expansion report
│   └── diagnosis_category_distribution.png      # Diagnosis category breakdown
│
├── scaling/
│   ├── run_scaling.py                           # StandardScaler standardization script
│   ├── scaling_report.txt                       # Pre/post scaling statistics
│   └── numeric_features_before_after.png        # Standardization histograms
│
├── split/
│   ├── run_split.py                             # Stratification & zero-leakage check
│   ├── split_report.txt                         # Partition report
│   └── train_test_class_balance.png             # Train vs test distribution plot
│
├── splitting_criteria/
│   ├── run_splitting_criteria.py                # Gini vs Entropy/Log-loss study
│   ├── splitting_criteria_report.txt            # Comparative split report
│   └── gini_vs_entropy_comparison.png           # Criteria performance plot
│
├── logistic_regression/
│   ├── run_logistic_regression.py               # Baseline linear classifier
│   ├── logistic_report.txt                      # Baseline metrics report
│   ├── roc_curve.png                            # Receiver Operating Characteristic plot
│   └── confusion_matrix.png                     # Confusion matrix heatmap
│
├── decision_tree/
│   ├── run_decision_tree.py                     # Tree depth study
│   ├── decision_tree_report.txt                 # Depth comparison report
│   └── tree_depth_performance.png               # Overfitting curve (depth vs accuracy)
│
├── regularization/
│   ├── run_regularization.py                    # L1 (Lasso) vs L2 (Ridge) across C values
│   ├── regularization_report.txt                # Sparsity & penalty comparison
│   └── regularization_comparison.png            # Regularization strength curve
│
├── bias_variance/
│   ├── run_bias_variance.py                     # Bias-variance complexity regimes
│   ├── bias_variance_report.txt                 # Generalization gap report
│   └── learning_curves.png                      # Bias vs variance bar chart
│
├── coefficients/
│   ├── run_coefficients.py                      # Feature weight & Odds Ratio analysis
│   ├── coefficients_report.txt                  # Top risk and protective factors
│   └── top_coefficients.png                     # Feature importance / log-odds plot
│
├── pipeline/
│   ├── verify_pipeline.py                       # Pipeline integrity & sample inference
│   └── pipeline_verification_report.txt         # End-to-end verification report
│
├── co6/
│   ├── run_co6_docker.py                        # CO6 container audit runner & architecture visualizer
│   ├── co6_docker_report.txt                    # Certified 100/100 production audit report
│   ├── docker_container_architecture.png        # Publication-grade container architecture
│   ├── Dockerfile                               # Dedicated CO6 container manifest
│   ├── docker-compose.yml                       # Dedicated CO6 compose definition
│   ├── build_docker.bat                         # 1-click Windows container build script
│   ├── run_docker.bat                           # 1-click Windows Docker compose runner
│   ├── build_docker.sh                          # Linux/macOS container build script
│   └── run_docker.sh                           # Linux/macOS container runner script
│
├── models/
│   ├── diabetes_pipeline.joblib                 # Serialized scikit-learn Pipeline
│   └── model_metadata.json                      # Actual test metrics and feature metadata
│
├── src/
│   ├── config.py                                # Central paths, feature lists, constants
│   ├── preprocessing.py                         # Data hygiene, ICD-9 mapping, ColumnTransformer
│   ├── train.py                                 # Training script with model comparison & export
│   ├── evaluate.py                              # Standalone test set evaluation module
│   └── predict.py                               # Prediction inference module with sample profiles
│
├── app/
│   ├── templates/index.html                     # Web interface with Terracotta & Warm Sand styling
│   ├── static/style.css                         # Modern Editorial CSS design system
│   └── app.py                                   # Flask web server, REST API & /health probe
│
├── tests/
│   ├── test_data.py                             # Data loading and cleaning unit tests
│   ├── test_pipeline.py                         # Pipeline loading and prediction tests
│   ├── test_api.py                              # Flask route and JSON API integration tests
│   └── test_docker.py                           # CO6 Docker security, hygiene, and health tests
│
├── Dockerfile                                   # Root production container manifest (python:3.11-slim)
├── .dockerignore                                # Context minimization (excludes raw dataset & venv)
├── docker-compose.yml                           # Multi-container orchestrator & resource limits
├── run_app.py                                   # Root launcher for PyCharm & container
├── requirements.txt                             # Project dependencies (includes gunicorn)
└── README.md                                    # Documentation
```

---

## 10. Web Interface & Terracotta & Warm Sand Design System

The web application features the **Option 1: Terracotta & Warm Sand (Modern Editorial)** visual identity:
* **Main Background**: Soft Linen (`#FDFBF7`)
* **Cards & Surfaces**: Pure White (`#FFFFFF`)
* **Header & Strong Headings**: Dark Roasted Bean (`#261E1A`)
* **Accent & Buttons**: Burnt Terracotta (`#C85A32`, hover `#AF4D29`)
* **Main Text**: Warm Near-Black (`#2A221E`)
* **Secondary Text**: Muted Taupe (`#786B63`)
* **Borders & Dividers**: Subtle Sandstone (`#E8E2D8`)
* **Positive / Baseline Risk**: Earthy Forest Green (`#4A7C59`)
* **Attention / Elevated Risk**: Rich Marigold Amber (`#D9822B`)
* **Interactive Samples**: 1-click loaders for baseline and elevated risk profiles.
* **No AI references**: No "AI-powered", "ChatGPT", "robot" graphics, or fake claims.

---

## 11. PyCharm Setup & Execution Guide

### Step 1: Open Project in PyCharm
1. Launch PyCharm.
2. Select **File &rarr; Open...** and select the `diabetes data project` directory.

### Step 2: Configure Python Interpreter
1. Navigate to **File &rarr; Settings &rarr; Project: diabetes data project &rarr; Python Interpreter**.
2. Select your Python 3.10+ (or Python 3.14) interpreter.

### Step 3: Install Dependencies
In the PyCharm Terminal (**Alt + F12**), install required packages:
```bash
pip install -r requirements.txt
```

### Step 4: Run Training (Optional)
The trained model is already saved in `models/diabetes_pipeline.joblib`. To retrain:
* Right-click [`src/train.py`](file:///c:/Users/Administrator/Desktop/diabetes%20data%20project/src/train.py) and select **Run 'train'**.

### Step 5: Run Course Outcome (CO1–CO6) Certified Audit Runners
Each Course Outcome has a dedicated, 1-click audit runner that executes complete verification and generates publication-quality figures and certification reports:
* **CO1 — Clinical ML Lifecycle, Ingestion & Leakage Controls**:
  ```bash
  py co1/run_co1_lifecycle.py
  ```
  *Report*: [`co1/co1_lifecycle_report.txt`](file:///c:/Users/Administrator/Desktop/diabetes%20data%20project/co1/co1_lifecycle_report.txt) &bull; *Visual*: [`co1/co1_lifecycle_architecture.png`](file:///c:/Users/Administrator/Desktop/diabetes%20data%20project/co1/co1_lifecycle_architecture.png)
* **CO2 — Supervised Learning: Linear Models & Regularization**:
  ```bash
  py co2/run_co2_linear_models.py
  ```
  *Report*: [`co2/co2_linear_models_report.txt`](file:///c:/Users/Administrator/Desktop/diabetes%20data%20project/co2/co2_linear_models_report.txt) &bull; *Visual*: [`co2/co2_odds_ratios_regularization.png`](file:///c:/Users/Administrator/Desktop/diabetes%20data%20project/co2/co2_odds_ratios_regularization.png)
* **CO3 — Supervised Learning: Tree Ensembles & Overfitting**:
  ```bash
  py co3/run_co3_tree_models.py
  ```
  *Report*: [`co3/co3_tree_models_report.txt`](file:///c:/Users/Administrator/Desktop/diabetes%20data%20project/co3/co3_tree_models_report.txt) &bull; *Visual*: [`co3/co3_tree_depth_overfitting.png`](file:///c:/Users/Administrator/Desktop/diabetes%20data%20project/co3/co3_tree_depth_overfitting.png)
* **CO4 — Unsupervised Learning: K-Means & PCA Phenotypes**:
  ```bash
  py co4/run_co4_unsupervised.py
  ```
  *Report*: [`co4/co4_unsupervised_report.txt`](file:///c:/Users/Administrator/Desktop/diabetes%20data%20project/co4/co4_unsupervised_report.txt) &bull; *Visual*: [`co4/patient_clusters_pca.png`](file:///c:/Users/Administrator/Desktop/diabetes%20data%20project/co4/patient_clusters_pca.png)
* **CO5 — Model Evaluation, Selection & Calibration**:
  ```bash
  py co5/run_co5_evaluation.py
  ```
  *Report*: [`co5/co5_evaluation_report.txt`](file:///c:/Users/Administrator/Desktop/diabetes%20data%20project/co5/co5_evaluation_report.txt) &bull; *Visual*: [`co5/co5_evaluation_calibration.png`](file:///c:/Users/Administrator/Desktop/diabetes%20data%20project/co5/co5_evaluation_calibration.png)
* **CO6 — ML Engineering, Production Docker Containerization**:
  ```bash
  py co6/run_co6_docker.py
  ```
  *Report*: [`co6/co6_docker_report.txt`](file:///c:/Users/Administrator/Desktop/diabetes%20data%20project/co6/co6_docker_report.txt) &bull; *Visual*: [`co6/docker_container_architecture.png`](file:///c:/Users/Administrator/Desktop/diabetes%20data%20project/co6/docker_container_architecture.png)

### Step 5B: Run Granular Workflow Scripts
Any individual module can also be executed independently:
* `py ingestion/ingest.py`
* `py eda/run_eda.py`
* `py cleaning/clean_data.py`
* `py encoding/run_encoding.py`
* `py scaling/run_scaling.py`
* `py split/run_split.py`
* `py splitting_criteria/run_splitting_criteria.py`
* `py logistic_regression/run_logistic_regression.py`
* `py decision_tree/run_decision_tree.py`
* `py regularization/run_regularization.py`
* `py bias_variance/run_bias_variance.py`
* `py coefficients/run_coefficients.py`
* `py pipeline/verify_pipeline.py`

### Step 6: Run Unit & Integration Tests
Run the complete automated test suite (including Docker validation tests):
```bash
py -m unittest discover tests
```

### Step 7: Launch the Web Application (Local Python)
* In PyCharm: Right-click [`run_app.py`](file:///c:/Users/Administrator/Desktop/diabetes%20data%20project/run_app.py) and select **Run 'run_app'**.
* Or in terminal:
  ```bash
  py run_app.py
  ```
* Open your browser at:
  ```text
  http://127.0.0.1:5000
  ```

### Step 8: Run via Production Docker Container (CO6 Implementation)
To run the clinical service inside an isolated, production-grade Linux container:

1. **Option A: One-command Docker Compose (Recommended)**
   ```bash
   docker compose up --build
   ```
   Or double-click [`co6/run_docker.bat`](file:///c:/Users/Administrator/Desktop/diabetes%20data%20project/co6/run_docker.bat) on Windows.

2. **Option B: Manual Docker CLI Build & Run**
   ```bash
   # Build the container image
   docker build -t diabetes-predict-app:1.0.0 .

   # Run container with port forwarding to 5000
   docker run -d -p 5000:5000 --name diabetes_app diabetes-predict-app:1.0.0
   ```

3. **Verify Container Health & Endpoints**
   * Web Interface: `http://localhost:5000`
   * Health Check: `http://localhost:5000/health`
   * Docker Status API: `http://localhost:5000/api/docker-status`
   * Run CO6 Container Audit: `py co6/run_co6_docker.py`

---

## 12. Prediction Workflow
```text
User fills web form (or clicks sample profile)
                     ↓
Flask app validates numeric ranges and categorical values
                     ↓
predict_patient() formats DataFrame with exact feature order
                     ↓
diabetes_pipeline.joblib applies ColumnTransformer:
  - StandardScaler (8 numeric features)
  - OneHotEncoder (14 categorical features)
                     ↓
HistGradientBoostingClassifier calculates:
  - Binary prediction class (0 or 1)
  - Predicted probability (e.g., 88.5% readmission risk)
                     ↓
Result card renders with risk level, progress meter, and disclaimer
```

---

## 13. Limitations
* This project is developed for **educational and research demonstration purposes**.
* Predictive accuracy is bounded by the historical nature of the 1999–2008 dataset and lack of social determinants of health (SDOH), post-discharge medication compliance, or transportation access.

---

## 14. Healthcare Disclaimer
> **Important Notice**: This application provides a machine-learning prediction based on the supplied dataset and is intended for educational purposes. It is **not** a medical diagnosis, clinical decision support tool, or medical device, and should **not** replace professional clinical judgment or advice from a qualified healthcare professional.
