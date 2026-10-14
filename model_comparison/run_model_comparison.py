"""
Module: Model Comparison
Compares classical classification algorithms on the identical test partition and outputs benchmark table.
Run in PyCharm:
    py model_comparison/run_model_comparison.py
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_and_clean_data, create_full_pipeline
from src.config import RANDOM_STATE, TEST_SIZE, PALETTE, show_or_save_plot

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_model_comparison():
    print("=" * 65)
    print("MODULE: CLASSIFICATION MODEL COMPARISON")
    print("=" * 65)
    print("[MODEL COMPARISON]")
    print("Loading data and creating stratified 80/20 train/test split...")
    X, y, _ = load_and_clean_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100, max_depth=8, random_state=RANDOM_STATE)
    }

    records = []
    print("\nTraining and evaluating candidate models...")
    for name, clf in models.items():
        print(f"  Training {name}...")
        pipeline = create_full_pipeline(clf)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        records.append({
            'Model': name,
            'Accuracy': f"{acc * 100:.2f}%",
            'Precision': f"{prec * 100:.2f}%",
            'Recall': f"{rec * 100:.2f}%",
            'F1': f"{f1:.4f}",
            'ROC-AUC': f"{auc:.4f}",
            '_f1_val': f1,
            '_auc_val': auc
        })

    df_comp = pd.DataFrame(records)
    print("\nModel                | Accuracy | Precision | Recall  | F1     | ROC-AUC")
    print("-" * 70)
    for _, r in df_comp.iterrows():
        print(f"{r['Model']:<20} | {r['Accuracy']:<8} | {r['Precision']:<9} | {r['Recall']:<7} | {r['F1']:<6} | {r['ROC-AUC']}")

    # Plot
    print("\n[MODEL COMPARISON] Generating Model Comparison Bar Chart...")
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=PALETTE['SOFT_LINEN'])
    ax.set_facecolor("#FFFFFF")
    
    df_plot = df_comp.set_index('Model')[['_f1_val', '_auc_val']].rename(columns={'_f1_val': 'F1-Score', '_auc_val': 'ROC-AUC'})
    df_plot.plot(kind='bar', ax=ax, color=[PALETTE['TERRACOTTA'], PALETTE['SLATE_DARK']],
                 edgecolor=PALETTE['WARM_SAND'], width=0.55)
    ax.set_title("Candidate Model Performance Comparison (Test Set)",
                 fontsize=12, fontweight='bold', color=PALETTE['SLATE_DARK'], pad=12)
    ax.set_ylabel("Score", fontsize=10, color=PALETTE['SLATE_DARK'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_ylim(0, 0.85)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plot_path = os.path.join(OUTPUT_DIR, "model_comparison_bar.png")
    show_or_save_plot(plot_path, plt)

    report_path = os.path.join(OUTPUT_DIR, "model_comparison_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""============================================================
DIABETES READMISSION — MODEL COMPARISON BENCHMARK
============================================================
{df_comp[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']].to_string(index=False)}
""")
    print(f"\n[MODEL COMPARISON] Report saved to: {report_path}")
    print("[MODEL COMPARISON] Module completed successfully.\n")


if __name__ == "__main__":
    run_model_comparison()
