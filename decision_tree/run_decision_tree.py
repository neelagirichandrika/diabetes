"""
Module 09: Decision Tree Classifier
Analyzes tree depth, complexity, and generalization performance.
Run in PyCharm:
    py decision_tree/run_decision_tree.py
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_and_clean_data, create_full_pipeline
from src.config import RANDOM_STATE, TEST_SIZE, PALETTE, show_or_save_plot

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "decision_tree")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_decision_tree():
    print("=" * 65)
    print("MODULE 09: DECISION TREE CLASSIFIER")
    print("=" * 65)
    print("[DECISION TREE]")
    print("Training model...")
    X, y, _ = load_and_clean_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Moderate depth 6 as default standalone
    clf_default = DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)
    pipe_default = create_full_pipeline(clf_default)
    pipe_default.fit(X_train, y_train)
    y_pred_def = pipe_default.predict(X_test)
    y_prob_def = pipe_default.predict_proba(X_test)[:, 1]

    acc_def = accuracy_score(y_test, y_pred_def)
    prec_def = precision_score(y_test, y_pred_def)
    rec_def = recall_score(y_test, y_pred_def)
    f1_def = f1_score(y_test, y_pred_def)
    auc_def = roc_auc_score(y_test, y_prob_def)

    print("\nActual metrics:")
    print(f"Accuracy:  {acc_def * 100:.2f}%")
    print(f"Precision: {prec_def * 100:.2f}%")
    print(f"Recall:    {rec_def * 100:.2f}%")
    print(f"F1-score:  {f1_def:.4f}")
    print(f"ROC-AUC:   {auc_def:.4f}")

    print("\nEvaluating tree depth progression (Overfitting analysis)...")
    depths = [3, 6, 8, 10, 15, 20]
    records = []

    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
        pipeline = create_full_pipeline(clf)
        pipeline.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, pipeline.predict(X_train))
        test_acc = accuracy_score(y_test, pipeline.predict(X_test))
        test_f1 = f1_score(y_test, pipeline.predict(X_test))
        test_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

        records.append({
            'Depth': d,
            'Train_Acc': round(train_acc * 100, 2),
            'Test_Acc': round(test_acc * 100, 2),
            'Test_F1': round(test_f1, 4),
            'Test_AUC': round(test_auc, 4)
        })

    df_depth = pd.DataFrame(records).set_index('Depth')
    print(df_depth)

    # Plot 1: Tree Depth Performance
    print("\n[DECISION TREE] Generating Tree Depth vs Performance Plot...")
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=PALETTE['SOFT_LINEN'])
    ax.set_facecolor("#FFFFFF")
    ax.plot(df_depth.index, df_depth['Train_Acc'], marker='o',
            color=PALETTE['TERRACOTTA'], lw=2, label='Training Accuracy')
    ax.plot(df_depth.index, df_depth['Test_Acc'], marker='s',
            color=PALETTE['SLATE_DARK'], lw=2, label='Testing Accuracy')
    ax.set_title("Decision Tree Depth vs Accuracy (Overfitting Curve)",
                 fontsize=12, fontweight='bold', color=PALETTE['SLATE_DARK'], pad=12)
    ax.set_xlabel("Tree Maximum Depth", fontsize=10, color=PALETTE['SLATE_DARK'])
    ax.set_ylabel("Accuracy (%)", fontsize=10, color=PALETTE['SLATE_DARK'])
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)
    
    plot_path = os.path.join(OUTPUT_DIR, "tree_depth_performance.png")
    show_or_save_plot(plot_path, plt)

    # Plot 2: Tree Structure Visualization
    print("\n[DECISION TREE] Generating Tree Structure Visualization (Depth=3 sub-tree)...")
    from sklearn.tree import plot_tree
    tree_clf = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    pipe_shallow = create_full_pipeline(tree_clf)
    pipe_shallow.fit(X_train, y_train)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=PALETTE['SOFT_LINEN'])
    plot_tree(
        tree_clf,
        max_depth=3,
        feature_names=pipe_shallow.named_steps['preprocessor'].get_feature_names_out(),
        class_names=['No Readmit', 'Readmitted'],
        filled=True,
        rounded=True,
        fontsize=7,
        ax=ax
    )
    ax.set_title("Decision Tree Architectural Diagram (Depth=3 Sub-tree)",
                 fontsize=12, fontweight='bold', color=PALETTE['SLATE_DARK'], pad=12)
    tree_structure_path = os.path.join(OUTPUT_DIR, "tree_structure.png")
    show_or_save_plot(tree_structure_path, plt)

    report_path = os.path.join(OUTPUT_DIR, "decision_tree_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""============================================================
DIABETES READMISSION — DECISION TREE STUDY
============================================================
Default Tree (Depth=6):
Accuracy:  {acc_def * 100:.2f}%
Precision: {prec_def * 100:.2f}%
Recall:    {rec_def * 100:.2f}%
F1-Score:  {f1_def:.4f}
ROC-AUC:   {auc_def:.4f}

Progression Across Depths:
{df_depth.to_string()}
""")
    print(f"\n[DECISION TREE] Report saved to: {report_path}")
    print("[DECISION TREE] Module completed successfully.\n")


if __name__ == "__main__":
    run_decision_tree()
