"""
Module 02: Exploratory Data Analysis (EDA)
Generates exactly 5 major visualizations from diabetic_data.csv:
  1. Readmission Distribution (Target Imbalance)
  2. Age vs Readmission (Risk Across Demographic Cohorts)
  3. Hospital Utilization (Length of Stay, Lab Tests, Medications)
  4. Previous Visits vs Readmission (Inpatient, Emergency, Outpatient Frequency)
  5. Numerical Feature Correlation (Clinical Variable Relationships)

Run in PyCharm:
    py eda/run_eda.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import RAW_DATA_PATH, PALETTE, show_or_save_plot

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "eda", "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)
REPORT_PATH = os.path.join(PROJECT_ROOT, "eda", "eda_summary_report.txt")


def run_eda():
    print("=" * 65)
    print("MODULE 02: EXPLORATORY DATA ANALYSIS (EDA) — 5 MAJOR VISUALIZATIONS")
    print("=" * 65)
    print("[EDA] Loading dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    rows, cols = df.shape
    print(f"[EDA] Dataset shape: {df.shape}")

    # -------------------------------------------------------------
    # 1. READMISSION DISTRIBUTION
    # -------------------------------------------------------------
    print("\n[EDA 1] Readmission distribution")
    target_counts = df['readmitted'].value_counts()
    total = len(df)
    for val in ['NO', '>30', '<30']:
        cnt = target_counts.get(val, 0)
        pct = (cnt / total) * 100
        print(f"  {val:<6}: {cnt:>6,} ({pct:.2f}%)")

    b_0 = target_counts.get('NO', 0)
    b_1 = target_counts.get('>30', 0) + target_counts.get('<30', 0)
    print(f"  Binary Aggregation (Any Readmission):")
    print(f"    0 (Not Readmitted): {b_0:,} ({b_0/total*100:.2f}%)")
    print(f"    1 (Readmitted):     {b_1:,} ({b_1/total*100:.2f}%)")

    fig1, ax1 = plt.subplots(figsize=(7, 5), facecolor=PALETTE['SOFT_LINEN'])
    ax1.set_facecolor("#FFFFFF")
    colors1 = [PALETTE['SLATE_DARK'], PALETTE['TERRACOTTA'], PALETTE['MARIGOLD']]
    bars1 = ax1.bar(['NO', '>30', '<30'],
                    [target_counts.get('NO', 0), target_counts.get('>30', 0), target_counts.get('<30', 0)],
                    color=colors1, edgecolor=PALETTE['WARM_SAND'], width=0.52)

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 700,
                 f"{yval:,}\n({yval/total*100:.1f}%)",
                 ha='center', va='bottom', fontsize=9.5, color=PALETTE['SLATE_DARK'], fontweight='bold')

    ax1.set_title("Readmission Distribution", fontsize=13, fontweight='bold', color=PALETTE['SLATE_DARK'], pad=15)
    ax1.set_xlabel("Readmission Category", fontsize=10.5, color=PALETTE['SLATE_DARK'], fontweight='bold')
    ax1.set_ylabel("Inpatient Encounters", fontsize=10.5, color=PALETTE['SLATE_DARK'], fontweight='bold')
    ax1.set_ylim(0, max(target_counts.values) * 1.18)
    ax1.grid(axis='y', linestyle='--', alpha=0.35, color=PALETTE['WARM_SAND'])

    path_1 = os.path.join(OUTPUT_DIR, "readmission_distribution.png")
    show_or_save_plot(path_1, plt)

    # -------------------------------------------------------------
    # 2. AGE VS READMISSION
    # -------------------------------------------------------------
    print("\n[EDA 2] Age vs readmission")
    age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                 '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
    age_cross = pd.crosstab(df['age'], df['readmitted'])[['NO', '>30', '<30']].reindex(age_order)
    age_pct = (age_cross.div(age_cross.sum(axis=1), axis=0) * 100).round(2)

    peak_age = df['age'].value_counts().index[0]
    peak_count = df['age'].value_counts().iloc[0]
    print(f"  Age brackets: 10 cohorts evaluated across {len(df):,} encounters")
    print(f"  Peak patient volume: {peak_age} with {peak_count:,} encounters ({peak_count/total*100:.1f}% of cohort)")
    print(f"  Highest readmission rate: [80-90) cohort at {100 - age_pct.loc['[80-90)', 'NO']:.2f}% readmission rate")

    fig2, ax2 = plt.subplots(figsize=(10, 5.5), facecolor=PALETTE['SOFT_LINEN'])
    ax2.set_facecolor("#FFFFFF")
    
    age_pct.plot(kind='bar', stacked=True, ax=ax2,
                 color=[PALETTE['SLATE_DARK'], PALETTE['TERRACOTTA'], PALETTE['MARIGOLD']],
                 edgecolor=PALETTE['WARM_SAND'], width=0.65)
    
    ax2.set_title("Readmission Across Age Groups", fontsize=13, fontweight='bold', color=PALETTE['SLATE_DARK'], pad=15)
    ax2.set_xlabel("Age Cohort", fontsize=10.5, color=PALETTE['SLATE_DARK'], fontweight='bold')
    ax2.set_ylabel("Percentage of Records (%)", fontsize=10.5, color=PALETTE['SLATE_DARK'], fontweight='bold')
    ax2.set_xticklabels(age_order, rotation=0, fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.legend(['NO (No Readmission)', '>30 Days Readmission', '<30 Days Readmission'],
               loc='upper right', frameon=True, facecolor=PALETTE['SOFT_LINEN'])
    ax2.grid(axis='y', linestyle='--', alpha=0.35, color=PALETTE['WARM_SAND'])

    path_2 = os.path.join(OUTPUT_DIR, "age_vs_readmission.png")
    show_or_save_plot(path_2, plt)

    # -------------------------------------------------------------
    # 3. HOSPITAL UTILIZATION
    # -------------------------------------------------------------
    print("\n[EDA 3] Hospital utilization")
    util_stats = {
        'time_in_hospital': (df['time_in_hospital'].mean(), df['time_in_hospital'].median(), df['time_in_hospital'].min(), df['time_in_hospital'].max()),
        'num_lab_procedures': (df['num_lab_procedures'].mean(), df['num_lab_procedures'].median(), df['num_lab_procedures'].min(), df['num_lab_procedures'].max()),
        'num_medications': (df['num_medications'].mean(), df['num_medications'].median(), df['num_medications'].min(), df['num_medications'].max())
    }
    print(f"  - time_in_hospital:   Mean {util_stats['time_in_hospital'][0]:.2f} days (Median: {util_stats['time_in_hospital'][1]:.0f}, Range: {util_stats['time_in_hospital'][2]}-{util_stats['time_in_hospital'][3]})")
    print(f"  - num_lab_procedures: Mean {util_stats['num_lab_procedures'][0]:.2f} tests (Median: {util_stats['num_lab_procedures'][1]:.0f}, Range: {util_stats['num_lab_procedures'][2]}-{util_stats['num_lab_procedures'][3]})")
    print(f"  - num_medications:    Mean {util_stats['num_medications'][0]:.2f} meds (Median: {util_stats['num_medications'][1]:.0f}, Range: {util_stats['num_medications'][2]}-{util_stats['num_medications'][3]})")

    fig3, axes3 = plt.subplots(1, 3, figsize=(13, 4.5), facecolor=PALETTE['SOFT_LINEN'])

    # 3A: Length of stay
    axes3[0].set_facecolor("#FFFFFF")
    axes3[0].hist(df['time_in_hospital'], bins=14, color=PALETTE['TERRACOTTA'], edgecolor=PALETTE['WARM_SAND'], alpha=0.9)
    axes3[0].axvline(util_stats['time_in_hospital'][0], color=PALETTE['SLATE_DARK'], linestyle='--', lw=1.5,
                     label=f"Mean: {util_stats['time_in_hospital'][0]:.1f}d")
    axes3[0].set_title("Length of Stay (Days)", fontsize=11, fontweight='bold', color=PALETTE['SLATE_DARK'])
    axes3[0].set_xlabel("Days in Hospital", fontsize=9.5, color=PALETTE['SLATE_DARK'])
    axes3[0].set_ylabel("Encounters", fontsize=9.5, color=PALETTE['SLATE_DARK'])
    axes3[0].legend(frameon=True, fontsize=8.5)
    axes3[0].grid(axis='y', linestyle='--', alpha=0.3)

    # 3B: Lab procedures
    axes3[1].set_facecolor("#FFFFFF")
    axes3[1].hist(df['num_lab_procedures'], bins=30, color=PALETTE['SLATE_DARK'], edgecolor=PALETTE['WARM_SAND'], alpha=0.9)
    axes3[1].axvline(util_stats['num_lab_procedures'][0], color=PALETTE['TERRACOTTA'], linestyle='--', lw=1.5,
                     label=f"Mean: {util_stats['num_lab_procedures'][0]:.1f}")
    axes3[1].set_title("Number of Lab Procedures", fontsize=11, fontweight='bold', color=PALETTE['SLATE_DARK'])
    axes3[1].set_xlabel("Distinct Lab Tests", fontsize=9.5, color=PALETTE['SLATE_DARK'])
    axes3[1].legend(frameon=True, fontsize=8.5)
    axes3[1].grid(axis='y', linestyle='--', alpha=0.3)

    # 3C: Medications administered
    axes3[2].set_facecolor("#FFFFFF")
    axes3[2].hist(df['num_medications'], bins=28, color=PALETTE['MARIGOLD'], edgecolor=PALETTE['WARM_SAND'], alpha=0.9)
    axes3[2].axvline(util_stats['num_medications'][0], color=PALETTE['SLATE_DARK'], linestyle='--', lw=1.5,
                     label=f"Mean: {util_stats['num_medications'][0]:.1f}")
    axes3[2].set_title("Medications Administered", fontsize=11, fontweight='bold', color=PALETTE['SLATE_DARK'])
    axes3[2].set_xlabel("Medications Count", fontsize=9.5, color=PALETTE['SLATE_DARK'])
    axes3[2].legend(frameon=True, fontsize=8.5)
    axes3[2].grid(axis='y', linestyle='--', alpha=0.3)

    plt.suptitle("Hospital Utilization", fontsize=13, fontweight='bold', color=PALETTE['SLATE_DARK'], y=1.02)
    plt.tight_layout()
    path_3 = os.path.join(OUTPUT_DIR, "hospital_utilization.png")
    show_or_save_plot(path_3, plt)

    # -------------------------------------------------------------
    # 4. PREVIOUS VISITS VS READMISSION
    # -------------------------------------------------------------
    print("\n[EDA 4] Previous visits vs readmission")
    visit_cols = ['number_inpatient', 'number_emergency', 'number_outpatient']
    mean_visits = df.groupby('readmitted')[visit_cols].mean().reindex(['NO', '>30', '<30'])

    for vcol in visit_cols:
        no_val = mean_visits.loc['NO', vcol]
        gt30_val = mean_visits.loc['>30', vcol]
        lt30_val = mean_visits.loc['<30', vcol]
        pct_diff = ((lt30_val - no_val) / no_val * 100) if no_val > 0 else 0
        print(f"  - {vcol:<18}: NO={no_val:.2f} | >30={gt30_val:.2f} | <30={lt30_val:.2f} (+{pct_diff:.1f}% for <30)")

    fig4, ax4 = plt.subplots(figsize=(8, 5.2), facecolor=PALETTE['SOFT_LINEN'])
    ax4.set_facecolor("#FFFFFF")
    
    x = np.arange(len(visit_cols))
    width = 0.25

    rects1 = ax4.bar(x - width, mean_visits.loc['NO'], width, label='NO (No Readmission)',
                     color=PALETTE['SLATE_DARK'], edgecolor=PALETTE['WARM_SAND'])
    rects2 = ax4.bar(x, mean_visits.loc['>30'], width, label='>30 Days Readmission',
                     color=PALETTE['TERRACOTTA'], edgecolor=PALETTE['WARM_SAND'])
    rects3 = ax4.bar(x + width, mean_visits.loc['<30'], width, label='<30 Days Readmission',
                     color=PALETTE['MARIGOLD'], edgecolor=PALETTE['WARM_SAND'])

    ax4.set_title("Previous Hospital Visits and Readmission", fontsize=13, fontweight='bold', color=PALETTE['SLATE_DARK'], pad=15)
    ax4.set_ylabel("Mean Number of Prior Visits (Past 12 Months)", fontsize=10, color=PALETTE['SLATE_DARK'], fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Inpatient Visits', 'Emergency Visits', 'Outpatient Visits'], fontsize=9.5, fontweight='bold')
    ax4.legend(frameon=True, facecolor=PALETTE['SOFT_LINEN'])
    ax4.grid(axis='y', linestyle='--', alpha=0.35, color=PALETTE['WARM_SAND'])

    # Value labels
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax4.annotate(f'{height:.2f}',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8.5, color=PALETTE['SLATE_DARK'])

    ax4.set_ylim(0, 1.25)
    path_4 = os.path.join(OUTPUT_DIR, "previous_visits_vs_readmission.png")
    show_or_save_plot(path_4, plt)

    # -------------------------------------------------------------
    # 5. NUMERICAL FEATURE CORRELATION
    # -------------------------------------------------------------
    print("\n[EDA 5] Numerical feature correlation")
    num_features = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses'
    ]
    corr_df = df[num_features].apply(pd.to_numeric, errors='coerce').corr()
    print("  Evaluated 8 core numerical clinical features (excluding identifiers and encounter IDs)")
    print(f"  Top positive correlation: num_medications vs time_in_hospital (r = {corr_df.loc['num_medications', 'time_in_hospital']:.2f})")
    print(f"  Secondary correlation:   num_medications vs num_procedures (r = {corr_df.loc['num_medications', 'num_procedures']:.2f})")
    print(f"  Prior visit correlation: number_emergency vs number_inpatient (r = {corr_df.loc['number_emergency', 'number_inpatient']:.2f})")

    fig5, ax5 = plt.subplots(figsize=(8.5, 6.5), facecolor=PALETTE['SOFT_LINEN'])
    ax5.set_facecolor("#FFFFFF")
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="Oranges", cbar=True,
                linewidths=1, linecolor=PALETTE['WARM_SAND'], annot_kws={"size": 9.5}, ax=ax5)
    ax5.set_title("Numerical Feature Correlation", fontsize=13,
                  fontweight='bold', color=PALETTE['SLATE_DARK'], pad=14)

    path_5 = os.path.join(OUTPUT_DIR, "numerical_correlation.png")
    show_or_save_plot(path_5, plt)

    # Final summary output
    print("\n" + "=" * 65)
    print("[EDA] 5 major visualizations generated successfully.")
    print("[EDA] Visualizations saved to:")
    print(OUTPUT_DIR)
    print("=" * 65 + "\n")

    # Generate summary report file
    report_content = f"""============================================================
DIABETES READMISSION — 5 MAJOR EDA VISUALIZATIONS REPORT
============================================================
Dataset:                     {RAW_DATA_PATH}
Encounters (Rows):           {rows:,}
Columns:                     {cols}
Visualizations Output Dir:   {OUTPUT_DIR}

1. Readmission Distribution:
{target_counts.to_string()}

2. Age vs Readmission (Counts by Age Cohort):
{age_cross.to_string()}

3. Hospital Utilization Statistics:
- time_in_hospital:   Mean {util_stats['time_in_hospital'][0]:.2f}, Median {util_stats['time_in_hospital'][1]:.0f}
- num_lab_procedures: Mean {util_stats['num_lab_procedures'][0]:.2f}, Median {util_stats['num_lab_procedures'][1]:.0f}
- num_medications:    Mean {util_stats['num_medications'][0]:.2f}, Median {util_stats['num_medications'][1]:.0f}

4. Previous Visits vs Readmission (Mean Visits):
{mean_visits.to_string()}

5. Numerical Correlation Matrix:
{corr_df.round(2).to_string()}

Generated Figures:
1. {path_1}
2. {path_2}
3. {path_3}
4. {path_4}
5. {path_5}
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)


if __name__ == "__main__":
    run_eda()
