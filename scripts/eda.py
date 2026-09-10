#!/usr/bin/env python3
"""
eda.py
------
PHASE 3 - Exploratory Data Analysis.

Reads output/Feature_Dataset.xlsx and produces:

  - Summary statistics, correlation matrix, missing-value report, and an
    IQR-based outlier report (all embedded in reports/EDA_Report.md)
  - Publication-quality figures saved to figures/:
      * Histograms: Quiz, TT1, TT2, TT3, ABA
      * Correlation heatmap
      * Student performance distribution (Average_Internal_Marks)
      * CO-wise performance (bar chart of mean CO scores)
      * Bloom-level performance (bar chart of mean L-level scores)
      * Learning trend (TT1 -> TT2 -> TT3 mean line plot)
      * Boxplots (assessment totals side by side)
      * Scatter plots (TT1 vs TT3, Average_Internal_Marks vs Avg_Question_Score)
      * Pair plot (assessment totals)

Runs independently: `python scripts/eda.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / no display backend needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from utils import get_logger, read_excel_safe

logger = get_logger("eda")

sns.set_theme(style="whitegrid", context="talk")
FIGSIZE = (9, 6)
DPI = 150


def _savefig(fig, name: str) -> Path:
    path = config.FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved -> %s", path)
    return path


def plot_histograms(feat: pd.DataFrame) -> list[str]:
    generated = []
    hist_specs = [
        ("Quiz_Score", "Quiz 1 Marks Distribution", "#4C72B0"),
        ("TT1_Total", "Theory Test 1 Marks Distribution", "#55A868"),
        ("TT2_Total", "Theory Test 2 Marks Distribution", "#C44E52"),
        ("TT3_Total", "Theory Test 3 Marks Distribution", "#8172B2"),
        ("ABA_Score", "ABA 1 Marks Distribution", "#CCB974"),
    ]
    for col, title, color in hist_specs:
        if col not in feat.columns:
            continue
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.histplot(feat[col].dropna(), kde=True, color=color, ax=ax, bins=10)
        ax.set_title(title)
        ax.set_xlabel("Marks")
        ax.set_ylabel("Number of Students")
        name = f"histogram_{col.lower()}"
        _savefig(fig, name)
        generated.append(name)
    return generated


def plot_correlation_heatmap(feat: pd.DataFrame) -> str | None:
    numeric = feat.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]  # drop constant columns
    if numeric.shape[1] < 2:
        logger.warning("Not enough numeric variability to draw a correlation heatmap.")
        return None
    corr = numeric.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(corr)), max(8, 0.5 * len(corr))))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Correlation Heatmap of Engineered Features")
    return _savefig(fig, "correlation_heatmap").stem


def plot_performance_distribution(feat: pd.DataFrame) -> str | None:
    if "Average_Internal_Marks" not in feat.columns:
        return None
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.histplot(feat["Average_Internal_Marks"].dropna(), kde=True, color="#4C72B0", ax=ax, bins=12)
    ax.set_title("Student Performance Distribution (Average Internal Marks)")
    ax.set_xlabel("Average Internal Marks")
    ax.set_ylabel("Number of Students")
    return _savefig(fig, "student_performance_distribution").stem


def plot_co_performance(feat: pd.DataFrame) -> str | None:
    co_cols = [c for c in feat.columns if c.startswith("CO") and c.endswith("_Score")]
    if not co_cols:
        return None
    means = feat[co_cols].mean().sort_index()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(x=means.index, y=means.values, hue=means.index, palette="viridis", legend=False, ax=ax)
    ax.set_title("Course Outcome (CO)-wise Average Performance")
    ax.set_xlabel("Course Outcome")
    ax.set_ylabel("Average Score (marks)")
    return _savefig(fig, "co_wise_performance").stem


def plot_bloom_performance(feat: pd.DataFrame) -> str | None:
    bloom_cols = [c for c in feat.columns if c.startswith("L") and c.endswith("_Score") and c[1:].split("_")[0].isdigit()]
    if not bloom_cols:
        return None
    means = feat[bloom_cols].mean().sort_index()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(x=means.index, y=means.values, hue=means.index, palette="magma", legend=False, ax=ax)
    ax.set_title("Bloom's Taxonomy Level-wise Average Performance")
    ax.set_xlabel("Bloom's Level")
    ax.set_ylabel("Average Score (marks)")
    return _savefig(fig, "bloom_level_performance").stem


def plot_learning_trend(feat: pd.DataFrame) -> str | None:
    tt_cols = [c for c in ("TT1_Total", "TT2_Total", "TT3_Total") if c in feat.columns]
    if len(tt_cols) < 2:
        return None
    means = feat[tt_cols].mean()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(tt_cols, means.values, marker="o", linewidth=2.5, color="#C44E52")
    for x, y in zip(tt_cols, means.values):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center")
    ax.set_title("Class-wide Learning Trend Across Theory Tests")
    ax.set_xlabel("Assessment")
    ax.set_ylabel("Mean Marks")
    return _savefig(fig, "learning_trend").stem


def plot_boxplots(feat: pd.DataFrame) -> str | None:
    cols = [c for c in ("Quiz_Score", "ABA_Score", "TT1_Total", "TT2_Total", "TT3_Total") if c in feat.columns]
    if not cols:
        return None
    melted = feat[cols].melt(var_name="Assessment", value_name="Marks")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.boxplot(data=melted, x="Assessment", y="Marks", hue="Assessment", palette="Set2", legend=False, ax=ax)
    ax.set_title("Marks Distribution by Assessment (Boxplot)")
    return _savefig(fig, "boxplot_assessments").stem


def plot_scatter(feat: pd.DataFrame) -> list[str]:
    generated = []
    pairs = [
        ("TT1_Total", "TT3_Total", "Theory Test 1 vs Theory Test 3"),
        ("Average_Internal_Marks", "Avg_Question_Score", "Average Internal Marks vs Average Question Score"),
    ]
    for x, y, title in pairs:
        if x not in feat.columns or y not in feat.columns:
            continue
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.regplot(data=feat, x=x, y=y, ax=ax, scatter_kws={"alpha": 0.7}, line_kws={"color": "red"})
        ax.set_title(title)
        name = f"scatter_{x.lower()}_vs_{y.lower()}"
        _savefig(fig, name)
        generated.append(name)
    return generated


def plot_pairplot(feat: pd.DataFrame) -> str | None:
    cols = [c for c in ("Quiz_Score", "ABA_Score", "TT1_Total", "TT2_Total", "TT3_Total") if c in feat.columns]
    if len(cols) < 2:
        return None
    sub = feat[cols].dropna(how="all")
    grid = sns.pairplot(sub, diag_kind="kde", plot_kws={"alpha": 0.6})
    grid.fig.suptitle("Pair Plot of Assessment Totals", y=1.02)
    path = config.FIGURES_DIR / "pair_plot.png"
    grid.fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(grid.fig)
    logger.info("Figure saved -> %s", path)
    return path.stem


# ---------------------------------------------------------------------------
# Tabular analyses embedded in the report
# ---------------------------------------------------------------------------

def summary_statistics_table(feat: pd.DataFrame) -> str:
    numeric = feat.select_dtypes(include=[np.number])
    desc = numeric.describe().T.round(2)
    lines = ["| Feature | count | mean | std | min | 25% | 50% | 75% | max |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, row in desc.iterrows():
        lines.append(f"| {name} | {int(row['count'])} | {row['mean']} | {row['std']} | "
                     f"{row['min']} | {row['25%']} | {row['50%']} | {row['75%']} | {row['max']} |")
    return "\n".join(lines)


def missing_value_report(feat: pd.DataFrame) -> str:
    missing = feat.isna().sum()
    pct = (missing / len(feat) * 100).round(2)
    table = pd.DataFrame({"Missing_Count": missing, "Missing_Percent": pct})
    table = table[table["Missing_Count"] > 0].sort_values("Missing_Count", ascending=False)
    if table.empty:
        return "No missing values in any engineered feature."
    lines = ["| Column | Missing Count | Missing % |", "|---|---|---|"]
    for name, row in table.iterrows():
        lines.append(f"| {name} | {int(row['Missing_Count'])} | {row['Missing_Percent']}% |")
    return "\n".join(lines)


def outlier_report(feat: pd.DataFrame) -> str:
    numeric = feat.select_dtypes(include=[np.number])
    rows = []
    for col in numeric.columns:
        series = numeric[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        if len(outliers):
            rows.append((col, len(outliers), round(lower, 2), round(upper, 2)))
    if not rows:
        return "No IQR-based outliers detected in any engineered feature."
    lines = ["| Feature | Outlier Count | Lower Bound | Upper Bound |", "|---|---|---|---|"]
    for col, n, lower, upper in rows:
        lines.append(f"| {col} | {n} | {lower} | {upper} |")
    return "\n".join(lines)


def main() -> pd.DataFrame:
    logger.info("PHASE 3: EDA starting.")
    feat = read_excel_safe(config.FEATURE_DATASET_FILE, logger)

    figure_names: list[str] = []
    figure_names += plot_histograms(feat)
    name = plot_correlation_heatmap(feat)
    if name: figure_names.append(name)
    name = plot_performance_distribution(feat)
    if name: figure_names.append(name)
    name = plot_co_performance(feat)
    if name: figure_names.append(name)
    name = plot_bloom_performance(feat)
    if name: figure_names.append(name)
    name = plot_learning_trend(feat)
    if name: figure_names.append(name)
    name = plot_boxplots(feat)
    if name: figure_names.append(name)
    figure_names += plot_scatter(feat)
    name = plot_pairplot(feat)
    if name: figure_names.append(name)

    lines = ["# Exploratory Data Analysis (EDA) Report", ""]
    lines.append(f"Based on `output/Feature_Dataset.xlsx` ({len(feat)} students, "
                 f"{feat.shape[1]} columns).")
    lines.append("")
    lines.append("## Summary Statistics")
    lines.append(summary_statistics_table(feat))
    lines.append("")
    lines.append("## Missing Value Report")
    lines.append(missing_value_report(feat))
    lines.append("")
    lines.append("## Outlier Report (IQR method, 1.5x whiskers)")
    lines.append(outlier_report(feat))
    lines.append("")
    lines.append("## Generated Figures")
    lines.append(f"All figures below are saved as PNG files in `figures/` at {DPI} DPI.")
    for fig_name in figure_names:
        lines.append(f"- `figures/{fig_name}.png`")
    lines.append("")

    config.EDA_REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("EDA report written -> %s", config.EDA_REPORT)
    logger.info("PHASE 3 complete: %d figures generated.", len(figure_names))
    return feat


if __name__ == "__main__":
    main()
