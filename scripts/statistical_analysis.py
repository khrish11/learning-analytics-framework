#!/usr/bin/env python3
"""
statistical_analysis.py
-------------------------
PHASE 4 - Statistical Analysis.

Computes, for every numeric engineered feature in Feature_Dataset.xlsx:
mean, median, variance, standard deviation, skewness, kurtosis.

Also computes Pearson and Spearman correlation matrices, and highlights
the strongest correlations with each theory test total (as a proxy
"outcome of interest" until CO_Attainment becomes available).

Writes reports/Statistical_Report.md. Runs independently:
`python scripts/statistical_analysis.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from utils import get_logger, read_excel_safe

logger = get_logger("statistical_analysis")


def descriptive_stats_table(numeric: pd.DataFrame) -> str:
    rows = []
    for col in numeric.columns:
        series = numeric[col].dropna()
        if series.empty:
            continue
        rows.append({
            "Feature": col,
            "Mean": round(series.mean(), 3),
            "Median": round(series.median(), 3),
            "Variance": round(series.var(ddof=0), 3),
            "Std_Dev": round(series.std(ddof=0), 3),
            "Skewness": round(stats.skew(series), 3) if len(series) > 2 else np.nan,
            "Kurtosis": round(stats.kurtosis(series), 3) if len(series) > 3 else np.nan,
        })
    table = pd.DataFrame(rows)
    lines = ["| Feature | Mean | Median | Variance | Std Dev | Skewness | Kurtosis |",
             "|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r['Feature']} | {r['Mean']} | {r['Median']} | {r['Variance']} | "
                     f"{r['Std_Dev']} | {r['Skewness']} | {r['Kurtosis']} |")
    return "\n".join(lines)


def correlation_tables(numeric: pd.DataFrame) -> tuple[str, str]:
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    pearson = numeric.corr(method="pearson").round(2)
    spearman = numeric.corr(method="spearman").round(2)

    def _fmt(corr: pd.DataFrame) -> str:
        cols = corr.columns.tolist()
        header = "| Feature | " + " | ".join(cols) + " |"
        sep = "|---|" + "|".join(["---"] * len(cols)) + "|"
        lines = [header, sep]
        for idx in corr.index:
            row_vals = " | ".join(str(v) for v in corr.loc[idx])
            lines.append(f"| {idx} | {row_vals} |")
        return "\n".join(lines)

    return _fmt(pearson), _fmt(spearman)


def strongest_correlations(numeric: pd.DataFrame, target_cols: list[str], top_n: int = 5) -> str:
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    pearson = numeric.corr(method="pearson")
    lines = []
    for target in target_cols:
        if target not in pearson.columns:
            continue
        series = pearson[target].drop(labels=[target], errors="ignore").dropna()
        top = series.reindex(series.abs().sort_values(ascending=False).index).head(top_n)
        lines.append(f"**Top {top_n} features correlated with `{target}`:**")
        lines.append("")
        for feat_name, val in top.items():
            lines.append(f"- {feat_name}: r = {round(val, 3)}")
        lines.append("")
    return "\n".join(lines) if lines else "No target columns available for correlation ranking."


def main() -> None:
    logger.info("PHASE 4: Statistical analysis starting.")
    feat = read_excel_safe(config.FEATURE_DATASET_FILE, logger)
    numeric = feat.select_dtypes(include=[np.number]).dropna(axis=1, how="all")

    desc_table = descriptive_stats_table(numeric)
    pearson_table, spearman_table = correlation_tables(numeric)
    target_candidates = [c for c in ("TT3_Total", "Average_Internal_Marks") if c in numeric.columns]
    top_corr = strongest_correlations(numeric, target_candidates)

    lines = ["# Statistical Report", ""]
    lines.append(f"Computed over `output/Feature_Dataset.xlsx` ({len(feat)} students, "
                 f"{numeric.shape[1]} numeric features analyzed).")
    lines.append("")
    lines.append("## Descriptive Statistics")
    lines.append("Mean, median, variance, standard deviation, skewness, and excess kurtosis "
                 "for every numeric engineered feature.")
    lines.append("")
    lines.append(desc_table)
    lines.append("")
    lines.append("## Pearson Correlation Matrix")
    lines.append(pearson_table)
    lines.append("")
    lines.append("## Spearman Correlation Matrix")
    lines.append(spearman_table)
    lines.append("")
    lines.append("## Strongest Correlations")
    lines.append(
        "Since `CO_Attainment` is not yet available, correlations below are ranked against "
        "`TT3_Total` (the most recent theory test) and `Average_Internal_Marks` as interim "
        "outcomes of interest for early-prediction analysis."
    )
    lines.append("")
    lines.append(top_corr)
    lines.append("")

    config.STATISTICAL_REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Statistical report written -> %s", config.STATISTICAL_REPORT)
    logger.info("PHASE 4 complete.")


if __name__ == "__main__":
    main()
