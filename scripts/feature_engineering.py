#!/usr/bin/env python3
"""
feature_engineering.py
------------------------
PHASE 2 - Feature Engineering.

Builds output/Feature_Dataset.xlsx from Master_Dataset.xlsx (+
Question_Metadata.xlsx for CO / Bloom-level aggregation), one row per
student, containing:

Assessment features
    Quiz_Score, ABA_Score, TT1_Total, TT2_Total, TT3_Total,
    Average_Internal_Marks, Max_Score, Min_Score, Std_Dev

Learning-trend features
    TT2_minus_TT1, TT3_minus_TT2, TT3_minus_TT1, Improvement_Percentage

Question-level features
    Avg_Question_Score, Highest_Question_Score, Lowest_Question_Score

CO features (computed from Question_Metadata's CO column)
    CO1_Score ... COn_Score                 (marks scored, absolute)
    CO1_Attainment_Pct ... COn_Attainment_Pct  (% of attempted-question max)
    Overall_CO_Attainment_Pct                (marks / max across all CO-mapped
                                               questions the student attempted)
    At_Risk                                  (1 if Overall_CO_Attainment_Pct is
                                               below config.CO_ATTAINMENT_THRESHOLD_PERCENT,
                                               else 0 -- see note below)

Bloom-level features (computed from Question_Metadata's Bloom_Level column)
    L1_Score ... Ln_Score (whichever levels actually appear in the data)

Consistency features
    Coefficient_of_Variation, Performance_Stability

Attendance / CO_Attainment (the raw placeholder column, as opposed to the
derived percentages above) are copied through as-is (blank, since the
source files do not exist yet) rather than estimated.

All totals are computed only over the questions a student actually has a
mark for -- an un-attempted OR-question is excluded from the sum/mean,
never treated as a zero. This matters especially for the attainment-%
denominator: Theory Test 2 and 3 both contain OR-question pairs, so a
question a student never had the option to lose marks on (the un-chosen
half of an OR pair) is excluded from BOTH the numerator and the
denominator of their CO-attainment percentage, not just the numerator.

NOTE ON METHODOLOGY: `Overall_CO_Attainment_Pct` / `At_Risk` here are a
per-student, per-course-offering proxy computed directly from continuous
assessment marks for early-warning ML purposes. This is NOT the same
calculation as the formal NBA/OBE course-level CO attainment metric (which
is typically "% of the COHORT crossing a per-CO target level", not a
per-student percentage) -- if the paper needs to report both, keep this
distinction explicit rather than presenting the ML label as the official
NBA attainment figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from utils import get_logger, get_question_columns, group_question_columns_by_assessment, read_excel_safe, write_excel_safe

logger = get_logger("feature_engineering")


def _safe_sum(row: pd.Series, cols: list[str]) -> float:
    vals = row[cols].dropna()
    return float(vals.sum()) if len(vals) else np.nan


def build_assessment_features(master: pd.DataFrame) -> pd.DataFrame:
    groups = group_question_columns_by_assessment(master)
    feat = pd.DataFrame(index=master.index)

    # Quiz / ABA scores: sum of their (single) question column(s).
    if "Quiz1" in groups:
        feat["Quiz_Score"] = master.apply(lambda r: _safe_sum(r, groups["Quiz1"]), axis=1)
    if "ABA1" in groups:
        feat["ABA_Score"] = master.apply(lambda r: _safe_sum(r, groups["ABA1"]), axis=1)

    for tt in config.THEORY_TEST_SEQUENCE:
        if tt in groups:
            feat[f"{tt}_Total"] = master.apply(lambda r, c=groups[tt]: _safe_sum(r, c), axis=1)

    internal_cols = [c for c in feat.columns if c.endswith("_Total") or c in ("Quiz_Score", "ABA_Score")]
    feat["Average_Internal_Marks"] = feat[internal_cols].mean(axis=1, skipna=True)
    feat["Max_Score"] = feat[internal_cols].max(axis=1, skipna=True)
    feat["Min_Score"] = feat[internal_cols].min(axis=1, skipna=True)
    feat["Std_Dev"] = feat[internal_cols].std(axis=1, skipna=True, ddof=0)

    return feat


def build_learning_trend_features(feat: pd.DataFrame) -> pd.DataFrame:
    trend = pd.DataFrame(index=feat.index)
    have = lambda c: c in feat.columns

    if have("TT2_Total") and have("TT1_Total"):
        trend["TT2_minus_TT1"] = feat["TT2_Total"] - feat["TT1_Total"]
    if have("TT3_Total") and have("TT2_Total"):
        trend["TT3_minus_TT2"] = feat["TT3_Total"] - feat["TT2_Total"]
    if have("TT3_Total") and have("TT1_Total"):
        trend["TT3_minus_TT1"] = feat["TT3_Total"] - feat["TT1_Total"]
        # Improvement % relative to TT1 (guard against divide-by-zero).
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(
                feat["TT1_Total"].fillna(0) > 0,
                100 * trend["TT3_minus_TT1"] / feat["TT1_Total"],
                np.nan,
            )
        trend["Improvement_Percentage"] = pct

    return trend


def build_question_level_features(master: pd.DataFrame) -> pd.DataFrame:
    q_cols = get_question_columns(master)
    feat = pd.DataFrame(index=master.index)
    if not q_cols:
        return feat
    feat["Avg_Question_Score"] = master[q_cols].mean(axis=1, skipna=True)
    feat["Highest_Question_Score"] = master[q_cols].max(axis=1, skipna=True)
    feat["Lowest_Question_Score"] = master[q_cols].min(axis=1, skipna=True)
    return feat


def build_co_features(master: pd.DataFrame, question_metadata: pd.DataFrame) -> pd.DataFrame:
    """Sum a student's marks across every question mapped to each CO."""
    feat = pd.DataFrame(index=master.index)
    meta = question_metadata[question_metadata["CO"].astype(str).str.match(r"^CO\d+$", na=False)]
    for co in sorted(meta["CO"].unique(), key=lambda c: int(c.replace("CO", ""))):
        co_questions = meta[meta["CO"] == co]
        cols = [f"{a}_{q}" for a, q in zip(co_questions["Assessment"], co_questions["Question"])]
        cols = [c for c in cols if c in master.columns]
        if not cols:
            continue
        feat[f"{co}_Score"] = master.apply(lambda r, c=cols: _safe_sum(r, c), axis=1)
    return feat


def build_bloom_features(master: pd.DataFrame, question_metadata: pd.DataFrame) -> pd.DataFrame:
    """Sum a student's marks across every question at each Bloom's level."""
    feat = pd.DataFrame(index=master.index)
    meta = question_metadata[question_metadata["Bloom_Level"].astype(str).str.match(r"^L\d+$", na=False)]
    for level in sorted(meta["Bloom_Level"].unique(), key=lambda l: int(l.replace("L", ""))):
        level_questions = meta[meta["Bloom_Level"] == level]
        cols = [f"{a}_{q}" for a, q in zip(level_questions["Assessment"], level_questions["Question"])]
        cols = [c for c in cols if c in master.columns]
        if not cols:
            continue
        feat[f"{level}_Score"] = master.apply(lambda r, c=cols: _safe_sum(r, c), axis=1)
    return feat


def build_consistency_features(feat: pd.DataFrame) -> pd.DataFrame:
    consistency = pd.DataFrame(index=feat.index)
    internal_cols = [c for c in feat.columns if c.endswith("_Total") or c in ("Quiz_Score", "ABA_Score")]
    if not internal_cols:
        return consistency

    mean_ = feat[internal_cols].mean(axis=1, skipna=True)
    std_ = feat[internal_cols].std(axis=1, skipna=True, ddof=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        cov = np.where(mean_ > 0, std_ / mean_, np.nan)
    consistency["Coefficient_of_Variation"] = cov

    # Performance_Stability: a normalized 0-1 score where 1 = perfectly
    # consistent (CoV = 0) and 0 = highly volatile. Values are clipped to
    # [0, 1] since CoV can technically exceed 1 for very erratic students.
    consistency["Performance_Stability"] = (1 - consistency["Coefficient_of_Variation"]).clip(lower=0, upper=1)

    return consistency


def build_co_attainment_features(master: pd.DataFrame, question_metadata: pd.DataFrame) -> pd.DataFrame:
    """Compute each student's CO-attainment percentage and derive the
    At_Risk label from config.CO_ATTAINMENT_THRESHOLD_PERCENT.

    Denominator handling for OR-questions: a question's Max_Marks only
    counts toward a student's denominator if that student actually
    attempted it (has a non-null mark). This keeps the un-chosen half of
    an OR pair out of BOTH the numerator and the denominator, so a
    student who took the Module 2 OR option is compared against the
    Module 2 max, not against a combined Module-2-plus-Module-2-OR max
    they were never able to earn.
    """
    feat = pd.DataFrame(index=master.index)
    meta = question_metadata[question_metadata["CO"].astype(str).str.match(r"^CO\d+$", na=False)].copy()

    co_names = sorted(meta["CO"].unique(), key=lambda c: int(c.replace("CO", "")))

    # Running totals across ALL COs combined, for the overall percentage.
    total_attained = pd.Series(0.0, index=master.index)
    total_possible = pd.Series(0.0, index=master.index)
    any_co_data = pd.Series(False, index=master.index)

    for co in co_names:
        co_questions = meta[meta["CO"] == co]
        col_max_pairs = [
            (f"{a}_{q}", m) for a, q, m in
            zip(co_questions["Assessment"], co_questions["Question"], co_questions["Max_Marks"])
            if f"{a}_{q}" in master.columns
        ]
        if not col_max_pairs:
            continue

        attained = pd.Series(0.0, index=master.index)
        possible = pd.Series(0.0, index=master.index)
        attempted_any = pd.Series(False, index=master.index)

        for col, max_marks in col_max_pairs:
            attempted_mask = master[col].notna()
            attained = attained + master[col].where(attempted_mask, 0.0)
            if pd.notna(max_marks):
                possible = possible + np.where(attempted_mask, max_marks, 0.0)
            attempted_any = attempted_any | attempted_mask

        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(possible > 0, 100 * attained / possible, np.nan)
        feat[f"{co}_Attainment_Pct"] = np.where(attempted_any, pct, np.nan)

        total_attained = total_attained + attained
        total_possible = total_possible + possible
        any_co_data = any_co_data | attempted_any

    if co_names:
        with np.errstate(divide="ignore", invalid="ignore"):
            overall_pct = np.where(total_possible > 0, 100 * total_attained / total_possible, np.nan)
        feat["Overall_CO_Attainment_Pct"] = np.where(any_co_data, overall_pct, np.nan)

        at_risk = np.where(
            feat["Overall_CO_Attainment_Pct"].notna(),
            (feat["Overall_CO_Attainment_Pct"] < config.CO_ATTAINMENT_THRESHOLD_PERCENT).astype("Int64"),
            pd.NA,
        )
        feat["At_Risk"] = pd.array(at_risk, dtype="Int64")

    return feat


def build_feature_dataset(master: pd.DataFrame, question_metadata: pd.DataFrame) -> pd.DataFrame:
    identity = master[[c for c in config.IDENTITY_COLUMNS if c in master.columns]].copy()

    assessment_feat = build_assessment_features(master)
    trend_feat = build_learning_trend_features(assessment_feat)
    question_feat = build_question_level_features(master)
    co_feat = build_co_features(master, question_metadata)
    co_attainment_feat = build_co_attainment_features(master, question_metadata)
    bloom_feat = build_bloom_features(master, question_metadata)
    consistency_feat = build_consistency_features(assessment_feat)

    placeholders = master[[c for c in config.PLACEHOLDER_COLUMNS if c in master.columns]].copy()

    feature_df = pd.concat(
        [identity, assessment_feat, trend_feat, question_feat, co_feat, co_attainment_feat,
         bloom_feat, consistency_feat, placeholders],
        axis=1,
    )
    return feature_df


def main() -> pd.DataFrame:
    logger.info("PHASE 2: Feature engineering starting.")
    master = read_excel_safe(config.MASTER_DATASET_FILE, logger)
    question_metadata = read_excel_safe(config.QUESTION_METADATA_FILE, logger)

    feature_df = build_feature_dataset(master, question_metadata)
    write_excel_safe(feature_df, config.FEATURE_DATASET_FILE, logger)

    _write_feature_engineering_report(feature_df)
    logger.info("PHASE 2 complete.")
    return feature_df


def _write_feature_engineering_report(feature_df: pd.DataFrame) -> None:
    engineered_cols = [c for c in feature_df.columns if c not in config.IDENTITY_COLUMNS + config.PLACEHOLDER_COLUMNS]

    lines = ["# Feature Engineering Report", ""]
    lines.append(f"`Feature_Dataset.xlsx` contains **{len(feature_df)} students** and "
                 f"**{len(engineered_cols)} engineered features** (plus identity columns and the "
                 f"blank Attendance / CO_Attainment placeholders).")
    lines.append("")
    lines.append("## Feature Groups")

    groups = {
        "Assessment features": ["Quiz_Score", "ABA_Score", "TT1_Total", "TT2_Total", "TT3_Total",
                                  "Average_Internal_Marks", "Max_Score", "Min_Score", "Std_Dev"],
        "Learning trend features": ["TT2_minus_TT1", "TT3_minus_TT2", "TT3_minus_TT1", "Improvement_Percentage"],
        "Question-level features": ["Avg_Question_Score", "Highest_Question_Score", "Lowest_Question_Score"],
        "CO features (absolute marks)": [c for c in engineered_cols if c.startswith("CO") and c.endswith("_Score")],
        "CO attainment % + At_Risk label": [c for c in engineered_cols if c.endswith("_Attainment_Pct") or c == "At_Risk"],
        "Bloom-level features": [c for c in engineered_cols if c.startswith("L") and c.endswith("_Score") and c[1].isdigit()],
        "Consistency features": ["Coefficient_of_Variation", "Performance_Stability"],
    }
    for group_name, cols in groups.items():
        present = [c for c in cols if c in feature_df.columns]
        lines.append(f"- **{group_name}** ({len(present)}): {', '.join(present) if present else 'none present'}")
    lines.append("")

    lines.append("## Missing-Value Notes")
    lines.append(
        "Totals (e.g. `TT2_Total`) are computed only over questions the student actually has a "
        "mark for; an un-attempted OR-question is excluded from the sum rather than counted as "
        "zero, so totals remain comparable across students who took different OR options."
    )
    lines.append("`Attendance` and `CO_Attainment` (the raw placeholder columns) are carried "
                 "through unchanged (blank) since their source files have not been supplied yet.")
    lines.append("")

    if "At_Risk" in feature_df.columns:
        lines.append("## At-Risk Label Definition")
        lines.append(
            f"`At_Risk` = 1 if a student's `Overall_CO_Attainment_Pct` is below "
            f"**{config.CO_ATTAINMENT_THRESHOLD_PERCENT}%**, else 0. `Overall_CO_Attainment_Pct` is "
            f"computed as (marks scored on CO-mapped questions the student attempted) / (maximum "
            f"marks of those SAME attempted questions) x 100 -- OR-question pairs are handled by "
            f"only counting the max marks of the option the student actually attempted, so a "
            f"student is never penalized for the un-chosen half of an OR pair."
        )
        lines.append("")
        lines.append(
            "**Methodology note:** this is a per-student proxy computed directly from continuous "
            "assessment data for early-warning ML purposes, not the formal NBA/OBE cohort-level CO "
            "attainment metric. Keep this distinction explicit in the paper if both are discussed."
        )
        lines.append("")
        value_counts = feature_df["At_Risk"].value_counts(dropna=True).sort_index()
        n_at_risk = int(value_counts.get(1, 0))
        n_not_at_risk = int(value_counts.get(0, 0))
        n_total = n_at_risk + n_not_at_risk
        pct_at_risk = round(100 * n_at_risk / n_total, 1) if n_total else 0.0
        lines.append(f"**Class balance:** {n_at_risk} At-Risk ({pct_at_risk}%), {n_not_at_risk} Not At-Risk, "
                     f"out of {n_total} students with a computable attainment percentage.")
        lines.append("")

    lines.append("## Summary Statistics (engineered numeric features)")
    numeric_engineered = feature_df[engineered_cols].select_dtypes(include=[np.number])
    if not numeric_engineered.empty:
        desc = numeric_engineered.describe().T.round(2)
        lines.append("| Feature | count | mean | std | min | 25% | 50% | 75% | max |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for feat_name, row in desc.iterrows():
            lines.append(
                f"| {feat_name} | {int(row['count'])} | {row['mean']} | {row['std']} | "
                f"{row['min']} | {row['25%']} | {row['50%']} | {row['75%']} | {row['max']} |"
            )
    lines.append("")

    config.FEATURE_ENGINEERING_REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Feature engineering report written -> %s", config.FEATURE_ENGINEERING_REPORT)


if __name__ == "__main__":
    main()
