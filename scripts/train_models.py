#!/usr/bin/env python3
"""
train_models.py
----------------
PHASE 6 - Machine Learning.

Trains, evaluates, and compares seven classifiers (Logistic Regression,
Decision Tree, Random Forest, Gradient Boosting [used in place of
XGBoost -- see note below], SVM, KNN, Naive Bayes) to predict a target
label from the engineered features.

TARGET AVAILABILITY
--------------------
The task brief is explicit: this phase runs "when Attendance and CO
Attainment become available." As of the current dataset, BOTH
`Attendance` and `CO_Attainment` in Feature_Dataset.xlsx are blank
(their source files have not been supplied yet). Training a model
against an empty target would either crash or force this script to
invent labels -- neither is acceptable.

This script therefore:
  1. Checks CANDIDATE_TARGET_COLUMNS (config.py) for enough non-null,
     multi-class data to train on.
  2. If a usable target is found, runs the FULL modeling pipeline below
     and writes Model_Report.md with real metrics.
  3. If no usable target is found (the current situation), it writes a
     Model_Report.md explaining exactly why training was skipped, and
     exits cleanly (exit code 0) -- no fabricated numbers, no crash.

Once Attendance.xlsx or a CO-Attainment source file is added upstream
and the pipeline is re-run end-to-end (run.py), this script will
automatically detect the new column and train for real -- no code
changes required.

Note on XGBoost: the execution environment used to build this project
has no internet access, so the `xgboost` package could not be
installed. `GradientBoostingClassifier` (scikit-learn) is used as a
readily-available substitute with comparable behavior. If `xgboost` is
available in your environment, install it and swap in
`xgboost.XGBClassifier` inside `build_model_zoo()` below -- the rest of
the pipeline (splitting, evaluation, comparison, saving) needs no
changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from utils import get_logger, read_excel_safe

logger = get_logger("train_models")

MIN_LABELED_SAMPLES = 20
MIN_CLASSES = 2


def diagnose_target_columns(feat: pd.DataFrame) -> list[dict]:
    """Return a diagnostic record for every candidate target column so the
    report can explain PRECISELY why training was or wasn't possible --
    distinguishing 'column is blank / not enough data' from 'column is
    populated but has only one class' (which is a real, reportable
    finding about the cohort, not a data-pipeline problem)."""
    diagnostics = []
    for col in config.CANDIDATE_TARGET_COLUMNS:
        if col not in feat.columns:
            diagnostics.append({"column": col, "status": "not present in Feature_Dataset.xlsx"})
            continue
        non_null = feat[col].dropna()
        n_labeled = len(non_null)
        n_classes = non_null.nunique()
        if n_labeled < MIN_LABELED_SAMPLES:
            diagnostics.append({
                "column": col, "status": f"only {n_labeled} labeled row(s) "
                f"(need >= {MIN_LABELED_SAMPLES})",
            })
        elif n_classes < MIN_CLASSES:
            value_summary = non_null.value_counts().to_dict()
            diagnostics.append({
                "column": col, "status": f"{n_labeled} labeled row(s) but only "
                f"{n_classes} distinct class present -- {value_summary}. No variation to "
                f"classify: every student falls in the same class.",
            })
        else:
            diagnostics.append({"column": col, "status": f"USABLE -- {n_labeled} labeled rows, {n_classes} classes"})
    return diagnostics


def find_usable_target(feat: pd.DataFrame) -> str | None:
    for col in config.CANDIDATE_TARGET_COLUMNS:
        if col not in feat.columns:
            continue
        non_null = feat[col].dropna()
        if len(non_null) >= MIN_LABELED_SAMPLES and non_null.nunique() >= MIN_CLASSES:
            return col
    return None


def prepare_target(series: pd.Series) -> tuple[pd.Series, str]:
    """Return a classification-ready label series plus a human-readable
    note describing how it was derived (used verbatim as-is if it's
    already categorical/low-cardinality; median-split into two classes
    otherwise)."""
    if series.nunique() <= 10:
        return series.astype("category").cat.codes, (
            f"Used '{series.name}' directly as a categorical label "
            f"({series.nunique()} classes)."
        )
    median = series.median()
    binned = (series > median).astype(int)
    note = (f"'{series.name}' is continuous ({series.nunique()} unique values); binned into two "
            f"classes via a median split (median={median:.2f}). Replace this rule with the "
            f"institution's actual CO-attainment level thresholds once defined.")
    return binned, note


def build_model_zoo() -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED),
        "DecisionTree": DecisionTreeClassifier(random_state=config.RANDOM_SEED),
        "RandomForest": RandomForestClassifier(random_state=config.RANDOM_SEED),
        "GradientBoosting": GradientBoostingClassifier(random_state=config.RANDOM_SEED),
        "SVM": SVC(probability=True, random_state=config.RANDOM_SEED),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
    }


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    scaled_cols = [c for c in df.columns if c.endswith("_scaled")]
    return df[scaled_cols]


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1_Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    try:
        if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics["ROC_AUC"] = roc_auc_score(y_test, y_proba)
        else:
            metrics["ROC_AUC"] = np.nan
    except Exception:
        metrics["ROC_AUC"] = np.nan
    metrics["Confusion_Matrix"] = confusion_matrix(y_test, y_pred).tolist()
    return metrics


def write_skip_report(reason: str, diagnostics: list[dict] | None = None) -> None:
    lines = ["# Model Report", ""]
    lines.append("## Status: Training Skipped")
    lines.append("")
    lines.append(reason)
    lines.append("")
    lines.append("## Candidate Target Columns Checked")
    if diagnostics:
        lines.append("| Column | Diagnosis |")
        lines.append("|---|---|")
        for d in diagnostics:
            lines.append(f"| {d['column']} | {d['status']} |")
    else:
        lines.append(f"`{', '.join(config.CANDIDATE_TARGET_COLUMNS)}` "
                     f"(minimum {MIN_LABELED_SAMPLES} labeled rows and {MIN_CLASSES} classes required).")
    lines.append("")
    lines.append("## What Happens Once Data Is Available")
    lines.append(
        "As soon as a target column has at least two represented classes (e.g. once "
        "Attendance data is added, or once this cohort/future cohort actually contains "
        "students below the CO-attainment threshold), re-running `python run.py` (or "
        "`python scripts/train_models.py` directly) will automatically detect it and "
        "train/evaluate all seven models below with no code changes:"
    )
    for model_name in config.MODEL_LIST:
        lines.append(f"- {model_name}")
    lines.append("")
    config.MODEL_REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.warning(reason)
    logger.info("Model report (skipped) written -> %s", config.MODEL_REPORT)


def main():
    logger.info("PHASE 6: Machine learning starting.")

    train_path, val_path, test_path = (
        config.PREPROCESSED_TRAIN_FILE, config.PREPROCESSED_VAL_FILE, config.PREPROCESSED_TEST_FILE,
    )
    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        write_skip_report(
            "Preprocessed train/validation/test files were not found. Run "
            "`python scripts/preprocessing.py` (or `python run.py`) before training models."
        )
        return None

    train_df = read_excel_safe(train_path, logger)
    val_df = read_excel_safe(val_path, logger)
    test_df = read_excel_safe(test_path, logger)
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    target_col = find_usable_target(full_df)
    if target_col is None:
        diagnostics = diagnose_target_columns(full_df)
        usable_none = all("USABLE" not in d["status"] for d in diagnostics)
        if any("distinct class present" in d["status"] for d in diagnostics):
            reason = (
                "A target column exists and is populated, but has zero class variation -- "
                "every student falls in the same class, so there is nothing for a classifier "
                "to distinguish. This is a genuine finding about the current cohort, not a "
                "pipeline error (see the per-column diagnosis below)."
            )
        else:
            reason = (
                "No usable target column was found. `At_Risk` is derived automatically from "
                "CO-attainment marks (see feature_engineering.py), but `CO_Attainment` and "
                "`Attendance` (the raw placeholder columns) are still blank since their source "
                "files have not been supplied yet. Model training was skipped rather than "
                "fabricating labels or predictions."
            )
        write_skip_report(reason, diagnostics)
        return None

    logger.info("Usable target column found: '%s'. Proceeding with training.", target_col)

    y_train_raw = train_df[target_col]
    y_val_raw = val_df[target_col]
    y_test_raw = test_df[target_col]

    # train_df/val_df/test_df each carry their own independent 0-based
    # index (they were read from three separate Excel files), so a plain
    # pd.concat + .loc split-back would silently misalign rows. Reset to
    # a single unique index first, remembering the original split sizes
    # so we can slice back by POSITION, not by (colliding) label.
    n_train, n_val = len(y_train_raw), len(y_val_raw)
    y_all_raw = pd.concat([y_train_raw, y_val_raw, y_test_raw], ignore_index=True)
    y_all_prepared, target_note = prepare_target(y_all_raw)
    y_train = y_all_prepared.iloc[:n_train].set_axis(y_train_raw.index)
    y_val = y_all_prepared.iloc[n_train:n_train + n_val].set_axis(y_val_raw.index)
    y_test = y_all_prepared.iloc[n_train + n_val:].set_axis(y_test_raw.index)

    X_train = get_feature_matrix(train_df)
    X_val = get_feature_matrix(val_df)
    X_test = get_feature_matrix(test_df)

    models = build_model_zoo()
    results = {}
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        logger.info("Training %s ...", name)
        model.fit(X_train, y_train)
        val_metrics = evaluate(model, X_val, y_val)
        test_metrics = evaluate(model, X_test, y_test)
        results[name] = {"validation": val_metrics, "test": test_metrics}
        joblib.dump(model, config.MODELS_DIR / f"{name}.joblib")
        logger.info("%s -> test accuracy=%.3f, F1=%.3f", name, test_metrics["Accuracy"], test_metrics["F1_Score"])

    write_model_report(target_col, target_note, results)
    logger.info("PHASE 6 complete.")
    return results


def write_model_report(target_col: str, target_note: str, results: dict) -> None:
    lines = ["# Model Report", ""]
    lines.append(f"## Status: Trained on target `{target_col}`")
    lines.append("")
    lines.append(f"**Target preparation:** {target_note}")
    lines.append("")
    lines.append("## Model Comparison (Test Set)")
    lines.append("| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |")
    lines.append("|---|---|---|---|---|---|")
    for name, res in results.items():
        m = res["test"]
        roc = "n/a" if pd.isna(m["ROC_AUC"]) else round(m["ROC_AUC"], 3)
        lines.append(f"| {name} | {round(m['Accuracy'], 3)} | {round(m['Precision'], 3)} | "
                     f"{round(m['Recall'], 3)} | {round(m['F1_Score'], 3)} | {roc} |")
    lines.append("")

    lines.append("## Confusion Matrices (Test Set)")
    for name, res in results.items():
        lines.append(f"**{name}:** `{res['test']['Confusion_Matrix']}`")
    lines.append("")

    best_model = max(results.items(), key=lambda kv: kv[1]["test"]["F1_Score"])
    lines.append(f"## Best Model (by test F1 score): **{best_model[0]}**")
    lines.append("")

    config.MODEL_REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Model report written -> %s", config.MODEL_REPORT)


if __name__ == "__main__":
    main()
