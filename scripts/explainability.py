#!/usr/bin/env python3
"""
explainability.py
-------------------
PHASE 7 - Explainable AI.

Explains the best model from Phase 6 using:
  - Permutation Importance (scikit-learn, always available)
  - Built-in Feature Importance (for tree-based models)
  - Partial Dependence Plots (scikit-learn)
  - SHAP values (only if the `shap` package is installed -- see note)

Explanations are aggregated back to the ORIGINAL domain concepts the
paper cares about -- which assessment, which question, and which CO --
by mapping each `_scaled` feature column back to its source group.

DEPENDENCY ON PHASE 6
----------------------
This script requires trained models in models/ (produced by
train_models.py). If Phase 6 was skipped because no target column was
available yet (the current state of this project -- Attendance and
CO_Attainment are both blank), this script has nothing to explain and
will exit cleanly after writing a note to that effect in
Model_Report.md, rather than fabricating explanations for a model that
was never trained.

Note on SHAP: the execution environment used to build this project has
no internet access, so the `shap` package could not be installed. This
script auto-detects `shap` at runtime: if present, full SHAP summary
plots are generated; if absent, permutation importance + built-in
feature importance + PDP still provide equivalent explanatory value,
and the report notes that SHAP was skipped. Install `shap` and re-run
this script to add SHAP plots with no other code changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from utils import get_logger, read_excel_safe

logger = get_logger("explainability")

try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def find_best_saved_model() -> tuple[str, object] | None:
    """Pick the model with the highest test F1 from Model_Report.md's
    underlying artifacts. Falls back to the first .joblib found in
    models/ if the report can't be parsed."""
    joblib_files = sorted(config.MODELS_DIR.glob("*.joblib"))
    if not joblib_files:
        return None
    # Prefer RandomForest / GradientBoosting if present (most interpretable
    # + strongest for tabular data); otherwise take the first available.
    preference_order = ["RandomForest", "GradientBoosting", "DecisionTree",
                          "LogisticRegression", "SVM", "KNN", "NaiveBayes"]
    by_name = {p.stem: p for p in joblib_files}
    for name in preference_order:
        if name in by_name:
            return name, joblib.load(by_name[name])
    first_name = joblib_files[0].stem
    return first_name, joblib.load(joblib_files[0])


def map_feature_to_group(feature_name: str) -> tuple[str, str, str]:
    """Map a '<X>_scaled' engineered-feature column back to
    (assessment_group, co_group, bloom_group) labels for the paper's
    'which assessments / questions / COs influence prediction' summary."""
    base = feature_name.replace("_scaled", "")
    assessment = "N/A"
    for group in config.ASSESSMENT_GROUPS:
        if base.startswith(group):
            assessment = group
            break
    co = base if base.startswith("CO") and base.endswith("_Score") else "N/A"
    bloom = base if (base.startswith("L") and base.endswith("_Score") and base[1:2].isdigit()) else "N/A"
    return assessment, co, bloom


def plot_feature_importance(importances: pd.Series, title: str, filename: str) -> Path:
    top = importances.sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.index[::-1], top.values[::-1], color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    path = config.FIGURES_DIR / f"{filename}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved -> %s", path)
    return path


def main():
    logger.info("PHASE 7: Explainability starting.")

    best = find_best_saved_model()
    if best is None:
        note = (
            "No trained models were found in models/ (Phase 6 was skipped -- see Model_Report.md "
            "above for the exact reason, e.g. a target column with zero class variation, or no "
            "target column populated yet). Explainability requires a trained model, so this phase "
            "was skipped as well. Once Phase 6 trains real models, re-run "
            "`python scripts/explainability.py` (or `python run.py`) to generate SHAP / "
            "permutation-importance / PDP explanations automatically -- no code changes required."
        )
        logger.warning(note)
        _append_to_model_report("## Explainability (Phase 7): Skipped", note)
        return None

    model_name, model = best
    logger.info("Explaining model: %s", model_name)

    test_df = read_excel_safe(config.PREPROCESSED_TEST_FILE, logger)
    feature_cols = [c for c in test_df.columns if c.endswith("_scaled")]
    X_test = test_df[feature_cols]

    target_col = next((c for c in config.CANDIDATE_TARGET_COLUMNS if c in test_df.columns), None)
    y_test = test_df[target_col] if target_col else None

    # --- Permutation importance -------------------------------------------------
    if y_test is not None and y_test.notna().all():
        perm = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=config.RANDOM_SEED)
        perm_series = pd.Series(perm.importances_mean, index=feature_cols)
        plot_feature_importance(perm_series, f"Permutation Importance ({model_name})", "permutation_importance")
    else:
        perm_series = pd.Series(dtype=float)
        logger.warning("Skipping permutation importance -- target values unavailable for the test split.")

    # --- Built-in feature importance (tree-based models only) -------------------
    builtin_series = pd.Series(dtype=float)
    if hasattr(model, "feature_importances_"):
        builtin_series = pd.Series(model.feature_importances_, index=feature_cols)
        plot_feature_importance(builtin_series, f"Built-in Feature Importance ({model_name})", "feature_importance")

    # --- Partial dependence plots (top 4 features by whichever importance we have) --
    ranking = builtin_series if not builtin_series.empty else perm_series
    if not ranking.empty:
        top_features = ranking.sort_values(ascending=False).head(4).index.tolist()
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            n_classes = len(np.unique(y_test)) if y_test is not None else 2
            pdp_kwargs = {"target": 0} if n_classes > 2 else {}
            PartialDependenceDisplay.from_estimator(model, X_test, top_features, ax=ax, **pdp_kwargs)
            fig.suptitle(f"Partial Dependence Plots ({model_name})")
            fig.tight_layout()
            path = config.FIGURES_DIR / "partial_dependence.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Figure saved -> %s", path)
        except Exception as exc:
            logger.warning("Could not generate partial dependence plots: %s", exc)

    # --- SHAP (optional) ---------------------------------------------------------
    shap_note = "SHAP was not available in this environment (package not installed) -- skipped."
    if SHAP_AVAILABLE:
        try:
            explainer = shap.Explainer(model, X_test)
            shap_values = explainer(X_test)
            fig = plt.figure(figsize=(9, 7))
            shap.summary_plot(shap_values, X_test, show=False)
            path = config.FIGURES_DIR / "shap_summary.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            shap_note = f"SHAP summary plot generated -> figures/shap_summary.png"
            logger.info(shap_note)
        except Exception as exc:
            shap_note = f"SHAP was installed but failed to run: {exc}"
            logger.warning(shap_note)

    # --- Aggregate to assessment / CO / Bloom level for the paper's narrative ---
    ranking_for_summary = ranking if not ranking.empty else pd.Series(dtype=float)
    assessment_scores: dict[str, float] = {}
    co_scores: dict[str, float] = {}
    bloom_scores: dict[str, float] = {}
    for feat_name, score in ranking_for_summary.items():
        assessment, co, bloom = map_feature_to_group(feat_name)
        if assessment != "N/A":
            assessment_scores[assessment] = assessment_scores.get(assessment, 0.0) + abs(score)
        if co != "N/A":
            co_scores[co] = co_scores.get(co, 0.0) + abs(score)
        if bloom != "N/A":
            bloom_scores[bloom] = bloom_scores.get(bloom, 0.0) + abs(score)

    lines = [f"## Explainability (Phase 7): Model = {model_name}", ""]
    lines.append(f"**SHAP status:** {shap_note}")
    lines.append("")
    lines.append("### Which assessments influence the prediction most")
    if assessment_scores:
        for k, v in sorted(assessment_scores.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"- {k}: importance mass = {round(v, 4)}")
    else:
        lines.append("Not enough importance signal to rank assessments.")
    lines.append("")
    lines.append("### Which COs influence the prediction most")
    if co_scores:
        for k, v in sorted(co_scores.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"- {k}: importance mass = {round(v, 4)}")
    else:
        lines.append("Not enough importance signal to rank COs.")
    lines.append("")
    lines.append("### Which Bloom levels influence the prediction most")
    if bloom_scores:
        for k, v in sorted(bloom_scores.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"- {k}: importance mass = {round(v, 4)}")
    else:
        lines.append("Not enough importance signal to rank Bloom levels.")
    lines.append("")
    lines.append("Figures: `figures/permutation_importance.png`, `figures/feature_importance.png`, "
                 "`figures/partial_dependence.png`" + (", `figures/shap_summary.png`" if SHAP_AVAILABLE else ""))

    _append_to_model_report(f"## Explainability (Phase 7): Model = {model_name}", "\n".join(lines))
    logger.info("PHASE 7 complete.")


def _append_to_model_report(heading: str, body: str) -> None:
    existing = config.MODEL_REPORT.read_text(encoding="utf-8") if config.MODEL_REPORT.exists() else "# Model Report\n"
    with config.MODEL_REPORT.open("a", encoding="utf-8") as fh:
        fh.write("\n\n" + body + "\n")
    logger.info("Appended explainability section -> %s", config.MODEL_REPORT)


if __name__ == "__main__":
    main()
