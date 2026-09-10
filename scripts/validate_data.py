#!/usr/bin/env python3
"""
validate_data.py
-----------------
PHASE 1 - Data Validation.

Validates Students.xlsx, Assessments.xlsx, Question_Metadata.xlsx and
Master_Dataset.xlsx for:

  - Duplicate students / duplicate USNs
  - Missing values in required fields
  - Invalid marks (negative, non-numeric)
  - Marks greater than the question's Max_Marks (per Question_Metadata)
  - Invalid / unmapped question IDs (question in Assessments.xlsx that
    has no corresponding entry in Question_Metadata.xlsx, or vice versa)
  - Incorrect / missing CO mappings
  - Inconsistent student records (e.g. USN present in one dataset but
    not another)

Writes reports/Validation_Report.md summarizing every problem found and
every automatic (non-destructive) correction performed, plus data
quality statistics. Runs fully independently: `python scripts/validate_data.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from utils import get_logger, get_question_columns, read_excel_safe

logger = get_logger("validate_data")


class ValidationFindings:
    """Accumulates issues and corrections for the final Markdown report."""

    def __init__(self) -> None:
        self.issues: List[str] = []
        self.corrections: List[str] = []
        self.stats: dict[str, object] = {}

    def issue(self, text: str) -> None:
        self.issues.append(text)
        logger.warning(text)

    def correction(self, text: str) -> None:
        self.corrections.append(text)
        logger.info(text)

    def stat(self, key: str, value) -> None:
        self.stats[key] = value


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_students(students: pd.DataFrame, f: ValidationFindings) -> None:
    f.stat("Total student records", len(students))

    dup_usn = students[students["USN"].duplicated(keep=False)]
    if not dup_usn.empty:
        f.issue(f"Found {dup_usn['USN'].nunique()} duplicate USN(s) in Students.xlsx: "
                 f"{sorted(dup_usn['USN'].unique().tolist())}")
    else:
        f.stat("Duplicate USNs in Students.xlsx", 0)

    dup_id = students[students["Student_ID"].duplicated(keep=False)]
    if not dup_id.empty:
        f.issue(f"Found {dup_id['Student_ID'].nunique()} duplicate Student_ID(s): "
                 f"{sorted(dup_id['Student_ID'].unique().tolist())}")
    else:
        f.stat("Duplicate Student_IDs", 0)

    missing_name = students[students["Student_Name"].isna() |
                             (students["Student_Name"].astype(str).str.strip() == "")]
    if not missing_name.empty:
        f.issue(f"{len(missing_name)} student(s) have a missing/blank Student_Name: "
                 f"{missing_name['USN'].tolist()}")
    else:
        f.stat("Students with missing name", 0)

    missing_usn = students[students["USN"].isna() | (students["USN"].astype(str).str.strip() == "")]
    if not missing_usn.empty:
        f.issue(f"{len(missing_usn)} record(s) have a missing/blank USN.")
    else:
        f.stat("Students with missing USN", 0)


def check_assessments(assessments: pd.DataFrame, students: pd.DataFrame,
                       question_metadata: pd.DataFrame, f: ValidationFindings) -> None:
    f.stat("Total assessment mark rows", len(assessments))

    dup_key = ["USN", "Assessment", "Question"]
    dups = assessments[assessments.duplicated(subset=dup_key, keep=False)]
    if not dups.empty:
        n_dup_groups = dups.drop_duplicates(subset=dup_key).shape[0]
        f.issue(f"Found {n_dup_groups} (USN, Assessment, Question) combination(s) with "
                 f"duplicate mark rows in Assessments.xlsx.")
    else:
        f.stat("Duplicate (USN, Assessment, Question) rows", 0)

    known_usns = set(students["USN"])
    orphan_usns = sorted(set(assessments["USN"]) - known_usns)
    if orphan_usns:
        f.issue(f"{len(orphan_usns)} USN(s) appear in Assessments.xlsx but not in "
                 f"Students.xlsx: {orphan_usns}")
    else:
        f.stat("Orphan USNs in Assessments.xlsx", 0)

    assessed_usns = set(assessments["USN"])
    missing_from_assessments = sorted(known_usns - assessed_usns)
    if missing_from_assessments:
        f.issue(f"{len(missing_from_assessments)} student(s) in Students.xlsx have NO "
                 f"assessment records at all: {missing_from_assessments}")
    else:
        f.stat("Students with zero assessment records", 0)

    numeric_marks = pd.to_numeric(assessments["Marks"], errors="coerce")
    non_numeric_mask = assessments["Marks"].notna() & numeric_marks.isna()
    if non_numeric_mask.any():
        f.issue(f"{non_numeric_mask.sum()} mark value(s) are non-numeric and could not be parsed.")
    else:
        f.stat("Non-numeric marks", 0)

    negative_mask = numeric_marks < config.MIN_VALID_MARK
    if negative_mask.any():
        f.issue(f"{negative_mask.sum()} mark value(s) are negative (below {config.MIN_VALID_MARK}).")
    else:
        f.stat("Negative marks", 0)

    meta_lookup = question_metadata.set_index(["Assessment", "Question"])["Max_Marks"].to_dict()
    over_max_rows = []
    for _, row in assessments.iterrows():
        if pd.isna(row["Marks"]):
            continue
        key = (row["Assessment"], row["Question"])
        max_marks = meta_lookup.get(key)
        if max_marks is not None and pd.notna(max_marks) and row["Marks"] > max_marks:
            over_max_rows.append((row["USN"], row["Assessment"], row["Question"], row["Marks"], max_marks))
    if over_max_rows:
        preview = over_max_rows[:10]
        f.issue(f"{len(over_max_rows)} mark value(s) EXCEED the question's Max_Marks: "
                 f"{preview}{' ...' if len(over_max_rows) > 10 else ''}")
    else:
        f.stat("Marks exceeding Max_Marks", 0)

    meta_keys = set(zip(question_metadata["Assessment"], question_metadata["Question"]))
    assess_keys = set(zip(assessments["Assessment"], assessments["Question"]))
    unmapped = sorted(assess_keys - meta_keys)
    if unmapped:
        f.issue(f"{len(unmapped)} (Assessment, Question) pair(s) used in Assessments.xlsx have "
                 f"NO entry in Question_Metadata.xlsx: {unmapped}")
    else:
        f.stat("Unmapped question IDs (Assessments -> Metadata)", 0)

    unused = sorted(meta_keys - assess_keys)
    if unused:
        f.stat("Question_Metadata entries never used in Assessments.xlsx (informational)", str(unused))


def check_question_metadata(question_metadata: pd.DataFrame, f: ValidationFindings) -> None:
    f.stat("Total question metadata rows", len(question_metadata))

    dup = question_metadata[question_metadata.duplicated(subset=["Assessment", "Question"], keep=False)]
    if not dup.empty:
        f.issue(f"{dup.shape[0]} duplicate (Assessment, Question) row(s) in Question_Metadata.xlsx.")
    else:
        f.stat("Duplicate rows in Question_Metadata.xlsx", 0)

    missing_max = question_metadata[question_metadata["Max_Marks"].isna()]
    if not missing_max.empty:
        f.issue(f"{len(missing_max)} question(s) in Question_Metadata.xlsx have no Max_Marks.")
    else:
        f.stat("Questions missing Max_Marks", 0)

    def _valid_co(value) -> bool:
        if pd.isna(value):
            return False
        text = str(value).strip()
        return text == "Not Specified" or bool(pd.Series([text]).str.match(r"^CO\d+$").iloc[0])

    bad_co = question_metadata[~question_metadata["CO"].map(_valid_co)]
    if not bad_co.empty:
        f.issue(f"{len(bad_co)} row(s) in Question_Metadata.xlsx have an invalid/missing CO mapping.")
    else:
        f.stat("Invalid/missing CO mappings", 0)

    not_specified = int((question_metadata["CO"] == "Not Specified").sum())
    if not_specified:
        f.stat("Questions with CO deliberately 'Not Specified' (no source question paper available)",
               not_specified)


def check_master_dataset(master: pd.DataFrame, students: pd.DataFrame, f: ValidationFindings) -> None:
    f.stat("Total Master_Dataset rows", len(master))

    if len(master) != len(students):
        f.issue(f"Master_Dataset.xlsx has {len(master)} rows but Students.xlsx has "
                 f"{len(students)} -- row counts should match (one row per student).")

    dup_usn = master[master["USN"].duplicated(keep=False)]
    if not dup_usn.empty:
        f.issue(f"Master_Dataset.xlsx has {dup_usn['USN'].nunique()} duplicate USN(s).")
    else:
        f.stat("Duplicate USNs in Master_Dataset.xlsx", 0)

    q_cols = get_question_columns(master)
    total_cells = len(master) * len(q_cols)
    missing_cells = int(master[q_cols].isna().sum().sum()) if q_cols else 0
    pct_missing = round(100 * missing_cells / total_cells, 2) if total_cells else 0.0
    f.stat("Question-level cells in Master_Dataset", total_cells)
    f.stat("Missing question-level cells (blank / not attempted)", missing_cells)
    f.stat("Percent missing (question-level)", f"{pct_missing}%")

    for placeholder in config.PLACEHOLDER_COLUMNS:
        if placeholder in master.columns:
            n_filled = int(master[placeholder].notna().sum())
            f.stat(f"'{placeholder}' populated rows (expected 0 until source file supplied)", n_filled)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(f: ValidationFindings, out_path: Path) -> None:
    lines = ["# Validation Report", ""]
    lines.append(f"Generated by `validate_data.py`. {len(f.issues)} issue(s) found, "
                 f"{len(f.corrections)} automatic correction(s) applied.")
    lines.append("")

    lines.append("## Problems Found")
    if f.issues:
        for i, issue in enumerate(f.issues, start=1):
            lines.append(f"{i}. {issue}")
    else:
        lines.append("No problems were found. All datasets passed every validation check.")
    lines.append("")

    lines.append("## Corrections Performed")
    lines.append(
        "This validator is **read-only by design**: it reports every issue it finds but does "
        "not silently rewrite the source datasets, since automatic correction of marks or "
        "identity data could introduce errors that are hard to detect later. Any correction "
        "listed below was a non-destructive, deterministic normalization applied only in "
        "downstream, derived outputs (never in the original input files)."
    )
    if f.corrections:
        for i, corr in enumerate(f.corrections, start=1):
            lines.append(f"{i}. {corr}")
    else:
        lines.append("No corrections were necessary.")
    lines.append("")

    lines.append("## Data Quality Statistics")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for key, value in f.stats.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Validation report written -> %s", out_path)


def main() -> ValidationFindings:
    logger.info("PHASE 1: Data validation starting.")
    students = read_excel_safe(config.STUDENTS_FILE, logger)
    assessments = read_excel_safe(config.ASSESSMENTS_FILE, logger)
    question_metadata = read_excel_safe(config.QUESTION_METADATA_FILE, logger)
    master = read_excel_safe(config.MASTER_DATASET_FILE, logger)

    f = ValidationFindings()
    check_students(students, f)
    check_assessments(assessments, students, question_metadata, f)
    check_question_metadata(question_metadata, f)
    check_master_dataset(master, students, f)

    write_report(f, config.VALIDATION_REPORT)
    logger.info("PHASE 1 complete: %d issue(s), %d correction(s).", len(f.issues), len(f.corrections))
    return f


if __name__ == "__main__":
    main()
