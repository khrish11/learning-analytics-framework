"""
utils.py
--------
Shared utilities used by every stage of the learning_analytics pipeline:
logging setup, safe Excel I/O, and small helpers for identifying
assessment/question columns inside Master_Dataset.xlsx by pattern rather
than by hardcoded name, so the pipeline keeps working if a future
semester adds/removes assessments or questions.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

import config


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger that writes to both stdout and a
    rotating-free log file under logs/, configured once per process."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured (e.g. re-imported)

    logger.setLevel(config.LOG_LEVEL)
    formatter = logging.Formatter(config.LOG_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_file = config.LOG_DIR / "pipeline.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def read_excel_safe(path: Path, logger: logging.Logger | None = None) -> pd.DataFrame:
    """Read an Excel file with a clear error message if it is missing,
    rather than letting pandas raise an opaque FileNotFoundError deep in
    a stack trace."""
    if not path.exists():
        msg = f"Required input file not found: {path}"
        if logger:
            logger.error(msg)
        raise FileNotFoundError(msg)
    return pd.read_excel(path)


def write_excel_safe(df: pd.DataFrame, path: Path, logger: logging.Logger | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)
    if logger:
        logger.info("Wrote %d rows x %d cols -> %s", df.shape[0], df.shape[1], path)


# ---------------------------------------------------------------------------
# Column pattern helpers for Master_Dataset.xlsx
# ---------------------------------------------------------------------------
QUESTION_COL_PATTERN = re.compile(r"^(?P<assessment>[A-Za-z0-9]+)_(?P<question>Q[0-9]+[a-zA-Z]?)$")


def get_question_columns(df: pd.DataFrame) -> List[str]:
    """Return every column in `df` that matches the '<Assessment>_Q..'
    pattern used throughout Master_Dataset.xlsx / Feature_Dataset.xlsx."""
    return [c for c in df.columns if QUESTION_COL_PATTERN.match(str(c))]


def group_question_columns_by_assessment(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Group question columns by their assessment prefix, e.g.
    {'TT1': ['TT1_Q1a', 'TT1_Q1b'], 'TT2': [...], ...}."""
    groups: Dict[str, List[str]] = {}
    for col in get_question_columns(df):
        match = QUESTION_COL_PATTERN.match(str(col))
        assessment = match.group("assessment")
        groups.setdefault(assessment, []).append(col)
    return groups


def question_col_to_assessment_question(col: str) -> tuple[str, str]:
    """'TT2_Q3b' -> ('TT2', 'Q3b')."""
    match = QUESTION_COL_PATTERN.match(str(col))
    if not match:
        raise ValueError(f"Column '{col}' does not match the Assessment_Question pattern.")
    return match.group("assessment"), match.group("question")
