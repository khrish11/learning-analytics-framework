#!/usr/bin/env python3
"""
preprocessing.py
-----------------
PHASE 5 - Data Preprocessing.

Prepares Feature_Dataset.xlsx for machine learning:

  - Handles missing values (numeric -> median imputation; the two
    placeholder columns Attendance / CO_Attainment are left untouched
    since they are not features, they are potential future targets)
  - Encodes categorical variables (Branch, Section -> one-hot; Semester
    is already numeric)
  - Scales numerical features (StandardScaler)
  - Splits students into Training / Validation / Testing sets using the
    ratios in config.py (default 70/15/15), with a fixed random seed for
    reproducibility

Writes output/train_set.xlsx, output/validation_set.xlsx,
output/test_set.xlsx (each with both the raw and scaled feature columns
retained, since a paper's descriptive tables usually want the raw
values while the models want the scaled ones).

Runs independently: `python scripts/preprocessing.py`. If Feature_Dataset.xlsx
does not exist yet, run feature_engineering.py first (or use run.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from utils import get_logger, read_excel_safe, write_excel_safe

logger = get_logger("preprocessing")


def handle_missing_values(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            n_missing = int(df[col].isna().sum())
            df[col] = df[col].fillna(median_val)
            logger.info("Imputed %d missing value(s) in '%s' with median=%.3f", n_missing, col, median_val)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    categorical_cols = [c for c in ("Branch", "Section") if c in df.columns]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
        logger.info("One-hot encoded categorical columns: %s", categorical_cols)
    return df


def scale_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[numeric_cols])
    scaled_df = pd.DataFrame(scaled_values, columns=[f"{c}_scaled" for c in numeric_cols], index=df.index)
    return pd.concat([df, scaled_df], axis=1), scaler


def split_dataset(df: pd.DataFrame):
    train_frac = config.TRAIN_FRACTION
    val_frac = config.VALIDATION_FRACTION
    test_frac = config.TEST_FRACTION
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Split fractions must sum to 1.0"

    train_df, temp_df = train_test_split(df, train_size=train_frac, random_state=config.RANDOM_SEED)
    relative_val_frac = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(temp_df, train_size=relative_val_frac, random_state=config.RANDOM_SEED)
    return train_df, val_df, test_df


def main():
    logger.info("PHASE 5: Preprocessing starting.")
    feat = read_excel_safe(config.FEATURE_DATASET_FILE, logger)

    identity_cols = [c for c in config.IDENTITY_COLUMNS if c in feat.columns]
    placeholder_cols = [c for c in config.PLACEHOLDER_COLUMNS if c in feat.columns]
    numeric_feature_cols = [
        c for c in feat.select_dtypes(include=[np.number]).columns
        if c not in placeholder_cols and c != "Semester"
    ]

    df = handle_missing_values(feat, numeric_feature_cols)
    df = encode_categoricals(df)

    # Re-resolve numeric_feature_cols after encoding (Semester + engineered features).
    numeric_feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in placeholder_cols
    ]
    df, scaler = scale_numeric(df, numeric_feature_cols)

    train_df, val_df, test_df = split_dataset(df)
    logger.info("Split sizes -> train: %d, validation: %d, test: %d", len(train_df), len(val_df), len(test_df))

    write_excel_safe(train_df, config.PREPROCESSED_TRAIN_FILE, logger)
    write_excel_safe(val_df, config.PREPROCESSED_VAL_FILE, logger)
    write_excel_safe(test_df, config.PREPROCESSED_TEST_FILE, logger)

    logger.info("PHASE 5 complete.")
    return train_df, val_df, test_df


if __name__ == "__main__":
    main()
