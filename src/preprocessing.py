"""Data ingestion, validation, splitting, and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "Class"


@dataclass
class SplitData:
    """Container for leakage-safe train/validation/test partitions."""

    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series


@dataclass
class DataQualityReport:
    """High-level summary of dataset health checks."""

    row_count: int
    column_count: int
    duplicate_rows: int
    missing_values_total: int
    missing_by_column: dict[str, int]


def load_dataset(csv_path: str | Path, target_col: str = TARGET_COLUMN) -> pd.DataFrame:
    """Load transaction data and enforce baseline schema checks.

    Parameters
    ----------
    csv_path:
        Path to a local CSV file (e.g., `data/raw/creditcard.csv`).
    target_col:
        Name of fraud label column. For Kaggle `mlg-ulb/creditcardfraud`, this is
        expected to be `Class` where fraud is 1 and non-fraud is 0.
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Download it from Kaggle and place it there."
        )

    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available columns: {df.columns.tolist()}"
        )

    # Defensive check to prevent silent metric corruption.
    if not set(df[target_col].dropna().unique()).issubset({0, 1}):
        raise ValueError(
            f"Target column '{target_col}' must be binary with labels {{0,1}}."
        )

    return df


def assess_data_quality(df: pd.DataFrame) -> DataQualityReport:
    """Produce a data quality snapshot for reporting and auditability."""

    missing_by_col = df.isnull().sum().to_dict()
    return DataQualityReport(
        row_count=df.shape[0],
        column_count=df.shape[1],
        duplicate_rows=int(df.duplicated().sum()),
        missing_values_total=int(df.isnull().sum().sum()),
        missing_by_column={k: int(v) for k, v in missing_by_col.items()},
    )


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop duplicate rows and return cleaned dataframe + removed count."""

    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(cleaned)
    return cleaned, removed


def stratified_train_valid_test_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = 0.20,
    valid_size: float = 0.20,
    random_state: int = 42,
) -> SplitData:
    """Create stratified train/validation/test splits.

    Why two-stage split?
    - Stage 1 (train+valid vs test): keeps the final test set untouched.
    - Stage 2 (train vs valid): gives a dedicated split for threshold tuning
      and model-selection diagnostics without contaminating the test set.
    """

    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")
    if not 0 < valid_size < 1:
        raise ValueError("valid_size must be in (0, 1)")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=valid_size,
        stratify=y_train_valid,
        random_state=random_state,
    )

    return SplitData(
        X_train=X_train.reset_index(drop=True),
        X_valid=X_valid.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_valid=y_valid.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


def summarize_class_balance(y: pd.Series) -> dict[str, Any]:
    """Return key imbalance indicators for any label vector."""

    class_counts = y.value_counts().sort_index()
    negative = int(class_counts.get(0, 0))
    positive = int(class_counts.get(1, 0))
    total = negative + positive

    fraud_rate = positive / total if total else 0.0
    ratio = (negative / positive) if positive else float("inf")

    return {
        "non_fraud_count": negative,
        "fraud_count": positive,
        "total_count": total,
        "fraud_rate": fraud_rate,
        "imbalance_ratio_nonfraud_to_fraud": ratio,
    }
