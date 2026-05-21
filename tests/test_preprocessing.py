from __future__ import annotations

import pandas as pd

from src.preprocessing import (
    TARGET_COLUMN,
    assess_data_quality,
    drop_duplicates,
    stratified_train_valid_test_split,
    summarize_class_balance,
)


def _build_df(n_rows: int = 1000, fraud_rate: float = 0.02) -> pd.DataFrame:
    fraud_count = int(n_rows * fraud_rate)
    nonfraud_count = n_rows - fraud_count

    data = {
        "Time": list(range(n_rows)),
        "Amount": [float((i % 100) + 1) for i in range(n_rows)],
        "V1": [float((i % 11) - 5) for i in range(n_rows)],
        TARGET_COLUMN: ([0] * nonfraud_count) + ([1] * fraud_count),
    }
    return pd.DataFrame(data)


def test_assess_quality_and_drop_duplicates() -> None:
    df = _build_df(100)
    # Introduce one duplicate
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    quality = assess_data_quality(df)
    assert quality.row_count == 101
    assert quality.duplicate_rows == 1

    cleaned, removed = drop_duplicates(df)
    assert removed == 1
    assert len(cleaned) == 100


def test_stratified_split_preserves_fraud_rate() -> None:
    df = _build_df(2000, fraud_rate=0.015)
    split = stratified_train_valid_test_split(
        df,
        target_col=TARGET_COLUMN,
        test_size=0.2,
        valid_size=0.2,
        random_state=42,
    )

    overall_rate = df[TARGET_COLUMN].mean()
    train_rate = split.y_train.mean()
    valid_rate = split.y_valid.mean()
    test_rate = split.y_test.mean()

    # Stratification should keep rates very close.
    assert abs(train_rate - overall_rate) < 0.005
    assert abs(valid_rate - overall_rate) < 0.005
    assert abs(test_rate - overall_rate) < 0.005


def test_class_balance_summary() -> None:
    y = pd.Series([0] * 90 + [1] * 10)
    summary = summarize_class_balance(y)

    assert summary["non_fraud_count"] == 90
    assert summary["fraud_count"] == 10
    assert summary["total_count"] == 100
    assert summary["fraud_rate"] == 0.10
    assert summary["imbalance_ratio_nonfraud_to_fraud"] == 9.0
