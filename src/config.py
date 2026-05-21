"""Centralized configuration constants for Fraud Shield AI."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TARGET_COLUMN = "Class"
RANDOM_STATE = 42

# Kaggle benchmark dataset reference used in this project.
KAGGLE_DATASET_SLUG = "mlg-ulb/creditcardfraud"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"

# Business weighting assumption:
# Missing true fraud (FN) is far more expensive than investigating a false alert (FP).
DEFAULT_COST_FALSE_POSITIVE = 1.0
DEFAULT_COST_FALSE_NEGATIVE = 25.0
