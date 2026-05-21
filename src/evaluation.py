"""Evaluation and threshold optimization utilities for fraud detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ThresholdSelection:
    """Metadata describing tuned decision threshold."""

    threshold: float
    objective_value: float
    objective_metric: str


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else 0.0


def compute_business_cost(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    cost_false_positive: float = 1.0,
    cost_false_negative: float = 25.0,
) -> dict[str, float]:
    """Compute simple expected cost for fraud operations trade-off analysis."""

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_cost = (fp * cost_false_positive) + (fn * cost_false_negative)

    return {
        "false_positive_count": float(fp),
        "false_negative_count": float(fn),
        "true_positive_count": float(tp),
        "true_negative_count": float(tn),
        "total_cost": float(total_cost),
    }


def evaluate_at_threshold(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    beta: float = 2.0,
    cost_false_positive: float = 1.0,
    cost_false_negative: float = 25.0,
    include_classification_report: bool = True,
) -> dict[str, Any]:
    """Evaluate one model at one explicit threshold."""

    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = _safe_divide(tn, (tn + fp))

    metrics = {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "specificity": specificity,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    metrics.update(
        compute_business_cost(
            y_true=y_true,
            y_pred=y_pred,
            cost_false_positive=cost_false_positive,
            cost_false_negative=cost_false_negative,
        )
    )
    if include_classification_report:
        metrics["classification_report"] = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

    return metrics


def evaluate_threshold_grid(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    beta: float = 2.0,
    cost_false_positive: float = 1.0,
    cost_false_negative: float = 25.0,
) -> pd.DataFrame:
    """Evaluate a dense threshold grid for robust threshold optimization."""

    thresholds = np.round(np.linspace(0.01, 0.99, 99), 2)
    rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        rows.append(
            evaluate_at_threshold(
                y_true=y_true,
                y_prob=y_prob,
                threshold=float(threshold),
                beta=beta,
                cost_false_positive=cost_false_positive,
                cost_false_negative=cost_false_negative,
                include_classification_report=False,
            )
        )

    return pd.DataFrame(rows)


def select_best_threshold(
    threshold_table: pd.DataFrame,
    objective: str = "f2",
    min_precision: float | None = None,
    min_recall: float | None = None,
) -> ThresholdSelection:
    """Select optimal threshold under objective and optional constraints."""

    candidates = threshold_table.copy()

    if min_precision is not None:
        candidates = candidates[candidates["precision"] >= min_precision]

    if min_recall is not None:
        candidates = candidates[candidates["recall"] >= min_recall]

    if candidates.empty:
        candidates = threshold_table.copy()

    max_val = candidates[objective].max()
    winners = candidates[candidates[objective] == max_val]

    # Tie-breaker: choose higher recall, then lower threshold for better catch rate.
    winner = winners.sort_values(by=["recall", "threshold"], ascending=[False, True]).iloc[0]

    return ThresholdSelection(
        threshold=float(winner["threshold"]),
        objective_value=float(winner[objective]),
        objective_metric=objective,
    )


def evaluate_train_valid_test(
    y_train: pd.Series,
    prob_train: np.ndarray,
    y_valid: pd.Series,
    prob_valid: np.ndarray,
    y_test: pd.Series,
    prob_test: np.ndarray,
    tuned_threshold: float,
    beta: float = 2.0,
    cost_false_positive: float = 1.0,
    cost_false_negative: float = 25.0,
) -> dict[str, dict[str, Any]]:
    """Compare train/validation/test metrics at a common tuned threshold."""

    return {
        "train": evaluate_at_threshold(
            y_train,
            prob_train,
            threshold=tuned_threshold,
            beta=beta,
            cost_false_positive=cost_false_positive,
            cost_false_negative=cost_false_negative,
        ),
        "validation": evaluate_at_threshold(
            y_valid,
            prob_valid,
            threshold=tuned_threshold,
            beta=beta,
            cost_false_positive=cost_false_positive,
            cost_false_negative=cost_false_negative,
        ),
        "test": evaluate_at_threshold(
            y_test,
            prob_test,
            threshold=tuned_threshold,
            beta=beta,
            cost_false_positive=cost_false_positive,
            cost_false_negative=cost_false_negative,
        ),
    }


def flatten_metrics_for_table(
    experiment_name: str,
    split_metrics: dict[str, dict[str, Any]],
    best_cv_pr_auc: float,
    best_cv_roc_auc: float,
    best_cv_f2: float,
) -> dict[str, Any]:
    """Flatten nested metric dictionaries for comparison dashboards."""

    row: dict[str, Any] = {
        "experiment": experiment_name,
        "cv_pr_auc": best_cv_pr_auc,
        "cv_roc_auc": best_cv_roc_auc,
        "cv_f2": best_cv_f2,
    }

    for split_name, metrics in split_metrics.items():
        row[f"{split_name}_precision"] = metrics["precision"]
        row[f"{split_name}_recall"] = metrics["recall"]
        row[f"{split_name}_f1"] = metrics["f1"]
        row[f"{split_name}_f2"] = metrics["f2"]
        row[f"{split_name}_roc_auc"] = metrics["roc_auc"]
        row[f"{split_name}_pr_auc"] = metrics["pr_auc"]
        row[f"{split_name}_fp"] = metrics["fp"]
        row[f"{split_name}_fn"] = metrics["fn"]
        row[f"{split_name}_cost"] = metrics["total_cost"]
        row[f"{split_name}_threshold"] = metrics["threshold"]

    return row
