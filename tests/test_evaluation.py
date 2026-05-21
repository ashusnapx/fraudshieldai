from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import (
    evaluate_at_threshold,
    evaluate_threshold_grid,
    select_best_threshold,
)


def test_evaluate_at_threshold_outputs_core_metrics() -> None:
    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.9])

    metrics = evaluate_at_threshold(y_true, y_prob, threshold=0.5)

    for key in ["precision", "recall", "f1", "f2", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"]:
        assert key in metrics

    assert metrics["tp"] == 3
    assert metrics["tn"] == 3


def test_threshold_grid_and_selection() -> None:
    rng = np.random.default_rng(42)
    y_true = pd.Series(([0] * 980) + ([1] * 20))
    y_prob = rng.uniform(0, 1, size=1000)
    # Slightly boost true-fraud probabilities so threshold search has signal.
    y_prob[980:] = np.clip(y_prob[980:] + 0.4, 0, 1)

    table = evaluate_threshold_grid(y_true, y_prob)
    assert len(table) == 99
    assert "f2" in table.columns

    selection = select_best_threshold(table, objective="f2", min_precision=0.01)
    assert 0.01 <= selection.threshold <= 0.99
    assert selection.objective_metric == "f2"
    assert selection.objective_value >= 0
