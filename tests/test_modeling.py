from __future__ import annotations

import numpy as np
import pandas as pd

from src.modeling import (
    ExperimentSpec,
    build_pipeline,
    generate_experiment_specs,
    predict_probabilities,
    train_with_random_search,
)


def _toy_data(n_rows: int = 800) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7)
    X = pd.DataFrame(
        {
            "Time": rng.uniform(0, 100000, size=n_rows),
            "Amount": rng.gamma(shape=2, scale=30, size=n_rows),
            "V1": rng.normal(0, 1, size=n_rows),
            "V2": rng.normal(0, 1, size=n_rows),
            "V3": rng.normal(0, 1, size=n_rows),
        }
    )

    # Create non-linear-ish fraud signal with heavy imbalance.
    linear = 0.03 * X["Amount"] - 0.8 * X["V1"] + 0.6 * X["V2"]
    prob = 1 / (1 + np.exp(-(linear - 2.4)))
    y = pd.Series((rng.uniform(0, 1, size=n_rows) < prob * 0.08).astype(int))

    # Ensure both classes exist for CV.
    if y.sum() < 20:
        y.iloc[:20] = 1
    if (y == 0).sum() < 100:
        y.iloc[20:120] = 0

    return X, y


def test_generate_experiment_specs_contains_required_variants() -> None:
    specs = generate_experiment_specs()
    assert len(specs) >= 18
    assert any(s.model_name == "xgboost" for s in specs)
    assert any(s.sampling_strategy == "smote_tomek" for s in specs)
    assert any(s.use_class_weight for s in specs)


def test_build_pipeline_and_predict_probabilities() -> None:
    X, y = _toy_data(400)
    spec = ExperimentSpec(model_name="logistic_regression", sampling_strategy="smote", use_class_weight=False)
    pipe = build_pipeline(spec, y_train=y, random_state=42)

    pipe.fit(X, y)
    probs = predict_probabilities(pipe, X)

    assert probs.shape[0] == len(X)
    assert np.all((probs >= 0) & (probs <= 1))


def test_train_with_random_search_smoke() -> None:
    X, y = _toy_data(500)
    spec = ExperimentSpec(model_name="logistic_regression", sampling_strategy="baseline", use_class_weight=True)

    trained = train_with_random_search(
        spec=spec,
        X_train=X,
        y_train=y,
        random_state=42,
        n_iter=2,
        cv_folds=2,
    )

    assert trained.best_estimator is not None
    assert isinstance(trained.best_params, dict)
    assert trained.best_cv_pr_auc >= 0
