"""Model building and hyperparameter optimization for fraud detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - handled in runtime environments.
    raise ImportError(
        "xgboost is required for this project. Install dependencies from requirements.txt"
    ) from exc


SAMPLING_STRATEGIES = [
    "baseline",
    "random_undersampling",
    "random_oversampling",
    "smote",
]

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]


@dataclass(frozen=True)
class ExperimentSpec:
    """Experiment configuration for one model + imbalance strategy run."""

    model_name: str
    sampling_strategy: str
    use_class_weight: bool = False


@dataclass
class TrainedExperiment:
    """Container storing trained model and tuning metadata."""

    spec: ExperimentSpec
    best_estimator: Any
    best_params: dict[str, Any]
    best_cv_pr_auc: float
    cv_roc_auc: float
    cv_f2: float


def _validate_spec(spec: ExperimentSpec) -> None:
    if spec.model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model_name: {spec.model_name}")
    if spec.sampling_strategy not in SAMPLING_STRATEGIES:
        raise ValueError(f"Unsupported sampling_strategy: {spec.sampling_strategy}")


def get_sampler(strategy_name: str, random_state: int) -> Any:
    """Return sampler object for imbalance treatment inside CV-safe pipelines."""

    if strategy_name == "baseline":
        return None
    if strategy_name == "random_undersampling":
        return RandomUnderSampler(random_state=random_state)
    if strategy_name == "random_oversampling":
        return RandomOverSampler(random_state=random_state)
    if strategy_name == "smote":
        return SMOTE(random_state=random_state, k_neighbors=5)
    if strategy_name == "smote_tomek":
        return SMOTETomek(random_state=random_state)

    raise ValueError(f"Unknown strategy: {strategy_name}")


def _build_classifier(
    model_name: str,
    random_state: int,
    use_class_weight: bool,
    scale_pos_weight: float,
) -> Any:
    """Build classifier with robust defaults for imbalanced fraud data."""

    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced" if use_class_weight else None,
            random_state=random_state,
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=50,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight="balanced_subsample" if use_class_weight else None,
            random_state=random_state,
            n_jobs=2,
        )

    if model_name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            reg_lambda=5.0,
            gamma=0.0,
            n_jobs=2,
            tree_method="hist",
            scale_pos_weight=scale_pos_weight if use_class_weight else 1.0,
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


def _get_param_space(model_name: str) -> dict[str, list[Any]]:
    """Hyperparameter search space per model."""

    if model_name == "logistic_regression":
        return {
            "model__C": [0.5, 1.0, 2.0], # Tiny search space to respect n_iter=3
            "model__penalty": ["l2"],
        }

    # Disable exhaustive search for tree models by providing a single fixed combination
    if model_name == "random_forest":
        return {
            "model__n_estimators": [50],
        }

    if model_name == "xgboost":
        return {
            "model__n_estimators": [50],
        }

    raise ValueError(f"Unsupported model_name: {model_name}")


def build_pipeline(spec: ExperimentSpec, y_train: pd.Series, random_state: int) -> ImbPipeline:
    """Build leakage-safe pipeline with optional resampling and scaling."""

    _validate_spec(spec)

    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = (negative / max(positive, 1))

    sampler = get_sampler(spec.sampling_strategy, random_state=random_state)
    classifier = _build_classifier(
        spec.model_name,
        random_state=random_state,
        use_class_weight=spec.use_class_weight,
        scale_pos_weight=scale_pos_weight,
    )

    steps: list[tuple[str, Any]] = []

    # For probabilistic linear models, scaling helps stable optimization.
    if spec.model_name == "logistic_regression":
        steps.append(("scaler", StandardScaler()))

    # Sampling is intentionally placed after scaling for LR so distances
    # used by SMOTE are computed in normalized feature space.
    if sampler is not None:
        if spec.model_name == "logistic_regression":
            steps = [("scaler", StandardScaler()), ("sampler", sampler)]
        else:
            steps.append(("sampler", sampler))

    steps.append(("model", classifier))

    return ImbPipeline(steps=steps)


def train_with_random_search(
    spec: ExperimentSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    n_iter: int = 20,
    cv_folds: int = 4,
) -> TrainedExperiment:
    """Train one experiment with PR-AUC-driven hyperparameter tuning."""

    pipeline = build_pipeline(spec=spec, y_train=y_train, random_state=random_state)
    param_space = _get_param_space(spec.model_name)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scorers = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "f2": make_scorer(fbeta_score, beta=2),
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring=scorers,
        refit="pr_auc",
        cv=cv,
        n_jobs=2,
        verbose=0,
        random_state=random_state,
        return_train_score=True,
    )

    search.fit(X_train, y_train)

    best_idx = search.best_index_
    cv_roc_auc = float(search.cv_results_["mean_test_roc_auc"][best_idx])
    cv_f2 = float(search.cv_results_["mean_test_f2"][best_idx])

    return TrainedExperiment(
        spec=spec,
        best_estimator=search.best_estimator_,
        best_params=search.best_params_,
        best_cv_pr_auc=float(search.best_score_),
        cv_roc_auc=cv_roc_auc,
        cv_f2=cv_f2,
    )


def generate_experiment_specs() -> list[ExperimentSpec]:
    """Create exactly 6 high-value, strategically meaningful experiment specs."""
    return [
        ExperimentSpec("logistic_regression", "baseline", False),
        ExperimentSpec("logistic_regression", "baseline", True),
        ExperimentSpec("logistic_regression", "smote", False),
        ExperimentSpec("random_forest", "baseline", False),
        ExperimentSpec("random_forest", "baseline", True),
        ExperimentSpec("xgboost", "smote", False),
    ]


def predict_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities for threshold-based evaluation."""

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    # Fallback for margin-only models.
    decision_scores = model.decision_function(X)
    return 1 / (1 + np.exp(-decision_scores))
