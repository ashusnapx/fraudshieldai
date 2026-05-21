"""Visualization utilities for model diagnostics and executive reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def configure_plot_style() -> None:
    """Configure a modern visual style for publication-quality charts."""

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk", font_scale=0.9)


def _save_plot(path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_roc_curves(
    prediction_payloads: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot ROC curves for multiple experiments in one comparison chart."""

    configure_plot_style()
    plt.figure(figsize=(11, 8))

    for payload in prediction_payloads:
        fpr, tpr, _ = roc_curve(payload["y_true"], payload["y_prob"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{payload['name']} (AUC={roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.title("ROC Curve Comparison", fontsize=16)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)

    _save_plot(output_path)


def plot_precision_recall_curves(
    prediction_payloads: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot Precision-Recall curves for multiple experiments."""

    configure_plot_style()
    plt.figure(figsize=(11, 8))

    for payload in prediction_payloads:
        precision, recall, _ = precision_recall_curve(payload["y_true"], payload["y_prob"])
        pr_auc = average_precision_score(payload["y_true"], payload["y_prob"])
        plt.plot(recall, precision, linewidth=2, label=f"{payload['name']} (PR-AUC={pr_auc:.4f})")

    plt.title("Precision-Recall Curve Comparison", fontsize=16)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(alpha=0.3)

    _save_plot(output_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Create annotated confusion matrix heatmap."""

    configure_plot_style()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred Non-Fraud", "Pred Fraud"],
        yticklabels=["True Non-Fraud", "True Fraud"],
    )
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    _save_plot(output_path)


def plot_threshold_tradeoff(threshold_table: pd.DataFrame, output_path: Path, title: str) -> None:
    """Visualize precision/recall/F2 dynamics across decision thresholds."""

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(threshold_table["threshold"], threshold_table["precision"], label="Precision", linewidth=2)
    ax.plot(threshold_table["threshold"], threshold_table["recall"], label="Recall", linewidth=2)
    ax.plot(threshold_table["threshold"], threshold_table["f2"], label="F2 Score", linewidth=2)

    best_row = threshold_table.loc[threshold_table["f2"].idxmax()]
    ax.scatter(best_row["threshold"], best_row["f2"], color="red", s=90, zorder=5)
    ax.annotate(
        f"Best F2 @ {best_row['threshold']:.2f}",
        xy=(best_row["threshold"], best_row["f2"]),
        xytext=(10, 10),
        textcoords="offset points",
        color="red",
        fontsize=11,
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Metric Value")
    ax.legend()
    ax.grid(alpha=0.3)

    _save_plot(output_path)


def plot_model_comparison_dashboard(
    comparison_df: pd.DataFrame,
    output_path: Path,
    metric: str = "test_pr_auc",
) -> None:
    """Bar chart dashboard comparing experiments on a selected metric."""

    configure_plot_style()
    chart_df = comparison_df.sort_values(metric, ascending=False).head(15)

    plt.figure(figsize=(16, 8))
    bars = sns.barplot(data=chart_df, x=metric, y="experiment", palette="crest")

    for idx, value in enumerate(chart_df[metric]):
        bars.text(value + 0.001, idx, f"{value:.4f}", va="center", fontsize=10)

    plt.title(f"Top Experiments by {metric}", fontsize=16)
    plt.xlabel(metric)
    plt.ylabel("Experiment")

    _save_plot(output_path)


def _extract_model_from_pipeline(model_pipeline: Any) -> Any:
    if hasattr(model_pipeline, "named_steps") and "model" in model_pipeline.named_steps:
        return model_pipeline.named_steps["model"]
    return model_pipeline


def plot_feature_importance(
    model_pipeline: Any,
    feature_names: list[str],
    output_path: Path,
    title: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """Plot model feature importance for tree or linear models."""

    model = _extract_model_from_pipeline(model_pipeline)

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance_values = np.abs(model.coef_).ravel()
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance_values}
    ).sort_values("importance", ascending=False)

    plot_df = importance_df.head(top_n)

    configure_plot_style()
    plt.figure(figsize=(12, 8))
    sns.barplot(data=plot_df, x="importance", y="feature", palette="mako")
    plt.title(title, fontsize=16)
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    _save_plot(output_path)
    return importance_df


def try_plot_shap_summary(
    model_pipeline: Any,
    X_sample: pd.DataFrame,
    output_path: Path,
    max_display: int = 20,
) -> bool:
    """Try SHAP summary plot for tree models. Returns True if successful."""

    try:
        import shap
    except ImportError:
        return False

    model = _extract_model_from_pipeline(model_pipeline)

    # SHAP tree explainer is robust for RF/XGBoost and gives practical interpretability.
    if not hasattr(model, "feature_importances_"):
        return False

    sample = X_sample.sample(min(len(X_sample), 2000), random_state=42)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

        configure_plot_style()
        shap.summary_plot(shap_values, sample, show=False, max_display=max_display)
        plt.title("SHAP Feature Impact Summary")
        _save_plot(output_path)
        return True
    except Exception:
        return False
