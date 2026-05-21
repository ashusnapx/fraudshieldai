"""Exploratory data analysis for fraud detection datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def configure_plot_style() -> None:
    """Apply a presentation-quality plotting style."""

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk", font_scale=0.9)
    sns.set_palette("viridis")


def _save_figure(output_path: Path, dpi: int = 300) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def compute_dataset_statistics(df: pd.DataFrame, target_col: str = "Class") -> dict[str, Any]:
    """Compute dataset-level stats used in reports and executive summary."""

    class_counts = df[target_col].value_counts().sort_index()
    fraud_count = int(class_counts.get(1, 0))
    non_fraud_count = int(class_counts.get(0, 0))
    total = int(len(df))

    return {
        "rows": total,
        "columns": int(df.shape[1]),
        "fraud_count": fraud_count,
        "non_fraud_count": non_fraud_count,
        "fraud_rate": fraud_count / total if total else 0.0,
        "imbalance_ratio_nonfraud_to_fraud": (non_fraud_count / fraud_count)
        if fraud_count
        else float("inf"),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values_total": int(df.isnull().sum().sum()),
    }


def plot_class_distribution(
    df: pd.DataFrame, output_path: Path, target_col: str = "Class"
) -> None:
    """Visualize class imbalance with both count and proportion annotation."""

    configure_plot_style()
    counts = df[target_col].value_counts().sort_index()
    total = len(df)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(["Non-Fraud (0)", "Fraud (1)"], counts.values, color=["#1f77b4", "#d62728"])
    ax.set_title("Transaction Class Distribution (Extreme Imbalance)", fontsize=16, pad=14)
    ax.set_ylabel("Number of Transactions")

    for bar, count in zip(bars, counts.values):
        pct = 100 * count / total
        ax.annotate(
            f"{count:,}\n({pct:.3f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    _save_figure(output_path)


def plot_amount_distribution(
    df: pd.DataFrame, output_path: Path, target_col: str = "Class", amount_col: str = "Amount"
) -> None:
    """Plot transaction amount distributions for fraud vs non-fraud."""

    if amount_col not in df.columns:
        return

    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(data=df, x=amount_col, bins=100, kde=True, ax=axes[0], color="#2ca02c")
    axes[0].set_title("Overall Transaction Amount Distribution")
    axes[0].set_xlim(0, df[amount_col].quantile(0.99))

    sns.boxplot(data=df, x=target_col, y=amount_col, ax=axes[1], palette=["#1f77b4", "#d62728"])
    axes[1].set_title("Amount by Class (Boxplot)")
    axes[1].set_xticklabels(["Non-Fraud", "Fraud"])
    axes[1].set_yscale("log")

    _save_figure(output_path)


def plot_time_based_patterns(
    df: pd.DataFrame, output_path: Path, target_col: str = "Class", time_col: str = "Time"
) -> None:
    """Analyze fraud dynamics over time if the dataset includes elapsed seconds."""

    if time_col not in df.columns:
        return

    configure_plot_style()
    df_tmp = df.copy()
    df_tmp["Hour"] = (df_tmp[time_col] / 3600.0) % 24

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(
        data=df_tmp,
        x="Hour",
        hue=target_col,
        bins=24,
        multiple="stack",
        palette={0: "#1f77b4", 1: "#d62728"},
        ax=axes[0],
    )
    axes[0].set_title("Transactions by Hour (Stacked by Class)")
    axes[0].set_xlabel("Hour of Day")

    fraud_rate_by_hour = df_tmp.groupby(pd.cut(df_tmp["Hour"], bins=np.arange(0, 25, 1), right=False))[target_col].mean()
    fraud_rate_by_hour.index = [interval.left for interval in fraud_rate_by_hour.index]

    axes[1].plot(fraud_rate_by_hour.index, fraud_rate_by_hour.values, marker="o", color="#d62728")
    axes[1].set_title("Fraud Rate by Hour")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Fraud Rate")

    _save_figure(output_path)


def plot_correlation_heatmap(
    df: pd.DataFrame, output_path: Path, target_col: str = "Class", top_k: int = 20
) -> None:
    """Create a target-focused correlation heatmap for interpretability."""

    configure_plot_style()
    corr = df.corr(numeric_only=True)

    if target_col in corr.columns:
        top_features = (
            corr[target_col]
            .abs()
            .sort_values(ascending=False)
            .head(top_k)
            .index
            .tolist()
        )
        corr_subset = corr.loc[top_features, top_features]
    else:
        corr_subset = corr

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr_subset,
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
    )
    plt.title("Correlation Heatmap (Top Features by |corr with Class|)", fontsize=16, pad=12)
    _save_figure(output_path)


def plot_feature_distributions(
    df: pd.DataFrame,
    output_path: Path,
    target_col: str = "Class",
    features: list[str] | None = None,
) -> None:
    """Plot KDE distributions for selected features by class label."""

    if features is None:
        candidate_features = [col for col in df.columns if col not in {target_col, "Time", "Amount"}]
        features = candidate_features[:8]

    configure_plot_style()
    n_features = len(features)
    n_cols = 2
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, feature in enumerate(features):
        sns.kdeplot(
            data=df,
            x=feature,
            hue=target_col,
            fill=True,
            common_norm=False,
            palette={0: "#1f77b4", 1: "#d62728"},
            alpha=0.35,
            ax=axes[idx],
        )
        axes[idx].set_title(f"Distribution of {feature} by Class")

    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    _save_figure(output_path)


def plot_skewness_analysis(df: pd.DataFrame, output_path: Path, top_k: int = 15) -> None:
    """Identify highly skewed features to motivate robust scaling choices."""

    configure_plot_style()
    skewness = df.select_dtypes(include=[np.number]).skew().sort_values(key=np.abs, ascending=False)
    skewness = skewness.head(top_k)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#d62728" if abs(v) > 1 else "#1f77b4" for v in skewness.values]
    ax.barh(skewness.index, skewness.values, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Top Skewed Features (Absolute Skewness)")
    ax.set_xlabel("Skewness")

    _save_figure(output_path)


def build_eda_report(df: pd.DataFrame, target_col: str = "Class") -> pd.DataFrame:
    """Generate a tabular EDA report with descriptive and imbalance insights."""

    numeric_summary = df.describe(include="all").transpose()
    numeric_summary["missing_count"] = df.isnull().sum()
    numeric_summary["missing_pct"] = numeric_summary["missing_count"] / len(df)

    if target_col in df.columns:
        class_stats = df[target_col].value_counts(normalize=True).rename("class_rate")
        for class_label, value in class_stats.items():
            numeric_summary.loc[f"target_rate_{class_label}", "mean"] = value

    return numeric_summary.reset_index().rename(columns={"index": "feature"})


def run_full_eda(
    df: pd.DataFrame,
    plots_dir: Path,
    reports_dir: Path,
    target_col: str = "Class",
) -> dict[str, Any]:
    """Execute EDA workflow and write plot/report artifacts to disk."""

    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    stats = compute_dataset_statistics(df, target_col=target_col)

    plot_class_distribution(df, plots_dir / "eda_class_distribution.png", target_col=target_col)
    plot_amount_distribution(df, plots_dir / "eda_amount_distribution.png", target_col=target_col)
    plot_time_based_patterns(df, plots_dir / "eda_time_analysis.png", target_col=target_col)
    plot_correlation_heatmap(df, plots_dir / "eda_correlation_heatmap.png", target_col=target_col)
    plot_feature_distributions(df, plots_dir / "eda_feature_distributions.png", target_col=target_col)
    plot_skewness_analysis(df, plots_dir / "eda_skewness_analysis.png")

    eda_table = build_eda_report(df, target_col=target_col)
    eda_table.to_csv(reports_dir / "eda_summary_table.csv", index=False)

    return stats
