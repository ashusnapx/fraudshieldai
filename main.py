"""Fraud Shield AI: enterprise-grade orchestration entry point.

Run example:
    python main.py \
        --data-path data/raw/creditcard.csv \
        --outputs-dir outputs \
        --random-state 42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.eda import run_full_eda
from src.evaluation import (
    evaluate_threshold_grid,
    evaluate_train_valid_test,
    flatten_metrics_for_table,
    select_best_threshold,
)
from src.modeling import (
    ExperimentSpec,
    generate_experiment_specs,
    predict_probabilities,
    train_with_random_search,
)
from src.preprocessing import (
    TARGET_COLUMN,
    assess_data_quality,
    drop_duplicates,
    load_dataset,
    stratified_train_valid_test_split,
    summarize_class_balance,
)
from src.utils import (
    build_run_context,
    ensure_directories,
    get_logger,
    save_dataframe,
    save_json,
    set_global_seed,
    setup_logging,
)
from src.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison_dashboard,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_threshold_tradeoff,
    try_plot_shap_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fraud Shield AI training pipeline")
    parser.add_argument("--data-path", type=str, default="data/raw/creditcard.csv")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--valid-size", type=float, default=0.20)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--threshold-objective", type=str, default="f2", choices=["f2", "f1", "recall", "precision"])
    parser.add_argument("--min-precision", type=float, default=0.0)
    parser.add_argument("--cost-fp", type=float, default=1.0)
    parser.add_argument("--cost-fn", type=float, default=25.0)
    parser.add_argument("--max-experiments", type=int, default=0, help="Optional cap for quick smoke runs. 0 means all experiments.")
    return parser.parse_args()


def _build_experiment_name(spec: ExperimentSpec) -> str:
    cw = "class_weighted" if spec.use_class_weight else "no_class_weight"
    return f"{spec.model_name}__{spec.sampling_strategy}__{cw}"


def _serialize_objective_summary(
    comparison_df: pd.DataFrame,
    champion_row: pd.Series,
    run_context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_context": run_context,
        "champion_experiment": champion_row["experiment"],
        "champion_metrics": {
            "test_pr_auc": champion_row["test_pr_auc"],
            "test_recall": champion_row["test_recall"],
            "test_precision": champion_row["test_precision"],
            "test_f2": champion_row["test_f2"],
            "test_cost": champion_row["test_cost"],
            "test_threshold": champion_row["test_threshold"],
        },
        "top_5_experiments": comparison_df.sort_values("test_pr_auc", ascending=False)
        .head(5)
        .to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()

    setup_logging(logging.INFO)
    logger = get_logger("fraudshield")

    set_global_seed(args.random_state)
    run_context = build_run_context(random_state=args.random_state)

    output_root = Path(args.outputs_dir)
    plots_dir = output_root / "plots"
    reports_dir = output_root / "reports"
    metrics_dir = output_root / "metrics"
    models_dir = output_root / "models"

    ensure_directories([plots_dir, reports_dir, metrics_dir, models_dir])

    # Kaggle source reference:
    # https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    logger.info("Loading dataset from %s", args.data_path)
    df = load_dataset(args.data_path, target_col=TARGET_COLUMN)

    dq_report = assess_data_quality(df)
    logger.info(
        "Dataset loaded: rows=%d, cols=%d, duplicates=%d, missing_total=%d",
        dq_report.row_count,
        dq_report.column_count,
        dq_report.duplicate_rows,
        dq_report.missing_values_total,
    )

    df, dropped_duplicates = drop_duplicates(df)
    if dropped_duplicates > 0:
        logger.info("Removed %d duplicate rows", dropped_duplicates)

    class_summary = summarize_class_balance(df[TARGET_COLUMN])
    logger.info(
        "Class balance before sampling: fraud_rate=%.5f, imbalance_ratio=%.2f:1",
        class_summary["fraud_rate"],
        class_summary["imbalance_ratio_nonfraud_to_fraud"],
    )

    # Aggressive Runtime Optimization: Sample dataset down to 50k
    logger.info("Downsampling dataset to 50,000 rows for rapid experimentation...")
    from sklearn.model_selection import train_test_split
    _, df = train_test_split(df, test_size=50000, random_state=args.random_state, stratify=df[TARGET_COLUMN])
    logger.info("Dataset sampled to %d rows", len(df))

    logger.info("Running EDA and saving report artifacts")
    eda_stats = run_full_eda(df, plots_dir=plots_dir, reports_dir=reports_dir, target_col=TARGET_COLUMN)

    logger.info("Creating stratified train/validation/test splits")
    split_data = stratified_train_valid_test_split(
        df,
        target_col=TARGET_COLUMN,
        test_size=args.test_size,
        valid_size=args.valid_size,
        random_state=args.random_state,
    )

    experiment_specs = generate_experiment_specs()
    if args.max_experiments > 0:
        experiment_specs = experiment_specs[: args.max_experiments]

    logger.info("Total experiments to execute: %d", len(experiment_specs))

    comparison_rows: list[dict[str, Any]] = []
    roc_pr_payloads: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}

    for idx, spec in enumerate(experiment_specs, start=1):
        experiment_name = _build_experiment_name(spec)
        logger.info("[%d/%d] Training experiment: %s", idx, len(experiment_specs), experiment_name)

        trained = train_with_random_search(
            spec=spec,
            X_train=split_data.X_train,
            y_train=split_data.y_train,
            random_state=args.random_state,
            n_iter=args.n_iter,
            cv_folds=args.cv_folds,
        )

        prob_train = predict_probabilities(trained.best_estimator, split_data.X_train)
        prob_valid = predict_probabilities(trained.best_estimator, split_data.X_valid)
        prob_test = predict_probabilities(trained.best_estimator, split_data.X_test)

        threshold_table = evaluate_threshold_grid(
            y_true=split_data.y_valid,
            y_prob=prob_valid,
            beta=2.0,
            cost_false_positive=args.cost_fp,
            cost_false_negative=args.cost_fn,
        )

        threshold_choice = select_best_threshold(
            threshold_table,
            objective=args.threshold_objective,
            min_precision=args.min_precision if args.min_precision > 0 else None,
        )

        split_metrics = evaluate_train_valid_test(
            y_train=split_data.y_train,
            prob_train=prob_train,
            y_valid=split_data.y_valid,
            prob_valid=prob_valid,
            y_test=split_data.y_test,
            prob_test=prob_test,
            tuned_threshold=threshold_choice.threshold,
            beta=2.0,
            cost_false_positive=args.cost_fp,
            cost_false_negative=args.cost_fn,
        )

        row = flatten_metrics_for_table(
            experiment_name=experiment_name,
            split_metrics=split_metrics,
            best_cv_pr_auc=trained.best_cv_pr_auc,
            best_cv_roc_auc=trained.cv_roc_auc,
            best_cv_f2=trained.cv_f2,
        )
        row["model_name"] = spec.model_name
        row["sampling_strategy"] = spec.sampling_strategy
        row["use_class_weight"] = spec.use_class_weight
        row["best_threshold"] = threshold_choice.threshold
        row["best_threshold_objective"] = threshold_choice.objective_metric
        row["best_threshold_objective_value"] = threshold_choice.objective_value
        row["best_params"] = str(trained.best_params)

        comparison_rows.append(row)
        trained_models[experiment_name] = trained.best_estimator

        save_dataframe(
            threshold_table,
            metrics_dir / f"threshold_table_{experiment_name}.csv",
            index=False,
        )

        y_pred_test = (prob_test >= threshold_choice.threshold).astype(int)
        plot_confusion_matrix(
            y_true=split_data.y_test.values,
            y_pred=y_pred_test,
            output_path=plots_dir / f"confusion_matrix_{experiment_name}.png",
            title=f"Confusion Matrix | {experiment_name}",
        )

        plot_threshold_tradeoff(
            threshold_table=threshold_table,
            output_path=plots_dir / f"threshold_tradeoff_{experiment_name}.png",
            title=f"Threshold Tradeoff | {experiment_name}",
        )

        roc_pr_payloads.append(
            {
                "name": experiment_name,
                "y_true": split_data.y_test.values,
                "y_prob": prob_test,
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values("test_pr_auc", ascending=False).reset_index(drop=True)

    save_dataframe(comparison_df, metrics_dir / "model_comparison_table.csv", index=False)

    plot_model_comparison_dashboard(
        comparison_df=comparison_df,
        output_path=plots_dir / "dashboard_top_experiments_test_pr_auc.png",
        metric="test_pr_auc",
    )
    plot_model_comparison_dashboard(
        comparison_df=comparison_df,
        output_path=plots_dir / "dashboard_top_experiments_test_f2.png",
        metric="test_f2",
    )

    # Use top-K experiments for legible global ROC/PR curves.
    top_payloads = [
        payload for payload in roc_pr_payloads if payload["name"] in set(comparison_df.head(8)["experiment"])
    ]
    plot_roc_curves(top_payloads, plots_dir / "roc_curve_top_experiments.png")
    plot_precision_recall_curves(top_payloads, plots_dir / "pr_curve_top_experiments.png")

    champion = comparison_df.iloc[0]
    champion_name = champion["experiment"]
    champion_model = trained_models[champion_name]

    # Save champion model for deployment experimentation.
    joblib.dump(champion_model, models_dir / f"champion_model_{champion_name}.joblib")

    # Feature importance (where available) and optional SHAP explainability.
    feature_importance_df = plot_feature_importance(
        model_pipeline=champion_model,
        feature_names=split_data.X_train.columns.tolist(),
        output_path=plots_dir / f"feature_importance_{champion_name}.png",
        title=f"Feature Importance | {champion_name}",
        top_n=20,
    )

    if not feature_importance_df.empty:
        save_dataframe(
            feature_importance_df,
            metrics_dir / f"feature_importance_{champion_name}.csv",
            index=False,
        )

    shap_ok = try_plot_shap_summary(
        model_pipeline=champion_model,
        X_sample=split_data.X_test,
        output_path=plots_dir / f"shap_summary_{champion_name}.png",
    )

    run_metadata = {
        "run_id": run_context.run_id,
        "started_at_utc": run_context.started_at_utc,
        "random_state": run_context.random_state,
        "dataset_stats": eda_stats,
        "data_quality": dq_report.__dict__,
        "class_summary": class_summary,
        "dropped_duplicates": dropped_duplicates,
        "num_experiments": len(experiment_specs),
        "shap_generated": shap_ok,
        "champion_experiment": champion_name,
    }

    save_json(run_metadata, reports_dir / "run_metadata.json")
    save_json(
        _serialize_objective_summary(comparison_df, champion, run_metadata),
        reports_dir / "executive_metrics_summary.json",
    )

    logger.info("Champion experiment: %s", champion_name)
    logger.info("Champion Test PR-AUC: %.5f", champion["test_pr_auc"])
    logger.info("Champion Test Recall: %.5f | Precision: %.5f | F2: %.5f", champion["test_recall"], champion["test_precision"], champion["test_f2"])
    logger.info("All artifacts saved under: %s", output_root.resolve())


if __name__ == "__main__":
    main()
