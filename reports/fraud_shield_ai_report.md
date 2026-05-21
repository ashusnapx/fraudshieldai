# Fraud Shield AI: Enterprise Fraud Detection Report

## Executive Summary
Fraud Shield AI delivers a production-grade fraud detection prototype for highly imbalanced financial transactions. The solution compares multiple imbalance-handling strategies (resampling, class weighting, and threshold tuning) across Logistic Regression, Random Forest, and XGBoost under leakage-safe pipelines and PR-AUC-first evaluation. The system is optimized for enterprise fraud objectives where missing fraud (false negatives) is substantially more expensive than reviewing false alerts.

## 1. Problem Statement
Credit-card fraud detection is a rare-event classification problem where legitimate transactions massively outnumber fraud. This imbalance makes naive classifiers and accuracy-centric evaluation misleading. The primary objective is to maximize fraud capture while controlling operational review cost.

## 2. Dataset Description
- Selected dataset: Kaggle `mlg-ulb/creditcardfraud`
- URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Size: 284,807 transactions
- Fraud class: 492 records
- Fraud prevalence: 0.172%
- Imbalance ratio: ~577:1 (non-fraud:fraud)

Why this dataset was chosen (as of May 2026):
- It remains the most established open benchmark on Kaggle for *extreme* card-fraud imbalance.
- Contains real transaction behavior (anonymized PCA components + Amount/Time).
- Dataset card explicitly recommends PR-AUC due imbalance severity.

## 3. Methodology
1. Data quality assessment (missing values, duplicates, schema checks)
2. Leakage-safe stratified split into train/validation/test
3. EDA and imbalance diagnostics
4. Model training with CV-based hyperparameter tuning (PR-AUC refit)
5. Imbalance strategy comparison:
   - Baseline
   - Random undersampling
   - Random oversampling
   - SMOTE
   - SMOTE + Tomek
   - Class-weighted baseline
6. Validation-driven threshold optimization using F2 objective
7. Test-set evaluation and business-cost analysis

## 4. EDA Findings
- Class imbalance is extreme and operationally critical.
- Amount distributions are heavy-tailed; robust preprocessing and threshold strategy matter.
- Fraud behavior differs from legitimate traffic in several anonymized principal components.
- Time windows may exhibit varying fraud intensity, motivating temporal monitoring in production.

## 5. Modeling Strategy
- Logistic Regression: strong interpretable baseline.
- Random Forest: robust non-linear ensemble, reduced sensitivity to scaling.
- XGBoost: high-capacity gradient boosting with strong ranking power in imbalanced settings.

All models use reproducible seeds and CV-driven hyperparameter optimization.

## 6. Evaluation Metrics
Reported for train/validation/test:
- ROC-AUC
- Precision
- Recall
- F1
- F2
- PR-AUC
- Confusion Matrix
- Classification Report

Why F2 is important:
- F2 weights recall higher than precision, aligning with fraud operations where missed fraud is more costly than extra manual reviews.

## 7. Results
Insert generated artifacts:
- `outputs/metrics/model_comparison_table.csv`
- `outputs/reports/executive_metrics_summary.json`
- `outputs/plots/roc_curve_top_experiments.png`
- `outputs/plots/pr_curve_top_experiments.png`
- `outputs/plots/dashboard_top_experiments_test_pr_auc.png`
- `outputs/plots/dashboard_top_experiments_test_f2.png`

## 8. Comparative Analysis
Expected practical trend:
- Baselines show high specificity but may under-detect fraud.
- Oversampling/SMOTE improve recall at potential precision cost.
- Class-weight and threshold tuning often provide efficient tradeoffs without synthetic-sample risks.
- PR-AUC differences better reflect production behavior than ROC-AUC alone.

## 9. Business Recommendations
1. Deploy PR-AUC and recall-focused model selection, not accuracy-led decisions.
2. Use dynamic threshold policies by risk appetite and analyst capacity.
3. Monitor FN-driven loss and FP-driven review load simultaneously.
4. Implement drift monitoring (feature drift + performance drift) with retraining triggers.
5. Maintain human-in-the-loop escalation for high-risk transactions.

## 10. Final Conclusion
Fraud Shield AI demonstrates that robust fraud detection requires integrated imbalance handling, calibrated thresholding, and business-aligned metrics. Best-performing models should be selected by PR-AUC, F2, and expected cost under defined operational constraints.

## 11. Future Improvements
- Add temporal validation and rolling-window backtesting.
- Add probability calibration (Platt/Isotonic) for risk-score reliability.
- Add anomaly signals and graph-based features.
- Introduce model governance artifacts (model cards, fairness checks, lineage).
- Extend to streaming scoring with concept-drift adaptation.
