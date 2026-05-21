"""
Enterprise Fraud Metrics Module.
Provides business-aligned metrics beyond standard ML evaluation.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Dict, Any

@dataclass
class EnterpriseMetrics:
    fraud_capture_rate: float
    alert_precision: float
    false_positive_burden: float
    investigation_efficiency: float
    revenue_protected_estimate: float
    cost_per_investigation: float
    analyst_review_load: int
    precision_at_top_k: float

def compute_enterprise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    avg_transaction_amount: float = 150.0,
    investigation_cost: float = 15.0,
    top_k: int = 100
) -> EnterpriseMetrics:
    """Compute enterprise business metrics."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    fraud_capture_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    alert_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    false_positive_burden = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    investigation_efficiency = alert_precision
    
    revenue_protected_estimate = tp * avg_transaction_amount
    cost_per_investigation = investigation_cost
    analyst_review_load = tp + fp
    
    # Precision at top K
    sorted_indices = np.argsort(y_prob)[::-1]
    top_k_indices = sorted_indices[:top_k]
    top_k_true = y_true[top_k_indices]
    precision_at_top_k = np.mean(top_k_true) if len(top_k_true) > 0 else 0.0
    
    return EnterpriseMetrics(
        fraud_capture_rate=fraud_capture_rate,
        alert_precision=alert_precision,
        false_positive_burden=false_positive_burden,
        investigation_efficiency=investigation_efficiency,
        revenue_protected_estimate=revenue_protected_estimate,
        cost_per_investigation=cost_per_investigation,
        analyst_review_load=analyst_review_load,
        precision_at_top_k=precision_at_top_k
    )

def compute_cost_at_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 25.0,
    avg_txn_amount: float = 150.0
) -> pd.DataFrame:
    """Compute business cost and metrics across a range of thresholds."""
    results = []
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        revenue_protected = tp * avg_txn_amount
        
        fraud_capture_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        alert_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        results.append({
            'threshold': thresh,
            'total_cost': total_cost,
            'revenue_protected': revenue_protected,
            'fp_count': fp,
            'fn_count': fn,
            'tp_count': tp,
            'tn_count': tn,
            'fraud_capture_rate': fraud_capture_rate,
            'alert_precision': alert_precision
        })
        
    return pd.DataFrame(results)

def simulate_business_impact(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    avg_txn_amount: float = 150.0,
    daily_transactions: int = 100000,
    investigation_cost_per_alert: float = 15.0
) -> Dict[str, Any]:
    """Simulate business impact for a production deployment scenario."""
    y_pred = (y_prob >= threshold).astype(int)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    # Scale from test set size to daily volume
    scale_factor = daily_transactions / len(y_true)
    
    daily_tp = tp * scale_factor
    daily_fp = fp * scale_factor
    daily_fn = fn * scale_factor
    
    daily_fraud_blocked = daily_tp
    daily_fraud_missed = daily_fn
    daily_alerts_generated = daily_tp + daily_fp
    
    daily_investigation_cost = daily_alerts_generated * investigation_cost_per_alert
    daily_revenue_protected = daily_fraud_blocked * avg_txn_amount
    daily_net_savings = daily_revenue_protected - daily_investigation_cost - (daily_fraud_missed * avg_txn_amount)
    
    return {
        'daily_fraud_blocked': daily_fraud_blocked,
        'daily_fraud_missed': daily_fraud_missed,
        'daily_alerts_generated': daily_alerts_generated,
        'daily_investigation_cost': daily_investigation_cost,
        'daily_revenue_protected': daily_revenue_protected,
        'daily_net_savings': daily_net_savings,
        'monthly_projection': {
            'net_savings': daily_net_savings * 30,
            'revenue_protected': daily_revenue_protected * 30,
            'investigation_cost': daily_investigation_cost * 30
        },
        'annual_projection': {
            'net_savings': daily_net_savings * 365,
            'revenue_protected': daily_revenue_protected * 365,
            'investigation_cost': daily_investigation_cost * 365
        }
    }
