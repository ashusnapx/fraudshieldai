"""
Drift Detection & Monitoring Module for FraudShield AI Platform.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional

@dataclass
class DriftReport:
    feature_name: str
    psi_value: float
    drift_detected: bool
    reference_distribution: np.ndarray
    current_distribution: np.ndarray
    n_bins: int

def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10, epsilon: float = 1e-4) -> float:
    """
    Compute the Population Stability Index (PSI) between two distributions.
    PSI < 0.1: No significant shift
    0.1 <= PSI < 0.25: Moderate shift
    PSI >= 0.25: Significant shift
    """
    bins = np.histogram_bin_edges(reference, bins=n_bins)
    
    ref_counts, _ = np.histogram(reference, bins=bins)
    curr_counts, _ = np.histogram(current, bins=bins)
    
    ref_props = ref_counts / len(reference)
    curr_props = curr_counts / len(current)
    
    ref_props = np.where(ref_props == 0, epsilon, ref_props)
    curr_props = np.where(curr_props == 0, epsilon, curr_props)
    
    psi_values = (curr_props - ref_props) * np.log(curr_props / ref_props)
    psi = np.sum(psi_values)
    
    return float(psi)

def detect_feature_drift(X_reference: pd.DataFrame, X_current: pd.DataFrame, threshold: float = 0.25) -> List[DriftReport]:
    """Detect drift across all features."""
    reports = []
    
    for col in X_reference.columns:
        if col in X_current.columns:
            ref_data = X_reference[col].values
            curr_data = X_current[col].values
            
            psi = compute_psi(ref_data, curr_data)
            
            reports.append(DriftReport(
                feature_name=col,
                psi_value=psi,
                drift_detected=bool(psi >= threshold),
                reference_distribution=ref_data,
                current_distribution=curr_data,
                n_bins=10
            ))
            
    return reports

def compute_prediction_drift(prob_reference: np.ndarray, prob_current: np.ndarray, n_bins: int = 10) -> float:
    """Compute PSI on prediction probabilities."""
    return compute_psi(prob_reference, prob_current, n_bins=n_bins)

def generate_drift_summary(drift_reports: List[DriftReport], output_path: Optional[str] = None) -> pd.DataFrame:
    """Generate summary DataFrame of drift reports."""
    summary = []
    
    for report in drift_reports:
        severity = "Significant" if report.psi_value >= 0.25 else "Moderate" if report.psi_value >= 0.1 else "None"
        summary.append({
            'feature': report.feature_name,
            'psi': report.psi_value,
            'drift_detected': report.drift_detected,
            'severity': severity
        })
        
    df = pd.DataFrame(summary).sort_values('psi', ascending=False).reset_index(drop=True)
    
    if output_path:
        df.to_csv(output_path, index=False)
        
    return df

def plot_drift_dashboard(drift_reports: List[DriftReport], output_path: str, top_k: int = 10) -> None:
    """Plot bar chart of PSI values for top K drifted features."""
    df = generate_drift_summary(drift_reports)
    
    plot_df = df.head(top_k)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(plot_df['feature'], plot_df['psi'], color='skyblue')
    
    for i, bar in enumerate(bars):
        if plot_df.iloc[i]['severity'] == 'Significant':
            bar.set_color('salmon')
        elif plot_df.iloc[i]['severity'] == 'Moderate':
            bar.set_color('gold')
            
    plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Moderate Drift Threshold (0.1)')
    plt.axhline(y=0.25, color='darkred', linestyle='--', label='Significant Drift Threshold (0.25)')
    
    plt.title('Feature Drift Analysis (Population Stability Index)', fontsize=14)
    plt.ylabel('PSI Value')
    plt.xlabel('Features')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()
