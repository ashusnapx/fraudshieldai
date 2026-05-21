"""
Premium SHAP Explainability Module for FraudShield AI Platform.
"""

import matplotlib.pyplot as plt
import shap
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def _extract_tree_model(pipeline):
    """Extract model from an imblearn or sklearn pipeline."""
    if hasattr(pipeline, 'steps'):
        return pipeline.steps[-1][1]
    return pipeline

def generate_shap_summary(model, X_sample, output_path, max_display=20) -> bool:
    """Generate and save SHAP summary beeswarm plot."""
    try:
        tree_model = _extract_tree_model(model)
        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(X_sample)
        
        # Handle XGBoost vs RF shap value output format
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class
            
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        logger.error(f"Failed to generate SHAP summary: {e}")
        return False

def generate_shap_waterfall(model, X_sample, output_path, sample_idx=0, max_display=15) -> bool:
    """Generate SHAP waterfall plot for a specific prediction."""
    try:
        tree_model = _extract_tree_model(model)
        explainer = shap.Explainer(tree_model)
        shap_values = explainer(X_sample)
        
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(shap_values[sample_idx], max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        logger.error(f"Failed to generate SHAP waterfall: {e}")
        return False

def generate_shap_dependence(model, X_sample, feature_name, output_path, interaction_feature=None) -> bool:
    """Generate SHAP dependence plot."""
    try:
        tree_model = _extract_tree_model(model)
        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        plt.figure(figsize=(10, 8))
        shap.dependence_plot(
            feature_name, shap_values, X_sample, 
            interaction_index=interaction_feature, show=False
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        logger.error(f"Failed to generate SHAP dependence: {e}")
        return False

def generate_shap_force(model, X_sample, output_path, sample_idx=0) -> bool:
    """Generate SHAP force plot for a single prediction and save as static image."""
    try:
        tree_model = _extract_tree_model(model)
        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(X_sample)
        expected_value = explainer.expected_value
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            expected_value = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
            
        plt.figure(figsize=(12, 4))
        # Force plots in matplotlib require passing matplotlib=True
        shap.force_plot(
            expected_value, shap_values[sample_idx,:], X_sample.iloc[sample_idx,:], 
            matplotlib=True, show=False
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        logger.error(f"Failed to generate SHAP force plot: {e}")
        return False

def generate_local_explanations(model, X_sample, feature_names, top_n=5, n_examples=3) -> pd.DataFrame:
    """Generate top N driving features for N examples."""
    try:
        tree_model = _extract_tree_model(model)
        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(X_sample.head(n_examples))
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        results = []
        for i in range(min(n_examples, len(X_sample))):
            sv = shap_values[i]
            # Get indices of top N absolute SHAP values
            top_indices = np.argsort(np.abs(sv))[::-1][:top_n]
            
            for rank, idx in enumerate(top_indices):
                val = sv[idx]
                feat = feature_names[idx]
                feat_val = X_sample.iloc[i, idx]
                results.append({
                    'example_idx': i,
                    'rank': rank + 1,
                    'feature': feat,
                    'shap_value': val,
                    'feature_value': feat_val,
                    'direction': 'Increases Risk' if val > 0 else 'Decreases Risk'
                })
        return pd.DataFrame(results)
    except Exception as e:
        logger.error(f"Failed to generate local explanations: {e}")
        return pd.DataFrame()
