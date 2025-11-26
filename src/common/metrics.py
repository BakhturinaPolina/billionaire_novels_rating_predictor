"""Metrics computation utilities."""

from typing import Dict, Any, List
import numpy as np


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    metric_names: List[str]
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Predicted values
        targets: Target/true values
        metric_names: List of metric names to compute
        
    Returns:
        Dictionary of metric name -> value
    """
    metrics = {}
    
    for metric_name in metric_names:
        if metric_name == "mse":
            metrics["mse"] = np.mean((predictions - targets) ** 2)
        elif metric_name == "mae":
            metrics["mae"] = np.mean(np.abs(predictions - targets))
        elif metric_name == "rmse":
            metrics["rmse"] = np.sqrt(np.mean((predictions - targets) ** 2))
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    
    return metrics

