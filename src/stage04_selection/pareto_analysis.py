"""Pareto efficiency analysis functions."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
from typing import List, Tuple, Dict, Any
import math


def identify_pareto(df: pd.DataFrame, metrics: List[str]) -> np.ndarray:
    """
    Identify Pareto-efficient points for given metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data points
    metrics : list
        List of column names to use for Pareto analysis
        
    Returns:
    --------
    np.ndarray
        Boolean array indicating whether each row is Pareto-efficient
    """
    pareto_efficient = np.ones(df.shape[0], dtype=bool)
    # Use enumerate to get positional index, not DataFrame index
    for pos_idx, (df_idx, row) in enumerate(df.iterrows()):
        # If there are any other points that are strictly better, mark as not Pareto-efficient
        # Compare current row with all rows (including itself)
        other_rows_better = (
            np.all(df[metrics].values >= row[metrics].values, axis=1) & 
            np.any(df[metrics].values > row[metrics].values, axis=1)
        )
        pareto_efficient[pos_idx] = not np.any(other_rows_better)
    return pareto_efficient


def clean_data(
    df: pd.DataFrame,
    remove_failed: bool = True,
    outlier_std_dev: float = 2.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean model evaluation data by removing failed runs and outliers.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with model evaluation results
    remove_failed : bool
        Remove runs where Topic_Diversity or Coherence equals 1.0
    outlier_std_dev : float
        Number of standard deviations for outlier bounds
        
    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Cleaned dataframe and statistics dictionary
    """
    df_clean = df.copy()
    stats = {}
    
    # Remove failed runs
    if remove_failed:
        initial_count = len(df_clean)
        df_clean = df_clean[
            (df_clean['Topic_Diversity'] < 1.0) & 
            (df_clean['Coherence'] < 1.0)
        ].copy()
        stats['removed_failed'] = initial_count - len(df_clean)
        stats['after_failed_removal'] = len(df_clean)
    
    # Compute bounds for outlier detection
    coherence_mean = df_clean['Coherence'].mean()
    coherence_std = df_clean['Coherence'].std()
    topic_diversity_mean = df_clean['Topic_Diversity'].mean()
    topic_diversity_std = df_clean['Topic_Diversity'].std()
    
    coherence_lower = coherence_mean - outlier_std_dev * coherence_std
    coherence_upper = coherence_mean + outlier_std_dev * coherence_std
    topic_diversity_lower = topic_diversity_mean - outlier_std_dev * topic_diversity_std
    topic_diversity_upper = topic_diversity_mean + outlier_std_dev * topic_diversity_std
    
    stats['coherence_bounds'] = [coherence_lower, coherence_upper]
    stats['topic_diversity_bounds'] = [topic_diversity_lower, topic_diversity_upper]
    
    # Remove outliers sequentially
    before_outliers = len(df_clean)
    df_clean = df_clean[
        (df_clean['Coherence'] >= coherence_lower) & 
        (df_clean['Coherence'] <= coherence_upper)
    ]
    df_clean = df_clean[
        (df_clean['Topic_Diversity'] >= topic_diversity_lower) & 
        (df_clean['Topic_Diversity'] <= topic_diversity_upper)
    ]
    df_clean = df_clean.reset_index(drop=True)
    
    stats['removed_outliers'] = before_outliers - len(df_clean)
    stats['final_count'] = len(df_clean)
    
    return df_clean, stats


def normalize_metrics(
    df: pd.DataFrame,
    method: str = "zscore"
) -> pd.DataFrame:
    """
    Normalize Coherence and Topic Diversity metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with Coherence and Topic_Diversity columns
    method : str
        Normalization method: "zscore" or "minmax"
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added normalized columns
    """
    df_norm = df.copy()
    
    if method == "zscore":
        scaler = StandardScaler()
        df_norm[['Coherence_norm', 'Topic_Diversity_norm']] = scaler.fit_transform(
            df_norm[['Coherence', 'Topic_Diversity']]
        )
    elif method == "minmax":
        scaler = MinMaxScaler()
        df_norm[['Coherence_norm', 'Topic_Diversity_norm']] = scaler.fit_transform(
            df_norm[['Coherence', 'Topic_Diversity']]
        )
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return df_norm


def calculate_combined_score(
    df: pd.DataFrame,
    weight_coherence: float,
    weight_topic_diversity: float
) -> pd.DataFrame:
    """
    Calculate combined score from normalized metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with normalized metrics
    weight_coherence : float
        Weight for coherence in combined score
    weight_topic_diversity : float
        Weight for topic diversity in combined score
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added Combined_Score column
    """
    df_scored = df.copy()
    df_scored['Combined_Score'] = (
        weight_coherence * df_scored['Coherence_norm'] + 
        weight_topic_diversity * df_scored['Topic_Diversity_norm']
    )
    return df_scored


def analyze_pareto_efficiency(
    df: pd.DataFrame,
    metrics: List[str],
    per_model: bool = True
) -> pd.DataFrame:
    """
    Identify Pareto-efficient models.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with normalized metrics
    metrics : list
        List of metric column names for Pareto analysis
    per_model : bool
        Also analyze Pareto efficiency per embedding model
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with Pareto efficiency flags
    """
    df_pareto = df.copy()
    
    # Overall Pareto efficiency
    df_pareto['Pareto_Efficient_All'] = identify_pareto(df_pareto, metrics)
    
    # Per-model Pareto efficiency
    if per_model:
        df_pareto['Pareto_Efficient_PerModel'] = False
        for model_name, group in df_pareto.groupby('Embeddings_Model'):
            pareto_flags = identify_pareto(group, metrics)
            df_pareto.loc[group.index, 'Pareto_Efficient_PerModel'] = pareto_flags
    
    return df_pareto


def cohen_d(group1: pd.Series, group2: pd.Series) -> float:
    """Calculate Cohen's d for two groups."""
    diff_mean = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt(((np.std(group1, ddof=1) ** 2) + (np.std(group2, ddof=1) ** 2)) / 2)
    return diff_mean / pooled_std if pooled_std > 0 else 0


def choose_test(df: pd.DataFrame, parameter: str, metric: str) -> Tuple[float, float, str]:
    """
    Choose between Pearson or Spearman based on data distribution.
    
    Returns:
    --------
    Tuple[float, float, str]
        Correlation coefficient, p-value, and test type
    """
    if abs(df[parameter].skew()) < 1:
        corr, p_value = pearsonr(df[parameter], df[metric])
        test_type = "Pearson"
    else:
        corr, p_value = spearmanr(df[parameter], df[metric])
        test_type = "Spearman"
    return corr, p_value, test_type


def calculate_cohens_d(df: pd.DataFrame, parameter: str, metric: str) -> float:
    """Split by median and calculate Cohen's d."""
    median_value = df[parameter].median()
    group1 = df[df[parameter] <= median_value][metric]
    group2 = df[df[parameter] > median_value][metric]
    return cohen_d(group1, group2)


def analyze_hyperparameters(
    df: pd.DataFrame,
    hyperparameters: List[str],
    performance_metrics: List[str]
) -> pd.DataFrame:
    """
    Analyze correlations and effect sizes between hyperparameters and performance metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with hyperparameters and performance metrics
    hyperparameters : list
        List of hyperparameter column names
    performance_metrics : list
        List of performance metric column names
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with correlation analysis results
    """
    results = {
        'Hyperparameter': [],
        'Metric': [],
        'Correlation': [],
        'p-value': [],
        'Test_Type': [],
        'Cohen_d': []
    }
    
    for param in hyperparameters:
        for metric in performance_metrics:
            corr, p_val, test_type = choose_test(df, param, metric)
            cohens_d_val = calculate_cohens_d(df, param, metric)
            
            results['Hyperparameter'].append(param)
            results['Metric'].append(metric)
            results['Correlation'].append(corr)
            results['p-value'].append(p_val)
            results['Test_Type'].append(test_type)
            results['Cohen_d'].append(cohens_d_val)
    
    return pd.DataFrame(results)

