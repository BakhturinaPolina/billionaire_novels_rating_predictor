"""Pareto efficiency analysis for model selection."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_evaluation_results(csv_path: Path) -> pd.DataFrame:
    """
    Load and validate model evaluation results CSV.
    
    Args:
        csv_path: Path to model_evaluation_results.csv
        
    Returns:
        DataFrame with evaluation results
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    logger.info(f"Loading evaluation results from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Reset index to ensure it's unique
    df = df.reset_index(drop=True)
    
    # Validate required columns
    required_columns = ['Embeddings_Model', 'Coherence', 'Topic_Diversity']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    logger.info(f"Loaded {len(df)} model evaluation results")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def normalize_metrics(
    df: pd.DataFrame,
    metrics: List[str] = None,
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Apply Z-score normalization to specified metrics.
    
    Args:
        df: DataFrame with metrics to normalize
        metrics: List of metric column names to normalize
        scaler: Optional pre-fitted scaler (if None, fits new scaler)
        
    Returns:
        Tuple of (DataFrame with normalized columns added, fitted scaler)
    """
    if metrics is None:
        metrics = ['Coherence', 'Topic_Diversity']
    
    # Check that all metrics exist
    missing_metrics = [m for m in metrics if m not in df.columns]
    if missing_metrics:
        raise ValueError(f"Missing metric columns: {missing_metrics}")
    
    # Ensure index is unique
    df = df.reset_index(drop=True).copy()
    
    # Check if normalized columns already exist
    normalized_columns = [f"{metric}_norm" for metric in metrics]
    existing_norm_cols = [col for col in normalized_columns if col in df.columns]
    
    if existing_norm_cols:
        logger.info(f"Normalized columns already exist: {existing_norm_cols}. Recalculating...")
        # Remove existing normalized columns
        df = df.drop(columns=existing_norm_cols)
    
    logger.info(f"Normalizing metrics: {metrics}")
    
    # Extract metric values
    metric_values = df[metrics].values
    
    # Fit or use existing scaler
    if scaler is None:
        scaler = StandardScaler()
        normalized_values = scaler.fit_transform(metric_values)
    else:
        normalized_values = scaler.transform(metric_values)
    
    # Add normalized columns
    for i, col_name in enumerate(normalized_columns):
        df[col_name] = normalized_values[:, i]
    
    logger.info(f"Added normalized columns: {normalized_columns}")
    logger.info(f"Normalized value ranges: {[(col, df[col].min(), df[col].max()) for col in normalized_columns]}")
    
    return df, scaler


def calculate_combined_score(
    df: pd.DataFrame,
    weight_coherence: float = 0.5,
    weight_diversity: float = 0.5,
    coherence_col: str = 'Coherence_norm',
    diversity_col: str = 'Topic_Diversity_norm'
) -> pd.DataFrame:
    """
    Calculate combined score from normalized metrics.
    
    Args:
        df: DataFrame with normalized metrics
        weight_coherence: Weight for coherence (default: 0.5)
        weight_diversity: Weight for diversity (default: 0.5)
        coherence_col: Column name for normalized coherence
        diversity_col: Column name for normalized diversity
        
    Returns:
        DataFrame with Combined_Score column added
    """
    # Validate weights sum to 1.0 (with small tolerance)
    weight_sum = weight_coherence + weight_diversity
    if abs(weight_sum - 1.0) > 1e-6:
        logger.warning(f"Weights sum to {weight_sum}, not 1.0. Normalizing...")
        weight_coherence = weight_coherence / weight_sum
        weight_diversity = weight_diversity / weight_sum
    
    # Check that normalized columns exist
    if coherence_col not in df.columns:
        raise ValueError(f"Column not found: {coherence_col}")
    if diversity_col not in df.columns:
        raise ValueError(f"Column not found: {diversity_col}")
    
    logger.info(f"Calculating combined score with weights: coherence={weight_coherence}, diversity={weight_diversity}")
    
    # Calculate combined score
    df['Combined_Score'] = (
        weight_coherence * df[coherence_col] +
        weight_diversity * df[diversity_col]
    )
    
    logger.info(f"Combined score range: [{df['Combined_Score'].min():.4f}, {df['Combined_Score'].max():.4f}]")
    
    return df


def identify_pareto_efficient(
    df: pd.DataFrame,
    metrics: List[str],
    groupby: Optional[str] = None
) -> pd.DataFrame:
    """
    Identify Pareto-efficient points.
    
    A point is Pareto-efficient if no other point dominates it (i.e., no other point
    has >= values in all metrics and > value in at least one metric).
    
    Args:
        df: DataFrame with metrics
        metrics: List of metric column names for Pareto analysis
        groupby: Optional column name to group by (e.g., 'Embeddings_Model')
                If provided, Pareto efficiency is calculated within each group
        
    Returns:
        DataFrame with 'Pareto_Efficient' boolean column added
    """
    # Validate metrics exist
    missing_metrics = [m for m in metrics if m not in df.columns]
    if missing_metrics:
        raise ValueError(f"Missing metric columns: {missing_metrics}")
    
    logger.info(f"Identifying Pareto-efficient points for metrics: {metrics}")
    if groupby:
        logger.info(f"Grouping by: {groupby}")
    
    # Initialize result column
    df['Pareto_Efficient'] = False
    
    if groupby:
        # Calculate Pareto efficiency within each group
        for group_name, group_df in df.groupby(groupby):
            group_indices = group_df.index
            pareto_flags = _calculate_pareto_flags(group_df[metrics].values)
            df.loc[group_indices, 'Pareto_Efficient'] = pareto_flags
            logger.info(f"Group '{group_name}': {pareto_flags.sum()}/{len(group_df)} Pareto-efficient")
    else:
        # Calculate Pareto efficiency across all points
        pareto_flags = _calculate_pareto_flags(df[metrics].values)
        df['Pareto_Efficient'] = pareto_flags
        logger.info(f"Overall: {pareto_flags.sum()}/{len(df)} Pareto-efficient")
    
    return df


def _calculate_pareto_flags(metric_values: np.ndarray) -> np.ndarray:
    """
    Calculate Pareto efficiency flags for a set of points.
    
    Args:
        metric_values: Array of shape (n_points, n_metrics) with metric values
        
    Returns:
        Boolean array of shape (n_points,) indicating Pareto efficiency
    """
    n_points = metric_values.shape[0]
    pareto_efficient = np.ones(n_points, dtype=bool)
    
    # For each point, check if any other point dominates it
    for i in range(n_points):
        point = metric_values[i]
        # Check if any other point has >= values in all metrics and > in at least one
        # A point j dominates point i if:
        #   - All metrics of j >= metrics of i (all(metric_values[j] >= point))
        #   - At least one metric of j > metric of i (any(metric_values[j] > point))
        dominated = np.any(
            np.all(metric_values >= point, axis=1) &
            np.any(metric_values > point, axis=1)
        )
        # If dominated, then not Pareto-efficient
        pareto_efficient[i] = not dominated
    
    return pareto_efficient


def apply_constraints(
    df: pd.DataFrame,
    constraints: Dict[str, Optional[float]]
) -> pd.DataFrame:
    """
    Apply constraints to filter DataFrame.
    
    Args:
        df: DataFrame to filter
        constraints: Dictionary of constraint_name -> threshold value
                    (None means constraint is not applied)
        
    Returns:
        Filtered DataFrame
    """
    original_len = len(df)
    df_filtered = df.copy()
    
    logger.info("Applying constraints:")
    
    # Apply min_nr_topics constraint (if specified and column exists)
    if 'min_nr_topics' in constraints and constraints['min_nr_topics'] is not None:
        threshold = constraints['min_nr_topics']
        if 'nr_topics' in df_filtered.columns:
            before = len(df_filtered)
            df_filtered = df_filtered[df_filtered['nr_topics'] >= threshold]
            logger.info(f"  min_nr_topics >= {threshold}: {before} -> {len(df_filtered)} rows")
        else:
            logger.warning("  min_nr_topics constraint specified but 'nr_topics' column not found. Skipping.")
    
    # Apply max_nr_topics constraint (if specified and column exists)
    if 'max_nr_topics' in constraints and constraints['max_nr_topics'] is not None:
        threshold = constraints['max_nr_topics']
        if 'nr_topics' in df_filtered.columns:
            before = len(df_filtered)
            df_filtered = df_filtered[df_filtered['nr_topics'] <= threshold]
            logger.info(f"  max_nr_topics <= {threshold}: {before} -> {len(df_filtered)} rows")
        else:
            logger.warning("  max_nr_topics constraint specified but 'nr_topics' column not found. Skipping.")
    
    # Apply min_coherence constraint
    if 'min_coherence' in constraints and constraints['min_coherence'] is not None:
        threshold = constraints['min_coherence']
        before = len(df_filtered)
        df_filtered = df_filtered[df_filtered['Coherence'] >= threshold]
        logger.info(f"  min_coherence >= {threshold}: {before} -> {len(df_filtered)} rows")
    
    # Apply min_diversity constraint
    if 'min_diversity' in constraints and constraints['min_diversity'] is not None:
        threshold = constraints['min_diversity']
        before = len(df_filtered)
        df_filtered = df_filtered[df_filtered['Topic_Diversity'] >= threshold]
        logger.info(f"  min_diversity >= {threshold}: {before} -> {len(df_filtered)} rows")
    
    logger.info(f"Constraints applied: {original_len} -> {len(df_filtered)} rows")
    
    return df_filtered


def select_top_models(
    df: pd.DataFrame,
    top_k: int = 10,
    tie_breaker: str = 'coherence',
    score_column: str = 'Combined_Score',
    pareto_column: str = 'Pareto_Efficient'
) -> pd.DataFrame:
    """
    Select top K models from Pareto-efficient models.
    
    Args:
        df: DataFrame with Pareto-efficient models
        top_k: Number of top models to select
        tie_breaker: Column name to use for breaking ties (default: 'coherence')
        score_column: Column name for sorting (default: 'Combined_Score')
        pareto_column: Column name indicating Pareto efficiency (default: 'Pareto_Efficient')
        
    Returns:
        DataFrame with top K models, sorted by score
    """
    # Ensure index is unique before filtering
    df = df.reset_index(drop=True).copy()
    
    # Filter to Pareto-efficient models if column exists
    if pareto_column in df.columns:
        # Get boolean mask as numpy array to avoid reindexing issues
        mask = df[pareto_column].fillna(False).values
        df_pareto = df.iloc[mask].reset_index(drop=True).copy()
        logger.info(f"Filtering to {len(df_pareto)} Pareto-efficient models (using column: {pareto_column})")
    elif 'Pareto_Efficient' in df.columns:
        mask = df['Pareto_Efficient'].fillna(False).values
        df_pareto = df.iloc[mask].reset_index(drop=True).copy()
        logger.info(f"Filtering to {len(df_pareto)} Pareto-efficient models (using default column)")
    else:
        df_pareto = df.copy()
        logger.warning("'Pareto_Efficient' column not found. Using all models.")
    
    if len(df_pareto) == 0:
        logger.warning("No Pareto-efficient models found. Returning empty DataFrame.")
        return df_pareto
    
    # Sort by score (descending), then by tie_breaker (descending)
    if score_column not in df_pareto.columns:
        raise ValueError(f"Score column not found: {score_column}")
    
    sort_columns = [score_column]
    if tie_breaker in df_pareto.columns and tie_breaker != score_column:
        sort_columns.append(tie_breaker)
    
    df_sorted = df_pareto.sort_values(
        by=sort_columns,
        ascending=False
    )
    
    # Select top K
    top_models = df_sorted.head(top_k).copy()
    
    # Add rank column
    top_models['pareto_rank'] = range(1, len(top_models) + 1)
    
    logger.info(f"Selected top {len(top_models)} models")
    logger.info(f"Score range: [{top_models[score_column].min():.4f}, {top_models[score_column].max():.4f}]")
    
    return top_models

