"""Visualization functions for Pareto efficiency analysis."""

import logging
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Set seaborn style
sns.set_style("whitegrid")


def plot_pareto_frontier(
    df: pd.DataFrame,
    pareto_df: Optional[pd.DataFrame] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 8),
    title: str = "Pareto Frontier: Coherence vs. Topic Diversity"
) -> None:
    """
    Plot Pareto frontier with all models and highlight Pareto-efficient points.
    
    Args:
        df: DataFrame with all models
        pareto_df: Optional DataFrame with only Pareto-efficient models
                  (if None, uses df where Pareto_Efficient=True)
        save_path: Optional path to save figure
        figsize: Figure size tuple
        title: Plot title
    """
    logger.info("Creating Pareto frontier plot")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all models
    if 'Embeddings_Model' in df.columns:
        sns.scatterplot(
            data=df,
            x='Topic_Diversity',
            y='Coherence',
            hue='Embeddings_Model',
            palette='Set2',
            s=70,
            alpha=0.7,
            ax=ax,
            legend='auto'
        )
    else:
        ax.scatter(
            df['Topic_Diversity'],
            df['Coherence'],
            s=70,
            alpha=0.7,
            label='All models'
        )
    
    # Highlight Pareto-efficient models
    if pareto_df is None:
        if 'Pareto_Efficient' in df.columns:
            pareto_df = df[df['Pareto_Efficient']]
        else:
            logger.warning("No Pareto_Efficient column found. Skipping Pareto highlight.")
            pareto_df = pd.DataFrame()
    
    if len(pareto_df) > 0:
        ax.scatter(
            pareto_df['Topic_Diversity'],
            pareto_df['Coherence'],
            facecolors='none',
            edgecolors='red',
            s=200,
            linewidths=2,
            label='Pareto-efficient',
            zorder=10
        )
    
    ax.set_xlabel('Topic Diversity', fontsize=12)
    ax.set_ylabel('Coherence', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Pareto frontier plot to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_pareto_by_model(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize_per_subplot: tuple = (7, 5)
) -> None:
    """
    Plot Pareto front for each embedding model separately.
    
    Args:
        df: DataFrame with models, must have 'Embeddings_Model' column
        save_path: Optional path to save figure
        figsize_per_subplot: Size of each subplot
    """
    if 'Embeddings_Model' not in df.columns:
        raise ValueError("DataFrame must have 'Embeddings_Model' column")
    
    logger.info("Creating per-model Pareto frontier plots")
    
    # Get unique embedding models
    unique_models = df['Embeddings_Model'].unique()
    num_models = len(unique_models)
    
    # Calculate grid dimensions
    cols = 2
    rows = math.ceil(num_models / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_subplot[0] * cols, figsize_per_subplot[1] * rows))
    
    # Flatten axes array for easier indexing
    if num_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for i, model_name in enumerate(unique_models):
        ax = axes[i]
        # Use iloc to avoid reindexing issues
        model_mask = (df['Embeddings_Model'] == model_name).values
        subset = df.iloc[model_mask].reset_index(drop=True).copy()
        
        # Plot all runs for this model
        ax.scatter(
            subset['Topic_Diversity'],
            subset['Coherence'],
            color='blue',
            alpha=0.6,
            s=70,
            label='All runs'
        )
        
        # Highlight Pareto-efficient runs
        if 'Pareto_Efficient_PerModel' in subset.columns:
            pareto_mask = subset['Pareto_Efficient_PerModel'].fillna(False).values
            pareto_subset = subset.iloc[pareto_mask].reset_index(drop=True).copy()
        elif 'Pareto_Efficient' in subset.columns:
            pareto_mask = subset['Pareto_Efficient'].fillna(False).values
            pareto_subset = subset.iloc[pareto_mask].reset_index(drop=True).copy()
        else:
            pareto_subset = pd.DataFrame()
        
        if len(pareto_subset) > 0:
            ax.scatter(
                pareto_subset['Topic_Diversity'],
                pareto_subset['Coherence'],
                facecolors='none',
                edgecolors='red',
                s=200,
                linewidths=2,
                label='Pareto-efficient',
                zorder=10
            )
        
        ax.set_xlabel('Topic Diversity', fontsize=10)
        ax.set_ylabel('Coherence', fontsize=10)
        ax.set_title(f'{model_name}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Pareto Front by Embedding Model', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved per-model Pareto plots to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_combined_score_distribution(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 5)
) -> None:
    """
    Plot distribution of combined scores.
    
    Args:
        df: DataFrame with Combined_Score column
        save_path: Optional path to save figure
        figsize: Figure size tuple
    """
    if 'Combined_Score' not in df.columns:
        raise ValueError("DataFrame must have 'Combined_Score' column")
    
    logger.info("Creating combined score distribution plot")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    axes[0].hist(df['Combined_Score'], bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Combined Score', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Distribution of Combined Scores', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    if 'Embeddings_Model' in df.columns:
        # Grouped box plot by embedding model
        df_sorted = df.sort_values('Combined_Score', ascending=False)
        sns.boxplot(
            data=df_sorted,
            x='Embeddings_Model',
            y='Combined_Score',
            ax=axes[1]
        )
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
        axes[1].set_title('Combined Score by Embedding Model', fontsize=12, fontweight='bold')
    else:
        # Simple box plot
        axes[1].boxplot(df['Combined_Score'], vert=True)
        axes[1].set_ylabel('Combined Score', fontsize=11)
        axes[1].set_title('Combined Score Distribution', fontsize=12, fontweight='bold')
    
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved combined score distribution plot to: {save_path}")
    else:
        plt.show()
    
    plt.close()

