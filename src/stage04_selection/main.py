"""Stage 04: Pareto-efficient model selection entrypoint."""

import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any
import math

from src.common.config import load_config, resolve_path
from src.common.logging import setup_logging
from src.stage04_selection.pareto_analysis import (
    clean_data,
    normalize_metrics,
    calculate_combined_score,
    analyze_pareto_efficiency,
    analyze_hyperparameters
)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


@click.group()
def cli():
    """Stage 04: Pareto-efficient model selection."""
    pass


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default="configs/selection.yaml",
    help="Path to selection configuration file"
)
@click.option(
    "--paths-config",
    type=click.Path(exists=True, path_type=Path),
    default="configs/paths.yaml",
    help="Path to paths configuration file"
)
def analyze(config: Path, paths_config: Path):
    """Run complete Pareto efficiency analysis."""
    print("[SELECTION] ========== Starting Pareto Efficiency Analysis ==========")
    print("=" * 80)
    print("Stage 04: Pareto-Efficient Model Selection")
    print("=" * 80)
    
    # Load configurations
    print(f"\n[SELECTION] Loading configurations...")
    print(f"[SELECTION]   Selection config: {config}")
    print(f"[SELECTION]   Paths config: {paths_config}")
    
    selection_cfg = load_config(config)
    paths_cfg = load_config(paths_config)
    
    # Resolve paths
    inputs = selection_cfg.get("inputs", {})
    outputs = selection_cfg.get("outputs", {})
    
    input_csv = resolve_path(Path(inputs.get("model_results_csv")))
    output_base = resolve_path(Path(outputs.get("base_dir")))
    figures_dir = resolve_path(Path(outputs.get("figures_dir")))
    tables_dir = resolve_path(Path(outputs.get("tables_dir")))
    top_models_dir = resolve_path(Path(outputs.get("top_models_dir")))
    
    # Create output directories
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    top_models_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logs_dir = resolve_path(Path(paths_cfg.get("outputs", {}).get("logs", "logs")))
    logger = setup_logging(logs_dir, log_file="stage04_selection.log")
    
    print(f"[SELECTION] ✓ Config loaded")
    print(f"[SELECTION]   Input CSV: {input_csv}")
    print(f"[SELECTION]   Output base: {output_base}")
    
    # Input/Output banner
    print("\n" + "=" * 80)
    print("Inputs → Outputs:")
    print("  Inputs:")
    print(f"    - model_results_csv: {input_csv}")
    print("  Outputs:")
    print(f"    - figures: {figures_dir}")
    print(f"    - tables: {tables_dir}")
    print(f"    - top_models: {top_models_dir}")
    print("=" * 80)
    
    # Load data
    print(f"\n[SELECTION] Loading model evaluation results...")
    if not input_csv.exists():
        print(f"[SELECTION] ❌ Input file not found: {input_csv}")
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    
    df = pd.read_csv(input_csv)
    print(f"[SELECTION] ✓ Loaded {len(df)} model evaluation results")
    print(f"[SELECTION]   Columns: {list(df.columns)}")
    
    # Data cleaning
    print(f"\n[SELECTION] Cleaning data...")
    cleaning_cfg = selection_cfg.get("cleaning", {})
    df_clean, cleaning_stats = clean_data(
        df,
        remove_failed=cleaning_cfg.get("remove_failed_runs", True),
        outlier_std_dev=cleaning_cfg.get("outlier_std_dev", 2.0)
    )
    
    print(f"[SELECTION] ✓ Data cleaning complete")
    print(f"[SELECTION]   Removed failed runs: {cleaning_stats.get('removed_failed', 0)}")
    print(f"[SELECTION]   Removed outliers: {cleaning_stats.get('removed_outliers', 0)}")
    print(f"[SELECTION]   Final model count: {cleaning_stats.get('final_count', 0)}")
    print(f"[SELECTION]   Coherence bounds: {cleaning_stats.get('coherence_bounds', [])}")
    print(f"[SELECTION]   Topic Diversity bounds: {cleaning_stats.get('topic_diversity_bounds', [])}")
    
    # Plot distributions with cutoff points
    print(f"\n[SELECTION] Creating distribution plots...")
    _plot_distributions_with_cutoffs(
        df_clean,
        cleaning_stats,
        figures_dir
    )
    print(f"[SELECTION] ✓ Saved distribution plots")
    
    # Normalize metrics
    print(f"\n[SELECTION] Normalizing metrics...")
    norm_method = selection_cfg.get("normalization", {}).get("method", "zscore")
    df_clean = normalize_metrics(df_clean, method=norm_method)
    print(f"[SELECTION] ✓ Metrics normalized using {norm_method}")
    
    # Run analysis for both weighting strategies
    weighting_strategies = selection_cfg.get("weighting_strategies", {})
    pareto_cfg = selection_cfg.get("pareto", {})
    selection_cfg_params = selection_cfg.get("selection", {})
    
    # Equal weights strategy
    print(f"\n[SELECTION] {'='*80}")
    print(f"[SELECTION] Analyzing with Equal Weights Strategy")
    print(f"[SELECTION] {'='*80}")
    
    equal_weights = weighting_strategies.get("equal_weights", {})
    df_equal = calculate_combined_score(
        df_clean.copy(),
        weight_coherence=equal_weights.get("weight_coherence", 0.5),
        weight_topic_diversity=equal_weights.get("weight_topic_diversity", 0.5)
    )
    
    df_equal = analyze_pareto_efficiency(
        df_equal,
        metrics=pareto_cfg.get("metrics", ["Coherence_norm", "Topic_Diversity_norm"]),
        per_model=pareto_cfg.get("analyze_per_model", True)
    )
    
    pareto_all_count = df_equal['Pareto_Efficient_All'].sum()
    pareto_per_model_count = df_equal['Pareto_Efficient_PerModel'].sum()
    print(f"[SELECTION]   Pareto-efficient models (overall): {pareto_all_count}")
    print(f"[SELECTION]   Pareto-efficient models (per model): {pareto_per_model_count}")
    
    # Create visualizations for equal weights
    _plot_pareto_fronts(df_equal, "Equal Weights", figures_dir)
    _plot_pareto_per_model(df_equal, figures_dir)
    
    # Select and save top models
    top_k = selection_cfg_params.get("top_k", 10)
    pareto_efficient_equal = df_equal[df_equal['Pareto_Efficient_All']]
    top_models_equal = pareto_efficient_equal.nlargest(top_k, 'Combined_Score')
    top_models_equal.to_csv(top_models_dir / "top_10_equal_weights.csv", index=False)
    print(f"[SELECTION] ✓ Saved top {top_k} models (equal weights)")
    
    # Coherence priority strategy
    print(f"\n[SELECTION] {'='*80}")
    print(f"[SELECTION] Analyzing with Coherence Priority Strategy")
    print(f"[SELECTION] {'='*80}")
    
    coherence_priority = weighting_strategies.get("coherence_priority", {})
    df_priority = calculate_combined_score(
        df_clean.copy(),
        weight_coherence=coherence_priority.get("weight_coherence", 0.7),
        weight_topic_diversity=coherence_priority.get("weight_topic_diversity", 0.3)
    )
    
    df_priority = analyze_pareto_efficiency(
        df_priority,
        metrics=pareto_cfg.get("metrics", ["Coherence_norm", "Topic_Diversity_norm"]),
        per_model=False
    )
    df_priority['Pareto_Efficient'] = df_priority['Pareto_Efficient_All']
    
    pareto_count_priority = df_priority['Pareto_Efficient'].sum()
    print(f"[SELECTION]   Pareto-efficient models: {pareto_count_priority}")
    
    # Create visualization for coherence priority
    _plot_pareto_front_coherence_priority(df_priority, figures_dir)
    
    # Select and save top models
    pareto_efficient_priority = df_priority[df_priority['Pareto_Efficient']]
    top_models_priority = pareto_efficient_priority.nlargest(top_k, 'Combined_Score')
    top_models_priority.to_csv(top_models_dir / "top_10_coherence_priority.csv", index=False)
    print(f"[SELECTION] ✓ Saved top {top_k} models (coherence priority)")
    
    # Hyperparameter analysis
    print(f"\n[SELECTION] {'='*80}")
    print(f"[SELECTION] Analyzing Hyperparameters")
    print(f"[SELECTION] {'='*80}")
    
    hp_cfg = selection_cfg.get("hyperparameter_analysis", {})
    hyperparameters = hp_cfg.get("hyperparameters", [])
    performance_metrics = hp_cfg.get("performance_metrics", [])
    
    # Verify hyperparameters are present
    missing_eq = [hp for hp in hyperparameters if hp not in top_models_equal.columns]
    missing_pri = [hp for hp in hyperparameters if hp not in top_models_priority.columns]
    
    if missing_eq:
        print(f"[SELECTION] ⚠ Warning: Missing hyperparameters in equal weights: {missing_eq}")
    if missing_pri:
        print(f"[SELECTION] ⚠ Warning: Missing hyperparameters in coherence priority: {missing_pri}")
    
    if not missing_eq and not missing_pri:
        print(f"[SELECTION] ✓ All {len(hyperparameters)} hyperparameters are present")
        
        # Analyze correlations
        correlation_equal = analyze_hyperparameters(
            top_models_equal,
            hyperparameters,
            performance_metrics
        )
        correlation_priority = analyze_hyperparameters(
            top_models_priority,
            hyperparameters,
            performance_metrics
        )
        
        correlation_equal.to_csv(
            tables_dir / "correlation_analysis_equal_weights.csv",
            index=False
        )
        correlation_priority.to_csv(
            tables_dir / "correlation_analysis_coherence_priority.csv",
            index=False
        )
        print(f"[SELECTION] ✓ Saved correlation analysis tables")
        
        # Create hyperparameter boxplots
        _plot_hyperparameter_boxplots(
            top_models_equal,
            top_models_priority,
            hyperparameters,
            figures_dir
        )
        print(f"[SELECTION] ✓ Saved hyperparameter boxplots")
    
    # Summary
    print("\n" + "=" * 80)
    print("[SELECTION] Analysis Summary")
    print("=" * 80)
    print(f"📊 Total models analyzed: {len(df_clean)}")
    print(f"✅ Pareto-efficient (equal weights): {pareto_all_count}")
    print(f"✅ Pareto-efficient (coherence priority): {pareto_count_priority}")
    print(f"📁 Output directory: {output_base}")
    print(f"   - Figures: {figures_dir}")
    print(f"   - Tables: {tables_dir}")
    print(f"   - Top models: {top_models_dir}")
    print("[SELECTION] ========== Analysis completed ==========")


def _plot_distributions_with_cutoffs(
    df: pd.DataFrame,
    stats: Dict[str, Any],
    output_dir: Path
):
    """Plot distributions with cutoff points."""
    plt.figure(figsize=(12, 5))
    
    coherence_bounds = stats.get('coherence_bounds', [])
    diversity_bounds = stats.get('topic_diversity_bounds', [])
    
    # Coherence distribution
    plt.subplot(1, 2, 1)
    sns.histplot(df['Coherence'], bins=20, kde=True)
    if len(coherence_bounds) == 2:
        plt.axvline(coherence_bounds[0], color='red', linestyle='dashed', linewidth=1, label='Lower Cutoff')
        plt.axvline(coherence_bounds[1], color='green', linestyle='dashed', linewidth=1, label='Upper Cutoff')
    plt.title('Coherence Distribution with Cutoff Points')
    plt.xlabel('Coherence')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Topic Diversity distribution
    plt.subplot(1, 2, 2)
    sns.histplot(df['Topic_Diversity'], bins=20, kde=True)
    if len(diversity_bounds) == 2:
        plt.axvline(diversity_bounds[0], color='red', linestyle='dashed', linewidth=1, label='Lower Cutoff')
        plt.axvline(diversity_bounds[1], color='green', linestyle='dashed', linewidth=1, label='Upper Cutoff')
    plt.title('Topic Diversity Distribution with Cutoff Points')
    plt.xlabel('Topic Diversity')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "distribution_with_cutoffs.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_pareto_fronts(df: pd.DataFrame, strategy_name: str, output_dir: Path):
    """Plot Pareto front for all models."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.scatterplot(
        data=df,
        x='Topic_Diversity',
        y='Coherence',
        hue='Embeddings_Model',
        palette='Set2',
        s=70,
        alpha=0.7,
        ax=ax
    )
    
    pareto_models = df[df['Pareto_Efficient_All']]
    ax.scatter(
        pareto_models['Topic_Diversity'],
        pareto_models['Coherence'],
        facecolors='none',
        edgecolors='red',
        s=200,
        linewidths=2,
        label='Pareto-efficient',
        zorder=10
    )
    
    ax.set_title(f'Pareto Front: Coherence vs. Topic Diversity ({strategy_name})', fontsize=14)
    ax.set_xlabel('Topic Diversity', fontsize=12)
    ax.set_ylabel('Coherence', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = "pareto_front_equal_weights.png" if "Equal" in strategy_name else "pareto_front_coherence_priority.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_pareto_per_model(df: pd.DataFrame, output_dir: Path):
    """Plot Pareto fronts per embedding model."""
    unique_models = df['Embeddings_Model'].unique()
    num_models = len(unique_models)
    cols = 2
    rows = math.ceil(num_models / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if num_models > 1 else [axes]
    
    for i, model_name in enumerate(unique_models):
        subset = df[df['Embeddings_Model'] == model_name]
        
        axes[i].scatter(
            subset['Topic_Diversity'],
            subset['Coherence'],
            label='All Runs',
            color='blue',
            alpha=0.6,
            s=50
        )
        
        pareto_subset = subset[subset['Pareto_Efficient_PerModel']]
        if len(pareto_subset) > 0:
            axes[i].scatter(
                pareto_subset['Topic_Diversity'],
                pareto_subset['Coherence'],
                label='Pareto Efficient',
                color='red',
                s=100,
                edgecolors='darkred',
                linewidths=1.5
            )
        
        axes[i].set_title(f'Pareto Front for {model_name}', fontsize=12)
        axes[i].set_xlabel('Topic Diversity', fontsize=10)
        axes[i].set_ylabel('Coherence', fontsize=10)
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_models, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "pareto_fronts_per_model.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_pareto_front_coherence_priority(df: pd.DataFrame, output_dir: Path):
    """Plot Pareto front for coherence priority strategy."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.scatterplot(
        data=df,
        x='Topic_Diversity',
        y='Coherence',
        hue='Embeddings_Model',
        palette='Set2',
        s=70,
        alpha=0.7,
        ax=ax
    )
    
    pareto_models = df[df['Pareto_Efficient']]
    ax.scatter(
        pareto_models['Topic_Diversity'],
        pareto_models['Coherence'],
        facecolors='none',
        edgecolors='red',
        s=200,
        linewidths=2,
        label='Pareto-efficient (Coherence Priority)',
        zorder=10
    )
    
    ax.set_title('Pareto Front: Coherence vs. Topic Diversity (Coherence Prioritized)', fontsize=14)
    ax.set_xlabel('Topic Diversity', fontsize=12)
    ax.set_ylabel('Coherence', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "pareto_front_coherence_priority.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_hyperparameter_boxplots(
    df_equal: pd.DataFrame,
    df_priority: pd.DataFrame,
    hyperparameters: list,
    output_dir: Path
):
    """Create normalized hyperparameter boxplots."""
    from sklearn.preprocessing import MinMaxScaler
    
    scaler_hp = MinMaxScaler()
    df_normalized_eq = df_equal.copy()
    df_normalized_pri = df_priority.copy()
    
    df_normalized_eq[hyperparameters] = scaler_hp.fit_transform(df_equal[hyperparameters])
    df_normalized_pri[hyperparameters] = scaler_hp.fit_transform(df_priority[hyperparameters])
    
    # Melt for plotting
    df_melted_eq = df_normalized_eq.melt(
        value_vars=hyperparameters,
        var_name='Hyperparameter',
        value_name='Value'
    )
    df_melted_eq['Strategy'] = 'Equal Weights'
    
    df_melted_pri = df_normalized_pri.melt(
        value_vars=hyperparameters,
        var_name='Hyperparameter',
        value_name='Value'
    )
    df_melted_pri['Strategy'] = 'Coherence Priority'
    
    df_melted = pd.concat([df_melted_eq, df_melted_pri], ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=df_melted,
        x='Value',
        y='Hyperparameter',
        hue='Strategy',
        orient='h',
        ax=ax
    )
    
    ax.set_title('Boxplots of Normalized Hyperparameters', fontsize=16)
    ax.set_xlabel('Normalized Values', fontsize=12)
    ax.set_ylabel('Hyperparameter', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "hyperparameter_boxplots.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    cli()

