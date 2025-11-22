"""Stage 04: Selection entrypoint - Pareto efficiency analysis."""

import logging
import sys
from pathlib import Path

import click

from src.common.config import load_config, resolve_path
from src.stage04_selection import pareto_analysis, pareto_plots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default="configs/selection.yaml",
    help="Path to selection configuration file"
)
def main(config: Path):
    """Stage 04: Pareto efficiency analysis and model selection."""
    print("=" * 80)
    print("Stage 04: Model Selection (Pareto Efficiency Analysis)")
    print("=" * 80)
    print(f"Loading config from: {config}")
    
    # Load configuration
    cfg = load_config(config)
    
    # Get project root (parent of src/)
    project_root = Path(__file__).parent.parent.parent
    
    # Load paths configuration
    paths_cfg = load_config(project_root / "configs/paths.yaml")
    outputs = paths_cfg.get("outputs", {})
    
    # Resolve input path
    experiments_dir = resolve_path(Path(outputs.get("experiments", "results/experiments")), project_root)
    input_csv = experiments_dir / "model_evaluation_results.csv"
    
    # Resolve output paths
    pareto_csv_path = resolve_path(Path(cfg.get("output", {}).get("pareto_csv", "results/pareto/pareto.csv")), project_root)
    plots_dir = resolve_path(Path(cfg.get("output", {}).get("plots_dir", "results/pareto")), project_root)
    
    # Get constraints
    constraints = cfg.get("constraints", {})
    
    # Get Pareto configuration
    pareto_cfg = cfg.get("pareto", {})
    objectives = pareto_cfg.get("objectives", ["coherence", "diversity"])
    
    # Get selection configuration
    selection_cfg = cfg.get("selection", {})
    top_k = selection_cfg.get("top_k", 10)
    tie_breaker = selection_cfg.get("tie_breaker", "coherence")
    
    # Get weights for combined score (default: equal weights)
    weight_coherence = cfg.get("pareto", {}).get("weight_coherence", 0.5)
    weight_diversity = cfg.get("pareto", {}).get("weight_diversity", 0.5)
    
    # Input/Output banner
    print("\nInputs → Outputs:")
    print("  Inputs:")
    print(f"    - model_evaluation_results.csv: {input_csv}")
    
    print("  Outputs:")
    print(f"    - pareto.csv: {pareto_csv_path}")
    print(f"    - plots: {plots_dir}")
    
    print(f"\n  Constraints:")
    if constraints.get("min_nr_topics") is not None:
        print(f"    - min_nr_topics >= {constraints.get('min_nr_topics')}")
    if constraints.get("max_nr_topics") is not None:
        print(f"    - max_nr_topics <= {constraints.get('max_nr_topics')}")
    if constraints.get("min_coherence") is not None:
        print(f"    - min_coherence >= {constraints.get('min_coherence')}")
    if constraints.get("min_diversity") is not None:
        print(f"    - min_diversity >= {constraints.get('min_diversity')}")
    
    print(f"\n  Selection:")
    print(f"    - top_k: {top_k}")
    print(f"    - tie_breaker: {tie_breaker}")
    print(f"    - weights: coherence={weight_coherence}, diversity={weight_diversity}")
    
    print("\n" + "=" * 80)
    
    try:
        # Step 1: Load evaluation results
        logger.info("Step 1: Loading evaluation results")
        df = pareto_analysis.load_evaluation_results(input_csv)
        
        # Step 2: Normalize metrics
        logger.info("Step 2: Normalizing metrics")
        df, scaler = pareto_analysis.normalize_metrics(df)
        
        # Step 3: Calculate combined score
        logger.info("Step 3: Calculating combined score")
        df = pareto_analysis.calculate_combined_score(
            df,
            weight_coherence=weight_coherence,
            weight_diversity=weight_diversity
        )
        
        # Ensure index is unique
        df = df.reset_index(drop=True)
        
        # Step 4: Identify Pareto efficiency (overall)
        logger.info("Step 4: Identifying Pareto-efficient models (overall)")
        df = pareto_analysis.identify_pareto_efficient(
            df,
            metrics=['Coherence_norm', 'Topic_Diversity_norm'],
            groupby=None
        )
        df = df.rename(columns={'Pareto_Efficient': 'Pareto_Efficient_All'})
        
        # Step 5: Identify Pareto efficiency (per model)
        logger.info("Step 5: Identifying Pareto-efficient models (per embedding model)")
        df = pareto_analysis.identify_pareto_efficient(
            df,
            metrics=['Coherence_norm', 'Topic_Diversity_norm'],
            groupby='Embeddings_Model'
        )
        df = df.rename(columns={'Pareto_Efficient': 'Pareto_Efficient_PerModel'})
        
        # Ensure index is unique again after operations
        df = df.reset_index(drop=True)
        
        # Step 6: Apply constraints
        logger.info("Step 6: Applying constraints")
        df_filtered = pareto_analysis.apply_constraints(df, constraints)
        
        # Step 7: Select top K models
        logger.info("Step 7: Selecting top K models")
        top_models = pareto_analysis.select_top_models(
            df_filtered,
            top_k=top_k,
            tie_breaker=tie_breaker,
            pareto_column='Pareto_Efficient_All'
        )
        
        # Step 8: Generate visualizations
        logger.info("Step 8: Generating visualizations")
        
        # Overall Pareto frontier
        pareto_all = df[df['Pareto_Efficient_All']].copy()
        pareto_plots.plot_pareto_frontier(
            df,
            pareto_df=pareto_all,
            save_path=plots_dir / "pareto_frontier_all.png",
            title="Pareto Frontier: Coherence vs. Topic Diversity (All Models)"
        )
        
        # Per-model Pareto frontiers
        pareto_plots.plot_pareto_by_model(
            df,
            save_path=plots_dir / "pareto_frontier_by_model.png"
        )
        
        # Combined score distribution
        pareto_plots.plot_combined_score_distribution(
            df,
            save_path=plots_dir / "combined_score_distribution.png"
        )
        
        # Step 9: Save results
        logger.info("Step 9: Saving results")
        pareto_csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save all Pareto-efficient models (overall)
        pareto_all_sorted = df[df['Pareto_Efficient_All']].sort_values(
            'Combined_Score',
            ascending=False
        )
        pareto_all_sorted['pareto_rank'] = range(1, len(pareto_all_sorted) + 1)
        pareto_all_sorted.to_csv(pareto_csv_path, index=False)
        
        logger.info(f"Saved {len(pareto_all_sorted)} Pareto-efficient models to: {pareto_csv_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("Analysis Complete")
        print("=" * 80)
        print(f"Total models analyzed: {len(df)}")
        print(f"Pareto-efficient (overall): {len(pareto_all_sorted)}")
        print(f"Pareto-efficient (per model): {len(df[df['Pareto_Efficient_PerModel']])}")
        print(f"Top {len(top_models)} models selected")
        print(f"\nResults saved to:")
        print(f"  - CSV: {pareto_csv_path}")
        print(f"  - Plots: {plots_dir}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

