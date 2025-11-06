"""Stage 05: Selection entrypoint."""

import click
from pathlib import Path
from src.common.config import load_config


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default="configs/selection.yaml",
    help="Path to selection configuration file"
)
def main(config: Path):
    """Stage 05: Pareto + constraints (nr_topics >= 200)."""
    print("=" * 80)
    print("Stage 05: Model Selection (Pareto)")
    print("=" * 80)
    print(f"Loading config from: {config}")
    
    cfg = load_config(config)
    
    # Enforce constraint
    min_topics = cfg.get("constraints", {}).get("min_nr_topics", 200)
    
    # Input/Output banner
    print("\nInputs → Outputs:")
    from src.common.config import load_config as load_paths
    from pathlib import Path as P
    paths_cfg = load_paths(P("configs/paths.yaml"))
    outputs = paths_cfg.get("outputs", {})
    
    print("  Inputs:")
    print(f"    - model_evaluation_results.csv: {outputs.get('experiments', 'N/A')}/model_evaluation_results.csv")
    print(f"    - topics: {outputs.get('topics', 'N/A')}")
    
    print("  Outputs:")
    pareto_csv = cfg.get("output", {}).get("pareto_csv", "results/pareto/pareto.csv")
    print(f"    - pareto.csv: {pareto_csv}")
    print(f"    - topics: {cfg.get('output', {}).get('topics_dir', 'results/pareto/topics')}")
    
    print(f"\n  Constraints:")
    print(f"    - min_nr_topics >= {min_topics}")
    
    print("\n" + "=" * 80)
    
    # TODO: Implement selection logic
    print("Selection stage - implementation pending")
    print("This stage will handle:")
    print("  - Pareto efficiency analysis")
    print("  - Model selection with constraints")
    print(f"  - Enforcing min_nr_topics >= {min_topics}")


if __name__ == "__main__":
    main()

