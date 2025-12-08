"""Stage 07: Analysis entrypoint."""

import click
from pathlib import Path
from src.common.config import load_config


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default="configs/scoring.yaml",
    help="Path to scoring configuration file"
)
def main(config: Path):
    """Stage 07: Goodreads scoring/strata, stats, FDR."""
    print("=" * 80)
    print("Stage 07: Statistical Analysis")
    print("=" * 80)
    print(f"Loading config from: {config}")
    
    cfg = load_config(config)
    
    # Input/Output banner
    print("\nInputs → Outputs:")
    from src.common.config import load_config as load_paths
    from pathlib import Path as P
    paths_cfg = load_paths(P("configs/paths.yaml"))
    inputs = paths_cfg.get("inputs", {})
    outputs = paths_cfg.get("outputs", {})
    
    print("  Inputs:")
    print(f"    - goodreads_csv: {inputs.get('goodreads_csv', 'N/A')}")
    print(f"    - topics: {outputs.get('topics', 'N/A')}")
    print(f"    - pareto.csv: {outputs.get('pareto', 'N/A')}/pareto.csv")
    
    if "output" in cfg:
        print("  Outputs:")
        for key, path in cfg["output"].items():
            print(f"    - {key}: {path}")
    
    print("\n" + "=" * 80)
    
    # TODO: Implement analysis logic
    print("Analysis stage - implementation pending")
    print("This stage will handle:")
    print("  - Goodreads scoring and stratification")
    print("  - Statistical analysis")
    print("  - FDR correction")


if __name__ == "__main__":
    main()

