"""Stage 06: Labeling entrypoint."""

import click
from pathlib import Path
from src.common.config import load_config


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default="configs/labeling.yaml",
    help="Path to labeling configuration file"
)
def main(config: Path):
    """Stage 06: Semi-supervised labels; composites."""
    print("=" * 80)
    print("Stage 06: Topic Labeling")
    print("=" * 80)
    print(f"Loading config from: {config}")
    
    cfg = load_config(config)
    
    # Input/Output banner
    print("\nInputs → Outputs:")
    from src.common.config import load_config as load_paths
    from pathlib import Path as P
    paths_cfg = load_paths(P("configs/paths.yaml"))
    outputs = paths_cfg.get("outputs", {})
    
    print("  Inputs:")
    print(f"    - topics: {outputs.get('topics', 'N/A')}")
    print(f"    - pareto.csv: {outputs.get('pareto', 'N/A')}/pareto.csv")
    
    if "output" in cfg:
        print("  Outputs:")
        for key, path in cfg["output"].items():
            print(f"    - {key}: {path}")
    
    print("\n" + "=" * 80)
    
    # TODO: Implement labeling logic
    print("Labeling stage - implementation pending")
    print("This stage will handle:")
    print("  - Semi-supervised topic labeling")
    print("  - Composite building")


if __name__ == "__main__":
    main()

