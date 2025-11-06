"""Stage 03: Modeling entrypoint."""

import click
from pathlib import Path
from src.common.config import load_config
from .retrain_from_tables import main as retrain_main


@click.group()
def cli():
    """Stage 03: BERTopic fit/retrain, OCTIS adapter."""
    pass


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default="configs/bertopic.yaml",
    help="Path to BERTopic configuration file"
)
def train(config: Path):
    """Train BERTopic models."""
    print("=" * 80)
    print("Stage 03: BERTopic Training")
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
    
    if inputs:
        print("  Inputs:")
        print(f"    - chapters_csv: {inputs.get('chapters_csv', 'N/A')}")
        print(f"    - custom_stoplist: {inputs.get('custom_stoplist', 'N/A')}")
    
    if outputs:
        print("  Outputs:")
        print(f"    - models: {outputs.get('models', 'N/A')}")
        print(f"    - topics: {outputs.get('topics', 'N/A')}")
    
    print("\n" + "=" * 80)
    
    # TODO: Implement training logic
    print("Training stage - implementation pending")
    print("This stage will handle:")
    print("  - BERTopic model training")
    print("  - OCTIS adapter integration")


@cli.command()
@click.option("--dataset_csv", type=Path, required=True)
@click.option("--out_dir", type=Path, required=True)
@click.option("--text_column", default="Sentence")
@click.option("--config", type=Path, default="configs/bertopic.yaml")
def retrain(dataset_csv: Path, out_dir: Path, text_column: str, config: Path):
    """Retrain BERTopic models from topic tables."""
    # Import argparse-style args and convert to retrain_from_tables format
    import sys
    sys.argv = [
        "retrain_from_tables.py",
        "--dataset_csv", str(dataset_csv),
        "--out_dir", str(out_dir),
        "--text_column", text_column,
    ]
    retrain_main()


if __name__ == "__main__":
    cli()

