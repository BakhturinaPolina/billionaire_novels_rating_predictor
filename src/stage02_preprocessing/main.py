"""Stage 02: Preprocessing entrypoint."""

import click
from pathlib import Path
from src.common.config import load_config


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default="configs/paths.yaml",
    help="Path to configuration file"
)
def main(config: Path):
    """Stage 02: Text cleaning, token/lemma, custom stoplist."""
    print("=" * 80)
    print("Stage 02: Preprocessing")
    print("=" * 80)
    print(f"Loading config from: {config}")
    
    cfg = load_config(config)
    
    # Input/Output banner
    print("\nInputs → Outputs:")
    inputs = cfg.get("inputs", {})
    outputs = cfg.get("outputs", {})
    
    if inputs:
        print("  Inputs:")
        for key, path in inputs.items():
            print(f"    - {key}: {path}")
    
    if outputs:
        print("  Outputs:")
        for key, path in outputs.items():
            print(f"    - {key}: {path}")
    
    print("\n" + "=" * 80)
    
    # TODO: Implement preprocessing logic
    print("Preprocessing stage - implementation pending")
    print("This stage will handle:")
    print("  - Text cleaning")
    print("  - Tokenization and lemmatization")
    print("  - Custom stoplist building")


if __name__ == "__main__":
    main()

