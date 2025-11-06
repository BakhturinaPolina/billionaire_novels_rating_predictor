"""Stage 04: Experiments entrypoint."""

import click
from pathlib import Path
from src.common.config import load_config


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default="configs/optuna.yaml",
    help="Path to Optuna configuration file"
)
def main(config: Path):
    """Stage 04: Bayesian search; ledgers."""
    print("=" * 80)
    print("Stage 04: Hyperparameter Experiments")
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
        print(f"    - octis_dataset: {inputs.get('octis_dataset', 'N/A')}")
    
    if outputs:
        print("  Outputs:")
        print(f"    - experiments: {outputs.get('experiments', 'N/A')}")
        print(f"    - model_evaluation_results.csv: {outputs.get('experiments', 'N/A')}/model_evaluation_results.csv")
    
    print("\n" + "=" * 80)
    
    # TODO: Implement experiment logic
    print("Experiments stage - implementation pending")
    print("This stage will handle:")
    print("  - Bayesian hyperparameter search")
    print("  - Experiment ledger management")


if __name__ == "__main__":
    main()

