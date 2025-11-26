"""Stage 03: Modeling entrypoint."""

import click
from pathlib import Path
from src.common.config import load_config
# Note: retrain_from_tables.py not found - retrain command may need implementation
# from .retrain_from_tables import main as retrain_main


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
    print("[TRAIN_CMD] ========== Starting train command ==========")
    print("=" * 80)
    print("Stage 03: BERTopic Training")
    print("=" * 80)
    print(f"[TRAIN_CMD] Loading config from: {config}")
    print(f"Loading config from: {config}")
    
    print("[TRAIN_CMD] Loading configuration file...")
    cfg = load_config(config)
    print(f"[TRAIN_CMD] ✓ Config loaded: {list(cfg.keys()) if isinstance(cfg, dict) else type(cfg)}")
    
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
    print("[RETRAIN_CMD] ========== Starting retrain command ==========")
    print(f"[RETRAIN_CMD] Arguments:")
    print(f"[RETRAIN_CMD]   dataset_csv: {dataset_csv}")
    print(f"[RETRAIN_CMD]   out_dir: {out_dir}")
    print(f"[RETRAIN_CMD]   text_column: {text_column}")
    print(f"[RETRAIN_CMD]   config: {config}")
    print("[RETRAIN_CMD] ⚠️  Retrain command - implementation pending")
    print("[RETRAIN_CMD] Note: retrain_from_tables.py not found - needs implementation")
    # TODO: Implement retrain functionality
    # The retrain_from_tables.py file is missing - this command needs to be implemented
    print("[RETRAIN_CMD] ========== retrain command completed ==========")


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default="configs/octis.yaml",
    help="Path to OCTIS configuration file"
)
def optimize(config: Path):
    """Run hyperparameter optimization with OCTIS."""
    print("[OPTIMIZE_CMD] ========== Starting optimize command ==========")
    print("=" * 80)
    print("Stage 03: Hyperparameter Optimization (merged from Stage 04)")
    print("=" * 80)
    print(f"[OPTIMIZE_CMD] Loading config from: {config}")
    
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
    print("[OPTIMIZE_CMD] Running bertopic_runner.py for hyperparameter optimization...")
    print("[OPTIMIZE_CMD] Note: bertopic_runner.py contains the full optimization implementation")
    
    # Import and execute bertopic_runner
    # Note: bertopic_runner.py runs as a script, so we import it to execute
    try:
        import sys
        import importlib.util
        bertopic_runner_path = Path(__file__).parent / "bertopic_runner.py"
        spec = importlib.util.spec_from_file_location("bertopic_runner", bertopic_runner_path)
        bertopic_runner = importlib.util.module_from_spec(spec)
        print(f"[OPTIMIZE_CMD] Executing bertopic_runner.py...")
        spec.loader.exec_module(bertopic_runner)
        print("[OPTIMIZE_CMD] ✓ Optimization completed")
    except Exception as e:
        print(f"[OPTIMIZE_CMD] ❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("[OPTIMIZE_CMD] ========== optimize command completed ==========")


if __name__ == "__main__":
    cli()

