"""Stage 05: Retraining entrypoint."""

import sys
import click
from pathlib import Path
from datetime import datetime
from src.common.config import load_config, resolve_path
from src.common.logging import setup_logging
from src.stage05_retraining.pareto_loader import load_top_models
from src.stage05_retraining.retrain_models import retrain_single_model


class Tee:
    """Write to both file and stdout/stderr with immediate flushing."""
    def __init__(self, file_path: Path, stream):
        # Open file in line-buffered mode for immediate flushing
        self.file = open(file_path, 'w', encoding='utf-8', buffering=1)  # Line buffering
        self.stream = stream
    
    def write(self, data):
        self.file.write(data)
        self.stream.write(data)
        # Force immediate flush for both
        self.file.flush()
        self.stream.flush()
    
    def flush(self):
        self.file.flush()
        self.stream.flush()
    
    def close(self):
        self.flush()
        self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


@click.group()
def cli():
    """Stage 05: Retrain top Pareto-efficient models."""
    pass


@cli.command()
@click.option(
    "--pareto_csv",
    type=click.Path(exists=True, path_type=Path),
    default="results/pareto/pareto.csv",
    help="Path to Pareto CSV file"
)
@click.option(
    "--top_n",
    type=int,
    default=4,
    help="Number of top models to retrain (default: 4)"
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default="configs/paths.yaml",
    help="Path to paths configuration file"
)
@click.option(
    "--output_dir",
    type=click.Path(path_type=Path),
    default="models/retrained/",
    help="Base output directory for retrained models"
)
@click.option(
    "--dataset_csv",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Override dataset CSV path (optional)"
)
def retrain(pareto_csv: Path, top_n: int, config: Path, output_dir: Path, dataset_csv: Path):
    """Retrain top N Pareto-efficient models from CSV."""
    # Load configuration first to get logs directory
    paths_cfg = load_config(config)
    outputs = paths_cfg.get("outputs", {})
    logs_dir = resolve_path(Path(outputs.get("logs", "logs")))
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"stage05_retraining_{timestamp}.log"
    logger = setup_logging(logs_dir, log_file=log_file)
    logger.info("=" * 80)
    logger.info("Stage 05: Retrain Top Pareto-Efficient Models")
    logger.info("=" * 80)
    
    # Set up Tee to capture all print output to log file
    log_path = logs_dir / log_file
    stdout_tee = Tee(log_path, sys.stdout)
    stderr_tee = Tee(log_path, sys.stderr)
    
    try:
        # Set unbuffered mode for immediate output
        import os
        os.environ['PYTHONUNBUFFERED'] = '1'
        
        # Redirect stdout and stderr to Tee
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = stdout_tee
        sys.stderr = stderr_tee
        
        # Force immediate flush
        sys.stdout.flush()
        sys.stderr.flush()
        
        logger.info(f"[RETRAIN_CMD] Log file: {log_path}")
        logger.info(f"[RETRAIN_CMD] ========== Starting retrain command ==========")
        print("[RETRAIN_CMD] ========== Starting retrain command ==========")
        print("=" * 80)
        print("Stage 05: Retrain Top Pareto-Efficient Models")
        print("=" * 80)
        print(f"[RETRAIN_CMD] Arguments:")
        print(f"[RETRAIN_CMD]   Pareto CSV: {pareto_csv}")
        print(f"[RETRAIN_CMD]   Top N models: {top_n}")
        print(f"[RETRAIN_CMD]   Config: {config}")
        print(f"[RETRAIN_CMD]   Output directory: {output_dir}")
        if dataset_csv:
            print(f"[RETRAIN_CMD]   Dataset Override: {dataset_csv}")
        
        # Configuration already loaded for logs setup
        print(f"\n[RETRAIN_CMD] Configuration:")
        print(f"[RETRAIN_CMD] ✓ Config loaded: {list(paths_cfg.keys()) if isinstance(paths_cfg, dict) else type(paths_cfg)}")
    
        # Get inputs from config (always needed for OCTIS path)
        inputs = paths_cfg.get("inputs", {})
        
        # Get dataset path
        if dataset_csv:
            dataset_path = resolve_path(dataset_csv)
            print(f"[RETRAIN_CMD]   Dataset CSV resolved (from override): {dataset_path}")
        else:
            dataset_path_str = inputs.get("chapters_csv", "data/processed/chapters.csv")
            dataset_path = resolve_path(Path(dataset_path_str))
            print(f"[RETRAIN_CMD]   Dataset CSV resolved (from config): {dataset_path}")
        
        # Get OCTIS dataset path
        octis_dataset_path = inputs.get("octis_dataset", "data/interim/octis")
        octis_dataset_path = resolve_path(Path(octis_dataset_path))
        print(f"[RETRAIN_CMD]   OCTIS dataset path resolved: {octis_dataset_path}")
        
        # Resolve output directory
        output_dir = resolve_path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[RETRAIN_CMD]   Output directory resolved: {output_dir}")
        
        # Input/Output banner
        print("\n" + "=" * 80)
        print("Inputs → Outputs:")
        print("  Inputs:")
        print(f"    - pareto_csv: {pareto_csv}")
        print(f"    - dataset_csv: {dataset_path}")
        print(f"    - octis_dataset: {octis_dataset_path}")
        print("  Outputs:")
        print(f"    - retrained_models: {output_dir}")
        print("=" * 80)
        
        # Load top models from Pareto CSV
        print(f"\n[RETRAIN_CMD] Loading top {top_n} models from Pareto CSV...")
        model_configs = load_top_models(pareto_csv, top_n=top_n)
        print(f"[RETRAIN_CMD] ✓ Loaded {len(model_configs)} model configurations")
        
        # Retrain each model
        print(f"\n[RETRAIN_CMD] ========== Starting Retraining Process ==========")
        print(f"[RETRAIN_CMD] Total models to retrain: {len(model_configs)}")
        results = []
        
        for idx, model_config in enumerate(model_configs, 1):
            print(f"\n[RETRAIN_CMD] {'='*80}")
            print(f"[RETRAIN_CMD] Processing model {idx}/{len(model_configs)}")
            print(f"[RETRAIN_CMD] {'='*80}")
            success = retrain_single_model(
                model_config=model_config,
                dataset_path=dataset_path,
                octis_dataset_path=octis_dataset_path,
                output_dir=output_dir
            )
            results.append({
                'model': model_config['embedding_model'],
                'pareto_rank': model_config['pareto_rank'],
                'success': success
            })
            if success:
                print(f"[RETRAIN_CMD] ✅ Model {idx}/{len(model_configs)} completed successfully")
            else:
                print(f"[RETRAIN_CMD] ❌ Model {idx}/{len(model_configs)} failed")
        
        # Summary
        print("\n" + "=" * 80)
        print("[RETRAIN_CMD] Retraining Summary")
        print("=" * 80)
        
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        print(f"📊 Total models: {len(results)}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        
        if successful > 0:
            print(f"\n✅ Successfully retrained models:")
            for r in results:
                if r['success']:
                    print(f"   - {r['model']} (rank {r['pareto_rank']})")
        
        if failed > 0:
            print(f"\n❌ Failed models:")
            for r in results:
                if not r['success']:
                    print(f"   - {r['model']} (rank {r['pareto_rank']})")
        
        print(f"\n📁 Output directory: {output_dir}")
        print(f"📄 Log file: {log_path}")
        print("[RETRAIN_CMD] ========== retrain command completed ==========")
        logger.info(f"[RETRAIN_CMD] ========== retrain command completed ==========")
        logger.info(f"[RETRAIN_CMD] Log file saved to: {log_path}")
    
    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        stdout_tee.close()
        stderr_tee.close()


if __name__ == "__main__":
    cli()

