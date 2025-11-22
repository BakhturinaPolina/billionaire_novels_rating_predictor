"""Archive and clean results folders for Stage 04 Selection.

This script archives existing experiments and pareto results to timestamped folders,
then cleans the original folders keeping only model_evaluation_results.csv in experiments
and removing all files from pareto.
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path


def get_timestamp() -> str:
    """Generate timestamp in format YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def archive_and_clean():
    """Archive experiments and pareto folders, then clean originals."""
    # Get project root (assuming script is in src/stage04_selection/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Define paths
    experiments_dir = project_root / "results" / "experiments"
    pareto_dir = project_root / "results" / "pareto"
    archive_dir = project_root / "archive"
    
    # CSV file to preserve
    csv_file = experiments_dir / "model_evaluation_results.csv"
    
    # Generate timestamp
    timestamp = get_timestamp()
    experiments_archive = archive_dir / f"experiments_{timestamp}"
    pareto_archive = archive_dir / f"pareto_{timestamp}"
    
    print("=" * 80)
    print("Archive and Clean Results Folders")
    print("=" * 80)
    print(f"Timestamp: {timestamp}")
    print()
    
    # Verify source directories exist
    if not experiments_dir.exists():
        print(f"ERROR: Experiments directory not found: {experiments_dir}")
        sys.exit(1)
    
    if not pareto_dir.exists():
        print(f"ERROR: Pareto directory not found: {pareto_dir}")
        sys.exit(1)
    
    # Verify CSV file exists
    if not csv_file.exists():
        print(f"ERROR: Required CSV file not found: {csv_file}")
        sys.exit(1)
    
    # Create archive directory if it doesn't exist
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Archive experiments folder
    print(f"Step 1: Archiving experiments folder...")
    print(f"  Source: {experiments_dir}")
    print(f"  Destination: {experiments_archive}")
    
    try:
        shutil.copytree(experiments_dir, experiments_archive)
        print(f"  ✓ Successfully archived experiments folder")
    except Exception as e:
        print(f"  ✗ ERROR archiving experiments: {e}")
        sys.exit(1)
    
    # Step 2: Archive pareto folder
    print(f"\nStep 2: Archiving pareto folder...")
    print(f"  Source: {pareto_dir}")
    print(f"  Destination: {pareto_archive}")
    
    try:
        shutil.copytree(pareto_dir, pareto_archive)
        print(f"  ✓ Successfully archived pareto folder")
    except Exception as e:
        print(f"  ✗ ERROR archiving pareto: {e}")
        sys.exit(1)
    
    # Step 3: Clean experiments folder (keep only CSV)
    print(f"\nStep 3: Cleaning experiments folder...")
    print(f"  Keeping: {csv_file}")
    
    try:
        # Remove all items except the CSV file
        for item in experiments_dir.iterdir():
            if item.name == "model_evaluation_results.csv":
                continue
            if item.is_dir():
                shutil.rmtree(item)
                print(f"  ✓ Removed directory: {item.name}")
            else:
                item.unlink()
                print(f"  ✓ Removed file: {item.name}")
        
        print(f"  ✓ Successfully cleaned experiments folder")
    except Exception as e:
        print(f"  ✗ ERROR cleaning experiments: {e}")
        sys.exit(1)
    
    # Step 4: Clean pareto folder (remove everything)
    print(f"\nStep 4: Cleaning pareto folder...")
    
    try:
        # Remove all contents
        for item in pareto_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
                print(f"  ✓ Removed directory: {item.name}")
            else:
                item.unlink()
                print(f"  ✓ Removed file: {item.name}")
        
        print(f"  ✓ Successfully cleaned pareto folder")
    except Exception as e:
        print(f"  ✗ ERROR cleaning pareto: {e}")
        sys.exit(1)
    
    # Verify final state
    print(f"\nStep 5: Verifying final state...")
    
    experiments_files = list(experiments_dir.iterdir())
    if len(experiments_files) == 1 and experiments_files[0].name == "model_evaluation_results.csv":
        print(f"  ✓ Experiments folder contains only model_evaluation_results.csv")
    else:
        print(f"  ✗ WARNING: Experiments folder contains unexpected files: {[f.name for f in experiments_files]}")
    
    pareto_files = list(pareto_dir.iterdir())
    if len(pareto_files) == 0:
        print(f"  ✓ Pareto folder is empty")
    else:
        print(f"  ✗ WARNING: Pareto folder is not empty: {[f.name for f in pareto_files]}")
    
    print()
    print("=" * 80)
    print("Archive and Clean Complete")
    print("=" * 80)
    print(f"Archived experiments to: {experiments_archive}")
    print(f"Archived pareto to: {pareto_archive}")
    print(f"Kept CSV file: {csv_file}")
    print()


if __name__ == "__main__":
    archive_and_clean()

