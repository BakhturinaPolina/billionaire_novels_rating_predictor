# Repository Restructure Migration Plan

## Overview
Refactoring to stage-first layout: `/src/stage01_ingestion` through `/src/stage07_analysis`, with `/src/common` for shared utilities.

## File Moves (git mv commands)

### Stage 01: Ingestion
```bash
# Currently empty, but structure exists
mkdir -p src/stage01_ingestion
git mv src/pipeline/ingestion/* src/stage01_ingestion/ 2>/dev/null || true
```

### Stage 02: Preprocessing
```bash
mkdir -p src/stage02_preprocessing
git mv src/pipeline/preprocessing/* src/stage02_preprocessing/ 2>/dev/null || true
```

### Stage 03: Modeling
```bash
mkdir -p src/stage03_modeling
git mv src/pipeline/modeling/bertopic_runner.py src/stage03_modeling/
git mv src/pipeline/modeling/retrain_from_tables.py src/stage03_modeling/
git mv src/pipeline/modeling/__init__.py src/stage03_modeling/
```

### Stage 04: Experiments
```bash
mkdir -p src/stage04_experiments
git mv src/pipeline/experiments/hparam_search.py src/stage04_experiments/
git mv src/pipeline/experiments/__init__.py src/stage04_experiments/
```

### Stage 05: Selection
```bash
mkdir -p src/stage05_selection
git mv src/pipeline/selection/* src/stage05_selection/ 2>/dev/null || true
```

### Stage 06: Labeling
```bash
mkdir -p src/stage06_labeling
git mv src/pipeline/labeling/composites.py src/stage06_labeling/
git mv src/pipeline/labeling/__init__.py src/stage06_labeling/
```

### Stage 07: Analysis
```bash
mkdir -p src/stage07_analysis
git mv src/pipeline/analysis/scoring_and_strata.py src/stage07_analysis/
git mv src/pipeline/analysis/bh_fdr.py src/stage07_analysis/
git mv src/pipeline/analysis/03_interaction_plots_PD.py src/stage07_analysis/
git mv src/pipeline/analysis/04_star_charts_and_pdf.py src/stage07_analysis/
git mv src/pipeline/analysis/__init__.py src/stage07_analysis/
```

### Common Utilities
```bash
mkdir -p src/common
# Move utils/utils/* to common/
git mv src/pipeline/utils/utils/training_utils.py src/common/
git mv src/pipeline/utils/utils/check_gpu_setup.py src/common/
git mv src/pipeline/utils/utils/hw_utils.py src/common/
git mv src/pipeline/utils/utils/thermal_monitor.py src/common/
# Remove empty utils directories
rmdir src/pipeline/utils/utils 2>/dev/null || true
rmdir src/pipeline/utils 2>/dev/null || true
```

### Notebooks Reorganization
```bash
# Stage 01: Ingestion (already 02_ingestion, rename to 01)
git mv notebooks/02_ingestion notebooks/01_ingestion

# Stage 02: Preprocessing (create if needed)
mkdir -p notebooks/02_preprocessing

# Stage 03: Modeling (already 03_modeling, keep)
# No change needed

# Stage 04: Experiments (create)
mkdir -p notebooks/04_experiments

# Stage 05: Selection (move from 06_pareto)
git mv notebooks/06_pareto notebooks/05_selection

# Stage 06: Labeling (create)
mkdir -p notebooks/06_labeling

# Stage 07: Analysis (move from 09_goodreads and 10_stats)
git mv notebooks/09_goodreads notebooks/07_analysis_goodreads
git mv notebooks/10_stats notebooks/07_analysis_stats
# Or merge into single 07_analysis directory
mkdir -p notebooks/07_analysis
git mv notebooks/07_analysis_goodreads/* notebooks/07_analysis/ 2>/dev/null || true
git mv notebooks/07_analysis_stats/* notebooks/07_analysis/ 2>/dev/null || true
rmdir notebooks/07_analysis_goodreads 2>/dev/null || true
rmdir notebooks/07_analysis_stats 2>/dev/null || true
```

### Results Reorganization
```bash
# Experiments: consolidate under results/experiments/<embedding>/
# Current structure already has this, but clean up naming
# Keep: results/experiments/Billionaire_OCTIS_ALL_Models_Results_CSV_Only/
# Keep: results/experiments/model_evaluation_results.csv

# Pareto: ensure all under results/pareto/
# Already exists: results/pareto/topics/

# Topics: ensure all under results/topics/
# Already exists: results/topics/by_book.csv
# Move topic JSON files to results/topics/ if scattered
```

### Scripts
```bash
# Keep scripts/convert_topics.py as-is (utility script)
# Keep scripts/restart_script.py as-is
# Keep scripts/ml_heat_diag.sh as-is
```

### Cleanup
```bash
# Remove empty pipeline directory
rmdir src/pipeline 2>/dev/null || true
```

## Import Updates Required

### Files needing import fixes:
1. `src/stage03_modeling/retrain_from_tables.py`
   - Change: `from src.utils.training_utils import ...`
   - To: `from src.common.training_utils import ...`

2. All stage modules that import from pipeline
   - Update relative imports to new stage structure
   - Update common utility imports to `from src.common import ...`

## Config Files to Create

1. `configs/paths.yaml` - Data paths
2. `configs/bertopic.yaml` - BERTopic configuration
3. `configs/octis.yaml` - OCTIS configuration
4. `configs/optuna.yaml` - Optuna/Bayesian search configuration
5. `configs/selection.yaml` - Pareto selection constraints (min_nr_topics >= 200)
6. `configs/labeling.yaml` - Labeling configuration
7. `configs/scoring.yaml` - Scoring and strata configuration

## Entrypoints to Create

Each stage needs `main.py`:
- `src/stage01_ingestion/main.py`
- `src/stage02_preprocessing/main.py`
- `src/stage03_modeling/main.py`
- `src/stage04_experiments/main.py`
- `src/stage05_selection/main.py`
- `src/stage06_labeling/main.py`
- `src/stage07_analysis/main.py`

