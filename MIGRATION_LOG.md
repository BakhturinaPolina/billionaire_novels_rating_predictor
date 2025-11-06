# Migration Log: Stage-First Repository Restructure

**Date:** 2025-01-XX  
**Branch:** refactor/structure-by-research-part  
**Session ID:** qual_1762465746861_1b4zclv5dzu

## Overview

Refactored repository from `src/pipeline/*` structure to stage-first layout:
- `/src/stage01_ingestion` through `/src/stage07_analysis`
- `/src/common` for shared utilities
- Stage-aligned notebook organization (`/notebooks/01_ingestion` through `/notebooks/07_analysis`)
- Centralized configuration in `/configs/*.yaml`

## File Moves (git mv)

### Source Code Restructure

#### Stage 01: Ingestion
- `src/pipeline/ingestion/__init__.py` → `src/stage01_ingestion/__init__.py`

#### Stage 02: Preprocessing
- `src/pipeline/preprocessing/__init__.py` → `src/stage02_preprocessing/__init__.py`

#### Stage 03: Modeling
- `src/pipeline/modeling/bertopic_runner.py` → `src/stage03_modeling/bertopic_runner.py`
- `src/pipeline/modeling/retrain_from_tables.py` → `src/stage03_modeling/retrain_from_tables.py`
- `src/pipeline/modeling/__init__.py` → `src/stage03_modeling/__init__.py`

#### Stage 04: Experiments
- `src/pipeline/experiments/hparam_search.py` → `src/stage04_experiments/hparam_search.py`
- `src/pipeline/experiments/__init__.py` → `src/stage04_experiments/__init__.py`

#### Stage 05: Selection
- `src/pipeline/selection/__init__.py` → `src/stage05_selection/__init__.py`

#### Stage 06: Labeling
- `src/pipeline/labeling/composites.py` → `src/stage06_labeling/composites.py`
- `src/pipeline/labeling/__init__.py` → `src/stage06_labeling/__init__.py`

#### Stage 07: Analysis
- `src/pipeline/analysis/scoring_and_strata.py` → `src/stage07_analysis/scoring_and_strata.py`
- `src/pipeline/analysis/bh_fdr.py` → `src/stage07_analysis/bh_fdr.py`
- `src/pipeline/analysis/03_interaction_plots_PD.py` → `src/stage07_analysis/03_interaction_plots_PD.py`
- `src/pipeline/analysis/04_star_charts_and_pdf.py` → `src/stage07_analysis/04_star_charts_and_pdf.py`
- `src/pipeline/analysis/__init__.py` → `src/stage07_analysis/__init__.py`

#### Common Utilities
- `src/pipeline/utils/utils/training_utils.py` → `src/common/training_utils.py`
- `src/pipeline/utils/utils/check_gpu_setup.py` → `src/common/check_gpu_setup.py`
- `src/pipeline/utils/utils/hw_utils.py` → `src/common/hw_utils.py`
- `src/pipeline/utils/utils/thermal_monitor.py` → `src/common/thermal_monitor.py`

### Notebook Reorganization

- `notebooks/02_ingestion/` → `notebooks/01_ingestion/`
- `notebooks/06_pareto/` → `notebooks/05_selection/`
- `notebooks/09_goodreads/*` → `notebooks/07_analysis/goodreads dataset EDA and score adjustment/`
- `notebooks/10_stats/*` → `notebooks/07_analysis/` (merged with goodreads)

### New Files Created

#### Configuration Files
- `configs/paths.yaml` - Data paths configuration
- `configs/bertopic.yaml` - BERTopic model configuration
- `configs/octis.yaml` - OCTIS framework configuration
- `configs/optuna.yaml` - Optuna/Bayesian optimization configuration
- `configs/selection.yaml` - Pareto selection constraints (min_nr_topics >= 200)
- `configs/labeling.yaml` - Topic labeling configuration
- `configs/scoring.yaml` - Scoring and statistical analysis configuration

#### Common Utilities
- `src/common/__init__.py` - Common package exports
- `src/common/config.py` - Configuration loading utilities
- `src/common/io.py` - I/O utilities (CSV, JSON, pickle, parquet)
- `src/common/logging.py` - Logging setup utilities
- `src/common/metrics.py` - Metrics computation utilities
- `src/common/seed.py` - Random seed utilities

#### Stage Entrypoints
- `src/stage01_ingestion/main.py` - Ingestion CLI entrypoint
- `src/stage02_preprocessing/main.py` - Preprocessing CLI entrypoint
- `src/stage03_modeling/main.py` - Modeling CLI entrypoint (train, retrain)
- `src/stage04_experiments/main.py` - Experiments CLI entrypoint
- `src/stage05_selection/main.py` - Selection CLI entrypoint
- `src/stage06_labeling/main.py` - Labeling CLI entrypoint
- `src/stage07_analysis/main.py` - Analysis CLI entrypoint

#### Build System
- `Makefile` - Stage-aware build targets

#### Documentation
- `MIGRATION_PLAN.md` - Migration plan document
- `MIGRATION_LOG.md` - This file

## Import Updates

### Fixed Imports
- `src/stage03_modeling/retrain_from_tables.py`:
  - Changed: `from src.utils.training_utils import ...`
  - To: `from src.common.training_utils import ...`

### Import Patterns
All stage entrypoints use:
```python
from src.common.config import load_config
```

All modules importing shared utilities use:
```python
from src.common.<module> import <functions>
```

## Validation

### Import Tests
- ✓ `src.common.config` imports successfully
- ✓ `src.stage01_ingestion.main` imports successfully
- ✓ `src.stage03_modeling.retrain_from_tables` imports successfully

### Entrypoint Tests
- ✓ `python -m src.stage01_ingestion.main --config configs/paths.yaml` runs
- ✓ `python -m src.stage05_selection.main --config configs/selection.yaml` runs
- ✓ Config loading works correctly
- ✓ Constraint enforcement (min_nr_topics >= 200) verified

## Remaining Tasks

### Code Updates Needed
1. Update any remaining hardcoded paths to use configs
2. Implement full logic in stage entrypoints (currently stubs)
3. Update any notebook imports that reference old paths
4. Update scripts that reference old module paths

### Results Directory
- Results directory structure already follows target layout
- No moves needed for `/results/experiments/`, `/results/pareto/`, `/results/topics/`

## Notes

- All moves performed with `git mv` to preserve history
- Empty directories removed after moves
- Config files use relative paths (resolved from project root)
- Stage entrypoints are thin CLI wrappers (full implementation pending)
- Makefile provides convenient stage execution targets

## Quick Wins Applied

1. ✓ Collapsed utils into `/src/common`
2. ✓ Moved Pareto notebooks to `/notebooks/05_selection`
3. ✓ Unified experiment structure (already organized by embedding)
4. ✓ Topics consolidated in `/results/topics`
5. ✓ Populated `/configs` with YAML stubs

## Next Steps

1. Implement full stage logic in entrypoints
2. Update any remaining hardcoded paths
3. Test full pipeline end-to-end
4. Update documentation (REPOSITORY_STRUCTURE.md)
5. Commit changes with descriptive message

---

## Refactor: Results Consolidation & Notebook Cleanup (2025-01-XX)

### Goals
- Consolidate experiment results into canonical structure: `results/experiments/<embedding>/<study_id>/`
- Unify all topic JSONs under `results/topics/` with clear subfolders
- Move duplicate notebooks to `notebooks/_legacy/`
- Add input/output banners to all stage entrypoints
- Enforce selection constraints (min_nr_topics >= 200)

### File Moves (git mv)

#### Experiment Results Consolidation
- `results/experiments/Billionaire_OCTIS_ALL_Models_Results_CSV_Only/all-MiniLM-L12-v2` → `results/experiments/all-MiniLM-L12-v2/study_001`
- `results/experiments/Billionaire_OCTIS_ALL_Models_Results_CSV_Only/multi-qa-mpnet-base-cos-v1` → `results/experiments/multi-qa-mpnet-base-cos-v1/study_001`
- `results/experiments/Billionaire_OCTIS_ALL_Models_Results_CSV_Only/paraphrase-distilroberta-base-v1` → `results/experiments/paraphrase-distilroberta-base-v1/study_001`
- `results/experiments/Billionaire_OCTIS_ALL_Models_Results_CSV_Only/paraphrase-MiniLM-L6-v2` → `results/experiments/paraphrase-MiniLM-L6-v2/study_001`
- `results/experiments/Billionaire_OCTIS_ALL_Models_Results_CSV_Only/paraphrase-mpnet-base-v2` → `results/experiments/paraphrase-mpnet-base-v2/study_001`
- `results/experiments/Billionaire_OCTIS_ALL_Models_Results_CSV_Only/whaleloops-phrase-bert` → `results/experiments/whaleloops-phrase-bert/study_001`
- `results/experiments/Billionaire_OCTIS_Two_Pretrained_Models_CSV_With_Evaluation_Results/optimization_results/all-MiniLM-L12-v2` → `results/experiments/all-MiniLM-L12-v2/study_002`

#### Topics Consolidation
- `results/topics/Bilionaires_JSON_Top_Models_Topics_with_Coherence_Scores` → `results/topics/top_models_with_coherence`
- `results/topics/Bilionaires_JSON_Topics_NO_Coherence_Scores` → `results/topics/top_models_no_coherence`
- `results/topics/Billionaire_Top_Models_JSON_Topics` → `results/topics/top_models`

#### Analysis Outputs
- `results/experiments/reproducible_scripts topics vs readers appreciation` → `results/analysis/reproducible_scripts_topics_vs_readers_appreciation`

#### Duplicate Notebooks → Legacy
- `notebooks/03_modeling/retrain_best_models/Copy of retrain_best_topics_coherence_proprity_with_add_repr.ipynb` → `notebooks/_legacy/Copy of retrain_best_topics_coherence_proprity_with_add_repr.ipynb`
- `notebooks/05_selection/post_processing_and_choosing_best_model/Another copy of 1o__best_bert_model.ipynb` → `notebooks/_legacy/Another copy of 1o__best_bert_model.ipynb`
- `notebooks/07_analysis/correlation analysis of topics/Copy of topics_analysis.ipynb` → `notebooks/_legacy/Copy of topics_analysis.ipynb`
- `notebooks/07_analysis/goodreads dataset EDA and score adjustment/Copy of Exploratory Data Analysis (EDA) of the Goodreads Dataset.ipynb` → `notebooks/_legacy/Copy of Exploratory Data Analysis (EDA) of the Goodreads Dataset.ipynb`
- `notebooks/07_analysis/goodreads dataset EDA and score adjustment/Copy of Exploratory Data Analysis (EDA) of the Goodreads Dataset(1).ipynb` → `notebooks/_legacy/Copy of Exploratory Data Analysis (EDA) of the Goodreads Dataset(1).ipynb`
- `notebooks/07_analysis/goodreads dataset EDA and score adjustment/Copy of RIGHT Exploratory Data Analysis (EDA) of the Goodreads Dataset.ipynb` → `notebooks/_legacy/Copy of RIGHT Exploratory Data Analysis (EDA) of the Goodreads Dataset.ipynb`
- `notebooks/07_analysis/mapping back probabilities to books ans statistical analysis of probabilities distribution/Copy of books_by_probabilities_analysis.ipynb` → `notebooks/_legacy/Copy of books_by_probabilities_analysis.ipynb`

### Code Updates

#### Stage Entrypoint Banners
All stage entrypoints (`src/stage0X/main.py`) now display:
- Input files/directories (from configs)
- Output files/directories (from configs)
- Constraints (where applicable, e.g., min_nr_topics >= 200 in stage05)

#### New Files Created
- `notebooks/_legacy/README.md` - Documentation for legacy notebooks
- `MOVE_PLAN.csv` - Move plan reference (can be deleted after review)

### Validation

#### Import Tests
- ✓ All stage entrypoints import successfully
- ✓ Config loading works correctly
- ✓ Path resolution functional

#### Structure Validation
- ✓ Experiments organized by `<embedding>/<study_id>/`
- ✓ Topics consolidated under `results/topics/`
- ✓ Duplicate notebooks moved to `_legacy/`
- ✓ Analysis outputs in `results/analysis/`

### Remaining Tasks

1. Update any notebooks that reference old experiment paths
2. Update any scripts that reference old topic paths
3. Test stage entrypoint banners display correctly
4. Verify selection constraint enforcement (min_nr_topics >= 200)

