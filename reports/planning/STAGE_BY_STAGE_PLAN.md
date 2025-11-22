# Stage-by-Stage Implementation Plan

This document breaks down the reliability & reproducibility work **by stage**, focusing only on files relevant to each stage at a time. Files are categorized by action needed: **VERIFY**, **DECIDE**, or **IMPLEMENT**.

---

## Stage 01: Ingestion

### Purpose
Load raw texts, Goodreads data, BookNLP I/O operations

### Files to Work With

#### **Code Files** (IMPLEMENT/VERIFY)
- `src/stage01_ingestion/main.py` - **VERIFY**: Check if it properly loads configs and shows inputs/outputs
- `src/common/config.py` - **VERIFY**: Path resolution works correctly
- `src/common/seed.py` - **IMPLEMENT**: Add `set_seed()` call at start of main.py
- `src/common/logging.py` - **VERIFY**: Seed is logged from config

#### **Config Files** (DECIDE)
- `configs/paths.yaml` - **VERIFY**: All paths are relative, no hardcoded absolutes
- `configs/*.yaml` - **DECIDE**: Do we need a `seed` field in paths.yaml or separate seed config?

#### **Input Files** (VERIFY)
- `data/raw/Billionaire_Novels_EPUB/` - **VERIFY**: Structure matches what code expects
- `data/raw/Billionaire_Full_Novels_TXT/` - **VERIFY**: Files exist and are readable
- `data/interim/booknlp/` - **VERIFY**: BookNLP outputs are in expected format

#### **Output Files** (DECIDE - Real vs Example)
- `data/processed/chapters.csv` - **REAL**: Main output, must exist
- `data/processed/goodreads.csv` - **REAL**: Required for analysis
- `data/processed/custom_stoplist.txt` - **REAL**: Used in preprocessing

#### **Actions Needed**
1. **VERIFY**: Run `python -m src.stage01_ingestion.main --config configs/paths.yaml --dry-run` shows correct inputs/outputs
2. **IMPLEMENT**: Add seed setting at start of main.py
3. **VERIFY**: Check for hardcoded paths (grep for `/home/`, `C:\`, absolute paths)
4. **DECIDE**: Where should seed be configured? (paths.yaml or separate file?)

---

## Stage 02: Preprocessing

### Purpose
Text cleaning, tokenization/lemmatization, custom stoplist building

### Files to Work With

#### **Code Files** (IMPLEMENT/VERIFY)
- `src/stage02_preprocessing/main.py` - **VERIFY**: Exists and shows inputs/outputs
- `src/common/seed.py` - **IMPLEMENT**: Add `set_seed()` call at start
- `src/common/logging.py` - **VERIFY**: Seed is logged

#### **Config Files** (DECIDE)
- `configs/paths.yaml` - **VERIFY**: Preprocessing input/output paths are correct
- **DECIDE**: Do we need preprocessing-specific config (tokenization params, stoplist rules)?

#### **Input Files** (VERIFY)
- `data/processed/chapters.csv` - **REAL**: From Stage 01
- `data/processed/custom_stoplist.txt` - **REAL**: From Stage 01

#### **Output Files** (DECIDE - Real vs Example)
- `data/processed/chapters.csv` (updated) - **REAL**: Preprocessed chapters
- `data/processed/chapters_subset_10000.csv` - **DECIDE**: Is this real or example/test data?
- `data/processed/Billionaire_ALL_MERGED_TXTs_by_POS/*.txt` - **DECIDE**: Real outputs or legacy/intermediate?

#### **Data Quality Gates** (IMPLEMENT)
- **IMPLEMENT**: Pandera schema for `chapters.csv` - validate columns, types, ranges
- **IMPLEMENT**: Pandera schema for `goodreads.csv` - validate ratings, dates, etc.

#### **Actions Needed**
1. **VERIFY**: main.py exists and works
2. **IMPLEMENT**: Add seed setting
3. **IMPLEMENT**: Create Pandera schemas for data validation
4. **DECIDE**: Which output files are real project outputs vs examples/test data?

---

## Stage 03: Modeling

### Purpose
BERTopic fit/retrain, OCTIS adapter integration

### Files to Work With

#### **Code Files** (IMPLEMENT/VERIFY)
- `src/stage03_modeling/main.py` - **VERIFY**: CLI commands (train, retrain) work
- `src/stage03_modeling/bertopic_runner.py` - **VERIFY**: No hardcoded paths, uses config
- `src/stage03_modeling/retrain_from_tables.py` - **VERIFY**: Uses config paths
- `src/common/seed.py` - **IMPLEMENT**: Add `set_seed()` call

#### **Config Files** (DECIDE)
- `configs/bertopic.yaml` - **VERIFY**: All model params are here, no hardcoded values
- `configs/octis.yaml` - **VERIFY**: OCTIS adapter settings
- **DECIDE**: Should seed be in bertopic.yaml or paths.yaml?

#### **Input Files** (VERIFY)
- `data/processed/chapters.csv` - **REAL**: From Stage 02
- `data/interim/octis/` - **DECIDE**: Real dataset or example/test?

#### **Output Files** (DECIDE - Real vs Example)
- `models/` - **REAL**: Trained BERTopic models (but currently empty - is this intentional?)
- `results/topics/by_book.csv` - **REAL**: Topic probabilities per book (per PIPELINE_CONTRACTS.md)
- `results/topics/top_models/*.json` - **DECIDE**: Real outputs or examples?
- `results/topics/top_models_with_coherence/*.json` - **DECIDE**: Real or examples?
- `results/topics/top_models_no_coherence/*.json` - **DECIDE**: Real or examples?

#### **Actions Needed**
1. **VERIFY**: All paths come from config, no hardcoded values
2. **IMPLEMENT**: Add seed setting
3. **VERIFY**: Output format matches PIPELINE_CONTRACTS.md schema
4. **DECIDE**: Which topic JSON files are real outputs vs examples/mockups?

---

## Stage 04: Experiments

### Purpose
Bayesian hyperparameter search; experiment ledgers

### Files to Work With

#### **Code Files** (IMPLEMENT/VERIFY)
- `src/stage04_experiments/main.py` - **VERIFY**: Shows inputs/outputs
- `src/stage04_experiments/hparam_search.py` - **VERIFY**: Uses config, logs seed
- `src/common/seed.py` - **IMPLEMENT**: Add `set_seed()` call

#### **Config Files** (DECIDE)
- `configs/octis.yaml` - **VERIFY**: OCTIS study settings, no hardcoded values
- **DECIDE**: Where should seed be configured for experiments?

#### **Input Files** (VERIFY)
- `data/processed/chapters.csv` - **REAL**: From Stage 02
- `models/` - **REAL**: Trained models from Stage 03

#### **Output Files** (IMPLEMENT - Critical!)
- `results/experiments/model_evaluation_results.csv` - **REAL**: **MUST IMPLEMENT** unified ledger schema
- `results/experiments/<embedding_model>/<study_id>/result.json` - **REAL**: Individual experiment logs
- `scripts/normalize_ledger.py` - **IMPLEMENT**: Script to normalize experiment logs into unified CSV

#### **Unified Ledger Schema** (IMPLEMENT)
- **IMPLEMENT**: Define `ledger_schema.py` with expected columns:
  - `study_id`, `embedding_model`, `trial_id`, `nr_topics`, `coherence_cv`, `topic_diversity`, `timestamp`, `config_hash`
- **IMPLEMENT**: Stage 04 writes directly in unified schema OR
- **IMPLEMENT**: `scripts/normalize_ledger.py` converts existing logs to unified format

#### **Actions Needed**
1. **IMPLEMENT**: Unified ledger schema definition
2. **IMPLEMENT**: Either modify Stage 04 to write unified CSV directly, OR create normalizer script
3. **VERIFY**: Output matches PIPELINE_CONTRACTS.md schema
4. **IMPLEMENT**: Add seed setting and logging

---

## Stage 05: Selection

### Purpose
Pareto efficiency analysis + constraints (nr_topics >= 200)

### Files to Work With

#### **Code Files** (IMPLEMENT/VERIFY)
- `src/stage05_selection/main.py` - **VERIFY**: Exists and shows inputs/outputs
- **IMPLEMENT**: Filter by `min_nr_topics >= 200` BEFORE Pareto calculation
- `src/common/seed.py` - **IMPLEMENT**: Add `set_seed()` call

#### **Config Files** (DECIDE)
- `configs/selection.yaml` - **VERIFY**: `min_nr_topics: 200` constraint is defined
- **DECIDE**: Should constraint be hardcoded or configurable?

#### **Input Files** (VERIFY)
- `results/experiments/model_evaluation_results.csv` - **REAL**: From Stage 04 (unified ledger)

#### **Output Files** (IMPLEMENT - Critical!)
- `results/pareto/pareto.csv` - **REAL**: **MUST IMPLEMENT** with constraint applied
- `results/pareto/BEST_MODEL.md` - **DECIDE**: Real output or example? Should this exist?

#### **Constraint Enforcement** (IMPLEMENT)
- **IMPLEMENT**: Filter models where `nr_topics < min_nr_topics` BEFORE Pareto calculation
- **VERIFY**: All rows in `pareto.csv` have `nr_topics >= 200`
- **IMPLEMENT**: Test to verify constraint is applied

#### **Actions Needed**
1. **IMPLEMENT**: Pre-filter by min_nr_topics before Pareto
2. **IMPLEMENT**: Write `pareto.csv` with required columns (per PIPELINE_CONTRACTS.md)
3. **DECIDE**: Do we need BEST_MODEL.md or is pareto.csv enough?
4. **VERIFY**: Output schema matches PIPELINE_CONTRACTS.md

---

## Stage 06: Labeling

### Purpose
Semi-supervised topic labeling; composite building

### Files to Work With

#### **Code Files** (IMPLEMENT/VERIFY)
- `src/stage06_labeling/main.py` - **VERIFY**: Shows inputs/outputs
- `src/stage06_labeling/composites.py` - **VERIFY**: Uses config paths
- `src/common/seed.py` - **IMPLEMENT**: Add `set_seed()` call

#### **Config Files** (DECIDE)
- `configs/labeling.yaml` - **VERIFY**: Labeling rules, seed terms, mappings
- **DECIDE**: Should labeling seeds/mappings be in config or hardcoded?

#### **Input Files** (VERIFY)
- `results/pareto/pareto.csv` - **REAL**: From Stage 05
- `results/topics/top_models/*.json` - **REAL**: Topic files from Stage 03

#### **Output Files** (IMPLEMENT)
- `results/topics/topic_labels.parquet` - **REAL**: **MUST IMPLEMENT** with columns:
  - `topic_id`, `label`, `confidence`, `seed_terms`
- `results/topics/labeling_report.md` - **DECIDE**: Real output or example? Should this exist?

#### **Labeling Consistency** (IMPLEMENT)
- **IMPLEMENT**: Config-based seed terms and mappings
- **IMPLEMENT**: Deterministic labeling (same inputs → same labels)
- **VERIFY**: Output format matches acceptance checks

#### **Actions Needed**
1. **IMPLEMENT**: Output `topic_labels.parquet` with required schema
2. **DECIDE**: Do we need labeling_report.md or is parquet enough?
3. **IMPLEMENT**: Config-based labeling rules
4. **VERIFY**: Labeling is deterministic (same seed → same labels)

---

## Stage 07: Analysis

### Purpose
Goodreads integration, statistical analysis, FDR correction, findings

### Files to Work With

#### **Code Files** (IMPLEMENT/VERIFY)
- `src/stage07_analysis/main.py` - **VERIFY**: Shows inputs/outputs
- `src/stage07_analysis/scoring_and_strata.py` - **VERIFY**: Uses config paths
- `src/stage07_analysis/bh_fdr.py` - **VERIFY**: FDR correction logic
- `src/stage07_analysis/03_interaction_plots_PD.py` - **VERIFY**: Plot generation
- `src/stage07_analysis/04_star_charts_and_pdf.py` - **VERIFY**: Chart generation
- `src/common/seed.py` - **IMPLEMENT**: Add `set_seed()` call

#### **Config Files** (DECIDE)
- `configs/scoring.yaml` - **VERIFY**: Scoring parameters, thresholds
- **DECIDE**: Are all analysis parameters in config or some hardcoded?

#### **Input Files** (VERIFY)
- `data/processed/goodreads.csv` - **REAL**: From Stage 01
- `results/pareto/pareto.csv` - **REAL**: From Stage 05
- `results/topics/topic_labels.parquet` - **REAL**: From Stage 06
- `results/topics/by_book.csv` - **REAL**: From Stage 03

#### **Output Files** (IMPLEMENT - Critical!)
- `data/processed/chapters_quartiled.parquet` - **REAL**: **MUST IMPLEMENT** quartile binning
  - Required columns: `quartile` (Q1-Q4), `normalized_position`
- `results/analysis/reproducible_scripts_topics_vs_readers_appreciation/*` - **DECIDE**: Real outputs or examples?
- `results/figures/*` - **DECIDE**: Real outputs or examples?

#### **Quartile Binning** (IMPLEMENT)
- **IMPLEMENT**: `src/stage07_analysis/quartile_binning.py` (new file)
- **IMPLEMENT**: Group chapters into quartiles (Q1-Q4) based on some metric
- **DECIDE**: What metric determines quartiles? (ratings? topic scores? something else?)
- **IMPLEMENT**: Calculate normalized positions (0.0-1.0) within quartiles

#### **Actions Needed**
1. **IMPLEMENT**: Quartile binning functionality
2. **DECIDE**: What determines quartile assignment?
3. **IMPLEMENT**: Output `chapters_quartiled.parquet` with required schema
4. **DECIDE**: Which analysis output files are real vs examples?

---

## Common Files (All Stages)

### Files to Work With

#### **Code Files** (VERIFY)
- `src/common/config.py` - **VERIFY**: `resolve_path()` works, no hardcoded paths
- `src/common/seed.py` - **VERIFY**: `set_seed()` is implemented correctly
- `src/common/logging.py` - **VERIFY**: Logs seed from config
- `src/common/io.py` - **VERIFY**: File I/O uses config paths

#### **Config Files** (DECIDE)
- `configs/paths.yaml` - **VERIFY**: All paths are relative, no hardcoded absolutes
- **DECIDE**: Where should `seed` be configured? (one global seed or per-stage?)

#### **Build Files** (IMPLEMENT)
- `Makefile` - **VERIFY**: All stage targets work
- `mcp_research_planning/Makefile.append` - **IMPLEMENT**: Add contract checks
- **IMPLEMENT**: `make contracts` target runs all checks

#### **Actions Needed**
1. **VERIFY**: No hardcoded paths anywhere (grep audit)
2. **IMPLEMENT**: Seed configuration strategy (decide where it lives)
3. **IMPLEMENT**: All stages call `set_seed()` at start
4. **IMPLEMENT**: Contract checks in Makefile

---

## Summary: Files by Action Type

### **IMPLEMENT** (Must Create/Modify)
1. `src/common/seed.py` - Add `set_seed()` function
2. All `src/stage0X/main.py` - Add seed call at start
3. `src/stage04_experiments/` - Unified ledger schema + normalizer
4. `src/stage05_selection/main.py` - Pre-filter by min_nr_topics
5. `src/stage07_analysis/quartile_binning.py` - New file for quartile binning
6. `scripts/normalize_ledger.py` - Normalize experiment logs (if needed)
7. `Makefile` - Add contract checks

### **VERIFY** (Check Existing)
1. All `main.py` files - Show inputs/outputs correctly
2. All config files - No hardcoded values
3. All code files - Use `resolve_path()`, no hardcoded paths
4. Output schemas - Match PIPELINE_CONTRACTS.md
5. Seed logging - Works in all stages

### **DECIDE** (Pending Decisions)
1. **Seed configuration**: Where should it live? (paths.yaml? separate file? per-stage?)
2. **Output files**: Which are real vs examples?
   - `chapters_subset_10000.csv` - Real or test?
   - `results/topics/top_models/*.json` - Real or examples?
   - `results/analysis/*` - Real or examples?
   - `results/figures/*` - Real or examples?
3. **Quartile metric**: What determines quartile assignment in Stage 07?
4. **Additional outputs**: Do we need BEST_MODEL.md, labeling_report.md, etc.?
5. **Preprocessing config**: Do we need separate config for tokenization params?

---

## Workflow Recommendation

1. **Start with Stage 01**: Verify inputs/outputs, add seed, check paths
2. **Move to Stage 02**: Add data validation schemas
3. **Continue sequentially**: Each stage builds on previous
4. **After all stages**: Add contract checks, CI, documentation

Focus on **one stage at a time** to avoid context switching and ensure each stage is complete before moving on.

