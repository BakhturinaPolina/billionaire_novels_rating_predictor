# Reproducibility Analysis: Running Pipeline Without OCTIS Optimization

## Executive Summary

**Question**: Can you repeat all pipeline steps (stages 01-05) without OCTIS+BERTopic when you've already trained 300+ models?

**Answer**: **Partially Yes** - You can skip Stage 04 (OCTIS hyperparameter optimization) if you already have experiment results, but you still need certain files and can use `retrain_from_tables.py` which bypasses OCTIS entirely.

---

## Current Pipeline State

### ✅ **What You Have**

1. **Processed Data** (Stage 01-02 outputs):
   - `data/processed/chapters.csv` - 707K sentences with columns: `Author, Book Title, Chapter, Sentence`
   - `data/processed/goodreads.csv` - Book ratings and metadata
   - `data/processed/custom_stoplist.txt` - Custom stopwords

2. **Experiment Results** (Stage 04 outputs):
   - `results/experiments/model_evaluation_results.csv` - Summary of all 300+ model trials
   - `results/experiments/<embedding_model>/<study_id>/result.json` - Individual experiment logs
   - Contains: coherence scores, diversity scores, hyperparameters for each trial

3. **Topic Results** (Stage 03 outputs):
   - `results/topics/by_book.csv` - Topic probabilities per book
   - `results/topics/top_models/*.json` - Topic word lists for top models

4. **Pareto Results** (Stage 05 outputs):
   - `results/pareto/pareto.csv` - Pareto-efficient models
   - `results/pareto/topics/` - Topic JSONs for Pareto models

### ⚠️ **What's Missing or Needs Generation**

1. **OCTIS Dataset Format** (`data/interim/octis/corpus.tsv`):
   - **Status**: Can be generated from `chapters.csv`
   - **Required by**: `bertopic_runner.py` (Stage 03 with OCTIS)
   - **NOT required by**: `retrain_from_tables.py` (direct BERTopic training)

2. **Cached Embeddings** (`.npy` files):
   - **Status**: Can be regenerated (takes time but is deterministic)
   - **Location**: `data/interim/octis/optimization_results/<model>/embeddings.npy`
   - **Required by**: OCTIS optimization to avoid recomputing embeddings

---

## Stage-by-Stage Reproducibility

### **Stage 01: Ingestion** 
- **Status**: ✅ **DATA EXISTS** - `chapters.csv` already processed
- **Current**: Stub implementation, but processed data available
- **What You Have**: 
  - ✅ `chapters.csv` - 707K sentences with columns: `Author, Book Title, Chapter, Sentence`
  - ✅ `goodreads.csv` - Book ratings and metadata
  - ✅ `custom_stoplist.txt` - Custom stopwords
- **Action**: **SKIP** - Use existing processed data files

### **Stage 02: Preprocessing**
- **Status**: ✅ **DATA EXISTS** - `chapters.csv` already preprocessed
- **Current**: Stub implementation, but preprocessed data available
- **What You Have**:
  - ✅ `chapters.csv` - Already cleaned and formatted
  - ✅ `chapters_subset_10000.csv` - Test subset (10K rows)
- **Action**: **SKIP** - Use existing preprocessed data files

### **Stage 03: Modeling** 
- **Status**: ✅ **IMPLEMENTED** - Two paths available

#### Path A: OCTIS Integration (`bertopic_runner.py`)
- **Requires**: 
  - `chapters.csv` ✅ (you have)
  - OCTIS dataset format (`data/interim/octis/corpus.tsv`) ⚠️ (can generate)
  - Embeddings cache ⚠️ (can regenerate)
- **Use Case**: Full OCTIS hyperparameter optimization
- **Testing**: Use `test_octis_pipeline.py` to validate pipeline without optimization:
  ```bash
  # Test with subset first
  python -m src.stage03_modeling.test_octis_pipeline --subset
  
  # Then test with full dataset
  python -m src.stage03_modeling.test_octis_pipeline --full
  ```

#### Path B: Direct BERTopic Retraining (`retrain_from_tables.py`) ⭐ **RECOMMENDED**
- **Requires**: 
  - `chapters.csv` ✅ (you have)
  - Hyperparameter combinations from `model_evaluation_results.csv` ✅ (you have)
- **Does NOT require**: OCTIS dataset, OCTIS framework
- **Use Case**: Retrain specific models with known hyperparameters
- **Advantage**: Bypasses OCTIS entirely, uses BERTopic directly

### **Stage 04: Experiments (OCTIS Optimization)**
- **Status**: ✅ **IMPLEMENTED** but **SKIP IF YOU HAVE RESULTS**
- **Current**: `hparam_search.py` runs Bayesian optimization via OCTIS
- **What You Need**:
  - If you already have `model_evaluation_results.csv` → **SKIP THIS STAGE**
  - If you want to run new experiments → Need OCTIS dataset + embeddings

### **Stage 05: Selection (Pareto Analysis)**
- **Status**: ⚠️ **STUB** - Implementation pending
- **Current**: Only shows inputs/outputs
- **What You Need**:
  - `model_evaluation_results.csv` ✅ (you have)
  - Implementation of Pareto efficiency calculation
  - Filter by `min_nr_topics >= 200` constraint

---

## Recommended Workflow for Testing Stages 01-05

### **Scenario 1: You Have All Results, Just Want to Test Pipeline**

**Skip Stages 01-02** (data already processed):
- Use existing `chapters.csv`, `goodreads.csv`, `custom_stoplist.txt`

**Skip Stage 04** (experiments already done):
- Use existing `results/experiments/model_evaluation_results.csv`

**Test Stage 03** (retrain specific models):
```bash
# Use retrain_from_tables.py - NO OCTIS needed
python -m src.stage03_modeling.main retrain \
  --dataset_csv data/processed/chapters.csv \
  --out_dir models/retrained/ \
  --text_column Sentence
```

**Test Stage 05** (Pareto selection):
- Need to implement Pareto calculation
- Input: `results/experiments/model_evaluation_results.csv`
- Output: `results/pareto/pareto.csv`

### **Scenario 2: You Want to Regenerate Everything**

**Stage 01-02**: Need implementation or use existing processed data

**Stage 03**: Generate OCTIS dataset first:
```python
# From bertopic_runner.py lines 194-214
# Creates data/interim/octis/corpus.tsv from chapters.csv
```

**Stage 04**: Run full optimization (300+ models, takes days/weeks)

**Stage 05**: Implement Pareto selection

---

## Files Needed for Testing Stages 01-05

### **Minimum Required Files** (You Have These ✅)

1. **Input Data**:
   - `data/processed/chapters.csv` - Main text corpus
   - `data/processed/goodreads.csv` - Ratings data  
   - `data/processed/custom_stoplist.txt` - Stopwords

2. **Experiment Results**:
   - `results/experiments/model_evaluation_results.csv` - All trial results
   - `results/experiments/<model>/<study>/result.json` - Detailed logs (optional)

3. **Configuration**:
   - `configs/paths.yaml` - Path definitions
   - `configs/bertopic.yaml` - BERTopic settings
   - `configs/octis.yaml` - OCTIS settings (if using OCTIS)
   - `configs/selection.yaml` - Selection criteria

### **Files You Can Generate** (If Needed)

1. **OCTIS Dataset** (if using `bertopic_runner.py`):
   - `data/interim/octis/corpus.tsv` - Generated from `chapters.csv`
   - Format: `sentence<TAB>partition<TAB>label`
   - Can be created by running `bertopic_runner.py` (it auto-generates)

2. **Embeddings Cache** (if using OCTIS optimization):
   - `data/interim/octis/optimization_results/<model>/embeddings.npy`
   - Regenerated automatically, but time-consuming

### **Files You Might Find in Archives**

Check your archives for:
- Legacy OCTIS dataset files (`corpus.tsv`, `vocabulary.txt`)
- Pre-computed embeddings (`.npy` files)
- Original experiment scripts (may have different hyperparameter sets)
- `archive/scripts/restart_script.py` - Crash-safe launcher

---

## Key Insights

### **1. Two Training Paths Exist**

- **OCTIS Path** (`bertopic_runner.py`): Full optimization framework, requires OCTIS dataset
- **Direct Path** (`retrain_from_tables.py`): Bypasses OCTIS, uses BERTopic directly ⭐

### **2. Stage 04 Can Be Skipped**

If you have `model_evaluation_results.csv` with all 300+ trials, you don't need to rerun Stage 04. The results contain:
- Hyperparameters for each trial
- Coherence and diversity scores
- Embedding model used
- Iteration number

### **3. Stage 03 Has Two Modes**

- **Train mode** (`main.py train`): Currently a stub, would use OCTIS
- **Retrain mode** (`main.py retrain`): ✅ Fully implemented, uses `retrain_from_tables.py`, **NO OCTIS needed**

### **4. Missing Implementations**

- Stage 01-02: Stubs only (but you have processed data)
- Stage 05: Stub only (needs Pareto calculation implementation)

---

## Recommendations

### **For Testing Without OCTIS** ⭐

1. **Use `retrain_from_tables.py`** for Stage 03:
   - No OCTIS dependency
   - Direct BERTopic training
   - Uses hyperparameters from your existing results

2. **Skip Stage 04**:
   - You already have all experiment results
   - No need to rerun 300+ model optimizations

3. **Implement Stage 05**:
   - Read `model_evaluation_results.csv`
   - Filter by `min_nr_topics >= 200`
   - Calculate Pareto efficiency
   - Output `results/pareto/pareto.csv`

### **For Full Reproducibility**

1. **Implement Stage 01-02** if you need to process raw data
2. **Generate OCTIS dataset** if you want to use `bertopic_runner.py`:
   ```python
   # From bertopic_runner.py, creates corpus.tsv
   # Can be extracted into a standalone script
   ```
3. **Keep experiment results** - they're your ground truth

---

## Next Steps

1. **Verify you have**: `chapters.csv`, `model_evaluation_results.csv`
2. **Decide**: Do you want to test retraining (use `retrain_from_tables.py`) or full pipeline?
3. **Implement**: Stage 05 Pareto selection (if needed)
4. **Check archives**: Look for any missing OCTIS dataset files or embeddings

---

*Last updated: 2025-01-XX*
*Based on codebase analysis of stages 01-05*

