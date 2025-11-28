# Testing Checklist: Stages 01-05

Quick reference for what files you need to test each stage.

## ✅ Files You Already Have

### Data Files
- [x] `data/processed/chapters.csv` - Main corpus (707K sentences)
- [x] `data/processed/goodreads.csv` - Ratings data
- [x] `data/processed/custom_stoplist.txt` - Stopwords
- [x] `data/processed/chapters_subset_10000.csv` - Test subset

### Results Files
- [x] `results/experiments/model_evaluation_results.csv` - All 300+ trial results
- [x] `results/experiments/<model>/<study>/result.json` - Detailed logs
- [x] `results/topics/by_book.csv` - Topic probabilities
- [x] `results/pareto/pareto.csv` - Pareto-efficient models

### Configuration Files
- [x] `configs/paths.yaml`
- [x] `configs/bertopic.yaml`
- [x] `configs/octis.yaml`
- [x] `configs/selection.yaml`

---

## ⚠️ Files You May Need to Generate

### OCTIS Dataset (Only if using `bertopic_runner.py`)
- [ ] `data/interim/octis/corpus.tsv` - TSV format for OCTIS
  - **How to create**: Run `bertopic_runner.py` (auto-generates) OR extract logic from lines 194-214
  - **Format**: `sentence<TAB>train<TAB>author,book_title`
  - **Source**: Generated from `chapters.csv`

### Embeddings Cache (Only if using OCTIS optimization)
- [ ] `data/interim/octis/optimization_results/<model>/embeddings.npy`
  - **How to create**: Automatically generated during training
  - **Time**: Slow to regenerate (hours per model)
  - **Note**: Not needed for `retrain_from_tables.py`

---

## 📋 Stage-by-Stage Testing Requirements

### Stage 01: Ingestion
**Status**: Stub (implementation pending)

**Can Test?**: ❌ No - needs implementation
**Skip?**: ✅ Yes - you have `chapters.csv` already

**Required Files**:
- Raw EPUB files (if starting from scratch)
- BookNLP outputs (if processing)

---

### Stage 02: Preprocessing  
**Status**: Stub (implementation pending)

**Can Test?**: ❌ No - needs implementation
**Skip?**: ✅ Yes - you have processed data already

**Required Files**:
- `chapters.csv` (input) ✅
- Preprocessing implementation (missing)

---

### Stage 03: Modeling
**Status**: ✅ Implemented (two paths)

#### Path A: OCTIS Integration (`bertopic_runner.py`)
**Can Test?**: ⚠️ Partial - needs OCTIS dataset
**Required Files**:
- `chapters.csv` ✅
- `data/interim/octis/corpus.tsv` ⚠️ (can generate)
- Embeddings cache ⚠️ (can regenerate, slow)

#### Path B: Direct BERTopic (`retrain_from_tables.py`) ⭐ **RECOMMENDED**
**Can Test?**: ✅ Yes - no OCTIS needed
**Required Files**:
- `chapters.csv` ✅
- Hyperparameters from `model_evaluation_results.csv` ✅

**Command**:
```bash
python -m src.stage03_modeling.main retrain \
  --dataset_csv data/processed/chapters.csv \
  --out_dir models/retrained/ \
  --text_column Sentence
```

---

### Stage 04: Experiments
**Status**: ✅ Implemented

**Can Test?**: ⚠️ Yes, but skip if you have results
**Skip?**: ✅ Yes - you have `model_evaluation_results.csv`

**Required Files** (if running new experiments):
- `data/interim/octis/corpus.tsv` ⚠️
- Embeddings cache ⚠️
- OCTIS framework installed

**Note**: You already have results from 300+ models, so no need to rerun.

---

### Stage 05: Selection
**Status**: Stub (implementation pending)

**Can Test?**: ❌ No - needs implementation
**Required Files**:
- `results/experiments/model_evaluation_results.csv` ✅
- Pareto calculation implementation (missing)

**What to Implement**:
1. Read `model_evaluation_results.csv`
2. Filter by `min_nr_topics >= 200` (from `configs/selection.yaml`)
3. Calculate Pareto efficiency
4. Write `results/pareto/pareto.csv`

---

## 🔍 Files to Check in Archives

Look for these in your archives:

1. **OCTIS Dataset Files**:
   - `corpus.tsv`
   - `vocabulary.txt`
   - `metadata.json`

2. **Pre-computed Embeddings**:
   - `*.npy` files (embeddings cache)
   - Location: `data/interim/octis/optimization_results/`

3. **Legacy Scripts**:
   - `archive/scripts/restart_script.py` - Crash-safe launcher
   - Original experiment notebooks/scripts

4. **Original Experiment Data**:
   - Different hyperparameter sets
   - Additional embedding models tested

---

## ✅ Recommended Testing Path

### Minimal Testing (No OCTIS)

1. **Skip Stages 01-02**: Use existing processed data ✅
2. **Test Stage 03**: Use `retrain_from_tables.py` (no OCTIS) ✅
3. **Skip Stage 04**: Use existing results ✅
4. **Implement Stage 05**: Pareto selection (needs implementation)

### Full Testing (With OCTIS)

1. **Generate OCTIS dataset**: Extract logic from `bertopic_runner.py` lines 194-214
2. **Test Stage 03**: Use `bertopic_runner.py` (requires OCTIS dataset)
3. **Test Stage 04**: Run optimization (time-consuming, 300+ models)
4. **Implement Stage 05**: Pareto selection

---

## 🚀 Quick Start Commands

### Test Retraining (No OCTIS)
```bash
# Stage 03: Retrain specific models
python -m src.stage03_modeling.main retrain \
  --dataset_csv data/processed/chapters.csv \
  --out_dir models/retrained/ \
  --text_column Sentence
```

### Generate OCTIS Dataset (If Needed)
```python
# Extract from bertopic_runner.py lines 194-214
# Creates data/interim/octis/corpus.tsv
```

### Check What You Have
```bash
# Verify processed data
ls -lh data/processed/*.csv

# Verify experiment results  
ls -lh results/experiments/model_evaluation_results.csv
head results/experiments/model_evaluation_results.csv

# Check for OCTIS dataset
ls -la data/interim/octis/ 2>/dev/null || echo "OCTIS dataset not found"
```

---

*Last updated: 2025-01-XX*

