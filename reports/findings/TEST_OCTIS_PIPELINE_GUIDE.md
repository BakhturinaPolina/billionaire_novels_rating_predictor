# Testing OCTIS+BERTopic Pipeline Guide

## Overview

The `test_octis_pipeline.py` script validates the OCTIS+BERTopic pipeline **without running optimization**. It tests:
- Data loading and validation
- OCTIS dataset format creation
- Embedding generation/caching
- BERTopic model instantiation
- Single training run validation

## Quick Start

### Test with Subset (Recommended First)

```bash
# Test with 10K subset - fast validation
python -m src.stage03_modeling.test_octis_pipeline --subset
```

This will:
1. Load `chapters_subset_10000.csv`
2. Create OCTIS dataset format
3. Generate/cache embeddings
4. Test a single BERTopic training run
5. Validate outputs

**Expected time**: 5-15 minutes (depending on GPU)

### Test with Full Dataset

```bash
# Test with full dataset - comprehensive validation
python -m src.stage03_modeling.test_octis_pipeline --full
```

**Expected time**: 30-60 minutes (depending on GPU and dataset size)

## Command Options

```bash
python -m src.stage03_modeling.test_octis_pipeline [OPTIONS]

Options:
  --subset              Test with chapters_subset_10000.csv (10K rows)
  --full                Test with full chapters.csv (~707K rows)
  --embedding-model     Embedding model to use (default: all-MiniLM-L12-v2)
  --no-cache            Disable embedding cache (force regeneration)
```

### Examples

```bash
# Test with different embedding model
python -m src.stage03_modeling.test_octis_pipeline --subset --embedding-model "paraphrase-mpnet-base-v2"

# Force regenerate embeddings (ignore cache)
python -m src.stage03_modeling.test_octis_pipeline --subset --no-cache
```

## What the Script Does

### Step 1: Load and Validate CSV
- Reads CSV with `latin1` encoding (as in legacy code)
- Validates column structure (`Author, Book Title, Chapter, Sentence`)
- Cleans sentences (remove newlines, normalize whitespace, lowercase)
- Checks for duplicates
- Tokenizes sentences

**Output**: DataFrame, list of strings, list of tokenized lists

### Step 2: Create OCTIS Dataset Format
- Creates `data/interim/octis/corpus.tsv`
- Format: `sentence<TAB>train<TAB>author,book_title`
- Validates OCTIS can load the dataset

**Output**: `data/interim/octis/corpus.tsv`

### Step 3: Load or Create Embeddings
- Checks for cached embeddings (`data/interim/octis/embeddings_cache/<model>_embeddings.npy`)
- If not found, generates embeddings using SentenceTransformer
- Saves to cache for future use

**Output**: NumPy array of embeddings (shape: [n_documents, embedding_dim])

### Step 4: Create BERTopic Model
- Instantiates `BERTopicOctisModelWithEmbeddings` class (from legacy code)
- Sets up representation models (KeyBERT, MMR, POS)
- Configures default hyperparameters

**Output**: Model instance ready for training

### Step 5: Test Single Training Run
- Runs one training iteration with default hyperparameters
- Validates topics are generated
- Checks output format (topics, topic-word matrix, topic-document matrix)

**Output**: Validation success/failure

## Expected Output

```
================================================================================
                    OCTIS+BERTopic Pipeline Test
================================================================================

📋 Test mode: SUBSET (10K)
🤖 Embedding model: all-MiniLM-L12-v2
💾 Cache embeddings: True

[STEP 1] Loading CSV file: data/processed/chapters_subset_10000.csv
--------------------------------------------------------------------------------
📁 File exists: data/processed/chapters_subset_10000.csv
📊 File size: 0.99 MB
📖 Reading CSV with latin1 encoding...
📋 Headers: ['Author', 'Book Title', 'Chapter', 'Sentence']
✅ Total rows loaded: 10386
📊 DataFrame shape: (10386, 4)
📋 Columns: ['Author', 'Book Title', 'Chapter', 'Sentence']

🧹 Cleaning sentences...
   - Removing newlines
   - Normalizing whitespace
   - Converting to lowercase

📝 Sample cleaned sentences:
   1. prologue h e was tired...
   2. dog-tired...
   3. amped up by pleasure mere minutes ago, his heartbeat was beginning to even out...

✅ Total sentences: 10386
🔍 Duplicate sentences: 0 (0.00%)

🔤 Tokenizing sentences...
✅ Non-empty documents: 10386
⚠️  Empty documents filtered: 0

[STEP 2] Creating OCTIS dataset format
--------------------------------------------------------------------------------
📁 OCTIS dataset directory: data/interim/octis
📄 Output file: data/interim/octis/corpus.tsv
...
```

## Troubleshooting

### Error: "CSV file not found"
- Check that `chapters.csv` or `chapters_subset_10000.csv` exists in `data/processed/`
- Verify paths in `configs/paths.yaml`

### Error: "Missing required library"
- Install dependencies: `pip install bertopic octis sentence-transformers umap-learn hdbscan`

### Error: "CUDA out of memory"
- Use `--subset` first to test with smaller dataset
- Reduce batch size in embedding generation (edit script)
- Use CPU: Set `device="cpu"` in script

### Error: "OCTIS dataset load failed"
- Check `corpus.tsv` format (should be TSV with 3 columns)
- Verify file encoding is UTF-8

## Comparison with Legacy Code

This test script uses the same features as `legacy_code_OCTIS/bertopic_plus_octis.py`:

✅ **Same features**:
- CSV loading with `latin1` encoding
- Sentence cleaning (newlines, whitespace, lowercase)
- OCTIS corpus.tsv format creation
- Embedding caching
- BERTopicOctisModelWithEmbeddings class structure
- Default hyperparameters
- RAPIDS support (optional, falls back to CPU)

❌ **Not included** (by design - this is a test, not optimization):
- Bayesian optimization loop
- Multiple embedding models
- Experiment result saving
- Email notifications
- Restart logic

## Next Steps

After successful test:

1. **Run full optimization** (if needed):
   ```bash
   python -m src.stage03_modeling.bertopic_runner
   ```

2. **Use retrain path** (if you have hyperparameters):
   ```bash
   python -m src.stage03_modeling.main retrain \
     --dataset_csv data/processed/chapters.csv \
     --out_dir models/retrained/
   ```

3. **Check outputs**:
   - `data/interim/octis/corpus.tsv` - OCTIS dataset
   - `data/interim/octis/embeddings_cache/` - Cached embeddings

---

*Last updated: 2025-01-XX*

