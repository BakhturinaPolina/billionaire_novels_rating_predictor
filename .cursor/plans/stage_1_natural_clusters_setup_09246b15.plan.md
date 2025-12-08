---
name: Stage 1 Natural Clusters Setup
overview: Document BERTopic model versions and structure, explore data files, and implement prepare_sentence_dataframe.py for Stage 1 natural clusters analysis.
todos:
  - id: doc_model_versions
    content: Create MODEL_VERSIONING.md documenting all BERTopic model versions, their paths, relationships, and how to load them
    status: in_progress
  - id: explore_model_structure
    content: Explore model_1_with_noise_labels structure, document BERTopic native format contents, verify noise labels storage
    status: pending
  - id: review_logs
    content: Review logs for model loading issues and document any fixes needed
    status: pending
  - id: explore_chapters_csv
    content: "Explore chapters.csv structure: columns, data types, missing values, text format validation"
    status: pending
  - id: explore_goodreads_csv
    content: "Explore goodreads.csv structure: rating distributions, missing values, Author/Title format"
    status: pending
  - id: locate_llm_labels
    content: Locate and document LLM labels JSON files structure and integration method
    status: pending
  - id: implement_prepare_dataframe
    content: Implement prepare_sentence_dataframe.py with fuzzy matching, rating class creation, position_norm calculation
    status: pending
    dependencies:
      - explore_chapters_csv
      - explore_goodreads_csv
  - id: update_readme
    content: Update stage1_natural_clusters README with discovered file paths and data structures
    status: pending
    dependencies:
      - doc_model_versions
      - explore_model_structure
      - locate_llm_labels
---

# Stage 1: Natural Clusters Setup Plan

## Phase 1: Model Versioning Documentation

### 1.1 Create Model Versioning Document

**File**: `docs/MODEL_VERSIONING.md`

Document all BERTopic model versions:

- **model_1**: Base retrained model (368 topics)
- **model_1_with_noise_labels**: Base model + noise candidate labels (20 noisy topics flagged)
- **model_1_with_categories**: Model with LLM-generated category labels (if exists)

For each version, document:

- File paths (both pickle wrapper and native directory formats)
- Creation date/timestamp
- Topic count
- Labels/metadata included
- Relationship to other versions (e.g., model_1_with_noise_labels extends model_1)
- How to load each version

### 1.2 Explore Model Structure

**Tasks**:

- Inspect `model_1_with_noise_labels` directory structure
- Document BERTopic native format contents (config.json, topics.json, safetensors files)
- Check if noise labels are stored in `custom_labels_` attribute
- Verify openrouter_experiments default uses `_with_noise_labels` suffix (already confirmed: default is `"_with_noise_labels"`)

### 1.3 Review Logs for Issues

**File**: `logs/stage06_labeling_openrouter_20251206_201600.log`

- Check for warnings/errors related to model loading
- Document any fixes needed

## Phase 2: Data Exploration

### 2.1 Explore chapters.csv Structure

**File**: `data/processed/chapters.csv`

- Columns: `Author`, `Book Title`, `Chapter`, `Sentence`
- Total rows: ~707,909
- Check for:
- Missing values
- Sentence text format (should match BERTopic training text)
- Chapter numbering consistency
- Author/Book Title variations

### 2.2 Explore goodreads.csv Structure  

**File**: `data/processed/goodreads.csv`

- Columns: `ID`, `Author`, `Title`, `Score`, `RatingsCount`, etc.
- Total rows: 98 books
- Check for:
- Rating distributions
- Missing ratings
- Author/Title format differences from chapters.csv

### 2.3 Document LLM Labels Location

**Reference**: `src/stage06_labeling/category_mapping/stage1_natural_clusters/README.md:27-29`

- Locate JSON files in `results/stage06_labeling_openrouter/`
- Document structure: `{topic_id: {label, keywords, scene_summary, ...}}`
- Note how labels integrate into BERTopic model (`custom_labels_` attribute)

## Phase 3: Implement prepare_sentence_dataframe.py

### 3.1 File Location

**File**: `src/stage06_labeling/category_mapping/stage1_natural_clusters/prepare_sentence_dataframe.py`

### 3.2 Implementation Steps

1. **Load Data**

- Load `chapters.csv` (handle large file efficiently)
- Load `goodreads.csv`

2. **Create book_id**

- Generate unique book identifier from Author + Title
- Use fuzzy matching to handle variations between chapters.csv and goodreads.csv
- Consider using `difflib.SequenceMatcher` or `fuzzywuzzy` for matching

3. **Merge Data**

- Merge chapters with goodreads on book_id (fuzzy matched)
- Attach book-level ratings to each sentence

4. **Filter Books**

- Filter books with `rating_count >= 100` (configurable threshold)
- Log how many books/sentences are filtered out

5. **Create rating_class**

- Calculate quantiles: 33rd and 66th percentile of `rating_mean`
- Assign classes:
 - `bad`: rating_mean < 33rd percentile
 - `mid`: 33rd percentile <= rating_mean <= 66th percentile  
 - `good`: rating_mean > 66th percentile

6. **Add position_norm**

- Group by book_id
- Calculate total sentences per book
- Compute `position_norm = sentence_index / total_sentences` for each sentence

7. **Verify text column**

- Ensure `text` column (from `Sentence` in chapters.csv) matches exactly what was used for BERTopic training
- Add validation/warning if format differs

8. **Add sentence_id**

- Create unique sentence identifier (optional but helpful)
- Format: `{book_id}_{chapter}_{sentence_index}` or sequential

9. **Output**

- Save cleaned dataframe to parquet format
- Log summary statistics (books, sentences, rating class distribution)

### 3.3 Configuration

- Make thresholds configurable (min_ratings, quantile cutoffs)
- Add command-line arguments or config file support

### 3.4 Error Handling

- Handle missing ratings gracefully
- Warn about books that can't be matched between files
- Validate data types and required columns

## Phase 4: Documentation

### 4.1 Update README

Update `src/stage06_labeling/category_mapping/stage1_natural_clusters/README.md` with:

- Actual file paths discovered
- Data structure details
- Model version used

### 4.2 Create Data Dictionary

Document the output dataframe schema in a separate markdown file or in the README.

## Dependencies

- pandas
- numpy  
- difflib or fuzzywuzzy (for fuzzy matching)
- pathlib

## Output Files

1. `docs/MODEL_VERSIONING.md` - Model version tracking
2. `src/stage06_labeling/category_mapping/stage1_natural_clusters/prepare_sentence_dataframe.py` - Implementation
3. Updated README with findings
4. Output parquet file: `data/processed/sentence_df_with_ratings.parquet` (or similar)