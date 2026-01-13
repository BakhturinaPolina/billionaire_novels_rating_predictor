# Stage 10 Data Preparation Pipeline: Comprehensive Report

**Date**: January 2025  
**Location**: `src/stage10_correlation_analysis/data_preparation/`  
**Outputs**: `results/stage10_correlation_analysis/data_preparation/`

---

## Executive Summary

This report documents the complete data preparation pipeline for Stage 10 correlation analysis. The pipeline transforms raw BERTopic topic assignments into book-level and segment-level features suitable for statistical analysis and hypothesis testing. The pipeline consists of four main scripts that process data sequentially, with comprehensive validation, error handling, and diagnostic outputs at each stage.

**Key Outputs**:
- Book-level topic probabilities (normalized per book)
- Segment-level topic probabilities (begin/middle/end tertiles)
- Topic lookup table with taxonomy and Radway mappings
- Book-level taxonomy category proportions
- Derived indices aligned to research hypotheses
- Comprehensive diagnostic reports

---

## Pipeline Overview

### Data Flow

```
Sentence-level data (sentence_df_with_topics.parquet)
    ↓
[Script 03] Generate Topic Probabilities
    ├── Book-level probabilities (book_topic_probs.parquet)
    ├── Chapter-level probabilities (chapter_topic_probs.parquet)
    └── Tertile probabilities (tertile_topic_probs.parquet) [Script 04]
    ↓
[Script 01] Data Validation & Extraction
    ├── Topic lookup table (topic_lookup.parquet)
    ├── Full model metadata (full_model_data.csv)
    └── Diagnostic reports (ID alignment, missing books)
    ↓
[Script 02] Book Aggregation
    ├── Book-level taxonomy proportions (long + wide format)
    ├── Segment-level taxonomy proportions (if available)
    └── Derived indices (love_over_sex, hea_index, etc.)
```

### Script Execution Order

1. **Script 03**: `03_generate_topic_probabilities_final.py` - Generate topic probabilities
2. **Script 04**: `04_generate_tertile_topic_probs_patched_v3.py` - Generate tertile probabilities (optional)
3. **Script 01**: `01_data_validation_extraction.py` - Extract topic metadata and validate
4. **Script 02**: `02_book_aggregation.py` - Aggregate to book-level and compute indices

---

## Script 03: Generate Topic Probabilities

**File**: `03_generate_topic_probabilities_final.py`

### Purpose

Generates normalized topic probabilities at book and chapter levels from sentence-level BERTopic assignments. This is the foundation for all downstream analysis.

### Key Features

1. **Goodreads-first book IDs**: Uses Goodreads IDs as primary identifier for reliable merging
2. **Robust ID normalization**: Handles `.0` suffixes, whitespace, and null values
3. **Cohort exclusion**: Optionally excludes books missing from sentence_df
4. **Caching**: Caches BERTopic transform outputs to save computation time (~2 hours)
5. **Probability normalization**: Ensures probabilities sum to ~1.0 per book/chapter
6. **NaN handling**: Replaces NaN values with 0.0 before aggregation (critical fix)

### Inputs

- **Sentence DataFrame**: `data/processed/sentence_df_with_topics.parquet`
  - Contains sentence-level text and topic assignments
  - Required columns: `sentence_text`, `book_id`, `topic_id`
  
- **BERTopic Model**: `models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_radway_mappings`
  - Final model with taxonomy and Radway mappings
  
- **Goodreads Metadata** (optional): `data/processed/goodreads.csv`
  - For ID alignment and validation
  
- **Excluded Book IDs** (optional): CSV with `book_id` column
  - Default: `['19561986', '19619918', '25781538', '52061964', '53491034']`
  - Books filtered out before sentence_df creation

### Outputs

**Location**: `results/stage10_correlation_analysis/data_preparation/topic_probabilities/`

1. **`book_topic_probs.parquet`**
   - Format: Long format `[book_id, topic_id, prob]`
   - Shape: ~33,856 rows (92 books × 368 topics)
   - Normalization: Sum(prob) ≈ 1.0 per book
   - Validation: 0% NaN values (NaN replaced with 0.0)

2. **`chapter_topic_probs.parquet`**
   - Format: Long format `[book_id, chapter_id, topic_id, prob]`
   - Normalization: Sum(prob) ≈ 1.0 per (book, chapter)
   - Validation: 0% NaN values

### Usage

```bash
python src/stage10_correlation_analysis/data_preparation/03_generate_topic_probabilities_final.py \
    --sentence-df data/processed/sentence_df_with_topics.parquet \
    --model-path models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_radway_mappings \
    --output-dir results/stage10_correlation_analysis/data_preparation \
    --book-id-source goodreads \
    --goodreads-id-col ID \
    --exclude-book-ids notebooks/07_analysis/statistical_analysis/excluded_book_ids.csv
```

### Critical Fix: NaN Handling

**Problem**: BERTopic's `transform()` can return NaN values, which propagate through aggregation.

**Solution**: Replace NaN with 0.0 **before** aggregation:
- After loading/transforming probabilities, check for NaN
- Replace all NaN with 0.0 using `np.nan_to_num()`
- Validate final output has zero NaN values

**Verification**:
```python
import pandas as pd
book_probs = pd.read_parquet('results/stage10_correlation_analysis/data_preparation/topic_probabilities/book_topic_probs.parquet')
print(f"NaN count: {book_probs['prob'].isna().sum()}")  # Should be 0
```

---

## Script 04: Generate Tertile Topic Probabilities

**File**: `04_generate_tertile_topic_probs_patched_v3.py`

### Purpose

Generates topic probabilities for begin/middle/end tertiles of each book, enabling narrative arc analysis.

### Key Features

1. **Tertile splitting**: Divides each book's ordered sentences into three equal tertiles
2. **Chunking**: Splits tertiles into manageable chunks to prevent embedding truncation
3. **Weighted aggregation**: Aggregates chunk probabilities within each tertile
4. **Book ordering**: Preserves sentence order within each book

### Inputs

- Same as Script 03 (sentence_df, model, excluded book IDs)

### Outputs

**Location**: `results/stage10_correlation_analysis/data_preparation/topic_probabilities/`

1. **`tertile_topic_probs.parquet`**
   - Format: `[book_id, segment, topic_id, prob]`
   - Segments: `begin`, `middle`, `end`
   - Shape: ~101,568 rows (92 books × 3 segments × 368 topics)
   - Normalization: Sum(prob) ≈ 1.0 per (book, segment)

### Usage

```bash
python src/stage10_correlation_analysis/data_preparation/04_generate_tertile_topic_probs_patched_v3.py \
    --sentence-df data/processed/sentence_df_with_topics.parquet \
    --model-path models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_radway_mappings \
    --output-dir results/stage10_correlation_analysis/data_preparation \
    --book-id-source goodreads \
    --goodreads-id-col ID
```

### Technical Details

- **Chunk size**: 40 sentences per chunk (configurable)
- **Minimum sentences**: 3 sentences required per tertile (books with fewer sentences are skipped)
- **Aggregation**: Weighted mean of chunk probabilities within tertile

---

## Script 01: Data Validation & Extraction

**File**: `01_data_validation_extraction.py`

### Purpose

Entry point for Stage 10 analysis. Loads the final BERTopic model, merges Stage 08 label metadata, and exports a topic-level lookup table for downstream aggregation.

### Key Features

1. **Model loading**: Loads BERTopic model with taxonomy & Radway mappings
2. **Label merging**: Merges Stage 08 LLM label metadata (labels, scene summaries, categories)
3. **QA checks**: Validates missing mappings, keyword quality, confidence distributions
4. **ID alignment**: Performs diagnostics between datasets (topic probs, goodreads metadata)
5. **Fallback support**: Can use exported CSV if model unavailable

### Inputs

- **BERTopic Model**: `models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_radway_mappings`
  - Required attributes: `topic_representations_`, `topic_taxonomy_`, `topic_radway_`
  
- **Stage 08 Labels**: `results/stage08_llm_labeling/labels_*.json`
  - Format: JSON dictionary mapping topic_id to metadata
  - Supports both rich format (dict with label, scene_summary, etc.) and simple format (topic_id: "label")
  
- **Goodreads Metadata**: `data/processed/goodreads.csv`
  - For ID alignment diagnostics
  - Required columns: `ID` (or `goodreads_book_id`, `book_id`)
  
- **Excluded Book IDs** (optional): CSV with `book_id` column

### Outputs

**Location**: `results/stage10_correlation_analysis/data_preparation/taxonomy_radway_eda/`

1. **`topic_lookup.parquet`**
   - Topic-level lookup table for merging with topic probabilities
   - Columns: `topic_id`, `name`, `keywords`, `taxonomy_main_id`, `taxonomy_main_name`, `radway_id`, `radway_name`, `label`, `scene_summary`, `primary_categories`, `secondary_categories`, `is_noise`
   - Shape: (369, 21) - one row per topic

2. **`full_model_data.csv` / `.parquet`**
   - Full topic metadata in CSV/Parquet format
   - Fallback for when model is unavailable

3. **`summary_statistics.json`**
   - QA summary: topic counts, mapping coverage, keyword quality

4. **`topics_needs_review.csv`**
   - Topics requiring manual review (missing mappings, poor keywords)

**Location**: `results/stage10_correlation_analysis/data_preparation/diagnostics/`

5. **`id_alignment_report.csv`**
   - ID overlap diagnostics between datasets
   - Shows overlap, coverage, and sample IDs

6. **`missing_books_in_outputs.csv`**
   - Books missing from outputs (with metadata)
   - Helps trace where books were lost in pipeline

### Usage

```bash
python src/stage10_correlation_analysis/data_preparation/01_data_validation_extraction.py \
    --output-dir results/stage10_correlation_analysis/data_preparation/taxonomy_radway_eda \
    --excluded-book-ids notebooks/07_analysis/statistical_analysis/excluded_book_ids.csv
```

### Auto-Detection

The script auto-detects:
- Model path (searches common locations)
- Labels path (finds newest `labels_*.json` in stage08 directory)
- Goodreads path (searches common locations)

---

## Script 02: Book Aggregation

**File**: `02_book_aggregation.py`

### Purpose

Takes the topic-level lookup table and joins it to book topic mixture data to produce book-level taxonomy proportions and derived indices aligned to research hypotheses.

### Key Features

1. **Taxonomy aggregation**: Aggregates topic probabilities to taxonomy category proportions
2. **Multiple formats**: Produces both long and wide format outputs
3. **Segment-level support**: Optionally processes chapter/segment data
4. **Derived indices**: Computes hypothesis-aligned indices (love_over_sex, hea_index, etc.)
5. **ID normalization**: Handles ID mismatches and normalization automatically

### Inputs

- **Topic Lookup**: `results/stage10_correlation_analysis/data_preparation/taxonomy_radway_eda/topic_lookup.parquet`
  - From Script 01 output
  
- **Book Topic Probabilities**: `results/stage10_correlation_analysis/data_preparation/topic_probabilities/book_topic_probs.parquet`
  - From Script 03 output
  
- **Chapter/Segment Topic Probabilities** (optional): `chapter_topic_probs.parquet` or `tertile_topic_probs.parquet`
  - Enables segment-level analysis
  
- **Goodreads Metadata**: `data/processed/goodreads.csv`
  - For book metadata and ID alignment

### Outputs

**Location**: `results/stage10_correlation_analysis/data_preparation/book_features/`

1. **`book_taxonomy_main_props_long.parquet`**
   - Long format: `[book_id, taxonomy_main_id, taxonomy_main_name, prop]`
   - One row per (book, category) pair
   - Shape: ~2,484 rows (92 books × 27 categories)

2. **`book_taxonomy_main_props_wide.parquet`**
   - Wide format: one row per book, columns are taxonomy categories
   - Shape: (92, 27+ categories)
   - Suitable for modeling and statistical analysis

3. **`indices_book_taxonomy_proxy.parquet`**
   - Derived indices aligned to research hypotheses
   - Columns: `book_id` + all derived indices
   - See "Derived Indices" section below

4. **`segment_taxonomy_main_props_long.parquet`** (if segment data available)
   - Segment-level proportions in long format
   - Format: `[book_id, segment, taxonomy_main_id, prop]`

### Usage

```bash
python src/stage10_correlation_analysis/data_preparation/02_book_aggregation.py \
    --topic-lookup results/stage10_correlation_analysis/data_preparation/taxonomy_radway_eda/topic_lookup.parquet \
    --output-dir results/stage10_correlation_analysis/data_preparation/book_features \
    --excluded-book-ids notebooks/07_analysis/statistical_analysis/excluded_book_ids.csv
```

### Auto-Discovery

The script auto-discovers:
- Book topic probabilities (searches organized structure first, then fallback locations)
- Chapter/segment probabilities (if available)
- Goodreads metadata (searches common locations)

---

## Derived Indices

Script 02 computes the following taxonomy-proxy indices aligned to research hypotheses:

### 1. `love_over_sex`
- Formula: `(commitment_hea + bonding_growth + positive_emotions + nonexplicit_affection) - explicit`
- Measures: Emphasis on emotional connection vs. explicit sexuality
- Hypothesis: H1 - Emotional intimacy vs. explicit sexuality

### 2. `hea_index`
- Formula: `commitment_hea`
- Measures: "Happily ever after" content (reconciliation, commitments)
- Hypothesis: H2 - HEA content prevalence

### 3. `explicitness_ratio`
- Formula: `explicit / (explicit + commitment_hea + positive_emotions + nonexplicit_affection)`
- Measures: Ratio of explicit sexual content to total romantic content
- Hypothesis: H1 variant - Sexual content proportion

### 4. `dark_vs_tender`
- Formula: `(neg_affect + breakup_conflict + violence_threat) - (positive_emotions + nonexplicit_affection)`
- Measures: Dark/conflict content vs. tender/positive content
- Hypothesis: H5 - Dark content vs. tender content

### 5. `miscommunication_balance`
- Formula: `(commitment_hea + bonding_growth + positive_emotions) - miscommunication`
- Measures: Resolution/connection vs. miscommunication/conflict
- Hypothesis: Q - Miscommunication vs. resolution

### 6. `luxury_saturation_proxy`
- Formula: `elite_work + public_leisure`
- Measures: Luxury/business world content
- Hypothesis: H3 - Luxury saturation

---

## Data Quality & Validation

### Probability Normalization

**Book-level probabilities**:
- Expected: Sum(prob) ≈ 1.0 per book
- Validation: 0% of books outside [0.99, 1.01] range
- Median density: 70.4% of topics have prob > 0.001 per book

**Segment-level probabilities**:
- Expected: Sum(prob) ≈ 1.0 per (book, segment)
- Validation: All segments normalized

### ID Alignment

All scripts perform ID alignment diagnostics:
- **Overlap checks**: Verify book IDs match between datasets
- **Coverage metrics**: Report how much of each dataset is covered
- **Sample IDs**: Show sample overlapping IDs for debugging
- **Missing books report**: Lists books missing from outputs

**Common Issues**:
- ID format mismatches (`.0` suffixes, whitespace)
- Missing books in sentence_df (excluded books)
- Column name mismatches (`ID` vs `book_id` vs `goodreads_book_id`)

**Solutions**:
- Scripts auto-normalize IDs (strip, remove `.0` suffixes)
- Use `--excluded-book-ids` to exclude known problematic books
- Check `id_alignment_report.csv` for detailed diagnostics

### NaN Handling

**Critical**: All scripts replace NaN with 0.0 before aggregation:
- Script 03: Replaces NaN in BERTopic transform outputs
- Script 04: Replaces NaN in tertile probabilities
- Script 02: Validates no NaN in final outputs

**Verification**:
```python
import pandas as pd
df = pd.read_parquet('path/to/output.parquet')
assert df.isna().sum().sum() == 0, "Found NaN values!"
```

---

## Directory Structure

All outputs are organized under `results/stage10_correlation_analysis/data_preparation/`:

```
data_preparation/
├── topic_probabilities/          # From Scripts 03 & 04
│   ├── book_topic_probs.parquet
│   ├── chapter_topic_probs.parquet
│   └── tertile_topic_probs.parquet
├── taxonomy_radway_eda/           # From Script 01
│   ├── topic_lookup.parquet
│   ├── full_model_data.csv
│   ├── summary_statistics.json
│   └── topics_needs_review.csv
├── book_features/                  # From Script 02
│   ├── book_taxonomy_main_props_long.parquet
│   ├── book_taxonomy_main_props_wide.parquet
│   ├── indices_book_taxonomy_proxy.parquet
│   └── segment_taxonomy_main_props_long.parquet
├── diagnostics/                   # From Script 01
│   ├── id_alignment_report.csv
│   └── missing_books_in_outputs.csv
├── cache/                         # Cached computations (Script 03)
└── logs/                          # Log files from all scripts
```

---

## Troubleshooting

### Common Issues

#### 1. Zero Overlap Between Datasets

**Problem**: `id_alignment_report.csv` shows zero overlap between book IDs.

**Solutions**:
1. Check that `book_id` in topic probability files matches `ID` column in `goodreads.csv`
2. Verify ID normalization (check for `.0` suffixes, whitespace)
3. Ensure Script 03 used `--book-id-source goodreads` and correct `--goodreads-id-col`
4. Review `id_alignment_report.csv` for sample IDs

#### 2. Missing Books in Outputs

**Problem**: `missing_books_in_outputs.csv` lists books that should be present.

**Solutions**:
1. Check excluded book IDs are correct
2. Verify books exist in sentence_df
3. Check if books were filtered upstream (before sentence_df creation)
4. Review log files for filtering messages

#### 3. NaN Values in Outputs

**Problem**: Output files contain NaN values.

**Solutions**:
1. Regenerate topic probabilities with Script 03 (fixes NaN handling)
2. Check BERTopic model is working correctly
3. Verify sentence_df has valid text data
4. Check log files for warnings about NaN values

#### 4. Model Not Found

**Problem**: Script 01 cannot find BERTopic model.

**Solutions**:
1. Check model path in script output/logs
2. Verify model exists at expected location
3. Script will attempt to load exported `full_model_data.csv` as fallback
4. Run Script 01 with model available to generate fallback CSV

#### 5. Labels File Not Found

**Problem**: Script 01 cannot find Stage 08 labels JSON.

**Solutions**:
1. Check `results/stage08_llm_labeling/` for available label files
2. Script will attempt to find newest `labels_*.json` file in directory
3. Provide explicit path with `--labels-path` if needed

### Cohort Exclusion Logic

The scripts exclude books that never produced any sentence rows. Default excluded books:
- `19561986` - The Tycoon's Vacation
- `19619918` - The Tycoon's Proposal
- `25781538` - The Tycoon's Revenge
- `52061964` - Reverie
- `53491034` - The Billionaire's Salvation: Max

These books are excluded because they were filtered out before `sentence_df` creation, meaning they have no usable text for analysis.

**To customize excluded books**:
1. Create/edit `excluded_book_ids.csv` with `book_id` column
2. Provide path with `--excluded-book-ids` argument

---

## Execution Workflow

### Complete Pipeline Run

```bash
# Step 1: Generate topic probabilities
python src/stage10_correlation_analysis/data_preparation/03_generate_topic_probabilities_final.py \
    --sentence-df data/processed/sentence_df_with_topics.parquet \
    --model-path models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_radway_mappings \
    --output-dir results/stage10_correlation_analysis/data_preparation \
    --book-id-source goodreads \
    --goodreads-id-col ID \
    --exclude-book-ids notebooks/07_analysis/statistical_analysis/excluded_book_ids.csv

# Step 2: Generate tertile probabilities (optional)
python src/stage10_correlation_analysis/data_preparation/04_generate_tertile_topic_probs_patched_v3.py \
    --sentence-df data/processed/sentence_df_with_topics.parquet \
    --model-path models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_radway_mappings \
    --output-dir results/stage10_correlation_analysis/data_preparation \
    --book-id-source goodreads \
    --goodreads-id-col ID

# Step 3: Extract topic metadata
python src/stage10_correlation_analysis/data_preparation/01_data_validation_extraction.py \
    --output-dir results/stage10_correlation_analysis/data_preparation/taxonomy_radway_eda \
    --excluded-book-ids notebooks/07_analysis/statistical_analysis/excluded_book_ids.csv

# Step 4: Aggregate to book-level
python src/stage10_correlation_analysis/data_preparation/02_book_aggregation.py \
    --topic-lookup results/stage10_correlation_analysis/data_preparation/taxonomy_radway_eda/topic_lookup.parquet \
    --output-dir results/stage10_correlation_analysis/data_preparation/book_features \
    --excluded-book-ids notebooks/07_analysis/statistical_analysis/excluded_book_ids.csv
```

### Partial Runs

Scripts can be run independently if inputs are available:
- Script 01 requires model and labels (but can use fallback CSV)
- Script 02 requires Script 01 output (topic_lookup) and Script 03 output (book_topic_probs)
- Script 04 requires same inputs as Script 03

---

## Next Steps

After running the data preparation pipeline:

1. **Exploratory Data Analysis (EDA)**
   - Use `book_taxonomy_main_props_wide.parquet` for visualizations
   - Analyze index distributions and correlations
   - Use `indices_book_taxonomy_proxy.parquet` for hypothesis exploration

2. **Statistical Testing**
   - Use `indices_book_taxonomy_proxy.parquet` for hypothesis tests
   - Use `book_taxonomy_main_props_wide.parquet` for category-level analysis
   - Use `tertile_topic_probs.parquet` for narrative arc analysis

3. **Modeling**
   - Use wide format proportions and indices as features
   - Predict ratings or popularity groups
   - Test interaction effects

4. **Composite Index Construction**
   - Use topic probabilities and taxonomy mappings to build theory-aligned composite indices
   - See `measurement_pipeline_composite_indices.md` for composite construction methodology

---

## Key Design Decisions

### 1. Goodreads-First Book IDs

**Decision**: Use Goodreads IDs as primary identifier.

**Rationale**:**
- Most reliable for merging across datasets
- Consistent with upstream data sources
- Handles ID normalization automatically

### 2. NaN Replacement (Not Ignoring)

**Decision**: Replace NaN with 0.0, not ignore.

**Rationale**:
- NaN means "no probability assigned" → interpret as 0.0
- Ensures clean data throughout pipeline
- Prevents propagation of NaN through aggregations

### 3. Probability Normalization

**Decision**: Normalize probabilities to sum to ~1.0 per book/chapter.

**Rationale**:
- Ensures valid probability distributions
- Makes proportions interpretable
- Required for downstream statistical analysis

### 4. Auto-Detection of Inputs

**Decision**: Scripts auto-detect file paths when possible.

**Rationale**:
- Reduces user burden
- Handles common file organization patterns
- Still allows explicit paths for flexibility

### 5. Comprehensive Diagnostics

**Decision**: Generate diagnostic reports at each stage.

**Rationale**:
- Enables troubleshooting
- Documents data quality
- Provides transparency in pipeline

---

## Dependencies

### Required Python Packages

- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `bertopic` - Topic modeling (for loading models)
- `pyarrow` or `fastparquet` - Parquet file support
- `tqdm` - Progress bars

### Required Data Files

See individual script sections for detailed input requirements.

### Project Structure

Scripts assume standard project structure:
```
project_root/
├── data/
│   └── processed/
│       ├── sentence_df_with_topics.parquet
│       └── goodreads.csv
├── models/
│   └── retrained/
│       └── paraphrase-MiniLM-L6-v2/
│           └── stage09_category_mapping/
│               └── model_1_with_radway_mappings/
├── results/
│   ├── stage08_llm_labeling/
│   │   └── labels_*.json
│   └── stage10_correlation_analysis/
│       └── data_preparation/
└── src/
    └── stage10_correlation_analysis/
        └── data_preparation/
```

---

## Reproducibility

### Version Control

- Scripts are versioned and documented
- Outputs include metadata (timestamps, script versions)
- Log files record all parameters and decisions

### Caching

- Script 03 caches BERTopic transform outputs
- Cache keyed by sentence_df file + model path (mtime/size)
- Use `--no-cache` to force regeneration

### Validation

- All scripts validate inputs before processing
- Outputs validated for expected structure and data quality
- Diagnostic reports document any issues

---

## References

### Related Documentation

- **Measurement Pipeline**: `measurement_pipeline_composite_indices.md` - Composite index construction methodology
- **Statistical Analysis**: `STATISTICAL_ANALYSIS_REPORT.md` - Results from downstream analysis
- **Data Preparation Guide**: Original detailed guide (archived)

### Related Scripts

- **Stage 08**: LLM labeling scripts that produce `labels_*.json`
- **Stage 09**: Category mapping that produces final BERTopic model
- **Stage 10 Analysis**: Downstream scripts that use these outputs

---

## Appendix: File Formats

### Topic Probability Format

**Long Format** (book_topic_probs.parquet):
```
book_id | topic_id | prob
--------|----------|-----
60416566|    0     | 0.0234
60416566|    1     | 0.0156
...
```

**Normalization**: Sum(prob) ≈ 1.0 per book_id

### Topic Lookup Format

**topic_lookup.parquet**:
```
topic_id | name | keywords | taxonomy_main_id | taxonomy_main_name | radway_id | radway_name | label | ...
---------|------|----------|------------------|-------------------|-----------|-------------|-------|----
   0     | ...  | ...      |       2.1        | Attraction        |    1      | Meeting     | ...  | ...
```

### Book Features Format

**Wide Format** (book_taxonomy_main_props_wide.parquet):
```
book_id | taxonomy_2.1 | taxonomy_2.2 | taxonomy_2.3 | ... | love_over_sex | hea_index | ...
--------|-------------|-------------|-------------|-----|--------------|-----------|-----
60416566|   0.0234    |   0.0156    |   0.0089    | ... |    0.1234     |  0.0456   | ...
```

---

**Report Generated**: January 2025  
**Pipeline Version**: v3 (patched)  
**Last Updated**: January 2025

