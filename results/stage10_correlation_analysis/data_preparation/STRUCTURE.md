# Data Preparation Directory Structure

This document describes the organized structure of the data preparation stage outputs.

## Directory Organization

All data preparation outputs are organized under:
```
results/stage10_correlation_analysis/data_preparation/
```

### Subdirectories

#### `topic_probabilities/`
**Source**: `generate_topic_probabilities_final.py`

Contains topic probability files:
- `book_topic_probs.parquet` - Book-level topic probabilities
- `chapter_topic_probs.parquet` - Chapter-level topic probabilities
- `book_topic_probs.csv` - (Optional) CSV version
- `chapter_topic_probs.csv` - (Optional) CSV version

#### `taxonomy_radway_eda/`
**Source**: `01_data_validation_extraction.py`

Contains topic lookup and metadata:
- `topic_lookup.parquet` - Topic lookup table (primary output for merging)
- `full_model_data.csv` - Full topic metadata (CSV)
- `full_model_data.parquet` - Full topic metadata (Parquet)
- `summary_statistics.json` - QA summary statistics
- `topics_needs_review.csv` - Topics requiring manual review
- `crosstab_taxonomy_group_x_radway_phase.csv` - Cross-tabulation tables
- `crosstab_taxonomy_main_x_radway_id.csv` - Cross-tabulation tables

#### `book_features/`
**Source**: `02_book_aggregation.py`

Contains book-level aggregated features and indices:
- `book_taxonomy_main_props_long.parquet` - Long format category proportions
- `book_taxonomy_main_props_wide.parquet` - Wide format for modeling
- `indices_book_taxonomy_proxy.parquet` - Derived indices (love_over_sex, hea_index, etc.)
- `segment_taxonomy_main_props_long.parquet` - (Optional) Segment-level proportions

#### `diagnostics/`
**Source**: `01_data_validation_extraction.py`

Contains diagnostic reports:
- `id_alignment_report.csv` - ID overlap diagnostics between datasets
- `missing_books_in_outputs.csv` - Books missing from outputs (with metadata)

#### `cache/`
**Source**: `generate_topic_probabilities_final.py`

Contains cached computations:
- `transform_*.pkl` - Cached transform outputs
- `topic_probs_*.npz` - Cached topic probability arrays

#### `logs/`
**Source**: All scripts

Contains log files:
- `generate_topic_probabilities.log` - Logs from topic probability generation
- `01_data_validation_extraction.log` - Logs from validation script
- `02_book_aggregation.log` - Logs from aggregation script

## Migration Notes

### Old Structure (Archived)

Old outputs have been moved to:
```
archive/stage10_correlation_analysis_old_outputs/
```

This includes:
- `taxonomy_radway_eda/` (old version)
- `notebook2_book_features/` (old version)
- `notebook3_eda/` (old EDA outputs)
- `category_statistical_analysis/` (old statistical analysis)

### Backward Compatibility

Scripts maintain backward compatibility by checking multiple locations:
1. New organized structure (e.g., `data_preparation/topic_probabilities/`)
2. Old flat structure (e.g., `data_preparation/`)
3. Legacy locations (e.g., root of `stage10_correlation_analysis/`)

This ensures existing workflows continue to work while new outputs use the organized structure.

## Usage

When running scripts, outputs will automatically be saved to the appropriate subdirectories:

```bash
# Generate topic probabilities (saves to topic_probabilities/)
python generate_topic_probabilities_final.py \
    --output-dir results/stage10_correlation_analysis/data_preparation

# Run validation (saves to taxonomy_radway_eda/ and diagnostics/)
python 01_data_validation_extraction.py

# Run aggregation (saves to book_features/)
python 02_book_aggregation.py
```

All scripts use sensible defaults and will create the necessary subdirectories automatically.

