# Stage 10 Correlation Analysis - Results Directory

This directory contains outputs from Stage 10 correlation and statistical analysis.

## Directory Structure

### `data_preparation/`
**Purpose**: Data preparation stage outputs (topic probabilities, topic lookup, book features)

This is the main directory for all data preparation outputs. See `data_preparation/STRUCTURE.md` for detailed organization.

**Subdirectories**:
- `topic_probabilities/` - Topic probability files (book and chapter level)
- `taxonomy_radway_eda/` - Topic lookup and metadata
- `book_features/` - Book-level aggregated features and indices
- `diagnostics/` - Diagnostic reports (ID alignment, missing books)
- `cache/` - Cached computations
- `logs/` - Log files from scripts

### Other Directories

Other directories in this location may contain:
- Legacy outputs (moved to `archive/stage10_correlation_analysis_old_outputs/`)
- Intermediate analysis results
- Experimental outputs

## Scripts Location

Data preparation scripts are located at:
```
src/stage10_correlation_analysis/data_preparation/
```

See the scripts' README.md for usage instructions.

## Archive

Old outputs from previous code versions have been moved to:
```
archive/stage10_correlation_analysis_old_outputs/
```

This includes:
- Old `taxonomy_radway_eda/` outputs
- Old `notebook2_book_features/` outputs
- Old `notebook3_eda/` outputs
- Old `category_statistical_analysis/` outputs
