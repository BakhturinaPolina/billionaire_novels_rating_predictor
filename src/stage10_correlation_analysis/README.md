# Stage 10: Correlation Analysis

## Overview

Stage 10 performs statistical analysis combining topic probabilities with Goodreads metadata. The analysis examines relationships between thematic content and book outcomes (reach and perceived quality) at three levels:

- **Macro-axis level**: Weighted combinations of CORE predictors
- **Topic-level**: Individual BERTopic probabilities (368 topics)  
- **Taxonomy-group level**: Aggregated probability mass (8 main groups, 27 subgroups)

## Structure

```
stage10_correlation_analysis/
└── data_preparation/
    ├── 01_data_validation_extraction.py  # Extract topic metadata, validate IDs
    ├── 02_book_aggregation.py            # Aggregate to book-level, compute indices
    ├── 03_generate_topic_probabilities_final.py  # Book/chapter topic probabilities
    └── 04_generate_tertile_topic_probs.py        # Tertile probabilities (begin/middle/end)
```

## Data Preparation Pipeline

### Execution Order

```bash
# Step 1: Generate topic probabilities
python src/stage10_correlation_analysis/data_preparation/03_generate_topic_probabilities_final.py \
    --sentence-df data/processed/sentence_df_with_topics.parquet \
    --model-path models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_radway_mappings \
    --output-dir results/stage10_correlation_analysis/data_preparation \
    --book-id-source goodreads --goodreads-id-col ID

# Step 2: Generate tertile probabilities (optional, for arc analysis)
python src/stage10_correlation_analysis/data_preparation/04_generate_tertile_topic_probs.py \
    --sentence-df data/processed/sentence_df_with_topics.parquet \
    --model-path models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_radway_mappings \
    --output-dir results/stage10_correlation_analysis/data_preparation \
    --book-id-source goodreads --goodreads-id-col ID

# Step 3: Extract topic metadata
python src/stage10_correlation_analysis/data_preparation/01_data_validation_extraction.py \
    --output-dir results/stage10_correlation_analysis/data_preparation/taxonomy_radway_eda

# Step 4: Aggregate to book-level
python src/stage10_correlation_analysis/data_preparation/02_book_aggregation.py \
    --topic-lookup results/stage10_correlation_analysis/data_preparation/taxonomy_radway_eda/topic_lookup.parquet \
    --output-dir results/stage10_correlation_analysis/data_preparation/book_features
```

### Key Outputs

**Topic Probabilities** (`data_preparation/topic_probabilities/`):
- `book_topic_probs.parquet`: Book-level topic probabilities (92 books × 368 topics)
- `chapter_topic_probs.parquet`: Chapter-level topic probabilities
- `tertile_topic_probs.parquet`: Tertile-level probabilities (begin/middle/end per book)

**Topic Metadata** (`data_preparation/taxonomy_radway_eda/`):
- `topic_lookup.parquet`: Topic-level lookup with taxonomy and Radway mappings

**Book Features** (`data_preparation/book_features/`):
- `book_taxonomy_main_props_wide.parquet`: Book-level taxonomy proportions
- `indices_book_taxonomy_proxy.parquet`: Derived hypothesis-aligned indices

### Derived Indices

| Index | Formula | Hypothesis |
|-------|---------|------------|
| love_over_sex | emotional_content - explicit | H1: Emotional intimacy vs. explicit sexuality |
| hea_index | commitment_hea | H2: HEA content prevalence |
| explicitness_ratio | explicit / total_romantic | H1 variant |
| dark_vs_tender | dark_content - tender_content | H5: Dark vs. tender content |

## Inputs

- **Sentence DataFrame**: `data/processed/sentence_df_with_topics.parquet`
- **BERTopic Model**: `models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_radway_mappings`
- **Goodreads Metadata**: `data/processed/goodreads.csv`

## Analysis Notebooks

Located in `notebooks/07_analysis/`:
- `01_topic_analysis/`: Topic-level comparisons across rating tiers
- `02_taxonomy_group_analysis/`: Taxonomy group comparisons, diversity analysis
- `03_composite_index_construction/`: Theory-aligned composite index building
- `04_hypothesis_testing/`: Statistical hypothesis tests

## Dependencies

- `pandas`, `numpy`: Data manipulation
- `scipy`: Statistical tests (Kruskal-Wallis, Mann-Whitney U, bootstrap)
- `matplotlib`, `seaborn`: Visualization
- `bertopic`: Model loading
- `pyarrow`: Parquet support

## Reports

- **Main Report**: `reports/01_stage_reports/stage10_correlation_analysis/stage10_correlation_analysis_report.md`
- **Drafts**: `reports/01_stage_reports/stage10_correlation_analysis/drafts/` (working documents, git-ignored)
