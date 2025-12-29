# Stage 10: Correlation Analysis

## Overview

Stage 10 performs comprehensive statistical analysis and exploratory data analysis (EDA) combining topic probabilities with Goodreads metadata. This includes:

- **Statistical analysis** of taxonomy category differences across rating classes
- **Exploratory data analysis** of taxonomy and Radway narrative function mappings
- **Hypothesis testing** with effect size calculations
- **Visualization** of results

## Structure

```
stage10_correlation_analysis/
└── data_preparation/      # Data preparation scripts
    ├── 01_data_validation_extraction.py
    ├── 02_book_aggregation.py
    ├── 03_generate_topic_probabilities_final.py  # Generate book/chapter topic probabilities
    └── 04_generate_tertile_topic_probs.py  # Generate tertile topic probabilities
```

## Analysis Scripts

### 1. Topic Probability Generation (`data_preparation/03_generate_topic_probabilities_final.py`)

Generates book-level and chapter-level topic probabilities from sentence-level data. This is a prerequisite for statistical analysis and correlation studies.

**Features:**
- Aggregates sentence-level topic probabilities to book and chapter levels
- Supports Goodreads ID-based book identification for reliable metadata merging
- Caching system for efficient recomputation (saves ~515MB cache)
- Batch processing for large datasets
- Normalized probability distributions (sum to 1.0 per book/chapter)

**Usage:**
```bash
python src/stage10_correlation_analysis/data_preparation/03_generate_topic_probabilities_final.py \
    --sentence-df data/processed/sentence_df_with_topics.parquet \
    --model-path models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_radway_mappings \
    --output-dir results/stage10_correlation_analysis/data_preparation \
    --book-id-source goodreads \
    --goodreads-id-col ID \
    [--batch-size 32] \
    [--no-cache]
```

**Outputs:**
- `book_topic_probs.parquet`: Book-level topic probabilities (book_id, topic_id, prob)
  - Format: One row per (book, topic) pair
  - Example: 33,856 rows for 92 books × 368 topics
- `chapter_topic_probs.parquet`: Chapter-level topic probabilities (book_id, chapter_id, topic_id, prob)
  - Format: One row per (chapter, topic) pair
  - Example: 1,089,280 rows for 2,960 chapters × 368 topics
- `cache/transform_*.pkl`: Cached transform outputs for efficient recomputation

**Key Features:**
- **Book ID Handling**: Supports Goodreads IDs (recommended), existing book_id column, or Author+Title fallback
- **Caching**: File-based cache keys detect input changes (file size + modification time)
- **Normalization**: Probabilities sum to 1.0 per book/chapter (validated automatically)
- **Error Handling**: Comprehensive diagnostics for zero-probability chapters and NaN values

### 2. Tertile Topic Probabilities (`data_preparation/04_generate_tertile_topic_probs.py`)

Generates topic probabilities for begin/middle/end tertiles of each book by splitting the book's token stream into three equal parts and re-inferring topic mixtures per tertile.

**Features:**
- Splits each book's sentences into three tertiles (begin/middle/end)
- Re-infers topic probabilities for each tertile using BERTopic model
- Preserves sentence order within books for accurate tertile boundaries
- Normalized probability distributions (sum to 1.0 per tertile)

**Usage:**
```bash
python src/stage10_correlation_analysis/data_preparation/04_generate_tertile_topic_probs.py \
    --sentence-df data/processed/sentence_df_with_topics.parquet \
    --model-path models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_radway_mappings \
    --output-dir results/stage10_correlation_analysis/data_preparation \
    --book-id-source goodreads \
    --goodreads-id-col ID
```

**Outputs:**
- `tertile_topic_probs.parquet`: Tertile-level topic probabilities (book_id, tertile, topic_id, prob)
  - Format: One row per (book, tertile, topic) pair
  - Tertile values: "begin", "middle", "end"
  - Example: 92 books × 3 tertiles × 368 topics = 101,568 rows

**Key Features:**
- **Tertile Splitting**: Divides each book's sentences into three equal parts based on sentence order
- **Order Preservation**: Maintains original sentence order (by chapter_id or sentence_index if available)
- **Normalization**: Probabilities sum to 1.0 per tertile (validated automatically)
- **Statistical Analysis Ready**: Output format suitable for analyzing topic distribution differences across book parts and rating classes

## Inputs

- **Sentence DataFrame**: `data/processed/sentence_df_with_topics.parquet`
  - Required columns: `text`, `book_id` (or `goodreads_book_id`), `chapter_id` (optional)
  - Contains sentence-level topic assignments from Stage 09
- **BERTopic Model**: Model with taxonomy and Radway mappings (from Stage 9)
  - Recommended: `models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_categories`
- **Book Category Proportions**: `results/stage09_category_mapping/stage2_theory_driven_categories/book_category_proportions.parquet`
- **Taxonomy Mappings**: `results/stage09_category_mapping/stage2_theory_driven_categories/taxonomy_mappings_*.json`

## Outputs

- **Topic Probabilities** (Production): `results/stage10_correlation_analysis/data_preparation/topic_probabilities/`
  - `book_topic_probs.parquet`: Book-level topic probabilities (production version)
  - `chapter_topic_probs.parquet`: Chapter-level topic probabilities (production version)
  - `tertile_topic_probs.parquet`: Tertile-level topic probabilities (begin/middle/end per book)
  - `cache/`: Cached transform outputs for efficient recomputation

## Dependencies

- `pandas`, `numpy` for data manipulation
- `scipy` for statistical tests (Kruskal-Wallis, Mann-Whitney U)
- `matplotlib`, `seaborn` for visualization
- `bertopic` for model loading

