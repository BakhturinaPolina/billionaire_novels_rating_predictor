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
├── analysis/              # Main analysis scripts (entry points)
│   ├── generate_topic_probabilities_goodreads.py  # Generate book/chapter topic probabilities
│   ├── category_statistics.py      # Statistical analysis of categories
│   └── taxonomy_radway_eda.py      # EDA for taxonomy & Radway mappings
├── utils/                # Reusable helper modules
│   ├── statistics.py              # Statistical test functions
│   └── visualization.py            # Plotting and visualization functions
└── docs/                 # Documentation and reports
    ├── STATISTICAL_ANALYSIS_REPORT.md
    ├── VISUALIZATION_EXAMPLES.md
    └── VISUALIZATION_IMPROVEMENTS.md
```

## Analysis Scripts

### 1. Topic Probability Generation (`analysis/generate_topic_probabilities_goodreads.py`)

Generates book-level and chapter-level topic probabilities from sentence-level data. This is a prerequisite for statistical analysis and correlation studies.

**Features:**
- Aggregates sentence-level topic probabilities to book and chapter levels
- Supports Goodreads ID-based book identification for reliable metadata merging
- Caching system for efficient recomputation (saves ~515MB cache)
- Batch processing for large datasets
- Normalized probability distributions (sum to 1.0 per book/chapter)

**Usage:**
```bash
python -m src.stage10_correlation_analysis.analysis.generate_topic_probabilities_goodreads \
    --sentence-df data/processed/sentence_df_with_topics.parquet \
    --model-path models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_categories \
    --output-dir results/stage10_correlation_analysis \
    --book-id-source existing \
    [--batch-size 32] \
    [--cache-dir results/stage10_correlation_analysis/cache] \
    [--no-cache]
```

**Outputs:**
- `book_topic_probs.parquet`: Book-level topic probabilities (book_id, topic_id, prob)
  - Format: One row per (book, topic) pair
  - Example: 33,856 rows for 92 books × 368 topics
- `chapter_topic_probs.parquet`: Chapter-level topic probabilities (book_id, chapter_id, topic_id, prob)
  - Format: One row per (chapter, topic) pair
  - Example: 1,089,280 rows for 2,960 chapters × 368 topics
- `cache/topic_probs_*.npz`: Cached probability arrays for efficient recomputation

**Key Features:**
- **Book ID Handling**: Supports Goodreads IDs (recommended), existing book_id column, or Author+Title fallback
- **Caching**: MD5-based cache keys detect input changes (file size + modification time)
- **Normalization**: Probabilities sum to 1.0 per book/chapter (validated automatically)
- **Error Handling**: Comprehensive diagnostics for zero-probability chapters and NaN values

### 2. Category Statistics (`analysis/category_statistics.py`)

Runs statistical analysis to identify taxonomy categories that differ significantly across book rating classes.

**Features:**
- Kruskal-Wallis tests for each category
- Effect size calculations (eta-squared)
- Post-hoc pairwise comparisons
- Comprehensive visualizations (volcano plots, effect size bars, prevalence plots)

**Usage:**
```bash
python -m src.stage10_correlation_analysis.analysis.category_statistics \
    --book-cat results/stage09_category_mapping/stage2_theory_driven_categories/book_category_proportions.parquet \
    --output-dir results/stage10_correlation_analysis/category_statistical_analysis \
    --top-n 10 \
    --alpha 0.05
```

**Outputs:**
- `kruskal_wallis_results.csv`: Statistical test results
- `figures/volcano_plot.png`: P-value vs effect size visualization
- `figures/effect_size_bars.png`: Effect sizes for top categories
- `figures/category_*_prevalence.png`: Individual category plots
- `figures/category_*_pairwise.png`: Pairwise comparison plots

### 3. Taxonomy & Radway EDA (`analysis/taxonomy_radway_eda.py`)

Exploratory data analysis of the BERTopic model with taxonomy and Radway narrative function mappings.

**Features:**
- Distribution analysis of taxonomy categories
- Distribution analysis of Radway narrative functions
- Cross-tabulations between taxonomy and Radway
- Summary statistics and data exports

**Usage:**
```bash
python -m src.stage10_correlation_analysis.analysis.taxonomy_radway_eda \
    --output-dir results/stage10_correlation_analysis/taxonomy_radway_eda
```

**Outputs:**
- `taxonomy_distribution.png`: Taxonomy category distributions
- `radway_distribution.png`: Radway function distributions
- `cross_tabulations.png`: Taxonomy vs Radway cross-tabulations
- `summary_statistics.json`: Summary statistics
- `full_model_data.csv` / `.parquet`: Complete extracted data

## Helper Modules

### `utils/statistics.py`
- Kruskal-Wallis test functions
- Effect size calculations
- Post-hoc pairwise comparisons

### `utils/visualization.py`
- Volcano plots
- Effect size bar charts
- Category prevalence plots (box, violin, strip plots)
- Pairwise comparison visualizations
- P-value heatmaps

## Inputs

- **Sentence DataFrame**: `data/processed/sentence_df_with_topics.parquet`
  - Required columns: `text`, `book_id` (or `goodreads_book_id`), `chapter_id` (optional)
  - Contains sentence-level topic assignments from Stage 09
- **BERTopic Model**: Model with taxonomy and Radway mappings (from Stage 9)
  - Recommended: `models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_categories`
- **Book Category Proportions**: `results/stage09_category_mapping/stage2_theory_driven_categories/book_category_proportions.parquet`
- **Taxonomy Mappings**: `results/stage09_category_mapping/stage2_theory_driven_categories/taxonomy_mappings_*.json`

## Outputs

- **Topic Probabilities** (Production): `results/stage10_correlation_analysis/`
  - `book_topic_probs.parquet`: Book-level topic probabilities (production version)
  - `chapter_topic_probs.parquet`: Chapter-level topic probabilities (production version)
  - `cache/`: Cached probability arrays for efficient recomputation

- **Statistical Results**: `results/stage10_correlation_analysis/category_statistical_analysis/`
  - Test results, effect sizes, visualizations

- **EDA Results**: `results/stage10_correlation_analysis/taxonomy_radway_eda/`
  - Distribution plots, cross-tabulations, summary statistics

## Dependencies

- `pandas`, `numpy` for data manipulation
- `scipy` for statistical tests (Kruskal-Wallis, Mann-Whitney U)
- `matplotlib`, `seaborn` for visualization
- `bertopic` for model loading

