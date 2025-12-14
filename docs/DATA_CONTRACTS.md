# Data Contracts

## Overview

This document specifies the input and output data formats for each stage of the pipeline. All data contracts must be satisfied for the pipeline to execute correctly.

## Input Data Contracts

### Stage 01: Ingestion

#### Raw Text Files
- **Location**: `data/raw/Billionaire_Full_Novels_TXT/`
- **Format**: Plain text files (`.txt`)
- **Naming**: `{book_title}.txt` or `{book_id}.txt`
- **Content**: Full novel text, one file per book
- **Encoding**: UTF-8

#### Goodreads Metadata
- **Location**: `data/processed/goodreads.csv`
- **Format**: CSV
- **Required Columns**:
  - `Title`: Book title
  - `Author`: Author name
  - `RatingsCount`: Number of ratings
  - `Score`: Average rating
  - `Popularity_ReadingNow`: Current reading popularity metric
  - `Popularity_Wishlisted`: Wishlist popularity metric
  - `Pages`: Number of pages
  - `PublishedDate`: Publication date (format flexible)

#### BookNLP Outputs (Optional)
- **Location**: `data/interim/booknlp/`
- **Format**: Multiple files per book
  - `.book`: BookNLP book file
  - `.entities`: Named entities
  - `.tokens`: Tokenized text
  - `.txt`: Processed text

### Stage 02: Preprocessing

#### Input
- Raw text files from Stage 01
- Custom stoplist: `data/processed/custom_stoplist.txt`

#### Output
- **Location**: `data/processed/chapters.csv`
- **Format**: CSV
- **Columns**:
  - `Book_Title`: Book identifier
  - `Sentence`: Preprocessed sentence text
  - Additional metadata columns as needed

### Stage 03: Modeling

#### Input
- **Chapters CSV**: `data/processed/chapters.csv`
- **OCTIS Dataset**: `data/interim/octis/` (if using OCTIS format)

#### Output
- **Model Evaluation Results**: `results/experiments/model_evaluation_results.csv`
  - **Columns**:
    - `Embeddings_Model`: Sentence transformer model name
    - `Coherence`: Topic coherence score
    - `Topic_Diversity`: Topic diversity score
    - `n_topics`: Number of topics discovered
    - Hyperparameter columns (UMAP, HDBSCAN, etc.)
    - `model_path`: Path to saved model

- **Trained Models**: `models/` directory
  - BERTopic model files (`.pkl` or native format)
  - Model metadata JSON files

- **Topic Outputs**: `results/topics/`
  - `by_book.csv`: Topic probabilities per book
  - `top_models/*.json`: Topic word lists for top models

### Stage 04: Selection

#### Input
- **Model Results CSV**: `results/experiments/model_evaluation_results.csv`
  - Must contain: `Coherence`, `Topic_Diversity`, `Embeddings_Model`, hyperparameter columns

#### Output
- **Top Models CSV**: `results/pareto/top_10_equal_weights.csv`, `top_10_coherence_priority.csv`
  - **Columns**: All input columns plus:
    - `Combined_Score`: Weighted combination of metrics
    - `Pareto_Efficient_All`: Boolean for overall Pareto efficiency
    - `Pareto_Efficient_PerModel`: Boolean for per-model Pareto efficiency
    - `pareto_rank`: Ranking within Pareto-efficient models

- **Visualizations**: `results/pareto/figures/`
  - `pareto_front_equal_weights.png`
  - `pareto_front_coherence_priority.png`
  - `pareto_fronts_per_model.png`
  - `distribution_with_cutoffs.png`
  - `hyperparameter_boxplots.png`

- **Correlation Tables**: `results/pareto/tables/`
  - `correlation_analysis_equal_weights.csv`
  - `correlation_analysis_coherence_priority.csv`

### Stage 05: Retraining

#### Input
- **Pareto CSV**: `results/pareto/top_10_equal_weights.csv` (or coherence priority)
  - Must contain hyperparameter columns for model reconstruction

- **Dataset CSV**: `data/processed/chapters.csv`
  - Same format as Stage 03 input

#### Output
- **Retrained Models**: `models/retrained/{embedding_model}/`
  - `model_{n}.pkl`: Pickle format (full wrapper)
  - `model_{n}/`: BERTopic native format directory
  - `model_{n}_metadata.json`: Training metadata

### Stage 06: Topic Exploration

#### Input
- **Retrained Models**: `models/retrained/{embedding_model}/model_{n}.pkl` or `model_{n}/`
- **OCTIS Corpus**: `data/interim/octis/corpus.tsv` (for gensim dictionary)
- **Dataset CSV**: `data/processed/chapters.csv` (optional, for document loading)

#### Output
- **Metrics JSON/CSV**: `results/stage06_topic_exploration/metrics_{model}.json`
  - Coherence (c_v) and diversity scores per representation
- **Topics JSON**: `results/stage06_topic_exploration/topics_all_representations_{model}.json`
  - All topics with all representations (Main, KeyBERT, POS, MMR)

### Stage 07: Topic Quality Analysis

#### Input
- **Retrained Models**: `models/retrained/{embedding_model}/model_{n}.pkl` or `model_{n}/`
- **OCTIS Corpus**: `data/interim/octis/corpus.tsv` (for gensim dictionary)
- **Dataset CSV**: `data/processed/chapters.csv` (optional, for document loading)

#### Output
- **Topic Quality CSV**: `results/stage07_topic_quality/topic_quality_{model}.csv`
  - Full topic quality table with all metrics
- **Noise Candidates CSV**: `results/stage07_topic_quality/topic_noise_candidates_{model}.csv`
  - Filtered view of only candidate noisy topics
- **Model with Labels** (optional): `models/retrained/{embedding_model}/stage07_topic_quality/model_{n}_with_noise_labels.pkl`

### Stage 08: LLM Labeling

#### Input
- **BERTopic Model**: `models/retrained/{embedding_model}/stage07_topic_quality/model_{n}_with_noise_labels/` (or base model)
- **Topics JSON** (optional): `results/stage06_topic_exploration/topics_all_representations_{model}.json`

#### Output
- **Labels JSON**: `results/stage08_llm_labeling/labels_pos_openrouter_{model_name}_{embedding_model}.json`
  - Format: `{"topic_id": {"label": "...", "keywords": [...], "scene_summary": "...", ...}}`
- **Model with Labels**: `models/retrained/{embedding_model}/stage08_llm_labeling/model_{n}_with_llm_labels.pkl` and `model_{n}_with_llm_labels/`

### Stage 09: Category Mapping

#### Input
- **BERTopic Model with Labels**: `models/retrained/{embedding_model}/stage08_llm_labeling/model_{n}_with_llm_labels/`
- **Sentence DataFrame**: `data/processed/sentence_df_with_ratings.parquet` (optional, for book-level aggregation)
- **Chapters CSV**: `data/processed/chapters.csv`
- **Goodreads CSV**: `data/processed/goodreads.csv`

#### Output
- **Topic-to-Category Mappings**: `results/stage09_category_mapping/stage2_theory_driven_categories/topic_to_category_probs.json`
  - Per-topic soft category assignments (weights sum to 1.0)
- **Topic-to-Category CSV**: `results/stage09_category_mapping/stage2_theory_driven_categories/topic_to_category_final.csv`
  - Flat table format for inspection
- **Book Category Proportions**: `results/stage09_category_mapping/stage2_theory_driven_categories/book_category_proportions.parquet`
  - Book-level category proportions
- **Indices CSV**: `results/stage09_category_mapping/stage2_theory_driven_categories/indices_book.csv` (optional)
  - All derived indices per book

### Stage 10: Correlation Analysis

#### Input
- **Book Category Proportions**: `results/stage09_category_mapping/stage2_theory_driven_categories/book_category_proportions.parquet`
  - Book-level category proportions from Stage 09
- **Taxonomy Mappings**: `results/stage09_category_mapping/stage2_theory_driven_categories/taxonomy_mappings_*.json`
  - Taxonomy and Radway function mappings
- **BERTopic Model**: Model with taxonomy and Radway mappings (from Stage 09)
- **Goodreads CSV**: `data/processed/goodreads.csv`
  - Book metadata with ratings

#### Output
- **Statistical Results**: `results/stage10_correlation_analysis/category_statistical_analysis/`
  - `kruskal_wallis_results.csv`: Statistical test results
  - Effect sizes and post-hoc comparisons
  - Visualizations: volcano plots, effect size bars, prevalence plots
- **EDA Results**: `results/stage10_correlation_analysis/taxonomy_radway_eda/`
  - Distribution plots: `taxonomy_distribution.png`, `radway_distribution.png`
  - Cross-tabulations: `cross_tabulations.png`
  - Summary statistics: `summary_statistics.json`
  - Full model data: `full_model_data.csv` / `.parquet`

## Data Validation

### Schema Validation

Each stage should validate input schemas:

```python
def validate_schema(df, required_columns, optional_columns=None):
    """Validate DataFrame has required columns."""
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True
```

### Type Validation

- **Numeric columns**: Check for NaN, inf, negative values where inappropriate
- **Categorical columns**: Check for valid categories
- **Date columns**: Validate date formats

### Range Validation

- **Probabilities**: Should be in [0, 1] range
- **Scores**: Check for expected ranges
- **Counts**: Should be non-negative integers

## Data Normalization

### Topic Probabilities

**Per-book normalization:**
```python
# Sum must equal 1.0 per book (within tolerance)
row_sums = topic_probs.sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-6)
```

### Missing Values

- **Topic probabilities**: Fill with 0.0 (topic not present)
- **Metadata**: Handle according to analysis needs
- **Ratings**: May have NaN for unrated books

## File Formats

### CSV
- **Encoding**: UTF-8
- **Delimiter**: Comma (`,`)
- **Header**: First row contains column names
- **Quoting**: Handle special characters appropriately

### Parquet
- **Format**: Apache Parquet
- **Compression**: Snappy (default)
- **Schema**: Preserved with column types

### JSON
- **Format**: JSON (UTF-8)
- **Structure**: Nested dictionaries/lists as appropriate
- **Indentation**: 2 spaces (for readability)

## Data Versioning

### Naming Conventions

- Include timestamps or version numbers in filenames for experiments:
  - `model_evaluation_results_20250115.csv`
  - `pareto_analysis_v2.csv`

### Backup Strategy

- Keep raw data immutable
- Version intermediate results
- Archive old model outputs

## Data Privacy

- **No personal information**: Ensure no PII in outputs
- **Aggregated statistics**: Use aggregated data for sharing
- **Anonymization**: Author/book identifiers should be anonymized if needed

---

For index definitions, see [INDICES.md](INDICES.md).  
For technical methodology, see [METHODOLOGY.md](METHODOLOGY.md).

