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

### Stage 06: Labeling

#### Input
- **Prepared Books**: `data/processed/prepared_books.parquet`
  - Contains topic probabilities per book
  - Must have topic columns (topic names as column names)

- **Manual Mapping**: `results/experiments/.../Billionaire_Manual_Mapping_Topics_by_Thematic_Clusters.md`
  - Markdown format with hierarchical topic clusters

- **Codebook** (Optional): `results/experiments/.../focused_topic_codebook.csv`
  - **Columns**:
    - `Topic`: Topic name
    - `PrimaryCategory`: Main category
    - `IntimacyFramingSubtype`: Subtype (if applicable)

#### Output
- **Composites Parquet**: `results/topics/ap_composites.parquet`
  - **Columns**:
    - `Title`, `Author`, `Group`, `Pages`, `Year`: Metadata
    - `A_Reassurance_Commitment` through `P_Tech_Media_Presence`: Composite scores (0.0-1.0)

### Stage 07: Analysis

#### Input
- **Topics CSV**: `results/topics/by_book.csv`
  - Topic probabilities per book

- **Goodreads CSV**: `data/processed/goodreads.csv`
  - Book metadata

- **Composites Parquet** (from Stage 06): `results/topics/ap_composites.parquet`

#### Output
- **Prepared Books**: `data/processed/prepared_books.parquet`
  - **Columns**:
    - Metadata: `Title`, `Author`, `RatingsCount`, `Score`, etc.
    - Topic columns: Normalized topic probabilities
    - Derived: `popularity_index`, `Group` (Top/Medium/Trash), `Year`
    - Z-scores: `z_RatingsCount`, `z_Score`, etc.

- **Statistical Results**: `results/analysis/`
  - Statistical test results
  - Model coefficients
  - Effect sizes

- **Visualizations**: `results/figures/`
  - Heatmaps
  - Group comparison plots
  - UMAP visualizations
  - Time-course plots

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

