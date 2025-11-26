# Technical Methodology

## Overview

This document provides detailed technical information about the computational pipeline, algorithms, and implementation details.

## Topic Modeling Pipeline

### Stage 03: BERTopic Training

#### Architecture

1. **Embedding Generation**
   - Uses SentenceTransformers to generate contextual embeddings
   - Embeddings are cached to avoid recomputation
   - Supports 6 pre-trained models (see SCIENTIFIC_README.md)

2. **Dimensionality Reduction (UMAP)**
   - GPU-accelerated via RAPIDS cuML
   - Parameters optimized via OCTIS:
     - `n_neighbors`: Controls local vs global structure
     - `n_components`: Embedding dimensionality (typically 5-15)
     - `min_dist`: Minimum distance between points in embedding space

3. **Clustering (HDBSCAN)**
   - GPU-accelerated via RAPIDS cuML
   - Parameters optimized:
     - `min_cluster_size`: Minimum points to form a cluster
     - `min_samples`: Conservative estimate of cluster stability

4. **Topic Representation**
   - Uses class-based TF-IDF (c-TF-IDF) for topic word extraction
   - Representation models:
     - **KeyBERTInspired**: Extracts keywords using KeyBERT
     - **MaximalMarginalRelevance**: Balances relevance and diversity
     - **PartOfSpeech**: Filters by part of speech

#### OCTIS Integration

OCTIS (Optimization of Computational Tools for Interpretable Science) provides:
- Bayesian optimization for hyperparameter search
- Multi-objective optimization support
- Automated experiment tracking

**Optimization Process:**
1. Define hyperparameter search space
2. Run Bayesian optimization (typically 50-100 iterations)
3. Evaluate models on coherence and diversity metrics
4. Select Pareto-efficient models

### Stage 04: Pareto Efficiency Analysis

#### Algorithm

**Pareto Efficiency Definition:**
A model is Pareto-efficient if no other model dominates it in all objectives while being strictly better in at least one.

**Implementation:**
```python
def identify_pareto(df, metrics):
    pareto_efficient = np.ones(df.shape[0], dtype=bool)
    for i, row in df.iterrows():
        # Check if any other row dominates this one
        dominated = np.all(df[metrics] >= row[metrics], axis=1) & \
                   np.any(df[metrics] > row[metrics], axis=1)
        pareto_efficient[i] = not np.any(dominated)
    return pareto_efficient
```

#### Data Cleaning

1. **Failed Run Removal**
   - Removes models where Coherence = 1.0 or Topic_Diversity = 1.0
   - These indicate clustering failures

2. **Outlier Detection**
   - Uses z-score method (default: 2 standard deviations)
   - Removes extreme values that may skew analysis

3. **Normalization**
   - Z-score normalization (default)
   - Min-max normalization (alternative)
   - Applied to Coherence and Topic_Diversity before combination

#### Weighting Strategies

1. **Equal Weights** (50/50)
   - `Combined_Score = 0.5 * Coherence_norm + 0.5 * Topic_Diversity_norm`

2. **Coherence Priority** (70/30)
   - `Combined_Score = 0.7 * Coherence_norm + 0.3 * Topic_Diversity_norm`

### Stage 05: Model Retraining

#### Process

1. **Load Pareto Results**
   - Reads top N models from CSV
   - Extracts hyperparameters for each model

2. **Direct Training**
   - No optimization - uses fixed hyperparameters
   - Trains each model independently
   - Failures in one model don't stop others

3. **Model Persistence**
   - **Pickle format**: Full wrapper with embeddings
   - **BERTopic native**: Direct loading with `BERTopic.load()`
   - **Metadata JSON**: Hyperparameters, scores, timestamps

### Stage 06: Thematic Mapping

#### Semi-Supervised Approach

1. **Manual Mapping** (Markdown)
   - Researcher-defined topic-to-cluster mappings
   - Hierarchical structure with headers
   - Keyword-based matching

2. **Codebook** (CSV)
   - Structured category definitions
   - Primary categories and subtypes
   - Systematic assignment rules

3. **Automated Assignment**
   - Weighted topic-to-composite mapping
   - Special handling for ambiguous topics
   - Example: "Woman Pleasure and Arousal" → 70% Explicit, 30% Mutual Intimacy

#### Composite Building

Topics are mapped to 16 composites (A-P) with weighted assignments:
- Each topic can map to multiple composites
- Weights sum to 1.0 per topic
- Final composite scores = sum of (topic_prob × weight) for all topics

### Stage 07: Statistical Analysis

#### Popularity Index

**Formula:**
```
popularity_index = mean(z_RatingsCount, z_Score, 
                        z_Popularity_ReadingNow, z_Popularity_Wishlisted)
```

Where z-scores are computed as:
```
z_X = (X - mean(X)) / std(X)
```

#### Stratification

**Three-Tier System:**
- **Top**: popularity_index > 66.67th percentile
- **Medium**: 33.33rd < popularity_index ≤ 66.67th percentile
- **Trash**: popularity_index ≤ 33.33rd percentile

#### Topic Probability Normalization

Per-book normalization ensures fair comparison:
```python
# Sum topic probabilities per book
row_sum = topic_probs.sum(axis=1)

# Normalize (handle zero sums)
topic_probs_normalized = topic_probs.div(row_sum.replace(0, np.nan), axis=0)
```

## GPU Acceleration

### RAPIDS cuML

**Why GPU-Only?**
- UMAP and HDBSCAN are computationally intensive
- GPU acceleration provides 10-100x speedup
- Consistency: all models use same hardware

**Requirements:**
- CUDA 12.x compatible GPU
- RAPIDS cuML installed
- Sufficient GPU memory (typically 8GB+)

**Memory Management:**
- Embedding caching to avoid recomputation
- Batch processing for large datasets
- Thermal monitoring for long runs
- Automatic cleanup on errors

## Data Flow

```
Raw Text Files
    ↓
Stage 01: Ingestion
    ↓
Stage 02: Preprocessing
    ↓
Stage 03: Modeling (BERTopic + OCTIS)
    ├─→ Model Evaluation Results CSV
    └─→ Trained Models
    ↓
Stage 04: Selection (Pareto Analysis)
    ├─→ Top Models CSV
    └─→ Visualizations
    ↓
Stage 05: Retraining
    └─→ Retrained Models (pickle + native)
    ↓
Stage 06: Labeling (Thematic Mapping)
    └─→ Composite Scores (parquet)
    ↓
Stage 07: Analysis
    ├─→ Prepared Books (parquet)
    ├─→ Statistical Tests
    └─→ Visualizations
```

## Configuration Management

### YAML Configuration Files

1. **`configs/paths.yaml`**
   - Input/output paths
   - Data directory structure

2. **`configs/bertopic.yaml`**
   - BERTopic model parameters
   - Embedding model selection

3. **`configs/octis.yaml`**
   - OCTIS optimization settings
   - Hyperparameter search space

4. **`configs/selection.yaml`**
   - Pareto analysis parameters
   - Weighting strategies
   - Cleaning thresholds

5. **`configs/scoring.yaml`**
   - Statistical analysis settings
   - Index definitions

### Path Resolution

All paths are resolved relative to project root:
```python
from src.common.config import resolve_path
path = resolve_path(Path("data/processed/chapters.csv"))
```

## Error Handling

### GPU Memory Errors
- Automatic batch size adjustment
- Memory cleanup on OOM
- Logging for debugging

### Model Training Failures
- Individual model failures don't stop pipeline
- Error logging with full tracebacks
- Failed models excluded from results

### Data Validation
- Schema checks on input files
- Missing value handling
- Type validation

## Performance Optimization

1. **Embedding Caching**
   - Embeddings saved to disk
   - Reused across model training runs
   - Significant time savings

2. **Parallel Processing**
   - Model training can be parallelized
   - GPU batch processing
   - Multi-threaded data loading

3. **Incremental Processing**
   - Checkpoint support for long runs
   - Resume from last checkpoint
   - Progress tracking

## Reproducibility

### Random Seeds
- Fixed seeds for all random operations
- Reproducible model training
- Deterministic clustering

### Version Control
- Model metadata includes:
  - Software versions
  - Hyperparameters
  - Training timestamps
  - Random seeds used

### Logging
- Comprehensive logging at each stage
- Execution logs saved to `logs/`
- Memory and thermal monitoring

---

For research questions and hypotheses, see [SCIENTIFIC_README.md](../SCIENTIFIC_README.md).  
For data contract specifications, see [DATA_CONTRACTS.md](DATA_CONTRACTS.md).

