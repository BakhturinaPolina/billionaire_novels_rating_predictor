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

### Stage 06: Topic Exploration

#### Overview

Interactive tooling for inspecting retrained BERTopic models from Stage 05. Focuses on fast, instrumented loading of models and computing coherence/diversity diagnostics.

#### Process

1. **Model Loading**
   - Loads pickle wrapper (default) or native safetensors format
   - Streams documents in batches (50k default) from wrapper cache or CSV fallback
   - Builds gensim dictionary by streaming OCTIS corpus for coherence scoring

2. **Multiple Representations**
   - Attaches additional representations using BERTopic's `update_topics()`:
     - **Main**: Standard c-TF-IDF representation (BERTopic default)
     - **KeyBERT**: Keyword extraction using KeyBERT-inspired representation
     - **POS (Part-of-Speech)**: Filters keywords by part-of-speech patterns
     - **MMR (Maximal Marginal Relevance)**: Balances keyword relevance with diversity (diversity=0.3)

3. **Metrics Computation**
   - **c_v Coherence**: Measures semantic consistency using gensim's coherence model
   - **Topic Diversity**: Ratio of unique words to total extracted terms
   - Metrics computed per representation for comparison

4. **Topic Extraction**
   - Extracts all topics with all representations to JSON format
   - Enables close reading evaluation and qualitative analysis
   - Saves structured output for downstream stages

### Stage 07: Topic Quality Analysis

#### Overview

Exploratory data analysis to identify candidate noisy topics before LLM labeling. Non-destructive analysis that flags topics for manual inspection without modifying the model.

#### Process

1. **Quality Metrics Computation**
   - **Topic Size**: Number of documents assigned to each topic
   - **POS Representation Statistics**: Count of POS-filtered keywords per topic
   - **Per-Topic POS Coherence**: c_v coherence computed on POS-filtered keywords using the same gensim dictionary as training

2. **Noise Candidate Detection**
   - Flags topics based on configurable thresholds:
     - Topics with few POS words (< 3): Indicates lack of interpretable keywords
     - Topics with low or missing POS coherence (< 0.0): Suggests semantic incoherence
     - Topics below minimum size threshold (< 30 documents): May represent outliers or noise

3. **Labeling for Inspection**
   - Noisy topics are labeled with inspection tags (e.g., `[NOISE:few_pos<3]`, `[NOISE:low_coh<0.00]`)
   - Labels are applied to both wrapper pickle and native BERTopic model formats
   - Quality tables saved to CSV for review

4. **Output Files**
   - `topic_quality_{model}.csv`: Full topic quality table with all metrics
   - `topic_noise_candidates_{model}.csv`: Filtered view of only candidate noisy topics

### Stage 08: LLM Labeling

#### Overview

Automated generation of human-readable topic labels using Large Language Models. Supports both cloud-based (OpenRouter API) and local inference approaches.

#### Label Generation Process

1. **Keyword Extraction**
   - Extracts top keywords from POS representation (default: 15 keywords per topic)
   - Uses BERTopic's topic representation system

2. **MMR Reranking**
   - Applies Maximal Marginal Relevance (MMR) reranking to balance keyword relevance with diversity
   - Ensures the model receives a diverse set of representative keywords

3. **Domain Detection**
   - Automatically detects semantic domains (e.g., BodyParts, FoodDrink, TimeSpan, Marriage) from keywords
   - Provides context-aware hints for more accurate labeling

4. **Label Generation**
   - **OpenRouter API** (recommended): Uses `mistralai/mistral-nemo` via OpenRouter API
     - No local GPU required
     - Faster iteration
     - Cloud-based inference
   - **Local Mistral-7B-Instruct**: Local inference with 4-bit quantization
     - Memory requirements: ~6GB VRAM with quantization
     - GPU-accelerated when available, with CPU fallback

5. **Label Integration**
   - Automatically integrates generated labels back into BERTopic models
   - Labels stored in BERTopic's `custom_labels_` attribute
   - Available in BERTopic visualizations and topic info

#### Enhanced Features

- **Representative Document Snippets**: Uses actual scene-level context for more precise labels
- **Romance-Aware Prompts**: Optimized for modern romantic and erotic fiction
- **Structured JSON Output**: With `--use-improved-prompts`, generates structured JSON with:
  - `label`: Short noun phrase (2-6 words)
  - `scene_summary`: One complete sentence describing typical scene
  - `primary_categories`: High-level tags
  - `secondary_categories`: Specific tags with dimension:value format
  - `is_noise`: Boolean indicating if topic is technical artifact
  - `rationale`: Explanation of label choice

### Stage 09: Category Mapping

#### Overview

Deterministic mapping from topic labels to theory-aligned categories using a three-stage approach. Operationalizes theoretical constructs from Radway (1984), Propp functions, and Ogas & Gaddam (2011).

#### Three-Stage Approach

1. **Stage 1: Natural Clusters**
   - Data-driven topic groupings without theoretical priors
   - Uses BERTopic's hierarchical topics to build tree structure
   - Reduces to interpretable meta-topics (40-80 topics)

2. **Stage 2: Theory-Driven Categories**
   - Maps topics to predefined theoretical categories using zero-shot classification
   - Categories include: luxury lifestyle, emotional depth, erotic content, etc.
   - Uses regex-based inference for deterministic mapping

3. **Stage 3: Radway Narrative Functions**
   - Maps topics to Radway's 13 narrative functions (R1-R13)
   - Uses zero-shot classification via Mistral-Nemo through OpenRouter
   - Integrates Radway mappings into taxonomy structure

#### Category Schema

**Core Composites (A-P)**: 16 thematic categories:
- **A**: Reassurance/Commitment (HEA centrality)
- **B**: Mutual Intimacy (non-explicit; love-over-sex preference)
- **C**: Explicit Eroticism
- **D**: Power/Wealth/Luxury
- **E**: Coercion/Brutality/Danger
- **F**: Angst/Negative Affect
- **G**: Courtship Rituals/Gifts
- **H**: Domestic Nesting
- **I**: Humor/Lightness
- **J**: Social Support/Kin
- **K**: Professional Intrusion
- **L**: Vices/Addictions
- **M**: Health/Recovery/Growth
- **N**: Separation/Reunion
- **O**: Aesthetics/Appearance
- **P**: Tech/Media Presence

**Cross-Cutting Categories**:
- **Q**: Miscommunication vs Repair
- **R**: Protectiveness vs Jealousy

#### Mapping Logic

1. **Regex-Based Inference**: Case-insensitive regex patterns match topic labels to categories
2. **Soft Assignments**: When multiple categories match, weights are normalized to sum to 1.0
3. **Fallback Heuristics**: If no patterns match, coarse POS-like heuristics assign categories

#### Output Files

- `topic_to_category_probs.json`: Per-topic soft category assignments
- `topic_to_category_final.csv`: Flat table format for inspection
- `book_category_props.csv`: Book-level category proportions
- `indices_book.csv`: All derived indices per book

### Stage 10: Correlation Analysis

#### Overview

Comprehensive statistical analysis and exploratory data analysis combining topic probabilities with Goodreads metadata. Tests research hypotheses with effect size calculations and generates visualizations.

#### Analysis Components

1. **Statistical Analysis**
   - Kruskal-Wallis tests for each category across rating classes
   - Effect size calculations (eta-squared)
   - Post-hoc pairwise comparisons with Holm correction
   - Comprehensive visualizations (volcano plots, effect size bars, prevalence plots)

2. **Exploratory Data Analysis**
   - Distribution analysis of taxonomy categories
   - Distribution analysis of Radway narrative functions
   - Cross-tabulations between taxonomy and Radway
   - Summary statistics and data exports

3. **Hypothesis Testing**
   - Group comparisons across Top/Medium/Trash popularity tiers
   - Effect sizes: Cohen's d for continuous, Cramér's V for categorical
   - Statistical tests with FDR correction

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
Stage 06: Topic Exploration
    ├─→ Metrics (coherence, diversity)
    └─→ Topics JSON (all representations)
    ↓
Stage 07: Topic Quality Analysis
    ├─→ Topic Quality CSV
    └─→ Model with Noise Labels
    ↓
Stage 08: LLM Labeling
    ├─→ Labels JSON
    └─→ Model with LLM Labels
    ↓
Stage 09: Category Mapping
    ├─→ Topic-to-Category Mappings
    ├─→ Book Category Proportions
    └─→ Derived Indices
    ↓
Stage 10: Correlation Analysis
    ├─→ Statistical Test Results
    ├─→ Visualizations
    └─→ Research Findings
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

