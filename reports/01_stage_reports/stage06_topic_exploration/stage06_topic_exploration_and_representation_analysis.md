# Stage 06: Topic Exploration and Quality Analysis

## Overview

Stage 06 provides comprehensive tooling for exploring and evaluating retrained BERTopic models from Stage 05. The stage focuses on three main objectives:

1. **Topic Representation Enhancement**: Attaching multiple representation models (Main, KeyBERT, POS, MMR) to enrich topic word lists
2. **Quantitative Evaluation**: Computing coherence and diversity metrics across different representations
3. **Quality Assessment**: Identifying noisy topics and preparing topics for downstream labeling and analysis

This stage bridges model training (Stage 05) and topic labeling (Stage 08) by providing both quantitative metrics and qualitative inspection tools.

## Purpose and Research Rationale

### Why Multiple Representations?

BERTopic's default c-TF-IDF representation (Main) captures statistical word importance but may miss semantically coherent or diverse word selections. We implement four complementary representations:

1. **Main (c-TF-IDF)**: Default statistical representation based on term frequency-inverse document frequency within topics
2. **KeyBERT**: Extracts keywords using BERT embeddings to capture semantic similarity
3. **POS (Part-of-Speech)**: Filters to content words (nouns, verbs, adjectives) to improve interpretability
4. **MMR (Maximal Marginal Relevance)**: Balances relevance and diversity to reduce redundancy

Each representation serves different analytical purposes:
- **Main**: Baseline statistical representation
- **KeyBERT**: Semantic coherence for close reading
- **POS**: Interpretability for human labeling
- **MMR**: Diversity for exploratory analysis

### Evaluation Metrics

We compute two complementary metrics:

1. **Coherence (c_v)**: Measures semantic coherence of topic word lists using Gensim's c_v coherence metric, which evaluates word co-occurrence patterns in sliding windows
2. **Topic Diversity**: Ratio of unique words to total words across all topics, measuring lexical diversity

These metrics help identify which representations produce the most interpretable and diverse topics for downstream analysis.

## Implementation

### Core Module: `explore_retrained_model.py`

The main exploration module provides:

#### Model Loading

**Wrapper Format (Default)**:
- Loads `RetrainableBERTopicModel` pickle files containing the trained model and original training dataset
- **Advantage**: Guarantees exact dataset match (stored in `wrapper.dataset_as_list_of_strings`)
- **Use case**: Recommended for EDA and quality analysis

**Native BERTopic Format**:
- Loads standard BERTopic safetensors directory
- **Advantage**: Portable, standard format
- **Use case**: Sharing/deployment, requires separate dataset provision

#### Document Preparation

Documents are loaded in batches (default: 50,000) with progress logging:
- From wrapper cache (if available)
- From CSV fallback (`chapters.csv` or `chapters_subset_10000.csv`)
- Normalized (whitespace, lowercase) for consistency

#### Dictionary Building

Gensim dictionary built by streaming `data/interim/octis/corpus.tsv`:
- Same vocabulary used during model training
- Required for coherence computation
- Batched processing for memory efficiency

#### Representation Attachment

```python
representations = {
    "Main": None,  # Default c-TF-IDF (already present)
    "KeyBERT": KeyBERTInspired(),
    "POS": PartOfSpeech("en_core_web_sm"),
    "MMR": MaximalMarginalRelevance(diversity=0.3),
}
```

After attachment, `topic_model.update_topics(docs)` regenerates topic word lists for all representations.

#### Metrics Computation

For each representation:
1. Extract top-K words per topic (default: K=10)
2. Compute c_v coherence using Gensim `CoherenceModel`
3. Compute topic diversity (unique words / total words)
4. Log results with timing information

#### Topic Extraction

When `--save-topics` flag is used, extracts all topics with all representations to JSON:
```json
{
  "Main": {
    "0": [{"word": "example", "score": 0.123}, ...],
    ...
  },
  "KeyBERT": {...},
  "POS": {...},
  "MMR": {...}
}
```

### Usage

#### Basic Exploration

```bash
python -m src.stage06_topic_exploration.explore_retrained_model \
  --embedding-model paraphrase-MiniLM-L6-v2 \
  --pareto-rank 1 \
  --dictionary-path data/interim/octis/corpus.tsv \
  --batch-size 50000
```

Outputs:
- Console metrics table
- `metrics.json` (or `metrics.csv`)

#### Full Topic Extraction

```bash
python -m src.stage06_topic_exploration.explore_retrained_model \
  --embedding-model paraphrase-MiniLM-L6-v2 \
  --pareto-rank 1 \
  --save-topics \
  --output-dir results/stage06_topic_exploration
```

Outputs:
- `metrics.json`
- `topics_all_representations.json` (all topics with all representations)

## Results: Model Evaluation

### Metrics Summary

For the selected model (`paraphrase-MiniLM-L6-v2`, Pareto rank 1):

| Representation | Topics | Coherence (c_v) | Topic Diversity |
|---------------|--------|-----------------|------------------|
| Main          | 368    | 0.404           | 0.602            |
| KeyBERT       | 368    | 0.278           | 0.645            |
| POS           | 368    | 0.315           | 0.692            |
| MMR           | 368    | 0.260           | 0.756            |

### Interpretation

1. **Main representation** achieves highest coherence (0.404), indicating strong statistical word co-occurrence patterns
2. **MMR representation** achieves highest diversity (0.756), providing the most lexically diverse topic word lists
3. **POS representation** balances coherence (0.315) and diversity (0.692), making it suitable for human interpretation and labeling
4. **KeyBERT** shows lower coherence (0.278) but moderate diversity (0.645), suggesting semantic similarity may not always align with statistical patterns

### Representation Selection for Downstream Analysis

Based on these metrics:
- **For LLM labeling (Stage 08)**: POS representation recommended (interpretable content words)
- **For exploratory analysis**: MMR representation recommended (diverse word lists)
- **For statistical validation**: Main representation recommended (highest coherence)

## Topic Quality Analysis

### Notebook: `06_topic_quality_eda.ipynb`

This notebook performs exploratory data analysis (EDA) on topics and flags candidate noisy topics for manual inspection.

#### Objectives

1. **Keep model intact**: No topics removed, only flagged for inspection
2. **EDA on POS representation**: Analyze topic size, POS word counts, per-topic POS coherence
3. **Noise candidate detection**: Flag topics with:
   - Few POS words (< 3)
   - Low POS coherence (< 0.0 threshold)
   - Small topic size (< 30 documents)

#### Methodology

1. **Load retrained model** (wrapper format recommended for dataset consistency)
2. **Load documents** from wrapper cache (exact training dataset)
3. **Build Gensim dictionary** from OCTIS corpus
4. **Build topic quality table** with:
   - Topic size (document count)
   - POS word count
   - Per-topic POS coherence (c_v)
   - Noise candidate flags
   - Inspection labels

#### Quality Metrics

For each topic:
- **Topic size**: Number of documents assigned to topic
- **n_pos_words**: Count of POS-filtered words in topic representation
- **coherence_c_v_pos**: Per-topic coherence using POS representation words
- **noise_candidate**: Boolean flag based on thresholds
- **noise_reason**: Explanation of why topic flagged (e.g., "few_pos<3", "low_coh<0.00")

#### Results

From the analysis of `paraphrase-MiniLM-L6-v2` model:

- **Total topics (excluding -1)**: 368
- **Candidate noisy topics**: 13
- **Topics with POS words < 10**: 20
- **Total unique noisy topics labeled**: 20

#### Noise Detection Criteria

Topics flagged as noisy based on:
1. **Few POS words** (< 3): Topics with insufficient content words for interpretation
2. **Low coherence** (< 0.0): Topics with poor semantic coherence
3. **Combined criteria**: Topics meeting both conditions

Examples of noisy topics:
- Topic 17, 18: Large topics (2064, 2062 docs) with no POS words (NaN coherence)
- Topic 141, 182, 183: Topics with only 1-2 POS words
- Topic 359, 283, 224: Topics with no POS words and NaN coherence

#### Labeling Strategy

Noisy topics are labeled in the model with inspection labels:
- Format: `[NOISE:reason] topic_name`
- Examples:
  - `[NOISE:noise_candidate] 17____`
  - `[NOISE:pos<10(6)] 106_wasa_thata_likea_sa`
  - `[NOISE:noise_candidate;pos<10(1)] 141_rush___`

These labels are saved to:
- Native BERTopic model: `model_1_with_noise_labels/`
- Wrapper pickle: `model_1_with_noise_labels.pkl`

**Note**: Topics are labeled but not removed, allowing for manual inspection and potential recovery of valid topics.

## Taxonomy Category Analysis

### Notebook: `06_taxonomy_category_analysis.ipynb`

This notebook provides helper functions for analyzing taxonomy category prevalence across rating classes, preparing for correlation analysis (Stage 10).

#### Functions

1. **`kruskal_by_rating(book_cat)`**:
   - Runs Kruskal-Wallis tests for each taxonomy category
   - Tests category prevalence differences across rating classes (low/mid/high)
   - Returns DataFrame with:
     - `category_id`: Taxonomy category
     - `groups`: Rating classes tested
     - `n_books_per_group`: Sample sizes
     - `H_statistic`: Kruskal-Wallis statistic
     - `p_value`: Statistical significance

2. **`plot_category_prevalence(book_cat, category_id)`**:
   - Creates box plots with jitter for category prevalence by rating class
   - Visualizes distribution differences across rating classes
   - Useful for identifying categories with significant rating differences

#### Prerequisites

Expects book-level category proportions from `aggregate_taxonomy_by_book.py`:
- DataFrame columns: `book_id`, `rating_class`, `main_category_id`, `prop`

#### Use Case

These functions support hypothesis testing in Stage 10 by:
- Identifying categories with significant prevalence differences across ratings
- Visualizing distribution patterns
- Preparing statistical evidence for correlation analysis

## Output Files

### Stage 06 Exploration Outputs

1. **`metrics_paraphrase-MiniLM-L6-v2.json`**:
   - Coherence and diversity scores for all representations
   - Format: Array of objects with `representation`, `n_topics`, `coherence_c_v`, `topic_diversity`

2. **`topics_all_representations_paraphrase-MiniLM-L6-v2.json`**:
   - All topics with word lists for all representations
   - Nested structure: `{representation: {topic_id: [word_objects]}}`
   - Useful for close reading and qualitative analysis

### Stage 07 Quality Analysis Outputs

1. **`topic_quality_paraphrase-MiniLM-L6-v2.csv`**:
   - Full quality table for all topics
   - Columns: Topic, Count, Name, Representation, KeyBERT, MMR, POS, Representative_Docs, coherence_c_v_pos, n_pos_words, noise_candidate, noise_reason, inspection_label

2. **`topic_noise_candidates_paraphrase-MiniLM-L6-v2.csv`**:
   - Subset of topics flagged as noise candidates
   - Same columns as quality table, filtered to `noise_candidate == True`

3. **Model files with noise labels**:
   - `model_1_with_noise_labels/`: Native BERTopic directory
   - `model_1_with_noise_labels.pkl`: Wrapper pickle with noise labels

## Integration with Downstream Stages

### Stage 08: LLM Labeling

- **Input**: Topics with POS representation (interpretable content words)
- **Quality filtering**: Noise candidate flags help identify topics to skip or prioritize
- **Representation choice**: POS representation recommended for labeling

### Stage 09: Category Mapping

- **Input**: Topics with all representations for flexible mapping
- **Quality filtering**: Noise candidates excluded from mapping
- **Representation choice**: Multiple representations available for different mapping strategies

### Stage 10: Correlation Analysis

- **Input**: Taxonomy category proportions (from Stage 09)
- **Statistical testing**: Kruskal-Wallis functions from taxonomy analysis notebook
- **Visualization**: Box plots for category prevalence by rating class

## Technical Considerations

### Memory Efficiency

- **Batched processing**: Documents and dictionary built in batches (50K default)
- **Streaming**: Dictionary built by streaming corpus TSV, not loading entire file
- **Progress logging**: Detailed batch-level logs for monitoring

### Dataset Consistency

- **Wrapper format preferred**: Guarantees exact training dataset match
- **CSV fallback**: Available when wrapper not present
- **Normalization**: Consistent text normalization (whitespace, lowercase)

### Model Compatibility

- **Topic count validation**: Warns if metadata topic count doesn't match loaded model
- **Representation availability**: Handles missing representations gracefully
- **Label preservation**: Merges new labels with existing custom labels

## Future Directions

1. **Automated noise filtering**: Implement automatic removal of confirmed noisy topics
2. **Representation comparison**: Statistical tests comparing representation quality
3. **Interactive visualization**: Topic quality dashboard for manual inspection
4. **Quality thresholds**: Adaptive thresholds based on topic size distributions

## Summary

Stage 06 provides essential tooling for exploring and evaluating retrained BERTopic models. By computing multiple representations and quality metrics, the stage enables:

1. **Informed representation selection** for downstream analysis
2. **Quality assessment** to identify problematic topics
3. **Statistical preparation** for correlation analysis

The stage's outputs directly support topic labeling (Stage 08), category mapping (Stage 09), and correlation analysis (Stage 10), making it a critical bridge between model training and research analysis.

