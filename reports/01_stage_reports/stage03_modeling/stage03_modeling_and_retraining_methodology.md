# Stage 03: Modeling — Methodology Report

## Purpose

Train BERTopic models with OCTIS hyperparameter optimization, using GPU-accelerated topic modeling to extract thematic patterns from romance novels.

## Research Rationale

### Why BERTopic?

Unlike traditional LDA, BERTopic uses BERT embeddings to capture contextual word meanings, producing more interpretable topics for literary analysis. This enables identification of semantic relationships beyond simple word co-occurrence.

**Character Name Exclusion**: Stage 03 uses the custom stoplist from Stage 02 preprocessing, which excludes 4,444 character names to ensure topics focus on thematic content rather than character co-occurrence. See [Stage 02 methodology](../stage02_preprocessing/stage02_preprocessing_methodology.md) for details.

---

## Model Training

### Hyperparameter Optimization

**OCTIS Framework**: Bayesian optimization across:
- 6 embedding models (SentenceTransformers)
- UMAP, HDBSCAN, and vectorizer parameters
- 300+ configurations evaluated

### GPU Acceleration

**RAPIDS cuML** (CUDA 12.x) for:
- UMAP dimensionality reduction
- HDBSCAN clustering

No CPU fallback — GPU required.

### Embedding Caching

Embeddings are cached to avoid recomputation across training iterations, significantly reducing processing time.

---

## Integration with Pipeline

| Stage | Relationship |
|-------|--------------|
| ← Stage 02 | Receives preprocessed text (`chapters.csv`) |
| → Stage 04 | Provides models for Pareto efficiency analysis |
| → Stage 05 | Top models retrained with optimal hyperparameters |

### Typical Workflow

1. **Stage 03**: Train BERTopic models with OCTIS optimization
2. **Stage 04**: Identify Pareto-efficient models
3. **Stage 05**: Retrain top models for final deployment

---
