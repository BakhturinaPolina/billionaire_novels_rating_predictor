# Stage 03: Modeling — Methodology Report

## Purpose

Train BERTopic models with OCTIS hyperparameter optimization, using GPU-accelerated topic modeling to extract thematic patterns from romance novels.

## Research Rationale

### Why BERTopic?

Unlike traditional LDA, BERTopic uses BERT embeddings to capture contextual word meanings, producing more interpretable topics for literary analysis. This enables identification of semantic relationships beyond simple word co-occurrence.

### Why Character Name Exclusion?

Character names present a unique challenge: when they appear frequently, they can dominate topic word distributions, creating topics that reflect character co-occurrence rather than thematic relationships.

**Impact**:
- **Before exclusion**: Topics dominated by names (e.g., "Alex, Stella, Weston, love")
- **After exclusion**: Topics focus on themes (e.g., "love, relationship, emotion, connection")

This aligns with computational literary analysis practices where character names are treated as structural elements rather than semantic content (Bamman et al., 2013; Jockers, 2013).

---

## Character Name Exclusion

### Processing Statistics

| Metric | Value |
|--------|-------|
| Lines processed | 7,525 |
| Lines filtered | 254 (3.4%) |
| Valid name lines | 7,271 |
| Multi-word names | 3,313 |
| Unique tokens | 4,497 |
| Final stopwords | 4,444 |

### Stoplist Composition

| Component | Count | Percentage |
|-----------|-------|------------|
| Character names | 4,444 | 93% |
| Standard English | 318 | 7% |
| **Total** | **4,762** | 100% |

This represents a **14× expansion** over standard English stopword lists.

### Processing Pipeline

1. **Cleaning**: Remove prefixes ("Mr.", "Miss"), numbers, quotes, punctuation
2. **Filtering**: Remove long lines (>50 chars), common phrases, descriptive patterns
3. **Extraction**: Split multi-word names ("Alex Crane" → "alex", "crane")

### Precision Trade-off

The pipeline retains some non-name words (estimated 1–2%). We prioritize **coverage over precision** to ensure comprehensive character name exclusion. False positives have minimal impact as they appear infrequently.

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

## References

- Bamman, D., Underwood, T., & Smith, N. A. (2013). A Bayesian Mixed Effects Model of Literary Character. *ACL*.
- Jockers, M. L. (2013). *Macroanalysis: Digital Methods and Literary History*. University of Illinois Press.
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv:2203.05794*.
