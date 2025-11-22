### Pareto Analysis Results

Short summary of Pareto-efficient model configurations from hyperparameter optimization.

---

#### Top Performers

**1. paraphrase-mpnet-base-v2 (Rank 1)**
- **Combined Score:** 1.75
- **Coherence:** 0.463 (highest)
- **Topic Diversity:** 0.82
- **Best balance** between coherence and diversity

**2. multi-qa-mpnet-base-cos-v1, iteration 26 (Rank 3)**
- **Combined Score:** 1.66
- **Coherence:** 0.402
- **Topic Diversity:** 0.90 (higher diversity, lower coherence)

**3. multi-qa-mpnet-base-cos-v1, iteration 11 (Rank 5)**
- **Combined Score:** 1.33
- **Coherence:** 0.419
- **Topic Diversity:** 0.83

---

#### Key Observations

- **Trade-off pattern:** mpnet models favor coherence; MiniLM models achieve higher diversity (0.92-0.93) but lower coherence (0.28-0.32)
- **All 12 configurations are Pareto-efficient** (both globally and per-model)
- **Model hierarchy:** mpnet variants outperform MiniLM variants in combined score
- **Recommendation:** `paraphrase-mpnet-base-v2` iteration 0 offers the best overall performance

---

#### Configuration Details

Top configuration (paraphrase-mpnet-base-v2):
- `min_topic_size`: 127
- `top_n_words`: 31
- `min_cluster_size`: 494
- `min_samples`: 28
- `umap_min_dist`: 0.058
- `umap_n_components`: 10
- `umap_n_neighbors`: 11
- `vectorizer_min_df`: 0.007

---

*Analysis date: 2025-01-27*  
*Source: `results/pareto/pareto.csv`*

