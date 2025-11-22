### Pareto Analysis Results

Short summary of Pareto-efficient model configurations from hyperparameter optimization.

**Note:** Results filtered to remove outliers (models with too few topics that had artificially high diversity scores).

---

#### Top Performers (After Outlier Filtering)

**1. paraphrase-mpnet-base-v2, iteration 0 (Rank 1)**
- **Combined Score:** 1.75
- **Coherence:** 0.463 (highest)
- **Topic Diversity:** 0.82
- **Best balance** between coherence and diversity

**2. multi-qa-mpnet-base-cos-v1, iteration 11 (Rank 3)**
- **Combined Score:** 1.33
- **Coherence:** 0.419
- **Topic Diversity:** 0.83

---

#### Key Observations

- **Outlier filtering applied:** Removed 21 models (4 with diversity >0.9, 17 via IQR method) that likely had too few topics
- **Final results:** 4 Pareto-efficient models (down from 12 before filtering)
- **Model hierarchy:** mpnet variants outperform other models in combined score
- **Recommendation:** `paraphrase-mpnet-base-v2` iteration 0 offers the best overall performance with highest coherence (0.463) and good diversity (0.82)

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

#### Outlier Filtering Details

Applied two-stage outlier filtering:
1. **Max diversity threshold:** Removed models with diversity > 0.9 (likely too few topics)
2. **IQR method:** Removed statistical outliers using 1.5×IQR multiplier

**Filtering results:**
- Original models: 272
- After max_diversity filter: 268 (removed 4)
- After IQR filter: 251 (removed 17 additional)
- Final Pareto-efficient: 4 models

---

*Analysis date: 2025-01-27*  
*Source: `results/pareto/pareto.csv`*  
*Outlier filtering: max_diversity=0.9, IQR multiplier=1.5*

