# Stage 04: Pareto-Efficient Model Selection
## Presentation Summary

**Key Points for Research Presentation**

---

## Slide 1: Problem Statement

### The Multi-Objective Challenge

**Topic modeling requires balancing competing objectives:**

- **Coherence** → Interpretable, semantically consistent topics
- **Topic Diversity** → Comprehensive thematic coverage

**The Trade-off:**
- Maximizing coherence → Fewer, more focused topics (lower diversity)
- Maximizing diversity → More topics with less consistency (lower coherence)

**Solution:** Pareto-efficient multi-objective optimization

---

## Slide 2: Methodology Overview

### Three-Stage Approach

1. **Data Cleaning**
   - Removed failed runs (Coherence/Diversity = 1.0)
   - Statistical outlier removal (z-score, 2σ threshold)
   - Domain-specific filtering (diversity > 0.9, IQR method)
   - **Result:** 251 valid configurations from 272 initial

2. **Pareto Efficiency Identification**
   - Identifies non-dominated solutions
   - No other configuration is better in BOTH metrics
   - Overall + per-model analysis

3. **Hyperparameter Correlation Analysis**
   - Statistical correlation (Pearson/Spearman)
   - Linear regression with assumption checks
   - Tree-based feature importance (Random Forest, XGBoost)

---

## Slide 3: Key Results - Top Models

### Pareto-Efficient Configurations (4 models)

<p align="center">
  <img src="../../results/stage04_selection/pareto_frontier_all.png" alt="Pareto Frontier" width="700">
</p>

**Top Performer: paraphrase-mpnet-base-v2, iteration 0**

| Metric | Value |
|--------|-------|
| **Combined Score** | 1.75 |
| **Coherence** | **0.463** (highest) |
| **Topic Diversity** | 0.82 |
| **Status** | Best balance |

**Key Hyperparameters:**
- `min_cluster_size`: 494
- `umap_n_components`: 10
- `umap_min_dist`: 0.058
- `vectorizer_min_df`: 0.007

---

## Slide 4: Performance Visualization

### Pareto Frontier Analysis

<p align="center">
  <img src="../../results/stage04_selection/figures/pareto_front_equal_weights.png" alt="Pareto Front Equal Weights" width="650">
</p>

**Key Observations:**
- Pareto-efficient models cluster in upper-right (high coherence + high diversity)
- Clear trade-off curve visible
- No single embedding model dominates all others
- Optimal balance: Coherence 0.4-0.47, Diversity 0.8-0.85

---

## Slide 5: Model Comparison

### Per-Model Pareto Frontiers

<p align="center">
  <img src="../../results/stage04_selection/figures/pareto_fronts_per_model.png" alt="Pareto Fronts Per Model" width="700">
</p>

**Model Hierarchy:**
- **mpnet variants** dominate combined scores
- **paraphrase-mpnet-base-v2**: Highest coherence (0.463)
- **paraphrase-MiniLM-L6-v2**: Highest diversity (0.94) but lower coherence

---

## Slide 6: Critical Hyperparameters

### Statistical Analysis Results

**High Priority Parameters** (Strong, Significant Effects):

| Parameter | Effect on | Correlation | p-value | Effect Size |
|----------|-----------|-------------|---------|-------------|
| `umap__min_dist` | Diversity | **r = 0.794** | **p = 0.006** | Large (d = -1.10) |
| `vectorizer__min_df` | Combined Score | **r = 0.745** | **p = 0.014** | Very Large (d = -2.56) |
| `umap__n_components` | Combined Score | **r = 0.712** | **p = 0.021** | Large (d = -1.46) |

**Medium Priority Parameters** (Strong Effects, Marginal Significance):

| Parameter | Effect on | Correlation | p-value |
|----------|-----------|-------------|---------|
| `bertopic__min_topic_size` | Coherence | r = 0.585 | p = 0.075 |
| `hdbscan__min_cluster_size` | Coherence | r = 0.563 | p = 0.090 |
| `umap__n_neighbors` | Coherence | r = -0.573 | p = 0.084 |

---

## Slide 7: The Coherence-Diversity Trade-off

### Opposing Parameter Effects

<p align="center">
  <img src="../../results/stage04_selection/figures/hyperparameter_boxplots.png" alt="Hyperparameter Distributions" width="700">
</p>

**Key Trade-offs Identified:**

1. **`bertopic__min_topic_size`**
   - ✅ Increases coherence (r = 0.585)
   - ❌ Decreases diversity (r = -0.591)
   - **Trade-off:** Filter small topics → better coherence, fewer topics

2. **`umap__min_dist`**
   - ❌ Decreases coherence (r = -0.528)
   - ✅ Increases diversity (r = 0.794, **significant**)
   - **Trade-off:** Tighter clustering → better coherence, less separation

**Synergistic Effect:**
- **`hdbscan__min_cluster_size`**: Primarily benefits coherence with minimal diversity cost

---

## Slide 8: Optimization Recommendations

### Evidence-Based Guidance

**For Maximizing Coherence:**
- ⬆️ Increase `min_topic_size` (100-130)
- ⬆️ Increase `min_cluster_size` (400-500)
- ⬇️ Decrease `umap_n_neighbors`
- ⬇️ Decrease `umap_min_dist`

**For Maximizing Diversity:**
- ⬆️ Increase `umap_min_dist` (0.02-0.08)
- ⬆️ Increase `vectorizer_min_df` (0.007-0.009)
- ⬇️ Decrease `min_topic_size`

**For Balanced Performance:**
- Optimize `umap_min_dist` (critical trade-off parameter)
- Optimize `vectorizer_min_df` (strong combined score effect)
- Set `umap_n_components` to 8-10

---

## Slide 9: Distribution Analysis

### Outlier Filtering Impact

<p align="center">
  <img src="../../results/stage04_selection/figures/distribution_with_cutoffs.png" alt="Distribution with Cutoffs" width="650">
</p>

**Filtering Results:**
- Original: 272 configurations
- After failed run removal: 268 (removed 4)
- After outlier removal: 251 (removed 17)
- **Final Pareto-efficient: 4 models** (down from 12 before filtering)

**Why Filtering Matters:**
- Removed models with artificially high diversity (>0.9) from too few topics
- Ensured Pareto-efficient models represent genuine performance trade-offs
- Statistical artifacts removed, focus on legitimate optimal solutions

---

## Slide 10: Feature Importance Validation

### Multi-Method Agreement

**Tree-Based Methods Confirm Correlation Findings:**

**Random Forest & XGBoost Feature Importance:**
1. `umap__min_dist` (highest for diversity/combined)
2. `vectorizer__min_df` (high for diversity/combined)
3. `hdbscan__min_cluster_size` (high for coherence)
4. `bertopic__min_topic_size` (moderate for coherence)
5. `umap__n_components` (moderate for combined)

**Cross-Validated R²:** 0.3-0.6 (moderate predictive power)

**Key Insight:** Multiple independent methods identify the same critical parameters → **Robust findings**

---

## Slide 11: Final Recommendations

### Model Selection

**Primary Recommendation:**
- **Model:** `paraphrase-mpnet-base-v2`, iteration 0
- **Rationale:** Highest coherence (0.463) with good diversity (0.82)
- **Best for:** General-purpose topic modeling requiring interpretable topics

**Alternative Options:**
- **High Diversity:** `paraphrase-MiniLM-L6-v2`, iteration 19 (diversity = 0.94)
- **High Coherence:** `paraphrase-mpnet-base-v2`, iteration 0 (coherence = 0.463)

### Hyperparameter Tuning Priorities

**Focus optimization on:**
1. `umap__min_dist` (most critical trade-off parameter)
2. `vectorizer__min_df` (strongest combined score effect)
3. `umap__n_components` (important for balanced performance)

**Lower priority:**
- `bertopic__top_n_words` (minimal effect)
- `hdbscan__min_samples` (weak effects)

---

## Slide 12: Methodological Contributions

### Key Strengths

1. **Assumption-Driven Analysis**
   - Explicit statistical assumption checking
   - Robust alternatives when assumptions violated
   - Valid inference ensured

2. **Multi-Method Validation**
   - Correlation analysis
   - Linear regression
   - Tree-based feature importance
   - Convergent validity across methods

3. **Robust Outlier Filtering**
   - Domain knowledge + statistical methods
   - Removes invalid configurations
   - Ensures genuine performance trade-offs

4. **Transparent Trade-offs**
   - Visualizations make trade-offs explicit
   - Reproducible selection criteria
   - Objective, algorithmic approach

---

## Slide 13: Limitations & Future Work

### Current Limitations

1. **Sample Size:** 251 configurations may limit statistical power
2. **Non-Linear Effects:** Linear methods may miss complex relationships
3. **Interactions:** Parameter interactions not explicitly modeled
4. **Model Heterogeneity:** Effects may vary by embedding model

### Future Directions

- **Expand search space** for promising hyperparameters
- **Bayesian optimization** for adaptive search
- **Interaction terms** in regression models
- **Stratified analysis** by embedding model type
- **Additional metrics** (stability, interpretability, coverage)

---

## Slide 14: Summary

### Key Takeaways

✅ **Pareto efficiency** provides principled multi-objective optimization

✅ **4 Pareto-efficient models** identified after robust filtering

✅ **Top performer:** `paraphrase-mpnet-base-v2` iteration 0
   - Coherence: 0.463 (highest)
   - Diversity: 0.82 (good)
   - Best overall balance

✅ **Critical hyperparameters identified:**
   - `umap__min_dist` (diversity)
   - `vectorizer__min_df` (combined score)
   - `umap__n_components` (combined score)

✅ **Trade-offs quantified:**
   - Coherence vs. diversity trade-offs explicit
   - Evidence-based optimization guidance
   - Parameter range recommendations

**Ready for downstream analysis with optimal configurations!**

---

**Analysis Date:** 2025-01-27  
**Source:** Stage 03 hyperparameter optimization results  
**Results:** `results/stage04_selection/`  
**Code:** `src/stage04_selection/`

