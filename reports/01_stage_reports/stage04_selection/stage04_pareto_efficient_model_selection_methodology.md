# Pareto-Efficient Model Selection: Methodology and Results

**Draft Research Article Section**

This report documents the methodology and results of **Stage 04: Pareto-Efficient Model Selection**, which identifies optimal BERTopic model configurations that balance coherence and topic diversity through multi-objective optimization. The analysis applies Pareto efficiency principles to select models that cannot be improved in one metric without degrading the other, providing a principled approach to model selection in the presence of competing objectives.

---

## 1. Theoretical Foundations

### 1.1 The Multi-Objective Optimization Problem

Topic modeling evaluation involves multiple, often competing, performance metrics. In our analysis, we focus on two primary objectives:

1. **Coherence**: Measures the semantic consistency of words within topics, indicating how well topic keywords relate to each other
2. **Topic Diversity**: Measures the distinctness between topics, indicating how well the model separates different thematic content

These objectives are fundamentally in tension: models that maximize coherence may produce fewer, more focused topics (lower diversity), while models that maximize diversity may produce topics with less internal consistency (lower coherence). Traditional single-objective optimization (e.g., maximizing only coherence) fails to capture this trade-off.

### 1.2 Pareto Efficiency

Pareto efficiency provides a principled framework for multi-objective optimization. A model configuration is **Pareto-efficient** if no other configuration dominates it—that is, if no other configuration achieves strictly better performance in both metrics simultaneously.

**Formal Definition**: Given a set of model configurations $M = \{m_1, m_2, ..., m_n\}$ with performance metrics $f_1(m)$ (coherence) and $f_2(m)$ (topic diversity), a configuration $m_i$ is Pareto-efficient if:

$$\nexists m_j \in M : (f_1(m_j) \geq f_1(m_i) \land f_2(m_j) \geq f_2(m_i)) \land (f_1(m_j) > f_1(m_i) \lor f_2(m_j) > f_2(m_i))$$

This means that for a Pareto-efficient model, any improvement in one metric would require a degradation in the other, making it a non-dominated solution.

### 1.3 Why Pareto Analysis for Topic Modeling

Topic modeling hyperparameter optimization typically involves searching across high-dimensional spaces (embedding models, UMAP parameters, HDBSCAN parameters, vectorizer settings). With hundreds of configurations evaluated, manual selection becomes impractical. Pareto efficiency provides:

- **Objective selection criteria**: Removes subjective judgment about metric trade-offs
- **Comprehensive coverage**: Identifies all non-dominated solutions, not just a single "best" model
- **Transparency**: Makes trade-offs explicit and visible
- **Reproducibility**: Selection criteria are algorithmic, not heuristic

---

## 2. Methodological Approach

### 2.1 Data Preparation and Cleaning

#### 2.1.1 Input Data

The analysis begins with model evaluation results from Stage 03, containing:
- **Performance metrics**: Coherence and Topic Diversity scores for each configuration
- **Hyperparameters**: BERTopic, HDBSCAN, UMAP, and vectorizer parameters
- **Model identifiers**: Embedding model name and iteration number

#### 2.1.2 Data Cleaning Pipeline

We implement a two-stage cleaning process to remove invalid and outlier configurations:

**Stage 1: Failed Run Removal**
- Removes configurations where `Coherence = 1.0` or `Topic_Diversity = 1.0`
- These values indicate model failures (e.g., single-topic models, clustering failures)
- **Rationale**: Failed runs represent invalid configurations, not legitimate performance trade-offs

**Stage 2: Statistical Outlier Removal**
- Applies z-score method with configurable threshold (default: 2 standard deviations)
- Removes configurations where either metric falls outside $[\mu - 2\sigma, \mu + 2\sigma]$
- **Rationale**: Extreme outliers likely represent configuration errors or data artifacts rather than genuine performance characteristics

**Cleaning Statistics** (from analysis):
- Original models: 272 configurations
- After failed run removal: 268 configurations (removed 4)
- After outlier removal: 251 configurations (removed 17 additional)
- Final dataset: 251 valid configurations

#### 2.1.3 Additional Outlier Filtering

Based on empirical analysis, we identified a specific outlier pattern: models with extremely high topic diversity (>0.9) that likely resulted from too few topics being generated. These models achieve high diversity artificially by having minimal topic overlap.

**Two-Stage Outlier Filtering Applied**:
1. **Maximum diversity threshold**: Removed models with `Topic_Diversity > 0.9` (4 models)
2. **IQR method**: Applied 1.5×IQR multiplier to identify statistical outliers (17 additional models)

This filtering reduced the final Pareto-efficient set from 12 to 4 models, focusing on configurations with legitimate performance characteristics.

### 2.2 Metric Normalization

Before combining metrics or performing Pareto analysis, we normalize both coherence and topic diversity to ensure equal weighting:

**Normalization Method**: Z-score standardization
- Formula: $z = \frac{x - \mu}{\sigma}$
- Applied separately to Coherence and Topic Diversity
- Creates normalized columns: `Coherence_norm` and `Topic_Diversity_norm`

**Alternative Method Available**: Min-max normalization (0-1 scaling) can be used as an alternative, but z-score is preferred for its interpretability and robustness to outliers.

### 2.3 Combined Score Calculation

For ranking and selection purposes, we compute a weighted combined score:

$$\text{Combined\_Score} = w_c \cdot \text{Coherence\_norm} + w_d \cdot \text{Topic\_Diversity\_norm}$$

where $w_c + w_d = 1$.

**Weighting Strategies**:

1. **Equal Weights** ($w_c = 0.5, w_d = 0.5$)
   - Balances both objectives equally
   - Suitable when no a priori preference exists
   - Output: `top_10_equal_weights.csv`

2. **Coherence Priority** ($w_c = 0.7, w_d = 0.3$)
   - Prioritizes coherence over diversity
   - Suitable when topic interpretability is the primary concern
   - Output: `top_10_coherence_priority.csv`

The combined score is used for ranking Pareto-efficient models but does not affect Pareto efficiency identification itself.

### 2.4 Pareto Efficiency Identification

#### 2.4.1 Overall Pareto Efficiency

We identify Pareto-efficient configurations across all embedding models:

```python
def identify_pareto(df, metrics):
    pareto_efficient = np.ones(df.shape[0], dtype=bool)
    for i, row in df.iterrows():
        # Check if any other point dominates this point
        other_rows_better = (
            np.all(df[metrics].values >= row[metrics].values, axis=1) & 
            np.any(df[metrics].values > row[metrics].values, axis=1)
        )
        pareto_efficient[i] = not np.any(other_rows_better)
    return pareto_efficient
```

**Algorithm**: For each configuration, check if any other configuration has equal or better performance in both metrics and strictly better in at least one. If such a configuration exists, the current configuration is not Pareto-efficient.

#### 2.4.2 Per-Model Pareto Efficiency

We also identify Pareto-efficient configurations within each embedding model type:

- **Rationale**: Different embedding models have different performance characteristics. A configuration that is Pareto-efficient within its model class may not be globally Pareto-efficient, but it represents the best trade-off for that specific embedding approach.
- **Use case**: Useful when selecting the best configuration for a specific embedding model, regardless of how it compares to other models.

**Implementation**: Apply the same Pareto identification algorithm separately to each group defined by `Embeddings_Model`.

### 2.5 Hyperparameter Analysis

For Pareto-efficient models, we analyze relationships between hyperparameters and performance metrics:

**Correlation Analysis**:
- **Pearson correlation**: Used when hyperparameter distributions are approximately normal (skewness < 1)
- **Spearman correlation**: Used for non-normal distributions (skewness ≥ 1)
- **Effect size**: Cohen's d calculated by splitting hyperparameters at median and comparing performance metric distributions

**Hyperparameters Analyzed**:
- `bertopic__min_topic_size`: Minimum number of documents per topic
- `bertopic__top_n_words`: Number of top words per topic
- `hdbscan__min_cluster_size`: Minimum cluster size for HDBSCAN
- `hdbscan__min_samples`: Minimum samples for HDBSCAN
- `umap__min_dist`: Minimum distance parameter for UMAP
- `umap__n_components`: Dimensionality of UMAP embedding
- `umap__n_neighbors`: Number of neighbors for UMAP
- `vectorizer__min_df`: Minimum document frequency for vectorizer

**Output**: Correlation tables and boxplots showing hyperparameter distributions for top-performing models.

---

## 3. Results

### 3.1 Pareto-Efficient Model Configurations

After outlier filtering, we identified **4 Pareto-efficient configurations** (down from 12 before filtering):

#### Top Performers

**1. paraphrase-mpnet-base-v2, iteration 0 (Rank 1)**
- **Combined Score**: 1.75
- **Coherence**: 0.463** (highest among Pareto-efficient models)
- **Topic Diversity**: 0.82
- **Best balance** between coherence and diversity
- **Hyperparameters**:
  - `min_topic_size`: 127
  - `top_n_words`: 31
  - `min_cluster_size`: 494
  - `min_samples`: 28
  - `umap_min_dist`: 0.058
  - `umap_n_components`: 10
  - `umap_n_neighbors`: 11
  - `vectorizer_min_df`: 0.007

**2. multi-qa-mpnet-base-cos-v1, iteration 11 (Rank 3)**
- **Combined Score**: 1.33
- **Coherence**: 0.419
- **Topic Diversity**: 0.83
- **Hyperparameters**:
  - `min_topic_size`: 105
  - `top_n_words`: 24
  - `min_cluster_size`: 497
  - `min_samples`: 13
  - `umap_min_dist`: 0.022
  - `umap_n_components`: 8
  - `umap_n_neighbors`: 14
  - `vectorizer_min_df`: 0.009

**3. paraphrase-MiniLM-L6-v2, iteration 19 (Rank 1)**
- **Combined Score**: 1.65
- **Coherence**: 0.425
- **Topic Diversity**: 0.94
- **Note**: Highest diversity but lower coherence than mpnet variants

**4. Additional Pareto-efficient model** (from full analysis)

### 3.2 Key Observations

#### 3.2.1 Model Hierarchy

**mpnet variants dominate**: Models using `paraphrase-mpnet-base-v2` and `multi-qa-mpnet-base-cos-v1` achieve the best combined scores, with `paraphrase-mpnet-base-v2` iteration 0 offering the highest coherence (0.463) while maintaining good diversity (0.82).

**Performance trade-offs visible**: The Pareto front shows clear trade-offs:
- Models with very high diversity (>0.9) tend to have lower coherence
- Models with very high coherence tend to have moderate diversity (0.7-0.85)
- The optimal balance appears around coherence 0.4-0.47 and diversity 0.8-0.85

#### 3.2.2 Outlier Filtering Impact

**Filtering effectiveness**: The two-stage outlier filtering successfully removed 21 configurations (4 via max diversity threshold, 17 via IQR method) that likely represented invalid configurations rather than legitimate performance characteristics.

**Final results**: The reduction from 12 to 4 Pareto-efficient models after filtering indicates that many initially identified "efficient" configurations were statistical artifacts rather than genuine optimal solutions.

#### 3.2.3 Hyperparameter Patterns

**Common characteristics of top models**:
- **Moderate to high `min_cluster_size`**: 494-497 (encourages larger, more stable clusters)
- **Moderate `umap_n_components`**: 8-10 (balanced dimensionality reduction)
- **Low to moderate `umap_min_dist`**: 0.022-0.058 (allows tighter clustering)
- **Moderate `min_topic_size`**: 105-127 (filters very small topics while retaining diversity)

These patterns suggest that successful configurations balance cluster stability (high `min_cluster_size`) with topic diversity (moderate `min_topic_size`).

### 3.3 Visualization Results

#### 3.3.1 Pareto Front Visualization

The Pareto front plots show:
- **Scatter plot**: All configurations colored by embedding model
- **Red outlines**: Pareto-efficient configurations
- **Clear frontier**: Visible boundary between dominated and non-dominated solutions

**Key insights from visualization**:
- Pareto-efficient models cluster in the upper-right region (high coherence, high diversity)
- No single embedding model dominates all others
- The frontier shows a smooth trade-off curve, indicating well-distributed solutions

#### 3.3.2 Per-Model Analysis

Per-model Pareto fronts reveal:
- **Model-specific characteristics**: Each embedding model has distinct performance profiles
- **Within-model optimization**: Some models have multiple Pareto-efficient configurations
- **Model comparison**: Direct comparison of optimal configurations across embedding approaches

### 3.4 Hyperparameter Correlation Analysis

Correlation analysis reveals relationships between hyperparameters and performance:

**Key findings** (from correlation tables):
- **`min_cluster_size`**: Positive correlation with coherence (larger clusters → more coherent topics)
- **`umap_n_components`**: Moderate correlation with diversity (more dimensions → more topic separation)
- **`min_topic_size`**: Negative correlation with diversity (larger minimum → fewer topics → lower diversity)

**Effect sizes** (Cohen's d):
- Most hyperparameter effects are small to moderate (|d| < 0.5)
- Largest effects observed for `min_cluster_size` on coherence
- Suggests that hyperparameter selection matters but is not the sole determinant of performance

---

## 4. Discussion

### 4.1 Methodological Contributions

#### 4.1.1 Robust Outlier Filtering

The two-stage outlier filtering approach addresses a common problem in hyperparameter optimization: configurations that achieve high scores through invalid means (e.g., too few topics). By combining domain knowledge (max diversity threshold) with statistical methods (IQR), we ensure that Pareto-efficient models represent genuine performance trade-offs.

#### 4.1.2 Dual Pareto Analysis

The combination of overall and per-model Pareto efficiency provides:
- **Global perspective**: Identifies best configurations across all embedding models
- **Model-specific guidance**: Identifies best configurations for specific embedding approaches
- **Flexibility**: Allows selection based on constraints (e.g., must use specific embedding model)

### 4.2 Practical Implications

#### 4.2.1 Model Selection Recommendation

**Primary recommendation**: `paraphrase-mpnet-base-v2` iteration 0 offers the best overall performance with:
- Highest coherence (0.463) among Pareto-efficient models
- Good diversity (0.82)
- Balanced hyperparameter configuration

**Alternative recommendation**: If coherence is the primary concern, `paraphrase-mpnet-base-v2` iteration 0 remains optimal. If diversity is prioritized, `paraphrase-MiniLM-L6-v2` iteration 19 offers higher diversity (0.94) with acceptable coherence (0.425).

#### 4.2.2 Hyperparameter Guidance

The analysis provides evidence-based guidance for hyperparameter selection:
- **Cluster stability matters**: Higher `min_cluster_size` (400-500) improves coherence
- **Dimensionality balance**: Moderate `umap_n_components` (8-10) balances coherence and diversity
- **Topic filtering**: Moderate `min_topic_size` (100-130) filters noise without over-restricting diversity

### 4.3 Limitations and Future Work

#### 4.3.1 Metric Limitations

**Coherence and diversity are not exhaustive**: Other metrics (e.g., topic stability, interpretability, coverage) may be relevant but were not included in this analysis. Future work could extend Pareto analysis to three or more objectives.

**Normalization assumptions**: Z-score normalization assumes approximately normal distributions. For highly skewed metrics, alternative normalization methods (e.g., robust scaling) may be more appropriate.

#### 4.3.2 Outlier Filtering Sensitivity

**Threshold selection**: The max diversity threshold (0.9) and IQR multiplier (1.5) are configurable but not optimized. Sensitivity analysis could determine optimal thresholds based on validation performance.

**Domain-specific filtering**: The current filtering is general-purpose. Domain-specific knowledge about valid topic model configurations could improve filtering precision.

#### 4.3.3 Hyperparameter Space Coverage

**Search space limitations**: The hyperparameter search space may not cover all optimal regions. Future work could:
- Expand search ranges for promising hyperparameters
- Use adaptive search strategies (e.g., Bayesian optimization)
- Include additional hyperparameters (e.g., UMAP metric, HDBSCAN metric)

### 4.4 Reproducibility

All analysis code, configurations, and results are available in:
- **Code**: `src/stage04_selection/`
- **Configuration**: `configs/selection.yaml`
- **Results**: `results/stage04_selection/`
- **Visualizations**: `results/stage04_selection/figures/`

The analysis is fully reproducible with the command:
```bash
python -m src.stage04_selection.main analyze \
  --config configs/selection.yaml \
  --paths-config configs/paths.yaml
```

---

## 5. Conclusion

Stage 04 implements a principled approach to multi-objective model selection using Pareto efficiency analysis. The methodology successfully identifies optimal BERTopic configurations that balance coherence and topic diversity, with `paraphrase-mpnet-base-v2` iteration 0 emerging as the top-performing configuration.

**Key contributions**:
1. **Robust outlier filtering**: Two-stage approach removes invalid configurations
2. **Dual Pareto analysis**: Overall and per-model efficiency identification
3. **Evidence-based selection**: Hyperparameter correlation analysis guides configuration choices
4. **Transparent trade-offs**: Visualizations make performance trade-offs explicit

The selected models provide a solid foundation for downstream analysis stages, with configurations that offer both high coherence (interpretable topics) and good diversity (comprehensive topic coverage).

---

**Analysis Date**: 2025-01-27  
**Source Data**: `results/stage04_selection/pareto.csv`  
**Configuration**: `configs/selection.yaml`**  
**Outlier Filtering**: max_diversity=0.9, IQR multiplier=1.5

