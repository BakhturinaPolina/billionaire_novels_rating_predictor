# Stage 04: Pareto-Efficient Model Selection and Hyperparameter Analysis

**Research Article Draft Section**

This document synthesizes the methodology and findings from Stage 04, which implements a principled multi-objective optimization approach to identify optimal BERTopic configurations for billionaire romance novel analysis. The analysis combines Pareto efficiency principles with comprehensive statistical hyperparameter analysis to guide model selection.

---

## 1. Introduction

Topic modeling for large-scale text corpora requires balancing multiple, often competing, performance objectives. In our analysis of billionaire romance novels, we focus on two critical metrics:

- **Coherence**:** Measures semantic consistency of words within topics, indicating topic interpretability
- **Topic Diversity**: Measures distinctness between topics, indicating comprehensive thematic coverage

These objectives are fundamentally in tension: maximizing coherence may reduce diversity (fewer, more focused topics), while maximizing diversity may reduce coherence (more topics with less internal consistency). Traditional single-objective optimization fails to capture this trade-off, necessitating a multi-objective approach.

---

## 2. Methodology

### 2.1 Data Preparation

**Source Data**: Model evaluation results from Stage 03 hyperparameter optimization
- **Total Configurations Evaluated**: 272 initial configurations
- **Embedding Models**: Multiple variants (paraphrase-mpnet-base-v2, multi-qa-mpnet-base-cos-v1, paraphrase-MiniLM-L6-v2, etc.)
- **Hyperparameters**: 8 parameters across BERTopic, HDBSCAN, UMAP, and vectorizer components

**Data Cleaning Pipeline**:

1. **Failed Run Removal**: Removed configurations with `Coherence = 1.0` or `Topic_Diversity = 1.0` (indicating model failures)
   - Removed: 4 configurations

2. **Statistical Outlier Removal**: Applied z-score method (threshold: 2 standard deviations)
   - Removed: 17 additional configurations

3. **Domain-Specific Outlier Filtering**: 
   - Maximum diversity threshold: Removed models with `Topic_Diversity > 0.9` (likely too few topics)
   - IQR method: Applied 1.5×IQR multiplier for statistical outliers
   - Removed: 21 total configurations

**Final Dataset**: 251 valid configurations for analysis

### 2.2 Metric Normalization and Combination

**Z-Score Normalization**:
- Applied separately to Coherence and Topic Diversity
- Formula: $z = \frac{x - \mu}{\sigma}$
- Creates normalized columns: `Coherence_norm` and `Topic_Diversity_norm`

**Combined Score Calculation**:
- **Equal Weights**: $0.5 \cdot \text{Coherence\_norm} + 0.5 \cdot \text{Topic\_Diversity\_norm}$
- **Coherence Priority**: $0.7 \cdot \text{Coherence\_norm} + 0.3 \cdot \text{Topic\_Diversity\_norm}$

### 2.3 Pareto Efficiency Identification

**Formal Definition**: A model configuration is Pareto-efficient if no other configuration dominates it—i.e., no other configuration achieves strictly better performance in both metrics simultaneously.

**Algorithm**: For each configuration, check if any other configuration has equal or better performance in both metrics and strictly better in at least one. If such a configuration exists, the current configuration is not Pareto-efficient.

**Analysis Types**:
- **Overall Pareto Efficiency**: Identifies non-dominated solutions across all embedding models
- **Per-Model Pareto Efficiency**: Identifies best configurations within each embedding model type

### 2.4 Hyperparameter Correlation Analysis

**Statistical Methods**:

1. **Correlation Analysis**:
   - **Normality Testing**: Shapiro-Wilk test (α = 0.05)
   - **Test Selection**: Pearson correlation for normal distributions, Spearman for non-normal
   - **Effect Size**: Cohen's d calculated by splitting hyperparameters at median

2. **Linear Regression Analysis**:
   - Multiple linear regression: $y = \beta_0 + \beta_1 x_1 + ... + \beta_8 x_8 + \epsilon$
   - **Assumption Checks**: Homoscedasticity (Breusch-Pagan), normality of residuals (Shapiro-Wilk), linearity (residual plots)

3. **Multicollinearity Assessment**:
   - Variance Inflation Factor (VIF) analysis
   - Interpretation: VIF < 5 (low), 5-10 (moderate), >10 (high)

4. **Tree-Based Feature Importance**:
   - Random Forest and XGBoost regression
   - 5-fold cross-validation with R² scoring
   - Non-parametric alternative when regression assumptions are violated

**Hyperparameters Analyzed** (8 total):
- `bertopic__min_topic_size`: Minimum documents per topic
- `bertopic__top_n_words`: Number of top words per topic
- `hdbscan__min_cluster_size`: Minimum cluster size for HDBSCAN
- `hdbscan__min_samples`: Minimum samples for HDBSCAN stability
- `umap__min_dist`: Minimum distance parameter for UMAP
- `umap__n_components`: Dimensionality of UMAP embedding
- `umap__n_neighbors`: Number of neighbors for UMAP
- `vectorizer__min_df`: Minimum document frequency for vocabulary filtering

---

## 3. Results

### 3.1 Pareto-Efficient Model Configurations

After comprehensive outlier filtering, we identified **4 Pareto-efficient configurations**:

#### Top Performers

**1. paraphrase-mpnet-base-v2, iteration 0 (Rank 1)**
- **Combined Score (Equal Weights)**: 1.75
- **Coherence**: 0.463 (highest among Pareto-efficient models)
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
- **Topic Diversity**: 0.94 (highest diversity)
- **Note**: Highest diversity but lower coherence than mpnet variants

**4. Additional Pareto-efficient model** (from full analysis)

![Pareto Frontier - All Models](../../results/stage04_selection/pareto_frontier_all.png)

*Figure 1: Pareto frontier showing all model configurations. Red outlines indicate Pareto-efficient models.*

![Pareto Frontier by Model](../../results/stage04_selection/pareto_frontier_by_model.png)

*Figure 2: Pareto frontier separated by embedding model type, showing model-specific performance characteristics.*

![Combined Score Distribution](../../results/stage04_selection/combined_score_distribution.png)

*Figure 3: Distribution of combined scores across all configurations, showing the performance landscape.*

### 3.2 Hyperparameter Correlation Analysis Results

#### 3.2.1 Coherence

**Strong Positive Correlations** (|r| > 0.5):
- **`bertopic__min_topic_size`**: r = 0.585, p = 0.075 (marginal significance)
  - **Cohen's d**: -1.18 (large effect)
  - **Interpretation**: Larger minimum topic size → higher coherence
  - **Practical**: Filtering small topics improves topic quality

- **`hdbscan__min_cluster_size`**: r = 0.563, p = 0.090 (marginal significance)
  - **Cohen's d**: -1.06 (large effect)
  - **Interpretation**: Larger clusters → more coherent topics
  - **Practical**: Encouraging larger, more stable clusters improves coherence

**Strong Negative Correlations**:
- **`umap__n_neighbors`**: r = -0.573, p = 0.084 (marginal significance)
  - **Cohen's d**: 0.88 (large effect)
  - **Interpretation**: Fewer neighbors → higher coherence
  - **Practical**: Tighter local structure in UMAP improves topic consistency

- **`umap__min_dist`**: r = -0.528, p = 0.117 (not significant)
  - **Cohen's d**: 0.41 (small-medium effect)
  - **Interpretation**: Lower minimum distance → higher coherence

#### 3.2.2 Topic Diversity

**Strong Positive Correlations**:
- **`umap__min_dist`**: r = 0.794, **p = 0.006** (significant)
  - **Cohen's d**: -1.10 (large effect)
  - **Interpretation**: Higher minimum distance → higher diversity
  - **Practical**: Spreading out clusters in embedding space increases topic separation

- **`vectorizer__min_df`**: r = 0.620, p = 0.056 (marginal significance)
  - **Cohen's d**: -1.07 (large effect)
  - **Interpretation**: Higher minimum document frequency → higher diversity
  - **Practical**: Filtering rare words increases topic distinctness

**Strong Negative Correlations**:
- **`bertopic__min_topic_size`**: r = -0.591, p = 0.072 (marginal significance)
  - **Cohen's d**: 1.62 (very large effect)
  - **Interpretation**: Larger minimum topic size → lower diversity
  - **Practical**: Filtering small topics reduces total topic count, decreasing diversity

#### 3.2.3 Combined Score (Equal Weights)

**Strong Positive Correlations**:
- **`vectorizer__min_df`**: r = 0.745, **p = 0.014** (significant)
  - **Cohen's d**: -2.56 (very large effect)
  - **Interpretation**: Higher minimum document frequency → better combined score
  - **Practical**: Vocabulary filtering is crucial for overall performance

- **`umap__n_components`**: r = 0.712, **p = 0.021** (significant)
  - **Cohen's d**: -1.46 (large effect)
  - **Interpretation**: More UMAP dimensions → better combined score
  - **Practical**: Preserving more dimensionality improves performance

- **`umap__min_dist`**: r = 0.663, **p = 0.037** (significant)
  - **Cohen's d**: -1.75 (very large effect)
  - **Interpretation**: Higher minimum distance → better combined score
  - **Practical**: Balancing cluster tightness and separation optimizes performance

![Hyperparameter Boxplots](../../results/stage04_selection/figures/hyperparameter_boxplots.png)

*Figure 4: Distribution of hyperparameter values for top-performing models, showing optimal parameter ranges.*

![Distribution with Cutoffs](../../results/stage04_selection/figures/distribution_with_cutoffs.png)

*Figure 5: Distribution of performance metrics with outlier filtering thresholds indicated.*

### 3.3 Key Patterns and Trade-offs

#### 3.3.1 Coherence vs. Diversity Trade-off

**Opposing Effects**:
- **`bertopic__min_topic_size`**: 
  - Positive effect on coherence (r = 0.585)
  - Negative effect on diversity (r = -0.591)
  - **Trade-off**: Larger minimum topic size improves coherence but reduces diversity

- **`umap__min_dist`**:
  - Negative effect on coherence (r = -0.528)
  - Positive effect on diversity (r = 0.794, significant)
  - **Trade-off**: Lower minimum distance improves coherence but reduces diversity

**Synergistic Effects**:
- **`hdbscan__min_cluster_size`**: 
  - Positive effect on coherence (r = 0.563)
  - Weak negative effect on diversity (r = -0.451)
  - **Interpretation**: Primarily benefits coherence with minimal diversity cost

#### 3.3.2 UMAP Parameter Effects

**UMAP parameters show strong, significant effects**:
- **`umap__min_dist`**: Most influential for diversity (r = 0.794, p = 0.006)
- **`umap__n_components`**: Important for combined score (r = 0.712, p = 0.021)
- **`umap__n_neighbors`**: Important for coherence (r = -0.573, p = 0.084)

**Interpretation**: UMAP dimensionality reduction parameters are critical for balancing coherence and diversity.

#### 3.3.3 Vectorizer Parameter Effects

**`vectorizer__min_df` shows strong effects**:
- Strong positive effect on diversity (r = 0.620, p = 0.056)
- Strong positive effect on combined score (r = 0.745, p = 0.014)
- Weak negative effect on coherence (r = -0.302, p = 0.396)

**Interpretation**: Vocabulary filtering is crucial for overall performance, primarily through diversity improvements.

![Pareto Front - Equal Weights](../../results/stage04_selection/figures/pareto_front_equal_weights.png)

*Figure 6: Pareto frontier visualization with equal-weight combined score ranking.*

![Pareto Front - Coherence Priority](../../results/stage04_selection/figures/pareto_front_coherence_priority.png)

*Figure 7: Pareto frontier visualization with coherence-priority combined score ranking.*

![Pareto Fronts Per Model](../../results/stage04_selection/figures/pareto_fronts_per_model.png)

*Figure 8: Per-model Pareto frontiers, showing optimal configurations for each embedding model type.*

### 3.4 Tree-Based Feature Importance

**Random Forest and XGBoost Results**:
- **Cross-Validated R² Scores**: 0.3-0.6 (moderate predictive power)
- **Feature Importance Rankings** (consistent across methods):
  1. **`umap__min_dist`**: Highest importance for diversity and combined scores
  2. **`vectorizer__min_df`**: High importance for diversity and combined scores
  3. **`hdbscan__min_cluster_size`**: High importance for coherence
  4. **`bertopic__min_topic_size`**: Moderate importance for coherence
  5. **`umap__n_components`**: Moderate importance for combined scores

**Agreement**: Both tree-based methods identify similar hyperparameters as most important, providing robust evidence for parameter effects.

---

## 4. Discussion

### 4.1 Methodological Contributions

#### 4.1.1 Assumption-Driven Analysis

The explicit checking of statistical assumptions (normality, homoscedasticity, multicollinearity) ensures valid inference. When assumptions are violated, the methodology employs robust alternatives (Spearman correlation, tree-based models) rather than proceeding with invalid parametric tests.

#### 4.1.2 Multi-Method Validation

The combination of correlation analysis, regression, and tree-based methods provides:
- **Convergent validity**: When multiple methods agree, findings are robust
- **Complementary insights**: Different methods reveal different aspects (linear vs. non-linear relationships)
- **Robustness**: Assumption violations in one method don't invalidate all findings

#### 4.1.3 Robust Outlier Filtering

The two-stage outlier filtering approach (domain knowledge + statistical methods) addresses configurations that achieve high scores through invalid means (e.g., too few topics), ensuring Pareto-efficient models represent genuine performance trade-offs.

### 4.2 Practical Implications

#### 4.2.1 Model Selection Recommendation

**Primary recommendation**: `paraphrase-mpnet-base-v2` iteration 0 offers the best overall performance with:
- Highest coherence (0.463) among Pareto-efficient models
- Good diversity (0.82)
- Balanced hyperparameter configuration

**Alternative recommendation**: If coherence is the primary concern, `paraphrase-mpnet-base-v2` iteration 0 remains optimal. If diversity is prioritized, `paraphrase-MiniLM-L6-v2` iteration 19 offers higher diversity (0.94) with acceptable coherence (0.425).

#### 4.2.2 Hyperparameter Tuning Priorities

**High Priority** (strong, significant effects):
1. **`umap__min_dist`**: Critical for diversity (r = 0.794, p = 0.006)
2. **`vectorizer__min_df`**: Critical for combined score (r = 0.745, p = 0.014)
3. **`umap__n_components`**: Important for combined score (r = 0.712, p = 0.021)

**Medium Priority** (strong effects, marginal significance):
1. **`bertopic__min_topic_size`**: Important for coherence (r = 0.585, p = 0.075)
2. **`hdbscan__min_cluster_size`**: Important for coherence (r = 0.563, p = 0.090)
3. **`umap__n_neighbors`**: Important for coherence (r = -0.573, p = 0.084)

**Low Priority** (weak or non-significant effects):
- **`bertopic__top_n_words`**: Minimal effect across metrics
- **`hdbscan__min_samples`**: Weak effects

#### 4.2.3 Optimization Strategy Recommendations

**For Maximizing Coherence**:
- Increase `bertopic__min_topic_size` (filter small topics)
- Increase `hdbscan__min_cluster_size` (encourage larger clusters)
- Decrease `umap__n_neighbors` (tighter local structure)
- Decrease `umap__min_dist` (allow tighter clustering)

**For Maximizing Diversity**:
- Increase `umap__min_dist` (spread out clusters)
- Increase `vectorizer__min_df` (filter rare words)
- Decrease `bertopic__min_topic_size` (allow more topics)

**For Balanced Performance**:
- Optimize `umap__min_dist` (balance coherence-diversity trade-off)
- Optimize `vectorizer__min_df` (strong effect on combined score)
- Set `umap__n_components` to moderate-high values (8-10)

#### 4.2.4 Parameter Range Guidance

Based on correlation patterns and top-performing configurations:

- **`umap__min_dist`**: 0.02-0.08 (balance coherence and diversity)
- **`umap__n_components`**: 8-10 (optimal for combined score)
- **`vectorizer__min_df`**: 0.007-0.009 (strong effect on diversity)
- **`hdbscan__min_cluster_size`**: 400-500 (improves coherence)
- **`bertopic__min_topic_size`**: 100-130 (balance coherence-diversity trade-off)

### 4.3 Limitations and Future Work

#### 4.3.1 Sample Size Limitations

- 251-303 configurations may limit statistical power
- Marginal significance (p ≈ 0.05-0.10) may reflect insufficient power rather than weak effects
- **Future work**: Collect more configurations or use Bayesian methods for small-sample inference

#### 4.3.2 Non-Linear Relationships

- Linear methods may miss complex effects
- Tree-based methods capture non-linearities but may overfit
- **Future work**: Use spline regression or Gaussian process models to capture non-linear effects

#### 4.3.3 Interaction Effects

- Current analysis examines main effects only
- Interactions between parameters may be important
- **Future work**: Include interaction terms in regression or use interaction-aware tree models

#### 4.3.4 Embedding Model Heterogeneity

- Analysis pools across embedding models
- Parameter effects may differ for different embedding approaches
- **Future work**: Stratified analysis by embedding model or include embedding model as a factor

---

## 5. Conclusion

Stage 04 implements a principled approach to multi-objective model selection using Pareto efficiency analysis combined with comprehensive statistical hyperparameter analysis. The methodology successfully identifies optimal BERTopic configurations that balance coherence and topic diversity, with `paraphrase-mpnet-base-v2` iteration 0 emerging as the top-performing configuration.

**Key contributions**:
1. **Robust outlier filtering**: Two-stage approach removes invalid configurations
2. **Dual Pareto analysis**: Overall and per-model efficiency identification
3. **Evidence-based selection**: Hyperparameter correlation analysis guides configuration choices
4. **Transparent trade-offs**: Visualizations make performance trade-offs explicit
5. **Assumption-driven methodology**: Statistical validity ensured through explicit assumption checking

**Critical Parameters Identified**:
- **UMAP parameters** (`umap__min_dist`, `umap__n_components`) show strong, significant effects on diversity and combined scores
- **Vectorizer parameter** (`vectorizer__min_df`) is crucial for overall performance
- **Cluster size parameters** (`bertopic__min_topic_size`, `hdbscan__min_cluster_size`) primarily affect coherence

**Trade-offs Identified**:
- Coherence and diversity show opposing relationships with several parameters
- Optimal configurations balance these trade-offs through careful parameter selection
- Combined score optimization requires balancing multiple objectives

The selected models provide a solid foundation for downstream analysis stages, with configurations that offer both high coherence (interpretable topics) and good diversity (comprehensive topic coverage).

---

**Analysis Date**: 2025-01-27  
**Source Data**: Stage 03 hyperparameter optimization results  
**Statistical Methods**: Correlation analysis, linear regression, tree-based feature importance, Pareto efficiency analysis  
**Results Location**: `results/stage04_selection/`  
**Code Location**: `src/stage04_selection/`

