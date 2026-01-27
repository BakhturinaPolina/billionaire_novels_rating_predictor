# Stage 04: Pareto-Efficient Model Selection and Hyperparameter Analysis

**Research Report**

This report documents the methodology and results of Stage 04, which implements a principled multi-objective optimization approach to identify optimal BERTopic configurations that balance coherence and topic diversity through Pareto efficiency analysis and comprehensive statistical hyperparameter analysis.

---

## 1. Introduction

Topic modeling evaluation requires balancing multiple, often competing, performance objectives. In our analysis, we focus on two critical metrics:

- **Coherence**: Measures semantic consistency of words within topics, indicating topic interpretability
- **Topic Diversity**: Measures distinctness between topics, indicating comprehensive thematic coverage

These objectives are fundamentally in tension: maximizing coherence may reduce diversity (fewer, more focused topics), while maximizing diversity may reduce coherence (more topics with less internal consistency). Traditional single-objective optimization fails to capture this trade-off, necessitating a multi-objective approach using Pareto efficiency principles.

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

**3. paraphrase-MiniLM-L6-v2, iteration 19 (Rank 1)**
- **Combined Score**: 1.65
- **Coherence**: 0.425
- **Topic Diversity**: 0.94 (highest diversity)

**Key Observations**:
- **Model hierarchy**: mpnet variants dominate combined scores
- **Performance trade-offs**: Optimal balance appears around coherence 0.4-0.47 and diversity 0.8-0.85
- **Outlier filtering impact**: Reduction from 12 to 4 Pareto-efficient models after filtering indicates many initially identified "efficient" configurations were statistical artifacts

### 3.2 Hyperparameter Correlation Analysis Results

Summary of correlations between hyperparameters and performance metrics (|r| > 0.5 shown):

| Hyperparameter | Coherence | Diversity | Combined Score |
|----------------|-----------|-----------|----------------|
| `umap__min_dist` | r = -0.528 | **r = 0.794*** | **r = 0.663*** |
| `vectorizer__min_df` | r = 0.387 | r = 0.620† | **r = 0.745*** |
| `umap__n_components` | r = 0.445 | r = 0.502 | **r = 0.712*** |
| `bertopic__min_topic_size` | r = 0.585† | r = -0.591† | r = 0.156 |
| `hdbscan__min_cluster_size` | r = 0.563† | r = -0.451 | r = 0.198 |
| `umap__n_neighbors` | r = -0.573† | r = 0.312 | r = -0.101 |

*p < 0.05 (significant), †p < 0.10 (marginal)

### 3.3 Key Trade-offs

**Opposing Effects** (parameters that improve one metric but hurt the other):
- **`bertopic__min_topic_size`**: ↑ coherence (r = 0.585) but ↓ diversity (r = -0.591)
- **`umap__min_dist`**: ↑ diversity (r = 0.794) but ↓ coherence (r = -0.528)

**Synergistic Effects**:
- **`hdbscan__min_cluster_size`**: Primarily benefits coherence (r = 0.563) with minimal diversity cost (r = -0.451)

### 3.4 Tree-Based Feature Importance Validation

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

- **Primary**: `paraphrase-mpnet-base-v2` iteration 0 — best overall balance (see Section 3.1)
- **Diversity-focused alternative**: `paraphrase-MiniLM-L6-v2` iteration 19 — highest diversity (0.94)

#### 4.2.2 Hyperparameter Tuning Guide

| Parameter | Priority | Optimal Range | Primary Effect |
|-----------|----------|---------------|----------------|
| `umap__min_dist` | High | 0.02-0.08 | ↑ diversity, ↓ coherence |
| `vectorizer__min_df` | High | 0.007-0.009 | ↑ combined score |
| `umap__n_components` | High | 8-10 | ↑ combined score |
| `bertopic__min_topic_size` | Medium | 100-130 | ↑ coherence, ↓ diversity |
| `hdbscan__min_cluster_size` | Medium | 400-500 | ↑ coherence |
| `umap__n_neighbors` | Medium | Lower values | ↑ coherence |
| `bertopic__top_n_words` | Low | — | Minimal effect |
| `hdbscan__min_samples` | Low | — | Weak effects |

**Optimization Strategies**:
- **Maximize Coherence**: Increase cluster/topic sizes, decrease `umap__min_dist` and `umap__n_neighbors`
- **Maximize Diversity**: Increase `umap__min_dist` and `vectorizer__min_df`, decrease topic sizes
- **Balanced Performance**: Optimize `umap__min_dist` (critical trade-off) and `vectorizer__min_df`

### 4.3 Limitations and Future Work

#### 4.3.1 Sample Size Limitations

- 251 configurations may limit statistical power
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

Stage 04 implements a principled multi-objective model selection approach using Pareto efficiency analysis combined with statistical hyperparameter analysis. The methodology identifies `paraphrase-mpnet-base-v2` iteration 0 as the top-performing configuration, balancing coherence (0.463) and diversity (0.82).

**Key contributions**:
1. Two-stage outlier filtering removes invalid configurations
2. Multi-method validation (correlation, regression, tree-based) ensures robust findings
3. Evidence-based parameter guidance derived from statistical analysis
4. Explicit trade-off quantification between coherence and diversity

The analysis reveals that UMAP parameters (`min_dist`, `n_components`) and `vectorizer__min_df` are the most influential for overall performance, while cluster size parameters primarily affect coherence. The selected models provide a solid foundation for downstream analysis stages.

---

**Analysis Date**: 2025-01-27  
**Source Data**: Stage 03 hyperparameter optimization results  
**Statistical Methods**: Correlation analysis, linear regression, tree-based feature importance, Pareto efficiency analysis  
**Results Location**: `results/stage04_selection/`  
**Code Location**: `src/stage04_selection/`
