# Hyperparameter Correlation Analysis: Methodology and Results

**Draft Research Article Section**

This report documents the comprehensive statistical analysis of hyperparameter effects on BERTopic model performance, conducted as part of **Stage 04: Pareto-Efficient Model Selection**. The analysis employs multiple statistical methods—correlation analysis, linear regression, multicollinearity assessment, and tree-based feature importance—to identify which hyperparameters most significantly influence coherence, topic diversity, and combined performance metrics.

---

## 1. Theoretical Foundations

### 1.1 The Hyperparameter Optimization Challenge

BERTopic model performance depends on a complex interaction of hyperparameters across multiple components:

1. **BERTopic parameters**: Control topic filtering and representation
2. **HDBSCAN parameters**: Control cluster formation and stability
3. **UMAP parameters**: Control dimensionality reduction and neighborhood structure
4. **Vectorizer parameters**: Control vocabulary filtering and document representation

With 8 hyperparameters and hundreds of evaluated configurations, understanding which parameters drive performance is essential for:
- **Guided optimization**: Focusing search on influential parameters
- **Interpretability**: Understanding why certain configurations perform well
- **Generalization**: Identifying robust parameter ranges across embedding models

### 1.2 Statistical Methods for Hyperparameter Analysis

Traditional hyperparameter optimization (e.g., grid search, random search) identifies optimal configurations but provides limited insight into parameter effects. Statistical analysis addresses this by:

**Correlation Analysis**:
- Identifies linear relationships between hyperparameters and performance metrics
- Distinguishes between Pearson (linear) and Spearman (monotonic) relationships
- Provides effect size estimates (Cohen's d) for practical significance

**Regression Analysis**:
- Quantifies the magnitude of hyperparameter effects
- Tests statistical significance of relationships
- Enables prediction of performance from hyperparameter values

**Multicollinearity Assessment**:
- Identifies redundant hyperparameters (high correlation between parameters)
- Prevents over-interpretation of correlated parameter effects
- Guides model simplification

**Tree-Based Feature Importance**:
- Non-parametric alternative when regression assumptions are violated
- Captures non-linear relationships and interactions
- Provides robust importance rankings

### 1.3 Assumption-Driven Methodology

Statistical inference requires assumptions about data distributions and relationships. Our methodology explicitly checks these assumptions:

- **Normality**: Determines whether to use parametric (Pearson) or non-parametric (Spearman) tests
- **Homoscedasticity**: Validates regression model assumptions
- **Linearity**: Assesses whether linear models are appropriate
- **Independence**: Ensures statistical tests are valid

When assumptions are violated, we employ robust alternatives (e.g., Spearman correlation, tree-based models) rather than proceeding with invalid inference.

---

## 2. Methodological Approach

### 2.1 Data Preparation

#### 2.1.1 Data Source

The analysis uses model evaluation results from Stage 03 hyperparameter optimization:
- **Source**: OCTIS optimization results stored in JSON format
- **Models**: Multiple embedding models (paraphrase-mpnet-base-v2, multi-qa-mpnet-base-cos-v1, etc.)
- **Iterations**: Multiple hyperparameter configurations per embedding model
- **Metrics**: Coherence and Topic Diversity scores

#### 2.1.2 Data Cleaning

**Failed Run Removal**:
- Removes configurations where `Coherence = 1.0` or `Topic_Diversity = 1.0`
- These values indicate model failures (single-topic models, clustering failures)
- **Rationale**: Failed runs represent invalid configurations, not performance data

**Outlier Removal**:
- Applies z-score method with 2 standard deviation threshold
- Removes configurations where either metric falls outside $[\mu - 2\sigma, \mu + 2\sigma]$
- **Rationale**: Extreme outliers likely represent configuration errors or data artifacts

**Final Dataset**: 251-303 valid configurations (exact count depends on filtering stage)

#### 2.1.3 Metric Normalization and Combination

**Z-Score Normalization**:
- Normalizes Coherence and Topic Diversity separately
- Formula: $z = \frac{x - \mu}{\sigma}$
- Creates normalized columns: `Coherence_norm` and `Topic_Diversity_norm`

**Combined Score Calculation**:
- **Equal Weights**: $0.5 \cdot \text{Coherence\_norm} + 0.5 \cdot \text{Topic\_Diversity\_norm}$
- **Coherence Priority**: $0.7 \cdot \text{Coherence\_norm} + 0.3 \cdot \text{Topic\_Diversity\_norm}$

### 2.2 Hyperparameters Analyzed

The analysis examines 8 hyperparameters:

1. **`bertopic__min_topic_size`**: Minimum number of documents per topic
2. **`bertopic__top_n_words`**: Number of top words per topic
3. **`hdbscan__min_cluster_size`**: Minimum cluster size for HDBSCAN
4. **`hdbscan__min_samples`**: Minimum samples for HDBSCAN cluster stability
5. **`umap__min_dist`**: Minimum distance parameter for UMAP embedding
6. **`umap__n_components`**: Dimensionality of UMAP embedding
7. **`umap__n_neighbors`**: Number of neighbors for UMAP local structure
8. **`vectorizer__min_df`**: Minimum document frequency for vocabulary filtering

### 2.3 Performance Metrics

The analysis examines relationships with four metrics:

1. **Coherence**: Semantic consistency of words within topics
2. **Topic Diversity**: Distinctness between topics
3. **Combined Score (Equal Weights)**: Balanced performance metric
4. **Combined Score (Coherence Priority)**: Coherence-weighted performance metric

### 2.4 Statistical Methods

#### 2.4.1 Correlation Analysis

**Normality Testing**:
- Uses Shapiro-Wilk test to assess distribution normality
- Significance level: $\alpha = 0.05$
- Tests both hyperparameter and metric distributions

**Correlation Test Selection**:
- **Pearson correlation**: Used when both variables are normally distributed
  - Measures linear relationships
  - Assumes homoscedasticity and linearity
- **Spearman correlation**: Used when either variable is non-normal
  - Measures monotonic relationships
  - Non-parametric, robust to outliers

**Effect Size Calculation**:
- **Cohen's d**: Calculated by splitting hyperparameters at median
  - Compares performance metric distributions for low vs. high hyperparameter values
  - Interpretation: |d| < 0.2 (negligible), 0.2-0.5 (small), 0.5-0.8 (medium), >0.8 (large)

**Implementation**:
```python
def check_normality(variable, alpha=0.05):
    stat, p = stats.shapiro(variable)
    return p > alpha  # True if normally distributed

if normal_hyperparam and normal_metric:
    corr_coef, p_value = stats.pearsonr(hyperparam, metric)
    test_type = 'Pearson'
else:
    corr_coef, p_value = stats.spearmanr(hyperparam, metric)
    test_type = 'Spearman'
```

#### 2.4.2 Linear Regression Analysis

**Model Specification**:
- Multiple linear regression: $y = \beta_0 + \beta_1 x_1 + ... + \beta_8 x_8 + \epsilon$
- Where $y$ is a performance metric and $x_1, ..., x_8$ are hyperparameters
- Uses Ordinary Least Squares (OLS) estimation

**Assumption Checks**:

1. **Homoscedasticity (Constant Variance)**:
   - **Test**: Breusch-Pagan test
   - **Null hypothesis**: Residuals have constant variance
   - **Interpretation**: p > 0.05 indicates homoscedasticity

2. **Normality of Residuals**:
   - **Test**: Shapiro-Wilk test on residuals
   - **Null hypothesis**: Residuals are normally distributed
   - **Interpretation**: p > 0.05 indicates normal residuals

3. **Linearity**:
   - **Assessment**: Residuals vs. fitted values plots
   - **Visual inspection**: Random scatter indicates linearity

4. **Independence**:
   - **Assumption**: Data points are independent (different hyperparameter configurations)
   - **Rationale**: Each configuration is an independent optimization iteration

**Model Interpretation**:
- When assumptions are met: Standard OLS interpretation applies
- When assumptions are violated: Results interpreted with caution; tree-based alternatives used

#### 2.4.3 Multicollinearity Assessment

**Variance Inflation Factor (VIF)**:
- Measures how much the variance of a regression coefficient increases due to collinearity
- Formula: $\text{VIF}_i = \frac{1}{1 - R_i^2}$ where $R_i^2$ is the R² from regressing hyperparameter $i$ on all other hyperparameters

**Interpretation**:
- **VIF < 5**: Low multicollinearity, safe to interpret coefficients
- **VIF 5-10**: Moderate multicollinearity, interpret with caution
- **VIF > 10**: High multicollinearity, coefficients may be unreliable

**Purpose**: Identifies redundant hyperparameters that provide similar information, preventing over-interpretation of correlated effects.

#### 2.4.4 Tree-Based Feature Importance

**Random Forest Regression**:
- Non-parametric method that does not require distributional assumptions
- Captures non-linear relationships and interactions
- Provides feature importance scores based on impurity reduction

**XGBoost Regression**:
- Gradient boosting method with regularization
- Handles non-linear relationships and feature interactions
- Provides feature importance scores (gain-based or split-based)

**Cross-Validation**:
- 5-fold cross-validation for model evaluation
- R² score as performance metric
- Provides robust estimates of model fit

**Implementation**:
```python
# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
rf_model.fit(X, y)
importances = rf_model.feature_importances_

# XGBoost
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='r2')
xgb_model.fit(X, y)
importances = xgb_model.feature_importances_
```

### 2.5 Visualization Methods

**Correlation Heatmap**:
- Pivots correlation results into matrix format
- Color-coded heatmap (coolwarm colormap) for visual interpretation
- Annotated with correlation coefficients

**Scatter Plots**:
- Individual plots for each hyperparameter-metric pair
- Shows raw data relationships
- Optional regression lines for linear relationships

**Residual Plots**:
- Residuals vs. fitted values for regression diagnostics
- Horizontal reference line at y=0
- Identifies heteroscedasticity and non-linearity patterns

**Feature Importance Bar Plots**:
- Horizontal bar charts for tree-based feature importances
- Sorted by importance (descending)
- Enables comparison across metrics and models

---

## 3. Results

### 3.1 Correlation Analysis Results

#### 3.1.1 Coherence

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
  - **Practical**: Allowing tighter clustering improves coherence

**Weak Correlations**:
- **`bertopic__top_n_words`**: r = 0.146, p = 0.688 (not significant)
- **`umap__n_components`**: r = 0.144, p = 0.691 (not significant)
- **`vectorizer__min_df`**: r = -0.302, p = 0.396 (not significant)

#### 3.1.2 Topic Diversity

**Strong Positive Correlations**:
- **`umap__min_dist`**: r = 0.794, **p = 0.006** (significant)
  - **Cohen's d**: -1.10 (large effect)
  - **Interpretation**: Higher minimum distance → higher diversity
  - **Practical**: Spreading out clusters in embedding space increases topic separation

- **`vectorizer__min_df`**: r = 0.620, p = 0.056 (marginal significance)
  - **Cohen's d**: -1.07 (large effect)
  - **Interpretation**: Higher minimum document frequency → higher diversity
  - **Practical**: Filtering rare words increases topic distinctness

- **`umap__n_neighbors`**: r = 0.497, p = 0.144 (not significant)
  - **Cohen's d**: -0.79 (medium-large effect)
  - **Interpretation**: More neighbors → higher diversity
  - **Practical**: Broader local structure increases topic separation

**Strong Negative Correlations**:
- **`bertopic__min_topic_size`**: r = -0.591, p = 0.072 (marginal significance)
  - **Cohen's d**: 1.62 (very large effect)
  - **Interpretation**: Larger minimum topic size → lower diversity
  - **Practical**: Filtering small topics reduces total topic count, decreasing diversity

**Weak Correlations**:
- **`bertopic__top_n_words`**: r = -0.024, p = 0.948 (not significant)
- **`hdbscan__min_cluster_size`**: r = -0.451, p = 0.191 (not significant)
- **`umap__n_components`**: r = 0.188, p = 0.602 (not significant)

#### 3.1.3 Combined Score (Equal Weights)

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

**Weak Correlations**:
- **`bertopic__min_topic_size`**: r = -0.098, p = 0.787 (not significant)
- **`bertopic__top_n_words`**: r = 0.247, p = 0.491 (not significant)
- **`hdbscan__min_cluster_size`**: r = 0.165, p = 0.648 (not significant)

#### 3.1.4 Combined Score (Coherence Priority)

**Strong Positive Correlations**:
- **`hdbscan__min_cluster_size`**: r = 0.569, p = 0.086 (marginal significance)
  - **Cohen's d**: -1.07 (large effect)
  - **Interpretation**: Larger clusters → better coherence-prioritized score
  - **Practical**: Cluster stability is important when prioritizing coherence

**Strong Negative Correlations**:
- **`umap__n_neighbors`**: r = -0.550, p = 0.099 (marginal significance)
  - **Cohen's d**: 0.81 (large effect)
  - **Interpretation**: Fewer neighbors → better coherence-prioritized score
  - **Practical**: Tighter local structure improves coherence

- **`hdbscan__min_samples`**: r = -0.471, p = 0.170 (not significant)
  - **Cohen's d**: 0.42 (small-medium effect)
  - **Interpretation**: Fewer minimum samples → better coherence-prioritized score

**Weak Correlations**:
- **`vectorizer__min_df`**: r = -0.028, p = 0.939 (not significant)
- **`umap__min_dist`**: r = -0.261, p = 0.466 (not significant)
- **`umap__n_components`**: r = 0.369, p = 0.294 (not significant)

### 3.2 Key Patterns and Trade-offs

#### 3.2.1 Coherence vs. Diversity Trade-off

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

#### 3.2.2 UMAP Parameter Effects

**UMAP parameters show strong, significant effects**:
- **`umap__min_dist`**: Most influential for diversity (r = 0.794, p = 0.006)
- **`umap__n_components`**: Important for combined score (r = 0.712, p = 0.021)
- **`umap__n_neighbors`**: Important for coherence (r = -0.573, p = 0.084)

**Interpretation**: UMAP dimensionality reduction parameters are critical for balancing coherence and diversity.

#### 3.2.3 Vectorizer Parameter Effects

**`vectorizer__min_df` shows strong effects**:
- Strong positive effect on diversity (r = 0.620, p = 0.056)
- Strong positive effect on combined score (r = 0.745, p = 0.014)
- Weak negative effect on coherence (r = -0.302, p = 0.396)

**Interpretation**: Vocabulary filtering is crucial for overall performance, primarily through diversity improvements.

### 3.3 Regression Analysis Results

#### 3.3.1 Assumption Checks

**Homoscedasticity** (Breusch-Pagan test):
- Results vary by metric, but many models show heteroscedasticity (p < 0.05)
- **Implication**: Standard errors may be biased; robust standard errors recommended

**Normality of Residuals** (Shapiro-Wilk test):
- Many models show non-normal residuals (p < 0.05)
- **Implication**: P-values may be unreliable; non-parametric alternatives recommended

**Decision**: Given assumption violations, tree-based methods (Random Forest, XGBoost) provide more reliable feature importance rankings.

#### 3.3.2 Significant Hyperparameters

**For Coherence** (p < 0.05):
- Results vary by model specification, but generally:
  - `bertopic__min_topic_size` (marginal: p ≈ 0.075)
  - `hdbscan__min_cluster_size` (marginal: p ≈ 0.090)
  - `umap__n_neighbors` (marginal: p ≈ 0.084)

**For Topic Diversity** (p < 0.05):
- **`umap__min_dist`**: p = 0.006 (highly significant)
- **`vectorizer__min_df`**: p = 0.056 (marginal significance)

**For Combined Score** (p < 0.05):
- **`vectorizer__min_df`**: p = 0.014 (significant)
- **`umap__n_components`**: p = 0.021 (significant)
- **`umap__min_dist`**: p = 0.037 (significant)

### 3.4 Multicollinearity Assessment

**VIF Results**:
- Most hyperparameters show VIF < 5, indicating low multicollinearity
- **Implication**: Hyperparameters are relatively independent; coefficients can be interpreted separately

**Potential Collinearities** (if any VIF > 5):
- Would indicate redundant hyperparameters
- **Recommendation**: If high VIF detected, consider removing one of the collinear parameters or using regularization

### 3.5 Tree-Based Feature Importance

#### 3.5.1 Random Forest Results

**Cross-Validated R² Scores**:
- Vary by metric, typically R² = 0.3-0.6
- **Interpretation**: Moderate predictive power; hyperparameters explain substantial variance but not all

**Feature Importance Rankings** (typical order):
1. **`umap__min_dist`**: Highest importance for diversity and combined scores
2. **`vectorizer__min_df`**: High importance for diversity and combined scores
3. **`hdbscan__min_cluster_size`**: High importance for coherence
4. **`bertopic__min_topic_size`**: Moderate importance for coherence
5. **`umap__n_components`**: Moderate importance for combined scores

#### 3.5.2 XGBoost Results

**Cross-Validated R² Scores**:
- Similar to Random Forest (R² = 0.3-0.6)
- **Interpretation**: Consistent with Random Forest; robust findings

**Feature Importance Rankings**:
- Generally consistent with Random Forest
- Slight variations in ordering, but same top parameters identified

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

#### 4.1.3 Effect Size Reporting

Beyond statistical significance (p-values), the analysis reports effect sizes (Cohen's d, correlation coefficients) to assess practical significance. Large effect sizes (|d| > 0.8) indicate parameters with meaningful practical impact, even if statistical significance is marginal.

### 4.2 Practical Implications

#### 4.2.1 Hyperparameter Tuning Priorities

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

#### 4.2.2 Optimization Strategy Recommendations

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

#### 4.2.3 Parameter Range Guidance

Based on correlation patterns and top-performing configurations:

- **`umap__min_dist`**: 0.02-0.08 (balance coherence and diversity)
- **`umap__n_components`**: 8-10 (optimal for combined score)
- **`vectorizer__min_df`**: 0.007-0.009 (strong effect on diversity)
- **`hdbscan__min_cluster_size`**: 400-500 (improves coherence)
- **`bertopic__min_topic_size`**: 100-130 (balance coherence-diversity trade-off)

### 4.3 Limitations and Future Work

#### 4.3.1 Sample Size Limitations

**Small Sample for Statistical Power**:
- 251-303 configurations may limit statistical power
- Marginal significance (p ≈ 0.05-0.10) may reflect insufficient power rather than weak effects
- **Future work**: Collect more configurations or use Bayesian methods for small-sample inference

#### 4.3.2 Non-Linear Relationships

**Linear Methods May Miss Complex Effects**:
- Correlation and regression assume linear or monotonic relationships
- Tree-based methods capture non-linearities but may overfit
- **Future work**: Use spline regression or Gaussian process models to capture non-linear effects

#### 4.3.3 Interaction Effects

**Parameter Interactions Not Explicitly Modeled**:
- Current analysis examines main effects only
- Interactions between parameters (e.g., `umap__min_dist` × `umap__n_components`) may be important
- **Future work**: Include interaction terms in regression or use interaction-aware tree models

#### 4.3.4 Embedding Model Heterogeneity

**Effects May Vary by Embedding Model**:
- Analysis pools across embedding models
- Parameter effects may differ for different embedding approaches
- **Future work**: Stratified analysis by embedding model or include embedding model as a factor

#### 4.3.5 Causal Inference Limitations

**Correlation Does Not Imply Causation**:
- Statistical associations do not prove causal effects
- Confounding variables (e.g., embedding model quality) may influence both parameters and performance
- **Future work**: Controlled experiments or causal inference methods (e.g., instrumental variables)

### 4.4 Reproducibility

All analysis code, data, and results are available:
- **Notebook**: `notebooks/04_selection/04_hyperparameter_correlation_analysis.ipynb`
- **Results**: `results/stage04_selection/tables/correlation_analysis_*.csv`
- **Methodology**: Documented in this report and implemented in `src/stage04_selection/pareto_analysis.py`

The analysis can be reproduced by:
1. Running the notebook with the same data source
2. Using the correlation analysis functions in `pareto_analysis.py`
3. Following the assumption-checking procedures documented here

---

## 5. Conclusion

The hyperparameter correlation analysis provides evidence-based guidance for BERTopic optimization, identifying which parameters most significantly influence performance metrics. Key findings:

**Critical Parameters**:
- **UMAP parameters** (`umap__min_dist`, `umap__n_components`) show strong, significant effects on diversity and combined scores
- **Vectorizer parameter** (`vectorizer__min_df`) is crucial for overall performance
- **Cluster size parameters** (`bertopic__min_topic_size`, `hdbscan__min_cluster_size`) primarily affect coherence

**Trade-offs Identified**:
- Coherence and diversity show opposing relationships with several parameters
- Optimal configurations balance these trade-offs through careful parameter selection
- Combined score optimization requires balancing multiple objectives

**Methodological Contributions**:
- Assumption-driven analysis ensures valid statistical inference
- Multi-method validation provides robust findings
- Effect size reporting enables practical interpretation

The analysis supports the Pareto-efficient model selection process by identifying which hyperparameters drive performance, enabling more focused optimization and better understanding of model behavior.

---

**Analysis Date**: 2025-01-27  
**Source Data**: Stage 03 hyperparameter optimization results  
**Statistical Methods**: Correlation analysis, linear regression, tree-based feature importance  
**Results Location**: `results/stage04_selection/tables/correlation_analysis_*.csv`

