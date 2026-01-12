---
name: Hypothesis Testing Analysis
overview: Create a new notebook in `07_analysis/hypothesis_testing/` to perform statistical inference on composite indices, addressing reliability issues, testing hypotheses H1-H6, and building predictive models to identify success factors.
todos:
  - id: setup_notebook
    content: Create notebook structure with imports, config, and data loading cells following patterns from 04_build_theory_aligned_composites_indices_v3_with_temporal.ipynb
    status: completed
  - id: step1_reliability
    content: "Implement Step 1: Index Iteration and Refinement - load reliability metrics, analyze failing indices, perform EFA/correlation analysis, split composites if needed, export refined definitions"
    status: completed
    dependencies:
      - setup_notebook
  - id: step2_correlation
    content: "Implement Step 2: Exploratory Correlation Matrix - compute correlation matrix, create heatmaps with clustering, identify multicollinearity, document trope packages"
    status: completed
    dependencies:
      - setup_notebook
  - id: step3_static_tests
    content: "Implement Step 3: Static Hypothesis Testing (H1-H5) - Kruskal-Wallis tests, pairwise comparisons, effect sizes, H3 interaction regression, create visualizations"
    status: completed
    dependencies:
      - setup_notebook
  - id: step4_temporal
    content: "Implement Step 4: Temporal Arc Analysis (H6) - LMM models, slope analysis, narrative arc visualizations with tier comparisons"
    status: completed
    dependencies:
      - setup_notebook
  - id: step5_predictive
    content: "Implement Step 5: Predictive Modeling - Random Forest classifier, feature importance, partial dependence plots, model evaluation"
    status: completed
    dependencies:
      - setup_notebook
---

# Hypothesis Testing and Statistical Inference Analysis

## Overview

This notebook implements the statistical inference phase, moving from index construction to hypothesis testing. It addresses reliability issues, performs exploratory correlation analysis, tests static hypotheses (H1-H5), analyzes temporal arcs (H6), and builds predictive models.

## Notebook Structure

**Location**: `notebooks/07_analysis/hypothesis_testing/05_hypothesis_testing_and_inference.ipynb`

**Output Directory**: `results/stage10_correlation_analysis/hypothesis_testing/`

## Implementation Plan

### Step 1: Index Iteration and Refinement (Reliability Patch)

**Input**: `results/stage10_correlation_analysis/taxonomy_group_analysis/indices/index_reliability_cronbach_alpha.csv`

**Actions**:

- Load reliability metrics and identify indices with negative or low Cronbach's α (< 0.6)
- For failing indices (e.g., `R_protectiveness: -0.72`, `C_explicit_eroticism: -0.53`):
- Load composite topic membership from `composite_topic_membership.csv`
- Compute correlation matrix of sub-topics within each failing composite
- Perform Exploratory Factor Analysis (EFA) or cluster analysis to identify topic subgroups
- If topics are negatively correlated, split the composite into separate indices
- Recalculate Cronbach's α for refined indices
- Export refined index definitions and updated reliability metrics

**Outputs**:

- `refined_index_definitions.csv` - Updated composite specifications
- `index_reliability_refined.csv` - Updated reliability metrics
- `topic_correlation_matrices/` - Correlation matrices for each failing composite

### Step 2: Exploratory Correlation Matrix

**Input**: `results/stage10_correlation_analysis/taxonomy_group_analysis/indices/book_indices_z_with_derived.csv`

**Actions**:

- Load book-level indices (all A-S composites plus derived H1-H6 indices)
- Compute Pearson correlation matrix for all indices
- Create correlation heatmap with clustering (using seaborn.clustermap)
- Identify multicollinearity (r > 0.8) between indices
- Generate "trope package" visualization showing which themes cluster together
- Document multicollinearity issues for regression models

**Outputs**:

- `index_correlation_matrix.csv` - Full correlation matrix
- `index_correlation_heatmap.html` - Interactive heatmap (Plotly)
- `index_correlation_heatmap.png` - Static heatmap
- `multicollinearity_report.csv` - Pairs with r > 0.8

### Step 3: Static Hypothesis Testing (H1–H5)

**Input**:

- `results/stage10_correlation_analysis/taxonomy_group_analysis/indices/book_indices_z_with_derived.csv`
- Book metadata with `rating_tier` (top/middle/trash)

**Actions**:

- Merge indices with book metadata to get tier labels
- For each hypothesis (H1-H5):
- **H1 (Love-over-Sex)**: Test `H1_love_over_sex_log` across tiers
- **H2 (HEA Index)**: Test `A_reassurance_commitment + G_courtship_rituals_gifts` across tiers
- **H3 (Luxury × Love)**: OLS regression with interaction term
- Model: `Rating ~ H3_luxury_saturation + H3_love_depth + H3_interaction + Controls`
- **H4 (Protectiveness vs Jealousy)**: Test `H4_protect_over_jealous_log` across tiers
- **H5 (Darkness vs Tenderness)**: Test `H5_dark_over_tender_log` across tiers
- Statistical tests:
- **Kruskal-Wallis H-test** for tier differences (non-parametric, handles non-normal distributions)
- **Pairwise Mann-Whitney U tests** with Holm correction for multiple comparisons
- **Effect sizes**: Cohen's d (for pairwise) and Eta-squared (for Kruskal-Wallis)
- Visualizations:
- Violin plots with box plots for each index by tier
- Effect size heatmaps (Top vs Trash differences)

**Outputs**:

- `hypothesis_tests_static.csv` - Test results (p-values, effect sizes, test statistics)
- `hypothesis_effect_sizes.csv` - Effect size summaries
- `figures/static_hypothesis_tests/` - Violin plots and effect size visualizations
- `regression_models_h3.csv` - H3 interaction model results

### Step 4: Temporal Arc Analysis (H6)

**Input**: `results/stage10_correlation_analysis/taxonomy_group_analysis/indices/segment_indices_z_with_derived.csv`

**Actions**:

- Load segment-level indices (begin/middle/end for each book)
- For each key index (A, B, Q_repair, Q_miscommunication, F_angst_negative_affect):
- **Linear Mixed-Effects Model (LMM)**:
- Model: `Index_Score ~ Segment + Tier + (Segment * Tier) + (1|book_id)`
- Use `statsmodels` or `scikit-learn` for LMM
- Test interaction term: does the slope differ by tier?
- **Slope Analysis**:
- Calculate per-book slopes (begin→end) for repair and commitment indices
- Compare slope distributions across tiers
- **Trend Visualization**:
- Plot mean narrative arc for each tier (begin/middle/end) with confidence intervals
- Use consistent color scheme from taxonomy_group_analysis

**Outputs**:

- `temporal_arc_lmm_results.csv` - Mixed-effects model results
- `temporal_slopes_by_book.csv` - Per-book slopes
- `temporal_slope_comparisons.csv` - Slope differences across tiers
- `figures/temporal_arcs/` - Arc plots for each index by tier

### Step 5: Predictive Modeling (Success Detector)

**Input**: `results/stage10_correlation_analysis/taxonomy_group_analysis/indices/book_indices_z_with_derived.csv`

**Actions**:

- Prepare binary classification target: `is_top = (rating_tier == 'top')`
- **Random Forest Classifier**:
- Features: All A-S indices (exclude derived H1-H6 to avoid leakage)
- Target: `is_top` (binary)
- Train/test split (80/20) with stratification
- Extract feature importance (Gini importance)
- Evaluate with ROC-AUC, precision, recall, F1
- **Feature Importance Analysis**:
- Rank indices by importance
- Identify which themes are strongest predictors
- **Partial Dependence Plots (PDPs)**:
- For top 5-10 most important indices
- Visualize relationship between index value and predicted probability of "Top"
- Identify "Goldilocks zones" (optimal ranges)

**Outputs**:

- `random_forest_model.pkl` - Trained model
- `feature_importance.csv` - Ranked feature importance
- `model_performance_metrics.csv` - Classification metrics
- `figures/partial_dependence_plots/` - PDPs for top features

## Code Structure

### Imports and Setup

- Follow structure from `04_build_theory_aligned_composites_indices_v3_with_temporal.ipynb`
- Use plotting constants from `taxonomy_group_analysis.ipynb`:
- `TIER_COLORS`, `TIER_COLORS_DISPLAY`
- `EFFECT_CMAP_MPL`, `EFFECT_CMAP_PLOTLY`
- `symmetric_effect_norm()`, `ensure_tier_display()`

### Data Loading

- Load indices from `taxonomy_group_analysis/indices/`
- Load book metadata with tier labels
- Handle missing data appropriately

### Statistical Functions

- Kruskal-Wallis test wrapper
- Effect size calculators (Cohen's d, Eta-squared)
- Multiple comparison correction (Holm method)
- LMM fitting utilities

## Key Files Referenced

- **Index Construction Logic**: `notebooks/07_analysis/indexing_hypothesis_testing/04_build_theory_aligned_composites_indices_v3_with_temporal.ipynb`
- **Plotting Schemas**: `notebooks/07_analysis/taxonomy_group_analysis/taxonomy_group_analysis.ipynb`
- **Hypothesis Definitions**: `SCIENTIFIC_README.md` (lines 27-52)

## Output Organization

All outputs saved to `results/stage10_correlation_analysis/hypothesis_testing/`:

- `tables/` - CSV files with test results, correlations, model outputs
- `figures/` - Visualizations organized by analysis type
- `models/` - Saved model files (Random Forest)