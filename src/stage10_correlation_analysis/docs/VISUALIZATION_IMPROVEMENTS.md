# Visualization Improvements for Statistical Analysis

## Current State
- **Current plots**: Box plots with jittered points for individual categories
- **Statistical test**: Kruskal-Wallis H-test
- **Limitations**: 
  - No effect size visualization
  - No post-hoc pairwise comparisons
  - No overview of all categories
  - No indication of which specific groups differ

## Recommended New Visualizations

### 1. **Volcano Plot** (Priority: High)
Shows p-value vs effect size to identify both statistically significant AND practically meaningful differences.

**Why**: Helps identify categories that are both significant and have large effect sizes, or categories with large effects that might be underpowered.

### 2. **Effect Size Bar Chart** (Priority: High)
Rank categories by effect size (eta-squared or rank-biserial correlation) with significance indicators.

**Why**: Statistical significance doesn't mean practical significance. Effect sizes show magnitude of differences.

### 3. **Post-hoc Pairwise Comparisons** (Priority: High)
For significant categories, show which specific rating classes differ (bad vs mid, bad vs good, mid vs good).

**Why**: Kruskal-Wallis only tells you groups differ, not which ones. Critical for interpretation.

### 4. **P-value Overview Heatmap** (Priority: Medium)
Heatmap showing all categories with p-values color-coded, grouped by category group.

**Why**: Quick overview of all results at once, helps identify patterns by category group.

### 5. **Violin Plots** (Priority: Medium)
Replace or supplement box plots with violin plots to show full distribution shapes.

**Why**: Better visualization of distribution shapes, especially for skewed data.

### 6. **Forest Plot** (Priority: Medium)
Show effect sizes with confidence intervals for all categories.

**Why**: Standard way to present multiple effect sizes in meta-analysis, shows uncertainty.

### 7. **Significance Summary Bar Chart** (Priority: Low)
Bar chart of -log10(p-values) with significance threshold line.

**Why**: Quick visual summary of which categories are most significant.

### 8. **Pairwise Comparison Matrix** (Priority: Medium)
For top categories, show pairwise comparison results in a matrix format.

**Why**: Clear visualization of which specific groups differ for each category.

## Implementation Priority

1. **Immediate**: Volcano plot, effect size calculations, post-hoc tests
2. **Short-term**: Violin plots, pairwise comparison plots
3. **Long-term**: Heatmap, forest plot, summary charts
