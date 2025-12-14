# Statistical Analysis Visualization Improvements - Summary

## Overview
Enhanced the statistical analysis visualization suite with improved plots that provide better insights into category differences across rating classes.

## New Features Added

### 1. Enhanced Statistical Functions (`stats_helpers.py`)

#### Effect Size Calculation
- **Added**: `kruskal_eta_squared()` function
- **Purpose**: Calculate eta-squared (η²) effect size for Kruskal-Wallis tests
- **Interpretation**: 
  - 0.01 = small effect
  - 0.06 = medium effect  
  - 0.14 = large effect
- **Benefit**: Distinguishes statistical significance from practical significance

#### Post-hoc Pairwise Comparisons
- **Added**: `pairwise_comparisons()` function
- **Purpose**: Perform Mann-Whitney U tests between all pairs of rating classes
- **Features**:
  - Bonferroni correction for multiple comparisons
  - Median difference calculations
  - Significance indicators
- **Benefit**: Identifies which specific rating classes differ (bad vs mid, bad vs good, mid vs good)

### 2. New Visualization Functions (`visualization_helpers.py`)

#### Volcano Plot
- **Function**: `plot_volcano()`
- **Shows**: -log10(p-value) vs effect size (η²)
- **Features**:
  - Color-coded by significance and effect size
  - Labels for significant categories
  - Threshold lines for p-value and effect size
- **Benefit**: Identifies categories that are both statistically significant AND have large effects

#### Effect Size Bar Chart
- **Function**: `plot_effect_size_bars()`
- **Shows**: Top N categories ranked by effect size
- **Features**:
  - Color-coded by significance
  - P-value annotations
  - Horizontal bars for easy reading
- **Benefit**: Highlights categories with largest practical differences

#### Enhanced Category Prevalence Plots
- **Function**: `plot_category_prevalence()` (enhanced)
- **New options**: 
  - Violin plots (shows full distribution shape)
  - Box plots (traditional)
  - Both combined
- **Features**:
  - Better visualization of distribution shapes
  - Individual data points overlaid
- **Benefit**: Better understanding of data distributions, especially for skewed data

#### Pairwise Comparison Plots
- **Function**: `plot_pairwise_comparisons()`
- **Shows**: Post-hoc test results for significant categories
- **Features**:
  - Bar chart of -log10(corrected p-values)
  - Median difference indicators (↑/↓)
  - Significance threshold line
- **Benefit**: Clear visualization of which specific groups differ

#### P-value Heatmap
- **Function**: `plot_pvalue_heatmap()`
- **Shows**: All categories with p-values color-coded
- **Features**:
  - Optional grouping by category group
  - Color scale: red (significant) to green (not significant)
  - Category names as labels
- **Benefit**: Quick overview of all results at once

### 3. Updated Analysis Script (`analyze_category_differences.py`)

#### New Outputs Generated:
1. **volcano_plot.png** - Overview of significance vs effect size
2. **effect_size_bars.png** - Top categories by effect size
3. **pvalue_heatmap.png** - All categories at a glance
4. **category_{id}_prevalence.png** - Enhanced violin plots with p-values
5. **category_{id}_pairwise.png** - Post-hoc comparisons for significant categories

#### Improvements:
- Automatic generation of overview plots
- Enhanced individual category plots with violin distributions
- Post-hoc analysis for all significant categories
- Better error handling and progress reporting

## Usage

Run the analysis script as before:
```bash
python scripts/analyze_category_differences.py \
    --book-cat results/.../book_category_proportions.parquet \
    --output-dir results/.../analysis \
    --top-n 15 \
    --alpha 0.05
```

The script now generates:
- All previous outputs (CSV results, individual category plots)
- New overview visualizations (volcano plot, effect size chart, heatmap)
- Post-hoc pairwise comparisons for significant categories

## Benefits

1. **Better Statistical Interpretation**
   - Effect sizes show practical significance
   - Post-hoc tests identify specific group differences
   - Multiple visualizations provide different perspectives

2. **Improved Visual Communication**
   - Volcano plots highlight important findings
   - Violin plots show distribution shapes
   - Heatmaps provide quick overviews

3. **More Complete Analysis**
   - Pairwise comparisons answer "which groups differ?"
   - Effect sizes answer "how large are the differences?"
   - Multiple plots reduce risk of missing important patterns

## Technical Details

### Effect Size Calculation
- Uses eta-squared (η²) for Kruskal-Wallis tests
- Formula: η² = (H - k + 1) / (n - k)
- Where H = H-statistic, k = number of groups, n = total sample size

### Multiple Comparisons Correction
- Bonferroni correction applied to pairwise comparisons
- Adjusted p-value = p-value × number of comparisons
- More conservative but reduces false positives

### Plot Aesthetics
- Consistent color scheme: red (significant), gray (not significant)
- Professional styling with grid lines and clear labels
- High-resolution output (150 DPI) for publications

## Next Steps (Optional Enhancements)

1. **Forest Plot**: Show effect sizes with confidence intervals
2. **Interaction Plots**: Show category × rating class interactions
3. **Effect Size Confidence Intervals**: Bootstrap confidence intervals for η²
4. **Multiple Comparison Corrections**: Apply FDR or Bonferroni to main Kruskal-Wallis tests
5. **Interactive Plots**: Create Plotly versions for exploration
