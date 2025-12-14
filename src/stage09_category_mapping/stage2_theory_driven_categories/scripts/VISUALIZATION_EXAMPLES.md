# Visualization Examples and Usage Guide

## Quick Reference

### Overview Plots (Generated Automatically)

#### 1. Volcano Plot
**File**: `volcano_plot.png`

**What it shows**:
- X-axis: Effect size (η²)
- Y-axis: -log10(p-value)
- Points colored by significance and effect size

**How to read**:
- **Top-right quadrant**: Significant + Large effect (most important)
- **Top-left quadrant**: Significant but small effect
- **Bottom-right quadrant**: Large effect but not significant (may be underpowered)
- **Bottom-left quadrant**: Neither significant nor large effect

**Example interpretation**:
- Category 5.3 appears in top-right → both significant and has large effect
- Category 6.5 appears in bottom-right → large effect but not significant (may need more power)

---

#### 2. Effect Size Bar Chart
**File**: `effect_size_bars.png`

**What it shows**:
- Top N categories ranked by effect size (η²)
- Color-coded by significance
- P-values annotated on bars

**How to read**:
- **Red bars**: Statistically significant (p < 0.05)
- **Gray bars**: Not significant
- **Bar length**: Effect size magnitude
- **Text on bars**: P-value

**Example interpretation**:
- Category 5.3 has longest red bar → largest significant effect
- Category 6.5 has long gray bar → large effect but not significant

---

#### 3. P-value Heatmap
**File**: `pvalue_heatmap.png`

**What it shows**:
- All categories with p-values color-coded
- Red = significant (low p-value)
- Green = not significant (high p-value)

**How to read**:
- **Red cells**: Significant differences (p < 0.05)
- **Yellow cells**: Borderline (p ≈ 0.05-0.10)
- **Green cells**: No significant difference (p > 0.10)
- **Numbers**: Exact p-values

**Example interpretation**:
- Quick scan shows 3 red cells → 3 significant categories
- Most cells are green → most categories don't differ

---

### Individual Category Plots

#### 4. Enhanced Prevalence Plot (Violin)
**File**: `category_{id}_prevalence.png`

**What it shows**:
- Violin plots showing full distribution shape
- Box plots inside violins (quartiles)
- Individual data points (jittered)
- P-value in title

**How to read**:
- **Violin width**: Density of data at that value
- **Box**: Interquartile range (IQR)
- **Points**: Individual books
- **Title**: Includes p-value and significance markers (*, **, ***)

**Example interpretation**:
- Category 5.3: Violins show different shapes → distributions differ
- Category 3.1: Violins similar → distributions similar

---

#### 5. Pairwise Comparison Plot
**File**: `category_{id}_pairwise.png` (only for significant categories)

**What it shows**:
- Bar chart of -log10(corrected p-values) for each pair
- Red bars = significant differences
- Gray bars = no significant difference
- Arrow indicators show direction of median difference

**How to read**:
- **Red bars**: These pairs differ significantly
- **Gray bars**: These pairs don't differ
- **↑/↓ arrows**: Direction of difference (which group is higher)
- **Numbers**: Median difference values

**Example interpretation**:
- Category 5.3: "bad vs good" is red → these groups differ
- Category 5.3: "bad vs mid" is gray → these groups don't differ
- This tells you: bad and good differ, but mid is similar to both

---

## Statistical Interpretation Guide

### Effect Size Guidelines (η²)
- **< 0.01**: Negligible effect
- **0.01 - 0.06**: Small effect
- **0.06 - 0.14**: Medium effect
- **> 0.14**: Large effect

### P-value Interpretation
- **< 0.001**: Highly significant (***)
- **< 0.01**: Very significant (**)
- **< 0.05**: Significant (*)
- **≥ 0.05**: Not significant

### Combined Interpretation
1. **Significant + Large Effect**: Important finding, both statistically and practically meaningful
2. **Significant + Small Effect**: Statistically significant but may not be practically important
3. **Not Significant + Large Effect**: May be underpowered, worth investigating with larger sample
4. **Not Significant + Small Effect**: No meaningful difference

---

## Common Questions

### Q: Why are some categories significant but have small effect sizes?
**A**: Statistical significance depends on sample size. With enough data, even tiny differences become significant. Effect sizes show practical importance.

### Q: What if a category has large effect but isn't significant?
**A**: This suggests the test may be underpowered (not enough data). The difference exists but we can't detect it statistically. Consider:
- Increasing sample size
- Checking if assumptions are met
- Using more sensitive tests

### Q: How do I know which specific groups differ?
**A**: Check the pairwise comparison plot for that category. It shows which pairs (bad vs mid, bad vs good, mid vs good) differ significantly.

### Q: Should I use Bonferroni correction for the main tests?
**A**: Currently, only pairwise comparisons use Bonferroni. For the main Kruskal-Wallis tests, you could apply FDR or Bonferroni if you want to control family-wise error rate across all 27 tests.

---

## Best Practices

1. **Start with overview plots**: Volcano plot and effect size chart give you the big picture
2. **Focus on significant + large effect**: These are the most important findings
3. **Check pairwise comparisons**: For significant categories, see which groups differ
4. **Consider practical significance**: Even if significant, small effects may not matter
5. **Look for patterns**: Do categories in the same group show similar patterns?

---

## Example Workflow

1. **Run analysis**: Generate all plots
2. **Check volcano plot**: Identify categories in top-right quadrant
3. **Review effect size chart**: See which categories have largest effects
4. **Examine significant categories**: Look at individual plots and pairwise comparisons
5. **Interpret results**: Combine statistical and practical significance
6. **Report findings**: Focus on significant + large effect categories
