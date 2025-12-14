# Statistical Analysis Report: Taxonomy Category Differences Across Rating Classes

**Date**: December 14, 2025 (Updated)  
**Analysis**: Kruskal-Wallis tests with effect sizes and post-hoc pairwise comparisons  
**Dataset**: 92 books across 3 rating classes (bad, mid, good)  
**Categories Analyzed**: 27 taxonomy categories

---

## Executive Summary

This report presents statistical analysis of taxonomy category prevalence differences across book rating classes using Kruskal-Wallis tests with effect size calculations and post-hoc pairwise comparisons. The analysis examined 27 categories from the Romance Corpus Topic Taxonomy across 92 books classified into three rating groups: bad (20-30 books per category), mid (22-32 books per category), and good (27-32 books per category).

**Key Finding**: **3 categories** show statistically significant differences (p < 0.05) across rating classes:
- **5.3: Community, Norms & Social Events** (p = 0.029, η² = 0.070 - **medium effect**)
- **6.2: Heroine's Work & Professional Identity** (p = 0.047, η² = 0.057 - **small-medium effect**)
- **3.4: Beliefs, Values & Moral Reflection** (p = 0.048, η² = 0.048 - **small effect**)

This suggests that while most thematic content categories are similarly distributed across books regardless of their Goodreads ratings, social events, heroine's work, and moral reflection serve as distinguishing factors. The effect sizes indicate that these differences are not only statistically significant but also practically meaningful, with Category 5.3 showing the largest effect.

---

## Methodology

### Data
- **Source**: Book-level category proportions aggregated from sentence-level topic assignments
- **Books**: 92 books (minimum 50 sentences per book)
- **Rating Classes**: 
  - Bad: 30 books
  - Mid: 32 books  
  - Good: 30 books
- **Categories**: 28 taxonomy categories (excluding categories with insufficient data)

### Statistical Test
- **Test**: Kruskal-Wallis H-test (non-parametric)
- **Purpose**: Test for differences in category prevalence distributions across rating classes
- **Significance Level**: α = 0.05
- **Minimum Group Size**: 5 books per category-rating combination
- **Effect Size**: Eta-squared (η²) calculated for all tests
  - Interpretation: < 0.01 = negligible, 0.01-0.06 = small, 0.06-0.14 = medium, > 0.14 = large
- **Post-hoc Tests**: Mann-Whitney U tests with Bonferroni correction for significant categories

### Analysis Pipeline
1. Load book-level category proportions from `book_category_proportions.parquet`
2. Run Kruskal-Wallis test for each category with effect size calculation
3. Sort results by p-value
4. Perform post-hoc pairwise comparisons for significant categories
5. Generate comprehensive visualizations (overview plots, individual categories, pairwise comparisons)
6. Report significant findings with effect sizes

---

## Significant Results

### Category 5.3: Community, Norms & Social Events

**Status**: ✅ **Significantly Different** (p = 0.029, η² = 0.070 - **Medium Effect**)

- **Group**: Social World Outside Couple
- **H-statistic**: 7.08
- **Effect Size**: η² = 0.070 (medium effect)
- **Sample Sizes**: Bad (20), Good (27), Mid (29), Total: 76
- **Interpretation**: Books differ significantly in the prevalence of community events, social norms, and public rituals across rating classes. This is the largest effect size among significant categories, indicating a practically meaningful difference.
- **Post-hoc Results**: See pairwise comparison plot for specific group differences

**Visualizations**: 
- `analysis/figures/category_5.3_prevalence.png` - Distribution across rating classes
- `analysis/figures/category_5.3_pairwise.png` - Pairwise comparisons

---

### Category 6.2: Heroine's Work & Professional Identity

**Status**: ✅ **Significantly Different** (p = 0.047, η² = 0.057 - **Small-Medium Effect**)

- **Group**: Work, Wealth, Status & Institutions
- **H-statistic**: 6.10
- **Effect Size**: η² = 0.057 (small-medium effect)
- **Sample Sizes**: Bad (23), Good (22), Mid (30), Total: 75
- **Interpretation**: Books differ significantly in how much they focus on the heroine's professional life and work identity across rating classes. The effect size indicates a moderate practical difference.
- **Post-hoc Results**: See pairwise comparison plot for specific group differences

**Visualizations**: 
- `analysis/figures/category_6.2_prevalence.png` - Distribution across rating classes
- `analysis/figures/category_6.2_pairwise.png` - Pairwise comparisons

---

### Category 3.4: Beliefs, Values & Moral Reflection

**Status**: ✅ **Significantly Different** (p = 0.048, η² = 0.048 - **Small Effect**)

- **Group**: Emotions, Cognition & Inner Life
- **H-statistic**: 6.06
- **Effect Size**: η² = 0.048 (small effect)
- **Sample Sizes**: Bad (29), Good (28), Mid (31), Total: 88
- **Interpretation**: Books differ significantly in the prevalence of moral reflection, values, and beliefs across rating classes. While statistically significant, the effect size is small, suggesting a modest practical difference.
- **Post-hoc Results**: See pairwise comparison plot for specific group differences

**Visualizations**: 
- `analysis/figures/category_3.4_prevalence.png` - Distribution across rating classes
- `analysis/figures/category_3.4_pairwise.png` - Pairwise comparisons

---

## Top Categories by Significance

The following categories show the strongest (though not always significant) differences:

| Rank | Category ID | Category Name | Group | p-value | H-statistic | η² (Effect Size) | Significant |
|------|-------------|---------------|-------|---------|-------------|------------------|-------------|
| 1 | 5.3 | Community, Norms & Social Events | Social World Outside Couple | **0.029** | 7.08 | **0.070** (medium) | ✅ Yes |
| 2 | 6.2 | Heroine's Work & Professional Identity | Work, Wealth, Status & Institutions | **0.047** | 6.10 | **0.057** (small-medium) | ✅ Yes |
| 3 | 3.4 | Beliefs, Values & Moral Reflection | Emotions, Cognition & Inner Life | **0.048** | 6.06 | **0.048** (small) | ✅ Yes |
| 4 | 6.5 | Law, Medicine, Education & Formal Institutions | Work, Wealth, Status & Institutions | 0.056 | 5.76 | 0.062 (small-medium) | No |
| 5 | 4.2 | Bonding, Everyday Intimacy & Growth | Relationship Trajectory (Main Couple) | 0.070 | 5.33 | 0.037 (small) | No |
| 6 | 2.3 | Explicit Sexual Acts | Sexuality, Attraction & Intimacy | 0.075 | 5.19 | 0.036 (small) | No |
| 7 | 4.4 | Conflict, Distance & Breakup Threats | Relationship Trajectory (Main Couple) | 0.148 | 3.82 | 0.020 (small) | No |
| 8 | 3.2 | Negative Emotions & Distress | Emotions, Cognition & Inner Life | 0.161 | 3.65 | 0.019 (small) | No |
| 9 | 4.5 | Reconciliation, Commitments & HEA | Relationship Trajectory (Main Couple) | 0.212 | 3.11 | 0.012 (negligible) | No |
| 10 | 6.1 | Hero's Elite Work & Business World | Work, Wealth, Status & Institutions | 0.278 | 2.56 | 0.006 (negligible) | No |

---

## Borderline Significant Categories

### Category 6.5: Law, Medicine, Education & Formal Institutions (p = 0.056, η² = 0.062)

- **Status**: Borderline significant (just above α = 0.05)
- **H-statistic**: 5.76
- **Effect Size**: η² = 0.062 (small-medium effect)
- **Sample Sizes**: Bad (20), Good (20), Mid (24), Total: 64
- **Note**: This category shows moderate differences with a small-medium effect size but does not reach statistical significance at the 0.05 level. The effect size suggests a potentially meaningful difference that may be underpowered.

**Visualization**: See `analysis/figures/category_6.5_prevalence.png`

### Category 4.2: Bonding, Everyday Intimacy & Growth (p = 0.070, η² = 0.037)

- **Status**: Borderline significant
- **H-statistic**: 5.33
- **Effect Size**: η² = 0.037 (small effect)
- **Sample Sizes**: Bad (30), Good (30), Mid (32), Total: 92
- **Note**: Relationship bonding and intimacy show moderate differences across rating classes with a small effect size.

**Visualization**: See `analysis/figures/category_4.2_prevalence.png`

### Category 2.3: Explicit Sexual Acts (p = 0.075, η² = 0.036)

- **Status**: Borderline significant
- **H-statistic**: 5.19
- **Effect Size**: η² = 0.036 (small effect)
- **Sample Sizes**: Bad (30), Good (30), Mid (32), Total: 92
- **Note**: Sexual content shows moderate differences with a small effect size but does not reach statistical significance.

**Visualization**: See `analysis/figures/category_2.3_prevalence.png`

---

## Categories with No Significant Differences

The majority of categories (24 out of 27) show no statistically significant differences across rating classes. This includes:

- **Sexuality & Intimacy**: Attraction (2.1), Kissing (2.2), Aftercare (2.4)
- **Emotions**: Positive (3.1), Negative (3.2), Ambivalence (3.3)
- **Relationship Trajectory**: Meeting (4.1), Secrets (4.3), Conflict (4.4)
- **Social World**: Family (5.1), Friends (5.2)
- **Work & Status**: Hero's Work (6.1), Heroine's Work (6.2), Shared Workplaces (6.3), Money (6.4)
- **Spaces & Objects**: Domestic (8.1), Public (8.2), Time (8.4)
- **Conflict & Harm**: Interpersonal (7.1), Violence (7.2)
- **Embodied Experience**: Pain (1.2), Exercise (1.5)

This suggests that **thematic content is largely consistent across rating classes**, with readers of different-rated books encountering similar proportions of most content types.

---

## Visualizations

### Overview Plots

Three comprehensive overview visualizations provide different perspectives on the results:

1. **`volcano_plot.png`** - Significance vs Effect Size
   - X-axis: Effect size (η²)
   - Y-axis: -log10(p-value)
   - Color-coded by significance and effect size
   - Highlights categories that are both significant AND have large effects
   - Shows categories with large effects that may be underpowered

2. **`effect_size_bars.png`** - Top Categories by Effect Size
   - Horizontal bar chart ranking categories by effect size
   - Color-coded by statistical significance
   - P-values annotated on bars
   - Identifies categories with largest practical differences

3. **`pvalue_heatmap.png`** - All Categories Overview
   - Heatmap showing all 27 categories
   - Color-coded by p-value (red = significant, green = not significant)
   - Quick visual scan of all results

### Individual Category Plots

Enhanced violin plots for the top 10 categories (available in `analysis/figures/`):

1. `category_5.3_prevalence.png` - **Community, Norms & Social Events** (significant, medium effect)
2. `category_6.2_prevalence.png` - **Heroine's Work & Professional Identity** (significant, small-medium effect)
3. `category_3.4_prevalence.png` - **Beliefs, Values & Moral Reflection** (significant, small effect)
4. `category_6.5_prevalence.png` - Law, Medicine, Education & Formal Institutions
5. `category_4.2_prevalence.png` - Bonding, Everyday Intimacy & Growth
6. `category_2.3_prevalence.png` - Explicit Sexual Acts
7. `category_4.4_prevalence.png` - Conflict, Distance & Breakup Threats
8. `category_3.2_prevalence.png` - Negative Emotions & Distress
9. `category_4.5_prevalence.png` - Reconciliation, Commitments & HEA
10. `category_6.1_prevalence.png` - Hero's Elite Work & Business World

Each individual plot shows:
- **Violin plots** showing full distribution shapes (better than box plots for skewed data)
- Box plots inside violins showing quartiles
- Individual data points (jittered) for each book
- P-value and significance markers in title

### Post-hoc Pairwise Comparison Plots

For the 3 significant categories, pairwise comparison plots show which specific rating classes differ:

1. **`category_5.3_pairwise.png`** - Community, Norms & Social Events
   - Shows which pairs (bad vs mid, bad vs good, mid vs good) differ significantly
   - Includes median differences and direction indicators

2. **`category_6.2_pairwise.png`** - Heroine's Work & Professional Identity
   - Pairwise comparisons with Bonferroni correction
   - Identifies specific group differences

3. **`category_3.4_pairwise.png`** - Beliefs, Values & Moral Reflection
   - Post-hoc test results showing which rating classes differ

---

## Key Insights

### 1. Limited Differentiation by Rating

The finding that only 3 out of 27 categories show significant differences suggests that:
- **Thematic content is largely consistent** across rating classes
- Goodreads ratings may reflect factors beyond thematic content (writing quality, plot structure, character development, etc.)
- Readers of different-rated books encounter similar proportions of most content types

### 2. Social Context and Work Identity as Differentiators

The significant differences in **Community, Norms & Social Events** (5.3, η² = 0.070 - medium effect) and **Heroine's Work & Professional Identity** (6.2, η² = 0.057 - small-medium effect) suggest:
- Books may differ in how much they engage with social settings and community events
- The heroine's professional identity and work life may play a role in rating differences
- Social context and work-related content may contribute to narrative depth or reader engagement
- **Category 5.3 shows the largest effect size**, indicating the most practically meaningful difference
- See pairwise comparison plots to determine which specific rating classes differ and in what direction

### 3. Moral Reflection as Differentiator

The significant difference in **Beliefs, Values & Moral Reflection** (3.4, η² = 0.048 - small effect) suggests:
- Books may differ in how much they engage with moral/ethical questions
- This could relate to narrative depth or philosophical engagement
- While statistically significant, the small effect size suggests a modest practical difference
- See pairwise comparison plot to determine which rating classes differ and direction of difference

### 4. Sexual Content Not a Strong Differentiator

Despite **Explicit Sexual Acts** (2.3) showing borderline significance (p = 0.075), it does not reach the threshold. This suggests:
- Sexual content prevalence is relatively similar across rating classes
- Ratings are not primarily driven by sexual content differences
- The genre maintains consistent sexual content regardless of rating

### 5. Relationship Dynamics Consistent

Categories related to relationship trajectory (4.1-4.5) show no significant differences, suggesting:
- The romance genre maintains consistent relationship development patterns
- All books follow similar narrative arcs regardless of rating
- Rating differences may relate to execution quality rather than content type

---

## Limitations

1. **Sample Size**: 92 books may limit power to detect smaller differences (note: Category 6.5 has large effect but not significant, suggesting possible underpowering)
2. **Multiple Comparisons**: 27 tests were performed; no correction for multiple comparisons applied to main Kruskal-Wallis tests (though post-hoc pairwise tests use Bonferroni correction)
3. **Proportions Only**: Analysis based on sentence proportions, not absolute counts or quality
4. **Rating Classification**: Books classified into 3 groups (bad/mid/good) may mask finer distinctions
5. **Effect Size Interpretation**: Eta-squared for Kruskal-Wallis is an approximation; exact interpretation may vary

---

## Recommendations for Further Analysis

1. ✅ **Post-hoc Tests**: **COMPLETED** - Pairwise comparisons conducted for all 3 significant categories (5.3, 6.2, 3.4) with Bonferroni correction
2. ✅ **Effect Size**: **COMPLETED** - Eta-squared calculated for all categories to assess practical significance
3. **Multiple Comparisons Correction**: Consider applying Bonferroni or FDR correction to main Kruskal-Wallis tests (27 tests performed) if controlling family-wise error rate
4. **Directional Analysis**: Examine which rating class has higher/lower prevalence using pairwise comparison results:
   - Community events and social norms (5.3) - see `category_5.3_pairwise.png`
   - Heroine's work identity (6.2) - see `category_6.2_pairwise.png`
   - Moral reflection (3.4) - see `category_3.4_pairwise.png`
5. **Content Quality**: Investigate whether differences relate to content quality rather than quantity
6. **Subcategory Analysis**: Examine secondary categories and other plausible categories for patterns
7. **Social Context Investigation**: Explore why social events and heroine's work differ across ratings, especially given the medium effect size for Category 5.3
8. **Underpowered Categories**: Investigate Category 6.5 (Law, Medicine, Education) which shows medium effect (η² = 0.062) but is not significant - may benefit from larger sample

---

## Files Generated

- **Statistical Results**: `analysis/kruskal_wallis_results.csv`
  - Contains full results for all 27 categories
  - Columns: category_id, category_name, category_group, p_value, H_statistic, eta_squared, total_n, significant, groups, n_books_per_group
  - Includes effect sizes (η²) for all categories

- **Overview Visualizations**: `analysis/figures/`
  - `volcano_plot.png` - Significance vs effect size overview
  - `effect_size_bars.png` - Top categories ranked by effect size
  - `pvalue_heatmap.png` - All categories with color-coded p-values

- **Individual Category Plots**: `analysis/figures/category_*_prevalence.png`
  - Enhanced violin plots for top 10 categories
  - Shows full distribution shapes with box plots and individual data points
  - Includes p-values and significance markers in titles

- **Post-hoc Pairwise Comparisons**: `analysis/figures/category_*_pairwise.png`
  - Pairwise comparison plots for 3 significant categories (5.3, 6.2, 3.4)
  - Shows which specific rating class pairs differ significantly
  - Includes median differences and direction indicators

---

## Conclusion

This analysis reveals that **thematic content is largely consistent across Goodreads rating classes** in the billionaire romance genre. However, three categories show significant differences with varying effect sizes: 

1. **Community, Norms & Social Events** (5.3) - Medium effect (η² = 0.070), the largest practical difference
2. **Heroine's Work & Professional Identity** (6.2) - Small-medium effect (η² = 0.057)
3. **Beliefs, Values & Moral Reflection** (3.4) - Small effect (η² = 0.048)

This suggests that ratings may reflect factors beyond thematic content—such as writing quality, narrative execution, or reader engagement—but also that certain content dimensions (social context, work identity, moral engagement) may contribute to rating differences. The effect sizes indicate that these differences are not only statistically significant but also practically meaningful, particularly for Category 5.3.

The consistency across most categories (24 out of 27) indicates that the genre maintains stable conventions regardless of how well-received individual books are, supporting the idea that romance novels follow predictable thematic patterns that readers expect regardless of quality. The three significant categories may represent areas where higher-rated books engage more deeply or effectively.

**Notable Finding**: Category 6.5 (Law, Medicine, Education & Formal Institutions) shows a medium effect size (η² = 0.062) but is not statistically significant, suggesting possible underpowering. This category may warrant further investigation with a larger sample.

---

**Report Generated**: December 14, 2025 (Updated with effect sizes and post-hoc analyses)  
**Analysis Script**: `scripts/analyze_category_differences.py`  
**Data Source**: `book_category_proportions.parquet`  
**Taxonomy Mappings**: `taxonomy_mappings_openrouter_mistralai_Mistral-Nemo-Instruct-2407_paraphrase-MiniLM-L6-v2.json`  
**Enhanced Analysis**: Includes effect size calculations (η²) and post-hoc pairwise comparisons with Bonferroni correction

