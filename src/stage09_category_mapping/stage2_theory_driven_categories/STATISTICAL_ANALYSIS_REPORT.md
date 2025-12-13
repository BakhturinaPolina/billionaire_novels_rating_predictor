# Statistical Analysis Report: Taxonomy Category Differences Across Rating Classes

**Date**: December 13, 2025  
**Analysis**: Kruskal-Wallis tests for category prevalence differences  
**Dataset**: 92 books across 3 rating classes (bad, mid, good)  
**Categories Analyzed**: 28 taxonomy categories

---

## Executive Summary

This report presents statistical analysis of taxonomy category prevalence differences across book rating classes using Kruskal-Wallis tests. The analysis examined 28 categories from the Romance Corpus Topic Taxonomy across 92 books classified into three rating groups: bad (30 books), mid (32 books), and good (30 books).

**Key Finding**: Only **1 category** shows statistically significant differences (p < 0.05) across rating classes:
- **3.4: Beliefs, Values & Moral Reflection** (p = 0.018)

This suggests that most thematic content categories are similarly distributed across books regardless of their Goodreads ratings, with moral reflection being the primary distinguishing factor.

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

### Analysis Pipeline
1. Load book-level category proportions from `book_category_proportions.parquet`
2. Run Kruskal-Wallis test for each category
3. Sort results by p-value
4. Generate visualizations for top 15 categories
5. Report significant findings

---

## Significant Results

### Category 3.4: Beliefs, Values & Moral Reflection

**Status**: ✅ **Significantly Different** (p = 0.018)

- **Group**: Emotions, Cognition & Inner Life
- **H-statistic**: 8.07
- **Sample Sizes**: Bad (29), Good (28), Mid (32)
- **Interpretation**: Books differ significantly in the prevalence of moral reflection, values, and beliefs across rating classes.

**Visualization**: See `analysis/figures/category_3.4_prevalence.png`

---

## Top Categories by Significance

The following categories show the strongest (though not always significant) differences:

| Rank | Category ID | Category Name | Group | p-value | H-statistic | Significant |
|------|-------------|---------------|-------|---------|-------------|-------------|
| 1 | 3.4 | Beliefs, Values & Moral Reflection | Emotions, Cognition & Inner Life | **0.018** | 8.07 | ✅ Yes |
| 2 | 2.3 | Explicit Sexual Acts | Sexuality, Attraction & Intimacy | 0.051 | 5.95 | No |
| 3 | 5.3 | Community, Norms & Social Events | Social World Outside Couple | 0.054 | 5.82 | No |
| 4 | 4.5 | Reconciliation, Commitments & HEA | Relationship Trajectory (Main Couple) | 0.104 | 4.53 | No |
| 5 | 4.2 | Bonding, Everyday Intimacy & Growth | Relationship Trajectory (Main Couple) | 0.107 | 4.47 | No |
| 6 | 1.1 | Body Parts & Physical Reactions | Embodied & Sensory Experience | 0.133 | 4.03 | No |
| 7 | 6.5 | Law, Medicine, Education & Formal Institutions | Work, Wealth, Status & Institutions | 0.139 | 3.94 | No |
| 8 | 5.2 | Friends & Social Circles | Social World Outside Couple | 0.152 | 3.77 | No |
| 9 | 4.4 | Conflict, Distance & Breakup Threats | Relationship Trajectory (Main Couple) | 0.185 | 3.38 | No |
| 10 | 8.3 | Objects, Technology & Everyday Artefacts | Spaces, Time, Activities & Objects | 0.363 | 2.03 | No |

---

## Borderline Significant Categories

### Category 2.3: Explicit Sexual Acts (p = 0.051)

- **Status**: Borderline significant (just above α = 0.05)
- **H-statistic**: 5.95
- **Sample Sizes**: Bad (30), Good (30), Mid (32)
- **Note**: This category shows the second-strongest difference but does not reach statistical significance at the 0.05 level.

**Visualization**: See `analysis/figures/category_2.3_prevalence.png`

### Category 5.3: Community, Norms & Social Events (p = 0.054)

- **Status**: Borderline significant
- **H-statistic**: 5.82
- **Sample Sizes**: Bad (28), Good (29), Mid (32)
- **Note**: Social events and community norms show moderate differences across rating classes.

**Visualization**: See `analysis/figures/category_5.3_prevalence.png`

---

## Categories with No Significant Differences

The majority of categories (27 out of 28) show no statistically significant differences across rating classes. This includes:

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

Visualizations for the top 15 categories are available in `analysis/figures/`:

1. `category_3.4_prevalence.png` - **Beliefs, Values & Moral Reflection** (significant)
2. `category_2.3_prevalence.png` - Explicit Sexual Acts
3. `category_5.3_prevalence.png` - Community, Norms & Social Events
4. `category_4.5_prevalence.png` - Reconciliation, Commitments & HEA
5. `category_4.2_prevalence.png` - Bonding, Everyday Intimacy & Growth
6. `category_1.1_prevalence.png` - Body Parts & Physical Reactions
7. `category_6.5_prevalence.png` - Law, Medicine, Education & Formal Institutions
8. `category_5.2_prevalence.png` - Friends & Social Circles
9. `category_4.4_prevalence.png` - Conflict, Distance & Breakup Threats
10. `category_8.3_prevalence.png` - Objects, Technology & Everyday Artefacts
11. `category_6.2_prevalence.png` - Heroine's Work & Professional Identity
12. `category_2.2_prevalence.png` - Kissing & Non-Explicit Affection
13. `category_3.3_prevalence.png` - Ambivalence & Internal Conflict
14. `category_6.3_prevalence.png` - Shared Workplaces & Professional Interaction
15. `category_8.2_prevalence.png` - Public & Leisure Spaces

Each visualization shows:
- Box plots for each rating class (bad, mid, good)
- Individual data points (jittered) for each book
- Distribution of category proportions across books

---

## Key Insights

### 1. Limited Differentiation by Rating

The finding that only 1 out of 28 categories shows significant differences suggests that:
- **Thematic content is remarkably consistent** across rating classes
- Goodreads ratings may reflect factors beyond thematic content (writing quality, plot structure, character development, etc.)
- Readers of different-rated books encounter similar proportions of most content types

### 2. Moral Reflection as Differentiator

The significant difference in **Beliefs, Values & Moral Reflection** (3.4) suggests:
- Books may differ in how much they engage with moral/ethical questions
- This could relate to narrative depth or philosophical engagement
- Further investigation needed to determine direction of difference (which rating class has more/less)

### 3. Sexual Content Not a Strong Differentiator

Despite **Explicit Sexual Acts** (2.3) showing borderline significance (p = 0.051), it does not reach the threshold. This suggests:
- Sexual content prevalence is relatively similar across rating classes
- Ratings are not primarily driven by sexual content differences
- The genre maintains consistent sexual content regardless of rating

### 4. Relationship Dynamics Consistent

Categories related to relationship trajectory (4.1-4.5) show no significant differences, suggesting:
- The romance genre maintains consistent relationship development patterns
- All books follow similar narrative arcs regardless of rating
- Rating differences may relate to execution quality rather than content type

---

## Limitations

1. **Sample Size**: 92 books may limit power to detect smaller differences
2. **Non-parametric Test**: Kruskal-Wallis tests for distribution differences but doesn't indicate direction or magnitude
3. **Multiple Comparisons**: 28 tests were performed; no correction for multiple comparisons applied
4. **Proportions Only**: Analysis based on sentence proportions, not absolute counts or quality
5. **Rating Classification**: Books classified into 3 groups (bad/mid/good) may mask finer distinctions

---

## Recommendations for Further Analysis

1. **Post-hoc Tests**: Conduct pairwise comparisons for Category 3.4 to determine which rating classes differ
2. **Effect Size**: Calculate effect sizes (e.g., eta-squared) to assess practical significance
3. **Multiple Comparisons Correction**: Apply Bonferroni or FDR correction if testing multiple hypotheses
4. **Directional Analysis**: Examine which rating class has higher/lower prevalence of moral reflection
5. **Content Quality**: Investigate whether differences relate to content quality rather than quantity
6. **Subcategory Analysis**: Examine secondary categories and other plausible categories for patterns

---

## Files Generated

- **Statistical Results**: `analysis/kruskal_wallis_results.csv`
  - Contains full results for all 28 categories
  - Columns: category_id, category_name, category_group, p_value, H_statistic, significant, groups, n_books_per_group

- **Visualizations**: `analysis/figures/category_*.png`
  - 15 box plots with jitter for top categories
  - Shows distribution of category proportions across rating classes

---

## Conclusion

This analysis reveals that **thematic content is largely consistent across Goodreads rating classes** in the billionaire romance genre. Only moral reflection shows significant differences, suggesting that ratings may reflect factors beyond thematic content—such as writing quality, narrative execution, or reader engagement—rather than the types of content present.

The consistency across most categories indicates that the genre maintains stable conventions regardless of how well-received individual books are, supporting the idea that romance novels follow predictable thematic patterns that readers expect regardless of quality.

---

**Report Generated**: December 13, 2025  
**Analysis Script**: `scripts/analyze_category_differences.py`  
**Data Source**: `book_category_proportions.parquet`

