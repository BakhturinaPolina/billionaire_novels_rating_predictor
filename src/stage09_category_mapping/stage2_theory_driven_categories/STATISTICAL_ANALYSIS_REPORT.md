# Statistical Analysis Report: Taxonomy Category Differences Across Rating Classes

**Date**: December 14, 2025  
**Analysis**: Kruskal-Wallis tests for category prevalence differences  
**Dataset**: 92 books across 3 rating classes (bad, mid, good)  
**Categories Analyzed**: 27 taxonomy categories

---

## Executive Summary

This report presents statistical analysis of taxonomy category prevalence differences across book rating classes using Kruskal-Wallis tests. The analysis examined 27 categories from the Romance Corpus Topic Taxonomy across 92 books classified into three rating groups: bad (20-30 books per category), mid (22-32 books per category), and good (27-32 books per category).

**Key Finding**: **3 categories** show statistically significant differences (p < 0.05) across rating classes:
- **5.3: Community, Norms & Social Events** (p = 0.029)
- **6.2: Heroine's Work & Professional Identity** (p = 0.047)
- **3.4: Beliefs, Values & Moral Reflection** (p = 0.048)

This suggests that while most thematic content categories are similarly distributed across books regardless of their Goodreads ratings, social events, heroine's work, and moral reflection serve as distinguishing factors.

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

### Category 5.3: Community, Norms & Social Events

**Status**: ✅ **Significantly Different** (p = 0.029)

- **Group**: Social World Outside Couple
- **H-statistic**: 7.08
- **Sample Sizes**: Bad (20), Good (27), Mid (29)
- **Interpretation**: Books differ significantly in the prevalence of community events, social norms, and public rituals across rating classes.

**Visualization**: See `analysis/figures/category_5.3_prevalence.png`

---

### Category 6.2: Heroine's Work & Professional Identity

**Status**: ✅ **Significantly Different** (p = 0.047)

- **Group**: Work, Wealth, Status & Institutions
- **H-statistic**: 6.10
- **Sample Sizes**: Bad (23), Good (22), Mid (30)
- **Interpretation**: Books differ significantly in how much they focus on the heroine's professional life and work identity across rating classes.

**Visualization**: See `analysis/figures/category_6.2_prevalence.png`

---

### Category 3.4: Beliefs, Values & Moral Reflection

**Status**: ✅ **Significantly Different** (p = 0.048)

- **Group**: Emotions, Cognition & Inner Life
- **H-statistic**: 6.06
- **Sample Sizes**: Bad (29), Good (28), Mid (31)
- **Interpretation**: Books differ significantly in the prevalence of moral reflection, values, and beliefs across rating classes.

**Visualization**: See `analysis/figures/category_3.4_prevalence.png`

---

## Top Categories by Significance

The following categories show the strongest (though not always significant) differences:

| Rank | Category ID | Category Name | Group | p-value | H-statistic | Significant |
|------|-------------|---------------|-------|---------|-------------|-------------|
| 1 | 5.3 | Community, Norms & Social Events | Social World Outside Couple | **0.029** | 7.08 | ✅ Yes |
| 2 | 6.2 | Heroine's Work & Professional Identity | Work, Wealth, Status & Institutions | **0.047** | 6.10 | ✅ Yes |
| 3 | 3.4 | Beliefs, Values & Moral Reflection | Emotions, Cognition & Inner Life | **0.048** | 6.06 | ✅ Yes |
| 4 | 6.5 | Law, Medicine, Education & Formal Institutions | Work, Wealth, Status & Institutions | 0.056 | 5.76 | No |
| 5 | 4.2 | Bonding, Everyday Intimacy & Growth | Relationship Trajectory (Main Couple) | 0.070 | 5.33 | No |
| 6 | 2.3 | Explicit Sexual Acts | Sexuality, Attraction & Intimacy | 0.075 | 5.19 | No |
| 7 | 4.4 | Conflict, Distance & Breakup Threats | Relationship Trajectory (Main Couple) | 0.148 | 3.82 | No |
| 8 | 3.2 | Negative Emotions & Distress | Emotions, Cognition & Inner Life | 0.161 | 3.65 | No |
| 9 | 4.5 | Reconciliation, Commitments & HEA | Relationship Trajectory (Main Couple) | 0.212 | 3.11 | No |
| 10 | 6.1 | Hero's Elite Work & Business World | Work, Wealth, Status & Institutions | 0.278 | 2.56 | No |

---

## Borderline Significant Categories

### Category 6.5: Law, Medicine, Education & Formal Institutions (p = 0.056)

- **Status**: Borderline significant (just above α = 0.05)
- **H-statistic**: 5.76
- **Sample Sizes**: Bad (20), Good (20), Mid (24)
- **Note**: This category shows moderate differences but does not reach statistical significance at the 0.05 level.

**Visualization**: See `analysis/figures/category_6.5_prevalence.png`

### Category 4.2: Bonding, Everyday Intimacy & Growth (p = 0.070)

- **Status**: Borderline significant
- **H-statistic**: 5.33
- **Sample Sizes**: Bad (30), Good (30), Mid (32)
- **Note**: Relationship bonding and intimacy show moderate differences across rating classes.

**Visualization**: See `analysis/figures/category_4.2_prevalence.png`

### Category 2.3: Explicit Sexual Acts (p = 0.075)

- **Status**: Borderline significant
- **H-statistic**: 5.19
- **Sample Sizes**: Bad (30), Good (30), Mid (32)
- **Note**: Sexual content shows moderate differences but does not reach statistical significance.

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

The finding that only 3 out of 27 categories show significant differences suggests that:
- **Thematic content is largely consistent** across rating classes
- Goodreads ratings may reflect factors beyond thematic content (writing quality, plot structure, character development, etc.)
- Readers of different-rated books encounter similar proportions of most content types

### 2. Social Context and Work Identity as Differentiators

The significant differences in **Community, Norms & Social Events** (5.3) and **Heroine's Work & Professional Identity** (6.2) suggest:
- Books may differ in how much they engage with social settings and community events
- The heroine's professional identity and work life may play a role in rating differences
- Social context and work-related content may contribute to narrative depth or reader engagement
- Further investigation needed to determine direction of differences (which rating class has more/less)

### 3. Moral Reflection as Differentiator

The significant difference in **Beliefs, Values & Moral Reflection** (3.4) suggests:
- Books may differ in how much they engage with moral/ethical questions
- This could relate to narrative depth or philosophical engagement
- Further investigation needed to determine direction of difference (which rating class has more/less)

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

1. **Sample Size**: 92 books may limit power to detect smaller differences
2. **Non-parametric Test**: Kruskal-Wallis tests for distribution differences but doesn't indicate direction or magnitude
3. **Multiple Comparisons**: 28 tests were performed; no correction for multiple comparisons applied
4. **Proportions Only**: Analysis based on sentence proportions, not absolute counts or quality
5. **Rating Classification**: Books classified into 3 groups (bad/mid/good) may mask finer distinctions

---

## Recommendations for Further Analysis

1. **Post-hoc Tests**: Conduct pairwise comparisons for the 3 significant categories (5.3, 6.2, 3.4) to determine which rating classes differ
2. **Effect Size**: Calculate effect sizes (e.g., eta-squared) to assess practical significance
3. **Multiple Comparisons Correction**: Apply Bonferroni or FDR correction if testing multiple hypotheses (27 tests performed)
4. **Directional Analysis**: Examine which rating class has higher/lower prevalence of:
   - Community events and social norms (5.3)
   - Heroine's work identity (6.2)
   - Moral reflection (3.4)
5. **Content Quality**: Investigate whether differences relate to content quality rather than quantity
6. **Subcategory Analysis**: Examine secondary categories and other plausible categories for patterns
7. **Social Context Investigation**: Explore why social events and heroine's work differ across ratings

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

This analysis reveals that **thematic content is largely consistent across Goodreads rating classes** in the billionaire romance genre. However, three categories show significant differences: social events/community norms, heroine's work identity, and moral reflection. This suggests that ratings may reflect factors beyond thematic content—such as writing quality, narrative execution, or reader engagement—but also that certain content dimensions (social context, work identity, moral engagement) may contribute to rating differences.

The consistency across most categories (24 out of 27) indicates that the genre maintains stable conventions regardless of how well-received individual books are, supporting the idea that romance novels follow predictable thematic patterns that readers expect regardless of quality. The three significant categories may represent areas where higher-rated books engage more deeply or effectively.

---

**Report Generated**: December 14, 2025  
**Analysis Script**: `scripts/analyze_category_differences.py`  
**Data Source**: `book_category_proportions.parquet`  
**Taxonomy Mappings**: `taxonomy_mappings_openrouter_mistralai_Mistral-Nemo-Instruct-2407_paraphrase-MiniLM-L6-v2.json`

