# Taxonomy Group Landscape Across Popularity Tiers: Broad Thematic Allocation, Fine-Grained Subgroup Drivers, and Diversity Patterns

## Abstract

This analysis examines how thematic content allocation differs across popularity tiers (Top/Mid/Trash) in billionaire romance novels. We aggregated normalized book-level topic probability mass into a theory-guided taxonomy with 8 main groups and ~27 subgroups. Each book's total modeled probability mass is effectively 1.0 (after upstream normalization), meaning group shares can be interpreted as "how much of the book's thematic attention goes to X" rather than an artifact of missing probability mass.

We report descriptive shifts across tiers, nonparametric comparisons (Kruskal–Wallis + Top vs Trash Mann–Whitney), and effect sizes (Cliff's delta) as the primary interpretation tool, given that small N makes p-values a weak success criterion.

## Data and Methods

### Data Structure

- **Sample size**: N = 92 books (Top/Mid/Trash tiers)
- **Taxonomy structure**: 8 main groups, ~27 subgroups
- **Normalization**: Book-level probability mass sums to 1.0
- **Coverage**: Modeled mass ≈ 0.998 (very high coverage; unmapped/noise/paratext shares are extremely small, ≈ 0.000–0.002 range)

### Statistical Approach

1. **Descriptive statistics**: Median, mean, quartiles by tier
2. **Nonparametric tests**: 
   - Kruskal–Wallis for overall tier differences
   - Mann–Whitney U for Top vs Trash comparisons
3. **Effect sizes**: Cliff's delta (δ) as primary interpretation metric
4. **Multiple comparisons**: Family-wise adjusted p-values where applicable

## 1. Main-Group Landscape (8 Macro Buckets)

### 1.1 What Differs at the Broadest Level?

At the main-group level, differences are present but modest (as expected: these are very large buckets that average many subtypes).

The strongest Top vs Trash effect sizes (Cliff's δ) are:

1. **Relationship Trajectory (Main Couple)**: Top higher (δ ≈ +0.37)
   - Top mean ≈ 0.365 vs Trash mean ≈ 0.350

2. **Social World Outside Couple**: Top higher (δ ≈ +0.30)

3. **Embodied & Sensory Experience**: Top higher (δ ≈ +0.29)

4. **Sexuality, Attraction & Intimacy**: Trash higher (δ ≈ −0.25)

5. **Emotions, Cognition & Inner Life**: slightly Trash higher (δ ≈ −0.14; weak)

These are directionally coherent with the inference story: Trash is more "sex/intimacy mass" heavy, while Top puts relatively more mass into relationship mechanics, social embedding, and embodied/sensory texture.

![Main Group Composition by Tier](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_T1_main_group_composition.png)

*Figure 1: Main group shares by tier. Small but interpretable shifts without overstating "significance."*

### 1.2 Plain Language Interpretation

At the broadest level, higher-tier books do not radically change what they talk about (romance is still romance). Instead, they slightly rebalance where thematic attention goes:

- **A bit more** toward relationship dynamics and social context
- **A bit less** toward pure sexuality/intimacy mass

This is the classic "allocation shift" pattern expected in a genre where most books share a common skeleton.

**Table 1: Main Group Comparisons**

See: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/tables/Table_T1_main_group_comparisons.csv`

![Main Group Effect Sizes Heatmap](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_T2_main_group_heatmap.png)

*Figure 2: Effect sizes (Cliff's delta) for main group comparisons across tiers.*

## 2. Subgroup Landscape (27 Finer Buckets)

This is where the signal sharpens. Several subgroups show meaningful Top vs Trash differentiation, and some survive within-family correction more strongly than main groups.

### 2.1 Strongest Subgroup Effects (Top vs Trash)

#### The Clearest Top-Associated Subgroup

**Beliefs, Values & Moral Reflection** (Emotions/Cognition/Inner Life)
- δ ≈ +0.46
- Top mean ≈ 0.134 vs Trash mean ≈ 0.116
- Adjusted p (family) ≈ 0.008

This is a strong and interpretable result: Top books allocate more mass to value/identity/moral reflection moments.

#### Other Notable Subgroup Differences

Directional differences (weaker after correction but still notable):

1. **Negative Emotions & Distress**: Trash higher (δ ≈ −0.37)
2. **Time/Seasons/Temporal framing**: Top higher (δ ≈ +0.36)
3. **Shared Workplaces & Professional Interaction**: Top higher (δ ≈ +0.35)
4. **Public & Leisure Spaces**: Trash higher (δ ≈ −0.28)
5. **Interpersonal Non-Romantic Conflict**: Top higher (δ ≈ +0.30)
6. **Violence/Threats/Coercion**: Trash higher (δ ≈ −0.30)

![Subgroup Effect Sizes](results/stage10_correlation_analysis/02_taxonomy_group_analysis/figures/subgroup_distributions/subgroup_effect_sizes_heatmap.png)

*Figure 3: Subgroup effect sizes (Cliff's delta) for Top vs Trash comparisons. Strongest effects highlighted.*

**Table 2: Subgroup Comparisons (Sorted)**

See: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/tables/Table_S1_subgroup_comparisons.csv`

### 2.2 Interpretation: What These Subgroup Patterns Mean

A simple interpretation that avoids overreach:

**Top-tier books lean into:**
- Values/identity reflection (moral reasoning, self-concept, beliefs)
- Professional/workplace interaction (institutional social texture)
- Time framing (temporal structuring cues)
- Non-romantic conflict/social embedding

**Trash-tier books lean into:**
- Distress/negative emotion intensity (panic, sobbing, emotional dysregulation)
- Threat/coercion/violence texture (risk/harm)
- Public/leisure spaces + procedural scene motion (signals of pacing/scene filler)
- Heavier sex/intimacy mass at the macro level

This aligns well with the "two channels" story: reach is not purely "more sex," quality beyond reach tends to relate to payoff/safety and coherent pacing, while "darkness intensity" can behave differently depending on arc timing.

## 3. Micro-Scene Drivers (Concrete Examples for Writing + Interpretation)

The `subgroup_topic_drivers.csv` provides topic-level examples explaining why some subgroups differ. This is perfect for a paper because it turns an abstract subgroup into recognizable narrative moments.

### 3.1 Beliefs/Values/Moral Reflection (Top higher)

Top-driving topics include:

- **"Feminist Identity Affirmation"** (topic 310)
- **"Sisters Reflecting on Childhood"** (topic 350)
- **"Belief Discussion"** (topic 249)

**Plain-language interpretation**: Top books include more explicit "who am I / what do I value / what do we believe" moments, often embedded in intimate dialogue scenes.

![Topic Drivers: Beliefs, Values & Moral Reflection](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_D3_drivers_Beliefs_ Values _ Moral Reflection.png)

*Figure 4: Topic-level drivers for the Beliefs, Values & Moral Reflection subgroup. Topics with positive Cliff's delta are more prevalent in Top-tier books.*

**Table 3: Topic Drivers for Beliefs, Values & Moral Reflection**

See: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/tables/Table_D3_drivers_Beliefs_ Values _ Moral Reflection.csv`

### 3.2 Negative Emotions & Distress (Trash higher)

Trash-driving examples include topics where trash > top, such as:

- **"Emotional Panic Attack"** (topic 160; negative Cliff's δ)

**Interpretation**: Trash-tier books allocate more mass to high-intensity distress cues (panic, breakdown) rather than reflective processing.

![Topic Drivers: Negative Emotions & Distress](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_D4_drivers_Negative Emotions _ Distress.png)

*Figure 5: Topic-level drivers for the Negative Emotions & Distress subgroup. Topics with negative Cliff's delta are more prevalent in Trash-tier books.*

**Table 4: Topic Drivers for Negative Emotions & Distress**

See: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/tables/Table_D4_drivers_Negative Emotions _ Distress.csv`

### 3.3 Professional Interaction (Top higher)

Top-driving example:

- **"Office Work at Desk"** (topic 212)

**Interpretation**: This doesn't mean "desk work = quality." It means Top books more often situate relationship dynamics inside a workplace/institutional setting (social context + constraints), which is consistent with billionaire-romance scaffolding.

![Topic Drivers: Shared Workplaces & Professional Interaction](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_D14_drivers_Shared Workplaces _ Professional Interaction.png)

*Figure 6: Topic-level drivers for the Shared Workplaces & Professional Interaction subgroup.*

**Table 5: Topic Drivers for Shared Workplaces & Professional Interaction**

See: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/tables/Table_D14_drivers_Shared Workplaces _ Professional Interaction.csv`

## 4. Coverage and Mapping Quality (Sanity Checks)

The coverage audit indicates that "non-thematic" buckets are tiny:

- **Modeled mass** ≈ 0.998 (very high)
- **Unmapped/noise/paratext/other shares** are all extremely small (≈ 0.000–0.002 range)

**Interpretation**: Subgroup comparisons are not being driven by "missing mapping" artifacts.

![Coverage Distribution](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_C1_coverage_distribution.png)

*Figure 7: Coverage audit showing high modeled mass across all tiers. Non-thematic components are negligible.*

**Table 6: Coverage Audit by Tier**

See: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/tables_csv/coverage_audit_by_tier.csv`

## 5. Diversity and Dispersion: Do Better Books Use More Topic Variety?

This is one of the most theory-relevant and intuitive findings.

### 5.1 Tier Differences in Diversity

Median values show a clear monotonic trend:

- **Entropy**: bad ≈ 5.33 → mid ≈ 5.43 → good ≈ 5.48
- **Effective topics**: bad ≈ 207 → mid ≈ 229 → good ≈ 240
- **Richness** (topics > 1e-3): bad ≈ 247 → mid ≈ 258 → good ≈ 265
- **HHI**: lower in good (more distributed)

Kruskal tests are suggestive:
- entropy/effective_topics p ≈ 0.019 (adjusted ≈ 0.077)
- richness p ≈ 0.015 (adjusted ≈ 0.073)

So it's not "conclusive" at N≈92, but it is directionally stable and coherent: **higher-tier books appear more thematically diverse / less concentrated**.

![Diversity Metrics by Tier](results/stage10_correlation_analysis/02_taxonomy_group_analysis/figures/diversity_metrics/fig_diversity_by_tier_faceted.png)

*Figure 8: Diversity metrics (entropy, effective topics, richness, HHI) by tier. Higher-tier books show greater thematic diversity.*

**Table 7: Diversity Tests**

See: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/tables_csv/diversity_tests.csv`

### 5.2 Interpretation

This supports a clean hypothesis for the large corpus: **Higher-quality books may succeed partly by combining more thematic ingredients (or distributing attention more evenly), rather than over-concentrating on a few motifs.**

This is especially useful because it's not tied to any single romance trope — it's a structural property.

![Diversity vs Other Metrics](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_C2_diversity_vs_other.png)

*Figure 9: Relationship between diversity metrics and other book characteristics.*

## 6. Supplementary Material: Subgroup Panels by Main Group

For detailed examination, subgroup distributions within each main group are available:

- **Conflict, Risk & Harm**: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_S1a_subgroup_panel_Conflict_ Risk _ Harm.png`
- **Embodied & Sensory Experience**: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_S1b_subgroup_panel_Embodied _ Sensory Experience.png`
- **Emotions, Cognition & Inner Life**: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_S1c_subgroup_panel_Emotions_ Cognition _ Inner Life.png`
- **Relationship Trajectory (Main Couple)**: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_S1d_subgroup_panel_Relationship Trajectory _Main Couple_.png`
- **Sexuality, Attraction & Intimacy**: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_S1e_subgroup_panel_Sexuality_ Attraction _ Intimacy.png`
- **Social World Outside Couple**: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_S1f_subgroup_panel_Social World Outside Couple.png`
- **Spaces, Time, Activities & Objects**: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_S1g_subgroup_panel_Spaces_ Time_ Activities _ Objects.png`
- **Work, Wealth, Status & Institutions**: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_S1h_subgroup_panel_Work_ Wealth_ Status _ Institutions.png`

## 7. Summary and Conclusions

### Key Findings

1. **Main groups**: Broad groups barely differ (expected), but directionally: Top books allocate slightly more to relationship + social embedding; Trash allocates more to sexuality mass.

2. **Subgroups**: Show sharper structure: Top is higher on values/moral reflection and professional context; Trash is higher on distress intensity and coercion/threat texture.

3. **Diversity**: Better books look more diverse (higher entropy/effective topics), suggesting a structural "variety/dispersion" hypothesis for the large corpus.

### Implications

The thematic allocation patterns suggest that successful billionaire romance novels achieve popularity not by fundamentally changing genre conventions, but by:

1. **Rebalancing attention** toward relationship mechanics and social embedding
2. **Incorporating value/identity reflection** moments that add depth
3. **Maintaining thematic diversity** rather than over-concentrating on a few motifs
4. **Moderating intensity** of distress/coercion elements while maintaining narrative tension

These findings provide actionable insights for understanding what distinguishes higher-tier books in this genre, while acknowledging that the genre's core structure remains consistent across popularity levels.

## References

All data, code, and supplementary materials are available in:
- Notebook: `notebooks/07_analysis/02_taxonomy_group_analysis/02_taxonomy_group_analysis_v2_contract_normalized.ipynb`
- Results: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/`

