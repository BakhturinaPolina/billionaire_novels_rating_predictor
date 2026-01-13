# Taxonomy Group Landscape Across Popularity Tiers
## Broad Thematic Allocation, Fine-Grained Subgroup Drivers, and Diversity Patterns

---

## Slide 1: Title Slide

**Taxonomy Group Landscape Across Popularity Tiers**

Broad Thematic Allocation, Fine-Grained Subgroup Drivers, and Diversity Patterns

*Billionaire Romance Novels Analysis*

---

## Slide 2: Research Question

**What differs thematically between Top, Mid, and Trash-tier books?**

- Aggregated normalized book-level topic probability mass
- Theory-guided taxonomy: **8 main groups**, **~27 subgroups**
- Each book's probability mass ≈ 1.0 (normalized)
- Group shares = "how much thematic attention goes to X"

**Sample**: N = 92 books across three popularity tiers

---

## Slide 3: Methods Overview

**Statistical Approach**

1. **Descriptive shifts** across tiers (good/mid/bad)
2. **Nonparametric comparisons**: 
   - Kruskal–Wallis (overall differences)
   - Mann–Whitney (Top vs Trash)
3. **Effect sizes**: Cliff's delta (δ) as primary metric
4. **Multiple comparisons**: Family-wise adjusted p-values

**Why effect sizes?** Small N makes p-values a weak success criterion

---

## Slide 4: Main Groups - The Big Picture

**At the broadest level: differences are modest but interpretable**

![Main Group Composition](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_T1_main_group_composition.png)

**Key insight**: Romance is still romance. Higher-tier books **rebalance** where attention goes, not what they talk about.

---

## Slide 5: Main Groups - Effect Sizes

**Strongest Top vs Trash differences (Cliff's δ):**

1. **Relationship Trajectory (Main Couple)**: Top higher (δ ≈ +0.37)
2. **Social World Outside Couple**: Top higher (δ ≈ +0.30)
3. **Embodied & Sensory Experience**: Top higher (δ ≈ +0.29)
4. **Sexuality, Attraction & Intimacy**: Trash higher (δ ≈ −0.25)

![Main Group Heatmap](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_T2_main_group_heatmap.png)

---

## Slide 6: Main Groups - Plain Language

**Top-tier books allocate:**
- ✅ **More** to relationship dynamics and social context
- ✅ **More** to embodied/sensory texture

**Trash-tier books allocate:**
- ❌ **More** to pure sexuality/intimacy mass

**This is the classic "allocation shift" pattern** in a genre with shared skeleton

---

## Slide 7: Subgroups - Where Signal Sharpens

**Subgroup level shows clearer differentiation**

![Subgroup Effect Sizes](results/stage10_correlation_analysis/02_taxonomy_group_analysis/figures/subgroup_distributions/subgroup_effect_sizes_heatmap.png)

**Strongest effects survive family-wise correction**

---

## Slide 8: Strongest Subgroup Effect

**Beliefs, Values & Moral Reflection** (Top higher)

- **Effect size**: δ ≈ +0.46
- **Top mean**: 0.134 vs **Trash mean**: 0.116
- **Adjusted p**: ≈ 0.008

**Interpretation**: Top books allocate more mass to value/identity/moral reflection moments

![Topic Drivers: Beliefs & Values](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_D3_drivers_Beliefs_ Values _ Moral Reflection.png)

---

## Slide 9: Other Notable Subgroup Differences

**Top-tier books lean into:**
- ✅ Values/identity reflection (moral reasoning, self-concept)
- ✅ Professional/workplace interaction
- ✅ Time framing (temporal structuring)
- ✅ Non-romantic conflict/social embedding

**Trash-tier books lean into:**
- ❌ Distress/negative emotion intensity (panic, sobbing)
- ❌ Threat/coercion/violence texture
- ❌ Public/leisure spaces (pacing/scene filler)

---

## Slide 10: Concrete Examples - Beliefs & Values

**Top-driving topics:**

- **"Feminist Identity Affirmation"** (topic 310)
- **"Sisters Reflecting on Childhood"** (topic 350)
- **"Belief Discussion"** (topic 249)

**What this means**: Top books include more explicit "who am I / what do I value" moments, often in intimate dialogue scenes

![Beliefs & Values Drivers](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_D3_drivers_Beliefs_ Values _ Moral Reflection.png)

---

## Slide 11: Concrete Examples - Negative Emotions

**Trash-driving topics:**

- **"Emotional Panic Attack"** (topic 160)
- High-intensity distress cues (panic, breakdown)

**What this means**: Trash-tier books allocate more mass to high-intensity distress rather than reflective processing

![Negative Emotions Drivers](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_D4_drivers_Negative Emotions _ Distress.png)

---

## Slide 12: Concrete Examples - Professional Context

**Top-driving example:**

- **"Office Work at Desk"** (topic 212)

**What this means**: Top books situate relationship dynamics inside workplace/institutional settings (social context + constraints)

*Not "desk work = quality"* — but institutional embedding matters

![Professional Interaction Drivers](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_D14_drivers_Shared Workplaces _ Professional Interaction.png)

---

## Slide 13: Coverage Quality Check

**Sanity check: mapping is comprehensive**

- **Modeled mass** ≈ 0.998 (very high)
- **Unmapped/noise/paratext** ≈ 0.000–0.002 (negligible)

**Interpretation**: Subgroup comparisons are NOT driven by missing mapping artifacts

![Coverage Distribution](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_C1_coverage_distribution.png)

---

## Slide 14: Diversity - The Key Finding

**Do better books use more topic variety?**

**Median values show clear monotonic trend:**

- **Entropy**: bad 5.33 → mid 5.43 → **good 5.48**
- **Effective topics**: bad 207 → mid 229 → **good 240**
- **Richness**: bad 247 → mid 258 → **good 265**

**Kruskal tests**: p ≈ 0.019 (adjusted ≈ 0.077)

![Diversity by Tier](results/stage10_correlation_analysis/02_taxonomy_group_analysis/figures/diversity_metrics/fig_diversity_by_tier_faceted.png)

---

## Slide 15: Diversity - Interpretation

**Higher-tier books are more thematically diverse / less concentrated**

**Hypothesis for large corpus**: Higher-quality books succeed partly by:
- Combining more thematic ingredients
- Distributing attention more evenly
- Avoiding over-concentration on a few motifs

**This is a structural property** — not tied to any single romance trope

![Diversity Metrics](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_C2_diversity_vs_other.png)

---

## Slide 16: Summary - Three Key Findings

### 1. Main Groups
Broad groups barely differ, but directionally:
- Top: more relationship + social embedding
- Trash: more sexuality mass

### 2. Subgroups
Sharper structure:
- Top: values/moral reflection, professional context
- Trash: distress intensity, coercion/threat texture

### 3. Diversity
Better books = more diverse (higher entropy/effective topics)
- Structural "variety/dispersion" hypothesis

---

## Slide 17: Implications

**Successful billionaire romance novels achieve popularity by:**

1. ✅ **Rebalancing attention** toward relationship mechanics and social embedding
2. ✅ **Incorporating value/identity reflection** moments that add depth
3. ✅ **Maintaining thematic diversity** rather than over-concentrating
4. ✅ **Moderating intensity** of distress/coercion while maintaining tension

**Not by fundamentally changing genre conventions** — the skeleton stays the same

---

## Slide 18: Lab Meeting Summary (One Slide)

**What to say out loud:**

1. **Broad groups barely differ** (expected), but directionally: Top books allocate slightly more to relationship + social embedding; Trash allocates more to sexuality mass.

2. **Subgroups show sharper structure**: Top is higher on values/moral reflection and professional context; Trash is higher on distress intensity and coercion/threat texture.

3. **Better books look more diverse** (higher entropy/effective topics), suggesting a structural "variety/dispersion" hypothesis for the large corpus.

---

## Slide 19: Supplementary Material Available

**Detailed subgroup panels by main group:**
- Conflict, Risk & Harm
- Embodied & Sensory Experience
- Emotions, Cognition & Inner Life
- Relationship Trajectory (Main Couple)
- Sexuality, Attraction & Intimacy
- Social World Outside Couple
- Spaces, Time, Activities & Objects
- Work, Wealth, Status & Institutions

**All tables and figures available in results directory**

---

## Slide 20: Questions?

**Data & Code:**
- Notebook: `notebooks/07_analysis/02_taxonomy_group_analysis/`
- Results: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/`

**Contact:** [Your contact information]

---

## Appendix Slides (Optional)

### A1: Main Group Comparison Table

See: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/tables/Table_T1_main_group_comparisons.csv`

### A2: Subgroup Comparison Table (Sorted)

See: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/tables/Table_S1_subgroup_comparisons.csv`

### A3: Topic Drivers Tables

- Beliefs, Values & Moral Reflection: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/tables/Table_D3_drivers_Beliefs_ Values _ Moral Reflection.csv`
- Negative Emotions & Distress: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/tables/Table_D4_drivers_Negative Emotions _ Distress.csv`
- Shared Workplaces: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/tables/Table_D14_drivers_Shared Workplaces _ Professional Interaction.csv`

### A4: Coverage Audit

See: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/tables_csv/coverage_audit_by_tier.csv`

### A5: Diversity Tests

See: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/tables_csv/diversity_tests.csv`

