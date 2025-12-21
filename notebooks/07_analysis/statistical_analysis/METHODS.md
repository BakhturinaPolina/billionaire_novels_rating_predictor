# Methods

## Data and Units of Analysis

The analysis operates at the **book level**, using outputs from a topic-model–based annotation pipeline applied to romance novels. For each book, sentences were assigned to theory-driven taxonomy categories derived from prior qualitative and domain-specific frameworks. Category assignments were aggregated to produce **book-level category proportions**, defined as the proportion of sentences in a book assigned to a given taxonomy category.

The primary analytical dataset consists of book–category pairs, where each row represents the proportion of a specific taxonomy category within a given book. Outcome labels (e.g., rating class) are attached at the book level and inherited by all corresponding category rows.

A separate topic lookup table maps low-level model topics to higher-order constructs, including:

* taxonomy main categories and groups,
* Radway narrative phases,
* and noise indicators.

This lookup table is used exclusively for **interpretation and grouping**, not for feature construction or statistical aggregation.

---

## Analytical Rationale

Rather than constructing composite indices that aggregate multiple thematic dimensions into single summary measures, the analysis adopts a **direct, category-level statistical approach**. This design choice was motivated by three considerations:

1. **Interpretability**: Individual taxonomy categories retain clear semantic meaning, whereas composite indices require additional assumptions about weighting and aggregation.

2. **Methodological transparency**: Avoiding index construction eliminates researcher-imposed structures not directly supported by the model outputs.

3. **Theoretical alignment**: The taxonomy and Radway frameworks are preserved explicitly rather than being subsumed into synthetic metrics.

All analyses therefore rely on **existing category proportions and observed labels**, without generating new composite variables.

---

## Preprocessing and Inclusion Criteria

Only categories with valid, non-missing proportion values were included. Topics or categories flagged as noise in upstream processing were excluded from analysis. Category proportions were treated as bounded, continuous variables in the interval ([0,1]).

No normalization or rescaling beyond the original proportion calculation was performed.

---

## Category-Level Comparisons by Outcome Label

To assess whether specific thematic dimensions differ systematically across outcome labels (e.g., higher- vs. lower-rated books), category proportions were compared between groups for each taxonomy main category.

Because category proportions are non-normally distributed and often zero-inflated, **non-parametric statistical tests** were employed:

* **Mann–Whitney U tests** were used for pairwise comparisons between outcome classes.

* **Cliff's delta** was calculated as a measure of effect size, capturing the direction and magnitude of distributional differences.

* **False discovery rate (FDR) correction** using the Benjamini–Hochberg procedure was applied across all category-level tests to control for multiple comparisons.

This approach identifies individual categories whose prevalence differs meaningfully across outcome labels, without aggregating them into composite indices.

---

## Group-Level Analysis Using Taxonomy Main Groups

To examine higher-level thematic patterns, category proportions were aggregated at the level of **taxonomy main groups** (e.g., *Sexuality, Attraction & Intimacy*; *Relationship Trajectory*). For each book, group-level proportions were computed as the sum of all category proportions belonging to that group.

These group-level proportions were then compared across outcome labels using the same non-parametric framework described above. This analysis provides a theory-aligned perspective on how broad thematic domains relate to outcomes, while maintaining a transparent and minimal aggregation rule.

---

## Correlation Structure Among Categories

To explore the empirical structure of thematic co-occurrence, **Spearman rank correlations** were computed between category proportions across books. Correlation analyses were conducted both globally and within selected taxonomy groups.

This analysis serves two purposes:

1. To identify clusters of categories that tend to co-occur within books.

2. To assess whether theoretically grouped categories also exhibit empirical cohesion.

Correlation analysis was descriptive in nature and not used to construct latent variables or indices.

---

## Mapping Results to Radway Narrative Phases

For interpretive synthesis, statistically salient taxonomy categories were mapped to **Radway narrative phases** using the topic lookup table. Each taxonomy category was associated with the Radway phase most frequently assigned to its underlying topics.

This mapping enables interpretation of quantitative findings in terms of narrative structure (e.g., *Initial Conflict*, *Turning Point*, *Commitment & Restoration*), allowing assessment of which narrative phases are overrepresented among categories associated with different outcome labels.

Radway phases were used **only as interpretive descriptors**; no Radway-based indices or numerical scores were constructed.

---

## Summary of Analytical Approach

In summary, the methods prioritize:

* direct use of model-derived category proportions,
* non-parametric, distribution-sensitive statistical testing,
* explicit theoretical grounding via taxonomy groups and narrative phases,
* and avoidance of composite index construction.

This approach emphasizes transparency, interpretability, and alignment between quantitative analysis and qualitative theory.

