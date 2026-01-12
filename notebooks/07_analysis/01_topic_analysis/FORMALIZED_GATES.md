# Formalized Three-Gate Rule for Topic Filtering

## Rationale

With only 30 books per tier, FDR-corrected p-values are underpowered. Even large effect sizes (|δ| > 0.35) cannot survive FDR correction. The smallest adjusted p-value is ~0.20.

**Solution:** A formalized three-gate rule for meaningful interpretation that makes the bottom layer reproducible and defensible.

## The Three Gates

### Gate 1: Effect Gate
- **Criterion:** |Cliff's δ| ≥ 0.20
- **Rationale:** Meaningful effect size threshold
- **Purpose:** Filters out topics with trivial differences between tiers

### Gate 2: Impact Gate
- **Criterion:** mass ≥ 0.002 OR |top_mean − trash_mean| ≥ 0.001
- **Rationale:** Ensures the topic has meaningful presence or meaningful difference
- **Purpose:** Filters out topics that are either too rare or have negligible differences

### Gate 3: Stability Gate
- **Criterion:** prevalence ≥ 0.10 AND NOT author-dominant
- **Rationale:** Ensures topic is stable across books and not driven by single author
- **Purpose:** Filters out author-specific style topics and ensures generalizability

## Two-Tier Structure

### Tier 1: High Confidence Topics
- **Criteria:**
  - |Cliff's δ| ≥ 0.35 (large effect)
  - Raw p < 0.05 (statistical significance)
  - All three gates passed
- **Use:** Primary findings, most defensible claims

### Tier 2: Exploratory Topics
- **Criteria:**
  - |Cliff's δ| ≥ 0.20 (moderate effect)
  - All three gates passed
  - No p-value filter (acknowledges power limitations)
- **Use:** Hypothesis-generating, exploratory analysis

## Implementation Notes

1. **Author Dominance:** Gate 3 requires merging author dominance analysis results. If `is_author_driven` column is available, it's included in Gate 3. Otherwise, Gate 3 uses prevalence only and author-dominance is flagged separately.

2. **Reproducibility:** All thresholds are explicit and fixed:
   - EFFECT_THRESHOLD = 0.20
   - MASS_THRESHOLD = 0.002
   - MEAN_DIFF_THRESHOLD = 0.001
   - PREVALENCE_THRESHOLD = 0.10

3. **Appropriateness:** This approach is appropriate for hypothesis-generating exploratory research with limited sample size.

## Outputs

- **Tier 1 topics:** High-confidence findings suitable for primary claims
- **Tier 2 topics:** Exploratory findings suitable for hypothesis generation
- **Filtered leaderboard:** Combined Tier 2 set (broader exploratory set)

