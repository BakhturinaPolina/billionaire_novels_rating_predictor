---
name: Fix Coverage Tracking and Diversity Analysis
overview: "Implement comprehensive fixes to taxonomy group analysis notebook: add OTHER bucket creation, fix coverage metrics tracking, add paratext/noise handling, implement diversity diagnostics, fix family-wise p-value correction, and ensure proper exclusion of low-coverage/author-dominant topics in composition vs inference contexts."
todos: []
---

# Fix Coverage Tracking and Diversity Analysis in Taxonomy Group Analysis Notebook

## Overview

This plan implements 9 major fixes to make coverage a first-class variable, properly track OTHER/paratext/noise, add diversity diagnostics, fix p-value corrections, and ensure proper handling of low-coverage and author-dominant topics.

## Implementation Steps

### 1. Add OTHER Bucket Creation (New Cell After Cell 7)

**Location**: Insert new cell after Cell 7 (after rating_tier merge)

**Action**: Create a new code cell that:

- Computes missing probability mass per book (1.0 - sum of existing probs)
- Creates topic_id = -1 rows for OTHER bucket
- Appends to book_topic_probs
- Validates that sums are ~1.0 after adding OTHER
- Preserves rating_tier in OTHER rows

**Key Code**: Compute `missing_mass = (1.0 - prob_sum).clip(lower=0.0)`, create rows with `topic_id=-1`, append and validate.

### 2. Enhance Topic Taxonomy Mapping with Analysis Flags (Cell 9)

**Location**: Edit Cell 9 (build_topic_taxonomy_mapping)

**Action**: Add after `gate3_pass` computation:

- `is_other` flag (topic_id == -1)
- `is_paratext` flag (detect via label/scene_summary/keywords patterns: "acknowledg", "author's note", etc.)
- `is_low_coverage` flag (prevalence < 0.10)
- `is_author_dominant_high` and `is_author_dominant_med` flags
- Ensure OTHER row exists in mapping_export (add if missing with topic_id=-1, label="OTHER (Missing/Noise Mass)")

**Constants to Add**: `PREV_MIN = 0.10`, `AUTHOR_DOM_MIN = 0.50`

### 3. Replace compute_book_group_shares Function (Cell 11)

**Location**: Replace entire Cell 11 function

**Action**: Replace `compute_book_group_shares` with new version that:

- Adds `exclude_paratext` parameter (default True)
- Changes `apply_gate3` to `apply_prevalence_gate` (default False for composition)
- Computes `other_share` from actual OTHER rows (topic_id == -1)
- Computes `paratext_share` explicitly
- Computes `noise_share` explicitly
- Computes `modeled_mass` as sum of thematic topics after filters
- Merges all coverage metrics (other_share, paratext_share, noise_share, modeled_mass) into result
- Returns DataFrame with coverage columns attached to every row

**Key Change**: Always exclude OTHER from thematic group sums, but measure it separately. Don't apply prevalence gate for composition (keep long-tail topics).

### 4. Update Main Group Share Computation (Cell 13)

**Location**: Edit Cell 13

**Action**:

- Change function call to use `apply_prevalence_gate=False` instead of `apply_gate3=True`
- Add `exclude_paratext=True` parameter
- Add sanity checks after computation:
- Verify `abs_share` sums equal `modeled_mass` per book
- Verify `modeled_mass + other_share + paratext_share + noise_share` ≈ 1.0 per book
- Print validation results

### 5. Add Coverage Audit Report (New Cell After Cell 13)

**Location**: Insert new cell after Cell 13

**Action**: Create cell that:

- Groups book_main_group_shares by rating_tier
- Computes mean/median of other_share, paratext_share, noise_share, modeled_mass
- Displays summary table
- Saves to `coverage_audit_by_tier.csv`

### 6. Add Diversity Diagnostics Section (New Cell After Cell 13)

**Location**: Insert new cell after coverage audit (or after Cell 13)

**Action**: Create comprehensive diversity analysis cell that:

- Filters to thematic topics only (exclude OTHER/noise/paratext, keep long-tail)
- Renormalizes probabilities per book
- Computes per-book metrics:
- Shannon entropy: `H = -sum(p * log(p))`
- Effective topics: `exp(H)`
- HHI (Herfindahl): `sum(p^2)`
- Richness: count of topics with prob > 0.001
- Merges with coverage metrics (other_share, etc.) and rating_tier
- Runs Kruskal-Wallis tests for diversity metrics across tiers (with Holm correction)
- Computes Spearman correlations: OTHER vs effective_topics within each tier
- Saves `diversity_metrics_by_book.csv` and `diversity_tests.csv`
- Displays tier summaries and correlation results

**Interpretation**: Tests whether higher OTHER in Top books correlates with higher thematic diversity (entropy/richness).

### 7. Fix Subgroup Share Computation (Cell 21)

**Location**: Edit Cell 21

**Action**:

- Replace filter block to explicitly exclude:
- `topic_id != -1` (OTHER)
- `author_dominance_flag != 'high'`
- `~is_noise` (label_is_noise and taxonomy_is_noise)
- `~is_paratext` (if column exists)
- Remove `gate3_pass` filter (don't exclude low-coverage for composition)
- After computing `book_subgroup_shares`, merge coverage columns from `book_main_group_shares`:
- Merge `['book_id', 'modeled_mass', 'other_share', 'paratext_share', 'noise_share']`
- Fix subgroup counting loop (ensure correct column names)

### 8. Fix Family-Wise P-Value Correction (Cell 23)

**Location**: Edit Cell 23 (subgroup statistical comparisons)

**Action**: Replace global Holm correction block with family-wise version:

- Group `subgroup_results` by `taxonomy_main_group`
- Apply Holm correction within each main group family
- Create `kruskal_p_adj_family` column
- Also apply family-wise correction to pairwise p-values:
- `top_vs_trash_p_adj_family`
- `top_vs_middle_p_adj_family`  
- `middle_vs_trash_p_adj_family`
- Keep existing save code (will now include family-adjusted columns)

**Key Change**: Instead of correcting across all 26 subgroups globally, correct within each main group's subgroup family (e.g., Sexuality has 3 subgroups, Work has 5).

### 9. Update Topic Driver Analysis (Cell 27)

**Location**: Edit Cell 27

**Action**:

- Ensure topic driver filtering uses strict criteria:
- `prevalence >= 0.10` (low-coverage exclusion for inference)
- `author_dominance_flag != 'high'` (author-dominant exclusion)
- This is separate from composition analysis (which keeps long-tail)

### 10. Add Sensitivity Analysis Summary (Optional New Cell)

**Location**: Insert new cell after main group comparisons (Cell 17 or 19)

**Action**: Create cell that:

- Recomputes main group shares with different filter combinations:
- Main: exclude author-dominant (high only)
- Sensitivity A: include everything except OTHER/noise/paratext
- Sensitivity B: exclude high + medium author dominance
- Compares key effects across sensitivity runs
- Saves `sensitivity_summary.csv`

### 11. Update Export Code to Include Coverage Columns

**Location**: Multiple cells (13, 21, 17, 23)

**Action**: Ensure all saved CSVs include coverage columns:

- `book_main_group_shares.csv`: already includes other_share, modeled_mass (verify paratext_share, noise_share added)
- `book_subgroup_shares.csv`: add coverage columns via merge
- `main_group_comparisons.csv`: add tier-level coverage summaries
- `subgroup_comparisons.csv`: verify family-adjusted p-values are saved

## Files to Modify

1. `notebooks/07_analysis/taxonomy_group_analysis/taxonomy_group_analysis.ipynb`

- Cell 7: Add new cell after (OTHER creation)
- Cell 9: Enhance mapping with flags
- Cell 11: Replace function entirely
- Cell 13: Update function call and add sanity checks
- New cell: Coverage audit report
- New cell: Diversity diagnostics
- Cell 21: Fix filters and add coverage merge
- Cell 23: Fix p-value correction
- Cell 27: Ensure strict filtering for drivers

## Expected Outcomes

1. `other_share` will have real values (not all zeros) in saved CSVs
2. `modeled_mass + other_share + paratext_share + noise_share` ≈ 1.0 per book
3. Subgroup p-values will be interpretable (family-adjusted, not all 1.0)
4. Diversity metrics will test "OTHER = thematic diversity" hypothesis
5. Coverage metrics tracked explicitly throughout pipeline
6. Paratext and noise excluded from thematic analysis but measured
7. Low-coverage topics kept in composition but excluded from inference

## Validation Checks

After implementation, verify:

- OTHER rows exist in book_topic_probs (topic_id == -1)
- other_share > 0 for most books in saved CSVs
- Coverage columns sum to ~1.0 per book
- Family-adjusted p-values are more interpretable than global correction
- Diversity metrics show tier differences if they exist