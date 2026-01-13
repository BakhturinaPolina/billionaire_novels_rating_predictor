# Hypothesis Testing Results

This folder contains findings from the hypothesis testing analysis (`04_hypothesis_testing_inference_only_v3_2.ipynb`).

## Main Document

**[HYPOTHESIS_TESTING_RESULTS.md](./HYPOTHESIS_TESTING_RESULTS.md)** — Comprehensive summary of all findings, including:
- Tier structure analysis (two-channel differences)
- Mass appeal predictors
- Perceived quality predictors
- Weighted/Bayesian validation methods
- Cross-validation results
- Arc analysis connections

## Folder Structure

The subfolders are organized by analysis type for future reference:

- `mass_appeal/` — Results related to popularity/popularity predictors (log_n_ratings)
- `perceived_quality/` — Results related to rating quality predictors (avg_rating/rating_mean)
- `arc_analysis/` — Results related to narrative arc/pacing analysis
- `validation_methods/` — Results from validation techniques (weighted, Bayesian, CV)

## Source Data

All raw CSV outputs are saved to:
```
results/measurement_v5/bundle/inference_outputs/
```

Key files:
- `tier_summary_goodreads_channels.csv`
- `goodreads_index_correlations.csv`
- `ridge_joint_coeffs_log_rating_count.csv`
- `ridge_joint_coeffs_rating_mean.csv`
- `partial_corr_rating_mean.csv`
- `partial_corr_log_rating_count.csv`
- `goodreads_weighted_quality_table.csv`
- `cv_repeats_summary.csv`

## Key Findings

1. **Tiers differ in both channels**: Quality (avg_rating) AND visibility (n_ratings)
2. **Themes predict popularity better than star ratings** (at N=92)
3. **Mass appeal signature**: Luxury + alpha + repair + safety + kin
4. **Quality signature**: Care + safety (positive); anger/anxiety + explicit erotics (negative)
5. **Pacing matters**: Higher-rated books have lower baseline negativity but stronger late crisis escalation

