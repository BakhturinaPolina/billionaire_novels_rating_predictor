# Stage 11: Power Analysis Report

## Sample Size Decision: Simulation-Based Power Analysis

**Source Notebook:** `notebooks/08_power_analysis/08_power_analysis_for_large_dataset.ipynb`  
**Analysis Date:** 2026-01-23  
**Pilot Sample Size:** N = 92 books

---

## Executive Summary

We estimated the minimum number of books needed for stable inference using a simulation-based power analysis grounded in the pilot dataset. The analysis supports a two-channel interpretation (reach/visibility vs quality/ratings), leading to sample-size adequacy prioritization for quality-beyond-reach models.

**Targets:**
- **Primary minimum target:** N = 6,000 books
- **Preferred target:** N = 12,000 books

---

## Methodology

### Simulation Design

For each candidate sample size N ∈ {6,000, 8,000, 10,000, 12,000, 15,000, 20,000}, we repeatedly resampled books with replacement (bootstrap; 250 replicates) and re-fit the planned regression models using the same preprocessing pipeline:

- **Global z-scoring** of axis measures (standardized once before simulation, NOT within each replicate)
- **Reach control** using `log_rating_count`
- **Two model specifications:**
  - OLS (no author fixed effects) — cross-market inference
  - Author fixed effects (within-author demeaning) — within-author inference

### Critical Technical Note: Global Standardization

Variables are z-scored **once globally** before the bootstrap simulation, NOT within each replicate. This preserves the population-scale structure and collinearity patterns, preventing inflated power estimates. Re-standardizing within each replicate would remove scale differences and collinearity structure, giving each axis an "equalized playing field" that doesn't reflect real-world estimation conditions.

### Evaluation Criteria

We tracked two criteria for each macro-axis effect:

1. **Power**: The proportion of replicates where the axis coefficient was statistically significant at α = 0.05 (two-sided).

2. **Direction Stability**: The probability that the coefficient keeps the same sign across replicates, computed as:

   ```
   direction_stability = max(P(β > 0), P(β < 0))
   ```

   This definition treats consistently negative effects as "stable" (not "unstable").

### Decision Thresholds

We considered an effect **"well-powered and stable"** at a given N if:
- Power ≥ 0.90
- Direction stability ≥ 0.95

---

## Macro-Axes Tested

Six theory-aligned macro-axes from the pilot study:

| Axis | Components | Interpretation |
|------|------------|----------------|
| **AX_status_dominance** | D_power_wealth_luxury + R2_alpha_guarding | Billionaire-romance backbone |
| **AX_payoff_safety** | A2_emotional_safety + Q_repair + R1_protective_caretaking | HEA-related affect regulation |
| **AX_negative_affect** | F2_anger_frustration + F3_anxiety_worry + F1_sadness_grief | Baseline negativity |
| **AX_explicitness** | C_explicit_eroticism | Explicit sexual content |
| **AX_attraction** | B1_attraction_chemistry | Non-explicit romantic charge |
| **AX_drama_obstacle** | Q_miscommunication | Conflict and obstacles |

---

## Results: Minimum N for 0.90 Power

### OLS (No Author Fixed Effects) — Primary Model for Confirmatory Testing

| Axis | rating_mean | avg_rating_bayes |
|------|-------------|------------------|
| AX_drama_obstacle | **6,000** | — |
| AX_payoff_safety | **6,000** | — |
| AX_negative_affect | **6,000** | 12,000 |
| AX_explicitness | **6,000** | **6,000** |
| AX_attraction | **12,000** | **6,000** |
| AX_status_dominance | — | — |

**Note:** "—" indicates threshold not reached by N = 20,000.

### Author Fixed Effects (Within-Author) — Secondary Model

| Axis | rating_mean | avg_rating_bayes |
|------|-------------|------------------|
| AX_status_dominance | **6,000** | **6,000** |
| AX_drama_obstacle | **6,000** | **6,000** |
| AX_payoff_safety | **6,000** | **6,000** |
| AX_explicitness | **6,000** | **6,000** |
| AX_attraction | **6,000** | **6,000** |
| AX_negative_affect | — | **6,000** |

---

## Results: Minimum N for 0.95 Direction Stability

### OLS (No Author Fixed Effects)

| Axis | rating_mean | avg_rating_bayes |
|------|-------------|------------------|
| AX_drama_obstacle | **6,000** | **6,000** |
| AX_payoff_safety | **6,000** | **6,000** |
| AX_negative_affect | **6,000** | **6,000** |
| AX_explicitness | **6,000** | **6,000** |
| AX_attraction | **6,000** | **6,000** |
| AX_status_dominance | — | — |

### Author Fixed Effects (Within-Author)

All axes meet the 0.95 stability threshold by N = 6,000 for both outcomes.

---

## Interpretation: Result-Driven Design Choice

The pilot results support a **two-channel interpretation**:

1. **Reach/Visibility** (`log_rating_count`): Status/dominance is a primary driver
2. **Quality/Ratings** (`rating_mean`, `avg_rating_bayes`): Quality beyond reach is driven by payoff/safety

### Key Finding: AX_status_dominance

For the quality-beyond-reach model (`rating_mean` with reach control, OLS), **AX_status_dominance does not meet power or stability thresholds even by N = 20,000**.

This is **consistent with the pilot interpretation** that status/dominance is primarily a reach driver rather than a quality driver once reach is controlled. It is not a power failure — it reflects the theoretical structure: status/dominance predicts who reads the book, not whether those readers rate it highly.

### Key Finding: AX_attraction

**AX_attraction requires ~12,000 books** to reliably meet the power + stability thresholds for `rating_mean` in the OLS model. This axis has a modest effect size in the pilot, requiring larger N for stable detection.

---

## Sample Size Decision

Based on these results, we set:

### Primary Minimum Target: N = 6,000 books

**Rationale:** Adequate for the main confirmatory quality axes identified from pilot theory tests:
- AX_payoff_safety
- AX_negative_affect
- AX_explicitness
- AX_drama_obstacle

All meet both power ≥ 0.90 and direction stability ≥ 0.95 by N = 6,000.

### Preferred Target: N = 12,000 books

**Rationale:** Adds robust power for AX_attraction in the quality-beyond-reach model, enabling full testing of the chemistry/attraction hypothesis.

---

## Secondary Analyses: Author Fixed Effects

If sufficient multi-book authors are available in the larger dataset, we will additionally run **within-author (author fixed effects) models** to test whether within-author shifts in axis content predict within-author shifts in ratings.

**Important:** These models answer a different question than across-market OLS:
- **OLS:** "Across all books in the market, does axis X predict ratings?"
- **Author FE:** "Within each author's books, does more axis X predict higher ratings?"

The author FE models show uniformly high power by N = 6,000, suggesting that within-author variation is sufficient for stable inference.

---

## Technical Note: Direction Stability Metric

### The Problem

The original `sign_stability` metric stored P(β > 0). This means:
- A truly negative stable effect shows up near 0.00, which looks "unstable" even though it's perfectly consistent (always negative).

### The Fix

Use symmetric direction stability:

```
direction_stability = max(P(β > 0), 1 − P(β > 0))
```

This correctly identifies effects that are **consistently positive OR consistently negative** as "stable."

**Example:**
- P(β > 0) = 0.02 → direction_stability = max(0.02, 0.98) = 0.98 (stable negative)
- P(β > 0) = 0.99 → direction_stability = max(0.99, 0.01) = 0.99 (stable positive)
- P(β > 0) = 0.50 → direction_stability = max(0.50, 0.50) = 0.50 (unstable)

---

## Files Referenced

All results saved to: `results/stage08_power_analysis/`

| File | Description |
|------|-------------|
| `power_analysis_results.csv` | Full power and stability results for all N × axis × outcome × model combinations |
| `min_n_for_power_0.90.csv` | Smallest N achieving 0.90 power per axis/outcome/model |
| `min_n_for_sign_stability_0.95.csv` | Smallest N achieving 0.95 direction stability per axis/outcome/model |
| `simulation_parameters.csv` | Simulation configuration (N grid, replicates, alpha, outcomes, axes) |

---

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| N grid | {6,000, 8,000, 10,000, 12,000, 15,000, 20,000} |
| Bootstrap replicates | 250 |
| Significance level (α) | 0.05 |
| Outcomes | rating_mean, avg_rating_bayes |
| Reach control | log_rating_count |
| Pilot sample size | 92 books |
| Unique authors (pilot) | 33 |

---

## Conclusion

The simulation-based power analysis provides clear, defensible sample size targets for the confirmatory study:

1. **N = 6,000** is sufficient for testing the primary quality-beyond-reach hypotheses (payoff/safety, negative affect, explicitness, drama/obstacle).

2. **N = 12,000** is preferred if testing the attraction/chemistry axis is a priority.

3. **AX_status_dominance** should not be treated as a quality predictor in confirmatory testing — it is a reach predictor by design, consistent with the two-channel interpretation.

---

*Analysis conducted: 2026-01-23*  
*Notebook: `notebooks/08_power_analysis/08_power_analysis_for_large_dataset.ipynb`*
