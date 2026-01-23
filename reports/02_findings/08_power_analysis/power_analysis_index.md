# Power Analysis Results

This folder contains findings from the simulation-based power analysis for sample size planning (`notebooks/08_power_analysis/08_power_analysis_for_large_dataset.ipynb`).

## Main Document

**[sample_size_decision.md](./sample_size_decision.md)** — Comprehensive summary of sample size planning, including:
- Simulation methodology (bootstrap with global z-scoring)
- Power and direction stability criteria
- Minimum N thresholds for each macro-axis
- Sample size decision rationale
- Technical notes on stability metric correction

## Source Data

All raw CSV outputs are saved to:
```
results/stage08_power_analysis/
```

Key files:
- `power_analysis_results.csv` — Full power and stability results
- `min_n_for_power_0.90.csv` — Minimum N for 0.90 power threshold
- `min_n_for_sign_stability_0.95.csv` — Minimum N for 0.95 stability threshold
- `simulation_parameters.csv` — Simulation configuration

## Key Findings

1. **Primary target: N = 6,000 books** — Adequate for main confirmatory quality axes
2. **Preferred target: N = 12,000 books** — Adds robust power for AX_attraction
3. **AX_status_dominance** is a reach predictor, not a quality predictor (consistent with two-channel interpretation)
4. **Author FE models** show uniformly high power by N = 6,000
5. **Direction stability** uses symmetric metric: max(P(β>0), 1−P(β>0))

## Macro-Axes Tested

| Axis | Quality N (OLS) | Interpretation |
|------|-----------------|----------------|
| AX_payoff_safety | 6,000 | HEA-related affect regulation |
| AX_negative_affect | 6,000 | Baseline negativity |
| AX_explicitness | 6,000 | Explicit sexual content |
| AX_drama_obstacle | 6,000 | Conflict and obstacles |
| AX_attraction | 12,000 | Non-explicit romantic charge |
| AX_status_dominance | — | Reach driver (not quality) |
