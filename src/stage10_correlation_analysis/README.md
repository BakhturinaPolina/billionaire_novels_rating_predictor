# Stage 07: Statistical Analysis

## Overview

Stage 07 performs comprehensive statistical analysis combining topic probabilities with Goodreads metadata, including hypothesis testing, index computation, and visualization.

## Status

✅ **Fully Implemented**

## Functionality

### Popularity Stratification

Creates three-tier classification:
- **Top**: popularity_index > 66.67th percentile
- **Medium**: 33.33rd < popularity_index ≤ 66.67th percentile
- **Trash**: popularity_index ≤ 33.33rd percentile

**Popularity Index Formula:**
```
popularity_index = mean(z_RatingsCount, z_Score, 
                        z_Popularity_ReadingNow, z_Popularity_Wishlisted)
```

### Topic Probability Normalization

Normalizes topic probabilities per book (sum to 1.0):
```python
row_sum = topic_probs.sum(axis=1)
topic_probs_normalized = topic_probs.div(row_sum.replace(0, np.nan), axis=0)
```

### Index Computation

Computes all derived indices (see [docs/INDICES.md](../../docs/INDICES.md)):
- Love-over-Sex Index
- HEA Index
- Explicitness Ratio
- Luxury Saturation
- And 6 more indices

### Statistical Tests

- **Group Comparisons**: ANOVA/Kruskal-Wallis across Top/Medium/Trash
- **Regression Models**: Logistic (Top vs Trash), OLS (avg_rating)
- **Interaction Effects**: Luxury × (Commitment+Tenderness), etc.
- **Time-Course Analysis**: Repeated-measures ANOVA for H6

### FDR Correction

Benjamini-Hochberg FDR correction for multiple comparisons.

## Key Files

- **`main.py`**: CLI entry point
- **`scoring_and_strata.py`**: Popularity index and stratification
- **`bh_fdr.py`**: Benjamini-Hochberg FDR correction
- **`03_interaction_plots_PD.py`**: Interaction plot generation
- **`04_star_charts_and_pdf.py`**: Star chart visualizations

## Usage

```bash
python -m src.stage07_analysis.main --config configs/scoring.yaml
```

## Inputs

- **Topics CSV**: `results/topics/by_book.csv`
- **Goodreads CSV**: `data/processed/goodreads.csv`
- **Composites Parquet** (from Stage 06): `results/topics/ap_composites.parquet`

## Outputs

- **Prepared Books**: `data/processed/prepared_books.parquet`
  - Contains all metadata, normalized topic probabilities, popularity index, Group

- **Statistical Results**: `results/analysis/`
  - Test results, model coefficients, effect sizes

- **Visualizations**: `results/figures/`
  - Heatmaps, group comparisons, UMAP plots, time-course plots

## Research Hypotheses Tested

- **H1**: Love-over-Sex higher in Top vs Trash
- **H2**: Explicitness Ratio higher in Trash
- **H3**: Luxury × (Commitment+Tenderness) interaction
- **H4**: Protective–Jealousy Delta higher in Top
- **H5**: Dark-vs-Tender lower in Top
- **H6**: Time-course trends (begin→end)

See [SCIENTIFIC_README.md](../../SCIENTIFIC_README.md) for detailed hypotheses.

## Data Contracts

See [docs/DATA_CONTRACTS.md](../../docs/DATA_CONTRACTS.md) for detailed specifications.

## Statistical Analysis Plan

See [SCIENTIFIC_README.md](../../SCIENTIFIC_README.md) for complete statistical analysis plan.

## Dependencies

- `pandas`, `numpy` for data manipulation
- `scipy`, `statsmodels` for statistical tests
- `matplotlib`, `seaborn` for visualization
- `sklearn` for regression models

