# Stage 04: Pareto-Efficient Model Selection

## Overview

Stage 04 performs Pareto efficiency analysis to identify the best-performing models that balance coherence and topic diversity.

## Status

✅ **Fully Implemented**

## Functionality

### Pareto Efficiency Analysis

Identifies models that are not dominated by any other model in both coherence and topic diversity metrics.

### Weighting Strategies

1. **Equal Weights** (50/50)
   - Balances coherence and diversity equally
   - Output: `top_10_equal_weights.csv`

2. **Coherence Priority** (70/30)
   - Prioritizes coherence over diversity
   - Output: `top_10_coherence_priority.csv`

### Data Cleaning

- Removes failed runs (Coherence = 1.0 or Topic_Diversity = 1.0)
- Removes outliers using z-score method (default: 2 std dev)
- Normalizes metrics before combination

### Hyperparameter Analysis

- Correlation analysis between hyperparameters and performance
- Boxplots of hyperparameter distributions
- Identifies optimal hyperparameter ranges

## Key Files

- **`main.py`**: CLI entry point with `analyze` command
- **`pareto_analysis.py`**: Core analysis functions
  - `identify_pareto()`: Pareto efficiency identification
  - `clean_data()`: Data cleaning and outlier removal
  - `normalize_metrics()`: Metric normalization
  - `analyze_hyperparameters()`: Hyperparameter correlation analysis

## Usage

```bash
python -m src.stage04_selection.main analyze \
  --config configs/selection.yaml \
  --paths-config configs/paths.yaml
```

## Configuration

See `configs/selection.yaml` for:
- Input/output paths
- Cleaning thresholds
- Weighting strategy parameters
- Hyperparameter analysis settings

## Outputs

### CSV Files
- `results/pareto/top_10_equal_weights.csv`
- `results/pareto/top_10_coherence_priority.csv`
- `results/pareto/tables/correlation_analysis_*.csv`

### Visualizations
- `results/pareto/figures/pareto_front_equal_weights.png`
- `results/pareto/figures/pareto_front_coherence_priority.png`
- `results/pareto/figures/pareto_fronts_per_model.png`
- `results/pareto/figures/distribution_with_cutoffs.png`
- `results/pareto/figures/hyperparameter_boxplots.png`

## Data Flow

```
Stage 03 Output (model_evaluation_results.csv)
    ↓
Data Cleaning (remove failed runs, outliers)
    ↓
Metric Normalization
    ↓
Pareto Efficiency Analysis
    ↓
Top Model Selection
    ↓
Hyperparameter Analysis
    ↓
Outputs (CSV + Visualizations)
```

## Algorithm Details

See [docs/METHODOLOGY.md](../../docs/METHODOLOGY.md) for detailed algorithm descriptions.

## Dependencies

- `pandas`, `numpy` for data manipulation
- `matplotlib`, `seaborn` for visualization
- `scipy` for statistical tests
- `sklearn` for normalization

