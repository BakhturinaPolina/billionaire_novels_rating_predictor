# Stage 10: Correlation Analysis

## Overview

Stage 10 performs comprehensive statistical analysis and exploratory data analysis (EDA) combining topic probabilities with Goodreads metadata. This includes:

- **Statistical analysis** of taxonomy category differences across rating classes
- **Exploratory data analysis** of taxonomy and Radway narrative function mappings
- **Hypothesis testing** with effect size calculations
- **Visualization** of results

## Structure

```
stage10_correlation_analysis/
├── analysis/              # Main analysis scripts (entry points)
│   ├── category_statistics.py      # Statistical analysis of categories
│   └── taxonomy_radway_eda.py      # EDA for taxonomy & Radway mappings
├── utils/                # Reusable helper modules
│   ├── statistics.py              # Statistical test functions
│   └── visualization.py            # Plotting and visualization functions
└── docs/                 # Documentation and reports
    ├── STATISTICAL_ANALYSIS_REPORT.md
    ├── VISUALIZATION_EXAMPLES.md
    └── VISUALIZATION_IMPROVEMENTS.md
```

## Analysis Scripts

### 1. Category Statistics (`analysis/category_statistics.py`)

Runs statistical analysis to identify taxonomy categories that differ significantly across book rating classes.

**Features:**
- Kruskal-Wallis tests for each category
- Effect size calculations (eta-squared)
- Post-hoc pairwise comparisons
- Comprehensive visualizations (volcano plots, effect size bars, prevalence plots)

**Usage:**
```bash
python -m src.stage10_correlation_analysis.analysis.category_statistics \
    --book-cat results/stage09_category_mapping/stage2_theory_driven_categories/book_category_proportions.parquet \
    --output-dir results/stage10_correlation_analysis/category_statistical_analysis \
    --top-n 10 \
    --alpha 0.05
```

**Outputs:**
- `kruskal_wallis_results.csv`: Statistical test results
- `figures/volcano_plot.png`: P-value vs effect size visualization
- `figures/effect_size_bars.png`: Effect sizes for top categories
- `figures/category_*_prevalence.png`: Individual category plots
- `figures/category_*_pairwise.png`: Pairwise comparison plots

### 2. Taxonomy & Radway EDA (`analysis/taxonomy_radway_eda.py`)

Exploratory data analysis of the BERTopic model with taxonomy and Radway narrative function mappings.

**Features:**
- Distribution analysis of taxonomy categories
- Distribution analysis of Radway narrative functions
- Cross-tabulations between taxonomy and Radway
- Summary statistics and data exports

**Usage:**
```bash
python -m src.stage10_correlation_analysis.analysis.taxonomy_radway_eda \
    --output-dir results/stage10_correlation_analysis/taxonomy_radway_eda
```

**Outputs:**
- `taxonomy_distribution.png`: Taxonomy category distributions
- `radway_distribution.png`: Radway function distributions
- `cross_tabulations.png`: Taxonomy vs Radway cross-tabulations
- `summary_statistics.json`: Summary statistics
- `full_model_data.csv` / `.parquet`: Complete extracted data

## Helper Modules

### `utils/statistics.py`
- Kruskal-Wallis test functions
- Effect size calculations
- Post-hoc pairwise comparisons

### `utils/visualization.py`
- Volcano plots
- Effect size bar charts
- Category prevalence plots (box, violin, strip plots)
- Pairwise comparison visualizations
- P-value heatmaps

## Inputs

- **Book Category Proportions**: `results/stage09_category_mapping/stage2_theory_driven_categories/book_category_proportions.parquet`
- **Taxonomy Mappings**: `results/stage09_category_mapping/stage2_theory_driven_categories/taxonomy_mappings_*.json`
- **BERTopic Model**: Model with taxonomy and Radway mappings (from Stage 9)

## Outputs

- **Statistical Results**: `results/stage10_correlation_analysis/category_statistical_analysis/`
  - Test results, effect sizes, visualizations

- **EDA Results**: `results/stage10_correlation_analysis/taxonomy_radway_eda/`
  - Distribution plots, cross-tabulations, summary statistics

## Dependencies

- `pandas`, `numpy` for data manipulation
- `scipy` for statistical tests (Kruskal-Wallis, Mann-Whitney U)
- `matplotlib`, `seaborn` for visualization
- `bertopic` for model loading

