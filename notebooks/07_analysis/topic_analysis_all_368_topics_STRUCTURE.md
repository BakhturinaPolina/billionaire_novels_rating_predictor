# Notebook Structure: Analysis of All 368 Topics Across Top/Medium/Trash Tiers

**Purpose:** Comprehensive bottom-to-top analysis of all 368 BERTopic topics across popularity tiers (Top/Medium/Trash), following a systematic screening and interpretation workflow.

**Reference Documentation:**
- BERTopic API: https://maartengr.github.io/BERTopic/index.html#citation
- BERTopic approximate_distribution: https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.approximate_distribution

---

## 0. Analysis-Ready Structure & Data Integrity

### 0.1 Setup & Imports
- Project root resolution
- Import libraries: pandas, numpy, matplotlib, seaborn, plotly, scipy.stats, statsmodels
- BERTopic import (for reference/documentation)
- Set plotting styles and output directories

### 0.2 Define Data Paths
```python
PROJECT_ROOT = Path("/home/polina/Documents/goodreads_romance_research_cursor/billionaire_novels_rating_predictor")

# Data paths (from data preparation stage)
BOOK_WIDE_PATH = PROJECT_ROOT / "results" / "stage10_correlation_analysis" / "data_preparation" / "book_features" / "book_taxonomy_main_props_wide.parquet"
BOOK_LONG_PATH = PROJECT_ROOT / "results" / "stage10_correlation_analysis" / "data_preparation" / "book_features" / "book_taxonomy_main_props_long.parquet"
BOOK_TOPIC_PROBS_PATH = PROJECT_ROOT / "results" / "stage10_correlation_analysis" / "data_preparation" / "topic_probabilities" / "book_topic_probs.parquet"
CHAPTER_TOPIC_PROBS_PATH = PROJECT_ROOT / "results" / "stage10_correlation_analysis" / "data_preparation" / "topic_probabilities" / "chapter_topic_probs.parquet"
TOPIC_LOOKUP_PATH = PROJECT_ROOT / "results" / "stage10_correlation_analysis" / "data_preparation" / "taxonomy_radway_eda" / "topic_lookup.parquet"
GOODREADS_PATH = PROJECT_ROOT / "data" / "processed" / "goodreads.csv"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "results" / "stage10_correlation_analysis" / "topic_analysis_all_368"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
```

### 0.3 Define Units of Analysis
- **Book-level**: one row per book, full topic mixture (sums to 1)
- **Segment-level**: one row per (book × segment {begin, middle, end}), topic mixture per segment
- **Topic-level**: topic as feature; outcomes are group differences/correlations

### 0.4 Data Integrity Checks
- Load all data files
- Verify book_topic_probs: each book has all 368 topics; check sum(prob) ≈ 1 per book
- Verify chapter_topic_probs: for each (book, segment), check sums; confirm all segments exist
- Verify books_meta: confirm group labels (rating_class: good/mid/bad), rating ranges, n_ratings, length, author counts
- Record inference procedure (sentence-level aggregation vs per-segment inference)

### 0.5 Handle Compositional Data Reality
- Document that topic probabilities are compositional (increasing one decreases others)
- Plan for composition-aware checks (log-ratio contrasts, Dirichlet-style modeling)

### 0.6 Topic Prevalence Filters (Prevent 300+ Topic Chaos)
For each topic, compute:
- **Prevalence**: fraction of books where prob > ε (e.g., > 0.001)
- **Mass**: mean prob across all books
- **Concentration**: how "spiky" it is (Gini coefficient or similar)

Create "topic health table" with:
- topic_id
- prevalence
- mass
- concentration
- manual label (from topic_lookup)
- taxonomy_main_name
- taxonomy_main_group
- radway_phase_name (if available)

**Deliverable:** `topic_health_table.parquet` saved to TABLE_DIR

---

## 1. Bottom Layer: Individual Topic Distributions Across Top/Medium/Trash

### 1.1 Merge Topic Probabilities with Book Metadata
- Merge book_topic_probs with books_meta to get rating_class
- Map rating_class: good → "Top", mid → "Medium", bad → "Trash"
- Verify all books have rating_class assigned

### 1.2 Visual Exploration (Distribution-First, Not Mean-First)

For each topic, create distribution comparisons across groups:

**1.2.1 Distribution Plots (per topic)**
- Violin plots (good for "shape" differences)
- Ridge plots / density plots (alternative visualization)
- ECDF curves (great when topic is mostly zero-ish and only sometimes spikes)
- Boxplots + jitter (good for seeing individual books)

**1.2.2 Batch Visualization Strategy**
- Create summary visualizations for all topics (grid/facet plots)
- Create individual detailed plots for top N topics (by prevalence/mass)
- Interactive Plotly figures saved to FIG_DIR

**Key questions per topic:**
- Is the topic present in all tiers but stronger in one?
- Or nearly absent in Top but common in Trash?
- Does it show bimodality (two clusters) suggesting subtypes or author effects?

### 1.3 Quantify Differences Per Topic (Screen, Don't Overpromise)

For each topic, compute:

**1.3.1 Group-wise Central Tendency**
- Median (robust to outliers)
- Mean (for comparison)
- Q1, Q3 (quartiles)

**1.3.2 Effect Sizes**
- Top vs Trash (primary comparison)
- Top vs Medium (optional)
- Medium vs Trash (optional)
- Use robust effect size: Cliff's delta or rank-biserial correlation

**1.3.3 Significance Testing (Optional but Recommended)**
- Non-parametric group test: Kruskal-Wallis for 3 groups
- Post-hoc pairwise comparisons: Mann-Whitney U tests
- Multiple comparisons correction: FDR/Benjamini-Hochberg

**1.3.4 Topic-Level Leaderboard**
Create tables:
- "Most Top-associated topics" (highest median/mean in Top)
- "Most Trash-associated topics" (highest median/mean in Trash)
- "Most Medium-peaked topics" (highest in Medium relative to others)
- "Topics with biggest Top–Trash separation" (largest effect size)

**Deliverables:**
- `topic_leaderboard_all.parquet` - full results for all 368 topics
- `topic_leaderboard_top_associated.parquet` - top N topics associated with Top tier
- `topic_leaderboard_trash_associated.parquet` - top N topics associated with Trash tier
- `topic_leaderboard_effect_sizes.parquet` - sorted by effect size

### 1.4 Tame Multiple Comparisons Problem

Apply sensible gates:
1. **Minimum prevalence**: appears in ≥ 15–20% of books
2. **Meaningful effect size**: not just p < .05, but |Cliff's delta| > threshold
3. **Survives FDR correction**: if testing, p_adj < 0.05
4. **Interpretable label**: topic actually coheres on inspection (from topic_lookup)

Create filtered leaderboard: `topic_leaderboard_filtered.parquet`

### 1.5 Author as "Shadow Confounder"

Before interpreting a topic as "Top loves X":
- Check whether topic is dominated by 1–2 authors
- Quick diagnostic: compute topic prevalence per author
- Flag topics: tier-stable vs author-driven

**Deliverable:** `topic_author_dominance.parquet` with flags

---

## 2. Mid Layer: Topic Clusters & Patterns

### 2.1 Topic Similarity Analysis
- Compute topic-topic correlation matrix (based on book-level probabilities)
- Identify topic clusters using hierarchical clustering
- Visualize topic similarity heatmap

### 2.2 Taxonomy Group Aggregations
- Aggregate topic probabilities by taxonomy_main_group
- Compare group-level distributions across tiers
- Statistical tests at group level

### 2.3 Radway Phase Aggregations
- Aggregate topic probabilities by radway_phase_name
- Compare phase distributions across tiers
- Test narrative arc hypotheses

---

## 3. Top Layer: Summary & Interpretation

### 3.1 Summary Statistics
- Number of topics significantly associated with each tier
- Effect size distributions
- Prevalence distributions

### 3.2 Key Findings Table
- Top 20 topics by effect size (Top vs Trash)
- Top 20 topics by prevalence
- Topics with strongest tier associations

### 3.3 Export All Results
- All tables to TABLE_DIR
- All figures to FIG_DIR
- Summary report (markdown/text)

---

## 4. Helper Functions

### 4.1 Cliff's Delta Implementation
```python
def cliffs_delta(x, y):
    """Compute Cliff's delta effect size."""
    # Implementation
    pass
```

### 4.2 Topic Prevalence Metrics
```python
def compute_topic_prevalence(book_topic_probs, threshold=0.001):
    """Compute prevalence, mass, and concentration for each topic."""
    pass
```

### 4.3 Distribution Comparison
```python
def compare_topic_distributions(topic_id, book_topic_probs, books_meta):
    """Compare topic distributions across rating classes."""
    pass
```

### 4.4 Plotly Figure Helper
```python
def show_plotly_fig(fig, save_html=True, output_dir=FIG_DIR):
    """Display Plotly figure with fallback to HTML save."""
    pass
```

---

## 5. Output Structure

```
results/stage10_correlation_analysis/topic_analysis_all_368/
├── figures/
│   ├── topic_distributions/
│   │   ├── all_topics_violin.html
│   │   ├── top_20_topics_detailed/
│   │   └── ...
│   ├── topic_leaderboards/
│   │   ├── top_associated_topics.html
│   │   └── ...
│   └── topic_similarity/
│       └── topic_correlation_heatmap.html
├── tables/
│   ├── topic_health_table.parquet
│   ├── topic_leaderboard_all.parquet
│   ├── topic_leaderboard_filtered.parquet
│   ├── topic_author_dominance.parquet
│   └── ...
└── summary_report.md
```

---

## Notes

1. **Compositional Data**: Always remember topic probabilities sum to 1 per book. Raw comparisons are informative, but consider log-ratio or Dirichlet modeling for formal tests.

2. **Multiple Comparisons**: With 368 topics, expect many "significant" results by chance. Use FDR correction and effect size thresholds.

3. **Author Effects**: Romance authors can imprint topics strongly. Always check author dominance before interpreting tier associations.

4. **BERTopic Reference**: When working with topic probabilities, refer to BERTopic's `approximate_distribution` method documentation for understanding how probabilities are computed.

5. **Performance**: For 368 topics × 92 books, batch processing and efficient data structures are essential. Use vectorized operations where possible.

