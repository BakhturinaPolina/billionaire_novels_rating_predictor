# Statistical Analysis Master Plan

This directory contains a structured set of notebooks for comprehensive statistical analysis aligned with research questions (RQ) and hypotheses (H1–H6).

## 📁 Structure

### Notebook 1: `01_data_validation_and_preprocessing.ipynb`
**Purpose**: Ensure clean, normalized data, build derived indices, segment books.

**Tasks**:
- Validate and load:
  - `book_topic_probs.csv`, `chapter_topic_probs.csv`
  - `books_meta.csv`
  - `taxonomy_mappings_*.json`, `taxonomy_with_radway.json`
- Normalize topic probabilities (per book/segment sum to 1)
- Merge topic proportions with metadata
- Segment books into **begin/middle/end** if chapter-level topic probs missing
- Compute derived **composite category proportions** per book

**Output**:
- `book_category_props.csv`
- `chapter_category_props.csv` (if available)
- Cleaned merged dataset with metadata + category values

---

### Notebook 2: `02_index_computation.ipynb`
**Purpose**: Compute **theory-aligned indices** for all books & segments.

**Indices** (as per hypotheses):
- `Love-over-Sex`: `(commitment_hea + tenderness_emotion) - explicit`
- `HEA Index`: `commitment_hea + symbolic_gifts + festive_rituals`
- `Luxury × Love`: `luxury × (commitment_hea + tenderness)`
- `Protective – Jealous`: `protectiveness - jealousy`
- `Dark-vs-Tender`: `(neg_affect + threat_dark) - tenderness`
- `Miscommunication Balance`: `(commitment + tenderness + repair) - miscommunication`
- Segment-wise: same indices per begin/middle/end

**Output**:
- `indices_book.csv`, `indices_chapter.csv`

---

### Notebook 3: `03_exploratory_analysis.ipynb`
**Purpose**: General EDA to understand category & index distributions.

**Visuals**:
- Heatmaps of category proportions (grouped by Top/Mid/Trash)
- Distribution of indices (histograms, KDE)
- UMAP/TSNE of books by category proportions (colored by group)
- Correlation matrix of indices

**Output**:
- Figures: `eda/` folder
- Insights about patterns & outliers

---

### Notebook 4: `04_group_comparisons.ipynb`
**Purpose**: Test hypotheses H1–H5 across popularity groups.

**Tests**:
- **Kruskal-Wallis** for non-parametric group comparison
- **Post-hoc tests** with Holm correction
- **Effect sizes**: Eta-squared, Cohen's d
- **Group-wise violin plots** of category proportions and indices

**Output**:
- `kruskal_wallis_results.csv`
- Visuals: volcano plots, violin plots, pairwise tests

**Aligned Hypotheses**:
- H1: `(Love-over-Sex index) → Top > Trash`
- H2: `(HEA Index) → Top > Trash`
- H4: `(Protective–Jealousy Delta) → Top > Trash`
- H5: `(Darkness–Tenderness) → Top < Trash`

---

### Notebook 5: `05_modeling_prediction.ipynb`
**Purpose**: Model whether theory-aligned categories & indices predict:
1. **Goodreads rating**
2. **Popularity tier (Top vs Trash)**

**Models**:
- **Logistic regression** (Top vs Trash)
- **OLS regression** (`avg_rating`)
- **Key interactions** (e.g. `Luxury × Love`, `Protective–Jealousy`)
- **Controls**: `length`, `author_id`, `year` (if available)

**Output**:
- Coefficients table + CI
- Interaction plots
- Predictive accuracy

---

### Notebook 6: `06_timecourse_analysis.ipynb`
**Purpose**: Test **H6**: Emotional arc over story time (begin→end).

**Analysis**:
- **Repeated Measures ANOVA** (or mixed effects) with:
  - Within factor: `segment (begin, middle, end)`
  - Dependent vars: `commitment`, `repair`, `miscommunication`, `neg_affect`
- **Trend plots** of arcs for Top vs Trash

**Output**:
- Statistical test results
- Timecourse plots for indices per group

---

### Notebook 7: `07_robustness_and_sensitivity.ipynb`
**Purpose**: Verify robustness of results.

**Tasks**:
- **Leave-one-author-out** validation
- **Bootstrap confidence intervals**
- **Alternative thresholds** (e.g., redefining "Top")

---

### Notebook 8: `08_final_report_generation.ipynb`
**Purpose**: Assemble final outputs into publishable format.

**Includes**:
- Summary statistics
- Key figures and statistical tables
- Interpretation of hypotheses
- Save as `.docx` and `.pdf`

---

## 📊 Data Dependencies

### Input Files (Expected Locations)
- `results/stage09_category_mapping/stage2_theory_driven_categories/book_category_proportions.parquet`
- `results/stage09_category_mapping/stage2_theory_driven_categories/taxonomy_mappings_*.json`
- `results/stage09_category_mapping/stage3_radway_functions/taxonomy_with_radway.json`
- `data/processed/goodreads.csv`
- `data/processed/chapters.csv`

### Output Files
All outputs will be saved to:
- `results/stage10_correlation_analysis/statistical_analysis/`

---

## 🔄 Execution Order

Execute notebooks in numerical order (01 → 08). Each notebook depends on outputs from previous notebooks.

---

## 📝 Notes

- All notebooks should include clear markdown documentation
- Statistical tests should report effect sizes, not just p-values
- Visualizations should be publication-ready (high DPI, clear labels)
- Results should be reproducible (set random seeds where applicable)
