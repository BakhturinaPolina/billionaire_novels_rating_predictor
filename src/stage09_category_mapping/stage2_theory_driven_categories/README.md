# Stage 2: Theory-Driven Categories

## Overview

**Goal**: Map existing BERTopic topics to the **Romance Corpus Topic Taxonomy** using zero-shot classification with Mistral-Nemo via OpenRouter. This upgrades the pipeline from "LLM gives me nice labels" to "LLM + BERTopic do actual zero-shot taxonomy classification."

**Status**: ✅ **Implemented** - Zero-shot taxonomy classification module ready

## Approach

### Primary Method: Zero-Shot Taxonomy Classification

Map existing topics to a fixed **Romance Corpus Topic Taxonomy** (30+ nodes across 8 groups) without retraining. Uses `mistralai/Mistral-Nemo-Instruct-2407` via OpenRouter API, following the same style and patterns as Stage 08 labeling.

## Romance Corpus Topic Taxonomy

The taxonomy consists of 8 main groups with 30+ specific nodes:

1. **Embodied & Sensory Experience** (1.1, 1.2, 1.5)
   - Body Parts & Physical Reactions
   - Pain, Injury & Vulnerability
   - Exercise & Physical Activity

2. **Sexuality, Attraction & Intimacy** (2.1, 2.2, 2.3, 2.4)
   - Attraction & Sexual Tension
   - Kissing & Non-Explicit Affection
   - Explicit Sexual Acts
   - Aftercare & Post-Sex Reflection

3. **Emotions, Cognition & Inner Life** (3.1, 3.2, 3.3, 3.4)
   - Positive Emotions & Security
   - Negative Emotions & Distress
   - Ambivalence & Internal Conflict
   - Beliefs, Values & Moral Reflection

4. **Relationship Trajectory (Main Couple)** (4.1, 4.2, 4.3, 4.4, 4.5)
   - Meeting, First Impressions & Setup
   - Bonding, Everyday Intimacy & Growth
   - Secrets, Misunderstandings & Hidden Information
   - Conflict, Distance & Breakup Threats
   - Reconciliation, Commitments & HEA

5. **Social World Outside Couple** (5.1, 5.2, 5.3)
   - Family & Kinship
   - Friends & Social Circles
   - Community, Norms & Social Events

6. **Work, Wealth, Status & Institutions** (6.1, 6.2, 6.3, 6.4, 6.5)
   - Hero's Elite Work & Business World
   - Heroine's Work & Professional Identity
   - Shared Workplaces & Professional Interaction
   - Money, Housing & Economic Security
   - Law, Medicine, Education & Formal Institutions

7. **Conflict, Risk & Harm** (7.1, 7.2, 7.3)
   - Interpersonal Non-Romantic Conflict
   - Violence, Threats & Coercion
   - Risk, Danger & External Crises

8. **Spaces, Time, Activities & Objects** (8.1, 8.2, 8.3, 8.4)
   - Domestic Spaces & Routines
   - Public & Leisure Spaces
   - Objects, Technology & Everyday Artefacts
   - Time, Seasons & Temporal Framing

**Special Category**: `noise` - For boilerplate, technical artefacts, or paratext

## Implementation

### Module: `scripts/zeroshot_taxonomy_openrouter.py`

A self-contained module that:

- **Reuses existing helpers**: Imports `load_openrouter_client`, `rerank_snippets_centrality`, and `format_snippets` from Stage 08
- **Zero-shot classification**: Maps each topic to taxonomy nodes using Mistral-Nemo with JSON-only output
- **Input format**: Accepts Stage 08 labels JSON (with `keywords`, `label`, `scene_summary`, `primary_categories`, `secondary_categories`, `is_noise`)
- **Output format**: JSON mapping `{topic_id: {main_category_id, secondary_category_id, other_plausible_ids, is_noise, rationale}}`

### Key Features

- **Same API pattern**: Uses `OpenAI` client with OpenRouter base URL, same authentication as Stage 08
- **JSON-only output**: System prompt enforces JSON-only, with defensive parsing (strips markdown fences)
- **Taxonomy validation**: Validates taxonomy IDs against the fixed taxonomy list
- **Noise handling**: Respects `is_noise` from Stage 08 and enforces noise semantics
- **Automatic snippet extraction**: By default, loads BERTopic model (from Stage 08) and extracts representative documents for improved classification accuracy
- **Integration with Stage 1 & Stage 8**: Reuses model loading helpers from Stage 1 and snippet extraction from Stage 08
- **Error handling**: JSON parsing fallbacks, validation and correction of invalid taxonomy IDs

## Usage

### Basic Usage

```bash
python -m src.stage09_category_mapping.stage2_theory_driven_categories.scripts.zeroshot_taxonomy_openrouter \
  --labels-json results/stage08_llm_labeling/labels_pos_openrouter_mistralai_Mistral-Nemo-Instruct-2407_romance_aware_paraphrase-MiniLM-L6-v2.json \
  --output-json results/stage09_category_mapping/stage2_theory_driven_categories/taxonomy_mappings_mistral_nemo.json \
  --model-name mistralai/Mistral-Nemo-Instruct-2407
```

**Note**: By default, the script automatically loads the BERTopic model (from Stage 08) to extract representative document snippets for each topic. This improves taxonomy classification accuracy. Use `--no-snippets` to skip this step if you don't have the model available.

### Command-Line Options

**Required Arguments:**
- `--labels-json`: Path to Stage 08 labels JSON file
- `--output-json`: Output path for taxonomy mappings JSON

**OpenRouter API Options:**
- `--model-name`: Model name (default: `mistralai/Mistral-Nemo-Instruct-2407`)
- `--api-key`: OpenRouter API key (optional, uses `OPENROUTER_API_KEY` env var if not provided)
- `--temperature`: Sampling temperature (default: 0.25, low for stable classification)
- `--max-tokens`: Max tokens for JSON output (default: 220)

**BERTopic Model Options (for snippet extraction):**
- `--base-dir`: Base directory for BERTopic models (default: `models/retrained`)
- `--embedding-model`: Embedding model name (default: `paraphrase-MiniLM-L6-v2`)
- `--model-suffix`: Model suffix (default: `_with_llm_labels`)
- `--model-stage`: Stage subfolder for model (default: `stage08_llm_labeling`)
- `--max-docs-per-topic`: Maximum number of representative docs to extract per topic (default: 10)
- `--no-snippets`: Skip loading BERTopic model and extracting representative snippets

### Input Requirements

**Stage 08 Labels JSON** should contain topic metadata with:
- `keywords`: List of topic keywords (from BERTopic)
- `label`: LLM-generated label (from Stage 08)
- `scene_summary`: Scene summary (optional but recommended)
- `primary_categories`: List of primary categories (e.g., `["romance_core", "sexual_content"]`)
- `secondary_categories`: List of secondary categories (e.g., `["setting:bedroom", "activity:kissing"]`)
- `is_noise`: Boolean indicating if topic is noise/technical

### Output Format

The output JSON maps each topic to taxonomy classification:

```json
{
  "33": {
    "topic_id": 33,
    "main_category_id": "6.1",
    "secondary_category_id": "5.1",
    "other_plausible_ids": ["4.2"],
    "is_noise": false,
    "rationale": "Keywords and scene summary indicate business discussions in elite work context (6.1), with family elements (5.1)."
  },
  ...
}
```

**Fields**:
- `topic_id`: Integer topic ID (echoed from input)
- `main_category_id`: Required taxonomy ID (e.g., "4.2", "2.3", "noise")
- `secondary_category_id`: Optional second taxonomy ID (null if not applicable)
- `other_plausible_ids`: Optional list (0-3 items) of other plausible taxonomy IDs
- `is_noise`: Boolean (if true, `main_category_id` must be "noise")
- `confidence`: Required confidence level ("low", "medium", "high")
- `rationale`: 1-3 sentences explaining the classification

## Integration with Stage 1

After running Stage 1 (natural clusters), you can:

1. **Map meta-topics to taxonomy**: Use taxonomy mappings to understand what theoretical categories your natural meta-topics represent
2. **Compare approaches**: See if natural clusters align with theory-driven categories
3. **Combine insights**: Use taxonomy IDs for interpretable statistical analysis

## Integration Points

### Input Sources

- **Stage 08 labels**: Reads topic metadata from `results/stage08_llm_labeling/labels_pos_openrouter_*.json`
  - Includes: `keywords`, `label`, `scene_summary`, `primary_categories`, `secondary_categories`, `is_noise`
- **Stage 08 BERTopic model**: Automatically loads model from `models/retrained/paraphrase-MiniLM-L6-v2/stage08_llm_labeling/model_1_with_llm_labels/`
  - Extracts representative documents for each topic using `extract_representative_docs_per_topic()`
  - Improves classification accuracy by providing context beyond keywords

### Output Location

- **Taxonomy mappings**: Writes to `results/stage09_category_mapping/stage2_theory_driven_categories/taxonomy_mappings_*.json`

### Helper Functions Reused

- **From Stage 08**: `load_openrouter_client`, `rerank_snippets_centrality`, `format_snippets`, `extract_representative_docs_per_topic`
- **From Stage 1**: `load_native_bertopic_model` (via Stage 06 helpers)

## Validation & Error Handling

- **Taxonomy ID validation**: Validates all taxonomy IDs against the fixed taxonomy list
- **JSON parsing fallbacks**: Handles missing/invalid JSON responses with defensive parsing
- **Noise semantics enforcement**: If `is_noise=true`, `main_category_id` must be "noise" and `secondary_category_id` must be null
- **Fallback assignments**: Provides fallback taxonomy assignments based on previous categories if classification fails
- **Logging**: Comprehensive logging for mismatched topic IDs or invalid taxonomy assignments

## Analysis Helpers

After taxonomy classification, use these helper modules for downstream analysis:

### 1. Book-Level Category Proportions

**Module**: `scripts/aggregate_taxonomy_by_book.py`

Aggregates sentence-level topic assignments to book-level category proportions.

```bash
python -m src.stage09_category_mapping.stage2_theory_driven_categories.scripts.aggregate_taxonomy_by_book \
  --sentences results/stage06_topic_exploration/sentence_df_with_topics.parquet \
  --taxonomy-mapping results/stage09_category_mapping/stage2_theory_driven_categories/taxonomy_mappings_mistral_nemo.json \
  --output results/stage09_category_mapping/stage2_theory_driven_categories/book_category_proportions.parquet \
  --min-sentences-per-book 50
```

**Output**: Parquet file with columns:
- `book_id`, `rating_class`, `main_category_id`
- `n_sentences`, `total_sentences`, `prop` (proportion)

### 2. Statistical Analysis

**Module**: `scripts/stats_helpers.py`

Enhanced statistical analysis with Kruskal-Wallis tests, effect sizes, and post-hoc pairwise comparisons.

**Main Function**: `kruskal_by_rating()`

```python
from src.stage09_category_mapping.stage2_theory_driven_categories.scripts.stats_helpers import (
    kruskal_by_rating,
    pairwise_comparisons
)
import pandas as pd

book_cat = pd.read_parquet("book_category_proportions.parquet")
kw_results = kruskal_by_rating(book_cat)
kw_results.sort_values("p_value").head(15)  # See which categories differ most
```

**Output**: DataFrame with:
- `category_id`, `groups`, `n_books_per_group`
- `H_statistic`, `p_value`
- `eta_squared` (effect size: < 0.01 = negligible, 0.01-0.06 = small, 0.06-0.14 = medium, > 0.14 = large)
- `total_n` (total sample size)
- `significant` (boolean)

**Post-hoc Pairwise Comparisons**: `pairwise_comparisons()`

For significant categories, identify which specific rating classes differ:

```python
# Get pairwise comparisons for a significant category
pairwise_res = pairwise_comparisons(book_cat, "5.3", alpha=0.05)
# Returns DataFrame with group1, group2, U_statistic, p_value, p_value_corrected, 
# significant, median_diff
```

**Complete Analysis Script**: `scripts/analyze_category_differences.py`

Run comprehensive analysis with all visualizations:

```bash
python -m src.stage09_category_mapping.stage2_theory_driven_categories.scripts.analyze_category_differences \
  --book-cat results/stage09_category_mapping/stage2_theory_driven_categories/book_category_proportions.parquet \
  --output-dir results/stage09_category_mapping/stage2_theory_driven_categories/analysis \
  --top-n 10 \
  --alpha 0.05 \
  --taxonomy-json results/stage09_category_mapping/stage2_theory_driven_categories/taxonomy_mappings_openrouter_mistralai_Mistral-Nemo-Instruct-2407_paraphrase-MiniLM-L6-v2.json
```

This generates:
- Statistical results CSV with effect sizes
- Overview plots (volcano plot, effect size bars, p-value heatmap)
- Individual category plots (enhanced violin plots)
- Pairwise comparison plots for significant categories

### 3. Visualization

**Module**: `scripts/visualization_helpers.py`

Comprehensive visualization suite with multiple plot types for statistical analysis.

**Individual Category Plots**: `plot_category_prevalence()`

Enhanced violin plots (or box plots) showing full distribution shapes:

```python
from src.stage09_category_mapping.stage2_theory_driven_categories.scripts.visualization_helpers import (
    plot_category_prevalence,
    plot_volcano,
    plot_effect_size_bars,
    plot_pairwise_comparisons,
    plot_pvalue_heatmap
)
import pandas as pd

book_cat = pd.read_parquet("book_category_proportions.parquet")

# Enhanced violin plot (default) - shows full distribution shape
plot_category_prevalence(book_cat, "4.4", plot_type="violin")

# Traditional box plot
plot_category_prevalence(book_cat, "2.3", plot_type="box")

# Both combined
plot_category_prevalence(book_cat, "5.3", plot_type="both")
```

**Overview Plots**:

```python
# Load statistical results
kw_results = pd.read_csv("kruskal_wallis_results.csv")

# Volcano plot: significance vs effect size
fig, ax = plot_volcano(kw_results, alpha=0.05, effect_threshold=0.01)

# Effect size bar chart: top categories by effect size
fig, ax = plot_effect_size_bars(kw_results, top_n=15, alpha=0.05)

# P-value heatmap: all categories overview
fig, ax = plot_pvalue_heatmap(kw_results, group_by="category_group")
```

**Post-hoc Pairwise Comparisons**:

```python
from src.stage09_category_mapping.stage2_theory_driven_categories.scripts.stats_helpers import pairwise_comparisons

# Get pairwise results
pairwise_res = pairwise_comparisons(book_cat, "5.3", alpha=0.05)

# Visualize which groups differ
fig, ax = plot_pairwise_comparisons(
    pairwise_res, 
    category_id="5.3",
    category_name="Community, Norms & Social Events"
)
```

**Available Plot Types**:
- **Volcano Plot**: Identifies categories that are both significant AND have large effects
- **Effect Size Bars**: Ranks categories by practical significance (effect size)
- **P-value Heatmap**: Quick overview of all categories with color-coded significance
- **Violin Plots**: Better visualization of distribution shapes than box plots
- **Pairwise Comparisons**: Shows which specific rating classes differ for significant categories

**Documentation**: See `scripts/VISUALIZATION_EXAMPLES.md` for detailed usage guide and interpretation examples.

## Statistical Analysis Results

**Key Findings** (from `STATISTICAL_ANALYSIS_REPORT.md`):

- **3 categories** show statistically significant differences (p < 0.05) across rating classes:
  1. **5.3: Community, Norms & Social Events** (p = 0.029, η² = 0.070 - **medium effect**)
  2. **6.2: Heroine's Work & Professional Identity** (p = 0.047, η² = 0.057 - small-medium effect)
  3. **3.4: Beliefs, Values & Moral Reflection** (p = 0.048, η² = 0.048 - small effect)

- **24 out of 27 categories** show no significant differences, indicating thematic content is largely consistent across rating classes

- **Effect sizes** help distinguish statistical significance from practical significance

**Visualization Outputs** (in `analysis/figures/`):
- `volcano_plot.png` - Overview of significance vs effect size
- `effect_size_bars.png` - Top categories by effect size
- `pvalue_heatmap.png` - All categories at a glance
- `category_*_prevalence.png` - Enhanced violin plots for individual categories
- `category_*_pairwise.png` - Post-hoc comparisons for significant categories

**Documentation**:
- `STATISTICAL_ANALYSIS_REPORT.md` - Complete analysis report with effect sizes and interpretations
- `scripts/VISUALIZATION_EXAMPLES.md` - Usage guide and interpretation examples
- `scripts/IMPROVEMENTS_SUMMARY.md` - Summary of visualization enhancements
- `scripts/VISUALIZATION_IMPROVEMENTS.md` - Technical overview of improvements

## Next Steps

After taxonomy classification and analysis:

1. **Interpret results**: Use statistical tests, effect sizes, and visualizations to understand category differences
2. **Review pairwise comparisons**: For significant categories, examine which specific rating classes differ
3. **Compare with Stage 1**: See how theory-driven categories compare to natural clusters
4. **Refine taxonomy**: Use confidence scores and manual review to improve mappings
5. **Investigate effect sizes**: Focus on categories with large effects, even if not statistically significant (may be underpowered)

## Dependencies

```python
from openai import OpenAI
from pathlib import Path
import json
import logging

# Reuses helpers from Stage 08:
from src.stage08_llm_labeling.openrouter_experiments.core.generate_labels_openrouter import (
    load_openrouter_client,
    rerank_snippets_centrality,
    format_snippets,
)
```

## References

- Stage 08 LLM Labeling: `src/stage08_llm_labeling/openrouter_experiments/core/generate_labels_openrouter.py`
- Stage 1 Natural Clusters: `../stage1_natural_clusters/README.md`
- OpenRouter API: https://openrouter.ai/
