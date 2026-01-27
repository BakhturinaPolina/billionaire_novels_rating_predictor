# Stage 2: Theory-Driven Categories

Zero-shot classification of BERTopic topics to the Romance Corpus Topic Taxonomy using Mistral-Nemo via OpenRouter.

## Status

✅ **Implemented** — Zero-shot taxonomy classification module ready.

## Usage

### Basic Classification

```bash
python -m src.stage09_category_mapping.stage2_theory_driven_categories.scripts.zeroshot_taxonomy_openrouter \
  --labels-json results/stage08_llm_labeling/labels_pos_openrouter_*.json \
  --output-json results/stage09_category_mapping/stage2_theory_driven_categories/taxonomy_mappings.json \
  --model-name mistralai/Mistral-Nemo-Instruct-2407
```

### With Snippet Extraction (Recommended)

```bash
python -m src.stage09_category_mapping.stage2_theory_driven_categories.scripts.zeroshot_taxonomy_openrouter \
  --labels-json results/stage08_llm_labeling/labels_pos_openrouter_*.json \
  --output-json results/stage09_category_mapping/stage2_theory_driven_categories/taxonomy_mappings.json \
  --base-dir models/retrained \
  --embedding-model paraphrase-MiniLM-L6-v2 \
  --max-docs-per-topic 10
```

## Inputs

| Source | Path | Description |
|--------|------|-------------|
| Labels | `results/stage08_llm_labeling/labels_pos_openrouter_*.json` | Stage 08 topic labels |
| Model | `models/retrained/{embedding_model}/stage08_llm_labeling/model_1_with_llm_labels/` | BERTopic model (for snippets) |

## Outputs

| Output | Path | Description |
|--------|------|-------------|
| Taxonomy | `results/stage09_category_mapping/stage2_theory_driven_categories/taxonomy_mappings_*.json` | Topic-to-taxonomy mappings |

## Module Structure

| File | Purpose |
|------|---------|
| `scripts/zeroshot_taxonomy_openrouter.py` | Main classification script |
| `scripts/aggregate_taxonomy_by_book.py` | Book-level category aggregation |
| `scripts/stats_helpers.py` | Kruskal-Wallis tests, effect sizes |
| `scripts/visualization_helpers.py` | Volcano plots, violin plots, heatmaps |
| `scripts/analyze_category_differences.py` | Complete analysis pipeline |
| `scripts/compare_models.py` | Model comparison utility |

## CLI Options

### Required

- `--labels-json`: Path to Stage 08 labels JSON
- `--output-json`: Output path for taxonomy mappings

### OpenRouter API

- `--model-name`: Model name (default: `mistralai/Mistral-Nemo-Instruct-2407`)
- `--api-key`: OpenRouter API key (or set `OPENROUTER_API_KEY` env var)
- `--temperature`: Sampling temperature (default: 0.25)
- `--max-tokens`: Max tokens for output (default: 220)

### BERTopic Model (for snippets)

- `--base-dir`: Base directory for models (default: `models/retrained`)
- `--embedding-model`: Embedding model name (default: `paraphrase-MiniLM-L6-v2`)
- `--model-suffix`: Model suffix (default: `_with_llm_labels`)
- `--model-stage`: Stage subfolder (default: `stage08_llm_labeling`)
- `--max-docs-per-topic`: Max representative docs per topic (default: 10)
- `--no-snippets`: Skip snippet extraction

## Output Format

```json
{
  "33": {
    "topic_id": 33,
    "main_category_id": "6.1",
    "secondary_category_id": "5.1",
    "other_plausible_ids": ["4.2"],
    "is_noise": false,
    "confidence": "medium",
    "rationale": "Keywords indicate business discussions..."
  }
}
```

## Analysis Pipeline

After taxonomy classification:

```bash
# 1. Aggregate to book level
python -m src.stage09_category_mapping.stage2_theory_driven_categories.scripts.aggregate_taxonomy_by_book \
  --sentences results/stage06_topic_exploration/sentence_df_with_topics.parquet \
  --taxonomy-mapping results/stage09_category_mapping/stage2_theory_driven_categories/taxonomy_mappings.json \
  --output results/stage09_category_mapping/stage2_theory_driven_categories/book_category_proportions.parquet

# 2. Run statistical analysis
python -m src.stage09_category_mapping.stage2_theory_driven_categories.scripts.analyze_category_differences \
  --book-cat results/stage09_category_mapping/stage2_theory_driven_categories/book_category_proportions.parquet \
  --output-dir results/stage09_category_mapping/stage2_theory_driven_categories/analysis \
  --top-n 10
```

## Notes

- Uses same API pattern as Stage 08 LLM labeling
- JSON-only output with defensive parsing
- Validates taxonomy IDs against fixed taxonomy list
- Respects `is_noise` from Stage 08

## See Also

- [Methodology Report](../../../reports/01_stage_reports/stage09_category_mapping/02_stage2_theory_driven_categories/stage09_stage2_taxonomy_classification_methodology.md) — Research rationale and results
- [Model Comparison](../../../reports/01_stage_reports/stage09_category_mapping/02_stage2_theory_driven_categories/stage09_stage2_model_comparison.md) — Model comparison report
