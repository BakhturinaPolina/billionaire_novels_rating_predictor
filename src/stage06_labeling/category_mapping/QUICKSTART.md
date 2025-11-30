# Category Mapping Quick Start Guide

Quick reference for common workflows in the category mapping module.

## Common Workflows

### 1. Basic Category Mapping

Map topic labels to categories:

```bash
python -m src.stage06_labeling.category_mapping.main_category_mapping \
    --labels results/stage06_labeling_openrouter/labels_pos_openrouter_romance_aware_paraphrase-MiniLM-L6-v2.json \
    --outdir results/stage06_labeling/category_mapping
```

**Output**: `topic_to_category_probs.json`, `topic_to_category_final.csv`, `topic_to_category_summary.csv`

---

### 2. Category Mapping with Fix Z

Automatically fix misclassified Z topics:

```bash
python -m src.stage06_labeling.category_mapping.main_category_mapping \
    --labels results/stage06_labeling_openrouter/labels_pos_openrouter_romance_aware_paraphrase-MiniLM-L6-v2.json \
    --outdir results/stage06_labeling/category_mapping \
    --fix-z
```

**Requirements**: `OPENROUTER_API_KEY` environment variable must be set.

---

### 3. Run Fix Z Standalone

Fix Z topics independently (useful for testing or re-running):

```bash
# First, run basic mapping
python -m src.stage06_labeling.category_mapping.main_category_mapping \
    --labels results/stage06_labeling_openrouter/labels_pos_openrouter_romance_aware_paraphrase-MiniLM-L6-v2.json \
    --outdir results/stage06_labeling/category_mapping

# Then run fix Z
python -m src.stage06_labeling.category_mapping.fix_z_topics \
    --labels results/stage06_labeling_openrouter/labels_pos_openrouter_romance_aware_paraphrase-MiniLM-L6-v2.json \
    --category-probs results/stage06_labeling/category_mapping/topic_to_category_probs.json \
    --outdir results/stage06_labeling/category_mapping \
    --limit 5  # Test with 5 topics first
```

---

### 4. Compare Before/After Fix Z

Analyze the impact of fix Z:

```bash
python -m src.stage06_labeling.category_mapping.compare_fix_z_results \
    --original results/stage06_labeling/category_mapping/topic_to_category_probs.json \
    --fixed results/stage06_labeling/category_mapping/topic_to_category_probs_fixed.json \
    --fix-results results/stage06_labeling/category_mapping/fix_z_results.json \
    --labels results/stage06_labeling_openrouter/labels_pos_openrouter_romance_aware_paraphrase-MiniLM-L6-v2.json \
    --output results/stage06_labeling/category_mapping/fix_z_comparison_report.md
```

**Output**: Markdown report with statistics and detailed changes.

---

### 5. Run Tests

Verify category mapping correctness:

```bash
pytest src/stage06_labeling/category_mapping/tests/test_category_mapping.py -v
```

---

### 6. Generate Labels with Improved Prompts

Use improved prompts that include category information:

```bash
python -m src.stage06_labeling.openrouter_experiments.main_openrouter \
    --topics-json results/stage06_exploration/topics_all_representations_paraphrase-MiniLM-L6-v2.json \
    --use-improved-prompts \
    --output-dir results/stage06_labeling_openrouter
```

**Note**: Labels generated with `--use-improved-prompts` include category information in the JSON output.

---

## Command-Line Cheat Sheet

### main_category_mapping

```bash
# Required
--labels <path>              # Labels JSON file

# Optional
--outdir <path>              # Output directory (default: results/stage06_labeling/category_mapping)
--book-topic-probs <path>    # Book-topic probability CSV (for aggregation)
--config <path>              # Config file for path resolution
--fix-z                      # Run LLM-based fix for Z topics
```

### fix_z_topics

```bash
# Required
--labels <path>              # Labels JSON file
--category-probs <path>      # Category probabilities JSON
--outdir <path>              # Output directory

# Optional
--api-key <key>              # OpenRouter API key (or set OPENROUTER_API_KEY)
--model <name>               # Model name (default: mistralai/mistral-nemo)
--temperature <float>        # Temperature (default: 0.3)
--limit <int>                # Limit topics to process
```

### compare_fix_z_results

```bash
# Required
--original <path>            # Original category probabilities JSON
--fixed <path>               # Fixed category probabilities JSON
--output <path>              # Output markdown report

# Optional
--fix-results <path>         # Fix Z results JSON (for LLM statistics)
--labels <path>              # Labels JSON (for topic context)
```

---

## File Structure

```
results/stage06_labeling/category_mapping/
├── topic_to_category_probs.json          # Category mappings (original)
├── topic_to_category_probs_fixed.json    # Category mappings (after fix Z)
├── topic_to_category_final.csv           # Long format CSV
├── topic_to_category_summary.csv         # One row per topic
├── fix_z_results.json                     # LLM responses from fix Z
└── fix_z_comparison_report.md            # Comparison report
```

---

## Environment Variables

```bash
# Required for fix Z functionality
export OPENROUTER_API_KEY=your_key_here
```

---

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| `OPENROUTER_API_KEY not set` | Set environment variable: `export OPENROUTER_API_KEY=...` |
| Tests failing | Run `pytest -v` to see which tests fail, then check regex patterns |
| JSON parsing errors | Check `fix_z_results.json` for malformed responses |
| Too many API calls | Use `--limit` flag to test with fewer topics |

---

## Next Steps

1. **Run basic mapping** to get baseline categories
2. **Run tests** to verify correctness
3. **Run fix Z** (with `--limit`) to test LLM integration
4. **Compare results** to see impact
5. **Integrate into pipeline** for production use

For detailed documentation, see [README.md](README.md).

