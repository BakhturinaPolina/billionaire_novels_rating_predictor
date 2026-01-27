# Stage 3: Radway Narrative Functions

Zero-shot classification of BERTopic topics to Radway's 13 narrative functions for analyzing story structure.

## Status

✅ **Implemented** — Classification completed for all 361 topics with heuristic overrides.

## Usage

### Basic Classification

```bash
python -m src.stage09_category_mapping.stage3_radway_functions.scripts.zeroshot_radway_openrouter \
  --taxonomy-json results/stage09_category_mapping/stage2_theory_driven_categories/taxonomy_mappings.json \
  --output-json results/stage09_category_mapping/stage3_radway_functions/taxonomy_with_radway.json \
  --model-name mistralai/Mistral-Nemo-Instruct-2407 \
  --api-key YOUR_API_KEY
```

### From BERTopic Model (Recommended)

```bash
python -m src.stage09_category_mapping.stage3_radway_functions.scripts.zeroshot_radway_openrouter \
  --taxonomy-json models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_taxonomy_mappings \
  --output-json results/stage09_category_mapping/stage3_radway_functions/taxonomy_with_radway.json \
  --model-name mistralai/Mistral-Nemo-Instruct-2407
```

### Update Model with Mappings

```bash
python -m src.stage09_category_mapping.stage3_radway_functions.scripts.update_model_with_radway \
  --merged-json results/stage09_category_mapping/stage3_radway_functions/taxonomy_with_radway.json \
  --source-model-suffix _with_taxonomy_mappings \
  --target-model-suffix _with_radway_mappings
```

## Inputs

| Source | Path | Description |
|--------|------|-------------|
| Taxonomy | `results/stage09_category_mapping/stage2_theory_driven_categories/taxonomy_mappings_*.json` | Stage 2 taxonomy mappings |
| Model | `models/retrained/{embedding_model}/stage09_category_mapping/model_1_with_taxonomy_mappings` | BERTopic model with taxonomy |

## Outputs

| Output | Path | Description |
|--------|------|-------------|
| Merged JSON | `results/stage09_category_mapping/stage3_radway_functions/taxonomy_with_radway.json` | Taxonomy + Radway mappings |
| Updated Model | `models/retrained/{embedding_model}/stage09_category_mapping/model_1_with_radway_mappings` | Model with Radway attached |

## Module Structure

| File | Purpose |
|------|---------|
| `scripts/zeroshot_radway_openrouter.py` | Main classification script |
| `scripts/update_model_with_radway.py` | Model update script |

## CLI Options

### Required

- `--taxonomy-json`: Path to Stage 2 taxonomy JSON or BERTopic model
- `--output-json`: Output path for merged JSON

### OpenRouter API

- `--model-name`: Model name (default: `mistralai/Mistral-Nemo-Instruct-2407`)
- `--api-key`: OpenRouter API key (or set `OPENROUTER_API_KEY` env var)
- `--temperature`: Sampling temperature (default: 0.0 for deterministic)
- `--max-tokens`: Max tokens for output (default: 220)

### Testing

- `--limit-topics`: Limit to first N topics (for testing)
- `--no-snippets`: Skip loading BERTopic model for snippets
- `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

## Output Format

```json
{
  "33": {
    "topic_id": 33,
    "main_category_id": "4.2",
    "source_metadata": { ... },
    "radway_functions": {
      "radway_main_id": "R8",
      "radway_secondary_id": "R9",
      "radway_other_plausible_ids": ["R10"],
      "radway_phase": "II",
      "radway_is_none": false,
      "radway_confidence": "medium",
      "radway_rationale": "...",
      "radway_main_name": "Hero treats heroine tenderly",
      "radway_phase_name": "Turning Point & Recognition"
    }
  }
}
```

## Heuristic Overrides

Post-LLM corrections for systematic errors:

| Pattern | Override |
|---------|----------|
| Explicit sex scenes (taxonomy 2.3) | → R12 (not R4) |
| Wedding/marriage/commitment cues | → R11/R13 (not none/R8) |
| R7 only with actual breakup cues | Narrowed scope |
| R4 sanity checks | Non-sexual contexts flagged |

## Notes

- Uses deterministic decoding (temperature=0.0) for consistency
- Reuses Stage 08 OpenRouter helpers
- Taxonomy can be loaded from JSON or model `topic_metadata_` attribute

## See Also

- [Research Report](../../../reports/01_stage_reports/stage09_category_mapping/03_stage3_radway_functions/stage09_stage3_radway_narrative_functions_research_report.md) — Methodology and results
