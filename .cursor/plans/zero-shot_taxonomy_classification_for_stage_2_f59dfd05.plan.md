---
name: Zero-shot taxonomy classification for Stage 2
overview: Implement zero-shot taxonomy classification module that maps BERTopic topics to the Romance Corpus Topic Taxonomy using Mistral-Nemo via OpenRouter, following the same style and patterns as Stage 08 labeling.
todos:
  - id: create-scripts-folder
    content: Create scripts/ directory in stage2_theory_driven_categories with __init__.py
    status: completed
  - id: implement-taxonomy-module
    content: Create zeroshot_taxonomy_openrouter.py with taxonomy definition, prompts, and classification functions
    status: completed
    dependencies:
      - create-scripts-folder
  - id: add-cli-interface
    content: Add command-line argument parsing and main entry point to the module
    status: completed
    dependencies:
      - implement-taxonomy-module
  - id: update-readme
    content: Update stage2_theory_driven_categories/README.md with usage instructions, examples, and integration notes
    status: completed
    dependencies:
      - implement-taxonomy-module
---

# Zero-Shot Taxonomy Classification for Stage 2

## Overview

Implement a zero-shot taxonomy classification module in `stage2_theory_driven_categories` that maps BERTopic topics to the Romance Corpus Topic Taxonomy using `mistralai/Mistral-Nemo-Instruct-2407` via OpenRouter. This upgrades the pipeline from simple LLM labeling to structured taxonomy classification.

## Implementation Plan

### 1. Create Module Structure

Create the following files in `src/stage09_category_mapping/stage2_theory_driven_categories/`:

- **`scripts/zeroshot_taxonomy_openrouter.py`** - Main module for taxonomy classification
- **`scripts/__init__.py`** - Package initialization (if needed)

### 2. Core Module Implementation

The module will:

- **Reuse existing helpers**: Import `load_openrouter_client`, `rerank_snippets_centrality`, and `format_snippets` from `src.stage08_llm_labeling.openrouter_experiments.core.generate_labels_openrouter`
- **Define taxonomy**: Include the complete Romance Corpus Topic Taxonomy (30+ nodes across 8 groups) as a constant
- **Zero-shot classification**: Map each topic to taxonomy nodes using Mistral-Nemo with JSON-only output
- **Input format**: Accept Stage 08 labels JSON (with `keywords`, `label`, `scene_summary`, `primary_categories`, `secondary_categories`, `is_noise`)
- **Output format**: JSON mapping `{topic_id: {main_category_id, secondary_category_id, other_plausible_ids, is_noise, rationale}}`

### 3. Key Features

- **Same API pattern**: Uses `OpenAI` client with OpenRouter base URL, same authentication
- **JSON-only output**: System prompt enforces JSON-only, with defensive parsing (strips markdown fences)
- **Taxonomy validation**: Validates taxonomy IDs against the fixed taxonomy list
- **Noise handling**: Respects `is_noise` from Stage 08 and enforces noise semantics
- **Representative snippets**: Optional support for including representative docs (reuses existing snippet extraction)
- **Error handling**: Retry logic, JSON parsing fallbacks, validation and correction of invalid taxonomy IDs

### 4. Command-Line Interface

Create a CLI that accepts:

- `--labels-json`: Path to Stage 08 labels JSON file
- `--output-json`: Output path for taxonomy mappings
- `--model-name`: Model name (default: `mistralai/Mistral-Nemo-Instruct-2407`)
- `--api-key`: OpenRouter API key (optional, uses env var if not provided)
- `--temperature`: Sampling temperature (default: 0.25)
- `--max-tokens`: Max tokens for JSON output (default: 220)

### 5. Update README

Update `stage2_theory_driven_categories/README.md` to include:

- Overview of zero-shot taxonomy classification approach
- Input requirements (Stage 08 labels JSON)
- Usage instructions with example command
- Output format description
- Integration with Stage 1 results (if applicable)
- Link to taxonomy definition

## Files to Create/Modify

1. **`src/stage09_category_mapping/stage2_theory_driven_categories/scripts/zeroshot_taxonomy_openrouter.py`** (new, ~600 lines)

   - Taxonomy definition (30+ nodes)
   - System and user prompts for zero-shot classification
   - `classify_topic_to_taxonomy_openrouter()` function
   - `map_all_topics_to_taxonomy()` batch function
   - CLI entry point

2. **`src/stage09_category_mapping/stage2_theory_driven_categories/scripts/__init__.py`** (new, minimal)

   - Package initialization

3. **`src/stage09_category_mapping/stage2_theory_driven_categories/README.md`** (update)

   - Add zero-shot taxonomy classification section
   - Usage examples
   - Input/output specifications
   - Integration notes

## Technical Details

### Taxonomy Structure

The Romance Corpus Topic Taxonomy includes:

- 8 main groups (Embodied & Sensory, Sexuality & Intimacy, Emotions & Cognition, Relationship Trajectory, Social World, Work & Wealth, Conflict & Risk, Spaces & Time)
- 30+ specific taxonomy nodes (e.g., "1.1 Body Parts & Physical Reactions", "2.3 Explicit Sexual Acts", "4.2 Bonding & Everyday Intimacy")
- Special "noise" category for technical/paratext topics

### Prompt Design

- **System prompt**: Defines taxonomy, output schema, field rules, special rules (e.g., violence vs exercise distinction)
- **User prompt**: Formats topic metadata (keywords, label, scene_summary, categories, snippets) for classification
- **JSON schema**: Strict schema with `topic_id`, `main_category_id`, `secondary_category_id`, `other_plausible_ids`, `is_noise`, `rationale`

### Integration Points

- **Stage 08 labels**: Reads from `results/stage08_llm_labeling/labels_pos_openrouter_*.json`
- **Output location**: Writes to `results/stage09_category_mapping/stage2_theory_driven_categories/taxonomy_mappings_*.json`
- **Helper functions**: Reuses `load_openrouter_client`, `rerank_snippets_centrality`, `format_snippets` from Stage 08

## Example Usage

```bash
python -m src.stage09_category_mapping.stage2_theory_driven_categories.scripts.zeroshot_taxonomy_openrouter \
  --labels-json results/stage08_llm_labeling/labels_pos_openrouter_mistralai_Mistral-Nemo-Instruct-2407_romance_aware_paraphrase-MiniLM-L6-v2.json \
  --output-json results/stage09_category_mapping/stage2_theory_driven_categories/taxonomy_mappings_mistral_nemo.json \
  --model-name mistralai/Mistral-Nemo-Instruct-2407
```

## Validation & Error Handling

- Validate taxonomy IDs against the fixed taxonomy list
- Handle missing/invalid JSON responses with fallbacks
- Enforce noise semantics (if `is_noise=true`, `main_category_id` must be "noise")
- Log warnings for mismatched topic IDs or invalid taxonomy assignments
- Provide fallback taxonomy assignments based on previous categories if classification fails