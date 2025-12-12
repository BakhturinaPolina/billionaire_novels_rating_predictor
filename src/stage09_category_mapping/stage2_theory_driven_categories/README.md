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

## Next Steps

After taxonomy classification:

1. **Aggregate per book**: Compute taxonomy category proportions per book
2. **Statistical analysis**: Test category differences across rating classes (similar to Stage 1)
3. **Visualization**: Create visualizations of category prevalence by quality
4. **Compare with Stage 1**: See how theory-driven categories compare to natural clusters

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
