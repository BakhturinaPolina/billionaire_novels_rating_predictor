# Stage 3: Radway Narrative Functions

## Overview

**Goal**: Map BERTopic topics to Radway's 13 narrative functions to analyze story structure and track narrative progression across books. Compare narrative patterns between bad/mid/good rated books.

**Status**: ✅ **Implemented** (can run independently or after Stages 1-2)

## Radway's 13 Functions

The implementation uses Radway's 13 narrative functions organized into three phases:

### Phase I: Initial Conflict & Isolation (The Setup)
- **R1**: Heroine's social identity is destroyed
- **R2**: Heroine reacts antagonistically to the hero
- **R3**: Hero responds ambiguously to heroine
- **R4**: Heroine interprets hero's behaviour as purely sexual interest
- **R5**: Heroine responds with anger or coldness
- **R6**: Hero retaliates or punishes heroine
- **R7**: Hero and heroine are physically or emotionally separated

### Phase II: Turning Point & Recognition (Developing Empathy)
- **R8**: Hero treats heroine tenderly
- **R9**: Heroine responds warmly to hero's tenderness
- **R10**: Heroine reinterprets hero's behaviour as result of previous hurt

### Phase III: Commitment & Restoration (The Happy Ending)
- **R11**: Hero declares love and demonstrates commitment
- **R12**: Heroine responds sexually and emotionally
- **R13**: Heroine's identity is restored

Additionally, topics that don't fit any function are classified as **"none"**.

## Implementation

### Architecture

The implementation uses the **taxonomy JSON from Stage 2 as the single source of truth**, which already contains:
- Taxonomy mappings (main_category_id, secondary_category_id, etc.)
- Source metadata (label, keywords, scene_summary, primary/secondary categories)

The Radway function mappings are merged back into the taxonomy JSON under a `"radway_functions"` key, preserving all existing fields.

### Zero-Shot Classification Approach

1. **Topic-level classification**: Maps each BERTopic topic to Radway functions using:
   - Topic keywords from BERTopic
   - LLM-generated labels and scene summaries (from Stage 8)
   - Stage 1 primary/secondary categories
   - Stage 2 taxonomy classifications
   - Optional representative document snippets

2. **LLM-based classification**: Uses Mistral-Nemo via OpenRouter for zero-shot classification with structured prompts that include interpretation hints linking taxonomy groups to Radway functions.

3. **Output structure**: Each topic gets a `radway_functions` object with:
   - `radway_main_id`: Primary Radway function (R1-R13 or "none")
   - `radway_secondary_id`: Optional secondary function
   - `radway_other_plausible_ids`: List of other plausible functions
   - `radway_phase`: Phase (I, II, III, or NA)
   - `radway_is_none`: Boolean flag
   - `radway_confidence`: Confidence level (low/medium/high)
   - `radway_rationale`: Explanation for the classification
   - `radway_main_name`: Human-readable function name
   - `radway_phase_name`: Human-readable phase name

## Usage

### Main Script: `zeroshot_radway_openrouter.py`

Classify all topics to Radway functions:

```bash
python src/stage09_category_mapping/stage3_radway_functions/scripts/zeroshot_radway_openrouter.py \
    --taxonomy-json path/to/taxonomy_mappings.json \
    --output-json path/to/taxonomy_with_radway.json \
    --model-name mistralai/Mistral-Nemo-Instruct-2407 \
    --temperature 0.25 \
    --max-tokens 220 \
    --limit-topics 10  # Optional: for testing
```

**Key Arguments**:
- `--taxonomy-json`: Path to Stage 2 taxonomy mappings JSON (required)
- `--output-json`: Path to save merged JSON (required)
- `--model-name`: OpenRouter model (default: `mistralai/Mistral-Nemo-Instruct-2407`)
- `--api-key`: OpenRouter API key (optional, uses env var if not provided)
- `--temperature`: Sampling temperature (default: 0.25)
- `--max-tokens`: Max tokens for JSON output (default: 220)
- `--limit-topics`: Limit to first N topics (for testing)
- `--no-snippets`: Skip loading BERTopic model for snippets
- `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

### Model Update Script: `update_model_with_radway.py`

Attach Radway mappings to BERTopic model:

```bash
python src/stage09_category_mapping/stage3_radway_functions/scripts/update_model_with_radway.py \
    --merged-json path/to/taxonomy_with_radway.json \
    --source-model-suffix _with_taxonomy_mappings \
    --target-model-suffix _with_radway_mappings
```

## Output Format

The merged JSON preserves all original taxonomy fields and adds Radway mappings:

```json
{
  "33": {
    "topic_id": 33,
    "main_category_id": "4.2",
    "main_category_name": "...",
    "main_category_group": "...",
    "secondary_category_id": "5.1",
    ... (all existing taxonomy fields) ...,
    "source_metadata": {
      "label": "...",
      "keywords": [...],
      ...
    },
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

## Files

- `scripts/zeroshot_radway_openrouter.py`: Main classification module with CLI
- `scripts/update_model_with_radway.py`: Model attachment script

## Dependencies

- Reuses code from Stage 2 (`zeroshot_taxonomy_openrouter.py`) for taxonomy lookups
- Reuses OpenRouter helpers from Stage 8
- Reuses model loading from Stage 6
- Uses Mistral-Nemo via OpenRouter for zero-shot classification

## Key Research Questions

- Do good books follow Radway's structure more closely?
- How does narrative arc differ by quality?
- Which narrative phases are most associated with high ratings?

## Next Steps

1. Run classification on full topic set
2. Analyze distribution of Radway functions across topics
3. Compare narrative patterns between bad/mid/good rated books
4. Visualize function prevalence by narrative phase
5. Statistical analysis of narrative structure differences

