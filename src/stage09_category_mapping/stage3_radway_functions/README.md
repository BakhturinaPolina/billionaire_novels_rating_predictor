# Stage 3: Radway Narrative Functions

## Overview

**Goal**: Map BERTopic topics to Radway's 13 narrative functions to analyze story structure and track narrative progression across books. Compare narrative patterns between bad/mid/good rated books.

**Status**: ✅ **Implemented & Improved** (can run independently or after Stages 1-2)

**Latest Update**: Classification completed for all 361 topics with improved accuracy through:
- Deterministic decoding (temperature=0.0) for consistent results
- Heuristic override system to fix common systematic errors
- Enhanced prompt disambiguation rules
- Improved handling of explicit sex scenes (2.3 → R12), commitment topics (→ R11/R13), and R7 separation definition

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

The implementation uses **taxonomy + source metadata as the single source of truth**, which can be loaded from:
1. **JSON file** (e.g., `taxonomy_mappings_*.json` from Stage 2)
2. **BERTopic model** (recommended: `model_1_with_llm_labels_and_metadata_disambiguated.pkl`)

Models can store taxonomy in either:
- `topic_taxonomy_` attribute (e.g., `model_1_with_taxonomy_mappings`) - taxonomy stored separately
- `topic_metadata_` attribute (e.g., `model_1_with_llm_labels_and_metadata_disambiguated.pkl`) - taxonomy merged with source metadata

The code automatically detects and merges both sources. Recommended models:
- Taxonomy mappings (main_category_id, secondary_category_id, etc.)
- Source metadata (label, keywords, scene_summary, primary/secondary categories)

The Radway function mappings are merged back into the taxonomy data under a `"radway_functions"` key, preserving all existing fields.

### Zero-Shot Classification Approach

1. **Topic-level classification**: Maps each BERTopic topic to Radway functions using:
   - Topic keywords from BERTopic
   - LLM-generated labels and scene summaries (from Stage 8)
   - Stage 1 primary/secondary categories
   - Stage 2 taxonomy classifications
   - Optional representative document snippets

2. **LLM-based classification**: Uses Mistral-Nemo via OpenRouter for zero-shot classification with structured prompts that include:
   - Interpretation hints linking taxonomy groups to Radway functions
   - Disambiguation rules for common confusions (R4 vs R12, R7 narrowing, commitment overrides)
   - Micro-examples for key distinctions
   - Gated "none" decision process

3. **Post-LLM heuristic overrides**: Conservative rule-based corrections for systematic errors:
   - Explicit sex scenes (taxonomy_main_id = 2.3) → R12 (not R4)
   - Wedding/marriage/commitment cues → R11/R13 (not none/R8)
   - R7 only when actual breakup/separation cues exist
   - R4 sanity checks for non-sexual contexts

4. **Output structure**: Each topic gets a `radway_functions` object with:
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

Classify all topics to Radway functions. Supports loading taxonomy + source metadata from either:
1. **JSON file** (e.g., `taxonomy_mappings_*.json`)
2. **BERTopic model** (recommended: `model_1_with_llm_labels_and_metadata_disambiguated.pkl`)

#### Using Recommended Model (Recommended)

Load directly from the recommended model with embedded taxonomy metadata:

```bash
python src/stage09_category_mapping/stage3_radway_functions/scripts/zeroshot_radway_openrouter.py \
    --taxonomy-json models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_taxonomy_mappings \
    --output-json path/to/taxonomy_with_radway.json \
    --model-name mistralai/Mistral-Nemo-Instruct-2407 \
    --api-key YOUR_API_KEY \
    --max-tokens 220 \
    --limit-topics 10  # Optional: for testing
```

#### Using JSON File (Alternative)

If you have a taxonomy mappings JSON file:

```bash
python src/stage09_category_mapping/stage3_radway_functions/scripts/zeroshot_radway_openrouter.py \
    --taxonomy-json path/to/taxonomy_mappings.json \
    --output-json path/to/taxonomy_with_radway.json \
    --model-name mistralai/Mistral-Nemo-Instruct-2407 \
    --api-key YOUR_API_KEY \
    --max-tokens 220 \
    --limit-topics 10  # Optional: for testing
```

**Key Arguments**:
- `--taxonomy-json`: Path to Stage 2 taxonomy mappings JSON file OR BERTopic model with embedded taxonomy metadata (required)
  - If JSON: loads from file (e.g., `taxonomy_mappings_*.json`)
  - If model (.pkl or directory): loads from model's `topic_metadata_` attribute (recommended)
  - Recommended models: `model_1_with_taxonomy_mappings` or `model_1_with_llm_labels_and_metadata_disambiguated.pkl`
- `--output-json`: Path to save merged JSON (required)
- `--model-name`: OpenRouter model (default: `mistralai/Mistral-Nemo-Instruct-2407`)
- `--api-key`: OpenRouter API key (required, or set OPENROUTER_API_KEY env var)
- `--temperature`: Sampling temperature (default: 0.0 for deterministic classification)
- `--max-tokens`: Max tokens for JSON output (default: 220)
- `--limit-topics`: Limit to first N topics (for testing)
- `--no-snippets`: Skip loading BERTopic model for snippets
- `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

**Note**: The classification uses deterministic decoding (temperature=0.0) by default for consistent, reproducible results. Heuristic overrides are automatically applied after LLM classification to fix common systematic errors.

### Model Update Script: `update_model_with_radway.py`

Attach Radway mappings to BERTopic model:

```bash
python src/stage09_category_mapping/stage3_radway_functions/scripts/update_model_with_radway.py \
    --merged-json path/to/taxonomy_with_radway.json \
    --source-model-suffix _with_taxonomy_mappings \
    --target-model-suffix _with_radway_mappings
```

**Note**: The default source model is `_with_taxonomy_mappings`. The code supports both:
- `_with_taxonomy_mappings`: Taxonomy in `topic_taxonomy_`, source metadata in `topic_metadata_` (if available)
- `_with_llm_labels_and_metadata_disambiguated`: Taxonomy merged into `topic_metadata_` (see MODEL_COMPARISON_REPORT.md)

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

## Current Status

✅ **Completed**:
- Classification run for all 361 topics
- Model updated with Radway mappings (`model_1_with_radway_mappings`)
- Heuristic override system implemented and tested
- Classification accuracy improved (2.3 → R12, commitment → R11/R13, reduced false negatives)

## Next Steps

1. ✅ ~~Run classification on full topic set~~ (Completed)
2. Analyze distribution of Radway functions across topics
3. Compare narrative patterns between bad/mid/good rated books
4. Visualize function prevalence by narrative phase
5. Statistical analysis of narrative structure differences
6. Export to CSV/Parquet for correlation analysis (see `stage10_correlation_analysis`)

## Classification Accuracy Improvements

The implementation includes several improvements to address systematic classification errors:

1. **Explicit sex scenes (2.3 → R12)**: Topics with `taxonomy_main_id = 2.3` are now correctly mapped to R12 (heroine responds sexually and emotionally) rather than R4 (purely sexual interest).

2. **Commitment topics (→ R11/R13)**: Topics mentioning wedding/marriage/engagement/proposal are correctly mapped to R11 (commitment) or R13 (restored identity) rather than "none" or R8 (tenderness).

3. **R7 narrowing**: R7 (separation) is now only used when actual breakup/separation cues exist, not for arguments or apologies.

4. **"None" false negatives**: Improved detection prevents romance-core topics from being incorrectly marked as "none".

These improvements are achieved through:
- Enhanced prompt disambiguation rules
- Post-LLM heuristic override system with regex-based pattern matching
- Deterministic decoding for consistency
- Gated "none" decision process

