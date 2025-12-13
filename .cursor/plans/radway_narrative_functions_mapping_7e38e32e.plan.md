---
name: Radway Narrative Functions Mapping
overview: Implement Stage 3 Radway narrative functions mapping that uses the taxonomy JSON as single source of truth, classifies topics to Radway's 13 functions using Mistral via OpenRouter, and merges results back into the taxonomy JSON structure.
todos:
  - id: radway-functions-def
    content: Define RADWAY_FUNCTIONS list with 13 functions (R1-R13) + none, create RADWAY_BY_ID dict and RADWAY_TEXT_BLOCK for prompts
    status: pending
  - id: load-taxonomy-json
    content: Implement load_topics_with_taxonomy() to load full taxonomy JSON structure preserving all fields
    status: pending
  - id: radway-prompts
    content: Create RADWAY_ZEROSHOT_SYSTEM_PROMPT and RADWAY_ZEROSHOT_USER_PROMPT templates with interpretation hints
    status: pending
    dependencies:
      - radway-functions-def
  - id: classify-function
    content: Implement classify_topic_to_radway_openrouter() that uses all taxonomy JSON fields to build prompts and classify topics
    status: pending
    dependencies:
      - radway-prompts
      - load-taxonomy-json
  - id: batch-mapping
    content: Implement map_all_topics_to_radway() to process all topics and merge Radway results into taxonomy JSON under radway_functions key
    status: pending
    dependencies:
      - classify-function
  - id: cli-interface
    content: Add command-line interface with argparse for running the script standalone
    status: pending
    dependencies:
      - batch-mapping
  - id: model-attachment
    content: Implement update_model_with_radway_mappings() to optionally attach Radway mappings to BERTopic model (similar to Stage 2)
    status: pending
    dependencies:
      - batch-mapping
---

# Stage 3: Radway Narrative Functions Mapping

## Overview

Implement zero-shot classification of BERTopic topics to Radway's 13 narrative functions, using the existing taxonomy JSON as the single source of truth. The module will reuse Stage 2 patterns and merge Radway results into the taxonomy JSON structure.

## Architecture

```
Taxonomy JSON (with source_metadata)
    ↓
Load full topic objects (all fields)
    ↓
Extract: taxonomy fields + source_metadata + optional snippets
    ↓
Zero-shot Radway classification (Mistral-Nemo via OpenRouter)
    ↓
Merge Radway results into taxonomy JSON under "radway_functions" key
    ↓
Optional: Attach to BERTopic model
```

## Implementation Steps

### 1. Create Main Module: `zeroshot_radway_openrouter.py`

**Location**: `src/stage09_category_mapping/stage3_radway_functions/scripts/zeroshot_radway_openrouter.py`

**Key Components**:

#### 1.1 Radway Functions Definition

- Define `RADWAY_FUNCTIONS` list with 13 functions (R1-R13) + "none"
- Each function includes: `id`, `name`, `phase`, `phase_name`, `description`
- Create `RADWAY_BY_ID` dict for quick lookup
- Generate `RADWAY_TEXT_BLOCK` for prompts (similar to `TAXONOMY_TEXT_BLOCK`)

#### 1.2 Load Taxonomy JSON as Source of Truth

- Function: `load_topics_with_taxonomy(taxonomy_json_path: Path) -> Dict[int, Dict[str, Any]]`
- Loads full taxonomy JSON structure (preserves all fields)
- Extracts from each topic:
  - Taxonomy fields: `main_category_id`, `main_category_name`, `main_category_group`, etc.
  - Source metadata: `source_metadata.label`, `source_metadata.keywords`, `source_metadata.scene_summary`, etc.
  - Returns dict with int keys (topic_id → full topic object)

#### 1.3 Radway Classification Function

- Function: `classify_topic_to_radway_openrouter(...)`
- Parameters:
  - `topic_id`, `topic_entry` (full taxonomy JSON entry)
  - `client`, `model_name`, `temperature`, `max_new_tokens`
  - Optional: `representative_docs`, `max_snippets`, `max_chars_per_snippet`
- Builds user prompt using ALL fields from `topic_entry`:
  - Taxonomy: main/secondary category IDs, names, groups
  - Source metadata: label, keywords, scene_summary, primary/secondary categories
  - Optional snippets
- Returns dict with Radway mapping fields:
  - `radway_main_id`, `radway_secondary_id`, `radway_other_plausible_ids`
  - `radway_phase`, `radway_is_none`, `radway_confidence`, `radway_rationale`
  - `radway_main_name`, `radway_phase_name` (enriched from `RADWAY_BY_ID`)

#### 1.4 Batch Mapping Function

- Function: `map_all_topics_to_radway(...)`
- Parameters:
  - `taxonomy_json_path`: Input taxonomy JSON (single source of truth)
  - `output_path`: Output JSON path (merged taxonomy + Radway)
  - Optional: `client`, `model_name`, `api_key`, `temperature`, `max_new_tokens`
  - Optional: `topic_to_snippets`, `max_snippets`, `max_chars_per_snippet`
  - Optional: `limit_topics` (for testing)
- Process:

  1. Load taxonomy JSON via `load_topics_with_taxonomy()`
  2. For each topic, call `classify_topic_to_radway_openrouter()`
  3. Merge Radway results into topic entry under `"radway_functions"` key
  4. Save merged JSON (preserves all original fields + adds Radway)

#### 1.5 Model Attachment Function (Optional)

- Function: `update_model_with_radway_mappings(...)`
- Similar to `update_model_with_taxonomy_mappings()` from Stage 2
- Loads BERTopic model with taxonomy mappings
- Attaches Radway mappings to model (e.g., `topic_model.topic_radway_` or merge into `topic_metadata_`)
- Saves updated model to new location

### 2. System and User Prompts

#### 2.1 System Prompt (`RADWAY_ZEROSHOT_SYSTEM_PROMPT`)

- Defines Radway's 13 functions with descriptions
- Explains phases: I (R1-R7), II (R8-R10), III (R11-R13)
- Provides interpretation hints linking taxonomy groups to Radway functions
- JSON schema specification (same style as taxonomy prompts)
- Field rules and constraints

#### 2.2 User Prompt (`RADWAY_ZEROSHOT_USER_PROMPT`)

- Template that uses ALL fields from taxonomy JSON:
  - Topic ID
  - Keywords (from `source_metadata.keywords`)
  - Label and scene summary (from `source_metadata`)
  - Primary/secondary categories (from `source_metadata`)
  - Taxonomy main/secondary IDs, names, groups
  - Optional representative snippets
- Format helper: `build_radway_user_prompt(topic_id, topic_entry, ...)`

### 3. Helper Functions

#### 3.1 Snippet Loading (Reuse from Stage 2)

- Reuse `load_bertopic_model_for_snippets()` pattern from `zeroshot_taxonomy_openrouter.py`
- Or accept `topic_to_snippets` dict as parameter

#### 3.2 JSON Merging

- Function: `merge_radway_into_taxonomy_json(taxonomy_json_path, radway_results, output_path)`
- Loads taxonomy JSON
- For each topic in `radway_results`, adds `"radway_functions"` key
- Preserves all existing fields
- Saves merged JSON

### 4. Command-Line Interface

**Script**: `zeroshot_radway_openrouter.py` (with `if __name__ == "__main__"`)

**Arguments**:

- `--taxonomy-json`: Path to taxonomy JSON (required)
- `--output-json`: Path to save merged JSON (required)
- `--model-name`: OpenRouter model (default: `mistralai/Mistral-Nemo-Instruct-2407`)
- `--api-key`: OpenRouter API key (optional, uses env var if not provided)
- `--temperature`: Sampling temperature (default: 0.25)
- `--max-tokens`: Max tokens for JSON output (default: 220)
- `--limit-topics`: Limit to first N topics (for testing)
- `--no-snippets`: Skip loading BERTopic model for snippets
- `--log-level`: Logging verbosity

### 5. Integration Points

#### 5.1 Reuse Existing Code

- Import from `zeroshot_taxonomy_openrouter.py`:
  - `TAXONOMY_BY_ID` (for taxonomy name/group lookups in prompts)
  - OpenRouter helpers: `load_openrouter_client`, `rerank_snippets_centrality`, `format_snippets`
- Import from Stage 8: `extract_representative_docs_per_topic` (if needed)
- Import model loading: `load_native_bertopic_model` from Stage 6

#### 5.2 Output Structure

**Merged JSON format**:

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

### 6. Optional: Model Update Script

**Script**: `update_model_with_radway.py` (similar to `update_model_with_taxonomy.py`)

- Loads BERTopic model with taxonomy mappings
- Loads merged taxonomy+Radway JSON
- Attaches Radway mappings to model
- Saves updated model

## File Structure

```
src/stage09_category_mapping/stage3_radway_functions/
├── README.md (already exists)
└── scripts/
    ├── zeroshot_radway_openrouter.py (NEW - main module)
    └── update_model_with_radway.py (NEW - optional model attachment)
```

## Dependencies

- Reuse existing imports from Stage 2 and Stage 8
- No new external dependencies required
- Follows same patterns as `zeroshot_taxonomy_openrouter.py`

## Testing Strategy

1. Test with `--limit-topics 5` on small subset
2. Verify merged JSON preserves all original fields
3. Verify Radway fields are correctly structured
4. Test with/without snippets
5. Test model attachment (if implemented)

## Key Design Decisions

1. **Single Source of Truth**: Taxonomy JSON contains all needed information (including `source_metadata`), so no need to load separate Stage 8 labels JSON
2. **Merge Strategy**: Add `radway_functions` as nested key to preserve all existing data
3. **Reuse Patterns**: Follow exact same structure as Stage 2 taxonomy mapping for consistency
4. **Flexible Input**: Support taxonomy JSON with or without `source_metadata` (graceful fallback)