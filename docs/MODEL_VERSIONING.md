# BERTopic Model Versioning Documentation

This document tracks all versions of the BERTopic models used in the project, their relationships, and how to load them.

## Base Model Information

- **Embedding Model**: `paraphrase-MiniLM-L6-v2`
- **Pareto Rank**: 1
- **Base Directory**: `models/retrained/paraphrase-MiniLM-L6-v2/`
- **Original Topic Count**: 368 topics (from metadata, though actual count may differ)

## Model Folder Structure

Models are organized by stage in subfolders under `models/retrained/<embedding_model>/`:

```
models/retrained/paraphrase-MiniLM-L6-v2/
├── model_1/                    # Base retrained model (Stage 05)
├── model_1.pkl                 # Base retrained model (pickle)
├── stage07_topic_quality/      # Stage 07 outputs
│   ├── model_1_with_noise_labels/     # BERTopic format
│   └── model_1_with_noise_labels.pkl  # Pickle format
├── stage08_llm_labeling/       # Stage 08 outputs
│   ├── model_1_with_llm_labels/        # BERTopic format
│   └── model_1_with_llm_labels.pkl     # Pickle format
└── stage09_category_mapping/   # Stage 09 outputs (if models are saved)
    ├── model_1_with_categories/       # BERTopic format
    └── model_1_with_categories.pkl    # Pickle format
```

## Model Versions

### 1. model_1 (Base Retrained Model)

**Description**: Base retrained BERTopic model from Stage 05, without any additional labels.

**File Paths**:
- **Pickle wrapper**: `models/retrained/paraphrase-MiniLM-L6-v2/model_1.pkl` (4.7 GB)
- **Native BERTopic format**: `models/retrained/paraphrase-MiniLM-L6-v2/model_1/` (directory)
- **Metadata**: `models/retrained/paraphrase-MiniLM-L6-v2/model_1_metadata.json`

**Created**: November 29, 2025, 18:00:19

**Contents**:
- Full `RetrainableBERTopicModel` wrapper (pickle)
- Native BERTopic safetensors format (directory)
- Training hyperparameters and metadata

**How to Load**:
```python
# Load pickle wrapper
from src.stage06_exploration.explore_retrained_model import load_retrained_wrapper
wrapper, topic_model = load_retrained_wrapper(
    base_dir="models/retrained",
    embedding_model="paraphrase-MiniLM-L6-v2",
    pareto_rank=1,
    model_suffix=""  # No suffix for base model
)

# Or load native format
from src.stage06_exploration.explore_retrained_model import load_native_bertopic_model
topic_model = load_native_bertopic_model(
    base_dir="models/retrained",
    embedding_model="paraphrase-MiniLM-L6-v2",
    pareto_rank=1,
    model_suffix=""
)
```

**Metadata** (from `model_1_metadata.json`):
- `num_topics`: 9936 (initial, before reduction)
- `coherence`: 0.425
- `topic_diversity`: 0.94
- `combined_score`: 1.65

---

### 2. model_1_with_noise_labels (Stage 07)

**Description**: Base model extended with noise candidate labels. Noisy topics were flagged and labeled during topic quality analysis (Stage 07).

**File Paths**:
- **Pickle wrapper**: `models/retrained/paraphrase-MiniLM-L6-v2/stage07_topic_quality/model_1_with_noise_labels.pkl`
- **Native BERTopic format**: `models/retrained/paraphrase-MiniLM-L6-v2/stage07_topic_quality/model_1_with_noise_labels/` (directory)

**Created**: Stage 07 (Topic Quality Analysis)

**Relationship**: Extends `model_1` by adding noise labels to the `custom_labels_` attribute.

**Contents**:
- All contents from `model_1`
- Additional `custom_labels_` attribute with noise topic labels (prefixed with `[NOISE_CANDIDATE:...]`)
- Labels stored in BERTopic's native `custom_labels_` field

**How Labels Were Applied**:
1. Topic quality analysis identified noisy topics (low POS coherence, small size, few POS words)
2. Labels applied via `topic_model.set_topic_labels(labels_dict)`
3. Model saved in both pickle and native formats to `stage07_topic_quality/` subfolder

**How to Load**:
```python
# Load with noise labels using stage subfolder
from src.stage06_topic_exploration.explore_retrained_model import load_retrained_wrapper
wrapper, topic_model = load_retrained_wrapper(
    base_dir="models/retrained",
    embedding_model="paraphrase-MiniLM-L6-v2",
    pareto_rank=1,
    model_suffix="_with_noise_labels",
    stage_subfolder="stage07_topic_quality"
)

# Or using the labeling module
from src.stage08_llm_labeling.generate_labels import load_bertopic_model
wrapper, topic_model = load_bertopic_model(
    base_dir="models/retrained",
    embedding_model="paraphrase-MiniLM-L6-v2",
    pareto_rank=1,
    model_suffix="_with_noise_labels",
    stage_subfolder="stage07_topic_quality"
)
```

**Notes**:
- Noise labels are temporary markers for inspection
- LLM labels in Stage 08 will overwrite noise tags (intended behavior)
- Noise labels use prefix pattern `[NOISE_CANDIDATE:...]` for easy identification

---

### 3. model_1_with_llm_labels (Stage 08)

**Description**: Model with LLM-generated topic labels integrated. Labels are generated from POS representation using OpenRouter API with romance-aware prompts that produce structured JSON output.

**File Paths**:
- **Pickle wrapper**: `models/retrained/paraphrase-MiniLM-L6-v2/stage08_llm_labeling/model_1_with_llm_labels.pkl`
- **Native BERTopic format**: `models/retrained/paraphrase-MiniLM-L6-v2/stage08_llm_labeling/model_1_with_llm_labels/` (directory)

**Created**: Stage 08 (LLM Labeling)

**Relationship**: Extends `model_1_with_noise_labels` by replacing noise tags with LLM-generated labels.

**Contents**:
- All contents from `model_1_with_noise_labels`
- LLM-generated labels replace noise tags (via `integrate_labels_to_bertopic()`)
- Labels stored in BERTopic's `custom_labels_` field
- Full JSON metadata (scene_summary, categories, is_noise, rationale) stored in labels JSON file

**Label Format**:
When using `--use-improved-prompts`, labels are generated as structured JSON with:
- `label`: Short noun phrase (2-6 words)
- `scene_summary`: One complete sentence (12-25 words) describing typical scene
- `primary_categories`: 1-3 high-level tags (e.g., "romance_core", "sexual_content", "work_life")
- `secondary_categories`: 0-5 specific tags with dimension:value format (e.g., "setting:car", "activity:kissing")
- `is_noise`: Boolean indicating if topic is technical artifact
- `rationale`: 1-3 sentences explaining label choice

**How Labels Were Applied**:
1. Load model from `stage07_topic_quality/` subfolder
2. LLM labels generated via OpenRouter API from POS representation with romance-aware JSON prompts
3. Labels integrated via `integrate_labels_to_bertopic()` which merges with existing labels
4. LLM labels overwrite noise tags (intended behavior)
5. Model saved in both native BERTopic and pickle formats to `stage08_llm_labeling/` subfolder
6. Full JSON metadata saved to labels JSON file in `results/stage08_llm_labeling/`

**How to Load**:
```python
# Load with LLM labels using stage subfolder
from src.stage06_topic_exploration.explore_retrained_model import load_native_bertopic_model
topic_model = load_native_bertopic_model(
    base_dir="models/retrained",
    embedding_model="paraphrase-MiniLM-L6-v2",
    pareto_rank=1,
    model_suffix="_with_llm_labels",
    stage_subfolder="stage08_llm_labeling"
)
```

**Labels JSON File**:
- Location: `results/stage08_llm_labeling/labels_pos_openrouter_<model_name>_romance_aware_<embedding_model>.json`
- Structure (with `--use-improved-prompts`): `{topic_id: {label, keywords, scene_summary, primary_categories, secondary_categories, is_noise, rationale}}`
- Structure (without flag): `{topic_id: {label, keywords, scene_summary}}`
- The `label` field is integrated into the model's `custom_labels_` attribute
- Other fields (scene_summary, categories, etc.) are available in the JSON file for analysis

**Notes**:
- This is the default model for Stage 09 category mapping
- LLM labels replace noise tags from Stage 07
- Labels are romance-aware and generated from POS representation
- Model is saved in both native BERTopic (directory) and pickle wrapper formats
- Full JSON metadata is preserved in the labels JSON file even though only labels are integrated into the model

---

### 4. model_1_with_categories (Stage 09, if saved)

**Description**: Model with category mappings integrated. Categories are mapped from topic labels to research categories.

**File Paths** (if saved):
- **Pickle wrapper**: `models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_categories.pkl`
- **Native BERTopic format**: `models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_categories/` (directory)

**Created**: Stage 09 (Category Mapping, if models are saved)

**Relationship**: Extends `model_1_with_llm_labels` by adding category mappings.

**Contents**:
- All contents from `model_1_with_llm_labels`
- Category mappings integrated into model metadata
- Category probabilities stored in model

**How to Load**:
```python
# Load with categories using stage subfolder
from src.stage06_topic_exploration.explore_retrained_model import load_native_bertopic_model
topic_model = load_native_bertopic_model(
    base_dir="models/retrained",
    embedding_model="paraphrase-MiniLM-L6-v2",
    pareto_rank=1,
    model_suffix="_with_categories",
    stage_subfolder="stage09_category_mapping"
)
```

---

## Model Version Relationships

```
model_1 (base, Stage 05)
    │
    └── model_1_with_noise_labels (Stage 07: adds noise topic labels)
        │
        └── model_1_with_llm_labels (Stage 08: replaces noise tags with LLM labels)
            │
            └── model_1_with_categories (Stage 09: adds category mappings, if saved)
```

**Model Progression**:
1. **Stage 05**: Base retrained model (`model_1`)
2. **Stage 07**: Add noise labels (`model_1_with_noise_labels` in `stage07_topic_quality/`)
3. **Stage 08**: Replace noise tags with LLM labels (`model_1_with_llm_labels` in `stage08_llm_labeling/`)
4. **Stage 09**: Add category mappings (`model_1_with_categories` in `stage09_category_mapping/`, if saved)

---

## Loading Patterns

### Pattern 1: Using Helper Functions (Recommended)

```python
from src.stage06_topic_exploration.explore_retrained_model import (
    load_retrained_wrapper,
    load_native_bertopic_model
)

# For pickle wrapper from Stage 07
wrapper, topic_model = load_retrained_wrapper(
    model_suffix="_with_noise_labels",
    stage_subfolder="stage07_topic_quality"
)

# For native format from Stage 08
topic_model = load_native_bertopic_model(
    model_suffix="_with_llm_labels",
    stage_subfolder="stage08_llm_labeling"
)
```

### Pattern 2: Direct BERTopic Loading

```python
from bertopic import BERTopic

# Load native format directory from Stage 08
topic_model = BERTopic.load("models/retrained/paraphrase-MiniLM-L6-v2/stage08_llm_labeling/model_1_with_llm_labels")
```

### Pattern 3: Using Labeling Module

```python
from src.stage08_llm_labeling.generate_labels import load_bertopic_model

# Load from Stage 07
wrapper, topic_model = load_bertopic_model(
    model_suffix="_with_noise_labels",
    stage_subfolder="stage07_topic_quality",
    use_native=False  # True for native format
)

# Load from Stage 08
wrapper, topic_model = load_bertopic_model(
    model_suffix="_with_llm_labels",
    stage_subfolder="stage08_llm_labeling",
    use_native=False
)
```

---

## Model Format Details

### Pickle Format (`.pkl`)
- Contains full `RetrainableBERTopicModel` wrapper
- Includes embeddings, cache, and all wrapper state
- Large file size (~4.7 GB)
- Can have GPU array serialization issues

### Native BERTopic Format (directory)
- Safetensors format (recommended by BERTopic)
- Smaller file size
- Avoids GPU array issues
- Contains:
  - `config.json`: Model configuration
  - `ctfidf_config.json`: CTFIDF vectorizer config
  - `ctfidf.safetensors`: CTFIDF weights
  - `topic_embeddings.safetensors`: Topic embeddings
  - `topics.json`: Topic representations and labels

### Custom Labels Storage
- Stored in `topic_model.custom_labels_` attribute (dict: `{topic_id: label}`)
- Persisted in native format's `topics.json`
- Can be accessed via `topic_model.get_topic_info()`

---

## Logs and Creation History

### model_1
- Created: Stage 05 retraining (`stage05_retraining`)
- Log: `logs/stage05_retraining_*.log`
- Saved via `retrain_models.py:1256-1288`

### model_1_with_noise_labels
- Created: Stage 07 (Topic Quality Analysis)
- Script: `src/stage07_topic_quality/main.py`
- Notebook: `notebooks/07_topic_quality/topic_quality_eda.ipynb`
- Saved to: `models/retrained/<embedding_model>/stage07_topic_quality/`

### model_1_with_llm_labels
- Created: Stage 08 (LLM Labeling)
- Script: `src/stage08_llm_labeling/openrouter_experiments/core/main_openrouter.py`
- Saved to: `models/retrained/<embedding_model>/stage08_llm_labeling/`
- Loads from: `stage07_topic_quality/` subfolder
- Output format: JSON with label, scene_summary, categories, is_noise, rationale (when using `--use-improved-prompts`)
- Saved formats: Both native BERTopic (directory) and pickle wrapper formats

### model_1_with_categories
- Created: Stage 09 (Category Mapping, if models are saved)
- Saved to: `models/retrained/<embedding_model>/stage09_category_mapping/` (if saved)
- Loads from: `stage08_llm_labeling/` subfolder

---

## Recommendations for Stage 09

For Stage 09 category mapping analysis, use:
- **Model**: `model_1_with_llm_labels` from `stage08_llm_labeling/` (native format)
- **Reason**: Contains LLM-generated labels needed for category mapping
- **Path**: `models/retrained/paraphrase-MiniLM-L6-v2/stage08_llm_labeling/model_1_with_llm_labels/`

If categories are already integrated, use `model_1_with_categories` from `stage09_category_mapping/`.

---

## Troubleshooting

### Model Not Found
- Check file paths match exactly (case-sensitive)
- Verify model suffix matches file/directory name
- Ensure you're in the correct working directory

### Loading Errors
- For pickle files: Ensure Python environment matches training environment
- For native format: Ensure BERTopic version is compatible
- Check logs for specific error messages

### Missing Labels
- Verify `custom_labels_` attribute exists: `hasattr(topic_model, 'custom_labels_')`
- Check if labels were saved: `topic_model.custom_labels_` should be a dict
- Review creation logs to confirm labels were applied

---

## Future Model Versions

When creating new model versions:
1. Use descriptive suffixes (e.g., `_with_custom_labels`, `_reduced_topics`)
2. Save in both pickle and native formats
3. Save to appropriate stage subfolder (e.g., `stage07_topic_quality/`, `stage08_llm_labeling/`)
4. Use `stage_subfolder` parameter when loading models
5. Document creation date and source model
6. Update this document with new version information
7. Create backup before overwriting existing models

## Stage Subfolder Usage

All helper functions now support `stage_subfolder` parameter:
- `load_retrained_wrapper(stage_subfolder="stage07_topic_quality")`
- `load_native_bertopic_model(stage_subfolder="stage08_llm_labeling")`
- `load_bertopic_model(stage_subfolder="stage08_llm_labeling")`

This allows loading models from specific stage subfolders while maintaining backward compatibility with base directory models.

