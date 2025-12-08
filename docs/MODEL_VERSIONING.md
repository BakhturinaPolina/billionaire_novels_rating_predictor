# BERTopic Model Versioning Documentation

This document tracks all versions of the BERTopic models used in the project, their relationships, and how to load them.

## Base Model Information

- **Embedding Model**: `paraphrase-MiniLM-L6-v2`
- **Pareto Rank**: 1
- **Base Directory**: `models/retrained/paraphrase-MiniLM-L6-v2/`
- **Original Topic Count**: 368 topics (from metadata, though actual count may differ)

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

### 2. model_1_with_noise_labels

**Description**: Base model extended with noise candidate labels. 20 noisy topics were flagged and labeled during topic quality analysis (Stage 06 EDA).

**File Paths**:
- **Pickle wrapper**: `models/retrained/paraphrase-MiniLM-L6-v2/model_1_with_noise_labels.pkl` (4.7 GB)
- **Native BERTopic format**: `models/retrained/paraphrase-MiniLM-L6-v2/model_1_with_noise_labels` (3.6 GB data file - appears to be native format saved as file)
- **Backup files**: Multiple timestamped backups exist:
  - `model_1_with_noise_labels_backup_20251204_172303.pkl`
  - `model_1_with_noise_labels_backup_20251205_162958.pkl`
  - `model_1_with_noise_labels_backup_20251205_163032.pkl`

**Created**: December 5, 2025, 16:31:02

**Relationship**: Extends `model_1` by adding noise labels to the `custom_labels_` attribute.

**Contents**:
- All contents from `model_1`
- Additional `custom_labels_` attribute with noise topic labels
- Labels stored in BERTopic's native `custom_labels_` field

**How Labels Were Applied**:
1. Topic quality analysis identified noisy topics (low POS coherence, small size)
2. Labels applied via `topic_model.set_topic_labels(labels_dict)`
3. Model saved in both pickle and native formats

**How to Load**:
```python
# Load with noise labels (default for openrouter experiments)
from src.stage06_exploration.explore_retrained_model import load_retrained_wrapper
wrapper, topic_model = load_retrained_wrapper(
    base_dir="models/retrained",
    embedding_model="paraphrase-MiniLM-L6-v2",
    pareto_rank=1,
    model_suffix="_with_noise_labels"  # Use suffix
)

# Or using the labeling module
from src.stage06_labeling.generate_labels import load_bertopic_model
wrapper, topic_model = load_bertopic_model(
    base_dir="models/retrained",
    embedding_model="paraphrase-MiniLM-L6-v2",
    pareto_rank=1,
    model_suffix="_with_noise_labels"
)
```

**Notes**:
- This is the **default model** used by `openrouter_experiments` (see `main_openrouter.py:109`)
- Noise labels help identify low-quality topics that should be filtered or handled specially

---

### 3. model_1_with_categories

**Description**: Model with LLM-generated category labels integrated. Categories are mapped from topic labels to research categories (e.g., luxury/wealth, emotional introspection, erotic content).

**File Paths**:
- **Native BERTopic format**: `models/retrained/paraphrase-MiniLM-L6-v2/model_1_with_categories/` (directory)
- **Note**: No pickle wrapper found - only native format exists

**Created**: December 1, 2025, 17:41:59

**Relationship**: Extends `model_1` (and potentially `model_1_with_noise_labels`) by adding category mappings.

**Contents**:
- Native BERTopic safetensors format
- Category mappings integrated into model metadata
- LLM-generated labels from OpenRouter experiments
- Category probabilities stored in model

**How Categories Were Integrated**:
1. LLM labels generated via OpenRouter API (see `results/stage06_labeling_openrouter/`)
2. Labels mapped to categories via `main_category_mapping.py`
3. Categories integrated via `integrate_categories_to_bertopic.py` (referenced in `RUN_FULL_PIPELINE.sh`)
4. Model saved in native BERTopic format

**How to Load**:
```python
# Load native format with categories
from src.stage06_exploration.explore_retrained_model import load_native_bertopic_model
topic_model = load_native_bertopic_model(
    base_dir="models/retrained",
    embedding_model="paraphrase-MiniLM-L6-v2",
    pareto_rank=1,
    model_suffix="_with_categories"
)

# Or directly with BERTopic
from bertopic import BERTopic
topic_model = BERTopic.load("models/retrained/paraphrase-MiniLM-L6-v2/model_1_with_categories")
```

**Related Files**:
- Labels JSON: `results/stage06_labeling_openrouter/labels_pos_openrouter_romance_aware_paraphrase-MiniLM-L6-v2.json`
- Category mappings: `results/stage06_labeling/category_mapping/topic_to_category_probs.json`
- Validation: `results/stage06_labeling/category_mapping/validation_report_final.md`

**Notes**:
- This model is intended for Stage 1 natural clusters analysis
- Contains 361 topics with category mappings
- Used for statistical analysis of topic distributions by rating class

---

## Model Version Relationships

```
model_1 (base)
    │
    ├── model_1_with_noise_labels (adds noise topic labels)
    │
    └── model_1_with_categories (adds LLM category labels)
```

**Note**: It's unclear if `model_1_with_categories` was built from `model_1` or `model_1_with_noise_labels`. The timestamps suggest it may have been created from `model_1` directly (Dec 1 vs Dec 5 for noise labels).

---

## Loading Patterns

### Pattern 1: Using Helper Functions (Recommended)

```python
from src.stage06_exploration.explore_retrained_model import (
    load_retrained_wrapper,
    load_native_bertopic_model
)

# For pickle wrapper
wrapper, topic_model = load_retrained_wrapper(
    model_suffix="_with_noise_labels"
)

# For native format
topic_model = load_native_bertopic_model(
    model_suffix="_with_categories"
)
```

### Pattern 2: Direct BERTopic Loading

```python
from bertopic import BERTopic

# Load native format directory
topic_model = BERTopic.load("models/retrained/paraphrase-MiniLM-L6-v2/model_1_with_categories")
```

### Pattern 3: Using Labeling Module

```python
from src.stage06_labeling.generate_labels import load_bertopic_model

wrapper, topic_model = load_bertopic_model(
    model_suffix="_with_noise_labels",
    use_native=False  # True for native format
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
- Created: Stage 06 EDA topic quality analysis
- Logs: `logs/stage06_labeling_*.log`, `notebooks/06_labeling/topic_quality_eda.ipynb`
- Saved via `topic_quality_analysis.py:298-304` and notebook cells
- Multiple saves/backups indicate iterative refinement

### model_1_with_categories
- Created: Stage 06 labeling category mapping pipeline
- Script: `results/stage06_labeling/category_mapping/RUN_FULL_PIPELINE.sh`
- Integration: `src/stage06_labeling/category_mapping/integrate_categories_to_bertopic.py` (referenced but file not found in current codebase)

---

## Recommendations for Stage 1

For Stage 1 natural clusters analysis, use:
- **Model**: `model_1_with_categories` (native format)
- **Reason**: Contains category labels needed for analysis
- **Path**: `models/retrained/paraphrase-MiniLM-L6-v2/model_1_with_categories/`

If categories are not needed, `model_1_with_noise_labels` can be used as a fallback.

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
3. Document creation date and source model
4. Update this document with new version information
5. Create backup before overwriting existing models

