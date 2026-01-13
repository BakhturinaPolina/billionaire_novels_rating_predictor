# Attaching Full Metadata to BERTopic Models

## Overview

BERTopic models store labels in the `custom_labels_` attribute, but the full JSON structure (keywords, categories, scene_summary, rationale, etc.) is only stored in the JSON file. This document explains how to attach the full metadata to the model for easier analysis.

## What is Stored

When using `--attach-metadata`, the following fields from the JSON file are stored in the model's `topic_metadata_` attribute:

- `label`: Topic label (also stored in `custom_labels_`)
- `keywords`: List of topic keywords
- `primary_categories`: List of primary categories (e.g., "romance_core", "sexual_content")
- `secondary_categories`: List of secondary categories (e.g., "setting:bedroom", "activity:kissing")
- `is_noise`: Boolean indicating if topic is noise
- `rationale`: Explanation of why this label was chosen
- `scene_summary`: One-sentence summary of typical scenes in this topic

## Usage

### Attaching Metadata to Existing Model

To attach full metadata to an existing model (e.g., MISTRAL_BACKUP):

```bash
python -m src.stage09_category_mapping.stage1_natural_clusters.load_model_with_labels \
    --model-path models/retrained/paraphrase-MiniLM-L6-v2/stage08_llm_labeling/model_1_with_llm_labels_MISTRAL_BACKUP.pkl \
    --labels-json results/stage08_llm_labeling/labels_pos_openrouter_mistralai_Mistral-Nemo-Instruct-2407_romance_aware_paraphrase-MiniLM-L6-v2.json \
    --attach-metadata \
    --save-model
```

This will:
1. Load the existing model
2. Load the full metadata from the JSON file
3. Attach it to the model as `topic_metadata_` attribute
4. Save the updated model (metadata is preserved in pickle format)

### Accessing Metadata in Code

Once metadata is attached, you can access it like this:

```python
import pickle

# Load model
with open("model.pkl", "rb") as f:
    wrapper = pickle.load(f)
    model = wrapper.trained_topic_model

# Access metadata
if hasattr(model, "topic_metadata_"):
    metadata = model.topic_metadata_
    
    # Get metadata for a specific topic
    topic_0_metadata = metadata[0]
    print(f"Label: {topic_0_metadata['label']}")
    print(f"Keywords: {topic_0_metadata['keywords']}")
    print(f"Categories: {topic_0_metadata['primary_categories']}")
    print(f"Scene summary: {topic_0_metadata['scene_summary']}")
```

## Integration with Labeling Pipeline

The labeling pipeline (`main_openrouter.py`) now automatically stores full metadata when integrating labels. Future models created with this pipeline will have metadata attached by default.

## Notes

- **Pickle format**: Metadata is preserved when saving/loading pickle files
- **Native BERTopic format**: Metadata is NOT automatically saved in native format (directory). It's only preserved in pickle files.
- **Backward compatibility**: Models without metadata still work; the `topic_metadata_` attribute is optional

## Benefits

Having full metadata in the model makes it easier to:
- Access keywords for hierarchical analysis
- Filter topics by categories
- Understand topic context without loading separate JSON files
- Perform analysis that requires both labels and metadata
