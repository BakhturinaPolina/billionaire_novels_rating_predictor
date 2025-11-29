# Stage 06: Topic Labeling & Composite Building

## Overview

Stage 06 provides two main functionalities:
1. **POS Topic Labeling**: Generate human-readable labels for topics using FLAN-T5 from POS representation keywords
2. **Composite Building**: Map topics to 16 thematic composites using a semi-supervised approach combining manual mapping and codebook-guided assignment

## Status

✅ **Fully Implemented**

- ✅ `generate_labels.py`: POS topic labeling with FLAN-T5
- ✅ `main.py`: CLI entry point for topic labeling
- ✅ `composites.py`: Full implementation of composite building

## Functionality

### Thematic Composites (A-P)

1. **A**: Reassurance/Commitment
2. **B**: Mutual Intimacy
3. **C**: Explicit Eroticism
4. **D**: Power/Wealth/Luxury
5. **E**: Coercion/Brutality/Danger
6. **F**: Angst/Negative Affect
7. **G**: Courtship Rituals/Gifts
8. **H**: Domestic Nesting
9. **I**: Humor/Lightness
10. **J**: Social Support/Kin
11. **K**: Professional Intrusion
12. **L**: Vices/Addictions
13. **M**: Health/Recovery/Growth
14. **N**: Separation/Reunion
15. **O**: Aesthetics/Appearance
16. **P**: Tech/Media Presence

### Mapping Process

1. **Manual Mapping** (Markdown)
   - Researcher-defined topic-to-cluster mappings
   - Hierarchical structure with keyword matching

2. **Codebook** (CSV)
   - Structured category definitions
   - Primary categories and subtypes
   - Systematic assignment rules

3. **Automated Assignment**
   - Weighted topic-to-composite mapping
   - Special handling for ambiguous topics
   - Example: "Woman Pleasure and Arousal" → 70% Explicit, 30% Mutual Intimacy

### Composite Building

Topics are mapped to composites with weighted assignments:
- Each topic can map to multiple composites
- Weights sum to 1.0 per topic
- Final composite scores = sum of (topic_prob × weight) for all topics

## POS Topic Labeling with FLAN-T5

### Overview

Automatically generate human-readable labels for topics by:
1. Extracting top keywords from POS (Part-of-Speech) representation from BERTopic model
2. Using a local FLAN-T5 model to generate short noun phrase labels
3. Saving labels to JSON file for inspection
4. **Always integrating labels back into BERTopic model** (unless `--no-integrate` is set)

**Note:** The pipeline always loads from BERTopic model as the primary source. JSON file can be optionally provided for comparison/inspection purposes.

### Prerequisites

Before running the labeling pipeline, you must have:
- A retrained BERTopic model with POS representation generated
- Run `explore_retrained_model.py` with `--save-topics` to generate POS representation

### Usage

```bash
# Basic usage with defaults (loads BERTopic model)
python -m src.stage06_labeling.main

# With JSON comparison (loads from BERTopic + compares with JSON for validation)
python -m src.stage06_labeling.main \
  --topics-json results/stage06/topics_all_representations_paraphrase-MiniLM-L6-v2.json

# With custom parameters
python -m src.stage06_labeling.main \
  --embedding-model paraphrase-MiniLM-L6-v2 \
  --pareto-rank 1 \
  --num-keywords 15 \
  --max-tokens 20 \
  --output-dir results/stage06_labeling \
  --model-name google/flan-t5-small

# For even more descriptive labels, use more keywords
python -m src.stage06_labeling.main \
  --num-keywords 20 \
  --max-tokens 25

# Skip BERTopic integration (NOT RECOMMENDED - labels won't be saved to model)
python -m src.stage06_labeling.main --no-integrate

# Use native safetensors instead of pickle wrapper
python -m src.stage06_labeling.main --use-native
```

### Behavior

**Default behavior:**
- Always loads BERTopic model (primary source for topics)
- Extracts topics from BERTopic model
- Generates labels using FLAN-T5
- Saves labels to JSON file
- **Always integrates labels back into BERTopic model** (so labels are saved to the model)

**Optional JSON comparison:**
- If `--topics-json` is provided, also loads topics from JSON file
- Compares topics from both sources (BERTopic vs JSON)
- Reports comparison statistics (matches, differences, etc.)
- Useful for validation and manual inspection

**Note:** Labels are always saved to the BERTopic model by default. Use `--no-integrate` only if you explicitly don't want labels in the model (not recommended).

### Command-Line Arguments

- `--embedding-model`: Model name (default: `paraphrase-MiniLM-L6-v2`)
- `--pareto-rank`: Model rank (default: 1)
- `--base-dir`: Base directory for models (default: `models/retrained`)
- `--use-native`: Load native safetensors instead of pickle wrapper
- `--num-keywords`: Number of top keywords per topic (default: 8)
- `--max-tokens`: Max tokens for label generation (default: 16)
- `--output-dir`: Output directory for labels JSON (default: `results/stage06_labeling`)
- `--no-integrate`: Skip integrating labels back into BERTopic
- `--model-name`: FLAN-T5 model name (default: `google/flan-t5-small`)
- `--device`: Device to use (`cuda` or `cpu`, default: auto-detect)
- `--topics-json`: Path to topics JSON file (optional, for comparison/inspection with BERTopic topics)

### Output

The labeling pipeline generates:
- **Labels JSON**: `results/stage06_labeling/labels_pos_{model_name}.json`
  - Format: `{"topic_id": "label", ...}`
  - Example: `{"0": "Online shopping", "1": "Political elections"}`

If `--no-integrate` is not set, labels are also integrated into the BERTopic model and will appear in visualizations.

### Key Files

- **`generate_labels.py`**: Core labeling functions
  - `load_bertopic_model()`: Load retrained BERTopic model
  - `extract_pos_topics()`: Extract POS representation topics
  - `load_labeling_model()`: Load FLAN-T5 model
  - `generate_label_from_keywords()`: Generate label from keywords
  - `generate_all_labels()`: Batch process all topics
  - `save_labels()`: Save labels to JSON
  - `integrate_labels_to_bertopic()`: Integrate labels into BERTopic

- **`main.py`**: CLI entry point for topic labeling

- **`composites.py`**: Core composite building logic
  - `build_mapping()`: Creates topic-to-composite mapping
  - `main()`: Executes composite building pipeline

## Composite Building Usage

```bash
# Direct execution of composites.py
python -m src.stage06_labeling.composites
```

## Inputs

- **Prepared Books**: `data/processed/prepared_books.parquet`
  - Must contain topic probability columns

- **Manual Mapping**: `results/experiments/.../Billionaire_Manual_Mapping_Topics_by_Thematic_Clusters.md`

- **Codebook** (Optional): `results/experiments/.../focused_topic_codebook.csv`

## Outputs

- **Composites Parquet**: `results/topics/ap_composites.parquet`
  - Columns: Metadata + 16 composite scores (A-P)

## Data Contracts

See [docs/DATA_CONTRACTS.md](../../docs/DATA_CONTRACTS.md) for detailed specifications.

## Special Cases

### Weighted Topics

Some topics have special weighted assignments:
- **"Woman Pleasure and Arousal"**: 70% `C_Explicit_Eroticism`, 30% `B_Mutual_Intimacy`

### Default Assignment

If a topic has no mapping, it is assigned equally to all matched composites:
```python
dests = {d: 1.0/len(mapping[t]) for d in mapping.get(t, [])}
```

## Methodology

See [docs/METHODOLOGY.md](../../docs/METHODOLOGY.md) for detailed mapping algorithm.

## Dependencies

### POS Topic Labeling
- `transformers` for FLAN-T5 model
- `torch` for model inference
- `bertopic` for topic model loading
- `sentence-transformers` for embeddings

### Composite Building
- `pandas` for data manipulation
- `pathlib` for file handling
- `re` for pattern matching

