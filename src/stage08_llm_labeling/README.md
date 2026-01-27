# Stage 08: LLM-Based Topic Labeling

## Overview

Stage 08 generates interpretable labels for BERTopic topics using Large Language Models (LLMs) via the OpenRouter API. It transforms keyword clusters into descriptive, scene-level labels suitable for computational literary analysis.

## Status

✅ **Fully Implemented**

## Functionality

### LLM-Based Label Generation

- **Input**: BERTopic topics (keywords + representative document snippets)
- **Output**: Descriptive labels (2-6 words) + scene summaries + metadata
- **Model**: `mistralai/Mistral-Nemo-Instruct-2407` via OpenRouter API
- **Coverage**: 368/368 topics (100%)

### Key Features

- **Romance-aware prompts** optimized for modern romantic and erotic fiction
- **Representative document snippets** for scene-level disambiguation
- **Anti-hallucination constraints** based on empirical testing
- **Structured JSON output** for programmatic analysis
- **Streaming and batch processing** options
- **Automatic retry logic** for API failures

## Key Files

- **`generate_labels.py`**: Main labeling module (local inference)
- **`openrouter_experiments/`**: OpenRouter API implementation
  - **`core/generate_labels_openrouter.py`**: Main labeling logic
  - **`core/main_openrouter.py`**: CLI entry point
  - **`tools/compare_models_openrouter.py`**: Multi-model comparison
  - **`tools/validate_label_quality.py`**: Quality validation

## Usage

### Basic Usage

```bash
python -m src.stage08_llm_labeling.openrouter_experiments.core.main_openrouter \
    --embedding-model paraphrase-MiniLM-L6-v2 \
    --pareto-rank 1 \
    --num-keywords 15 \
    --max-tokens 40
```

### With Topics JSON (Streaming Mode)

```bash
python -m src.stage08_llm_labeling.openrouter_experiments.core.main_openrouter \
    --embedding-model paraphrase-MiniLM-L6-v2 \
    --topics-json results/stage06_topic_exploration/topics_all_representations_paraphrase-MiniLM-L6-v2.json \
    --num-keywords 15
```

### Model Comparison

```bash
python -m src.stage08_llm_labeling.openrouter_experiments.tools.compare_models_openrouter \
    --embedding-model paraphrase-MiniLM-L6-v2 \
    --topics-json results/stage06_topic_exploration/topics_all_representations_paraphrase-MiniLM-L6-v2.json \
    --limit-topics 30
```

## Configuration

### API Configuration

- **API Key**: Set via `--api-key` argument or modify `DEFAULT_OPENROUTER_API_KEY`
- **Model**: Default `mistralai/mistral-nemo`, configurable via `--model-name`
- **Base URL**: `https://openrouter.ai/api/v1`

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Temperature | 0.35 | Balanced for consistency + natural phrasing |
| Max tokens | 40 | Sufficient for 2-6 word labels |
| Rate limit delay | 4.0s | Conservative API rate limiting |

## Outputs

### Files

- `results/stage08_llm_labeling/labels_pos_openrouter_{model_name}.json`
- `logs/stage08_llm_labeling_{timestamp}.log`

### JSON Output Format

```json
{
  "label": "Makeout In Parked Car",
  "scene_summary": "In a parked car, they kiss and touch...",
  "primary_categories": ["romance_core", "sexual_content"],
  "secondary_categories": ["setting:car", "activity:kissing"],
  "is_noise": false,
  "rationale": "Keywords indicate car setting and physical intimacy..."
}
```

### Integration with BERTopic

Labels are integrated into the BERTopic model's `topic_metadata_` attribute (unless `--no-integrate` is set).

## Data Flow

```
BERTopic Model (topics + keywords)
    ↓
POS-Filtered Keywords (nouns, verbs, adjectives)
    ↓
Representative Document Snippets (6 per topic)
    ↓
LLM Labeling (OpenRouter API)
    ↓
JSON Output + BERTopic Integration
    ↓
Stage 09: Taxonomy Mapping
```

## Dependencies

- `openai` (OpenAI-compatible API client)
- `tenacity` (retry logic)
- `spacy` (POS tagging)
- `bertopic` (topic model loading)

## Algorithm Details

See [reports/01_stage_reports/stage08_llm_labeling/stage08_llm_labeling_report.md](../../reports/01_stage_reports/stage08_llm_labeling/stage08_llm_labeling_report.md) for detailed methodology.
