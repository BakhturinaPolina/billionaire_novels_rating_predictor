# OpenRouter API Experiments for Topic Labeling

This experimental module provides an alternative implementation for generating topic labels using the OpenRouter API instead of local LLM inference. It maintains the same logic and workflow as the main `generate_labels.py` module but uses cloud-based API calls.

## Overview

This module is designed for experiments with different LLM providers and models via OpenRouter. It uses the same prompt structure and domain detection logic as the main labeling module but replaces local model inference with API calls to OpenRouter.

## Key Differences from Main Module

| Feature | Main Module (`generate_labels.py`) | This Module (`generate_labels_openrouter.py`) |
|---------|-----------------------------------|-----------------------------------------------|
| **LLM Provider** | Local (Hugging Face models) | OpenRouter API |
| **Model** | `mistralai/Mistral-7B-Instruct-v0.2` (local) | `mistralai/mistral-nemo` (via API) |
| **Hardware Requirements** | GPU recommended (4-bit quantization) | No local GPU needed |
| **Model Loading** | Loads model into memory | API client initialization |
| **Inference** | Local model inference | HTTP API calls |
| **Rate Limiting** | N/A | Built-in delays and retry logic |
| **Cost** | Free (local compute) | Pay-per-use API costs |

## Features

- ✅ **Romance-aware prompts** optimized for modern romantic and erotic fiction
- ✅ Same domain detection and adaptive hints as main module
- ✅ Same POS topic extraction workflow
- ✅ **Representative document snippets** for improved label precision
- ✅ Streaming and batch processing options
- ✅ Automatic retry logic for API failures
- ✅ Rate limit handling with delays
- ✅ Same output format (JSON files)
- ✅ Integration with BERTopic models

## API Configuration

### Default Settings

- **API Key**: Hardcoded in `generate_labels_openrouter.py` (can be overridden via CLI)
- **Model**: `mistralai/mistral-nemo`
- **Base URL**: `https://openrouter.ai/api/v1`
- **Timeout**: 60 seconds

### API Key

The default API key is embedded in the code. To use your own key:

1. Pass it via `--api-key` command-line argument, or
2. Modify `DEFAULT_OPENROUTER_API_KEY` in `generate_labels_openrouter.py`

## Installation

No additional dependencies beyond the main project requirements. The module uses:
- `openai` package (for OpenAI-compatible API client)
- `tenacity` (for retry logic)

Both should already be in your environment if you have the main project dependencies installed.

## Snippets Feature

This module includes an enhanced labeling approach that uses **representative document snippets** alongside keywords to generate more precise, scene-level labels.

### What Are Snippets?

Snippets are short excerpts (typically 6 sentences) from documents that best represent each topic. They provide the LLM with actual scene-level context, enabling it to:

- Distinguish specific acts (e.g., "Blowjob in Car" vs "Erotic Intimacy")
- Identify settings (e.g., "Kitchen Argument" vs "General Conflict")
- Capture tone and style (e.g., "Rough Kissing" vs "Tender Kissing")
- Avoid vague, euphemistic labels

### How It Works

1. **Extraction**: Representative documents are extracted from the BERTopic model using `get_representative_docs()` or `representative_docs_` attribute
2. **Formatting**: Up to 6 snippets per topic are formatted as numbered quotes
3. **Integration**: Snippets are included in the user prompt between POS cues and the label request
4. **Fallback**: If snippets are unavailable, the system works with keywords only (backward compatible)

### Default Parameters

- **Max snippets per topic**: 6 (configurable via code)
- **Max characters per snippet**: 200 (truncates at word boundaries)
- **Token cost**: ~75 tokens per topic for snippets (~7.6% increase)

### Benefits

- **Precision**: Labels capture specific acts, settings, and tones
- **Neutrality**: Explicit instructions prevent euphemisms
- **Scene-level understanding**: Better than keyword-only associations
- **Cost-effective**: Small token increase for significant quality improvement

### Requirements

- BERTopic model must have representative documents available
- Automatically extracted when BERTopic model is loaded
- Works with both streaming and batch processing modes

For detailed information, see:
- `prompts_with_snippets.md`: Prompt structure and examples
- `SNIPPETS_LOGIC.md`: Theoretical reasoning and design decisions

## JSON Output Format

When using `--use-improved-prompts`, the romance-aware prompt produces structured JSON output with the following fields:

- **`label`**: Short noun phrase (2-6 words) describing the topic
- **`scene_summary`**: One complete sentence (12-25 words) describing a typical scene
- **`primary_categories`**: 1-3 high-level tags (e.g., "romance_core", "sexual_content", "work_life")
- **`secondary_categories`**: 0-5 specific tags with dimension:value format (e.g., "setting:car", "activity:kissing")
- **`is_noise`**: Boolean indicating if the topic is a technical artifact or meaningless
- **`rationale`**: 1-3 sentences explaining how keywords and snippets support the label

The JSON parsing pipeline automatically extracts all these fields and includes them in the output JSON file. This provides richer metadata for downstream analysis while maintaining backward compatibility with the label-only format.

## Usage

### Basic Usage

```bash
python -m src.stage08_llm_labeling.openrouter_experiments.core.main_openrouter \
    --embedding-model paraphrase-MiniLM-L6-v2 \
    --pareto-rank 1 \
    --num-keywords 15 \
    --max-tokens 40
```

### With Custom API Key

```bash
python -m src.stage08_llm_labeling.openrouter_experiments.core.main_openrouter \
    --embedding-model paraphrase-MiniLM-L6-v2 \
    --api-key YOUR_API_KEY_HERE \
    --model-name mistralai/mistral-nemo
```

### With Topics JSON (Streaming Mode)

```bash
python -m src.stage08_llm_labeling.openrouter_experiments.core.main_openrouter \
    --embedding-model paraphrase-MiniLM-L6-v2 \
    --topics-json results/stage06_topic_exploration/topics_all_representations_paraphrase-MiniLM-L6-v2.json \
    --num-keywords 15
```

### With Model Suffix (e.g., Noise Labels)

To load a model that was saved with noise labels (from `topic_quality_eda.ipynb`):

```bash
python -m src.stage08_llm_labeling.openrouter_experiments.core.main_openrouter \
    --embedding-model paraphrase-MiniLM-L6-v2 \
    --model-suffix "_with_noise_labels" \
    --num-keywords 15
```

### All Options

```bash
python -m src.stage08_llm_labeling.openrouter_experiments.core.main_openrouter \
    --embedding-model paraphrase-MiniLM-L6-v2 \
    --pareto-rank 1 \
    --base-dir models/retrained \
    --use-native \
    --model-suffix "_with_noise_labels" \
    --num-keywords 15 \
    --max-tokens 40 \
    --output-dir results/stage08_llm_labeling \
    --api-key sk-or-v1-... \
    --model-name mistralai/mistral-nemo \
    --temperature 0.3 \
    --batch-size 50 \
    --topics-json path/to/topics.json \
    --limit-topics 100 \
    --no-integrate
```

## Command-Line Arguments

### Required Arguments

None (uses defaults for all parameters)

### Optional Arguments

- `--embedding-model`: BERTopic embedding model name (default: `paraphrase-MiniLM-L6-v2`)
- `--pareto-rank`: Pareto rank of the model to load (default: `1`)
- `--base-dir`: Base directory containing retrained models (default: `models/retrained`)
- `--use-native`: Load native safetensors instead of pickle wrapper
- `--model-suffix`: Optional suffix to append to model filename/directory (e.g., `_with_noise_labels`) (default: empty)
- `--num-keywords`: Number of top keywords per topic (default: `15`)
- `--max-tokens`: Maximum tokens to generate per label (default: `40`)
- `--output-dir`: Output directory for labels JSON (default: `results/stage08_llm_labeling`)
- `--api-key`: OpenRouter API key (default: hardcoded key)
- `--model-name`: OpenRouter model name (default: `mistralai/mistral-nemo`)
- `--temperature`: Sampling temperature (default: `0.3`)
- `--batch-size`: Topics per progress log (default: `50`)
- `--topics-json`: Path to topics JSON file (enables streaming mode)
- `--limit-topics`: Limit number of topics to process (for testing)
- `--no-integrate`: Skip integrating labels into BERTopic model
- `--config`: Path to paths configuration file

## Output

The module generates:

1. **Labels JSON file**: `results/stage08_llm_labeling/labels_pos_openrouter_{model_name}.json`
   - Format: `{"topic_id": "label", ...}`
   - Same format as main module

2. **Log file**: `logs/stage08_llm_labeling_{timestamp}.log`
   - Contains all console output and errors
   - Useful for debugging API issues

3. **Integrated labels** (unless `--no-integrate` is set):
   - Labels are integrated into the BERTopic model
   - Available in BERTopic visualizations and topic info

## Error Handling

The module includes:

- **Automatic retries**: Uses `tenacity` for exponential backoff retries
- **Rate limit handling**: Built-in delays between API calls (0.5 seconds)
- **Fallback labels**: If API fails, generates simple fallback from keywords
- **Error logging**: All errors are logged to both console and log file

## Rate Limits and Costs

- **Rate Limits**: OpenRouter has rate limits based on your account tier
- **Costs**: Pay-per-use pricing (check OpenRouter pricing page)
- **Delays**: Module includes 0.5 second delays between calls to avoid rate limits
- **Batch Processing**: Processes topics in batches with progress logging

## Comparison with Main Module

### When to Use This Module

- ✅ You want to experiment with different models via OpenRouter
- ✅ You don't have GPU resources for local inference
- ✅ You want to test API-based labeling workflows
- ✅ You need faster iteration without model loading overhead

### When to Use Main Module

- ✅ You have GPU resources available
- ✅ You want to avoid API costs
- ✅ You need offline/private inference
- ✅ You're processing very large numbers of topics (cost considerations)

## Troubleshooting

### API Errors

If you encounter API errors:

1. Check your API key is valid
2. Verify you have sufficient credits/quota
3. Check rate limits (may need to increase delays)
4. Review log file for detailed error messages

### Rate Limit Errors

If you hit rate limits:

1. Increase delay between calls (modify `time.sleep(0.5)` in code)
2. Reduce batch size
3. Process topics in smaller chunks using `--limit-topics`

### Model Not Found

If the model name is invalid:

1. Check OpenRouter model list: https://openrouter.ai/models
2. Verify model name spelling
3. Try alternative model names

## Folder Structure

This module is organized into the following subdirectories:

- **`core/`**: Core implementation files
  - `generate_labels_openrouter.py`: Main labeling logic with OpenRouter API
  - `main_openrouter.py`: CLI entry point for single model labeling

- **`tools/`**: Utility scripts for specific tasks
  - `compare_models_openrouter.py`: Multi-model comparison tool
  - `validate_label_quality.py`: Quality validation script
  - `inspect_random_topics.py`: Topic inspection utility

- **`docs/`**: Documentation files
  - `prompts.md`: Prompt documentation (without snippets)
  - `prompts_with_snippets.md`: Enhanced prompts with snippets
  - `SNIPPETS_LOGIC.md`: Theoretical reasoning for snippets feature
  - `MODEL_SELECTION_RECOMMENDATION.md`: Model comparison analysis
  - `COST_EVALUATION_REPORT.md`: Cost analysis documentation

- **`evaluation/`**: Testing and evaluation scripts
  - `test_hybrid_performance.py`: Performance testing for hybrid approach
  - `eval_snippet_centrality.py`: Snippet centrality evaluation

- **`archive/`**: Historical files documenting resolved issues
  - `BERTOPIC_REVIEW.md`: Code review notes (resolved)
  - `HYBRID_APPROACH_IMPLEMENTED.md`: Historical implementation notes
  - `PERFORMANCE_REGRESSION_ANALYSIS.md`: Historical performance analysis

## Multi-Model Comparison

A new script `tools/compare_models_openrouter.py` allows you to test multiple free OpenRouter models side-by-side and generate comparison outputs for manual inspection.

### Free Models Available

The script includes these free models by default:

1. **mistralai/mistral-7b-instruct:free** - General-purpose instruct model, 32k context
2. **mistralai/mistral-nemo:free** - Instruction-following and reasoning model
3. **venice/uncensored:free** - Uncensored Mistral-24B variant (handles explicit content with fewer refusals)
4. **tngtech/tng-r1t-chimera:free** - General chat model

You can specify custom models using the `--models` argument.

### Usage

Compare multiple models with default settings (30 topics per model):

```bash
python -m src.stage08_llm_labeling.openrouter_experiments.tools.compare_models_openrouter \
    --embedding-model paraphrase-MiniLM-L6-v2 \
    --topics-json results/stage06_topic_exploration/topics_all_representations_paraphrase-MiniLM-L6-v2.json \
    --limit-topics 30
```

Compare specific models:

```bash
python -m src.stage08_llm_labeling.openrouter_experiments.tools.compare_models_openrouter \
    --embedding-model paraphrase-MiniLM-L6-v2 \
    --topics-json results/stage06_topic_exploration/topics_all_representations_paraphrase-MiniLM-L6-v2.json \
    --models mistralai/mistral-7b-instruct:free mistralai/mistral-nemo:free \
    --limit-topics 30
```

### Output Format

The comparison script generates:

1. **CSV file** (`comparison_models_{timestamp}.csv`):
   - Columns: Topic ID, Keywords, then one column per model with labels
   - Easy to open in Excel/Google Sheets for side-by-side comparison
   - Format: Each row shows one topic with all model labels

2. **JSON file** (`comparison_models_{timestamp}.json`):
   - Structured format with metadata
   - Format:
     ```json
     {
       "metadata": {
         "generated_at": "...",
         "models": ["model1", "model2", ...],
         "total_topics": 30
       },
       "topics": {
         "0": {
           "topic_id": 0,
           "keywords": ["keyword1", "keyword2", ...],
           "labels": {
             "model1": "Label from model1",
             "model2": "Label from model2"
           }
         }
       }
     }
     ```

3. **Individual model outputs**: Each model's labels are also saved separately as `labels_pos_openrouter_{model_name}_{embedding_model}.json`

### Comparison Workflow

1. Run comparison with `--limit-topics 30` for initial testing
2. Review CSV file to compare labels across models
3. Identify which model produces best labels for your use case
4. Run full labeling with the best model using `core/main_openrouter.py`

## Files

See the [Folder Structure](#folder-structure) section above for a complete overview. Key files:

- **Core**: `core/generate_labels_openrouter.py`, `core/main_openrouter.py`
- **Tools**: `tools/compare_models_openrouter.py`, `tools/validate_label_quality.py`, `tools/inspect_random_topics.py`
- **Documentation**: `docs/prompts.md`, `docs/prompts_with_snippets.md`, `docs/SNIPPETS_LOGIC.md`
- **Evaluation**: `evaluation/test_hybrid_performance.py`, `evaluation/eval_snippet_centrality.py`
- **Archive**: `archive/` contains historical documentation

## Future Enhancements

Potential improvements:

- Support for multiple model fallbacks
- Configurable retry strategies
- Cost tracking and reporting
- Batch API calls (if supported by OpenRouter)

## See Also

- Main module: `src/stage08_llm_labeling/generate_labels.py`
- Main CLI: `src/stage08_llm_labeling/main.py`
- Prompt documentation: `docs/prompts.md`
- Snippets-enhanced prompts: `docs/prompts_with_snippets.md`
- Snippets theoretical reasoning: `docs/SNIPPETS_LOGIC.md`

