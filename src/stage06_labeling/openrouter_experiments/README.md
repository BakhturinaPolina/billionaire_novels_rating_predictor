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

- ✅ Same prompt structure as main module (Universal System/User prompts)
- ✅ Same domain detection and adaptive hints
- ✅ Same POS topic extraction workflow
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

## Usage

### Basic Usage

```bash
python -m src.stage06_labeling.openrouter_experiments.main_openrouter \
    --embedding-model paraphrase-MiniLM-L6-v2 \
    --pareto-rank 1 \
    --num-keywords 15 \
    --max-tokens 40
```

### With Custom API Key

```bash
python -m src.stage06_labeling.openrouter_experiments.main_openrouter \
    --embedding-model paraphrase-MiniLM-L6-v2 \
    --api-key YOUR_API_KEY_HERE \
    --model-name mistralai/mistral-nemo
```

### With Topics JSON (Streaming Mode)

```bash
python -m src.stage06_labeling.openrouter_experiments.main_openrouter \
    --embedding-model paraphrase-MiniLM-L6-v2 \
    --topics-json results/stage06_exploration/topics_all_representations_paraphrase-MiniLM-L6-v2.json \
    --num-keywords 15
```

### All Options

```bash
python -m src.stage06_labeling.openrouter_experiments.main_openrouter \
    --embedding-model paraphrase-MiniLM-L6-v2 \
    --pareto-rank 1 \
    --base-dir models/retrained \
    --use-native \
    --num-keywords 15 \
    --max-tokens 40 \
    --output-dir results/stage06_labeling_openrouter \
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
- `--num-keywords`: Number of top keywords per topic (default: `15`)
- `--max-tokens`: Maximum tokens to generate per label (default: `40`)
- `--output-dir`: Output directory for labels JSON (default: `results/stage06_labeling_openrouter`)
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

1. **Labels JSON file**: `results/stage06_labeling_openrouter/labels_pos_openrouter_{model_name}.json`
   - Format: `{"topic_id": "label", ...}`
   - Same format as main module

2. **Log file**: `logs/stage06_labeling_openrouter_{timestamp}.log`
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

## Files

- `generate_labels_openrouter.py`: Main labeling logic with OpenRouter API
- `main_openrouter.py`: CLI entry point
- `prompts.md`: Documentation of all prompt versions
- `README.md`: This file
- `__init__.py`: Package initialization

## Future Enhancements

Potential improvements:

- Support for multiple model fallbacks
- Configurable retry strategies
- Cost tracking and reporting
- Batch API calls (if supported by OpenRouter)
- Support for other prompt templates (6-component format)

## See Also

- Main module: `src/stage06_labeling/generate_labels.py`
- Main CLI: `src/stage06_labeling/main.py`
- Prompt documentation: `prompts.md`

