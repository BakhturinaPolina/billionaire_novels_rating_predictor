# Stage 06 EDA – Topic Quality Analysis and Noisy Topic Detection

This stage performs exploratory data analysis (EDA) on BERTopic model topics and flags candidate noisy topics for manual inspection before proceeding with LLM-generated topic labeling.

## Overview

**Position in Pipeline**: Between `stage06_topic_exploration` and `stage08_llm_labeling`

**Purpose**: 
- Identify noisy topics before LLM labeling stage
- Keep the model as-is (no topics removed)
- Provide quality metrics for manual inspection
- Flag candidate noisy topics with labels and reasons

## Key Features

- **EDA on POS representation**: Size, POS length, per-topic POS coherence
- **Noise candidate detection**: Flags topics based on configurable thresholds
- **Manual inspection support**: Generates CSV files for easy review
- **Reuses Stage 06 infrastructure**: Same loading & batching logic as `explore_retrained_model.py`
- **Non-destructive**: Does not modify the model clustering, only flags topics

## Inputs

- `models/retrained/<embedding_model>/model_<rank>.pkl` – pickle wrapper containing `RetrainableBERTopicModel`
- `models/retrained/<embedding_model>/model_<rank>/` – native BERTopic folder (if using `--use-native`)
- `data/interim/octis/corpus.tsv` – OCTIS corpus used for gensim dictionary
- `data/processed/chapters.csv` – full cleaned corpus (optional, if wrapper docs unavailable)

## Outputs

1. **`topic_quality_<model>.csv`**: Full topic quality table with all metrics
2. **`topic_noise_candidates_<model>.csv`**: Filtered view of only candidate noisy topics
3. **Model with labels** (optional): If `--apply-labels` is used, saves model with inspection labels

## Usage

### Command Line Interface

```bash
# Basic usage
python -m src.stage07_topic_quality.main \
  --embedding-model paraphrase-MiniLM-L6-v2 \
  --pareto-rank 1 \
  --output-dir results/stage07_topic_quality

# With custom thresholds
python -m src.stage07_topic_quality.main \
  --embedding-model paraphrase-MiniLM-L6-v2 \
  --pareto-rank 1 \
  --min-topic-size 50 \
  --min-pos-words 5 \
  --min-pos-coherence 0.1 \
  --output-dir results/stage07_topic_quality

# Apply labels to noisy topics and save model
python -m src.stage07_topic_quality.main \
  --embedding-model paraphrase-MiniLM-L6-v2 \
  --pareto-rank 1 \
  --apply-labels \
  --save-model-with-labels models/retrained/paraphrase-MiniLM-L6-v2/model_1_with_noise_labels
```

### Jupyter Notebook

The notebook `notebooks/06_labeling/topic_quality_eda.ipynb` provides an interactive version with the same functionality. **Always run with the venv activated:**

```bash
# Activate virtual environment (REQUIRED)
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows

# Launch Jupyter
jupyter notebook notebooks/06_labeling/topic_quality_eda.ipynb
```

The notebook includes a venv verification cell that will warn you if you're not using the correct Python environment.

## Configuration Parameters

### Topic Quality Thresholds

- `--min-topic-size`: Minimum number of documents per topic (default: 30)
- `--min-pos-words`: Minimum number of POS words per topic (default: 3)
- `--min-pos-coherence`: Minimum per-topic POS coherence threshold (default: 0.0)

### Model Loading

- `--use-native`: Load native BERTopic safetensors instead of pickle wrapper
- `--base-dir`: Base directory containing retrained models
- `--embedding-model`: Embedding model name
- `--pareto-rank`: Pareto rank of the model to analyze

### Data Loading

- `--dataset-csv`: Override dataset CSV path
- `--dictionary-path`: Path to OCTIS corpus TSV for dictionary building
- `--batch-size`: Batch size for document loading (default: 50000)
- `--limit-docs`: Limit number of documents loaded (for testing)

## Output Format

The quality table CSV contains:

- `Topic`: Topic ID
- `Count`: Number of documents assigned to this topic
- `Name`: Topic name from BERTopic
- `Representation`: Topic representation type
- `n_pos_words`: Number of POS words in topic representation
- `pos_words`: List of POS words
- `coherence_c_v_pos`: Per-topic POS coherence score
- `flag_small`: Boolean flag if topic is below min size threshold
- `flag_few_pos`: Boolean flag if topic has few POS words
- `flag_low_coh`: Boolean flag if topic has low coherence
- `noise_reason`: Semicolon-separated list of reasons why topic is flagged
- `noise_candidate`: Boolean flag indicating if topic is a noise candidate
- `inspection_label`: Human-readable label with noise reasons prepended

## Workflow Integration

1. **After Stage 06 Exploration**: Run EDA to identify noisy topics
2. **Manual Review**: Inspect `topic_noise_candidates_*.csv` in Excel or similar
3. **Before Stage 06 Labeling**: Use flagged topics to inform labeling strategy
   - Optionally treat confirmed noise topics as `OTHER` in analysis
   - Or overwrite labels using LLM-generated labels later

## Example Output

```
=== Topic Quality Summary ===
Total topics (excluding -1): 150
Candidate noisy topics: 12

Topics flagged as small (<30 docs): 5
Topics flagged as few POS words (<3): 3
Topics flagged as low coherence (<0.00): 8

=== 20 topics with lowest POS coherence ===
[Table showing worst topics]

=== Candidate noisy topics (for manual inspection) ===
[Table showing flagged topics]
```

## Notes

- This stage does **not** remove topics from the model
- Topics are only flagged for manual inspection
- The model clustering remains unchanged
- Labels can be applied optionally for visualization purposes
- Later stages can use the noise candidate flags to filter or handle topics differently

