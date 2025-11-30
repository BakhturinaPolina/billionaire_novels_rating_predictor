# Stage 06 – BERTopic Topics Exploration

Interactive tooling for inspecting retrained BERTopic checkpoints coming out of Stage 05. The module focuses on fast, instrumented loading of the pickle wrapper or native safetensors folder, attaching richer representation models, and computing coherence/diversity diagnostics with aggressive logging so bottlenecks are easy to spot.

## Key Inputs
- `models/retrained/<embedding_model>/model_<rank>.pkl` – pickle wrapper containing `RetrainableBERTopicModel`.
- `models/retrained/<embedding_model>/model_<rank>/` – native BERTopic folder saved with `serialization="safetensors"`.
- `models/retrained/paraphrase-MiniLM-L6-v2/model_1_metadata.json` – hyperparameter summary + topic counts for quick reference.
- `data/processed/chapters.csv` – full cleaned corpus (Stage 05 output).
- `data/processed/chapters_subset_10000.csv` – lightweight subset for smoke tests.
- `data/interim/octis/corpus.tsv` – OCTIS corpus used as the canonical gensim dictionary source.

## What the Helper Does
1. Loads the wrapper (default) or native BERTopic folder with `--use-native`.
2. Streams documents in batches (50k default) either from the wrapper cache or CSV fallback, logging progress every batch.
3. Builds a gensim dictionary by streaming `corpus.tsv`, again in batches, so the same vocabulary used during retraining is reused for coherence scoring.
4. Attaches additional representations (Main, KeyBERT, POS, MMR), calls `update_topics`, and tracks timing for each phase.
5. Computes `c_v` coherence + topic diversity per representation and prints a compact summary table.
6. **Saves metrics** to CSV or JSON (default: JSON) for further analysis.
7. **Extracts all topics** with word lists for all representations when `--save-topics` is used, saving to JSON for close reading evaluation.

## Usage

### Basic Usage
```bash
python -m src.stage06_exploration.explore_retrained_model \
  --embedding-model paraphrase-MiniLM-L6-v2 \
  --pareto-rank 1 \
  --dictionary-path data/interim/octis/corpus.tsv \
  --batch-size 50000
```

This will:
- Compute and print metrics to console
- Save metrics to `metrics.json` (or `metrics.csv` if `--metrics-format csv` is used)

### Save Topics for Close Reading
```bash
python -m src.stage06_exploration.explore_retrained_model \
  --embedding-model paraphrase-MiniLM-L6-v2 \
  --pareto-rank 1 \
  --save-topics \
  --output-dir results/stage06_exploration
```

This will:
- Save metrics to `results/stage06_exploration/metrics.json`
- Extract all topics with all representations and save to `results/stage06_exploration/topics_all_representations.json`

The topics JSON file contains a nested structure:
```json
{
  "Main": {
    "0": [{"word": "example", "score": 0.123}, ...],
    "1": [{"word": "another", "score": 0.456}, ...]
  },
  "KeyBERT": {
    "0": [{"word": "example", "score": 0.123}, ...],
    ...
  },
  "POS": {...},
  "MMR": {...}
}
```

### Optional Flags
- `--use-native` – skip the pickle wrapper and load the safetensors folder instead.
- `--dataset-csv PATH` – override the document source (e.g., `data/processed/chapters_subset_10000.csv`).
- `--fallback-dataset {chapters,subset}` – pick the default CSV if wrapper docs are missing.
- `--limit-docs N` – stop streaming after `N` rows (handy for rapid iteration).
- `--top-k K` – number of keywords per topic when computing metrics (default: 10).
- `--output-dir PATH` – directory to save output files (default: current directory).
- `--metrics-format {csv,json}` – format for metrics file (default: json).
- `--save-topics` – extract and save all topics with all representations to JSON.

## Output Files

1. **Metrics File** (`metrics.json` or `metrics.csv`):
   - Contains coherence (c_v) and diversity scores for each representation
   - Always saved (to current directory or `--output-dir` if specified)

2. **Topics File** (`topics_all_representations.json`):
   - Contains all topics with word lists for all representations (Main, KeyBERT, POS, MMR)
   - Only saved when `--save-topics` flag is used
   - Useful for close reading evaluation and qualitative analysis

Logs include stage names, start/finish timestamps, and per-batch counters so it is obvious whether the code is processing slowly or stuck between stages.


