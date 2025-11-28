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

## Usage
```bash
python -m src.stage06_BERTopic_topics_exploration.explore_retrained_model \
  --embedding-model paraphrase-MiniLM-L6-v2 \
  --pareto-rank 1 \
  --dictionary-path data/interim/octis/corpus.tsv \
  --batch-size 50000
```

Optional flags:
- `--use-native` – skip the pickle wrapper and load the safetensors folder instead.
- `--dataset-csv PATH` – override the document source (e.g., `data/processed/chapters_subset_10000.csv`).
- `--fallback-dataset {chapters,subset}` – pick the default CSV if wrapper docs are missing.
- `--limit-docs N` – stop streaming after `N` rows (handy for rapid iteration).
- `--top-k K` – number of keywords per topic when computing metrics.

Logs include stage names, start/finish timestamps, and per-batch counters so it is obvious whether the code is processing slowly or stuck between stages.


