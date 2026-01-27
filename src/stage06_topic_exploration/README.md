# Stage 06: Topic Exploration

Interactive tooling for inspecting retrained BERTopic models with multiple representations and coherence/diversity metrics.

## Status

✅ **Active** — Full implementation.

## Usage

### Basic Exploration

```bash
python -m src.stage06_topic_exploration.explore_retrained_model \
  --embedding-model paraphrase-MiniLM-L6-v2 \
  --pareto-rank 1 \
  --dictionary-path data/interim/octis/corpus.tsv \
  --batch-size 50000
```

### Save Topics for Close Reading

```bash
python -m src.stage06_topic_exploration.explore_retrained_model \
  --embedding-model paraphrase-MiniLM-L6-v2 \
  --pareto-rank 1 \
  --save-topics \
  --output-dir results/stage06_topic_exploration
```

## Inputs

| Source | Path | Description |
|--------|------|-------------|
| Model | `models/retrained/{embedding_model}/model_{rank}.pkl` | Pickle wrapper (default) |
| Model (native) | `models/retrained/{embedding_model}/model_{rank}/` | BERTopic native format (`--use-native`) |
| Documents | `data/processed/chapters.csv` | Full corpus (fallback) |
| Dictionary | `data/interim/octis/corpus.tsv` | OCTIS corpus for Gensim dictionary |

## Outputs

| Output | Description |
|--------|-------------|
| `metrics.json` | Coherence (c_v) and diversity scores per representation |
| `topics_all_representations.json` | All topics with all representations (if `--save-topics`) |

## Module Structure

| File | Purpose |
|------|---------|
| `explore_retrained_model.py` | Main exploration script |

## Representations

Attaches four representations to topics:
- **Main**: Default c-TF-IDF (statistical)
- **KeyBERT**: Semantic similarity-based keywords
- **POS**: Part-of-speech filtered (nouns, verbs, adjectives)
- **MMR**: Maximal Marginal Relevance (diversity-focused)

## Optional Flags

- `--use-native`: Load native BERTopic safetensors instead of pickle wrapper
- `--dataset-csv PATH`: Override document source
- `--fallback-dataset {chapters,subset}`: Pick default CSV if wrapper docs missing
- `--limit-docs N`: Stop after N rows (for testing)
- `--top-k K`: Keywords per topic for metrics (default: 10)
- `--output-dir PATH`: Output directory (default: current directory)
- `--metrics-format {csv,json}`: Metrics file format (default: json)
- `--save-topics`: Extract and save all topics with all representations

## Notes

- Documents loaded in batches (50K default) with progress logging
- Dictionary built by streaming corpus TSV (memory-efficient)
- Wrapper format preferred (guarantees exact training dataset match)
- Metrics computed using same Gensim dictionary as training

## See Also

- [Methodology Report](../../reports/01_stage_reports/stage06_topic_exploration/stage06_topic_exploration_and_representation_analysis.md) — Research rationale and results
