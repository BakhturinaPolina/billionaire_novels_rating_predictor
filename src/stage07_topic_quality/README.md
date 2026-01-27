# Stage 07: Topic Quality Analysis

Exploratory data analysis to identify candidate noisy topics before LLM labeling.

## Status

✅ **Active** — Full implementation.

## Usage

### Basic Usage

```bash
python -m src.stage07_topic_quality.main \
  --embedding-model paraphrase-MiniLM-L6-v2 \
  --pareto-rank 1 \
  --output-dir results/stage07_topic_quality
```

### Custom Thresholds

```bash
python -m src.stage07_topic_quality.main \
  --embedding-model paraphrase-MiniLM-L6-v2 \
  --pareto-rank 1 \
  --min-topic-size 50 \
  --min-pos-words 5 \
  --min-pos-coherence 0.1 \
  --output-dir results/stage07_topic_quality
```

### Apply Labels to Model

```bash
python -m src.stage07_topic_quality.main \
  --embedding-model paraphrase-MiniLM-L6-v2 \
  --pareto-rank 1 \
  --apply-labels \
  --save-model-with-labels models/retrained/paraphrase-MiniLM-L6-v2/model_1_with_noise_labels
```

## Inputs

| Source | Path | Description |
|--------|------|-------------|
| Model | `models/retrained/{embedding_model}/model_{rank}.pkl` | Pickle wrapper (default) |
| Model (native) | `models/retrained/{embedding_model}/model_{rank}/` | BERTopic native format (`--use-native`) |
| Dictionary | `data/interim/octis/corpus.tsv` | OCTIS corpus for Gensim dictionary |
| Documents | `data/processed/chapters.csv` | Full corpus (optional fallback) |

## Outputs

| Output | Description |
|--------|-------------|
| `topic_quality_{model}.csv` | Full quality table with all metrics |
| `topic_noise_candidates_{model}.csv` | Filtered view of candidate noisy topics |
| Model with labels | Optional (if `--apply-labels` used) |

## Module Structure

| File | Purpose |
|------|---------|
| `main.py` | CLI entrypoint |
| `topic_quality_analysis.py` | Analysis functions |

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-topic-size` | 30 | Minimum documents per topic |
| `--min-pos-words` | 3 | Minimum POS words per topic |
| `--min-pos-coherence` | 0.0 | Minimum per-topic POS coherence |

## Output Format

CSV columns:
- `Topic`: Topic ID
- `Count`: Number of documents
- `n_pos_words`: Number of POS words
- `coherence_c_v_pos`: Per-topic POS coherence score
- `noise_candidate`: Boolean flag
- `noise_reason`: Semicolon-separated reasons
- `inspection_label`: Human-readable label

## Notes

- **Non-destructive**: Does not remove topics, only flags them
- Reuses Stage 06 infrastructure (loading & batching logic)
- Topics flagged for manual inspection before Stage 08 labeling

## See Also

- [Methodology Report](../../reports/01_stage_reports/stage07_topic_quality/stage07_topic_quality_analysis_research_report.md) — Research methodology and results
