# Stage 03: Modeling

BERTopic model training with OCTIS hyperparameter optimization and GPU acceleration.

## Status

✅ **Active** — Full implementation with RAPIDS GPU acceleration.

## Usage

### Test Pipeline (Recommended First)

```bash
# Test with subset (10K rows)
python -m src.stage03_modeling.test_octis_pipeline --subset

# Test with full dataset
python -m src.stage03_modeling.test_octis_pipeline --full
```

### Train Models

```bash
python -m src.stage03_modeling.main train --config configs/bertopic.yaml
```

### Optimize Hyperparameters

```bash
python -m src.stage03_modeling.main optimize --config configs/octis.yaml
```

### Retrain Models

```bash
python -m src.stage03_modeling.main retrain \
  --dataset_csv data/processed/chapters.csv \
  --out_dir models/retrained/
```

## Requirements

- **CUDA-compatible GPU** (required)
- **RAPIDS cuML** (CUDA 12.x) for GPU acceleration
- **BERTopic** for topic modeling
- **OCTIS** for hyperparameter optimization

## Module Structure

| File | Purpose | GPU |
|------|---------|-----|
| `main.py` | CLI entrypoint (`train`, `retrain`, `optimize`) | N/A |
| `bertopic_runner.py` | OCTIS integration | ✅ RAPIDS |
| `bertopic_octis_model.py` | BERTopic-OCTIS wrapper | ✅ RAPIDS |
| `test_octis_pipeline.py` | Pipeline validation | ✅ RAPIDS |
| `memory_utils.py` | GPU memory monitoring | N/A |
| `convert_topics.py` | Legacy utility (NumPy → JSON) | N/A |

## GPU Acceleration

**Mandatory RAPIDS (cuML)** — no CPU fallback:
- `cuml.manifold.UMAP` for dimensionality reduction
- `cuml.cluster.HDBSCAN` for clustering

Verify GPU setup:
```bash
python -m src.common.check_gpu_setup
```

## Configuration

| File | Purpose |
|------|---------|
| `configs/bertopic.yaml` | BERTopic model parameters |
| `configs/octis.yaml` | OCTIS optimization settings |
| `configs/paths.yaml` | Data and output paths |

## Outputs

| Output | Location | Description |
|--------|----------|-------------|
| Models | `models/` | Trained BERTopic models |
| Topic probs | `results/topics/by_book.csv` | Per-book topic probabilities |
| Topic words | `results/topics/top_models/*.json` | Topic word lists |

## Notes

- Embeddings are cached to avoid recomputation
- Models saved with full hyperparameter configuration
- Character names excluded via custom stoplist (from Stage 02)

## See Also

- [Methodology Report](../../reports/01_stage_reports/stage03_modeling/stage03_modeling_and_retraining_methodology.md) — Research rationale and character name exclusion