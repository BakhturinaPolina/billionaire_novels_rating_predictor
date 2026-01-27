# Stage 05: Retraining

Retrain top Pareto-efficient models with their optimal hyperparameters.

## Status

✅ **Active** — Full implementation with RAPIDS GPU acceleration.

## Usage

```bash
# Retrain top 4 models (default)
python -m src.stage05_retraining.main retrain

# Retrain top N models
python -m src.stage05_retraining.main retrain --top_n 4

# Custom paths
python -m src.stage05_retraining.main retrain \
  --pareto_csv results/stage04_selection/pareto.csv \
  --top_n 4 \
  --config configs/paths.yaml \
  --output_dir models/retrained/
```

## Requirements

- **CUDA-compatible GPU** (required)
- **RAPIDS cuML** (CUDA 12.x) for GPU acceleration
- **BERTopic** for topic modeling
- **OCTIS** (Dataset class only, not optimization)

## Module Structure

| File | Purpose | GPU |
|------|---------|-----|
| `main.py` | CLI entrypoint (`retrain` command) | N/A |
| `pareto_loader.py` | Load top N models from Pareto CSV | N/A |
| `retrain_models.py` | Core retraining logic | ✅ RAPIDS |
| `diagnose_data.py` | Data validation utilities | N/A |

## GPU Acceleration

**Mandatory RAPIDS (cuML)** — no CPU fallback:
- `cuml.manifold.UMAP` for dimensionality reduction
- `cuml.cluster.HDBSCAN` for clustering

## Configuration

| File | Purpose |
|------|---------|
| `configs/paths.yaml` | Data and output paths |
| `results/stage04_selection/pareto.csv` | Pareto-efficient model configurations |

## Outputs

Models saved in:
```
models/retrained/{embedding_model}/
├── model_1.pkl              # Pickle format (full wrapper)
├── model_1/                 # BERTopic native format (safetensors)
├── model_1_metadata.json    # Training metadata
└── ...
```

**Output formats**:
1. **Pickle** (`.pkl`): Full `RetrainableBERTopicModel` instance
2. **BERTopic native** (directory): Standard format for `BERTopic.load()`
3. **Metadata** (`.json`): Hyperparameters, scores, topic counts, timestamps

## Notes

- Embeddings cached (reuses Stage 03 cache)
- Independent model training (failures don't stop others)
- Character names excluded (same as Stage 03)
- No OCTIS optimization (hyperparameters from CSV)

## Differences from Stage 03

| Aspect | Stage 03 | Stage 05 |
|--------|----------|----------|
| Optimization | OCTIS hyperparameter search | Direct training with provided hyperparameters |
| Input | Configuration files | Pareto CSV |
| Output formats | OCTIS-compatible | Pickle, native, metadata JSON |

## See Also

- [Methodology Report](../../reports/01_stage_reports/stage05_retraining/stage05_retraining_methodology_and_results.md) — Research rationale and results
