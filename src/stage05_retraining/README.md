# Stage 05: Retraining

Retrain top Pareto-efficient models with their optimal hyperparameters.

## Overview

This stage retrains the top N Pareto-efficient models from `results/pareto/pareto.csv` using their specific hyperparameters. The implementation follows `stage03_modeling` patterns but without OCTIS optimization - directly training with provided hyperparameters.

## Requirements

- **RAPIDS cuML** (CUDA 12.x) - Required for GPU acceleration
- **CUDA-compatible GPU** - Required
- **BERTopic** - Topic modeling library
- **OCTIS** - For Dataset class only (not for optimization)
- **pandas** - For CSV reading

## Files

### Core Files

- **`main.py`** - CLI entry point with `retrain` command
- **`pareto_loader.py`** - CSV parsing and model selection
- **`retrain_models.py`** - Core retraining logic

### File Status

| File | Status | GPU | Notes |
|------|-------|-----|-------|
| `main.py` | ✅ Active | N/A | Entry point |
| `pareto_loader.py` | ✅ Active | N/A | CSV parsing |
| `retrain_models.py` | ✅ Active | ✅ RAPIDS | Core retraining |

## Usage

### Retrain Top Models

```bash
# Retrain top 4 models (default)
python -m src.stage05_retraining.main retrain

# Retrain top N models
python -m src.stage05_retraining.main retrain --top_n 4

# Specify custom paths
python -m src.stage05_retraining.main retrain \
  --pareto_csv results/pareto/pareto.csv \
  --top_n 4 \
  --config configs/paths.yaml \
  --output_dir models/retrained/
```

## GPU Acceleration

**This stage ALWAYS uses RAPIDS (cuML) for GPU acceleration.**

- Uses `cuml.manifold.UMAP` (not CPU `umap-learn`)
- Uses `cuml.cluster.HDBSCAN` (not CPU `hdbscan`)
- No CPU fallback - requires GPU

See `src/common/gpu_models.py` for GPU utilities.

### Verify GPU Setup

```bash
python -m src.common.check_gpu_setup
```

Or in Python:

```python
from src.common.gpu_models import print_gpu_status
print_gpu_status()
```

## Configuration

- **`configs/paths.yaml`** - Data and output paths
- **`results/pareto/pareto.csv`** - Pareto-efficient model configurations

## Outputs

Models are saved in the following structure:

```
models/retrained/
├── {embedding_model_1}/
│   ├── model_1.pkl                    # Pickle format (full wrapper)
│   ├── model_1/                      # BERTopic native format
│   ├── model_1_metadata.json         # Training metadata
│   ├── model_2.pkl
│   └── ...
└── {embedding_model_2}/
    └── ...
```

### Output Formats

1. **Pickle format** (`.pkl`): Full `RetrainableBERTopicModel` instance including embeddings and wrapper state
2. **BERTopic native format** (directory): Native BERTopic model format for direct loading with `BERTopic.load()`
3. **Metadata** (`.json`): Hyperparameters, scores, training timestamp, and model statistics

## Notes

- All models use GPU acceleration via RAPIDS
- Embeddings are cached to avoid recomputation (same cache as stage03)
- Models are saved with full hyperparameter configuration
- Each model is retrained independently - failures in one model don't stop others
- The stage selects top N models by `pareto_rank` column (handles duplicates by keeping all)

## Differences from Stage 03

- **No OCTIS optimization**: Hyperparameters are read directly from CSV
- **Direct training**: Models are trained with specific hyperparameters, not searched
- **Model saving**: Both pickle and BERTopic native formats are saved
- **Metadata tracking**: Each model includes detailed metadata JSON

