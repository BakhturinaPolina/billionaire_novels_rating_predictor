# Stage 03: Modeling

BERTopic model training and retraining with GPU acceleration.

## Overview

This stage handles:
- **BERTopic model training** with OCTIS integration
- **Model retraining** from hyperparameter tables
- **GPU-accelerated** UMAP and HDBSCAN using RAPIDS (cuML)

## Requirements

- **RAPIDS cuML** (CUDA 12.x) - Required for GPU acceleration
- **CUDA-compatible GPU** - Required
- **BERTopic** - Topic modeling library
- **OCTIS** - Optimization framework (for hyperparameter search)

## Files

### Core Files

- **`main.py`** - CLI entry point with commands: `train`, `retrain`, `optimize`
- **`bertopic_runner.py`** - OCTIS integration for hyperparameter optimization
- **`bertopic_octis_model.py`** - BERTopic-OCTIS integration wrapper (moved from `src/common/` as it's stage-specific)
- **`test_octis_pipeline.py`** - Test script to validate pipeline without optimization

### Utilities

- **`convert_topics.py`** - Convert NumPy topic arrays to JSON (legacy utility)
- **`memory_utils.py`** - GPU memory monitoring and management utilities

### File Status

| File | Status | GPU | Notes |
|------|-------|-----|-------|
| `main.py` | ✅ Active | N/A | Entry point |
| `bertopic_runner.py` | ✅ Active | ✅ RAPIDS | OCTIS integration |
| `test_octis_pipeline.py` | ✅ Active | ✅ RAPIDS | Testing |
| `bertopic_octis_model.py` | ✅ Active | ✅ RAPIDS | BERTopic-OCTIS wrapper |
| `convert_topics.py` | ⚠️ Legacy | N/A | Utility script |

## Usage

### Test Pipeline (Recommended First)

```bash
# Test with subset (10K rows) - fast validation
python -m src.stage03_modeling.test_octis_pipeline --subset

# Test with full dataset
python -m src.stage03_modeling.test_octis_pipeline --full
```

### Train Models

```bash
# Train BERTopic models (OCTIS integration)
python -m src.stage03_modeling.main train --config configs/bertopic.yaml
```

### Optimize Models

```bash
# Run hyperparameter optimization with OCTIS
python -m src.stage03_modeling.main optimize --config configs/octis.yaml
```

### Retrain Models

```bash
# Retrain from hyperparameter tables (implementation pending)
python -m src.stage03_modeling.main retrain \
  --dataset_csv data/processed/chapters.csv \
  --out_dir models/retrained/
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

- **`configs/bertopic.yaml`** - BERTopic model parameters
- **`configs/octis.yaml`** - OCTIS optimization settings
- **`configs/paths.yaml`** - Data and output paths

## Outputs

- **`models/`** - Trained BERTopic models
- **`results/topics/by_book.csv`** - Topic probabilities per book
- **`results/topics/top_models/*.json`** - Topic word lists

## Notes

- All models use GPU acceleration via RAPIDS
- Embeddings are cached to avoid recomputation
- Models are saved with full hyperparameter configuration

