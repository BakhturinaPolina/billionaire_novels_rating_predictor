# Romantic Novels NLP Research Pipeline

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A computational research pipeline for analyzing romantic novels using neural topic modeling (BERTopic), LLM-based labeling, and statistical correlation with reader appreciation metrics from Goodreads.

## Research Question

> Which thematic patterns differentiate highly-rated romance novels from lower-rated ones, and how do these patterns relate to reader appreciation metrics?

## Key Features

- **Neural Topic Modeling**: BERTopic with OCTIS Bayesian hyperparameter optimization
- **GPU Acceleration**: Mandatory RAPIDS cuML (CUDA 12.x) for UMAP and HDBSCAN
- **LLM Topic Labeling**: Automated labeling via OpenRouter API (Mistral-Nemo)
- **Theory-Aligned Categories**: Zero-shot classification to romance taxonomy and Radway narrative functions
- **Statistical Analysis**: Correlation with Goodreads ratings using bootstrap inference

## Installation

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU with CUDA 12.x drivers
- ~6GB VRAM (for quantized LLM inference)

### Setup

```bash
git clone <repository-url>
cd billionaire_novels_rating_predictor

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Verify GPU setup:
```bash
python -m src.common.check_gpu_setup
```

## Quick Start

```bash
# Run individual stages
make stage01  # Data ingestion
make stage02  # Preprocessing
make stage03  # BERTopic training with OCTIS optimization

# Run full pipeline
make all
```

Or run stages directly:
```bash
python -m src.stage01_ingestion.main --config configs/paths.yaml
python -m src.stage03_modeling.main train --config configs/bertopic.yaml
```

## Documentation

| Resource | Description |
|----------|-------------|
| [SCIENTIFIC_README.md](SCIENTIFIC_README.md) | Full methodology, hypotheses, and results |
| [reports/](reports/) | Stage-by-stage technical reports and findings |
| [configs/](configs/) | YAML configuration files |

## Project Structure

```
billionaire_novels_rating_predictor/
├── src/                    # Source code (stages 01-10)
├── configs/                # YAML configuration files
├── notebooks/              # Jupyter notebooks by stage
├── data/                   # Raw, interim, processed data
├── results/                # Pipeline outputs
├── reports/                # Technical reports and findings
├── models/                 # Trained BERTopic models
└── scripts/                # Utility scripts
```

## Pipeline Overview

| Stage | Name | Description |
|-------|------|-------------|
| 01 | Ingestion | Load novels and Goodreads metadata |
| 02 | Preprocessing | Text cleaning, tokenization, character name removal |
| 03 | Modeling | BERTopic training with OCTIS optimization |
| 04 | Selection | Pareto-efficient model selection |
| 05 | Retraining | Retrain top models |
| 06 | Topic Exploration | Multi-representation topic analysis |
| 07 | Topic Quality | Noisy topic detection |
| 08 | LLM Labeling | Automated topic labeling |
| 09 | Category Mapping | Theory-aligned taxonomy classification |
| 10 | Correlation Analysis | Statistical hypothesis testing |

See [SCIENTIFIC_README.md](SCIENTIFIC_README.md) for detailed stage documentation.

## Configuration

All settings are in `configs/`:

- `paths.yaml` — Data directories
- `bertopic.yaml` — BERTopic parameters
- `octis.yaml` — Hyperparameter search space
- `selection.yaml` — Model selection criteria
- `labeling.yaml` — LLM labeling settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow PEP 8 style guidelines
4. Submit a pull request

## License

[MIT License](LICENSE)

## Citation

```bibtex
@software{romantic_novels_nlp,
  title = {Romantic Novels NLP Research Pipeline},
  author = {Polina},
  year = {2025},
  url = {<repository-url>}
}
```

## Acknowledgments

- [BERTopic](https://github.com/MaartenGr/BERTopic) — Topic modeling
- [OCTIS](https://github.com/MIND-Lab/OCTIS) — Hyperparameter optimization
- [RAPIDS cuML](https://github.com/rapidsai/cuml) — GPU acceleration
- [SentenceTransformers](https://www.sbert.net/) — Embeddings
- [Mistral](https://mistral.ai/) — LLM labeling
