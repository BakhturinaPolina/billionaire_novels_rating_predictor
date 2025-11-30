# Romantic Novels NLP Research Project

A comprehensive NLP research pipeline for analyzing romantic novels using topic modeling, statistical analysis, and reader appreciation patterns.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Data](#data)
- [Pipeline Stages](#pipeline-stages)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Overview

This project implements a seven-stage research pipeline for analyzing romantic novels through:

- **Topic Modeling**: BERTopic-based topic extraction with multiple embedding models
- **Hyperparameter Optimization**: Bayesian optimization using OCTIS
- **Statistical Analysis**: Reader appreciation pattern analysis with FDR correction
- **Goodreads Integration**: Analysis of book ratings and metadata

The pipeline processes novel texts, extracts topics, and correlates them with reader ratings to identify patterns in reader appreciation.

**Research Question**: Which thematic patterns differentiate highly-rated romance novels from lower-rated ones, and how do these patterns relate to reader appreciation metrics?

## Project Structure

```
romantic_novels_project_code/
├── src/                          # Source code organized by pipeline stage
│   ├── common/                   # Shared utilities (config, GPU, logging)
│   ├── stage01_ingestion/        # Data loading (Goodreads, BookNLP)
│   ├── stage02_preprocessing/    # Text cleaning, tokenization
│   ├── stage03_modeling/         # BERTopic training & optimization
│   ├── stage04_experiments/      # Hyperparameter optimization
│   ├── stage05_selection/        # Pareto-efficient model selection
│   ├── stage05_retraining/       # Retrain top models
│   ├── stage06_BERTopic_topics_exploration/  # Topic exploration & evaluation
│   ├── stage06_labeling/         # Topic labeling and composite building
│   └── stage07_analysis/         # Statistical analysis & visualization
├── configs/                      # YAML configuration files
├── notebooks/                    # Jupyter notebooks by stage
├── data/                         # Data directories (see Data section)
├── results/                      # Pipeline outputs
├── reports/                      # Documentation and findings
└── scripts/                      # Utility scripts
```

For detailed structure information, see [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md).

## Installation

### Prerequisites

- **Python 3.12+**
- **CUDA-compatible GPU** (required for GPU acceleration)
- **CUDA 12.x drivers** (for RAPIDS GPU libraries)

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd romantic_novels_project_code
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install SpaCy model:**
```bash
python -m spacy download en_core_web_sm
```

5. **Verify GPU setup (Optional):**
```bash
python -m src.common.check_gpu_setup
```

## Quick Start

1. **Configure paths:**
   - Edit `configs/paths.yaml` to set your data directories
   - All paths are relative to the project root

2. **Run a single stage:**
```bash
# Using Makefile
make stage01  # Data ingestion
make stage03  # Model training
make stage07  # Analysis

# Or directly with Python
python -m src.stage03_modeling.main train --config configs/bertopic.yaml
```

3. **Run the full pipeline:**
```bash
make all
```

## Usage

### Stage-by-Stage Execution

Each stage can be run independently:

```bash
# Stage 01: Ingestion
python -m src.stage01_ingestion.main --config configs/paths.yaml

# Stage 02: Preprocessing
python -m src.stage02_preprocessing.main --config configs/paths.yaml

# Stage 03: Modeling
python -m src.stage03_modeling.main train --config configs/bertopic.yaml
# Retrain specific models
python -m src.stage03_modeling.main retrain --dataset_csv data/processed/chapters.csv --out_dir models/

# Stage 04: Experiments (Hyperparameter Optimization)
python -m src.stage04_experiments.main --config configs/octis.yaml

# Stage 05: Selection (Pareto Analysis)
python -m src.stage05_selection.main --config configs/selection.yaml

# Stage 05: Retraining (Retrain Top Models)
python -m src.stage05_retraining.main retrain --top_n 4

# Stage 06: Topic Exploration (Evaluate Retrained Models)
python -m src.stage06_BERTopic_topics_exploration.explore_retrained_model \
  --embedding-model paraphrase-MiniLM-L6-v2 \
  --pareto-rank 1 \
  --save-topics \
  --output-dir results/stage06

# Stage 06: Labeling (Generate Topic Labels with Mistral-7B-Instruct)
python -m src.stage06_labeling.main \
  --embedding-model paraphrase-MiniLM-L6-v2 \
  --pareto-rank 1 \
  --topics-json results/stage06/topics_all_representations_paraphrase-MiniLM-L6-v2.json

# Stage 07: Analysis
python -m src.stage07_analysis.main --config configs/scoring.yaml
```

### Configuration

All configuration is done via YAML files in `configs/`:

- **`paths.yaml`**: Data and output directory paths
- **`bertopic.yaml`**: BERTopic model parameters
- **`octis.yaml`**: Hyperparameter search space
- **`selection.yaml`**: Model selection criteria (min_nr_topics >= 200)
- **`scoring.yaml`**: Statistical analysis settings
- **`labeling.yaml`**: Topic labeling configuration

See individual config files for detailed parameter descriptions.

## Data

### Data Directory Structure

- **`data/raw/`**: Raw input data (excluded from git)
  - EPUB files and full text novels
- **`data/interim/`**: Intermediate processing outputs (excluded from git)
  - BookNLP outputs
  - OCTIS datasets
- **`data/processed/`**: Final processed data
  - `chapters.csv`: Processed novel chapters with sentences (~707K rows)
  - `chapters_subset_10000.csv`: Subset for testing (10K rows)
  - `goodreads.csv`: Cleaned Goodreads dataset with ratings
  - `custom_stoplist.txt`: Custom stopwords (character names)

### Data Exclusion Policy

The following are excluded from git via `.gitignore`:
- `data/raw/`
- `data/interim/`
- Large output files (`.npz`, large `.csv`)
- Model files (`.pt`, `.pkl`, `.h5`)

## Pipeline Stages

### Stage 01: Ingestion
Load raw texts, Goodreads data, and handle BookNLP I/O operations.

### Stage 02: Preprocessing
Text cleaning, tokenization, lemmatization, and custom stoplist building.

### Stage 03: Modeling
BERTopic model training with various embedding models. Supports retraining from topic tables with coherence priority.

### Stage 04: Experiments
Bayesian hyperparameter optimization using OCTIS. Optimizes BERTopic parameters across multiple embedding models.

### Stage 05: Selection
Pareto efficiency analysis with constraint enforcement (minimum 200 topics). Selects optimal models based on coherence and diversity metrics. Also handles retraining of top models.

### Stage 06: Topic Exploration & Labeling
**Topic Exploration**: Interactive tooling for inspecting retrained BERTopic models. Loads models (pickle wrapper or native safetensors), attaches multiple representations (Main, KeyBERT, POS, MMR), computes coherence (c_v) and diversity metrics, and extracts all topics with all representations for close reading evaluation.

**Topic Labeling**: Automated generation of human-readable topic labels using Mistral-7B-Instruct-v0.2 with 4-bit quantization. Extracts POS representation keywords, applies MMR reranking for diversity, and integrates labels back into BERTopic models. Also includes semi-supervised composite building to create Appreciation Pattern (AP) composites from topic groups.

### Stage 07: Analysis
Goodreads scoring/stratification, statistical analysis, and FDR correction. Generates visualizations and statistical reports.

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make changes following PEP 8
4. Test your changes
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{romantic_novels_nlp,
  title = {Modern Romantic Novels — Themes × Popularity: A Mixed-Methods Computational Analysis},
  author = {[Your Name/Institution]},
  year = {2025},
  url = {[Repository URL]}
}
```

## Acknowledgments

- **BERTopic** (Grootendorst, 2022) for topic modeling
- **OCTIS** (Terragni et al., 2021) for hyperparameter optimization
- **RAPIDS cuML** for GPU acceleration
- **SentenceTransformers** for embeddings
- **Mistral-7B-Instruct** (Jiang et al., 2023) for automated topic labeling
- **bitsandbytes** for efficient model quantization

---

**Note**: This is a research project. Some pipeline stages may be under active development. Check individual stage documentation for current implementation status.
