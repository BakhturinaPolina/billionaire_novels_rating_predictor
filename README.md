# Modern Romantic Novels — Themes × Popularity

**A Mixed-Methods Computational Analysis of Romance Novel Themes and Reader Appreciation**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This project quantifies and interprets which themes are more frequent in Top / Medium / Trash romance novels and how those themes relate to reader appreciation. We combine topic-model outputs (from the novels) with a theory-driven taxonomy and advanced, light-weight ML to map topics → categories, compute indices, and test hypotheses.

**Research Question**: Which thematic patterns differentiate highly-rated romance novels from lower-rated ones, and how do these patterns relate to reader appreciation metrics?

## Quick Start

### Prerequisites

- **Python 3.12+**
- **CUDA-compatible GPU** (required for Stages 03 and 05)
- **RAPIDS cuML** (CUDA 12.x) - for GPU-accelerated topic modeling
- See `requirements.txt` for full dependencies

### Installation

```bash
# Clone repository
git clone <repository-url>
cd romantic_novels_project_code

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU setup
python -m src.common.check_gpu_setup
```

### Running the Pipeline

The project follows a 7-stage pipeline:

```bash
# Stage 03: Train BERTopic models with hyperparameter optimization
python -m src.stage03_modeling.main optimize --config configs/octis.yaml

# Stage 04: Select Pareto-efficient models
python -m src.stage04_selection.main analyze --config configs/selection.yaml

# Stage 05: Retrain top models
python -m src.stage05_retraining.main retrain --top_n 4

# Stage 06: Build thematic composites
python -m src.stage06_labeling.composites

# Stage 07: Statistical analysis
python -m src.stage07_analysis.main --config configs/scoring.yaml
```

## Project Structure

```
romantic_novels_project_code/
├── src/                          # Source code
│   ├── common/                   # Shared utilities (config, GPU, logging)
│   ├── stage01_ingestion/        # Data loading (Goodreads, BookNLP)
│   ├── stage02_preprocessing/     # Text cleaning, tokenization
│   ├── stage03_modeling/         # BERTopic training & optimization
│   ├── stage04_selection/        # Pareto-efficient model selection
│   ├── stage05_retraining/       # Retrain top models
│   ├── stage06_labeling/         # Thematic composite building
│   └── stage07_analysis/         # Statistical analysis & hypothesis testing
├── notebooks/                    # Jupyter notebooks for exploration
│   ├── 03_modeling/              # Modeling experiments
│   ├── 04_selection/             # Selection analysis
│   └── 07_analysis/              # Statistical analysis
├── data/                         # Data directory
│   ├── raw/                      # Raw data files
│   ├── interim/                  # Intermediate processed data
│   └── processed/                # Final processed data
├── models/                       # Trained models
├── results/                      # Analysis results
│   ├── pareto/                   # Pareto analysis outputs
│   ├── topics/                   # Topic modeling outputs
│   └── figures/                  # Visualizations
├── configs/                      # Configuration files
└── logs/                         # Execution logs
```

## Research Pipeline

### Stage 01: Ingestion
Load raw text files, Goodreads metadata, and BookNLP outputs.

### Stage 02: Preprocessing
Text cleaning, tokenization, lemmatization, custom stoplist building.

### Stage 03: Modeling ⚡
**BERTopic model training** with GPU acceleration (RAPIDS cuML) and **OCTIS hyperparameter optimization**. Evaluates 300+ model configurations across 6 embedding models.

**Key Technologies**: BERTopic, OCTIS, RAPIDS cuML, SentenceTransformers

### Stage 04: Selection 📊
**Pareto-efficient model selection** using multi-objective optimization. Balances coherence and topic diversity with two weighting strategies (equal weights, coherence priority).

**Outputs**: Top-performing models, visualizations, correlation analysis

### Stage 05: Retraining 🔄
Retrain top Pareto-efficient models with optimal hyperparameters.

### Stage 06: Labeling 🏷️
Map topics to **16 thematic composites** (A-P) using semi-supervised approach:
- A: Reassurance/Commitment
- B: Mutual Intimacy
- C: Explicit Eroticism
- D: Power/Wealth/Luxury
- E: Coercion/Brutality/Danger
- F: Angst/Negative Affect
- G: Courtship Rituals/Gifts
- H: Domestic Nesting
- I: Humor/Lightness
- J: Social Support/Kin
- K: Professional Intrusion
- L: Vices/Addictions
- M: Health/Recovery/Growth
- N: Separation/Reunion
- O: Aesthetics/Appearance
- P: Tech/Media Presence

### Stage 07: Analysis 📈
**Statistical analysis** combining topic probabilities with Goodreads metadata:
- Popularity stratification (Top/Medium/Trash)
- Hypothesis testing (H1-H6)
- Index computation (Love-over-Sex, HEA Index, etc.)
- FDR correction (Benjamini-Hochberg)

## Key Features

- **GPU-Accelerated**: Mandatory RAPIDS cuML for fast topic modeling
- **Multi-Objective Optimization**: Pareto efficiency analysis for model selection
- **Theory-Driven Taxonomy**: 16 thematic composites based on romance novel theory
- **Statistical Rigor**: Comprehensive hypothesis testing with FDR correction
- **Reproducible**: Configuration-driven pipeline with clear data contracts

## Dataset

The dataset includes **105 standalone billionaire romance novels** by 35 different authors, selected from curated lists such as "100 Best Billionaire Romance Books of All Time". Each novel contains at least 100,000 words, resulting in **680,822 sentences** organized hierarchically by author, book, chapter, and sentence.

## Research Hypotheses

- **H1**: Higher Love-over-Sex and HEA Index in Top vs Trash
- **H2**: Explicitness Ratio higher in Trash
- **H3**: Luxury Saturation predicts appreciation only with high (commitment_hea + tenderness_emotion) (interaction)
- **H4**: Protective–Jealousy Delta higher in Top
- **H5**: Dark-vs-Tender lower (i.e., more tender) in Top
- **H6**: Time-course: commitment_hea and apology_repair rise from begin→end; miscommunication and neg_affect fall

## Documentation

- **[SCIENTIFIC_README.md](SCIENTIFIC_README.md)** - Detailed methodology, research questions, and statistical analysis plan
- **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** - Technical methodology details
- **[docs/DATA_CONTRACTS.md](docs/DATA_CONTRACTS.md)** - Input/output data specifications
- **[docs/INDICES.md](docs/INDICES.md)** - Derived index definitions and formulas
- Stage-specific READMEs in `src/stage*/README.md`

## Citation

If you use this code or findings, please cite:

```bibtex
@software{romantic_novels_themes_popularity,
  title = {Modern Romantic Novels — Themes × Popularity: A Mixed-Methods Computational Analysis},
  author = {[Your Name]},
  year = {2025},
  url = {[Repository URL]}
}
```

## License

See [LICENSE](LICENSE) file for details.

## Contributing

This is a research project. For questions or collaboration inquiries, please open an issue.

## Acknowledgments

- **BERTopic** (Grootendorst, 2022) for topic modeling
- **OCTIS** (Terragni et al., 2021) for hyperparameter optimization
- **RAPIDS cuML** for GPU acceleration
- **SentenceTransformers** for embeddings

---

**Note**: Stages 01, 02, and 06 (main.py) are placeholder implementations showing planned architecture. Core functionality is in specialized modules (e.g., `composites.py` in Stage 06).

