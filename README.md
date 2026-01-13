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

This project implements a ten-stage research pipeline for analyzing romantic novels through:

- **Topic Modeling**: BERTopic-based topic extraction with multiple embedding models
- **Hyperparameter Optimization**: Bayesian optimization using OCTIS
- **Statistical Analysis**: Reader appreciation pattern analysis with FDR correction
- **Goodreads Integration**: Analysis of book ratings and metadata

The pipeline processes novel texts, extracts topics, and correlates them with reader ratings to identify patterns in reader appreciation.

**Research Question**: Which thematic patterns differentiate highly-rated romance novels from lower-rated ones, and how do these patterns relate to reader appreciation metrics?

## Project Structure

```
billionaire_novels_rating_predictor/
├── src/                          # Source code organized by pipeline stage
│   ├── common/                   # Shared utilities (config, GPU, logging)
│   ├── stage01_ingestion/        # Data loading (Goodreads, BookNLP)
│   ├── stage02_preprocessing/    # Text cleaning, tokenization
│   ├── stage03_modeling/         # BERTopic training & optimization
│   ├── stage04_selection/        # Pareto-efficient model selection
│   ├── stage05_retraining/       # Retrain top models
│   ├── stage06_topic_exploration/ # Topic exploration & evaluation
│   ├── stage07_topic_quality/    # Topic quality analysis & noise detection
│   ├── stage08_llm_labeling/     # Topic labeling with LLM
│   ├── stage09_category_mapping/ # Category mapping & theory alignment
│   └── stage10_correlation_analysis/ # Statistical analysis & visualization
├── configs/                      # YAML configuration files
├── notebooks/                    # Jupyter notebooks by stage
│   ├── 01_ingestion/             # Data ingestion notebooks
│   ├── 02_preprocessing/         # Preprocessing notebooks
│   ├── 04_selection/             # Model selection analysis
│   ├── 05_retraining/            # Retraining notebooks
│   ├── 06_labeling/              # Labeling and taxonomy analysis
│   └── 07_analysis/              # Final analysis notebooks
│       ├── 01_topic_analysis/    # Individual topic distributions
│       ├── 02_taxonomy_group_analysis/ # Taxonomy group comparisons
│       ├── 03_composite_index_construction/ # Theory-aligned indices
│       └── 04_hypothesis_testing/ # Hypothesis testing (H1-H6)
├── data/                         # Data directories (see Data section)
│   ├── raw/                      # Raw input data (excluded from git)
│   ├── interim/                  # Intermediate processing outputs
│   └── processed/                # Final processed data
├── results/                      # Pipeline outputs by stage
├── reports/                      # Documentation and findings
│   ├── 01_stage_reports/         # Technical reports by pipeline stage
│   └── 02_findings/              # Research findings and results
├── models/                       # Trained BERTopic models
├── scripts/                      # Utility scripts
├── cache/                        # Cached models and embeddings
└── logs/                         # Pipeline execution logs
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
make stage10  # Correlation analysis

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

# Stage 04: Selection (Pareto Analysis)
python -m src.stage04_selection.main analyze --config configs/selection.yaml

# Stage 05: Retraining (Retrain Top Models)
python -m src.stage05_retraining.main retrain --top_n 4

# Stage 06: Topic Exploration (Evaluate Retrained Models)
python -m src.stage06_topic_exploration.explore_retrained_model \
  --embedding-model paraphrase-MiniLM-L6-v2 \
  --pareto-rank 1 \
  --save-topics \
  --output-dir results/stage06_topic_exploration

# Stage 07: Topic Quality Analysis (Noisy Topic Detection)
python -m src.stage07_topic_quality.main \
  --embedding-model paraphrase-MiniLM-L6-v2 \
  --pareto-rank 1 \
  --output-dir results/stage07_topic_quality

# Stage 08: LLM Labeling (Two-Step Process)

# Step 1: Generate Topic Labels with OpenRouter API
python -m src.stage08_llm_labeling.openrouter_experiments.core.main_openrouter \
  --embedding-model paraphrase-MiniLM-L6-v2 \
  --pareto-rank 1 \
  --topics-json results/stage06_topic_exploration/topics_all_representations_paraphrase-MiniLM-L6-v2.json

# Step 2: Map Labels to Theory-Aligned Categories
python -m src.stage09_category_mapping.stage1_natural_clusters.prepare_sentence_dataframe \
  --chapters data/processed/chapters.csv \
  --goodreads data/processed/goodreads.csv \
  --output data/processed/sentence_df_with_ratings.parquet

# Stage 09: Category Mapping
# See src/stage09_category_mapping/README.md for detailed usage

# Stage 10: Correlation Analysis
# Step 1: Generate topic probabilities (production)
python src/stage10_correlation_analysis/data_preparation/03_generate_topic_probabilities_final.py \
    --sentence-df data/processed/sentence_df_with_topics.parquet \
    --model-path models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_radway_mappings \
    --output-dir results/stage10_correlation_analysis/data_preparation \
    --book-id-source goodreads \
    --goodreads-id-col ID

# Step 2: Generate tertile probabilities (optional, for narrative arc analysis)
python src/stage10_correlation_analysis/data_preparation/04_generate_tertile_topic_probs_patched_v3.py \
    --sentence-df data/processed/sentence_df_with_topics.parquet \
    --model-path models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_radway_mappings \
    --output-dir results/stage10_correlation_analysis/data_preparation \
    --book-id-source goodreads \
    --goodreads-id-col ID

# Step 3: Extract topic metadata and validate
python src/stage10_correlation_analysis/data_preparation/01_data_validation_extraction.py \
    --output-dir results/stage10_correlation_analysis/data_preparation/taxonomy_radway_eda

# Step 4: Aggregate to book-level and compute indices
python src/stage10_correlation_analysis/data_preparation/02_book_aggregation.py \
    --topic-lookup results/stage10_correlation_analysis/data_preparation/taxonomy_radway_eda/topic_lookup.parquet \
    --output-dir results/stage10_correlation_analysis/data_preparation/book_features
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
Handles initial data collection and consolidation, serving as the foundational data loading step. Key features:

**Data Sources:**
- **Raw text files**: Loads from `data/raw/Billionaire_Full_Novels_TXT/` (TXT or EPUB formats)
- **Goodreads metadata**: Integrates ratings, review counts, and publication information from `data/processed/goodreads.csv`
- **BookNLP outputs** (optional): Processes character entity files for character name extraction

**Key Features:**
- **File loading**: Recursive scanning, format detection (TXT, EPUB), encoding handling (UTF-8 with mojibake correction)
- **Metadata integration**: Fuzzy matching (threshold: 0.85) to handle author/title format differences between text files and Goodreads
- **Data validation**: Early validation of data integrity (file existence, format consistency, encoding issues)
- **Error handling**: Robust error handling for missing files, encoding issues, metadata mismatches, and corrupted files
- **Progress tracking**: Logging for large-scale dataset processing (105 novels)

**Output**: Processed text data with merged metadata, maintaining hierarchical organization (Author → Book → Chapter) for downstream processing.

**Statistics**: 
- 97-98 books with complete Goodreads metadata
- Rating distribution: Mean 3.99, Std 0.21, Range 3.26-4.42
- All books have ≥100 ratings (minimum: 146, mean: 65,849)

See `reports/01_stage_reports/stage01_ingestion/stage01_data_ingestion_methodology.md` for detailed documentation.

### Stage 02: Preprocessing
Transforms raw text files into clean, structured format suitable for neural topic modeling. Key features:

**Text Cleaning Pipeline:**
- **Encoding fixes**: Mojibake correction (e.g., `â€™` → `'`, `â€œ` → `"`), Unicode normalization (NFKD)
- **Whitespace normalization**: Converts newlines to spaces, collapses multiple spaces/tabs, strips leading/trailing whitespace
- **Case normalization**: Converts all text to lowercase (after sentence segmentation)
- **Artifact removal**: Removes headers/footers, page numbers, formatting artifacts

**Sentence Segmentation:**
- Preserves sentence boundaries for BERTopic (operates on sentence-level embeddings)
- Handles edge cases: abbreviations (Mr., Dr., Inc.), decimal numbers, ellipses
- Maintains context: preserves sentence order and chapter structure

**Tokenization and Lemmatization:**
- **Tokenization**: Word tokenization with punctuation handling, hyphen handling
- **POS tagging**: Context-aware part-of-speech tagging for accurate lemmatization
- **Lemmatization**: Converts words to root forms (e.g., "running" → "run", "better" → "good")
- **Implementation**: Uses spaCy for sentence segmentation, tokenization, POS tagging, and lemmatization

**Custom Stoplist Building:**
- **Components**: Standard English stopwords (318) + Character names (4,444) = 4,762 total stopwords
- **Character name processing**: 
  - Processes 7,525 character name lines from romance novel texts
  - Extracts 4,497 unique name tokens (4,444 after overlap removal)
  - Cleaning steps: removes prefixes, numbers, punctuation; filters descriptive patterns
  - Multi-word name extraction: splits "Alex Crane" → extracts both "alex" and "crane"
- **Impact**: 14x expansion of stopwords (93% character names, 7% standard English) ensures topics focus on thematic content rather than character co-occurrence patterns

**Output Format:**
- **File**: `data/processed/chapters.csv`
- **Structure**: Columns: `Author`, `Book Title`, `Chapter`, `Sentence`
- **Statistics**: 680,822 sentences from 105 novels by 35 authors
- **Format**: One sentence per row, lowercase, normalized, cleaned, stopwords removed, lemmatized

**Precision Trade-offs:**
The preprocessing retains some non-name words (estimated 1-2% of tokens) but prioritizes coverage to ensure comprehensive character name exclusion. This approach successfully captures the vast majority of character references while maintaining reproducibility.

See `reports/01_stage_reports/stage02_preprocessing/stage02_preprocessing_methodology.md` for detailed documentation.

### Stage 03: Modeling
BERTopic model training with OCTIS hyperparameter optimization. Key features:
- **GPU-accelerated** using RAPIDS (cuML) - mandatory, no CPU fallback
- **Character name exclusion**: 4,444 character names added to stopwords (93% of expanded list) to improve topic interpretability by focusing on thematic content rather than character co-occurrence
- **Embedding caching**: Reuses embeddings to avoid recomputation across iterations
- **Multiple embedding models**: Evaluates 6 pre-trained SentenceTransformer models
- **Over 300 configurations** analyzed through Bayesian optimization

### Stage 04: Selection
Pareto efficiency analysis to identify optimal models balancing coherence and diversity. Features:
- **Two-stage outlier filtering**: Removes failed runs and statistical outliers (z-score + IQR methods)
- **Domain-specific filtering**: Removes models with artificially high diversity (>0.9) from too few topics
- **Final results**: 4 Pareto-efficient models identified (down from 12 before filtering)
- **Top performer**: `paraphrase-mpnet-base-v2`, iteration 0 (coherence: 0.463, diversity: 0.82)
- **Hyperparameter correlation analysis**: Statistical analysis identifies which hyperparameters most influence performance (UMAP parameters most influential)
- **Weighting schemes**: Equal weights (50/50) and coherence priority (70/30)

### Stage 05: Retraining
Retrains top N Pareto-efficient models from Stage 04 with their exact hyperparameters for final deployment. Features:
- **Direct retraining**: No OCTIS optimization - uses hyperparameters directly from Pareto CSV
- **Multiple output formats**: Pickle (full wrapper), BERTopic native format (safetensors), and metadata JSON
- **Independent model training**: Failures in one model don't stop others
- **Embedding caching**: Reuses embeddings from Stage 03 to avoid recomputation
- **Character name exclusion**: Same preprocessing pipeline as Stage 03
- **GPU acceleration**: Uses RAPIDS (cuML) - same as Stage 03
- **Comprehensive metadata**: Each model includes hyperparameters, evaluation scores, topic counts, and timestamps

### Stage 06: Topic Exploration
Interactive tooling for inspecting retrained BERTopic models. Loads models (pickle wrapper or native safetensors), attaches multiple representations (Main, KeyBERT, POS, MMR), computes coherence (c_v) and diversity metrics, and extracts all topics with all representations for close reading evaluation.

**Key Features:**
- **Multiple Representations**: Enriches topics with four complementary representations (Main c-TF-IDF, KeyBERT semantic similarity, POS content words, MMR diversity)
- **Quantitative Evaluation**: Computes c_v coherence and topic diversity metrics for each representation
- **Results**: For the selected model (paraphrase-MiniLM-L6-v2, 368 topics), Main achieves highest coherence (0.404), MMR achieves highest diversity (0.756), and POS balances both (0.315 coherence, 0.692 diversity)
- **Representation Selection**: POS recommended for LLM labeling, MMR for exploratory analysis, Main for statistical validation

### Stage 07: Topic Quality Analysis
Exploratory data analysis to identify candidate noisy topics before LLM labeling. Computes topic size statistics, POS representation statistics, and per-topic POS coherence. Flags candidate noisy topics based on configurable thresholds.

**Quality Metrics:**
- **Topic size**: Number of documents assigned to each topic
- **POS representation statistics**: Count of POS-filtered keywords per topic
- **Per-topic POS coherence**: c_v coherence computed on POS-filtered keywords using the same gensim dictionary as training

**Noise Detection Criteria:**
- Topics with few POS words (< 3): Indicates topics lacking interpretable keywords
- Topics with low or missing POS coherence (< 0.0): Suggests semantically incoherent topics
- Topics below minimum size threshold (< 30 documents): May represent outliers or noise

**Results for Selected Model** (paraphrase-MiniLM-L6-v2, 368 topics):
- **Candidate noisy topics**: 13 (3.5% of all topics)
- **Topics with POS words < 10**: 20 (5.4% of all topics)
- **Topics with valid coherence scores**: 361 (98.1% of all topics)
- **Quality threshold**: 94.6-96.5% of topics meet quality thresholds

Noisy topics are labeled with inspection tags (e.g., `[NOISE_CANDIDATE:few_pos<3]`) and saved to both wrapper pickle and native BERTopic model formats. Quality tables are saved to CSV for review. See `src/stage07_topic_quality/` for details.

### Stage 08: LLM Labeling
Automated generation of human-readable topic labels using either:
- **OpenRouter API** (recommended): Cloud-based labeling with `mistralai/Mistral-Nemo-Instruct-2407` via OpenRouter API. No local GPU required. Primary model selected for research reliability (low hallucination, format compliance, academic tone). Alternative literary models (e.g., `thedrummer/cydonia-24b-v4.1`, `thedrummer/anubis-70b-v1.1`) evaluated for enhanced genre awareness. Cost: ~$0.00005 per topic (~$0.018 for 368 topics). See `src/stage08_llm_labeling/openrouter_experiments/`.
- **Local Mistral-7B-Instruct**: Local inference with 4-bit quantization. Extracts POS representation keywords, applies MMR reranking for diversity, and integrates labels back into BERTopic models.

**Theoretical Foundation**: LLMs combine semantic understanding, domain knowledge, consistency, and scalability to solve the challenge of interpreting topic model keyword lists at scale. Zero-shot classification enables mapping to predefined theoretical frameworks without training data.

**Key Features:**
- **Romance-Aware Prompting**: Domain-specific system prompt designed for modern romantic and erotic fiction with anti-hallucination constraints
- **Representative Snippets**: Uses actual document snippets (3-6 sentences, 200 chars max) from BERTopic's representative documents as primary evidence for label generation. Enables fine distinctions (rough vs gentle kisses, specific sexual acts, scene-level context)
- **Improved Prompts** (`--use-improved-prompts`): Structured JSON output with `label`, `scene_summary`, `primary_categories`, `secondary_categories`, `is_noise`, and `rationale` fields
- **Model Evaluation**: Labels evaluated on specificity, genre awareness, discriminative power, scene summary quality, category accuracy, and format compliance
- **Quality Assurance**: Model comparison on 30 topics shows 100% success rate for Nemo-Instruct (2.30 avg words, 0% keyword copying). Taxonomy coverage: 98.1% (361/368 topics)
- **Computational Infrastructure**: OpenRouter API with 4.0s rate limiting, temperature=0.35, streaming mode for memory efficiency, caching/resumption support
- **Integration**: All metadata (labels, taxonomy, Radway mappings) stored in BERTopic's `topic_metadata_` attribute for single source of truth

### Stage 09: Category Mapping
Three-stage zero-shot classification approach mapping topics to theory-aligned categories:

**Stage 1: Natural Clusters** (optional): Data-driven topic groupings using BERTopic's hierarchical topics

**Stage 2: Theory-Driven Taxonomy Classification** ✅ **Implemented**: Zero-shot classification to **Romance Corpus Topic Taxonomy** (8 groups, 30+ categories) using Mistral-Nemo via OpenRouter. Maps topics to hierarchical taxonomy nodes (e.g., "4.2" = Relationship Trajectory, Bonding). Coverage: 98.1% (361/368 topics). **Statistical Analysis Results**: Kruskal-Wallis tests identified 3 categories with statistically significant differences across rating classes: 5.3 (Community, Norms & Social Events, p=0.029, η²=0.070), 6.2 (Heroine's Work & Professional Identity, p=0.047, η²=0.057), 3.4 (Beliefs, Values & Moral Reflection, p=0.048, η²=0.048). Taxonomy mappings embedded in BERTopic model's `topic_metadata_` attribute (recommended model: `model_1_with_llm_labels_and_metadata_disambiguated.pkl`).

**Stage 3: Radway Narrative Functions** ✅ **Implemented**: Zero-shot classification to **Radway's 13 narrative functions** (Radway, 1984) organized into 3 phases: Phase I (Initial Conflict & Isolation, R1-R7), Phase II (Turning Point & Recognition, R8-R10), Phase III (Commitment & Restoration, R11-R13). Includes heuristic overrides for systematic errors. **Classification Results**: 272 topics mapped to Radway functions, 96 classified as "none". Distribution: Phase I (147 topics, 54.0%), Phase II (96 topics, 35.3%), Phase III (28 topics, 10.3%). All 13 functions represented. Key finding: Phase I dominates, indicating conflict and tension are central to romance narrative structure.

**Output**: Unified JSON structure combining taxonomy and Radway mappings, stored in BERTopic's `topic_metadata_` attribute. Book-level category proportions (`book_category_proportions.parquet`) and statistical analysis results available. Final goal: construct 19 theory-aligned composite categories (A-S) for hypothesis testing. See `src/stage09_category_mapping/` for details.

### Stage 10: Correlation Analysis
Comprehensive statistical analysis combining topic probabilities with Goodreads metadata. The pipeline consists of four sequential scripts plus analysis notebooks:

**Data Preparation Pipeline**:

1. **Script 03: Generate Topic Probabilities** - Generates normalized topic probabilities at book and chapter levels
   ```bash
   python src/stage10_correlation_analysis/data_preparation/03_generate_topic_probabilities_final.py \
       --sentence-df data/processed/sentence_df_with_topics.parquet \
       --model-path models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_radway_mappings \
       --output-dir results/stage10_correlation_analysis/00_data_preparation \
       --book-id-source goodreads \
       --goodreads-id-col ID \
       --exclude-book-ids notebooks/07_analysis/statistical_analysis/excluded_book_ids.csv
   ```
   - **Outputs**: `book_topic_probs.parquet` (33,856 rows: 92 books × 368 topics), `chapter_topic_probs.parquet` (1,089,280 rows: 2,960 chapters × 368 topics)
   -    - **Key Features**: Goodreads-first book IDs, cohort exclusion (5 books: 19561986, 19619918, 25781538, 52061964, 53491034), NaN replacement (critical fix: BERTopic transform can return NaN, replaced with 0.0 before aggregation), caching (~2 hours saved), probability normalization (0% NaN, sums to ~1.0 per book), comprehensive ID alignment diagnostics

2. **Script 04: Generate Tertile Probabilities** (optional) - Generates begin/middle/end tertile probabilities for narrative arc analysis
   ```bash
   python src/stage10_correlation_analysis/data_preparation/04_generate_tertile_topic_probs_patched_v3.py \
       --sentence-df data/processed/sentence_df_with_topics.parquet \
       --model-path models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_radway_mappings \
       --output-dir results/stage10_correlation_analysis/00_data_preparation
   ```
   - **Outputs**: `tertile_topic_probs.parquet` (~101,568 rows: 92 books × 3 segments × 368 topics)

3. **Script 01: Data Validation & Extraction** - Extracts topic metadata and validates data alignment
   ```bash
   python src/stage10_correlation_analysis/data_preparation/01_data_validation_extraction.py \
       --output-dir results/stage10_correlation_analysis/00_data_preparation/taxonomy_radway_eda
   ```
   - **Outputs**: `topic_lookup.parquet` (369 topics × 21 columns), `full_model_data.csv`, `summary_statistics.json`, `topics_needs_review.csv`, diagnostic reports (`id_alignment_report.csv`, `missing_books_in_outputs.csv`)
   - **Key Features**: Auto-detection of model/labels paths, fallback CSV support, QA checks (missing mappings, keyword quality, confidence distributions)

4. **Script 02: Book Aggregation** - Aggregates to book-level and computes derived indices
   ```bash
   python src/stage10_correlation_analysis/data_preparation/02_book_aggregation.py \
       --topic-lookup results/stage10_correlation_analysis/00_data_preparation/taxonomy_radway_eda/topic_lookup.parquet \
       --output-dir results/stage10_correlation_analysis/00_data_preparation/book_features
   ```
   - **Outputs**: `book_taxonomy_main_props_wide.parquet` (92 books × 27+ categories), `book_taxonomy_main_props_long.parquet`, `indices_book_taxonomy_proxy.parquet` (hypothesis-aligned indices: love_over_sex, hea_index, explicitness_ratio, dark_vs_tender, miscommunication_balance, luxury_saturation_proxy), `segment_taxonomy_main_props_long.parquet` (if segment data available)
   - **Key Features**: Auto-discovery of input files, ID normalization, multiple output formats (long/wide), segment-level support

**Analysis Notebooks** (`notebooks/07_analysis/`):

5. **01_topic_analysis**: Individual topic distributions across Top/Middle/Trash tiers
   - Topic-level leaderboards, effect sizes (Cliff's Delta), two-gate filtering rule
   - **Results**: 85 discriminative topics from 342 analyzed (368 total)
     - **Tier 1 (High Confidence)**: 8 topics (7 Top-associated, 1 Trash-associated)
     - **Tier 2 (Exploratory)**: 85 topics (70 Top-associated, 15 Trash-associated)
   - **Top-tier differentiation**: Psychological credibility scenes (fear admissions, emotional delusion) and embodied intimacy cues (affectionate stares, lip biting)
   - **Trash-tier differentiation**: Explicit sexual content and procedural/transition scenes
   - **Author dominance**: 30 topics show high author dominance (>50% from single author)
   - Outputs: `results/stage10_correlation_analysis/01_topic_analysis/`

6. **02_taxonomy_group_analysis**: Taxonomy group-level distribution comparisons
   - Dual normalization (absolute vs conditional shares), Gate 3 filtering
   - **Results**: Main group differences modest but interpretable (Top allocates more to relationship dynamics/social context, less to sexuality mass). Subgroup level shows sharper differentiation: **Beliefs, Values & Moral Reflection** (Top higher, δ≈+0.46, p≈0.008), **Negative Emotions & Distress** (Trash higher, δ≈-0.37). **Diversity finding**: Higher-tier books show greater thematic diversity (entropy: bad≈5.33 → good≈5.48, p≈0.019 adjusted≈0.077)
   - Outputs: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/`

7. **03_composite_index_construction**: Theory-aligned composite indices (A-S) construction
   - Measurement pipeline v5.6 with reliability diagnostics (alpha, omega, PCA, stability)
   - Outputs: `results/measurement_v5/`

8. **04_hypothesis_testing**: Hypothesis testing (H1-H6) using composite indices
   - Macro-axes analysis (5-axis model: status/dominance, payoff/safety, drama/obstacle, explicitness, negative affect), arc trajectory tests using exported deltas (end−begin, middle−begin), bootstrap inference (800 iterations, 95% CI, P(β>0) for directional effects), cross-validation (20 repeats of 5-fold CV)
   - **Methodology**: Two-channel analysis separating mass appeal (`log_rating_count`) from perceived quality (`rating_mean`), bootstrap-based effect size estimation with sign stability metrics
   - Outputs: `results/measurement_v5/bundle/inference_outputs/`

**Hypothesis Testing Results** (N = 92 books):

**Two-Channel Analysis**: Separates mass appeal (visibility) from perceived quality (ratings)

**Mass Appeal Predictors** (`log_rating_count`):
- **Top predictors** (β with 95% CI, P(β>0)):
  1. **R2_alpha_guarding**: β≈ +0.44, CI [+0.23, +0.60], P=1.00
  2. **Luxury/wealth (PC1)**: β≈ +0.37, CI [+0.20, +0.55], P=1.00
  3. **Emotional safety (PC1)**: β≈ +0.32, CI [+0.13, +0.50], P=0.998
  4. **Repair**: β≈ +0.23, CI [+0.02, +0.41], P=0.985
  5. **Social support/kin**: β≈ +0.19, P≈0.95
- **Macro axes** (5-axis model):
  - **AX_status_dominance**: β≈ +0.46, CI [+0.28, +0.61], P=1.00
  - **AX_payoff_safety**: β≈ +0.33, CI [+0.14, +0.50], P≈0.999
  - **AX_drama_obstacle**: β≈ +0.34, CI [+0.07, +0.57], P≈0.993
  - **AX_explicitness**: β≈ −0.27, CI [−0.45, −0.06], P≈0.003 (strongly negative)
- **Interpretation**: Popularity associated with "billionaire-romance package": status/luxury + alpha guarding + repair + emotional safety

**Perceived Quality Predictors** (`rating_mean`, controlling for `log_rating_count`):
- **Top predictors**:
  1. **Protective caretaking**: β≈ +0.22, CI [+0.05, +0.36], P=0.995
  2. **Emotional safety (PC1)**: β≈ +0.15, P=0.95
- **Macro axes**:
  - **AX_payoff_safety**: β≈ +0.21, P=0.974
  - **AX_explicitness**: β≈ −0.15, P=0.095 (tends negative)
  - **AX_negative_affect**: β≈ −0.13, P=0.051 (borderline negative)
- **Partial correlations** (quality beyond popularity):
  - **R1_protective_caretaking**: +0.245
  - **A2_emotional_safety__pc1**: +0.152
  - **C_explicit_eroticism**: -0.172 (negative)
  - **F2_anger_frustration**: -0.121 (negative)
- **Interpretation**: Quality beyond reach associated with "care + safety", negatively with baseline negative affect and explicit erotics

**Narrative Arc / Pacing Results**:
- Higher-rated books show **better pacing**: lower baseline negativity but stronger late "crisis escalation" (third-act crisis)
- **Anger/frustration end−begin**: β≈ +0.24, CI [+0.08, +0.41], P=0.995
- **Anxiety/worry end−begin**: β≈ +0.19, CI [+0.02, +0.36], P=0.981

**Predictive Performance** (20 repeats of 5-fold CV):
- **rating_mean**: CV R² = 0.056 ± 0.041 (themes) vs 0.108 ± 0.031 (metadata only)
- **log_rating_count**: CV R² = 0.050 ± 0.037 (themes only)

**Key Finding**: Themes explain popularity (reach) better than star ratings. Star ratings likely influenced by factors beyond theme indices (prose quality, pacing, editing, reader expectations, etc.). The theme system is better at explaining market reach than "star rating," suggesting that market reach is more systematically related to thematic content, while star ratings may be influenced by factors beyond theme indices.

**Taxonomy Group Analysis Results** (N = 92 books):
- **Main groups**: Modest but interpretable differences. Top allocates more to Relationship Trajectory (δ≈+0.37), Social World Outside Couple (δ≈+0.30), Embodied & Sensory Experience (δ≈+0.29). Trash allocates more to Sexuality, Attraction & Intimacy (δ≈-0.25)
- **Subgroups**: Stronger differentiation. **Beliefs, Values & Moral Reflection** (Top higher, δ≈+0.46, adjusted p≈0.008), **Negative Emotions & Distress** (Trash higher, δ≈-0.37), **Shared Workplaces & Professional Interaction** (Top higher, δ≈+0.35)
- **Diversity metrics**: Higher-tier books show greater thematic diversity (entropy: bad≈5.33 → mid≈5.43 → good≈5.48, p≈0.019 adjusted≈0.077). Effective topics: bad≈207 → good≈240. Richness (topics > 1e-3): bad≈247 → good≈265
- **Coverage**: Modeled mass ≈ 0.998 (very high coverage), unmapped/noise/paratext shares are extremely small (≈0.000–0.002 range)

**Topic-Level Analysis** (N = 92 books, 342 topics analyzed):
- **85 discriminative topics** identified via two-gate filtering (effect size |Cliff's δ| ≥ 0.20 AND meaningful impact: mass ≥ 0.002 OR |mean diff| ≥ 0.001)
- **Two-tier structure**: Tier 1 (High Confidence, 8 topics: |δ| ≥ 0.35 AND raw p < 0.05), Tier 2 (Exploratory, 85 topics: |δ| ≥ 0.20)
- **Top-tier differentiation**: Psychological credibility scenes (fear admissions, emotional delusion, bluffing about feelings) and embodied intimacy cues (affectionate stares, lip biting, shared joy). Often map to Radway Phase I (Initial Conflict & Isolation) and Phase II (Turning Point & Recognition)
- **Trash-tier differentiation**: Explicit sexual content (dominatrix sessions, explicit erotics) and procedural/transition scenes (doors, phones, desk work). Often map to Radway Phase III (Commitment & Restoration) or "none" (background/contextual)
- **Author dominance**: 30 topics show high author dominance (>50% from single author), requiring control in modeling. 6 topics are both significant AND author-driven
- **Topic health**: Median prevalence = 0.924 (most topics appear in most books), median mass = 0.0020, median concentration ratio = 2.68
- **Top Tier 1 examples**: "Married Couple's Affectionate Stares" (δ = 0.453), "Frightened Admissions" (δ = 0.420), "Emotional Relationship Delusion" (δ = 0.404)

See `reports/01_stage_reports/stage10_correlation_analysis/` and `reports/02_findings/hypothesis_testing/` for detailed results.

## Reports

The `reports/` directory contains all research reports, findings, and technical documentation organized by pipeline stage and analysis type. All files use numbered prefixes (01_, 02_, etc.) for clear ordering and navigation.

### Structure

- **`01_stage_reports/`**: Technical reports organized by pipeline stage
  - `03_modeling/`: Modeling methodology, character name exclusion, GPU acceleration
  - `04_selection/`: Pareto efficiency analysis results, hyperparameter correlation analysis, outlier filtering methodology
  - `05_retraining/`: Retraining methodology, output formats, model statistics
  - `06_topic_exploration/`: Topic exploration methodology, multiple representation strategies, coherence and diversity evaluation results
  - `07_topic_quality/`: Topic quality analysis methodology, noisy topic detection, quality metrics and results
  - `08_llm_labeling/`: Model comparisons, prompt documentation, cost analysis (9 reports)
  - `09_category_mapping/`: Category mapping reports (Stage 1: natural clusters, Stage 2: theory-driven, Stage 3: Radway functions)
  - `10_correlation_analysis/`: Data preparation guides and technical documentation

- **`02_findings/`**: Research findings and analysis results
  - `01_hypothesis_testing/`: Hypothesis testing results with subdirectories for arc analysis, mass appeal, perceived quality, and validation methods
  - `02_exploratory_analysis/`: Exploratory data analysis (character names, etc.)
  - `03_statistical_analysis/`: Statistical analysis summaries and model comparisons

- **`03_methodology/`**: Reserved for high-level methodology documentation and pipeline summaries

For detailed information, see [`reports/README.md`](reports/README.md).

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
