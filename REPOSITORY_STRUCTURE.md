# Repository Structure Map

**Romantic Novels NLP Research Project**

This document provides a comprehensive map of the repository structure, organized by research phase and component type.

---

## Root Directory

### Configuration Files
- **`requirements.txt`** - Python package dependencies
- **`.gitignore`** - Git ignore patterns (excludes venv, __pycache__, etc.)

### Configuration Directories
- **`.cursor/`** - Cursor IDE configuration
- **`.rules/`** - Project-specific coding rules and protocols

---

## Data Directory (`data/`)

### `data/raw/` - Raw Input Data
**Purpose:** Unprocessed source materials (novels, EPUB files)

#### `Billionaire_Full_Novels_TXT/`
- **Description:** Full text files of romantic novels organized by author
- **Structure:** Author subdirectories (Asher, Bowen, Clare, Day, etc.) containing `.txt` files
- **Content:** Complete novel texts extracted from EPUBs

#### `Billionaire_Novels_EPUB/`
- **Description:** Original EPUB files organized by author
- **Structure:** 
  - Author subdirectories (35+ authors) with `.epub` files
  - `booknlp_outputs/` - BookNLP processing outputs (105 books)
  - `txt_books/` - Extracted text files from EPUBs (105 files)

---

### `data/interim/` - Intermediate Processing Data
**Purpose:** Data generated during processing steps, not final outputs

#### `data/interim/booknlp/`
- **Description:** BookNLP processing outputs (entities, tokens, quotes, etc.)
- **Structure:** `Bilionaire_Romantic_Novels_EPUB_BookNLP/` organized by author
- **File Types:**
  - `.book` - BookNLP book format files
  - `.entities` - Named entity recognition results
  - `.tokens` - Tokenized text
  - `.quotes` - Extracted dialogue/quotes
  - `.supersense` - Semantic supersense annotations
  - `.txt` - Token files by POS (adj_noun, verb_noun, noun, all)

---

### `data/processed/` - Final Processed Data
**Purpose:** Cleaned, processed data ready for analysis and modeling

#### Files:
- **`chapters.csv`** - Processed novel chapters with sentences (main dataset, ~707K rows)
- **`chapters_subset_10000.csv`** - Subset of chapters for testing (10K rows)
- **`goodreads.csv`** - Cleaned Goodreads dataset with ratings and metadata
- **`custom_stoplist.txt`** - Custom stopwords list (extracted character names)

#### `Billionaire_ALL_MERGED_TXTs_by_POS/`
- **Description:** Merged token files organized by part-of-speech
- **Files:**
  - `tokens_all.txt` - All tokens
  - `tokens_noun.txt` - Noun tokens only
  - `tokens_adj_noun.txt` - Adjective-noun pairs
  - `tokens_verb_noun.txt` - Verb-noun pairs

---

## Source Code (`src/`)

### `src/pipeline/` - Main Pipeline Package
**Purpose:** Modular pipeline components organized by research phase

#### `src/pipeline/ingestion/`
- **Purpose:** Data loading and initial processing
- **Files:**
  - `__init__.py` - Package initialization
  - (Additional ingestion modules to be added)

#### `src/pipeline/preprocessing/`
- **Purpose:** Text preprocessing, stoplist building, chapter binning
- **Files:**
  - `__init__.py` - Package initialization
  - (Additional preprocessing modules to be added)

#### `src/pipeline/modeling/`
- **Purpose:** BERTopic model training and management
- **Files:**
  - `__init__.py` - Package initialization
  - **`bertopic_runner.py`** - Main BERTopic wrapper integrating with OCTIS, handles model training with various embedding models
  - **`retrain_from_tables.py`** - Retrain BERTopic models from topic tables, includes coherence priority and representation improvements

#### `src/pipeline/experiments/`
- **Purpose:** Hyperparameter optimization and experimental runs
- **Files:**
  - `__init__.py` - Package initialization
  - **`hparam_search.py`** - Bayesian hyperparameter search using Optuna, optimizes BERTopic parameters

#### `src/pipeline/selection/`
- **Purpose:** Model selection using Pareto efficiency and constraints
- **Files:**
  - `__init__.py` - Package initialization
  - (Selection modules to be added)

#### `src/pipeline/labeling/`
- **Purpose:** Semi-supervised topic labeling and composite building
- **Files:**
  - `__init__.py` - Package initialization
  - **`composites.py`** - Builds AP (Appreciation Pattern) composites from topic groups

#### `src/pipeline/analysis/`
- **Purpose:** Statistical analysis, correlations, and hypothesis testing
- **Files:**
  - `__init__.py` - Package initialization
  - **`scoring_and_strata.py`** - Prepares data and groups for statistical analysis, builds quartiles
  - **`bh_fdr.py`** - Benjamini-Hochberg FDR correction for multiple comparisons, group delta analysis
  - **`03_interaction_plots_PD.py`** - Generates interaction plots for probability distributions
  - **`04_star_charts_and_pdf.py`** - Creates star charts and PDF visualizations

#### `src/pipeline/utils/`
- **Purpose:** Shared utilities and helper functions
- **Files:**
  - `__init__.py` - Package initialization
  - **`utils/check_gpu_setup.py`** - GPU availability and configuration checks
  - **`utils/hw_utils.py`** - Hardware utility functions
  - **`utils/thermal_monitor.py`** - GPU thermal monitoring
  - **`utils/training_utils.py`** - Training-related utilities

---

## Notebooks (`notebooks/`)

**Purpose:** Jupyter notebooks organized by research phase

### `notebooks/02_ingestion/` - Data Ingestion & Initial EDA
- **`dataset_building/`**
  - `bookNLP_parse_text_of_books.ipynb` - BookNLP text parsing pipeline
  - `parse_structured_text_from_epub.ipynb` - EPUB extraction and parsing
  - `scraping_ratings_information_romantic.ipynb` - Goodreads data scraping
- **`romantic novels dataset EDA/`**
  - `EDA romantic novels.ipynb` - Initial exploratory data analysis

### `notebooks/03_modeling/` - Model Training & Optimization
- **`retrain_best_models/`**
  - `retrain_best_topics_coherence_proprity.ipynb` - Retrain models with coherence priority
  - `retrain_best_topics_coherence_proprity_with_add_repr.ipynb` - Retrain with additional representation improvements
  - `Copy of retrain_best_topics_coherence_proprity_with_add_repr.ipynb` - Backup copy

### `notebooks/06_pareto/` - Pareto Analysis & Model Selection
- **`post_processing_and_choosing_best_model/`**
  - `Billionaire_Project_Pareto_Efficiency_Analysis_Choose_Top_Models.ipynb` - Main Pareto efficiency analysis
  - `pareto_ten_best_models.ipynb` - Top 10 Pareto-efficient models
  - `hyperparameters_corr.ipynb` - Hyperparameter correlation analysis
  - `coherence_diversity_calculations.ipynb` - Coherence and diversity metrics
  - `cleaning_topics_by_coherence_score.ipynb` - Topic cleaning by coherence
  - `post_processing_best_topics_coherence_priority.ipynb` - Post-processing with coherence priority
  - `choose_best_models_plot_stars.ipynb` - Model selection visualization
  - `best models.ipynb` - Best model identification
  - `1o__best_bert_model.ipynb` - Best BERTopic model analysis
  - `Another copy of 1o__best_bert_model.ipynb` - Backup copy

### `notebooks/09_goodreads/` - Goodreads Dataset Analysis
- **`goodreads dataset EDA and score adjustment/`**
  - `EDA_goodreads.ipynb` - Goodreads exploratory data analysis
  - `Exploratory Data Analysis (EDA) of the Goodreads Dataset.ipynb` - Main EDA notebook
  - `RIGHT Exploratory Data Analysis (EDA) of the Goodreads Dataset.ipynb` - Corrected EDA version
  - `bayesian_score_adjustment.ipynb` - Bayesian score adjustment for ratings
  - Multiple copies/variants of EDA notebooks

### `notebooks/10_stats/` - Statistical Analysis & Findings
- **`correlation analysis of topics/`**
  - `topics_analysis.ipynb` - Main topic correlation analysis
  - `topics_correlation_with_all_topics.ipynb` - Cross-topic correlations
  - `bert_topic_similarity_analysis.ipynb` - BERTopic similarity analysis
  - `clustering_topics_per_book_BERTopic.ipynb` - Topic clustering by book
  - `Topics by Broader Thematic Gropus.ipynb` - Thematic grouping analysis
  - `Fanfuction_Example_full_code_for_topics_BERTopic.ipynb` - Example code
  - `Copy of topics_analysis.ipynb` - Backup copy

- **`mapping back probabilities to books ans statistical analysis of probabilities distribution/`**
  - `books_by_probabilities_analysis.ipynb` - Probability distribution analysis
  - `Copy of books_by_probabilities_analysis.ipynb` - Backup copy

- **`clusterisation by subgenres/`**
  - `clusterisation of topis by subgenres.ipynb` - Topic clustering by subgenre

---

## Results (`results/`)

### `results/experiments/` - Experimental Results & Ledgers
- **`model_evaluation_results.csv`** - Summary ledger of all model evaluation results
- **`Billionaire_OCTIS_ALL_Models_Results_CSV_Only/`** - OCTIS experiment results by model
  - Subdirectories: `all-MiniLM-L12-v2/`, `multi-qa-mpnet-base-cos-v1/`, `paraphrase-distilroberta-base-v1/`, etc.
  - Files: `result.json`, `result_2.json`, `result_3.json` (experiment outputs)
- **`Billionaire_OCTIS_Two_Pretrained_Models_CSV_With_Evaluation_Results/`** - Evaluation results with pretrained models
  - `optimization_results/` - Optuna optimization results (`.npz` files)
- **`reproducible_scripts topics vs readers appreciation/`** - Analysis outputs
  - `signature_themes_by_group_long.csv` - Signature themes by reader group
  - `top20_signature_themes_*.csv` - Top 20 themes by quartile (top, middle, trash)
  - `top216_group_classification_refined.csv` - Refined topic group classifications
  - `topic_correlation_matrix.csv` - Topic correlation matrix
  - `topic_group_prevalence_with_eta2.csv` - Group prevalence with effect sizes
  - `top_correlated_pairs.csv` - Most correlated topic pairs
  - `all_topics_grouped_bars.pdf` - Visualization PDF
  - `Billionaire_Manual_Mapping_Topics_by_Thematic_Clusters.md/pdf` - Manual topic mapping documentation
  - `README.txt` - Documentation

### `results/pareto/` - Pareto-Efficient Models
- **`topics/pareto_efficient_per_model/`** - Topic sets for Pareto-efficient models
  - 28 JSON files, one per Pareto-efficient model configuration
  - Format: `{model_name}__{trial_id}__topics.json`

### `results/topics/` - Topic Model Outputs
- **`by_book.csv`** - Topic distribution by book (main output)
- **`Bilionaires_JSON_Top_Models_Topics_with_Coherence_Scores/`** - Top models with coherence scores
  - 14 JSON files with cleaned topics and coherence metrics
  - Format: `{model}__{trial}__coh_{score}__topics_clean.json`
- **`Bilionaires_JSON_Topics_NO_Coherence_Scores/`** - Topics without coherence scores
  - 14 JSON files
- **`Billionaire_Top_Models_JSON_Topics/`** - Top model topic outputs
  - 8 JSON files with topic representations

### `results/figures/` - Generated Visualizations
- **Purpose:** EDA plots, correlation visualizations, and analysis figures
- **Status:** Currently empty (figures generated on-demand)

---

## Reports (`reports/`)

### `reports/eda/` - Exploratory Data Analysis Reports
- **Purpose:** EDA documentation and findings
- **Status:** Currently empty

### `reports/findings/` - Research Findings & Documentation
- **`CHARACTER_NAMES_ANALYSIS.md`** - Analysis of character name extraction
- **`README_CHARACTER_NAMES.md`** - Character names documentation
- **`PIPELINE_OVERVIEW.md`** - Pipeline overview and methodology

---

## Scripts (`scripts/`)

**Purpose:** Standalone utility scripts

- **`convert_topics.py`** - Converts topic `.npy` files to JSON format
- **`restart_script.py`** - Script for restarting interrupted experiments
- **`ml_heat_diag.sh`** - Shell script for ML hardware diagnostics

---

## Configs (`configs/`)

**Purpose:** Configuration files for pipeline components

- **Status:** Currently empty (to be populated with YAML configs)
- **Planned:** `paths.yaml`, `bertopic.yaml`, `octis.yaml`, `optuna.yaml`, `selection.yaml`, `labeling.yaml`, `scoring.yaml`

---

## Models (`models/`)

**Purpose:** Saved trained models

- **Status:** Currently empty
- **Usage:** Will store serialized BERTopic models after training

---

## Research Pipeline Flow

1. **Ingestion (02)** → Raw data → BookNLP processing → Processed chapters
2. **Modeling (03)** → BERTopic training → Hyperparameter optimization → Model evaluation
3. **Selection (06)** → Pareto analysis → Best model selection
4. **Analysis (09-10)** → Goodreads integration → Statistical analysis → Findings

---

## File Naming Conventions

- **Models:** `{embedding_model}__{trial_id}__{additional_info}__topics.json`
- **Results:** `result.json`, `result_{n}.json` for multiple runs
- **Data:** Descriptive names (e.g., `chapters.csv`, `goodreads.csv`)
- **Notebooks:** Descriptive names indicating purpose and phase

---

## Notes

- All file moves were performed using `git mv` to preserve history
- Repository structure follows research phase organization
- Python packages use `__init__.py` for proper module structure
- Large files (parquet, npy, pt, mm, ipynb) should use Git LFS (not currently configured)

---

*Last updated: After refactoring to research-phase structure*

