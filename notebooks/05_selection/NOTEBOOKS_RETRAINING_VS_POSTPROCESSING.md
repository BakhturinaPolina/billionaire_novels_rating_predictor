# Notebooks: Retraining vs Postprocessing

## Overview

This document describes the difference between notebooks for **retraining models** and notebooks for **postprocessing trained models** in the `05_selection` directory.

## Retraining Notebooks

**Purpose**: These notebooks train/fit BERTopic models from scratch using the dataset and selected hyperparameters.

**Location**: 
- `legacy_select_best_model/1o__best_bert_model.ipynb`
- `post_processing_and_choosing_best_model/best models.ipynb`

**Key Characteristics**:
- Load raw/preprocessed text data
- Initialize BERTopic models with specific hyperparameters
- Fit/train models on the dataset (`model.fit()`)
- Generate topics from scratch
- Save trained model files
- May use GPU acceleration (RAPIDS/CUDA) for faster training

**Typical Workflow**:
1. Load dataset (sentences/text)
2. Preprocess text (tokenization, stop word removal)
3. Configure BERTopic with hyperparameters (UMAP, HDBSCAN, vectorizer settings)
4. Fit model on data
5. Extract topics
6. Save model for later use

## Postprocessing Notebooks

**Purpose**: These notebooks work with already-trained models and their topics to analyze, clean, filter, or improve them.

**Location**: `post_processing_and_choosing_best_model/`

**Notebooks**:
- `cleaning_topics_by_coherence_score.ipynb` - Removes low-quality topics based on coherence scores
- `coherence_diversity_calculations.ipynb` - Calculates coherence and diversity metrics for existing topics
- `post_processing_best_topics_coherence_priority.ipynb` - Post-processes and prioritizes topics by coherence
- `hyperparameters_corr.ipynb` - Analyzes correlations between hyperparameters and model performance

**Key Characteristics**:
- Load pre-trained models or topic files (JSON/PKL)
- Work with existing topics (no model training)
- Calculate metrics (coherence, diversity) on existing topics
- Filter/clean topics based on quality metrics
- Analyze relationships between hyperparameters and outcomes
- Generate visualizations and reports

**Typical Workflow**:
1. Load trained models or topic files
2. Extract topics from models
3. Calculate quality metrics (coherence, diversity)
4. Filter/clean topics based on thresholds
5. Analyze patterns and correlations
6. Generate visualizations

## Key Differences

| Aspect | Retraining | Postprocessing |
|--------|-----------|----------------|
| **Input** | Raw text data | Trained models/topic files |
| **Process** | Model training (`fit()`) | Topic analysis and filtering |
| **Output** | Trained model files | Cleaned topics, metrics, analyses |
| **Computational Cost** | High (GPU recommended) | Low (CPU sufficient) |
| **Purpose** | Generate topics | Improve/analyze existing topics |

## Current Status

- **Pareto efficiency analysis notebooks** have been moved to `archive/notebooks/05_selection_pareto_legacy/`
- **Main Pareto analysis**: `pareto_efficiency_analysis.ipynb` (kept in `05_selection/`)
- **Retraining notebooks**: Kept in `05_selection/` for future use
- **Postprocessing notebooks**: Kept in `05_selection/post_processing_and_choosing_best_model/` for ongoing analysis

