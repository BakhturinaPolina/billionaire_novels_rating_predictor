# Stage 05: Retraining Methodology and Results Report

## Overview

This report documents the retraining pipeline implemented in **Stage 05: Retraining**. This stage retrains the top N Pareto-efficient models identified in Stage 04, using their optimal hyperparameters for final model deployment. The implementation follows Stage 03 patterns but without OCTIS optimization—directly training models with hyperparameters read from the Pareto CSV file.

## Purpose

Stage 05 serves as the final model production stage, taking the top-performing models from the hyperparameter optimization and Pareto analysis pipeline and retraining them with their optimal configurations. This ensures:

1. **Reproducible model training**: Models are retrained with exact hyperparameters from Pareto analysis
2. **Production-ready outputs**: Models are saved in multiple formats for different use cases
3. **Complete metadata tracking**: Each model includes full training metadata and statistics
4. **Independent model training**: Failures in one model don't stop others

## Key Components

### Core Functionality

- **Direct retraining** from Pareto-efficient model configurations
- **GPU-accelerated** using RAPIDS (cuML) - same as Stage 03
- **Multiple output formats**: Pickle, BERTopic native format (safetensors), and metadata JSON
- **Independent model training**: Failures in one model don't stop others
- **Embedding caching**: Reuses embeddings from Stage 03 to avoid recomputation
- **Character name exclusion**: Same preprocessing pipeline as Stage 03

### Architecture

The retraining pipeline consists of three main components:

1. **`pareto_loader.py`**: Loads and parses top N models from `results/stage04_selection/pareto.csv`
2. **`retrain_models.py`**: Core retraining logic with GPU acceleration and model saving
3. **`main.py`**: CLI entry point with logging and error handling

## Differences from Stage 03

### Key Distinctions

| Aspect | Stage 03 | Stage 05 |
|--------|----------|----------|
| **Optimization** | OCTIS hyperparameter search | Direct training with provided hyperparameters |
| **Input** | Configuration files | Pareto CSV with optimal hyperparameters |
| **Output formats** | OCTIS-compatible outputs | Pickle, BERTopic native, and metadata JSON |
| **Model selection** | All models in search space | Top N Pareto-efficient models only |
| **Metadata** | Basic training info | Comprehensive metadata with scores and statistics |

### Implementation Details

- **No OCTIS optimization**: Hyperparameters are read directly from CSV
- **Direct training**: Models are trained with specific hyperparameters, not searched
- **Model saving**: Both pickle and BERTopic native formats (safetensors) are saved
- **Metadata tracking**: Each model includes detailed metadata JSON with:
  - Hyperparameters
  - Coherence and topic diversity scores
  - Combined score
  - Number of topics discovered
  - Training timestamp
  - Pareto rank

## Methodology

### Data Pipeline

The retraining process follows these steps:

1. **Load Pareto CSV**: Read top N models from `results/stage04_selection/pareto.csv`
2. **Load dataset**: Read and validate CSV file with text data
3. **Create OCTIS dataset**: Generate `corpus.tsv` format for BERTopic compatibility
4. **Load character names**: Apply same character name exclusion as Stage 03
5. **Load or create embeddings**: Reuse cached embeddings from Stage 03 when possible
6. **Train model**: Train BERTopic with specific hyperparameters
7. **Save models**: Save in multiple formats (pickle, BERTopic native, metadata)

### Character Names Preprocessing

Stage 05 uses the same character name exclusion pipeline as Stage 03:

- **Same preprocessing function**: `preprocess_character_name()` from Stage 03
- **Same stopwords**: 4,444 character name tokens + 318 standard English stopwords
- **Same filtering logic**: Removes prefixes, normalizes text, extracts multi-word names

This ensures consistency across the modeling pipeline and allows models to focus on thematic content rather than character co-occurrence patterns.

### Embedding Caching

The retraining pipeline reuses embeddings from Stage 03:

- **Cache location**: `cache/embeddings/{embedding_model_name}/`
- **Cache format**: NumPy arrays saved in batches
- **Cache validation**: Checks dataset size matches before using cached embeddings
- **Regeneration**: Automatically regenerates embeddings if cache is invalid or missing

This significantly reduces retraining time, as embedding computation is the most time-consuming step.

### GPU Acceleration

**This stage ALWAYS uses RAPIDS (cuML) for GPU acceleration.**

- Uses `cuml.manifold.UMAP` (not CPU `umap-learn`)
- Uses `cuml.cluster.HDBSCAN` (not CPU `hdbscan`)
- No CPU fallback - requires GPU
- Same GPU utilities as Stage 03: `src/common/gpu_models.py`

### Model Training Process

For each model configuration:

1. **Load embedding model**: Load SentenceTransformer model
2. **Create BERTopic wrapper**: `RetrainableBERTopicModel` extends `BERTopicOctisModelWithEmbeddings`
3. **Set hyperparameters**: Apply hyperparameters from Pareto CSV
4. **Create representation models**: KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
5. **Create GPU models**: UMAP and HDBSCAN using RAPIDS
6. **Create vectorizer**: CountVectorizer with character name stopwords
7. **Train BERTopic**: Fit model with embeddings and OCTIS dataset
8. **Extract topics**: Get number of topics discovered
9. **Save models**: Save in all three formats

## Usage

### Basic Usage

```bash
# Retrain top 4 models (default)
python -m src.stage05_retraining.main retrain

# Retrain top N models
python -m src.stage05_retraining.main retrain --top_n 4

# Specify custom paths
python -m src.stage05_retraining.main retrain \
  --pareto_csv results/stage04_selection/pareto.csv \
  --top_n 4 \
  --config configs/paths.yaml \
  --output_dir models/retrained/
```

### Command-Line Options

- `--pareto_csv`: Path to Pareto CSV file (default: `results/stage04_selection/pareto.csv`)
- `--top_n`: Number of top models to retrain (default: 4)
- `--config`: Path to paths configuration file (default: `configs/paths.yaml`)
- `--output_dir`: Base output directory for retrained models (default: `models/retrained/`)
- `--dataset_csv`: Override dataset CSV path (optional)

### Logging

All output is logged to:
- **Console**: Real-time progress updates
- **Log file**: `logs/stage05_retraining_{timestamp}.log`

The logging system captures:
- Model loading and validation
- Embedding processing
- Training progress
- Model saving
- Error messages and stack traces

## Outputs

### Directory Structure

Models are saved in the following structure:

```
models/retrained/
├── {embedding_model_1}/
│   ├── model_1.pkl                    # Pickle format (full wrapper)
│   ├── model_1/                      # BERTopic native format (safetensors)
│   │   ├── config.json
│   │   ├── topic_embeddings.safetensors
│   │   ├── ctfidf.safetensors
│   │   └── ...
│   ├── model_1_metadata.json         # Training metadata
│   ├── model_2.pkl
│   ├── model_2/
│   ├── model_2_metadata.json
│   └── ...
└── {embedding_model_2}/
    └── ...
```

### Output Formats

#### 1. Pickle Format (`.pkl`)

Full `RetrainableBERTopicModel` instance including:
- Trained BERTopic model
- Embeddings
- Wrapper state
- All hyperparameters

**Use case**: Direct Python loading for analysis and inference

**Loading**:
```python
import pickle
with open('models/retrained/{embedding_model}/model_1.pkl', 'rb') as f:
    model = pickle.load(f)
```

#### 2. BERTopic Native Format (directory)

Native BERTopic model format using safetensors:
- `config.json`: Model configuration
- `topic_embeddings.safetensors`: Topic embeddings
- `ctfidf.safetensors`: Class-based TF-IDF weights
- Embedding model reference

**Use case**: Direct loading with `BERTopic.load()` for production deployment

**Loading**:
```python
from bertopic import BERTopic
model = BERTopic.load('models/retrained/{embedding_model}/model_1')
```

**Advantages**:
- Smaller file size than pickle
- Avoids GPU array serialization issues
- Standard BERTopic format
- Safe tensor format (safetensors)

#### 3. Metadata JSON (`.json`)

Comprehensive training metadata:

```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "pareto_rank": 1,
  "hyperparameters": {
    "bertopic__min_topic_size": 15,
    "bertopic__top_n_words": 10,
    "hdbscan__min_cluster_size": 15,
    "hdbscan__min_samples": 5,
    "umap__min_dist": 0.0,
    "umap__n_components": 5,
    "umap__n_neighbors": 15,
    "vectorizer__min_df": 0.01
  },
  "coherence": 0.4523,
  "topic_diversity": 0.8234,
  "combined_score": 0.6379,
  "iteration": 42,
  "num_topics": 127,
  "training_timestamp": "2024-01-15T10:30:45.123456"
}
```

**Metadata fields**:
- `embedding_model`: Name of embedding model used
- `pareto_rank`: Rank in Pareto-efficient set
- `hyperparameters`: Full hyperparameter configuration
- `coherence`: Topic coherence score from Stage 04
- `topic_diversity`: Topic diversity score from Stage 04
- `combined_score`: Combined evaluation score from Stage 04
- `iteration`: OCTIS iteration number from Stage 03
- `num_topics`: Number of topics discovered during retraining
- `training_timestamp`: ISO format timestamp of retraining

## Data Results

### Model Statistics

Each retrained model includes:

1. **Topic count**: Number of topics discovered (excluding noise topic -1)
2. **Hyperparameters**: Full configuration used for training
3. **Evaluation scores**: Coherence, diversity, and combined scores from Pareto analysis
4. **Training metadata**: Timestamp, embedding model, Pareto rank

### Validation Checks

The retraining pipeline performs several validation checks:

1. **Dataset size validation**: Ensures training data, OCTIS corpus, and embeddings have matching sizes
2. **Hyperparameter validation**: Validates all required hyperparameters are present
3. **Model training validation**: Checks that topics were successfully generated
4. **File saving validation**: Verifies all output files were created successfully

### Error Handling

- **Independent model training**: Failures in one model don't stop others
- **Comprehensive logging**: All errors are logged with full stack traces
- **GPU memory cleanup**: Automatic cleanup after each model (success or failure)
- **Progress tracking**: Real-time progress updates for each model

## Technical Requirements

### Hardware Requirements

- **CUDA-compatible GPU** - Required
- **RAPIDS cuML** (CUDA 12.x) - Required for GPU acceleration
- **Sufficient GPU memory** - For embeddings and model training

### Software Dependencies

- **BERTopic** - Topic modeling library
- **OCTIS** - Dataset class (not for optimization)
- **RAPIDS cuML** - GPU-accelerated UMAP and HDBSCAN
- **pandas** - For CSV reading and data manipulation
- **sentence-transformers** - For embedding models
- **torch** - For GPU operations

### Configuration Files

- **`configs/paths.yaml`**: Data and output paths
- **`results/stage04_selection/pareto.csv`**: Pareto-efficient model configurations

## Integration with Other Stages

### Workflow Integration

1. **Stage 03 - Initial Training**: Train BERTopic models with OCTIS hyperparameter optimization
2. **Stage 04 - Pareto Analysis**: Identify top Pareto-efficient models
3. **Stage 05 - Retraining**: Retrain top models with optimal hyperparameters (this stage)

### Shared Components

Stage 05 shares the following with Stage 03:

- **GPU acceleration** via RAPIDS (cuML)
- **Character name exclusion** preprocessing
- **Embedding caching** (same cache location)
- **BERTopic model architecture**
- **OCTIS dataset format**

### Character Names Usage

Character names exclusion is applied identically to Stage 03:

- Same preprocessing function: `preprocess_character_name()`
- Same stopwords list: 4,444 character names + 318 standard stopwords
- Same filtering logic and cleaning steps

This ensures consistency across the modeling pipeline.

## Future Improvements

### Model Optimization

- **Batch retraining**: Parallel retraining of multiple models
- **Incremental retraining**: Retrain only models with updated hyperparameters
- **Model versioning**: Track model versions and retraining history
- **Performance metrics**: Add training time and resource usage to metadata

### Output Enhancements

- **Topic visualization**: Generate topic visualizations during retraining
- **Topic quality metrics**: Compute additional quality metrics during retraining
- **Model comparison**: Compare retrained models with original Stage 03 models
- **Export formats**: Additional export formats for different use cases

### Workflow Improvements

- **Automated retraining**: Trigger retraining automatically after Pareto analysis
- **Model validation**: Automated validation of retrained models
- **Deployment integration**: Direct integration with model deployment pipeline

## References

- **Stage 03 Report**: See `reports/01_stage_reports/stage03_modeling/stage03_modeling_and_retraining_methodology.md` for initial training methodology
- **Stage 04 Report**: See `reports/01_stage_reports/stage04_selection/` for Pareto analysis methodology
- **BERTopic Documentation**: https://maartengr.github.io/BERTopic/
- **RAPIDS cuML Documentation**: https://docs.rapids.ai/api/cuml/stable/

