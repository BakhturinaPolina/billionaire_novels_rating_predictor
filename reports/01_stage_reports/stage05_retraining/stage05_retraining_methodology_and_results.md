# Stage 05: Retraining — Methodology Report

## Purpose

Retrain top N Pareto-efficient models identified in Stage 04 with their optimal hyperparameters for final deployment.

## Research Rationale

### Why Retrain After Pareto Analysis?

1. **Reproducibility**: Models retrained with exact hyperparameters from Pareto analysis ensure consistent results
2. **Production-ready outputs**: Multiple output formats (pickle, native, metadata) support different use cases
3. **Complete metadata**: Full training metadata enables model comparison and validation
4. **Independent training**: Failures in one model don't affect others

### Differences from Stage 03

| Aspect | Stage 03 | Stage 05 |
|--------|----------|----------|
| **Optimization** | OCTIS hyperparameter search | Direct training with provided hyperparameters |
| **Input** | Configuration files | Pareto CSV with optimal hyperparameters |
| **Output formats** | OCTIS-compatible outputs | Pickle, BERTopic native, and metadata JSON |
| **Model selection** | All models in search space | Top N Pareto-efficient models only |

---

## Methodology

### Data Pipeline

1. **Load Pareto CSV**: Read top N models from `results/stage04_selection/pareto.csv`
2. **Load dataset**: Read and validate CSV file with text data
3. **Create OCTIS dataset**: Generate `corpus.tsv` format for BERTopic compatibility
4. **Load character names**: Apply same character name exclusion as Stage 03
5. **Load or create embeddings**: Reuse cached embeddings from Stage 03 when possible
6. **Train model**: Train BERTopic with specific hyperparameters
7. **Save models**: Save in multiple formats (pickle, BERTopic native, metadata)

### Character Name Exclusion

Stage 05 uses the same character name exclusion pipeline as Stage 03:
- Same preprocessing function and stopwords (4,444 character names + 318 standard English)
- Ensures consistency across the modeling pipeline

### Embedding Caching

The retraining pipeline reuses embeddings from Stage 03:
- **Cache location**: `cache/embeddings/{embedding_model_name}/`
- **Cache validation**: Checks dataset size matches before using cached embeddings
- **Regeneration**: Automatically regenerates if cache is invalid or missing

This significantly reduces retraining time, as embedding computation is the most time-consuming step.

### GPU Acceleration

**RAPIDS cuML** (CUDA 12.x) for:
- UMAP dimensionality reduction
- HDBSCAN clustering

No CPU fallback — GPU required.

---

## Output Formats

### 1. Pickle Format (`.pkl`)

Full `RetrainableBERTopicModel` instance including:
- Trained BERTopic model
- Embeddings
- Wrapper state
- All hyperparameters

**Use case**: Direct Python loading for analysis and inference

### 2. BERTopic Native Format (directory)

Native BERTopic model format using safetensors:
- `config.json`: Model configuration
- `topic_embeddings.safetensors`: Topic embeddings
- `ctfidf.safetensors`: Class-based TF-IDF weights

**Use case**: Direct loading with `BERTopic.load()` for production deployment

**Advantages**:
- Smaller file size than pickle
- Avoids GPU array serialization issues
- Standard BERTopic format

### 3. Metadata JSON (`.json`)

Comprehensive training metadata:
- Embedding model name
- Pareto rank
- Full hyperparameter configuration
- Coherence, diversity, and combined scores
- Number of topics discovered
- Training timestamp

---

## Integration with Pipeline

| Stage | Relationship |
|-------|--------------|
| ← Stage 03 | Reuses embedding cache and preprocessing pipeline |
| ← Stage 04 | Receives Pareto-efficient model configurations |
| → Stage 06 | Provides retrained models for exploration |

### Typical Workflow

1. **Stage 03**: Train BERTopic models with OCTIS optimization
2. **Stage 04**: Identify Pareto-efficient models
3. **Stage 05**: Retrain top models with optimal hyperparameters (this stage)

---

## References

- **Stage 03 Report**: See `reports/01_stage_reports/stage03_modeling/` for initial training methodology
- **Stage 04 Report**: See `reports/01_stage_reports/stage04_selection/` for Pareto analysis methodology
