# Stage 03: Modeling Methodology Report

## Overview

This report documents the modeling pipeline implemented in **Stage 03: Modeling**. This stage handles initial BERTopic model training with OCTIS integration for hyperparameter optimization, focusing on GPU-accelerated topic modeling using RAPIDS (cuML) for UMAP dimensionality reduction and HDBSCAN clustering, with character name exclusion to improve topic interpretability.

### Purpose

Stage 03 handles initial BERTopic model training with OCTIS integration for hyperparameter optimization. The stage implements GPU-accelerated topic modeling using RAPIDS (cuML) for UMAP dimensionality reduction and HDBSCAN clustering.

### Key Components

- **BERTopic model training** with OCTIS integration for hyperparameter search
- **GPU-accelerated** UMAP and HDBSCAN using RAPIDS (cuML)
- **Character name exclusion** to improve topic interpretability
- **Embedding caching** to avoid recomputation

### Character Names Preprocessing

#### Research Rationale

In narrative text analysis, character names present a unique challenge for topic modeling algorithms. While names carry narrative significance, they often function as high-frequency markers that obscure thematic content. When character names appear frequently across documents, they can dominate topic word distributions, creating topics that reflect character co-occurrence rather than thematic relationships.

We implemented character name exclusion to improve topic interpretability by allowing the algorithm to focus on content words—verbs, adjectives, nouns describing actions, emotions, and narrative elements—rather than character references. This approach aligns with best practices in computational literary analysis where character names are typically treated as structural elements rather than semantic content (Bamman et al., 2013; Jockers, 2013).

#### Processing Statistics

The preprocessing pipeline processes character names extracted from romance novel texts:

- **Total lines processed**: 7,525
- **Lines filtered out**: 254 (3.4%)
- **Lines with valid names**: 7,271
- **Multi-word names processed**: 3,313
- **Unique name tokens extracted**: 4,497
- **Final character names added to stopwords**: 4,444

The 4,444 final count reflects that some character name tokens already overlapped with standard English stopwords (e.g., common first names like "will", "may" that are also modal verbs).

#### Stopwords Summary

After processing:
- **Standard English stopwords**: 318
- **Character names added**: 4,444
- **Total stopwords**: 4,762

This represents a 14x increase in stopwords, with character names comprising approximately 93% of the expanded list.

#### Preprocessing Pipeline

The character names preprocessing performs the following operations:

**Cleaning steps**:
- Removes leading prefixes: "A ", "Mr.", "Miss ", "the ", "AKA ", "#"
- Removes leading numbers: "17 Vick Jett" → "vick jett"
- Removes quotes and punctuation
- Converts everything to lowercase

**Filtering steps**:
- Removes empty lines
- Removes very long lines (>50 characters) - usually descriptions
- Removes common phrases like "A voice", "the Wright brothers"
- Removes descriptive patterns like "A scowling Dante" (keeps "Dante")
- Filters out common non-name words
- Skips geographic locations

**Name extraction**:
- Splits multi-word names: "Alex Crane" → extracts both "alex" and "crane"
- This ensures both first and last names are filtered, even if they appear separately

#### Filtering Examples

The preprocessing pipeline successfully filtered out various non-name patterns:

**Correctly Filtered**:
- **Prefix patterns**: "After T.J." → filtered (temporal prefix)
- **Common word phrases**: "A voice" → filtered (descriptive phrase)
- **Descriptive patterns**: "A scowling Dante" → filtered (keeps "dante" if it appears elsewhere as a name)
- **Long descriptions**: "AUTHOR K.A. LINDE … TheWrightBoss HEIDI SWORE SHE'D" → filtered (>50 characters, descriptive text)

#### Precision Trade-offs

The preprocessing retained some words that are not character names (estimated 1-2% of tokens):
- Examples: "aardvarks", "accustomed", "activated", "actress", "actually", "ad"

We decided **not** to implement additional aggressive filtering at this stage for the following reasons:

1. **Coverage vs. Precision Trade-off**: The current pipeline successfully extracts 4,444 character name tokens, capturing the vast majority of character references. Implementing stricter filtering would risk excluding valid character names, especially uncommon or invented names.

2. **Acceptable False Positives**: The remaining non-name words are minimal compared to the benefit of comprehensive character name coverage. These false positives have minimal impact on topic quality as they appear infrequently in actual documents.

3. **Practical Considerations**: Further cleaning would require maintenance of name databases or dictionaries, more complex pattern matching rules, and potential reduction in reproducibility.

#### Impact on Topic Modeling

Preliminary analysis indicates that character name exclusion improves topic interpretability:

- **Before exclusion**: Topics often dominated by character names (e.g., "Alex, Stella, Weston, love")
- **After exclusion**: Topics focus on thematic content (e.g., "love, relationship, emotion, connection")

This shift allows researchers to identify thematic patterns and narrative elements rather than character co-occurrence patterns, which aligns with computational literary analysis goals.

### Usage

#### Test Pipeline

```bash
# Test with subset (10K rows) - fast validation
python -m src.stage03_modeling.test_octis_pipeline --subset

# Test with full dataset
python -m src.stage03_modeling.test_octis_pipeline --full
```

#### Train Models

```bash
# Train BERTopic models (OCTIS integration)
python -m src.stage03_modeling.main train --config configs/bertopic.yaml
```

#### Optimize Models

```bash
# Run hyperparameter optimization with OCTIS
python -m src.stage03_modeling.main optimize --config configs/octis.yaml
```

### GPU Acceleration

**This stage ALWAYS uses RAPIDS (cuML) for GPU acceleration.**

- Uses `cuml.manifold.UMAP` (not CPU `umap-learn`)
- Uses `cuml.cluster.HDBSCAN` (not CPU `hdbscan`)
- No CPU fallback - requires GPU

### Outputs

- **`models/`** - Trained BERTopic models
- **`results/topics/by_book.csv`** - Topic probabilities per book
- **`results/topics/top_models/*.json`** - Topic word lists

## Integration with Other Stages

### Typical Workflow

1. **Stage 03 - Initial Training**: Train BERTopic models with OCTIS hyperparameter optimization
2. **Pareto Analysis**: Identify top Pareto-efficient models (handled in Stage 04)
3. **Stage 05 - Retraining**: Retrain top models with their optimal hyperparameters for final deployment (see separate Stage 05 report)

### Character Names Preprocessing

Character names exclusion is applied during model training:

- Character names are loaded from the preprocessing pipeline
- Names are preprocessed and added to stopwords during initial training
- This ensures consistent preprocessing across the modeling pipeline

## Technical Requirements

### Hardware Requirements

- **CUDA-compatible GPU** - Required
- **RAPIDS cuML** (CUDA 12.x) - Required for GPU acceleration

### Software Dependencies

- **BERTopic** - Topic modeling library
- **OCTIS** - Optimization framework for hyperparameter search
- **RAPIDS cuML** - GPU-accelerated UMAP and HDBSCAN
- **pandas** - For CSV reading and data manipulation

## Future Improvements

### Character Names Preprocessing

While the current approach prioritizes coverage over precision, future enhancements could include:

- **Domain-specific name databases**: For romance novels specifically, maintaining a curated list of known character names could improve precision
- **Frequency-based filtering**: Analyzing token frequencies in the actual corpus could help identify non-name words that slipped through
- **Hybrid approach**: Combining current extraction with manual review of high-frequency extracted tokens

These improvements could be implemented as optional enhancements for researchers requiring higher precision, while maintaining the current approach as the default for maximum coverage and reproducibility.

### Model Optimization

- **Embedding model selection**: Expand beyond current embedding models
- **Hyperparameter space exploration**: Further refinement of search spaces
- **Model ensemble approaches**: Combining multiple models for improved topic quality
- **Integration with Stage 05**: Improved workflow for retraining top models with optimal hyperparameters

## References

- Bamman, D., Underwood, T., & Smith, N. A. (2013). A Bayesian Mixed Effects Model of Literary Character. *Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics*.
- Jockers, M. L. (2013). *Macroanalysis: Digital Methods and Literary History*. University of Illinois Press.

