# Stage 02: Preprocessing

## Overview

Stage 02 handles text cleaning, tokenization, lemmatization, and custom stoplist building.

## Status

⚠️ **Placeholder Implementation** - Core logic pending

The `main.py` file shows the planned structure, but implementation is pending.

## Planned Functionality

### Text Cleaning
- Remove special characters
- Handle encoding issues
- Normalize whitespace
- Remove headers/footers if present

### Tokenization & Lemmatization
- Sentence segmentation
- Word tokenization
- Part-of-speech tagging
- Lemmatization (using spaCy or similar)

### Custom Stoplist
- Load custom stoplist from `data/processed/custom_stoplist.txt`
- Remove stopwords
- Domain-specific filtering

### Output
- Cleaned, tokenized, lemmatized text
- One sentence per row in CSV format
- Ready for Stage 03 modeling

## Data Contracts

**Input**: Raw text files from Stage 01  
**Output**: `data/processed/chapters.csv` with columns:
- `Book_Title`: Book identifier
- `Sentence`: Preprocessed sentence text

See [docs/DATA_CONTRACTS.md](../../docs/DATA_CONTRACTS.md) for details.

## Usage

```bash
python -m src.stage02_preprocessing.main --config configs/paths.yaml
```

## Implementation Notes

When implementing, consider:
- Using spaCy for NLP processing
- Handling large files efficiently
- Preserving sentence boundaries
- Custom stoplist integration
- Memory management for large datasets

