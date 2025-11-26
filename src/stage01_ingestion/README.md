# Stage 01: Data Ingestion

## Overview

Stage 01 handles loading raw data files, Goodreads metadata, and BookNLP outputs for downstream processing.

## Status

⚠️ **Placeholder Implementation** - Core logic pending

The `main.py` file shows the planned structure and data contracts, but implementation is pending.

## Planned Functionality

### Inputs
- Raw text files from `data/raw/Billionaire_Full_Novels_TXT/`
- Goodreads metadata CSV: `data/processed/goodreads.csv`
- BookNLP outputs (optional): `data/interim/booknlp/`

### Outputs
- Processed text data for Stage 02
- Merged metadata
- BookNLP entity information

## Data Contracts

See [docs/DATA_CONTRACTS.md](../../docs/DATA_CONTRACTS.md) for detailed input/output specifications.

## Usage

```bash
python -m src.stage01_ingestion.main --config configs/paths.yaml
```

## Implementation Notes

When implementing, consider:
- Handling multiple file formats (TXT, EPUB, etc.)
- Merging Goodreads metadata with book texts
- Processing BookNLP outputs if available
- Error handling for missing files
- Progress tracking for large datasets

