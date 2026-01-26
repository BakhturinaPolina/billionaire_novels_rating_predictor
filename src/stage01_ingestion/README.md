# Stage 01: Data Ingestion

Load raw novel texts and Goodreads metadata for downstream processing.

## Status

⚠️ **Placeholder** — CLI structure defined, core logic pending.

## Usage

```bash
python -m src.stage01_ingestion.main --config configs/paths.yaml
```

## Inputs

| Source | Path | Description |
|--------|------|-------------|
| Raw texts | `data/raw/Billionaire_Full_Novels_TXT/` | TXT/EPUB novel files |
| Goodreads | `data/processed/goodreads.csv` | Ratings, review counts |
| BookNLP | `data/interim/booknlp/` | Character entities (optional) |

## Outputs

| Output | Description |
|--------|-------------|
| Processed texts | Book texts with metadata attached |
| Merged metadata | Book IDs, ratings, author info |
| Character names | For stoplist in Stage 02 |

## Module Structure

```
stage01_ingestion/
├── main.py      # CLI entrypoint
├── README.md    # This file
└── __init__.py
```

## See Also

- [Methodology Report](../../reports/01_stage_reports/stage01_ingestion/stage01_data_ingestion_methodology.md) — Research rationale and data decisions
