# Stage 02: Preprocessing

Text cleaning, sentence segmentation, tokenization, lemmatization, and custom stoplist application.

## Status

⚠️ **Placeholder** — CLI structure defined, core logic pending. Output data (`chapters.csv`) exists from earlier processing.

## Usage

```bash
python -m src.stage02_preprocessing.main --config configs/paths.yaml
```

## Inputs

| Source | Path | Description |
|--------|------|-------------|
| Raw texts | From Stage 01 | Novel texts with metadata |
| Stoplist | `data/processed/custom_stoplist.txt` | Character names + English stopwords |

## Outputs

| Output | Path | Description |
|--------|------|-------------|
| Chapters | `data/processed/chapters.csv` | 680,822 sentences, one per row |

**Output columns**: `Author`, `Book Title`, `Chapter`, `Sentence`

## Processing Steps

1. **Text cleaning**: Encoding fixes (mojibake), whitespace normalization
2. **Sentence segmentation**: Split into sentences (spaCy)
3. **Tokenization & lemmatization**: POS tagging, root form extraction
4. **Stopword removal**: 4,762 stopwords (4,444 character names + 318 English)

## Module Structure

```
stage02_preprocessing/
├── main.py      # CLI entrypoint
├── README.md    # This file
└── __init__.py
```

## See Also

- [Methodology Report](../../reports/01_stage_reports/stage02_preprocessing/stage02_preprocessing_methodology.md) — Research rationale and processing decisions
