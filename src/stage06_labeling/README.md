# Stage 06: Topic Labeling & Composite Building

## Overview

Stage 06 maps topics to 16 thematic composites using a semi-supervised approach combining manual mapping and codebook-guided assignment.

## Status

✅ **Partially Implemented**

- ✅ `composites.py`: Full implementation of composite building
- ⚠️ `main.py`: Placeholder (shows structure)

## Functionality

### Thematic Composites (A-P)

1. **A**: Reassurance/Commitment
2. **B**: Mutual Intimacy
3. **C**: Explicit Eroticism
4. **D**: Power/Wealth/Luxury
5. **E**: Coercion/Brutality/Danger
6. **F**: Angst/Negative Affect
7. **G**: Courtship Rituals/Gifts
8. **H**: Domestic Nesting
9. **I**: Humor/Lightness
10. **J**: Social Support/Kin
11. **K**: Professional Intrusion
12. **L**: Vices/Addictions
13. **M**: Health/Recovery/Growth
14. **N**: Separation/Reunion
15. **O**: Aesthetics/Appearance
16. **P**: Tech/Media Presence

### Mapping Process

1. **Manual Mapping** (Markdown)
   - Researcher-defined topic-to-cluster mappings
   - Hierarchical structure with keyword matching

2. **Codebook** (CSV)
   - Structured category definitions
   - Primary categories and subtypes
   - Systematic assignment rules

3. **Automated Assignment**
   - Weighted topic-to-composite mapping
   - Special handling for ambiguous topics
   - Example: "Woman Pleasure and Arousal" → 70% Explicit, 30% Mutual Intimacy

### Composite Building

Topics are mapped to composites with weighted assignments:
- Each topic can map to multiple composites
- Weights sum to 1.0 per topic
- Final composite scores = sum of (topic_prob × weight) for all topics

## Key Files

- **`composites.py`**: Core composite building logic
  - `build_mapping()`: Creates topic-to-composite mapping
  - `main()`: Executes composite building pipeline

- **`main.py`**: CLI entry point (placeholder)

## Usage

```bash
# Direct execution of composites.py
python -m src.stage06_labeling.composites

# Or via main.py (when implemented)
python -m src.stage06_labeling.main --config configs/labeling.yaml
```

## Inputs

- **Prepared Books**: `data/processed/prepared_books.parquet`
  - Must contain topic probability columns

- **Manual Mapping**: `results/experiments/.../Billionaire_Manual_Mapping_Topics_by_Thematic_Clusters.md`

- **Codebook** (Optional): `results/experiments/.../focused_topic_codebook.csv`

## Outputs

- **Composites Parquet**: `results/topics/ap_composites.parquet`
  - Columns: Metadata + 16 composite scores (A-P)

## Data Contracts

See [docs/DATA_CONTRACTS.md](../../docs/DATA_CONTRACTS.md) for detailed specifications.

## Special Cases

### Weighted Topics

Some topics have special weighted assignments:
- **"Woman Pleasure and Arousal"**: 70% `C_Explicit_Eroticism`, 30% `B_Mutual_Intimacy`

### Default Assignment

If a topic has no mapping, it is assigned equally to all matched composites:
```python
dests = {d: 1.0/len(mapping[t]) for d in mapping.get(t, [])}
```

## Methodology

See [docs/METHODOLOGY.md](../../docs/METHODOLOGY.md) for detailed mapping algorithm.

## Dependencies

- `pandas` for data manipulation
- `pathlib` for file handling
- `re` for pattern matching

