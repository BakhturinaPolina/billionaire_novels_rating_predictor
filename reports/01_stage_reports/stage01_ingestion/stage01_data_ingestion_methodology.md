# Stage 01: Data Ingestion — Methodology Report

## Purpose

Stage 01 serves as the foundational data loading step, consolidating raw text files, Goodreads metadata, and BookNLP outputs for downstream processing.

## Research Rationale

### Why Separate Ingestion from Preprocessing?

1. **Reproducibility**: Clear input/output contracts allow rerunning later stages independently
2. **Early Validation**: Catch data integrity issues (encoding, missing files) before expensive processing
3. **Metadata Integration**: Consistent book identification across all pipeline stages
4. **Modularity**: Experiment with preprocessing strategies without re-ingesting raw data

---

## Data Sources

### 1. Novel Corpus

- **105 standalone billionaire romance novels** by 35 authors
- Selected from curated "Best Billionaire Romance" lists
- Each novel ≥100,000 words
- Formats: Plain text (`.txt`) and EPUB

### 2. Goodreads Metadata

Provides popularity metrics for tier grouping (Top/Middle/Trash):

| Field | Description |
|-------|-------------|
| ID | Unique book identifier |
| Author, Title | Book identification |
| Score | Average rating (1–5) |
| RatingsCount | Number of user ratings |
| ReviewsCount | Number of reviews |
| Pages | Book length |

**Statistics**:
- 97–98 books with complete metadata
- Rating distribution: Mean 3.99, Std 0.21, Range 3.26–4.42
- All books have ≥100 ratings (min: 146, mean: 65,849)

### 3. BookNLP Outputs (Optional)

Character entity extraction for stopword generation:
- 7,525 character name lines processed
- 4,444 unique character names extracted
- Used in Stage 02 to build custom stoplist

---

## Processing Decisions

### Book Matching Strategy

Matching text files to Goodreads metadata requires handling format differences:

| Challenge | Solution |
|-----------|----------|
| Author format: `sarina_bowen` vs `sarina bowen` | Normalize underscores/spaces |
| Title case differences | Case-insensitive matching |
| Minor title variations | Fuzzy matching (threshold: 0.85) |

### Encoding Handling

- Primary: UTF-8
- Fallback: Detect and fix common mojibake artifacts (e.g., `â€™` → `'`)
- Strategy: Log encoding failures, continue processing

### Error Handling

| Issue | Approach |
|-------|----------|
| Missing files | Log warning, continue |
| Encoding errors | Attempt fix, log if failed |
| Metadata mismatch | Log unmatched books, proceed with available data |
| Corrupted files | Skip with error logging |

---

## Output Structure

The ingestion stage produces:

1. **Processed Text Data**: Book texts maintaining hierarchical organization (Author → Book → Chapter)
2. **Merged Metadata**: Integrated book identifiers with Goodreads ratings
3. **Character Names**: Entity lists for stopword generation (Stage 02)

---

## Integration with Pipeline

### Downstream Dependencies

| Stage | Uses from Stage 01 |
|-------|-------------------|
| 02 Preprocessing | Text data, character names for stoplist |
| 03 Modeling | Preprocessed text (via Stage 02) |
| 10 Correlation | Book metadata for tier grouping |

---