# Stage 1: Natural Clusters — Methodology Report

## Purpose

Prepare sentence-level data with topic assignments and ratings metadata for hierarchical topic exploration and category mapping.

## Research Rationale

### Why Natural Clusters Before Theory-Driven Categories?

1. **Data-driven exploration**: Understand what topics BERTopic discovered before imposing theoretical frameworks
2. **Hierarchical structure**: Explore how topics cluster naturally to inform meta-topic selection
3. **Validation**: Compare data-driven clusters with theory-driven categories (Stages 2-3)

---

## Methodology

### Data Preparation Pipeline

1. **Load chapters and Goodreads data**
2. **Fuzzy matching**: Match Author + Title between datasets (threshold: 0.85)
3. **Merge ratings**: Join ratings data to sentences
4. **Create rating classes**: bad/mid/good based on quantiles (0.33, 0.66)
5. **Position normalization**: 0.0 (book start) to 1.0 (book end)
6. **Topic assignment**: Transform sentences using BERTopic model

### Key Design Decisions

**Matched-only approach**: Final dataset contains only books present in BOTH chapters.csv and goodreads.csv. This ensures analysis uses only books with complete metadata (92 books, 612,692 sentences).

**Topic probabilities**: Include full probability vectors (soft assignments) to enable weighted analysis of multi-theme sentences.

**Duplicate label disambiguation**: 19 duplicate LLM-generated labels were disambiguated by appending topic IDs (e.g., "Unclear Relationship Feelings (T4)").

---

## Results

### Dataset Summary

| Metric | Value |
|--------|-------|
| Total sentences | 612,692 |
| Total books | 92 (matched) |
| Match rate | 90.0% of original sentences |
| Unique topics | 356 (excluding outlier -1) |

### Rating Class Distribution

| Class | Books | Sentences | % Sentences |
|-------|-------|-----------|-------------|
| bad | 30 | 165,607 | 27.0% |
| mid | 32 | 231,358 | 37.8% |
| good | 30 | 215,727 | 35.2% |

### Rating Statistics (Book-Level)

| Metric | Value |
|--------|-------|
| Mean rating | 3.999 |
| Median rating | 4.020 |
| Range | 3.26 – 4.42 |
| Quantile thresholds | 3.92, 4.07 |

### Hierarchical Topics

Hierarchical clustering of 333 topics (after noise filtering) enables:
- Dendrogram visualization
- Meta-topic selection
- Natural category discovery

---

## Integration with Pipeline

| Stage | Relationship |
|-------|--------------|
| ← Stage 08 | Receives BERTopic model with LLM labels |
| → Stage 02 | Provides sentence dataframe for taxonomy classification |
| → Stage 03 | Provides topics for Radway function mapping |

---

## Outputs

| Output | Path | Description |
|--------|------|-------------|
| Sentence dataframe | `data/processed/sentence_df_with_ratings.parquet` | Sentences with ratings metadata |
| Topic assignments | `data/processed/sentence_df_with_topics.parquet` | Sentences with topic assignments |

---

## Limitations

1. **11 unmatched books**: Below 0.85 fuzzy threshold (10% sentence loss)
2. **Duplicate labels**: 19 labels required disambiguation
3. **Hard topic assignment**: Despite soft probabilities, analysis typically uses argmax topic

---

## References

See `drafts/` for detailed processing reports:
- Initial data preparation
- Duplicate label disambiguation
- Hierarchical topics exploration
- Metadata attachment guide
- Probabilities inclusion decision
