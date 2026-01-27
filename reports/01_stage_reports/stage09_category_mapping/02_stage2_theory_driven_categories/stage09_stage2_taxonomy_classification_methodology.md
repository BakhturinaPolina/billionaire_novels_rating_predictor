# Stage 2: Theory-Driven Categories — Methodology Report

## Purpose

Map BERTopic topics to the Romance Corpus Topic Taxonomy using zero-shot classification, enabling theory-driven statistical analysis of thematic patterns.

## Research Rationale

### Why Theory-Driven Categories?

1. **Interpretability**: Raw topic keywords require domain expertise to interpret; taxonomy categories provide standardized, meaningful labels
2. **Comparability**: Fixed taxonomy enables comparison across studies and datasets
3. **Statistical Analysis**: Taxonomy categories serve as dependent variables for hypothesis testing
4. **Literature Alignment**: Categories map to established romance fiction scholarship

### Why Zero-Shot Classification?

Zero-shot classification via LLM avoids:
- Manual labeling of 368 topics
- Training data requirements
- Domain-specific model fine-tuning

The approach leverages:
- Stage 08 LLM labels and scene summaries
- Representative document snippets from BERTopic
- Structured prompts with taxonomy definitions

---

## Romance Corpus Topic Taxonomy

The taxonomy consists of **8 main groups with 30+ specific categories**:

| Group | Categories |
|-------|------------|
| **1. Embodied & Sensory Experience** | Body parts, pain/injury, physical activity |
| **2. Sexuality, Attraction & Intimacy** | Attraction, kissing, explicit acts, aftercare |
| **3. Emotions, Cognition & Inner Life** | Positive/negative emotions, ambivalence, values |
| **4. Relationship Trajectory** | Meeting, bonding, secrets, conflict, HEA |
| **5. Social World Outside Couple** | Family, friends, community |
| **6. Work, Wealth, Status** | Hero/heroine work, money, institutions |
| **7. Conflict, Risk & Harm** | Interpersonal conflict, violence, danger |
| **8. Spaces, Time, Activities** | Domestic/public spaces, objects, time |

**Special category**: `noise` — Boilerplate, technical artifacts, or paratext.

---

## Methodology

### Classification Pipeline

1. **Load Stage 08 labels**: Keywords, labels, scene summaries, primary/secondary categories
2. **Extract representative snippets**: Load BERTopic model, extract top documents per topic
3. **Zero-shot classification**: Prompt Mistral-Nemo with topic context and taxonomy definitions
4. **Validation**: Verify taxonomy IDs against fixed taxonomy list
5. **Noise handling**: Respect `is_noise` from Stage 08

### Prompt Design

The classification prompt includes:
- Full taxonomy definitions with category codes (e.g., "4.2: Bonding, Everyday Intimacy & Growth")
- Topic keywords and LLM-generated labels
- Representative document snippets (reranked by centrality)
- JSON-only output constraints

### Quality Control

- **Taxonomy ID validation**: All IDs checked against fixed taxonomy
- **Confidence scoring**: Low/medium/high confidence for each classification
- **Rationale capture**: 1-3 sentences explaining classification decision
- **Fallback assignments**: Heuristic fallbacks for failed classifications

---

## Results

### Classification Coverage

For `paraphrase-MiniLM-L6-v2` model (368 topics):

| Metric | Value |
|--------|-------|
| Topics classified | 361 (98.1%) |
| Topics with main category | 361 |
| Topics with secondary category | ~85% |
| Unique taxonomy categories used | 28 |
| High confidence classifications | ~36% |

### Category Distribution

**Most common categories:**

| Category | Count | Description |
|----------|-------|-------------|
| 4.2 | 47 | Bonding, Everyday Intimacy & Growth |
| 2.3 | 38 | Explicit Sexual Acts |
| 4.4 | 35 | Conflict, Distance & Breakup Threats |
| 3.2 | 32 | Negative Emotions & Distress |
| 5.1 | 28 | Family & Kinship |

### Statistical Analysis Results

Kruskal-Wallis tests comparing category proportions across rating classes (bad/mid/good):

| Category | p-value | η² (Effect Size) | Significance |
|----------|---------|------------------|--------------|
| 5.3: Community, Norms & Social Events | 0.029 | 0.070 (medium) | ✅ |
| 6.2: Heroine's Work & Professional Identity | 0.047 | 0.057 (small-medium) | ✅ |
| 3.4: Beliefs, Values & Moral Reflection | 0.048 | 0.048 (small) | ✅ |

**Key Finding**: 24 out of 27 categories show no significant differences, indicating thematic content is largely consistent across rating classes.

---

## Integration with Pipeline

| Stage | Relationship |
|-------|--------------|
| ← Stage 08 | Receives topic labels and scene summaries |
| → Stage 03 (Radway) | Provides taxonomy context for Radway classification |
| → Stage 10 | Category proportions used in correlation analysis |

---

## Recommended Model

**Use**: `model_1_with_llm_labels_and_metadata_disambiguated.pkl`

This model contains:
- 368 topics with LLM labels
- 361 taxonomy mappings embedded
- 98.1% taxonomy coverage

See [Model Comparison Report](stage09_stage2_model_comparison.md) for detailed comparison.

---

## References

- Radway, J. A. (1984). *Reading the Romance: Women, Patriarchy, and Popular Literature*.
- Stage 08 LLM Labeling methodology
