# Stage 06: Topic Exploration — Methodology Report

## Purpose

Explore and evaluate retrained BERTopic models by attaching multiple topic representations and computing coherence/diversity metrics to inform downstream analysis.

## Research Rationale

### Why Multiple Representations?

BERTopic's default c-TF-IDF representation (Main) captures statistical word importance but may miss semantically coherent or diverse word selections. We implement four complementary representations:

1. **Main (c-TF-IDF)**: Default statistical representation based on term frequency-inverse document frequency within topics
2. **KeyBERT**: Extracts keywords using BERT embeddings to capture semantic similarity
3. **POS (Part-of-Speech)**: Filters to content words (nouns, verbs, adjectives) to improve interpretability
4. **MMR (Maximal Marginal Relevance)**: Balances relevance and diversity to reduce redundancy

Each representation serves different analytical purposes:
- **Main**: Baseline statistical representation
- **KeyBERT**: Semantic coherence for close reading
- **POS**: Interpretability for human labeling
- **MMR**: Diversity for exploratory analysis

### Evaluation Metrics

We compute two complementary metrics:

1. **Coherence (c_v)**: Measures semantic coherence of topic word lists using Gensim's c_v coherence metric, which evaluates word co-occurrence patterns in sliding windows
2. **Topic Diversity**: Ratio of unique words to total words across all topics, measuring lexical diversity

These metrics help identify which representations produce the most interpretable and diverse topics for downstream analysis.

---

## Results

### Metrics Summary

For the selected model (`paraphrase-MiniLM-L6-v2`, Pareto rank 1, 368 topics):

| Representation | Coherence (c_v) | Topic Diversity |
|---------------|-----------------|------------------|
| Main          | 0.404           | 0.602            |
| KeyBERT       | 0.278           | 0.645            |
| POS           | 0.315           | 0.692            |
| MMR           | 0.260           | 0.756            |

### Interpretation

1. **Main representation** achieves highest coherence (0.404), indicating strong statistical word co-occurrence patterns
2. **MMR representation** achieves highest diversity (0.756), providing the most lexically diverse topic word lists
3. **POS representation** balances coherence (0.315) and diversity (0.692), making it suitable for human interpretation and labeling
4. **KeyBERT** shows lower coherence (0.278) but moderate diversity (0.645), suggesting semantic similarity may not always align with statistical patterns

### Representation Selection for Downstream Analysis

Based on these metrics:
- **For LLM labeling (Stage 08)**: POS representation recommended (interpretable content words)
- **For exploratory analysis**: MMR representation recommended (diverse word lists)
- **For statistical validation**: Main representation recommended (highest coherence)

---

## Integration with Pipeline

| Stage | Relationship |
|-------|--------------|
| ← Stage 05 | Receives retrained models |
| → Stage 07 | Provides topics for quality analysis |
| → Stage 08 | POS representation used for LLM labeling |
| → Stage 09 | Multiple representations available for category mapping |

---

## References

- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv:2203.05794*.
