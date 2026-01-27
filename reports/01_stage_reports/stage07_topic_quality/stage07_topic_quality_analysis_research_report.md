# Stage 07: Topic Quality Analysis — Research Report

## Purpose

Identify candidate noisy topics through exploratory data analysis (EDA) before proceeding to LLM-generated topic labeling. This stage serves as a quality control checkpoint.

## Research Rationale

### Why Quality Analysis Before Labeling?

1. **Efficiency**: Identifying noisy topics before LLM labeling saves computational costs
2. **Quality Control**: Flags topics that may not be suitable for downstream analysis
3. **Manual Review**: Provides structured data for human inspection
4. **Non-destructive**: Flags topics without modifying the model structure

### Quality Dimensions

The analysis evaluates topics across three dimensions:

1. **Topic size**: Number of documents assigned to each topic
2. **POS representation**: Quality and quantity of part-of-speech words in topic representations
3. **Semantic coherence**: Per-topic coherence scores using the c_v metric

---

## Methodology

### Noise Detection Criteria

Topics flagged as noise candidates based on:

1. **Few POS words** (< 3): Topics with insufficient content words for interpretation
2. **Low coherence** (< 0.0): Topics with poor semantic coherence
3. **Small size** (< 30 documents): Topics below minimum size threshold

A topic is flagged as a `noise_candidate` if it meets any of these conditions.

### Coherence Computation

Uses Gensim's c_v coherence metric:
- **Segmentation**: Sliding window approach over the corpus
- **Probability estimation**: Word co-occurrence probabilities
- **Score range**: Typically 0.0 to 1.0 (higher = more coherent)

Computed using the same Gensim dictionary built from OCTIS corpus used during training.

---

## Results

### Overall Statistics

For `paraphrase-MiniLM-L6-v2` model (368 topics):

| Metric | Value |
|--------|-------|
| Total topics (excluding -1) | 368 |
| Candidate noisy topics | 13 (3.5%) |
| Topics with POS words < 10 | 20 (5.4%) |
| Topics with valid coherence scores | 361 (98.1%) |

### Noise Candidate Characteristics

**Identified patterns**:

1. **Empty or near-empty representations**: Topics 17, 18, 132, 186, 224 have empty word lists, suggesting uninterpretable content
2. **Single-word topics**: Topics with only 1-2 POS words (141, 183, 241, 262, 347) lack semantic richness
3. **Large but incoherent topics**: Topics 17 and 18 are among the largest (2,062-2,064 documents) but have no valid POS words, suggesting catch-all clusters

### Key Findings

- **94.6-96.5% of topics** meet quality thresholds
- **Noise is concentrated**: A small number of topics (13-20) require special attention
- **Large topics can be noisy**: Some of the largest topics are actually noise candidates
- **Coherence computation is robust**: 98.1% of topics have computable coherence scores

---

## Implications for Downstream Analysis

### For LLM Labeling (Stage 08)

- 13-20 topics (3.5-5.4%) flagged for special handling
- These topics may benefit from:
  - Manual review before LLM labeling
  - Special labeling strategies (e.g., "OTHER" or "NOISE")
  - Exclusion from certain analyses

### For Model Quality

- Most topics (94.6-96.5%) have acceptable quality metrics
- The model successfully clusters most documents into interpretable topics
- Noise candidates represent a small but significant portion requiring attention

---

## Integration with Pipeline

| Stage | Relationship |
|-------|--------------|
| ← Stage 06 | Receives explored topics with multiple representations |
| → Stage 08 | Provides noise candidate flags to inform labeling strategy |

---

## References

- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv:2203.05794*.
