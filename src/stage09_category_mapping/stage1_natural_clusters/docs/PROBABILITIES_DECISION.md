# Decision: Including Topic Probabilities in Topic Assignment

**Date**: December 7, 2025  
**Step**: Step 3 - Assign Topics to Sentences  
**Script**: `assign_topics_to_sentences.py`  
**Decision**: ✅ **Include probabilities** (`--include-probs` flag used)

---

## The Choice

At Step 3, we had two options for how to store topic assignments:

```python
topics, probs = topic_model.transform(docs)

df["topic"] = topics              # hard assignment (always)
# optional:
df["topic_prob"] = probs         # soft assignment (full distribution)
```

### Option 1: Hard Assignments Only (Without Probabilities)

If we **do not** keep probabilities, we effectively treat each sentence as belonging to **one and only one topic**:

- `df["topic"]` is the **argmax** topic (the one with highest probability)
- All later counts are **hard counts**:
  - Book A has 100 sentences with topic 7 → that's it, no nuance
- This is simpler and usually fine, especially with many short sentences

**Downside**: Sentences that clearly mix two themes (e.g., "He signed the merger and then kissed her passionately") are forced into just one topic.

### Option 2: Include Probabilities (Soft Assignments)

If we **do** keep probabilities, we keep the **full probability vector** over topics for each sentence:

- `probs[i]` might look like `[0.05, 0.7, 0.1, 0.0, …]`
- We can then build **soft** book-level topic distributions:

```python
import numpy as np
import pandas as pd

# Assume probs is a 2D numpy array: n_sentences × n_topics
n_topics = probs.shape[1]

df_probs = pd.DataFrame(
    probs,
    columns=[f"topic_{i}" for i in range(n_topics)]
)
df_with_probs = pd.concat([df.reset_index(drop=True), df_probs], axis=1)

# Soft counts per book: sum probabilities instead of counting argmax
book_topic_soft = (
    df_with_probs
    .groupby("book_id")[df_probs.columns]
    .sum()
)

# Convert to proportions per book
book_topic_soft = book_topic_soft.div(book_topic_soft.sum(axis=1), axis=0)
```

**Benefits**:
- A sentence that's 0.6 "Luxury" + 0.4 "Emotional introspection" contributes **to both**
- Book-level topic proportions become smoother and often **more stable**, especially if sentences are noisy or short
- Enables more advanced analyses (e.g., topic entropy, uncertainty, "multi-theme" sentences)

---

## Our Decision: Include Probabilities

We chose to **include probabilities** for the following reasons:

1. **Mixed-content sentences**: Romance novels often contain sentences that blend multiple themes (e.g., business negotiations mixed with romantic tension, luxury settings with emotional introspection)

2. **Statistical stability**: Soft assignments provide smoother, more stable book-level topic distributions, which is important for statistical comparisons (ANOVA, effect sizes)

3. **Flexibility**: Having probabilities allows us to:
   - Run initial analyses with hard assignments (simple)
   - Rerun with soft assignments as a robustness check
   - Explore topic uncertainty and entropy
   - Identify sentences with high topic ambiguity

4. **Future analyses**: Probabilities enable more nuanced downstream analyses that may be valuable for understanding the relationship between topic distributions and book quality

---

## Implementation Results

**Execution Date**: December 7, 2025, 18:23:30 - 20:02:45  
**Processing Time**: ~1 hour 39 minutes (inference only, no retraining)

### Input
- **Sentences**: 612,692 sentences from 92 matched books
- **Model**: BERTopic model with 368 topics (excluding outlier -1), 369 custom labels
- **Model suffix**: `_with_categories`

### Output
- **File**: `data/processed/sentence_df_with_topics.parquet`
- **File size**: 1,312.94 MB (1.3 GB)
- **Columns added**:
  - `topic`: Hard topic assignment (argmax, integer)
  - `topic_prob`: Full probability distribution (array/list of floats)

### Topic Assignment Statistics
- **Unique topics assigned**: 369 (includes outlier topic -1)
- **Non-outlier topics**: 368
- **Outlier sentences (topic -1)**: 8,051 (1.31% of sentences)

**Top 10 topics by frequency** (hard assignments):
1. Topic 8: 19,300 sentences (3.15%)
2. Topic 7: 17,710 sentences (2.89%)
3. Topic 1: 12,276 sentences (2.00%)
4. Topic 2: 12,098 sentences (1.97%)
5. Topic 15: 11,865 sentences (1.94%)
6. Topic 337: 10,240 sentences (1.67%)
7. Topic 20: 10,213 sentences (1.67%)
8. Topic -1 (outlier): 8,051 sentences (1.31%)
9. Topic 50: 7,465 sentences (1.22%)
10. Topic 145: 7,443 sentences (1.21%)

### Storage Considerations

**File size impact**: The inclusion of probabilities increased the output file size significantly:
- Without probabilities: Estimated ~200-300 MB
- With probabilities: 1,312.94 MB (1.3 GB)
- **Increase**: ~4-6x larger

This is expected because:
- Each sentence now stores a probability vector of length 369 (one per topic)
- For 612,692 sentences: 612,692 × 369 × 8 bytes (float64) ≈ 1.8 GB theoretical
- Actual size (1.3 GB) is smaller due to parquet compression

**Trade-off**: The larger file size is acceptable given:
- The value of having soft assignments for statistical analyses
- Parquet compression helps reduce storage
- We can always extract hard assignments from probabilities if needed

---

## Usage in Downstream Analyses

### Hard Assignments (Current Default)
For most analyses, we can use the `topic` column directly:

```python
# Hard counts per book
book_topic_counts = (
    df.groupby(["book_id", "topic"])
      .size()
      .unstack(fill_value=0)
)
```

### Soft Assignments (Available When Needed)
For more nuanced analyses, we can use the `topic_prob` column:

```python
# Extract probability vectors
probs = np.array(df["topic_prob"].tolist())  # shape: (n_sentences, n_topics)

# Soft counts per book: sum probabilities
book_topic_soft = (
    df.groupby("book_id")
      .apply(lambda x: probs[x.index].sum(axis=0))
)

# Convert to proportions
book_topic_soft = book_topic_soft.div(book_topic_soft.sum(axis=1), axis=0)
```

### Recommended Approach
1. **Initial analyses**: Use hard assignments (`topic` column) for simplicity
2. **Robustness checks**: Rerun key analyses with soft assignments (`topic_prob`) to verify stability
3. **Advanced analyses**: Use probabilities for entropy calculations, uncertainty metrics, and multi-theme sentence identification

---

## Summary

**Decision**: ✅ Include probabilities (`--include-probs`)

**Rationale**:
- Enables soft book-level topic distributions
- Provides more stable statistical comparisons
- Handles mixed-content sentences better
- Allows flexibility for future analyses

**Trade-offs**:
- ✅ More nuanced and statistically stable analyses
- ✅ Flexibility for robustness checks
- ❌ Larger file size (~1.3 GB vs ~300 MB)
- ❌ Slightly slower processing (but still acceptable)

**Impact**: This decision affects all downstream analyses. We can use hard assignments for simplicity, but have the option to use soft assignments when needed for more nuanced or robust statistical comparisons.

---

## Next Steps

1. **Step 4**: Use `sentence_df_with_topics.parquet` for hierarchical topic exploration
2. **Step 6**: When computing book-level topic distributions, consider:
   - Primary analysis: Hard assignments (simple, interpretable)
   - Robustness check: Soft assignments (more stable, handles mixed sentences)
3. **Step 8**: For ANOVA analyses, compare results using both hard and soft assignments to verify robustness

---

## References

- **Script**: `assign_topics_to_sentences.py`
- **Log file**: `results/stage09_category_mapping/stage1_natural_clusters/logs/stage09_category_mapping_assign_topics_20251207_182330.log`
- **Output file**: `data/processed/sentence_df_with_topics.parquet`
- **Related documentation**: `README.md` (Step 3 section)

