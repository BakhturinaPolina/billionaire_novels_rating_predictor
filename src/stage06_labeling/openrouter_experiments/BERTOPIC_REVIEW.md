# BERTopic Code Review

## Issues Found

### 1. **Critical: Incorrect Usage of `get_document_info()`**

**Problem**: The code calls `topic_model.get_document_info(batch_docs)` with documents from a CSV file that may not be the same documents used during `fit_transform()`.

**According to BERTopic Documentation**:
- `get_document_info(docs)` expects the **same documents** that were used during training
- It relies on internal arrays (`topics_`, `probabilities_`) created during `fit_transform()`
- When called with new documents, these arrays don't match, causing: `"All arrays must be of the same length"`

**Error Location**: 
- Lines 648, 728 in `generate_labels_openrouter.py`
- Error message: `"All arrays must be of the same length"`

**Solution**: 
For new documents (not used during training), use `transform()` instead:

```python
# Instead of:
batch_doc_info = topic_model.get_document_info(batch_docs)

# Use:
topics, probs = topic_model.transform(batch_docs)
# Then manually create document info DataFrame
```

### 2. **Recommended Approach: Use `get_representative_docs()`**

According to BERTopic docs, `get_representative_docs()` is the correct method to get representative documents for topics. It returns pre-computed representative documents (typically 3-5 per topic) that were identified during training.

**Current Code**: Already has a fallback to `get_representative_docs()` (line 895+), which is correct.

**Recommendation**: Make `get_representative_docs()` the primary method when working with new documents, and only use `get_document_info()` when you have the exact same documents used during training.

### 3. **Code Structure Issues**

The batch processing logic (lines 621-821) is complex and error-prone. The current approach tries to:
1. Load documents from CSV in batches
2. Call `get_document_info()` on each batch
3. Accumulate results

**Better Approach**:
1. Use `get_representative_docs()` directly (simpler, more reliable)
2. If more documents are needed, use `transform()` on new documents to get topics/probabilities
3. Filter and sort manually

## Recommended Fixes

### Fix 1: Replace `get_document_info()` with `transform()` for new documents

```python
# In extract_representative_docs_per_topic(), replace:
batch_doc_info = topic_model.get_document_info(batch_docs)

# With:
topics, probs = topic_model.transform(batch_docs)
# Create DataFrame manually
batch_doc_info = pd.DataFrame({
    'Document': batch_docs,
    'Topic': topics,
    'Probability': probs if probs is not None else [0.0] * len(batch_docs)
})
```

### Fix 2: Simplify by using `get_representative_docs()` as primary method

The code already falls back to `get_representative_docs()` when `get_document_info()` fails. Consider making this the primary approach:

```python
def extract_representative_docs_per_topic(
    topic_model: BERTopic,
    max_docs_per_topic: int = 40,
    documents: list[str] | None = None,
    csv_path: Path | None = None,
    batch_size: int = 50000,
) -> dict[int, list[str]]:
    """Extract representative documents using BERTopic's built-in method."""
    
    # Primary method: Use get_representative_docs() (works with any model)
    try:
        rep_docs_dict = topic_model.get_representative_docs()
        topic_to_docs = {}
        
        for topic_id, docs in rep_docs_dict.items():
            if topic_id == -1:  # Skip outliers
                continue
            # Limit to max_docs_per_topic
            topic_to_docs[topic_id] = docs[:max_docs_per_topic]
        
        LOGGER.info(
            "Extracted representative docs via get_representative_docs() for %d topics",
            len(topic_to_docs),
        )
        return topic_to_docs
        
    except Exception as e:
        LOGGER.warning("get_representative_docs() failed: %s", e)
        return {}
```

### Fix 3: Handle the case when documents match training documents

If you need to verify whether documents match training documents, check the length:

```python
# Check if documents match training documents
if hasattr(topic_model, 'topics_'):
    training_doc_count = len(topic_model.topics_)
    if len(batch_docs) == training_doc_count:
        # Documents match - can use get_document_info()
        batch_doc_info = topic_model.get_document_info(batch_docs)
    else:
        # New documents - use transform()
        topics, probs = topic_model.transform(batch_docs)
        # Create DataFrame...
```

## BERTopic Best Practices

According to the documentation:

1. **For representative documents**: Use `get_representative_docs()` - returns pre-computed representative docs
2. **For document info on training docs**: Use `get_document_info(docs)` with the same docs used in `fit_transform()`
3. **For new documents**: Use `transform(docs)` to get topics/probabilities, then process manually
4. **For topic information**: Use `get_topic_info()` - returns DataFrame with topic statistics

## Summary

The main issue is using `get_document_info()` with documents that weren't used during training. The fix is to either:
1. Use `get_representative_docs()` (simplest, recommended)
2. Use `transform()` for new documents and create document info manually
3. Only use `get_document_info()` when you have the exact training documents

The current fallback to `get_representative_docs()` is working (as seen in the logs), but the batch processing approach is failing unnecessarily.

