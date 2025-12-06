# Performance Regression Analysis

## Timeline of Changes

### Before Performance Issues (Fast)
- **Commit `acb738b`** (Dec 4): "Add multi-model comparison for OpenRouter topic labeling"
  - Simple keyword-based labeling
  - No document loading from CSV
  - Fast API calls with just keywords

### Performance Degradation Started
1. **Commit `5ff8acc`** (Dec 4, 19:03): "Add representative snippets feature"
   - Added `extract_representative_docs_per_topic()` function
   - Started loading documents from CSV in batches
   - Added snippet formatting and inclusion in prompts
   - **Impact**: Slower due to CSV batch processing

2. **Commit `09a3f7c`** (Dec 4, 19:30): "Fix BERTopic get_representative_docs() API call"
   - Fixed API usage but still loading documents
   - **Impact**: Minor, just a bug fix

3. **Commit `7ec9961`** (Dec 5, 17:06): "Wire in centrality reranker and update snippet defaults"
   - Added centrality reranking (semantic similarity computation)
   - Increased snippets from **6 to 15**
   - Increased snippet chars from **200 to 1200**
   - **Impact**: Major slowdown - reranking + larger prompts = more tokens + slower processing

### Current State (Slow)
- Loading 40 docs per topic from CSV (batch processing)
- Centrality reranking for snippet selection
- 15 snippets × 1200 chars = ~18,000 chars per prompt
- Much larger API payloads

## Performance Impact

### Before (Commit `acb738b` or earlier)
- **Prompt size**: ~500-1000 tokens (keywords only)
- **Processing time**: ~0.5-1s per topic
- **No document loading**: Instant
- **Total for 15 topics**: ~10-15 seconds

### After (Current)
- **Prompt size**: ~3000-4000 tokens (keywords + 15 snippets)
- **Processing time**: ~2-6s per topic (including document loading)
- **Document loading**: 2-3 minutes for all topics (CSV batch processing)
- **Centrality reranking**: Additional ~0.5-1s per topic
- **Total for 15 topics**: ~3-4 minutes

## Recommended Recovery Points

### Option 1: **Commit `acb738b`** (RECOMMENDED)
**"Add multi-model comparison for OpenRouter topic labeling"**

**Pros:**
- Simple, fast keyword-only labeling
- No document loading overhead
- Smaller prompts = faster API calls
- Lower token costs

**Cons:**
- No snippets = potentially less precise labels
- But labels were still good quality before snippets

**Recovery command:**
```bash
git checkout acb738b -- src/stage06_labeling/openrouter_experiments/generate_labels_openrouter.py
git checkout acb738b -- src/stage06_labeling/openrouter_experiments/main_openrouter.py
```

### Option 2: **Commit `5ff8acc^`** (Before snippets)
**Right before snippets were added**

**Pros:**
- Latest prompt improvements
- No document loading complexity
- Fast performance

**Cons:**
- Might miss some prompt optimizations from later commits

**Recovery command:**
```bash
git checkout 5ff8acc^ -- src/stage06_labeling/openrouter_experiments/generate_labels_openrouter.py
git checkout 5ff8acc^ -- src/stage06_labeling/openrouter_experiments/main_openrouter.py
```

### Option 3: **Hybrid Approach** (Keep snippets but simplify)
**Keep commit `09a3f7c` but revert `7ec9961`**

**Pros:**
- Keep snippets feature (3-5 pre-computed docs from BERTopic)
- Remove slow CSV batch processing
- Remove centrality reranking
- Use fewer snippets (3-5 instead of 15)

**Cons:**
- Need manual code changes

**Changes needed:**
1. Remove CSV batch processing code
2. Use only `get_representative_docs()` (3-5 docs, fast)
3. Remove centrality reranking
4. Reduce snippet count to 3-5
5. Reduce snippet char limit to 200-400

## Recommendation

**Go back to commit `acb738b`** - This was the last commit before document loading complexity was added. The labels were still high quality with just keywords, and the performance was much better.

If you want to keep snippets but improve performance, use **Option 3** (Hybrid) - keep the simple snippet extraction but remove the slow parts.

## Verification

To verify which commit had good performance, check the logs:
- Look for generation times in logs before Dec 4, 19:03
- Compare with current logs showing 2-6s per topic

