# Fix for NaN Probabilities in Topic Probability Files

## Problem

The `book_topic_probs.parquet` and `chapter_topic_probs.parquet` files contained NaN values for ~66% of probability entries, affecting all 368 topics and 61 books.

## Root Cause

BERTopic's `transform()` method can return NaN values for certain topics/documents. When these NaN values are aggregated using `.mean(axis=0)`, the result is also NaN. The aggregation functions did not handle NaN values before computing means.

## Solution

**Replace NaN values with 0.0 BEFORE aggregation** to ensure clean data throughout the pipeline. This ensures:
1. No NaNs propagate to aggregation
2. Regular `.mean()` can be used (no need for `nanmean`)
3. Final output is guaranteed to have zero NaNs

### Changes Made

1. **Early NaN cleaning** (line ~552):
   - After loading/transforming probabilities, check for NaN values
   - If found, replace all NaN with 0.0 immediately using `np.nan_to_num()`
   - This ensures the probability matrix is clean before any aggregation

2. **`aggregate_to_book_level()` function** (line ~405):
   - Check for NaN values in each book's probability slice
   - Replace NaN with 0.0 BEFORE computing mean
   - Use regular `.mean()` on clean data
   - Add validation to ensure no NaNs remain after aggregation

3. **`aggregate_to_chapter_level()` function** (line ~437):
   - Check for NaN values in each chapter's probability slice
   - Replace NaN with 0.0 BEFORE computing mean
   - Use regular `.mean()` on clean data
   - Add validation to ensure no NaNs remain after aggregation

4. **Final validation** (line ~571):
   - After aggregation, verify that both output DataFrames have zero NaN values
   - Raise error if any NaNs are found (should never happen)
   - Log success message confirming clean output

## Regenerating the Data

To regenerate the topic probability files with the fix:

```bash
# Activate virtual environment
source venv/bin/activate

# Regenerate topic probabilities (this will use cached transform outputs if available)
python src/stage10_correlation_analysis/data_preparation/03_generate_topic_probabilities_final.py \
    --sentence-df data/processed/sentence_df_with_topics.parquet \
    --model-path models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_radway_mappings \
    --output-dir results/stage10_correlation_analysis/data_preparation \
    --book-id-source existing \
    --exclude-book-ids notebooks/07_analysis/statistical_analysis/excluded_book_ids.csv

# If you want to force regeneration (ignore cache), add --no-cache flag
```

The script will:
1. Use cached transform outputs if available (saves ~2 hours of computation)
2. Regenerate book-level and chapter-level aggregations with the fixed code
3. Output new parquet files to `results/stage10_correlation_analysis/data_preparation/topic_probabilities/`

## Verification

After regeneration, verify the fix worked:

```python
import pandas as pd

# Check book-level probabilities
book_probs = pd.read_parquet('results/stage10_correlation_analysis/data_preparation/topic_probabilities/book_topic_probs.parquet')
print(f"Total rows: {len(book_probs):,}")
print(f"NaN count: {book_probs['prob'].isna().sum():,}")
print(f"NaN percentage: {book_probs['prob'].isna().sum() / len(book_probs) * 100:.2f}%")

# Should show 0 NaN values (or very close to 0)
```

## Notes

- **NaN values are replaced with 0.0**, not ignored. This means:
  - If a topic has NaN probability for all sentences in a book, the book-level probability for that topic will be 0.0
  - This is the correct behavior: NaN means "no probability assigned", which we interpret as 0.0
- The cached transform outputs (from BERTopic.transform) may contain NaNs, but they will be cleaned during processing
- The final output is guaranteed to have **zero NaN values** - the script will raise an error if any are found
- If you want to regenerate from scratch, delete the cache file or use `--no-cache` flag

