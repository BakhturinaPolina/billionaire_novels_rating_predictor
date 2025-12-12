# Stage 1 Natural Clusters: Initial Step Report

**Date:** December 7, 2025  
**Script:** `prepare_sentence_dataframe.py`  
**Log File:** `logs/stage06_category_mapping_20251207_175435.log`

## Executive Summary

This report documents the initial step of Stage 1 natural clusters analysis: preparing a matched-only sentence-level dataframe with ratings data. The script successfully processed **680,822 sentences** from **105 books**, matched **94 books** during fuzzy matching (90% of sentences), and created a cleaned **matched-only** dataset of **612,692 sentences** from **92 books** ready for topic modeling analysis.

**Key Principle**: The final dataset contains only books present in BOTH chapters.csv and goodreads.csv. This ensures Stage 1 analysis uses only books with complete metadata, even though the BERTopic model may have been trained on all 105 books.

## Objectives

1. Load chapters and Goodreads data
2. Create `book_id` by fuzzy matching Author + Title between files
3. Merge ratings data to sentences
4. Create `rating_class` (bad/mid/good) based on quantiles
5. Calculate `position_norm` (normalized sentence position in book)
6. Generate unique `sentence_id` for each sentence
7. Output cleaned dataframe to parquet format

## Methodology

### Parameters

- **Input Files:**
  - `chapters.csv`: 680,822 sentences from 105 books
  - `goodreads.csv`: 97 books with ratings data
  
- **Processing Parameters:**
  - `min_ratings`: 100 (minimum ratings count to include book)
  - `fuzzy_threshold`: 0.85 (minimum similarity score for matching)
  - `quantiles`: (0.33, 0.66) for rating_class creation

### Processing Steps

#### Step 1: Data Loading
- Loaded chapters.csv: 680,822 sentences from 105 unique books
- Loaded goodreads.csv: 97 books with ratings
- Validated required columns present
- Filtered Goodreads by min_ratings (all 97 books passed threshold)

#### Step 2: Fuzzy Matching
- Created normalized match keys (lowercase, stripped punctuation)
- Performed fuzzy matching using SequenceMatcher
- **Results:**
  - Matched: 94 / 105 books (89.5% match rate)
  - Matched sentences: 612,692 / 680,822 (90.0%)
  - Match score statistics:
    - Mean: 0.985
    - Median: 1.000
    - Range: 0.861 - 1.000

#### Step 2b: Create Matched-Only Dataframe (Inner Join)
- Used inner join to create matched-only dataframe (only books in BOTH datasets)
- **Results:**
  - Books in chapters.csv (with book_id): 92
  - Books in goodreads.csv: 97
  - Books matched (in both): 92
  - Books only in texts (excluded): 0 (all unmatched already filtered during fuzzy matching)
  - Books only in metadata (excluded): 5 (IDs: 19561986, 19619918, 25781538, 52061964, 53491034)
  - Final matched-only dataframe: 612,692 sentences from 92 books

#### Step 3: Rating Class Creation
- Calculated quantiles on **book-level** ratings (not sentence-level)
- Quantiles: 3.920 (33rd percentile), 4.070 (66th percentile)
- Created three classes:
  - **bad**: rating < 3.920
  - **mid**: 3.920 ≤ rating ≤ 4.070
  - **good**: rating > 4.070

#### Step 4: Position Normalization
- Calculated normalized sentence position within each book (0.0 to 1.0)
- Sorted by book_id and chapter to ensure correct ordering
- Position norm range: 0.000 - 1.000
- Mean: 0.500, Std: 0.289

#### Step 5: Column Renaming
- `Sentence` → `text` (to match BERTopic training format)
- `Chapter` → `chapter_id`

#### Step 6: Sentence ID Creation
- Created unique `sentence_id` as: `{book_id}_{chapter_id}_{sentence_index}`
- Verified uniqueness: 612,692 unique IDs

#### Step 7: Final Dataframe Preparation
- Selected final columns
- Validated data quality (no null text values)
- Saved to parquet format

## Results

### Matched-Only Dataset Summary

**Final Matched-Only Dataset:**
- **Total sentences:** 612,692
- **Total books:** 92 (matched between chapters.csv and goodreads.csv)
- **Match rate:** 90.0% of sentences, 87.6% of books (92/105)

**Exclusions:**
- **11 books** from chapters.csv: Could not be matched (below 0.85 fuzzy threshold)
- **5 books** from goodreads.csv: No corresponding text data in chapters.csv
- **Total excluded sentences:** 68,130 (10.0% of original)

**Methodological Note:** The matched-only approach ensures all Stage 1 statistical analysis uses only books with complete metadata, avoiding contamination from unmatched books. The BERTopic model may have been trained on all 105 books, but analysis will use only these 92 matched books.

### Dataset Statistics

**Final Dataset:**
- **Total sentences:** 612,692
- **Total books:** 92
- **Total chapters:** 73
- **Output file:** `data/processed/sentence_df_with_ratings.parquet`

### Rating Statistics

#### Book-Level Statistics (Corrected for Sentence Weighting)
- **Total books:** 92
- **Mean rating:** 3.999
- **Median rating:** 4.020
- **Standard deviation:** 0.201
- **Range:** 3.260 - 4.420

#### Sentence-Level Statistics (Weighted by Sentence Count)
- **Mean rating:** 4.028
- **Median rating:** 4.040
- **Standard deviation:** 0.182
- **Range:** 3.260 - 4.420

**Note:** Sentence-level statistics weight books with more sentences more heavily. The difference (3.999 vs 4.028) demonstrates this weighting effect.

### Rating Class Distribution

#### Book-Level Distribution
- **bad:** 30 books (32.6%)
- **mid:** 32 books (34.8%)
- **good:** 30 books (32.6%)

#### Sentence-Level Distribution
- **bad:** 165,607 sentences (27.0%)
- **mid:** 231,358 sentences (37.8%)
- **good:** 215,727 sentences (35.2%)

### Position Statistics

- **Range:** 0.000 - 1.000
- **Mean:** 0.500
- **Standard deviation:** 0.289
- **Quartiles:**
  - Q1: 0.250
  - Q2: 0.500
  - Q3: 0.750

### Sentences per Book

- **Mean:** 6,659.7 sentences
- **Median:** 6,696.0 sentences
- **Min:** 855 sentences
- **Max:** 14,697 sentences
- **Standard deviation:** 2,755.3

## Issues and Warnings

### Unmatched Books

#### Books Only in Text Data (Excluded from Analysis)

**11 books** from chapters.csv could not be matched with Goodreads data (below 0.85 threshold):

1. **'The Tycoon's Vacation'** by 'Anne_Melody' (best score: 0.848)
2. **'The Tycoon's Proposal'** by 'Anne_Melody' (best score: 0.848)
3. **'The Tycoon's Revenge'** by 'Anne_Melody' (best score: 0.844)
4. **'The Takeover'** by 'LT_Swan' (best score: 0.800)
5. **'Brutal Secret'** by 'Laurelin_Paige' (best score: 0.787)
6. **'Reverie'** by 'Rose_Shain' (best score: 0.722)
7. **'Billionaire Bad Boy'** by 'Kendra_Little' (best score: 0.714)
8. **'Love Has a Name'** by 'Ann_Cole' (best score: 0.578)
9. **'Carter Grayson'** by 'Sandi_Lynn' (best score: 0.537)
10. **'Chase Calloway'** by 'Sandi_Lynn' (best score: 0.491)
11. **'Jamieson Finn'** by 'Sandi_Lynn' (best score: 0.409)

**Impact:** 68,130 sentences (10.0% of total) were excluded from the final dataset.

**Potential Solutions:**
- Lower fuzzy threshold (may introduce false matches)
- Manual matching for close scores (0.80-0.85 range)
- Check for alternative titles/author name variations in Goodreads data

#### Books Only in Metadata (Excluded from Analysis)

**5 books** from goodreads.csv have no corresponding text data in chapters.csv:
- Book IDs: 19561986, 19619918, 25781538, 52061964, 53491034

**Impact:** These books are excluded from the matched-only dataframe as they have no sentences to analyze.

**Note:** These books may have been excluded during text processing or may not have been in the original text corpus.

### Data Quality

- ✅ All text values are non-null
- ✅ All matched sentences have ratings
- ✅ Unique sentence_ids verified
- ✅ Position normalization working correctly

## Output Files

### Primary Output

- **File:** `data/processed/sentence_df_with_ratings.parquet`
- **Format:** Parquet (compressed columnar format)
- **Rows:** 612,692
- **Columns:**
  - `sentence_id`: Unique identifier
  - `book_id`: Goodreads book ID
  - `chapter_id`: Chapter number
  - `position_norm`: Normalized position (0.0-1.0)
  - `text`: Sentence text
  - `rating_mean`: Average Goodreads rating
  - `rating_count`: Number of ratings
  - `rating_class`: bad/mid/good
  - `Author`: Author name (reference)
  - `Book Title`: Book title (reference)

### Log File

- **File:** `logs/stage06_category_mapping_20251207_175435.log`
- Contains detailed step-by-step logging of the entire process
- Includes matched-only dataframe creation statistics

## Key Improvements Made

### Matched-Only Dataframe Approach

**Principle:** Stage 1 analysis uses only books present in BOTH text data and metadata.

- **Training**: BERTopic model can be trained on all 105 books (100% of sentences)
- **Analysis (Stage 1)**: Only use the 92 matched books (612,692 sentences, ~90%)
- **Unmatched books**: Keep them in the model, but exclude their sentences from all statistical analysis

**Implementation:**
- Changed merge from `how="left"` + filtering to `how="inner"` for explicit matched-only approach
- Added tracking of books only in texts vs only in metadata
- Enhanced logging to show which books were excluded and why

### Statistics Calculation Fix

**Issue:** Original code calculated statistics on sentence-level data, which weighted books with more sentences more heavily.

**Fix:** Statistics are now calculated on **book-level** data (one rating per book), with sentence-level statistics provided for comparison.

**Impact:** 
- Book-level mean: 3.999 (correct)
- Sentence-level mean: 4.028 (weighted, shown for comparison)

### Enhanced Logging

- Added comprehensive logging throughout all functions
- Logs saved to timestamped files in `logs/` directory
- Detailed statistics at each processing step
- Warnings for unmatched books with scores
- Explicit tracking of matched-only dataframe creation

### Code Quality

- Added validation checks for column existence
- Improved error handling
- Better progress indicators
- More detailed statistics reporting
- Clear documentation of matched-only approach

## Next Steps

1. **Review unmatched books:** Consider manual matching for books with scores 0.80-0.85
2. **Load prepared dataframe:** Use `sentence_df_with_ratings.parquet` for topic modeling
3. **Proceed to topic modeling:** Use the prepared dataframe with BERTopic
4. **Analyze rating classes:** Investigate topic distributions across bad/mid/good rating classes

## Technical Notes

### Fuzzy Matching Algorithm

- Uses `difflib.SequenceMatcher` for string similarity
- Normalizes author and title (lowercase, remove punctuation)
- Threshold: 0.85 (85% similarity required)
- Matching is one-to-one (each chapters book matches to best Goodreads match)

### Rating Class Quantiles

- Calculated on **book-level** ratings (not sentence-level)
- Ensures equal representation of books in each class
- Quantiles: 33rd and 66th percentiles
- Results in approximately equal distribution (30-32 books per class)

### Position Normalization

- Calculated as: `sentence_index / (total_sentences - 1)`
- Range: 0.0 (first sentence) to 1.0 (last sentence)
- Sorted by book_id and chapter_id to ensure correct ordering
- Useful for analyzing topic distribution across book positions

## Conclusion

The initial step successfully prepared a clean, validated **matched-only** sentence-level dataframe with ratings data. The dataset is ready for Stage 1 natural clusters analysis with:

- ✅ 612,692 sentences from 92 matched books
- ✅ Complete rating data (mean, count, class)
- ✅ Position normalization for temporal analysis
- ✅ Unique identifiers for all sentences
- ✅ Comprehensive logging and validation
- ✅ Explicit matched-only approach ensuring methodological rigor

**Methodological Note:** The 10% data loss from unmatched books is acceptable given the high match rate (90%). The strict matching threshold (0.85) ensures data quality, and the matched-only approach ensures that all Stage 1 statistical analysis (topics_per_class, ANOVA, etc.) uses only books with complete metadata, avoiding contamination from unmatched books.

