# Stage 1: Natural Clusters (Hierarchical Topics)

## Overview

**Goal**: Discover data-driven topic groupings without theoretical priors by using BERTopic's hierarchical topics feature. Identify which natural meta-topics are associated with book quality (bad/mid/good ratings).

**Status**: 🚧 **In Progress** (Step 1 ✅, Step 2 ✅, Step 3+ pending)

## Progress Summary

- ✅ **Step 1**: Prepare Sentence-Level DataFrame - **Complete**
  - Created matched-only dataset: 612,692 sentences from 92 books
  - All books have complete rating data and rating classes
  
- ✅ **Step 2**: Load BERTopic Model and Attach LLM Labels - **Complete**
  - Model loaded successfully (368 topics, 369 labels)
  - Labels verified and accessible
  
- ⏳ **Step 3+**: Topic Assignment and Analysis - **Pending**

## Important: Matched-Only Analysis

**Key Principle**: Stage 1 analysis uses only books present in BOTH the text data (chapters.csv) and metadata (goodreads.csv).

- **Training**: BERTopic model can be trained on all 105 books (100% of sentences)
- **Analysis (Stage 1)**: Only use the 92 matched books (~612,692 sentences, ~90%)
- **Unmatched books**: Keep them in the model, but exclude their sentences from all statistical analysis (topics_per_class, ANOVA, etc.)

This ensures methodological rigor: the unmatched 10% of sentences helped shape the topic space during training, but are excluded from rating-based comparisons to avoid contamination.

## Inputs Required

1. **Trained BERTopic Model**
   - Path: `models/retrained/paraphrase-MiniLM-L6-v2/model_1_with_categories/`
   - Format: Native BERTopic safetensors directory
   - Topic count: **368 topics** (excluding outlier topic -1)
   - Labels: **369 labels** (includes topic -1) already integrated
   - Contains: Category labels integrated via Stage 06 labeling pipeline
   - See `docs/MODEL_VERSIONING.md` for detailed model version information
   - **Verified**: Model loads successfully with labels accessible via `custom_labels_` and `get_topic_info()`

2. **Sentence-Level DataFrame**
   - Source: `data/processed/chapters.csv`
   - Rows: 680,822 sentences
   - Columns: `Author`, `Book Title`, `Chapter`, `Sentence`
   - Missing values: None
   - Author format: Underscore-separated (e.g., `Ann_Cole`, `sarina_bowen`)
   - Book Title format: Title case with punctuation
   - Chapter: Integer chapter numbers
   - Sentence: Raw sentence text (matches BERTopic training format)
   
   **Required output columns**:
     - `sentence_id`: Unique identifier (optional but helpful)
     - `book_id`: Book identifier (derived from Author + Title)
     - `chapter_id`: Chapter index (int)
     - `position_norm`: Normalized position in book [0,1]
     - `text`: Sentence text (from `Sentence` column)
     - `rating_mean`: Mean user rating (1-5 scale, from goodreads.csv)
     - `rating_count`: Number of ratings (from goodreads.csv)
     - `rating_class`: "bad" / "mid" / "good" (3-category quality label)

3. **Goodreads Metadata**
   - Source: `data/processed/goodreads.csv`
   - Rows: 97 books
   - Columns: `ID`, `Author`, `Title`, `Score`, `RatingsCount`, `ReviewsCount`, `Pages`, etc.
   - Rating distribution: Mean 3.99, Std 0.21, Range 3.26-4.42
   - RatingsCount: All books have >= 100 ratings (min: 146, mean: 65,849)
   - Author format: Lowercase with spaces (e.g., `sarina bowen`, `j. clare`)
   - Title format: Lowercase (e.g., `brooklynaire`, `hard hitter`)
   - **Note**: Author/Title format differs from chapters.csv - requires fuzzy matching

4. **LLM Labels/Descriptions**
   - Location: `results/stage06_labeling_openrouter/labels_pos_openrouter_mistralai_mistral-nemo_romance_aware_paraphrase-MiniLM-L6-v2_reasoning_high.json`
   - Structure: `{topic_id: {label, keywords, scene_summary}}`
   - Total topics: 361
   - Also stored in model's `topics.json` under `topic_labels` key
   - Integration: Labels are already integrated into `model_1_with_categories` via `custom_labels_` attribute

## Implementation Steps

### Step 1: Prepare Sentence-Level DataFrame

**File**: `prepare_sentence_dataframe.py`

**Status**: ✅ **Implemented**

**Usage**:
```bash
python -m src.stage06_labeling.category_mapping.stage1_natural_clusters.prepare_sentence_dataframe \
    --chapters data/processed/chapters.csv \
    --goodreads data/processed/goodreads.csv \
    --output data/processed/sentence_df_with_ratings.parquet \
    --min-ratings 100 \
    --fuzzy-threshold 0.85 \
    --quantiles 0.33 0.66
```

**Tasks**:
1. Load `chapters.csv` (680,822 sentences from 105 books) and `goodreads.csv` (97 books)
2. **Fuzzy matching**: Match books between files using normalized Author + Title
   - Uses `difflib.SequenceMatcher` for similarity scoring
   - Default threshold: 0.85 (configurable)
   - Handles format differences (underscores vs spaces, case differences)
3. **Create "matched only" dataframe**: Use inner join to keep only books present in BOTH datasets
   - Tracks which books were dropped (only in texts vs only in metadata)
   - Ensures Stage 1 analysis uses only books with complete metadata
   - Result: 92 matched books (612,692 sentences, ~90% of total)
   - Note: 94 books matched during fuzzy matching, but 92 books present in both datasets after inner join
4. Filter books with `rating_count >= 100` (configurable, default: 100)
   - Applied to goodreads.csv before matching
5. Create `rating_class` column using quantile-based buckets (only for matched books):
   - `bad`: rating_mean < 33rd percentile
   - `mid`: 33rd-66th percentile
   - `good`: > 66th percentile
6. Add `position_norm`: Normalized sentence position in book [0, 1]
   - Calculated as: `sentence_index / (total_sentences - 1)`
7. Add `sentence_id`: Unique identifier (`{book_id}_{chapter}_{sentence_index}`)
8. Rename `Sentence` → `text` (matches BERTopic training format)

**Key Principle**:
- **Training**: BERTopic model can be trained on all 105 books (100% of sentences)
- **Analysis (Stage 1)**: Only use the 92 matched books (612,692 sentences, ~90%)
- The unmatched 10% of sentences helped shape the topic space during training, but are excluded from statistical analysis

**Output**: 
- Parquet file: `data/processed/sentence_df_with_ratings.parquet`
- Columns: `sentence_id`, `book_id`, `chapter_id`, `position_norm`, `text`, `rating_mean`, `rating_count`, `rating_class`, `Author`, `Book Title`
- Logs matching statistics and rating class distribution

**Test Results** (Dec 7, 2025):
- ✅ Successfully processed 680,822 sentences from 105 books
- ✅ Matched 94 books during fuzzy matching (89.5% match rate)
- ✅ Created matched-only dataset: 612,692 sentences from 92 books (90.0% of sentences)
- ✅ All matched books have complete rating data
- ✅ Rating classes: bad (27.0%), mid (37.8%), good (35.2%) at sentence level

---

### Step 2: Load BERTopic Model and Attach LLM Labels

**File**: `load_model_with_labels.py`

**Status**: ✅ **Implemented and Tested**

**Usage**:
```bash
python -m src.stage06_labeling.category_mapping.stage1_natural_clusters.load_model_with_labels \
    --labels-json results/stage06_labeling_openrouter/labels_pos_openrouter_mistralai_mistral-nemo_romance_aware_paraphrase-MiniLM-L6-v2_reasoning_high.json \
    --expected-topics 368 \
    --model-suffix _with_categories
```

**Tasks**:
1. Load BERTopic model from `model_1_with_categories` (native format)
2. Check if LLM labels are already integrated (via `custom_labels_` attribute)
3. If not, load labels from JSON file and attach them using `set_topic_labels()`
4. Verify model has expected number of topics (368, excluding outlier -1)
5. Log verification results and model state

**Key Features**:
- Uses helper function `load_native_bertopic_model()` for consistent loading
- Handles multiple JSON label formats (nested dict with `label` key or flat dict)
- Detects if labels are already integrated (avoids redundant loading)
- `--force-reload-labels` flag to override existing labels
- Comprehensive logging and verification
- Verifies labels accessible via both `custom_labels_` and `get_topic_info()`

**Test Results** (Dec 7, 2025):
- ✅ Model loads successfully (~4.4 seconds)
- ✅ Found 369 labels in `custom_labels_` (includes topic -1)
- ✅ All 369 topics have names accessible via `get_topic_info()`
- ✅ Model has 368 topics (excluding outlier -1)
- ✅ Labels already integrated (no reload needed)

**Output**: 
- Loaded model with labels attached (in memory)
- Log file with verification results
- Model ready for topic assignment (Step 3)

---

### Step 3: Ensure Topics for All Sentences (Matched Only)

**File**: `assign_topics_to_sentences.py`

**Tasks**:
1. Load the matched-only dataframe from Step 1:
   ```python
   df = pd.read_parquet("data/processed/sentence_df_with_ratings.parquet")
   # df contains only matched books (92 books, ~612k sentences)
   ```

2. Extract sentences from matched-only dataframe:
   ```python
   docs = df["text"].tolist()
   ```

3. Load the BERTopic model (trained on all 105 books):
   ```python
   topic_model = BERTopic.load("path/to/model")
   # Model was trained on all 105 books, but we'll transform only matched sentences
   ```

4. Transform only the matched sentences (inference only, no retraining):
   ```python
   topics, probs = topic_model.transform(docs)
   # This applies the model to the 612k matched sentences only
   ```

5. Attach to dataframe:
   ```python
   df["topic"] = topics
   df["topic_prob"] = probs  # if needed
   ```

**Output**: DataFrame with topic assignments per sentence (matched books only)

---

### Step 4: Explore Natural Hierarchy (Matched Docs Only)

**File**: `explore_hierarchical_topics.py`

**Tasks**:
1. Use the matched-only dataframe with topic assignments:
   ```python
   df = pd.read_parquet("data/processed/sentence_df_with_topics.parquet")
   docs = df["text"].tolist()
   ```

2. Build hierarchical structure on matched docs:
   ```python
   hierarchical_topics = topic_model.hierarchical_topics(docs)
   # Uses only matched sentences for hierarchy construction
   ```

2. Visualize dendrogram:
   ```python
   fig = topic_model.visualize_hierarchy(
       hierarchical_topics=hierarchical_topics,
       custom_labels=True  # uses LLM labels if set
   )
   fig.write_html("hierarchy_dendrogram.html")
   ```

3. Print text tree for inspection:
   ```python
   tree = topic_model.get_topic_tree(hierarchical_topics)
   print(tree)
   ```

4. **Decision point**: Choose target number of meta-topics (typically 40-80)
   - Too few: Over-merged, lose interpretability
   - Too many: Still too granular, not much reduction
   - Look for natural breakpoints in the tree

**Output**: 
- Visualization files
- Chosen `nr_meta_topics` value

---

### Step 5: Reduce Topics to Meta-Topic Level (Matched Docs Only)

**File**: `reduce_to_meta_topics.py`

**Tasks**:
1. Use matched-only dataframe:
   ```python
   df = pd.read_parquet("data/processed/sentence_df_with_topics.parquet")
   docs = df["text"].tolist()
   ```

2. Create a copy of model (reduction is in-place):
   ```python
   import copy
   topic_model_reduced = copy.deepcopy(topic_model)
   ```

3. Reduce to chosen number using matched docs:
   ```python
   nr_meta_topics = 60  # your chosen value
   topic_model_reduced.reduce_topics(docs, nr_topics=nr_meta_topics)
   # Reduction based on matched sentences only
   ```

4. Get updated topic assignments:
   ```python
   topics_reduced = topic_model_reduced.topics_
   df["topic_meta"] = topics_reduced
   ```

4. Inspect new topic set:
   ```python
   topic_info = topic_model_reduced.get_topic_info()
   print(topic_info.head(20))
   ```

**Output**: 
- Reduced model
- DataFrame with `topic_meta` column

---

### Step 6: Compute Per-Book Topic Distributions (Matched Books Only)

**File**: `compute_book_topic_distributions.py`

**Tasks**:
1. Use matched-only dataframe with meta-topics:
   ```python
   df = pd.read_parquet("data/processed/sentence_df_with_meta_topics.parquet")
   # df contains only matched books (92 books)
   ```

2. Count sentences per book × meta-topic:
   ```python
   book_topic_counts = (
       df.groupby(["book_id", "topic_meta"])
         .size()
         .unstack(fill_value=0)
   )
   ```

3. Convert to proportions (within each book):
   ```python
   book_topic_props = book_topic_counts.div(
       book_topic_counts.sum(axis=1), axis=0
   )
   ```

4. Attach book-level metadata:
   ```python
   book_meta = df.groupby("book_id").agg({
       "rating_mean": "first",
       "rating_class": "first",
       "rating_count": "first",
       "sentence_id": "count"  # n_sentences
   })
   
   book_level = book_meta.join(book_topic_props)
   # book_level contains only matched books (92 books)
   ```

**Output**: DataFrame with one row per book, columns for each meta-topic proportion (matched books only)

---

### Step 7: Topics Per Class (Visualization & Sanity Check) - Matched Docs Only

**File**: `topics_per_class_analysis.py`

**Tasks**:
1. Use matched-only dataframe:
   ```python
   df = pd.read_parquet("data/processed/sentence_df_with_meta_topics.parquet")
   docs = df["text"].tolist()
   classes = df["rating_class"].tolist()
   # docs and classes only include matched books
   ```

2. Use BERTopic's built-in function on matched docs:
   ```python
   topics_per_class = topic_model_reduced.topics_per_class(
       docs, classes=classes
   )
   # Only matched sentences are used, ensuring no contamination from unmatched books
   ```

3. Visualize:
   ```python
   fig = topic_model_reduced.visualize_topics_per_class(
       topics_per_class,
       top_n_topics=30,
       custom_labels=True
   )
   fig.write_html("topics_per_class.html")
   ```

4. Cross-check with `book_level` table for consistency

**Output**: Visualization showing topic prevalence by rating class (matched books only)

---

### Step 8: Statistical Analysis (ANOVA) - Matched Books Only

**File**: `statistical_analysis.py`

**Tasks**:
1. Use book-level dataframe (matched books only):
   ```python
   book_level = pd.read_parquet("data/processed/book_level_topic_distributions.parquet")
   # book_level contains only matched books (92 books)
   ```

2. For each meta-topic, test if proportion differs across rating classes:
   ```python
   from scipy.stats import f_oneway
   
   topic_cols = [c for c in book_level.columns 
                 if isinstance(c, int) or c.startswith("Topic")]
   
   results = []
   for col in topic_cols:
       g_bad = book_level.loc[book_level["rating_class"] == "bad", col]
       g_mid = book_level.loc[book_level["rating_class"] == "mid", col]
       g_good = book_level.loc[book_level["rating_class"] == "good", col]
       
       F, p = f_oneway(g_bad, g_mid, g_good)
       results.append({
           "topic_meta": col,
           "F": F,
           "p": p
       })
   ```

2. Adjust for multiple testing (Benjamini-Hochberg):
   ```python
   from statsmodels.stats.multitest import multipletests
   
   p_values = [r["p"] for r in results]
   _, p_adjusted, _, _ = multipletests(
       p_values, method="fdr_bh"
   )
   ```

3. Merge with topic labels:
   ```python
   topic_info = topic_model_reduced.get_topic_info()[
       ["Topic", "Name"]
   ]
   anova_results = pd.DataFrame(results).merge(
       topic_info, left_on="topic_meta", right_on="Topic", how="left"
   )
   anova_results["p_adjusted"] = p_adjusted
   ```

4. Sort by significance:
   ```python
   anova_results = anova_results.sort_values("p_adjusted")
   ```

**Output**: Table of meta-topics with F-statistics and p-values

---

### Step 9: Effect Size Calculation (Optional but Recommended)

**File**: `effect_size_analysis.py`

**Tasks**:
1. Calculate Cohen's d for good vs bad (for significant topics):
   ```python
   from scipy.stats import cohen_d
   
   for _, row in anova_results.iterrows():
       if row["p_adjusted"] < 0.05:
           col = row["topic_meta"]
           g_bad = book_level.loc[book_level["rating_class"] == "bad", col]
           g_good = book_level.loc[book_level["rating_class"] == "good", col]
           
           d = cohen_d(g_good, g_bad)
           # add to results
   ```

2. Add effect size interpretation (small: 0.2, medium: 0.5, large: 0.8)

**Output**: Enhanced results table with effect sizes

---

### Step 10: Qualitative Inspection

**File**: `inspect_key_topics.py`

**Tasks**:
1. For most significant meta-topics:
   ```python
   # Get top words
   topic_model_reduced.get_topic(7)
   
   # Get representative sentences
   rep_docs = topic_model_reduced.get_representative_docs(7)
   ```

2. Print summaries for interpretation

**Output**: Human-readable summaries of key meta-topics

---

### Step 11: Decision Point: Is Stage 2 Needed?

**File**: `evaluate_stage1_results.py`

**Tasks**:
1. Check if natural meta-topics align with research questions:
   - Are there clear "luxury/wealth" meta-topics?
   - Are there "emotional introspection" meta-topics?
   - Are there "erotic content" meta-topics (light vs explicit)?

2. If yes → Consider skipping Stage 2, proceed to Stage 3
3. If no → Proceed to Stage 2 for theory-driven mapping

**Output**: Decision document with rationale

---

## Expected Outputs

1. **Data Files**:
   - `sentence_df_with_topics.parquet`: Sentence-level data with topic assignments
   - `book_level_topic_distributions.parquet`: Book-level meta-topic proportions
   - `anova_results.csv`: Statistical test results

2. **Visualizations**:
   - `hierarchy_dendrogram.html`: Hierarchical topic tree
   - `topics_per_class.html`: Topic prevalence by rating class
   - `significant_meta_topics.png`: Bar plot of significant topics

3. **Reports**:
   - `stage1_summary.md`: Key findings
   - `stage2_decision.md`: Whether Stage 2 is needed

## Key Parameters to Tune

- **`nr_meta_topics`**: Target number of meta-topics (40-80 recommended)
- **`min_ratings`**: Minimum ratings for book inclusion (100 recommended)
- **Rating class thresholds**: Quantile cutoffs (33rd/66th percentiles)

## Dependencies

```python
import pandas as pd
import numpy as np
from bertopic import BERTopic
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests
```

## Next Steps After Stage 1

1. Review results and decide on Stage 2
2. If proceeding to Stage 2: See `../stage2_theory_driven_categories/README.md`
3. If skipping to Stage 3: See `../stage3_radway_functions/README.md`

