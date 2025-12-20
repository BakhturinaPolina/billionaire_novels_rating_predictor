#!/usr/bin/env python3
"""Generate book-level and chapter-level topic probabilities from sentence-level data.

This script aggregates topic probabilities from sentence_df_with_topics.parquet
to create:
- book_topic_probs.parquet: book_id, topic_id, prob
- chapter_topic_probs.parquet: book_id, chapter_id, topic_id, prob

Usage:
    python scripts/generate_topic_probabilities.py \
        --sentence-df data/processed/sentence_df_with_topics.parquet \
        --output-dir results/stage10_correlation_analysis
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from bertopic import BERTopic
from tqdm import tqdm

from src.common.logging import setup_logging


def load_bertopic_model(
    model_path: Path,
    logger: Optional[logging.Logger] = None,
) -> BERTopic:
    """Load BERTopic model from path.
    
    Args:
        model_path: Path to BERTopic model (directory or .pkl file)
        logger: Logger instance
        
    Returns:
        Loaded BERTopic model
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Loading BERTopic model from: {model_path}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # Try loading from directory first
    if model_path.is_dir():
        logger.info("  Loading from directory...")
        topic_model = BERTopic.load(str(model_path))
    elif model_path.suffix == ".pkl":
        logger.info("  Loading from .pkl file...")
        with open(model_path, "rb") as f:
            loaded_obj = pickle.load(f)
        
        # Check if it's a wrapper
        if hasattr(loaded_obj, "trained_topic_model") and loaded_obj.trained_topic_model is not None:
            logger.info("  Extracted BERTopic model from wrapper")
            topic_model = loaded_obj.trained_topic_model
        elif isinstance(loaded_obj, BERTopic):
            topic_model = loaded_obj
        else:
            # Try BERTopic.load() as fallback
            topic_model = BERTopic.load(str(model_path))
    else:
        # Try loading as directory
        topic_model = BERTopic.load(str(model_path))
    
    # Log model info
    if hasattr(topic_model, "topic_representations_"):
        topic_ids = [tid for tid in topic_model.topic_representations_.keys() if tid != -1]
        logger.info(f"✓ Model loaded with {len(topic_ids)} topics (excluding outlier -1)")
    
    return topic_model


def create_book_identifier(author: str, title: str) -> str:
    """Create a unique book identifier from Author + Book Title.
    
    This creates an identifier based on Author + Book Title (useful as a fallback when no canonical ID is available).
    
    Args:
        author: Author name
        title: Book title
        
    Returns:
        Unique book identifier string
    """
    # Normalize and combine: Author_Title (preserving original format from chapters.csv)
    author_norm = str(author).strip().replace(" ", "_")
    title_norm = str(title).strip()
    return f"{author_norm}_{title_norm}"



def coerce_id_series(s: pd.Series, *, name: str, logger: Optional[logging.Logger] = None) -> pd.Series:
    """Coerce an identifier series into a consistent, join-friendly string form.

    - strips whitespace
    - converts to string (preserving numeric ids)
    - converts <NA>/nan to actual missing values
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    out = s.copy()
    # Preserve missingness
    out = out.astype("string")
    out = out.str.strip()
    missing = out.isna() | (out.str.lower().isin(["", "nan", "none"]))
    if missing.any():
        logger.warning(f"  {missing.sum():,} rows have missing {name} values")
        out[missing] = pd.NA
    return out


def load_sentence_dataframe(
    sentence_df_path: Path,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Load sentence dataframe with topic assignments.

    Uses Goodreads ID as the canonical book_id for reliable merging with metadata.
    This ensures consistent joins across the analysis pipeline.
    
    Args:
        sentence_df_path: Path to sentence_df_with_topics.parquet
        logger: Logger instance
        
    Returns:
        DataFrame with book_id (Goodreads ID), chapter_id, text, Author, Book Title columns
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Loading sentence dataframe from: {sentence_df_path}")
    
    if not sentence_df_path.exists():
        raise FileNotFoundError(f"Sentence dataframe not found: {sentence_df_path}")
    
    df = pd.read_parquet(sentence_df_path)
    
    logger.info(f"✓ Loaded {len(df):,} sentences")
    logger.info(f"  Columns: {', '.join(df.columns.tolist())}")
    
    # Verify required columns
    required_cols = ["text"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Select book_id source
    # CRITICAL: Goodreads ID is the canonical identifier for reliable merging
    goodreads_col = getattr(load_sentence_dataframe, "_goodreads_id_col", "goodreads_book_id")
    book_id_source = getattr(load_sentence_dataframe, "_book_id_source", "goodreads")

    if book_id_source == "goodreads":
        if goodreads_col not in df.columns:
            raise ValueError(
                f"❌ REQUIRED: book_id_source=goodreads but column '{goodreads_col}' not found in sentence dataframe.\n"
                f"   Available columns: {df.columns.tolist()}\n"
                f"   Fix: Add '{goodreads_col}' to sentence_df upstream (merge with Goodreads metadata)."
            )
        logger.info(f"  ✓ Using Goodreads-based book_id from column: {goodreads_col}")
        
        # Warn if Author/Title exist but we're not using them (helps catch confusion)
        if "Author" in df.columns and "Book Title" in df.columns:
            logger.info(f"  Note: Author/Book Title columns present but not used for book_id (using Goodreads ID instead)")
        
        df["book_id"] = coerce_id_series(df[goodreads_col], name=goodreads_col, logger=logger)
        
        # Strict validation: no missing Goodreads IDs allowed
        missing_count = df["book_id"].isna().sum()
        if missing_count > 0:
            raise ValueError(
                f"❌ CRITICAL: Found {missing_count:,} rows ({missing_count/len(df)*100:.1f}%) with missing Goodreads IDs in column '{goodreads_col}'.\n"
                f"   Cannot produce reliable book-level outputs.\n"
                f"   Fix: Ensure all sentences have valid Goodreads IDs upstream (check join with metadata)."
            )

    elif book_id_source == "existing":
        if "book_id" not in df.columns:
            raise ValueError("book_id_source=existing but no 'book_id' column found")
        logger.info("  Using existing book_id column from sentence dataframe")
        df["book_id"] = coerce_id_series(df["book_id"], name="book_id", logger=logger)

    elif book_id_source == "author_title":
        if "Author" not in df.columns or "Book Title" not in df.columns:
            raise ValueError("book_id_source=author_title requires 'Author' and 'Book Title' columns")
        logger.warning("  ⚠ WARNING: Using Author+Title for book_id (not recommended for merging with metadata)")
        logger.warning("     This will break joins with Goodreads metadata. Use --book-id-source goodreads instead.")
        df["book_id"] = df.apply(lambda row: create_book_identifier(row["Author"], row["Book Title"]), axis=1)
        df["book_id"] = coerce_id_series(df["book_id"], name="author_title", logger=logger)

    else:
        raise ValueError(f"Unknown book_id_source: {book_id_source}")

    logger.info(f"  Books: {df['book_id'].nunique()}")
    
    # Check if chapter_id exists (optional for chapter-level aggregation)
    has_chapter = "chapter_id" in df.columns
    if has_chapter:
        unique_chapter_pairs = df.groupby(['book_id', 'chapter_id']).size().shape[0]
        avg_chapters_per_book = df.groupby('book_id')['chapter_id'].nunique().mean()
        logger.info(f"  Found chapter_id column: {unique_chapter_pairs:,} unique (book, chapter) pairs")
        logger.info(f"    Average chapters per book: {avg_chapters_per_book:.1f}")
    else:
        logger.warning("  No chapter_id column found. Chapter-level aggregation will be skipped.")
    
    return df


def compute_cache_key(
    sentence_df_path: Path,
    model_path: Path,
    num_texts: int,
) -> str:
    """Compute a cache key based on input parameters.
    
    Args:
        sentence_df_path: Path to sentence dataframe
        model_path: Path to BERTopic model
        num_texts: Number of texts being processed
        
    Returns:
        Cache key string (hash)
    """
    # Create a hash from file paths, file sizes, modification times, and text count
    # This ensures cache invalidation when inputs change, even if row count stays the same
    stat = sentence_df_path.stat()
    
    # Handle both file and directory model paths
    if model_path.exists():
        model_stat = model_path.stat()
        model_fingerprint = f"{model_path}_{model_stat.st_size}_{model_stat.st_mtime_ns}"
    else:
        # Model path doesn't exist (shouldn't happen, but handle gracefully)
        model_fingerprint = str(model_path)
    
    key_string = f"{sentence_df_path}_{stat.st_size}_{stat.st_mtime_ns}_{model_fingerprint}_{num_texts}"
    return hashlib.md5(key_string.encode()).hexdigest()


def load_cached_probabilities(
    cache_dir: Path,
    cache_key: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[Tuple[list[int], np.ndarray]]:
    """Load cached topic probabilities if they exist.
    
    Args:
        cache_dir: Directory where cache files are stored
        cache_key: Cache key (hash) identifying this computation
        logger: Logger instance
        
    Returns:
        Tuple of (topics list, probabilities array) if cache exists, None otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    cache_file = cache_dir / f"topic_probs_{cache_key}.npz"
    
    if not cache_file.exists():
        return None
    
    try:
        logger.info(f"  Loading cached probabilities from: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        topics = data["topics"].tolist()
        probs_array = data["probs_array"]
        logger.info(f"  ✓ Loaded cached probabilities: shape {probs_array.shape}")
        return topics, probs_array
    except Exception as e:
        logger.warning(f"  Failed to load cache: {e}")
        return None


def save_cached_probabilities(
    topics: list[int],
    probs_array: np.ndarray,
    cache_dir: Path,
    cache_key: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Save computed topic probabilities to cache.
    
    Args:
        topics: List of topic assignments
        probs_array: Probability array shape (n_docs x n_topics)
        cache_dir: Directory where cache files are stored
        cache_key: Cache key (hash) identifying this computation
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"topic_probs_{cache_key}.npz"
    
    try:
        logger.info(f"  Saving probabilities to cache: {cache_file}")
        np.savez_compressed(
            cache_file,
            topics=np.array(topics),
            probs_array=probs_array,
        )
        file_size_mb = cache_file.stat().st_size / 1024**2
        logger.info(f"  ✓ Saved cache ({file_size_mb:.2f} MB)")
    except Exception as e:
        logger.warning(f"  Failed to save cache: {e}")


def compute_topic_probabilities(
    topic_model: BERTopic,
    texts: list[str],
    batch_size: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    cache_key: Optional[str] = None,
    use_cache: bool = True,
    logger: Optional[logging.Logger] = None,
) -> tuple[list[int], np.ndarray]:
    """Compute topic probabilities for texts using BERTopic model.
    
    Args:
        topic_model: BERTopic model
        texts: List of text documents
        batch_size: Optional batch size for processing
        cache_dir: Optional directory for caching probabilities
        cache_key: Optional cache key for this computation
        use_cache: Whether to use cache if available
        logger: Logger instance
        
    Returns:
        Tuple of (topics list, probabilities array shape: n_docs x n_topics)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Try to load from cache first
    if use_cache and cache_dir and cache_key:
        cached = load_cached_probabilities(cache_dir, cache_key, logger)
        if cached is not None:
            return cached
    
    logger.info(f"Computing topic probabilities for {len(texts):,} documents...")
    
    if batch_size and len(texts) > batch_size:
        logger.info(f"  Processing in batches of {batch_size:,}...")
        all_topics = []
        all_probs = []
        
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Create progress bar
        pbar = tqdm(
            total=len(texts),
            unit="doc",
            unit_scale=True,
            desc="  Processing batches",
            ncols=100,
            mininterval=1.0,  # Update at least every second
        )
        
        for batch_num, i in enumerate(range(0, len(texts), batch_size), start=1):
            batch = texts[i:i + batch_size]
            batch_topics, batch_probs = topic_model.transform(batch)
            
            all_topics.extend(batch_topics.tolist() if isinstance(batch_topics, np.ndarray) else batch_topics)
            all_probs.append(batch_probs if isinstance(batch_probs, np.ndarray) else np.array(batch_probs))
            
            # Update progress bar
            pbar.update(len(batch))
            pbar.set_postfix({"batch": f"{batch_num}/{total_batches}"})
        
        pbar.close()
        logger.info(f"  ✓ Processed all {len(texts):,} documents in {total_batches} batches")
        
        probs_array = np.vstack(all_probs)
    else:
        logger.info("  Processing all documents at once...")
        topics, probs = topic_model.transform(texts)
        
        all_topics = topics.tolist() if isinstance(topics, np.ndarray) else topics
        probs_array = probs if isinstance(probs, np.ndarray) else np.array(probs)
    
    logger.info(f"✓ Computed probabilities: shape {probs_array.shape}")
    
    # Save to cache if enabled
    if cache_dir and cache_key:
        save_cached_probabilities(all_topics, probs_array, cache_dir, cache_key, logger)
    
    return all_topics, probs_array


# NOTE: extract_topic_probabilities() is not currently used in main() pipeline.
# It was designed for extracting pre-computed probabilities from a topic_prob column,
# but the current pipeline computes probabilities via BERTopic.transform() instead.
# Keeping function for potential future use but removing broken topic_model reference.


def aggregate_to_book_level(
    probs_array: np.ndarray,
    metadata: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Aggregate topic probabilities to book level.
    
    Args:
        probs_array: Probability array shape (n_sentences, n_topics)
        metadata: DataFrame with book_id column (and optionally chapter_id)
        logger: Logger instance
        
    Returns:
        DataFrame with book_id, topic_id, prob (summed and normalized per book)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Aggregating to book level...")
    
    n_sentences, n_topics = probs_array.shape
    
    # Check for NaN values
    nan_sentences = np.isnan(probs_array.sum(axis=1))
    if nan_sentences.sum() > 0:
        logger.warning(f"  {nan_sentences.sum():,} sentences ({nan_sentences.sum()/n_sentences*100:.2f}%) have NaN probabilities")
        logger.warning(f"    These will be excluded from aggregation")
    
    # Group by book_id and sum probabilities
    book_topic_list = []
    
    for book_id in metadata["book_id"].unique():
        book_mask = metadata["book_id"] == book_id
        # Since metadata index is reset, we can use boolean indexing directly
        book_probs_raw = probs_array[book_mask.values]
        
        # Filter out NaN rows (replace with zeros for safe summation)
        book_probs_raw_clean = np.nan_to_num(book_probs_raw, nan=0.0)
        book_probs = book_probs_raw_clean.sum(axis=0)  # Sum over sentences
        
        # Normalize
        total = book_probs.sum()
        if total > 0:
            book_probs = book_probs / total
        else:
            # If total is 0, this shouldn't happen but ensure book is still represented
            # Set uniform distribution (all topics equal probability)
            logger.warning(f"  Book {book_id} has zero total probability, using uniform distribution")
            book_probs = np.ones(n_topics) / n_topics
        
        # Store ALL topics with ALL probabilities (no filtering)
        for topic_id in range(n_topics):
            prob = book_probs[topic_id]
            book_topic_list.append({
                "book_id": book_id,
                "topic_id": int(topic_id),
                "prob": float(prob),
            })
    
    book_topic = pd.DataFrame(book_topic_list)
    
    logger.info(f"  Aggregated to {len(book_topic):,} (book, topic) pairs")
    logger.info(f"  Books: {book_topic['book_id'].nunique()}")
    logger.info(f"  Topics: {book_topic['topic_id'].nunique()}")
    
    # Verify normalization
    book_sums = book_topic.groupby("book_id")["prob"].sum()
    if not np.allclose(book_sums, 1.0, atol=1e-6):
        logger.warning(f"  Some books don't sum to 1.0 (min: {book_sums.min():.6f}, max: {book_sums.max():.6f})")
    else:
        logger.info("  ✓ Probabilities normalized correctly (sum to 1.0 per book)")
    
    return book_topic


def aggregate_to_chapter_level(
    probs_array: np.ndarray,
    metadata: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Aggregate topic probabilities to chapter level.
    
    Args:
        probs_array: Probability array shape (n_sentences, n_topics)
        metadata: DataFrame with book_id and chapter_id columns
        logger: Logger instance
        
    Returns:
        DataFrame with book_id, chapter_id, topic_id, prob (summed and normalized per chapter)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if "chapter_id" not in metadata.columns:
        logger.warning("  No chapter_id column found. Skipping chapter-level aggregation.")
        return pd.DataFrame()
    
    logger.info("Aggregating to chapter level...")
    
    n_sentences, n_topics = probs_array.shape
    
    # Verify alignment
    if len(metadata) != n_sentences:
        logger.error(f"  MISMATCH: metadata has {len(metadata)} rows but probs_array has {n_sentences} rows!")
        raise ValueError("Metadata and probability array size mismatch")
    
    # Check if individual sentence probabilities sum to 1
    sentence_sums = probs_array.sum(axis=1)
    zero_sentence_count = np.sum(sentence_sums == 0)
    nan_sentence_count = np.isnan(sentence_sums).sum()
    if nan_sentence_count > 0:
        logger.warning(f"  {nan_sentence_count:,} sentences ({nan_sentence_count/n_sentences*100:.2f}%) have NaN probabilities")
        logger.warning(f"    These will be excluded from aggregation (replaced with zeros)")
    if zero_sentence_count > 0:
        logger.warning(f"  {zero_sentence_count:,} sentences ({zero_sentence_count/n_sentences*100:.2f}%) have zero total probability")
        logger.warning(f"    This may indicate an issue with BERTopic probability computation")
    
    # Group by (book_id, chapter_id) and sum probabilities
    chapter_topic_list = []
    zero_prob_chapters = []
    zero_prob_diagnostics = []
    
    # Get all chapter groups and create progress bar
    chapter_groups = metadata.groupby(["book_id", "chapter_id"]).groups
    total_chapters = len(chapter_groups)
    
    logger.info(f"  Processing {total_chapters:,} chapters...")
    pbar = tqdm(
        total=total_chapters,
        unit="chapter",
        desc="  Aggregating chapters",
        ncols=100,
        mininterval=1.0,
    )
    
    for (book_id, chapter_id), group_indices in chapter_groups.items():
        # Since metadata index is reset, group_indices are already array-aligned
        group_indices_list = list(group_indices)
        num_sentences_in_chapter = len(group_indices_list)
        
        if num_sentences_in_chapter == 0:
            logger.warning(f"  Chapter ({book_id}, {chapter_id}) has NO sentences in metadata!")
            zero_prob_chapters.append((book_id, chapter_id))
            zero_prob_diagnostics.append({
                'book_id': book_id,
                'chapter_id': chapter_id,
                'num_sentences': 0,
                'zero_sentence_count': 0,
                'pct_zero_sentences': 0.0,
                'mean_sentence_sum': 0.0,
                'min_sentence_sum': 0.0,
                'max_sentence_sum': 0.0,
            })
            chapter_probs = np.ones(n_topics) / n_topics
        else:
            # Get probabilities for this chapter's sentences
            chapter_probs_raw = probs_array[group_indices_list]
            
            # Filter out NaN values (replace with zeros for safe summation)
            chapter_probs_raw_clean = np.nan_to_num(chapter_probs_raw, nan=0.0)
            
            # Check individual sentence probabilities
            sentence_sums_chapter = chapter_probs_raw_clean.sum(axis=1)
            zero_sentences_in_chapter = np.sum(sentence_sums_chapter == 0)
            nan_sentences_in_chapter = np.isnan(chapter_probs_raw.sum(axis=1)).sum()
            
            # Sum over sentences
            chapter_probs = chapter_probs_raw_clean.sum(axis=0)
            
            # Normalize
            total = chapter_probs.sum()
            if total > 0:
                chapter_probs = chapter_probs / total
            else:
                # If total is 0, diagnose the issue
                zero_prob_chapters.append((book_id, chapter_id))
                # Use regular mean/min/max since array is already cleaned (NaN replaced with 0)
                zero_prob_diagnostics.append({
                    'book_id': book_id,
                    'chapter_id': chapter_id,
                    'num_sentences': num_sentences_in_chapter,
                    'zero_sentence_count': int(zero_sentences_in_chapter),
                    'pct_zero_sentences': float(zero_sentences_in_chapter / num_sentences_in_chapter * 100) if num_sentences_in_chapter > 0 else 0.0,
                    'nan_sentence_count': int(nan_sentences_in_chapter),
                    'mean_sentence_sum': float(sentence_sums_chapter.mean()) if num_sentences_in_chapter > 0 else 0.0,
                    'min_sentence_sum': float(sentence_sums_chapter.min()) if num_sentences_in_chapter > 0 else 0.0,
                    'max_sentence_sum': float(sentence_sums_chapter.max()) if num_sentences_in_chapter > 0 else 0.0,
                })
                chapter_probs = np.ones(n_topics) / n_topics
        
        # Get chapter metadata (Author, Book Title) if available
        # Use first index from group_indices for efficient lookup
        first_idx = group_indices[0]
        chapter_row = {
            "book_id": book_id,
            "chapter_id": chapter_id,
        }
        if "Author" in metadata.columns:
            chapter_row["Author"] = metadata.iloc[first_idx]["Author"]
        if "Book Title" in metadata.columns:
            chapter_row["Book Title"] = metadata.iloc[first_idx]["Book Title"]
        
        # Store ALL topics with ALL probabilities (no filtering)
        for topic_id in range(n_topics):
            prob = chapter_probs[topic_id]
            row = chapter_row.copy()
            row["topic_id"] = int(topic_id)
            row["prob"] = float(prob)
            chapter_topic_list.append(row)
        
        # Update progress bar
        pbar.update(1)
    
    pbar.close()
    chapter_topic = pd.DataFrame(chapter_topic_list)
    
    # Log summary of zero-probability chapters with diagnostics
    if zero_prob_chapters:
        total_chapters = len(metadata.groupby(["book_id", "chapter_id"]).groups)
        zero_pct = (len(zero_prob_chapters) / total_chapters) * 100
        logger.warning(f"  {len(zero_prob_chapters):,} chapters ({zero_pct:.1f}%) had zero total probability")
        logger.warning(f"    These chapters were assigned uniform distribution across all topics")
        
        # Log diagnostic information
        if zero_prob_diagnostics:
            diag_df = pd.DataFrame(zero_prob_diagnostics)
            logger.warning(f"\n  Diagnostic information for zero-probability chapters:")
            logger.warning(f"    Average sentences per chapter: {diag_df['num_sentences'].mean():.1f}")
            logger.warning(f"    Chapters with all-zero sentences: {(diag_df['pct_zero_sentences'] == 100).sum()}")
            logger.warning(f"    Average % zero sentences: {diag_df['pct_zero_sentences'].mean():.1f}%")
            if 'nan_sentence_count' in diag_df.columns:
                total_nan = diag_df['nan_sentence_count'].sum()
                logger.warning(f"    Total NaN sentences in affected chapters: {total_nan:,}")
            logger.warning(f"    Mean sentence probability sum: {diag_df['mean_sentence_sum'].mean():.6f}")
            logger.warning(f"    Min sentence probability sum: {diag_df['min_sentence_sum'].min():.6f}")
            logger.warning(f"    Max sentence probability sum: {diag_df['max_sentence_sum'].max():.6f}")
            
            # Show sample of problematic chapters
            if len(zero_prob_chapters) <= 10:
                logger.warning(f"\n    All affected chapters:")
                for diag in zero_prob_diagnostics[:10]:
                    logger.warning(f"      ({diag['book_id']}, {diag['chapter_id']}): "
                                 f"{diag['num_sentences']} sentences, "
                                 f"{diag['zero_sentence_count']} zero-prob ({diag['pct_zero_sentences']:.1f}%), "
                                 f"mean_sum={diag['mean_sentence_sum']:.6f}")
            else:
                logger.warning(f"\n    Sample affected chapters (showing first 5):")
                for diag in zero_prob_diagnostics[:5]:
                    logger.warning(f"      ({diag['book_id']}, {diag['chapter_id']}): "
                                 f"{diag['num_sentences']} sentences, "
                                 f"{diag['zero_sentence_count']} zero-prob ({diag['pct_zero_sentences']:.1f}%), "
                                 f"mean_sum={diag['mean_sentence_sum']:.6f}")
                logger.warning(f"      ... ({len(zero_prob_chapters)-5} more chapters)")
        else:
            # Fallback if diagnostics weren't collected
            if len(zero_prob_chapters) <= 10:
                logger.warning(f"    Affected chapters: {zero_prob_chapters}")
            else:
                logger.warning(f"    Sample affected chapters: {zero_prob_chapters[:5]} ... ({len(zero_prob_chapters)-5} more)")
    
    logger.info(f"  Aggregated to {len(chapter_topic):,} (chapter, topic) pairs")
    logger.info(f"  Books: {chapter_topic['book_id'].nunique()}")
    logger.info(f"  Chapters: {chapter_topic.groupby('book_id')['chapter_id'].nunique().sum()}")
    logger.info(f"  Topics: {chapter_topic['topic_id'].nunique()}")
    
    # Verify normalization
    chapter_sums = chapter_topic.groupby(["book_id", "chapter_id"])["prob"].sum()
    if not np.allclose(chapter_sums, 1.0, atol=1e-6):
        logger.warning(f"  Some chapters don't sum to 1.0 (min: {chapter_sums.min():.6f}, max: {chapter_sums.max():.6f})")
    else:
        logger.info("  ✓ Probabilities normalized correctly (sum to 1.0 per chapter)")
    
    return chapter_topic


def main():
    parser = argparse.ArgumentParser(
        description="Generate book-level and chapter-level topic probabilities"
    )
    parser.add_argument(
        "--sentence-df",
        type=Path,
        required=True,
        help="Path to sentence_df_with_topics.parquet",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to BERTopic model (directory or .pkl file)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage10_correlation_analysis"),
        help="Output directory for book_topic_probs.parquet and chapter_topic_probs.parquet",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for topic transformation (default: process all at once)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Directory for log files (default: output_dir/logs)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for caching computed probabilities (default: output_dir/cache)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (always recompute probabilities)",
    )

    parser.add_argument(
        "--book-id-source",
        choices=["goodreads", "existing", "author_title"],
        default="goodreads",
        help="Which identifier to output as book_id. DEFAULT: 'goodreads' (REQUIRED for reliable merges with metadata). "
             "Use 'author_title' only if Goodreads IDs are unavailable (will break downstream joins).",
    )
    parser.add_argument(
        "--goodreads-id-col",
        type=str,
        default="goodreads_book_id",
        help="Column name in sentence_df that contains Goodreads book id (used when --book-id-source=goodreads).",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logs_dir = args.logs_dir or args.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(
        logs_dir=logs_dir,
        log_file="generate_topic_probabilities.log",
    )
    
    logger.info("=" * 80)
    logger.info("Generate Topic Probabilities")
    logger.info("=" * 80)
    logger.info(f"Sentence dataframe: {args.sentence_df}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Book ID source: {args.book_id_source}")
    if args.book_id_source == "goodreads":
        logger.info(f"Goodreads ID column: {args.goodreads_id_col}")
    if args.book_id_source != "goodreads":
        logger.warning("⚠ WARNING: Not using Goodreads IDs. This will break merges with metadata!")
    
    # Step 1: Load BERTopic model
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Load BERTopic Model")
    logger.info("=" * 80)
    
    topic_model = load_bertopic_model(args.model_path, logger=logger)
    
    # Step 2: Load sentence dataframe
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Load Sentence Dataframe")
    logger.info("=" * 80)
    
    load_sentence_dataframe._book_id_source = args.book_id_source
    load_sentence_dataframe._goodreads_id_col = args.goodreads_id_col
    df = load_sentence_dataframe(args.sentence_df, logger=logger)
    
    # Step 3: Compute topic probabilities
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Compute Topic Probabilities")
    logger.info("=" * 80)
    
    # Setup cache
    cache_dir = args.cache_dir or args.output_dir / "cache"
    cache_key = compute_cache_key(args.sentence_df, args.model_path, len(df))
    use_cache = not args.no_cache
    
    if use_cache:
        logger.info(f"Cache directory: {cache_dir}")
        logger.info(f"Cache key: {cache_key}")
        logger.info("  Note: Cache key includes file size + mtime to detect content changes")
    else:
        logger.info("  Cache disabled (--no-cache)")
    
    texts = df["text"].tolist()
    topics, probs_array = compute_topic_probabilities(
        topic_model,
        texts,
        batch_size=args.batch_size,
        cache_dir=cache_dir if use_cache else None,
        cache_key=cache_key if use_cache else None,
        use_cache=use_cache,
        logger=logger,
    )
    n_topics = probs_array.shape[1]
    # Sanity: topic id alignment
    if hasattr(topic_model, "topic_representations_"):
        model_topic_ids = [tid for tid in topic_model.topic_representations_.keys() if tid != -1]
        if len(model_topic_ids) != n_topics:
            logger.warning(
                f"Topic count mismatch: model has {len(model_topic_ids)} topics (excluding -1) "
                f"but probability vectors have width {n_topics}. "
                "Will label topics by column index (0..n_topics-1)."
            )
    
    # Create metadata dataframe with book_id, Author, Book Title
    metadata_cols = ["book_id"]
    # Keep original Goodreads id column too if present (useful for debugging joins)
    if args.book_id_source == "goodreads" and args.goodreads_id_col in df.columns and args.goodreads_id_col != "book_id":
        metadata_cols.append(args.goodreads_id_col)
    if "Author" in df.columns:
        metadata_cols.append("Author")
    if "Book Title" in df.columns:
        metadata_cols.append("Book Title")
    if "chapter_id" in df.columns:
        metadata_cols.append("chapter_id")
    
    metadata = df[metadata_cols].copy().reset_index(drop=True)
    
    # Step 4: Aggregate to book level
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Aggregate to Book Level")
    logger.info("=" * 80)
    
    book_topic = aggregate_to_book_level(probs_array, metadata, logger=logger)
    
    # Step 5: Aggregate to chapter level (if chapter_id available)
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Aggregate to Chapter Level")
    logger.info("=" * 80)
    
    chapter_topic = aggregate_to_chapter_level(probs_array, metadata, logger=logger)
    
    # Step 6: Save outputs
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Save Outputs")
    logger.info("=" * 80)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save book-level probabilities
    book_output = args.output_dir / "book_topic_probs.parquet"
    book_topic.to_parquet(book_output, index=False)
    logger.info(f"✓ Saved book-level probabilities: {book_output}")
    logger.info(f"  Rows: {len(book_topic):,}")
    logger.info(f"  File size: {book_output.stat().st_size / 1024**2:.2f} MB")
    
    # Save chapter-level probabilities (if available)
    if len(chapter_topic) > 0:
        chapter_output = args.output_dir / "chapter_topic_probs.parquet"
        chapter_topic.to_parquet(chapter_output, index=False)
        logger.info(f"✓ Saved chapter-level probabilities: {chapter_output}")
        logger.info(f"  Rows: {len(chapter_topic):,}")
        logger.info(f"  File size: {chapter_output.stat().st_size / 1024**2:.2f} MB")
    else:
        logger.info("⚠ Skipped chapter-level output (no chapter_id available)")
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"Input sentences: {len(df):,}")
    logger.info(f"Books: {df['book_id'].nunique()}")
    logger.info(f"Topics: {n_topics}")
    logger.info(f"Book-level pairs: {len(book_topic):,}")
    if len(chapter_topic) > 0:
        logger.info(f"Chapter-level pairs: {len(chapter_topic):,}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("\n✓ Topic probability aggregation complete!")


if __name__ == "__main__":
    main()