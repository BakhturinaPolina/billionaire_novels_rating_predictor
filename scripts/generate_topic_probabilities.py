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
    
    This creates a stable identifier based on the original chapters.csv data,
    not the Goodreads ID.
    
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


def load_sentence_dataframe(
    sentence_df_path: Path,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Load sentence dataframe with topic assignments.
    
    Creates a book_id from Author + Book Title (from chapters.csv) instead of
    using the Goodreads ID. This ensures consistency with the original data source.
    
    Args:
        sentence_df_path: Path to sentence_df_with_topics.parquet
        logger: Logger instance
        
    Returns:
        DataFrame with book_id (from Author+Title), chapter_id, text, Author, Book Title columns
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
    
    # Check if we have Author and Book Title (needed to create book_id)
    if "Author" not in df.columns or "Book Title" not in df.columns:
        logger.warning("  Author or Book Title not found. Will try to use existing book_id.")
        if "book_id" not in df.columns:
            raise ValueError("Need either (Author, Book Title) or book_id column")
    else:
        # Create book_id from Author + Book Title (from chapters.csv)
        logger.info("  Creating book_id from Author + Book Title (chapters.csv identifier)")
        df["book_id"] = df.apply(
            lambda row: create_book_identifier(row["Author"], row["Book Title"]), axis=1
        )
        logger.info(f"  Created {df['book_id'].nunique()} unique book identifiers")
    
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
    # Create a hash from file paths and text count
    key_string = f"{sentence_df_path}_{model_path}_{num_texts}"
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


def extract_topic_probabilities(
    df: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract topic probabilities from topic_prob column.
    
    Args:
        df: DataFrame with topic_prob column (list/array of probabilities)
        logger: Logger instance
        
    Returns:
        Tuple of (probability array shape: n_sentences x n_topics, metadata dataframe)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Extracting topic probabilities from topic_prob column...")
    
    # Convert topic_prob to numpy array
    # topic_prob is stored as list/array in parquet
    probs_list = df["topic_prob"].tolist()
    
    # Handle different storage formats
    if isinstance(probs_list[0], (list, np.ndarray)):
        probs_array = np.array(probs_list)
    else:
        # If stored as string or other format, try to parse
        logger.warning("  topic_prob may not be in expected format, attempting conversion...")
        try:
            probs_array = np.array([np.array(p) if isinstance(p, (list, np.ndarray)) else eval(str(p)) for p in probs_list])
        except Exception as e:
            logger.error(f"  Failed to parse topic_prob: {e}")
            raise ValueError("Could not parse topic_prob column. Expected list/array format.")
    
    n_topics = probs_array.shape[1]
    logger.info(f"  Found {n_topics} topics in probability vectors")
    logger.info(f"  Probability array shape: {probs_array.shape}")
    logger.info(f"  Memory usage: {probs_array.nbytes / 1024**2:.2f} MB")
    
    # Create metadata dataframe with book_id and chapter_id
    # Reset index to ensure alignment with array rows
    metadata = df[["book_id"]].copy().reset_index(drop=True)
    if "chapter_id" in df.columns:
        metadata["chapter_id"] = df["chapter_id"].values
    
    return probs_array, metadata


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
    
    # Step 1: Load BERTopic model
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Load BERTopic Model")
    logger.info("=" * 80)
    
    topic_model = load_bertopic_model(args.model_path, logger=logger)
    
    # Step 2: Load sentence dataframe
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Load Sentence Dataframe")
    logger.info("=" * 80)
    
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
    
    # Create metadata dataframe with book_id, Author, Book Title
    metadata_cols = ["book_id"]
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
