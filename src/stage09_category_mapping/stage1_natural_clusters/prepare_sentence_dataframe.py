"""Prepare sentence-level dataframe with ratings for Stage 1 natural clusters analysis.

This script creates a "matched only" dataframe that contains only books present in BOTH
chapters.csv (text data) and goodreads.csv (metadata). This ensures Stage 1 analysis uses
only books with complete metadata, even though the BERTopic model may have been trained
on all books.

Key principle:
- Training: BERTopic model can be trained on all 105 books (100% of sentences)
- Analysis (Stage 1): Only use the 94 matched books (612,692 sentences, ~90%)

This script:
1. Loads chapters.csv and goodreads.csv
2. Creates book_id by fuzzy matching Author + Title between files
3. Creates "matched only" dataframe using inner join (only books in both datasets)
4. Tracks which books were dropped (only in texts vs only in metadata)
5. Creates rating_class (bad/mid/good) based on quantiles (only for matched books)
6. Calculates position_norm (normalized sentence position in book)
7. Outputs cleaned dataframe to parquet format
"""

from __future__ import annotations

import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from difflib import SequenceMatcher

from src.common.logging import setup_logging

# Suppress FutureWarning about groupby.apply operating on grouping columns
warnings.filterwarnings("ignore", category=FutureWarning, message=".*DataFrameGroupBy.apply operated on the grouping columns.*")


def normalize_author_title(author: str, title: str) -> str:
    """Normalize author and title for matching.
    
    Args:
        author: Author name
        title: Book title
        
    Returns:
        Normalized string for matching
    """
    # Convert to lowercase, remove extra spaces, strip punctuation
    author_norm = str(author).lower().strip().replace("_", " ").replace(".", "")
    title_norm = str(title).lower().strip()
    return f"{author_norm} {title_norm}"


def fuzzy_match_books(
    chapters_df: pd.DataFrame,
    goodreads_df: pd.DataFrame,
    threshold: float = 0.85,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Match books between chapters.csv and goodreads.csv using fuzzy matching.
    
    Args:
        chapters_df: DataFrame from chapters.csv
        goodreads_df: DataFrame from goodreads.csv
        threshold: Minimum similarity score for matching (0-1)
        logger: Logger instance for detailed logging
        
    Returns:
        DataFrame with book_id column added to chapters_df
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Starting fuzzy matching between chapters and goodreads data...")
    logger.info(f"  Matching threshold: {threshold}")
    
    # Create normalized keys for matching
    logger.info("  Creating normalized match keys...")
    chapters_df["_match_key"] = chapters_df.apply(
        lambda row: normalize_author_title(row["Author"], row["Book Title"]), axis=1
    )
    goodreads_df["_match_key"] = goodreads_df.apply(
        lambda row: normalize_author_title(row["Author"], row["Title"]), axis=1
    )
    
    # Get unique book combinations from chapters
    chapters_books = chapters_df[["Author", "Book Title", "_match_key"]].drop_duplicates()
    goodreads_books = goodreads_df[["Author", "Title", "_match_key"]].drop_duplicates()
    
    logger.info(f"  Found {len(chapters_books):,} unique books in chapters.csv")
    logger.info(f"  Found {len(goodreads_books):,} unique books in goodreads.csv")
    
    # Create mapping from chapters match_key to goodreads index
    match_mapping = {}
    unmatched_chapters = []
    match_scores = []
    
    logger.info("  Performing fuzzy matching...")
    for idx, (_, chapter_row) in enumerate(chapters_books.iterrows(), 1):
        if idx % 100 == 0:
            logger.info(f"    Processed {idx}/{len(chapters_books)} books...")
        
        chapter_key = chapter_row["_match_key"]
        best_match_idx = None
        best_score = 0.0
        
        for gr_idx, gr_row in goodreads_books.iterrows():
            gr_key = gr_row["_match_key"]
            # Use SequenceMatcher for fuzzy matching
            similarity = SequenceMatcher(None, chapter_key, gr_key).ratio()
            
            if similarity > best_score:
                best_score = similarity
                best_match_idx = gr_idx
        
        if best_score >= threshold:
            match_mapping[chapter_key] = best_match_idx
            match_scores.append(best_score)
            logger.debug(
                f"    Matched: '{chapter_row['Book Title']}' -> '{goodreads_books.loc[best_match_idx, 'Title']}' "
                f"(score: {best_score:.3f})"
            )
        else:
            unmatched_chapters.append((chapter_row["Book Title"], chapter_row["Author"], best_score))
            logger.warning(
                f"    No match found for '{chapter_row['Book Title']}' by '{chapter_row['Author']}' "
                f"(best score: {best_score:.3f} < threshold {threshold})"
            )
    
    # Add book_id to chapters_df based on matching
    def get_book_id(row):
        match_key = row["_match_key"]
        if match_key in match_mapping:
            gr_idx = match_mapping[match_key]
            # Use goodreads ID as book_id
            return goodreads_df.loc[gr_idx, "ID"]
        return None
    
    chapters_df["book_id"] = chapters_df.apply(get_book_id, axis=1)
    
    # Log matching statistics
    matched_count = chapters_df["book_id"].notna().sum()
    total_sentences = len(chapters_df)
    match_rate = matched_count / total_sentences if total_sentences > 0 else 0
    matched_books = len(match_mapping)
    unmatched_books = len(unmatched_chapters)
    
    logger.info(f"  Matching results:")
    logger.info(f"    Matched sentences: {matched_count:,} / {total_sentences:,} ({match_rate:.1%})")
    logger.info(f"    Matched books: {matched_books:,} / {len(chapters_books):,}")
    logger.info(f"    Unmatched books: {unmatched_books:,}")
    
    if match_scores:
        logger.info(f"    Match score statistics:")
        logger.info(f"      Mean: {np.mean(match_scores):.3f}")
        logger.info(f"      Median: {np.median(match_scores):.3f}")
        logger.info(f"      Min: {np.min(match_scores):.3f}")
        logger.info(f"      Max: {np.max(match_scores):.3f}")
    
    if unmatched_chapters:
        logger.warning(f"  Top 10 unmatched books (by score):")
        sorted_unmatched = sorted(unmatched_chapters, key=lambda x: x[2], reverse=True)
        for title, author, score in sorted_unmatched[:10]:
            logger.warning(f"    '{title}' by '{author}' (best score: {score:.3f})")
    
    # Clean up temporary column
    chapters_df = chapters_df.drop(columns=["_match_key"])
    
    return chapters_df


def create_rating_class(
    df: pd.DataFrame,
    rating_col: str = "rating_mean",
    quantiles: tuple[float, float] = (0.33, 0.66),
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Create rating_class column based on quantiles.
    
    Args:
        df: DataFrame with rating_mean column
        rating_col: Name of rating column
        quantiles: Tuple of (low_quantile, high_quantile) for bad/mid/good split
        logger: Logger instance for detailed logging
        
    Returns:
        DataFrame with rating_class column added
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Creating rating_class based on {rating_col} quantiles...")
    logger.info(f"  Input dataframe shape: {df.shape}")
    logger.info(f"  Unique books: {df['book_id'].nunique()}")
    logger.info(f"  Rating column: {rating_col}")
    logger.info(f"  Quantiles requested: {quantiles}")
    
    # Validate rating column exists
    if rating_col not in df.columns:
        raise ValueError(f"Rating column '{rating_col}' not found in dataframe. Available columns: {list(df.columns)}")
    
    # Check for missing values in rating column
    missing_ratings = df[rating_col].isna().sum()
    total_rows = len(df)
    logger.info(f"  Missing ratings: {missing_ratings:,} / {total_rows:,} ({missing_ratings/total_rows:.1%})")
    
    # Calculate quantiles (per book, then aggregate)
    # Get one rating per book (all sentences from same book have same rating)
    book_ratings = df.groupby("book_id")[rating_col].first()
    logger.info(f"  Book-level ratings extracted: {len(book_ratings)} unique books")
    logger.info(f"  Book-level rating stats: mean={book_ratings.mean():.3f}, std={book_ratings.std():.3f}, "
                f"min={book_ratings.min():.3f}, max={book_ratings.max():.3f}")
    
    # Check for missing book ratings
    missing_book_ratings = book_ratings.isna().sum()
    if missing_book_ratings > 0:
        logger.warning(f"  Found {missing_book_ratings} books with missing ratings")
    
    # Calculate quantiles on book-level ratings (not sentence-level)
    low_q, high_q = book_ratings.quantile(quantiles)
    
    logger.info(f"  Rating quantiles (book-level): {low_q:.3f} ({quantiles[0]:.0%}), {high_q:.3f} ({quantiles[1]:.0%})")
    logger.info(f"  Books below low quantile: {(book_ratings < low_q).sum()}")
    logger.info(f"  Books between quantiles: {((book_ratings >= low_q) & (book_ratings <= high_q)).sum()}")
    logger.info(f"  Books above high quantile: {(book_ratings > high_q).sum()}")
    
    def assign_class(rating):
        if pd.isna(rating):
            return None
        if rating < low_q:
            return "bad"
        elif rating <= high_q:
            return "mid"
        else:
            return "good"
    
    df["rating_class"] = df[rating_col].apply(assign_class)
    
    # Log distribution (sentence-level, but all sentences from same book have same class)
    class_counts = df["rating_class"].value_counts()
    logger.info(f"  Rating class distribution (sentence-level):")
    for cls, count in class_counts.items():
        pct = count / len(df) * 100
        logger.info(f"    {cls}: {count:,} ({pct:.1f}%)")
    
    # Also log book-level distribution
    book_classes = df.groupby("book_id")["rating_class"].first()
    book_class_counts = book_classes.value_counts()
    logger.info(f"  Rating class distribution (book-level):")
    for cls, count in book_class_counts.items():
        pct = count / len(book_classes) * 100
        logger.info(f"    {cls}: {count:,} ({pct:.1f}%)")
    
    return df


def add_position_norm(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Add position_norm column (normalized sentence position within book).
    
    Args:
        df: DataFrame with book_id and sentence ordering
        logger: Logger instance for detailed logging
        
    Returns:
        DataFrame with position_norm column added
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Calculating position_norm (normalized sentence position per book)...")
    logger.info(f"  Input dataframe shape: {df.shape}")
    logger.info(f"  Unique books: {df['book_id'].nunique()}")
    
    # Note: Chapter column will be renamed to chapter_id later, so check which exists
    chapter_col = "chapter_id" if "chapter_id" in df.columns else "Chapter"
    logger.info(f"  Using chapter column: {chapter_col}")
    
    if chapter_col not in df.columns:
        logger.warning(f"  Chapter column '{chapter_col}' not found. Using row order within each book.")
        # Sort by book_id only if chapter column doesn't exist
        df = df.sort_values(["book_id"]).reset_index(drop=True)
    else:
        # Sort by book_id and chapter to ensure correct ordering
        df = df.sort_values(["book_id", chapter_col]).reset_index(drop=True)
        logger.info(f"  Sorted by book_id and {chapter_col}")
    
    # Group by book and calculate position
    # IMPORTANT: Sort within each group to ensure correct position calculation
    def calc_position(group):
        total = len(group)
        group = group.copy()
        # Sort within group by chapter (if available) to ensure correct order
        if chapter_col in group.columns:
            group = group.sort_values([chapter_col]).reset_index(drop=True)
        # Calculate normalized position: 0.0 (first) to 1.0 (last)
        group["position_norm"] = np.arange(total) / max(total - 1, 1)  # Avoid division by zero
        return group
    
    df = df.groupby("book_id", group_keys=False).apply(calc_position)
    
    # Log statistics
    logger.info(f"  Position norm calculated for {df['book_id'].nunique()} books")
    logger.info(f"  Position norm range: {df['position_norm'].min():.3f} - {df['position_norm'].max():.3f}")
    logger.info(f"  Position norm mean: {df['position_norm'].mean():.3f}, std: {df['position_norm'].std():.3f}")
    
    # Log distribution by quartiles
    position_quartiles = df['position_norm'].quantile([0.25, 0.5, 0.75])
    logger.info(f"  Position norm quartiles: Q1={position_quartiles[0.25]:.3f}, "
                f"Q2={position_quartiles[0.5]:.3f}, Q3={position_quartiles[0.75]:.3f}")
    
    # Check for any books with only one sentence (position_norm would be 0.0)
    book_sizes = df.groupby("book_id").size()
    single_sentence_books = (book_sizes == 1).sum()
    if single_sentence_books > 0:
        logger.warning(f"  Found {single_sentence_books} books with only 1 sentence (position_norm=0.0)")
    
    return df


def add_sentence_id(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Add sentence_id column (unique identifier for each sentence).
    
    Args:
        df: DataFrame with book_id, chapter_id columns
        logger: Logger instance for detailed logging
        
    Returns:
        DataFrame with sentence_id column added
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Creating sentence_id column...")
    logger.info(f"  Input dataframe shape: {df.shape}")
    
    # Validate required columns
    if "book_id" not in df.columns:
        raise ValueError("book_id column not found in dataframe")
    if "chapter_id" not in df.columns:
        raise ValueError("chapter_id column not found in dataframe")
    
    # Create sentence_id as: book_id_chapter_sentence_index
    df = df.sort_values(["book_id", "chapter_id"]).reset_index(drop=True)
    logger.info(f"  Sorted by book_id and chapter_id")
    
    def create_ids(group):
        group = group.copy()
        # Sort within group by chapter_id to ensure consistent ordering
        group = group.sort_values(["chapter_id"]).reset_index(drop=True)
        # Group by chapter_id within this book group and create sequential indices
        # Since we're already grouped by book_id, we only need to group by chapter_id
        group["sentence_id"] = (
            group["book_id"].astype(str) + "_" +
            group["chapter_id"].astype(str) + "_" +
            group.groupby("chapter_id", group_keys=False).cumcount().astype(str)
        )
        return group
    
    df = df.groupby("book_id", group_keys=False).apply(create_ids)
    
    # Verify uniqueness
    duplicate_count = df["sentence_id"].duplicated().sum()
    if duplicate_count > 0:
        logger.warning(f"  Found {duplicate_count} duplicate sentence_ids! Using sequential IDs instead.")
        df["sentence_id"] = "sent_" + df.index.astype(str)
        logger.info(f"  Replaced with sequential IDs: sent_0 to sent_{len(df)-1}")
    else:
        unique_count = df["sentence_id"].nunique()
        total_count = len(df)
        logger.info(f"  Created {unique_count} unique sentence_ids (matches total rows: {total_count})")
        if unique_count != total_count:
            logger.warning(f"  Mismatch: {unique_count} unique IDs but {total_count} total rows!")
    
    # Log sample IDs for verification
    sample_ids = df["sentence_id"].head(5).tolist()
    logger.debug(f"  Sample sentence_ids: {sample_ids}")
    
    return df


def main(
    chapters_path: Path,
    goodreads_path: Path,
    output_path: Path,
    min_ratings: int = 100,
    fuzzy_threshold: float = 0.85,
    quantiles: tuple[float, float] = (0.33, 0.66),
    logs_dir: Optional[Path] = None,
) -> None:
    """Main function to prepare sentence dataframe.
    
    Args:
        chapters_path: Path to chapters.csv
        goodreads_path: Path to goodreads.csv
        output_path: Path to save output parquet file
        min_ratings: Minimum ratings count to include book
        fuzzy_threshold: Minimum similarity for fuzzy matching (0-1)
        quantiles: Quantiles for rating_class creation
        logs_dir: Directory for log files (defaults to 'logs')
    """
    # Set up logging to file
    if logs_dir is None:
        logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"stage06_category_mapping_{timestamp}.log"
    logger = setup_logging(logs_dir, log_file=log_file)
    
    logger.info("=" * 60)
    logger.info("Preparing matched-only sentence-level dataframe for Stage 1 analysis")
    logger.info("=" * 60)
    logger.info("NOTE: This creates a 'matched only' dataframe containing only books")
    logger.info("      present in BOTH chapters.csv and goodreads.csv.")
    logger.info("      Stage 1 analysis will use only these matched books, even if")
    logger.info("      the BERTopic model was trained on all books.")
    logger.info("=" * 60)
    logger.info(f"Parameters:")
    logger.info(f"  chapters_path: {chapters_path}")
    logger.info(f"  goodreads_path: {goodreads_path}")
    logger.info(f"  output_path: {output_path}")
    logger.info(f"  min_ratings: {min_ratings}")
    logger.info(f"  fuzzy_threshold: {fuzzy_threshold}")
    logger.info(f"  quantiles: {quantiles}")
    logger.info("=" * 60)
    
    # Load data
    logger.info(f"Loading chapters.csv from {chapters_path}...")
    if not chapters_path.exists():
        raise FileNotFoundError(f"Chapters file not found: {chapters_path}")
    chapters_df = pd.read_csv(chapters_path)
    unique_books_chapters = len(chapters_df[['Author', 'Book Title']].drop_duplicates())
    logger.info(f"  Loaded {len(chapters_df):,} sentences from {unique_books_chapters} unique books")
    logger.info(f"  Chapters dataframe shape: {chapters_df.shape}")
    logger.info(f"  Chapters dataframe columns: {list(chapters_df.columns)}")
    
    logger.info(f"Loading goodreads.csv from {goodreads_path}...")
    if not goodreads_path.exists():
        raise FileNotFoundError(f"Goodreads file not found: {goodreads_path}")
    goodreads_df = pd.read_csv(goodreads_path)
    logger.info(f"  Loaded {len(goodreads_df):,} books from Goodreads")
    logger.info(f"  Goodreads dataframe shape: {goodreads_df.shape}")
    logger.info(f"  Goodreads dataframe columns: {list(goodreads_df.columns)}")
    
    # Validate required columns
    required_chapters = ["Author", "Book Title", "Chapter", "Sentence"]
    required_goodreads = ["ID", "Author", "Title", "Score", "RatingsCount"]
    
    missing_chapters = set(required_chapters) - set(chapters_df.columns)
    missing_goodreads = set(required_goodreads) - set(goodreads_df.columns)
    
    if missing_chapters:
        logger.error(f"Missing columns in chapters.csv: {missing_chapters}")
        raise ValueError(f"Missing columns in chapters.csv: {missing_chapters}")
    if missing_goodreads:
        logger.error(f"Missing columns in goodreads.csv: {missing_goodreads}")
        raise ValueError(f"Missing columns in goodreads.csv: {missing_goodreads}")
    
    logger.info("  All required columns present")
    
    # Filter goodreads by min_ratings
    initial_count = len(goodreads_df)
    logger.info(f"  Filtering Goodreads data by min_ratings >= {min_ratings}...")
    logger.info(f"  Initial Goodreads books: {initial_count:,}")
    
    # Log rating count statistics before filtering
    if "RatingsCount" in goodreads_df.columns:
        rating_stats = goodreads_df["RatingsCount"].describe()
        logger.info(f"  RatingsCount statistics (before filter):")
        logger.info(f"    Mean: {rating_stats['mean']:.1f}, Median: {rating_stats['50%']:.1f}")
        logger.info(f"    Min: {rating_stats['min']:.1f}, Max: {rating_stats['max']:.1f}")
        logger.info(f"    Books below threshold: {(goodreads_df['RatingsCount'] < min_ratings).sum():,}")
    
    goodreads_df = goodreads_df[goodreads_df["RatingsCount"] >= min_ratings].copy()
    filtered_count = initial_count - len(goodreads_df)
    if filtered_count > 0:
        logger.info(f"  Filtered out {filtered_count:,} books with < {min_ratings} ratings")
        logger.info(f"  Remaining Goodreads books: {len(goodreads_df):,}")
    else:
        logger.info(f"  No books filtered (all have >= {min_ratings} ratings)")
    
    # Rename goodreads columns for merging
    logger.info("Renaming Goodreads columns for merging...")
    goodreads_df = goodreads_df.rename(columns={
        "Score": "rating_mean",
        "RatingsCount": "rating_count",
    })
    logger.info(f"  Renamed: Score -> rating_mean, RatingsCount -> rating_count")
    
    # Log rating statistics from Goodreads
    if "rating_mean" in goodreads_df.columns:
        rating_mean_stats = goodreads_df["rating_mean"].describe()
        logger.info(f"  Rating mean statistics (Goodreads):")
        logger.info(f"    Mean: {rating_mean_stats['mean']:.3f}, Median: {rating_mean_stats['50%']:.3f}")
        logger.info(f"    Min: {rating_mean_stats['min']:.3f}, Max: {rating_mean_stats['max']:.3f}")
        logger.info(f"    Std: {rating_mean_stats['std']:.3f}")
    
    # Fuzzy match books
    logger.info("=" * 60)
    logger.info("Step 1: Fuzzy matching books between chapters and Goodreads")
    logger.info("=" * 60)
    chapters_df = fuzzy_match_books(chapters_df, goodreads_df, threshold=fuzzy_threshold, logger=logger)
    
    # Create "matched only" dataframe using inner join
    # This ensures Stage 1 analysis uses only books with complete metadata
    logger.info("=" * 60)
    logger.info("Step 2: Creating matched-only dataframe (inner join)")
    logger.info("=" * 60)
    logger.info("Creating matched-only dataframe: only books present in BOTH datasets")
    logger.info(f"  Chapters dataframe shape before merge: {chapters_df.shape}")
    logger.info(f"  Goodreads dataframe shape: {goodreads_df.shape}")
    
    # Track which books are in each dataset
    text_book_ids = set(chapters_df["book_id"].dropna().unique())
    meta_book_ids = set(goodreads_df["ID"].unique())
    
    only_in_texts = text_book_ids - meta_book_ids
    only_in_meta = meta_book_ids - text_book_ids
    matched_book_ids = text_book_ids & meta_book_ids
    
    logger.info(f"  Books in chapters.csv (with book_id): {len(text_book_ids)}")
    logger.info(f"  Books in goodreads.csv: {len(meta_book_ids)}")
    logger.info(f"  Books matched (in both): {len(matched_book_ids)}")
    logger.info(f"  Books only in texts (will be excluded): {len(only_in_texts)}")
    logger.info(f"  Books only in metadata (will be excluded): {len(only_in_meta)}")
    
    if only_in_texts:
        logger.warning(f"  Books only in texts (no metadata): {sorted(only_in_texts)}")
    if only_in_meta:
        logger.warning(f"  Books only in metadata (no text): {sorted(only_in_meta)}")
    
    # Use inner join to create matched-only dataframe
    # This ensures we only keep books present in BOTH datasets
    df = chapters_df.merge(
        goodreads_df[["ID", "rating_mean", "rating_count"]],
        left_on="book_id",
        right_on="ID",
        how="inner",  # Only keep books present in both
    )
    
    logger.info(f"  Matched-only dataframe shape: {df.shape}")
    logger.info(f"  Unique books in matched dataframe: {df['book_id'].nunique()}")
    logger.info(f"  Total sentences in matched dataframe: {len(df):,}")
    
    # Verify we got the expected number of books
    if df["book_id"].nunique() != len(matched_book_ids):
        logger.warning(
            f"  Mismatch: Expected {len(matched_book_ids)} matched books, "
            f"but got {df['book_id'].nunique()} in dataframe"
        )
    
    # Log sentence counts for dropped books
    if only_in_texts:
        dropped_sentences = chapters_df[chapters_df["book_id"].isin(only_in_texts)]
        dropped_count = len(dropped_sentences)
        logger.info(f"  Sentences from unmatched books (excluded): {dropped_count:,}")
        logger.info(f"  Sentences from matched books (included): {len(df):,}")
    
    # Create rating_class (only for matched books)
    # Note: This is calculated on the matched-only dataframe, ensuring rating classes
    # are based only on books with complete metadata
    logger.info("=" * 60)
    logger.info("Step 3: Creating rating_class based on quantiles (matched books only)")
    logger.info("=" * 60)
    df = create_rating_class(df, quantiles=quantiles, logger=logger)
    
    # Add position_norm
    logger.info("=" * 60)
    logger.info("Step 4: Calculating position_norm")
    logger.info("=" * 60)
    df = add_position_norm(df, logger=logger)
    
    # Rename Sentence to text (to match BERTopic training format)
    logger.info("=" * 60)
    logger.info("Step 5: Renaming columns")
    logger.info("=" * 60)
    logger.info("Renaming columns: Sentence -> text, Chapter -> chapter_id")
    df = df.rename(columns={"Sentence": "text", "Chapter": "chapter_id"})
    logger.info(f"  Columns after rename: {list(df.columns)}")
    
    # Add sentence_id
    logger.info("=" * 60)
    logger.info("Step 6: Creating sentence_id")
    logger.info("=" * 60)
    df = add_sentence_id(df, logger=logger)
    
    # Select final columns
    logger.info("=" * 60)
    logger.info("Step 7: Preparing final dataframe")
    logger.info("=" * 60)
    final_columns = [
        "sentence_id",
        "book_id",
        "chapter_id",
        "position_norm",
        "text",
        "rating_mean",
        "rating_count",
        "rating_class",
    ]
    
    # Validate all final columns exist
    missing_final = set(final_columns) - set(df.columns)
    if missing_final:
        logger.error(f"Missing final columns: {missing_final}")
        raise ValueError(f"Missing final columns: {missing_final}")
    
    # Keep Author and Book Title for reference (optional)
    df_final = df[final_columns + ["Author", "Book Title"]].copy()
    logger.info(f"  Final dataframe shape: {df_final.shape}")
    logger.info(f"  Final columns: {list(df_final.columns)}")
    
    # Validate text column
    null_text_count = df_final["text"].isnull().sum()
    if null_text_count > 0:
        logger.warning(f"  Found {null_text_count:,} null text values ({null_text_count/len(df_final):.1%})")
    else:
        logger.info(f"  All text values are non-null")
    
    # Validate other critical columns
    null_book_id = df_final["book_id"].isnull().sum()
    if null_book_id > 0:
        logger.error(f"  Found {null_book_id:,} null book_id values (should be 0 after filtering)")
    
    null_rating = df_final["rating_mean"].isnull().sum()
    if null_rating > 0:
        logger.warning(f"  Found {null_rating:,} null rating_mean values")
    
    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Saving to {output_path}...")
    df_final.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(f"  Successfully saved {len(df_final):,} rows to parquet")
    
    # Summary statistics
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY STATISTICS (MATCHED-ONLY DATASET)")
    logger.info("=" * 60)
    logger.info(f"Total sentences: {len(df_final):,}")
    logger.info(f"Total books: {df_final['book_id'].nunique()}")
    logger.info(f"Total chapters: {df_final['chapter_id'].nunique()}")
    logger.info("")
    logger.info("NOTE: This dataset contains only matched books (present in both")
    logger.info("      chapters.csv and goodreads.csv). All Stage 1 analysis should")
    logger.info("      use this matched-only dataset to ensure consistency.")
    
    # Sentence-level statistics
    logger.info("")
    logger.info("Sentence-level statistics:")
    logger.info(f"  Rating class distribution:")
    class_counts = df_final['rating_class'].value_counts()
    for cls, count in class_counts.items():
        pct = count / len(df_final) * 100
        logger.info(f"    {cls}: {count:,} ({pct:.1f}%)")
    
    # IMPORTANT: Calculate statistics on book-level ratings, not sentence-level
    # (sentence-level would weight books with more sentences more heavily)
    logger.info("")
    logger.info("Book-level rating statistics (corrected for sentence weighting):")
    book_ratings = df_final.groupby("book_id")["rating_mean"].first()
    logger.info(f"  Total books: {len(book_ratings):,}")
    logger.info(f"  Mean: {book_ratings.mean():.3f}")
    logger.info(f"  Median: {book_ratings.median():.3f}")
    logger.info(f"  Std: {book_ratings.std():.3f}")
    logger.info(f"  Min: {book_ratings.min():.3f}")
    logger.info(f"  Max: {book_ratings.max():.3f}")
    
    # Also show sentence-level for comparison (but note the weighting issue)
    logger.info("")
    logger.info("Sentence-level rating statistics (for comparison - weighted by sentence count):")
    logger.info(f"  Mean: {df_final['rating_mean'].mean():.3f}")
    logger.info(f"  Median: {df_final['rating_mean'].median():.3f}")
    logger.info(f"  Std: {df_final['rating_mean'].std():.3f}")
    logger.info(f"  Min: {df_final['rating_mean'].min():.3f}")
    logger.info(f"  Max: {df_final['rating_mean'].max():.3f}")
    logger.info("  NOTE: Sentence-level stats weight books with more sentences more heavily")
    
    # Book-level rating class distribution
    logger.info("")
    logger.info("Book-level rating class distribution:")
    book_classes = df_final.groupby("book_id")["rating_class"].first()
    book_class_counts = book_classes.value_counts()
    for cls, count in book_class_counts.items():
        pct = count / len(book_classes) * 100
        logger.info(f"    {cls}: {count:,} ({pct:.1f}%)")
    
    # Position statistics
    logger.info("")
    logger.info("Position statistics:")
    logger.info(f"  position_norm range: {df_final['position_norm'].min():.3f} - {df_final['position_norm'].max():.3f}")
    logger.info(f"  position_norm mean: {df_final['position_norm'].mean():.3f}")
    
    # Sentences per book statistics
    sentences_per_book = df_final.groupby("book_id").size()
    logger.info("")
    logger.info("Sentences per book statistics:")
    logger.info(f"  Mean: {sentences_per_book.mean():.1f}")
    logger.info(f"  Median: {sentences_per_book.median():.1f}")
    logger.info(f"  Min: {sentences_per_book.min():,}")
    logger.info(f"  Max: {sentences_per_book.max():,}")
    logger.info(f"  Std: {sentences_per_book.std():.1f}")
    
    logger.info("")
    logger.info(f"Output saved to: {output_path}")
    logger.info(f"Log file saved to: {logs_dir / log_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare sentence-level dataframe with ratings for Stage 1 analysis"
    )
    parser.add_argument(
        "--chapters",
        type=Path,
        default=Path("data/processed/chapters.csv"),
        help="Path to chapters.csv",
    )
    parser.add_argument(
        "--goodreads",
        type=Path,
        default=Path("data/processed/goodreads.csv"),
        help="Path to goodreads.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/sentence_df_with_ratings.parquet"),
        help="Path to save output parquet file",
    )
    parser.add_argument(
        "--min-ratings",
        type=int,
        default=100,
        help="Minimum ratings count to include book",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.85,
        help="Minimum similarity score for fuzzy matching (0-1)",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs=2,
        default=[0.33, 0.66],
        help="Quantiles for rating_class creation (low, high)",
    )
    
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for log files",
    )
    
    args = parser.parse_args()
    
    main(
        chapters_path=args.chapters,
        goodreads_path=args.goodreads,
        output_path=args.output,
        min_ratings=args.min_ratings,
        fuzzy_threshold=args.fuzzy_threshold,
        quantiles=tuple(args.quantiles),
        logs_dir=args.logs_dir,
    )

