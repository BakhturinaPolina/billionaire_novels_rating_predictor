#!/usr/bin/env python3
"""
generate_tertile_topic_probs.py

Generates topic probabilities for begin/middle/end tertiles of each book by:
1. Loading sentence dataframe with text ordered by book
2. For each book, splitting sentences into 3 equal tertiles (begin/middle/end)
3. Inferring topic mixtures for each tertile using BERTopic model
4. Outputting tertile_topic_probs.parquet: [book_id, tertile, topic_id, prob]

Usage:
  python generate_tertile_topic_probs.py \
    --sentence-df data/processed/sentence_df_with_topics.parquet \
    --model-path models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_radway_mappings \
    --output-dir results/stage10_correlation_analysis/data_preparation \
    --book-id-source goodreads \
    --goodreads-id-col ID
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from bertopic import BERTopic
from tqdm import tqdm

# Import utilities from data_preparation module (same directory)
# Use importlib since module name starts with a number
import importlib.util
spec = importlib.util.spec_from_file_location(
    "generate_topic_probabilities_final",
    Path(__file__).parent / "03_generate_topic_probabilities_final.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
# Import from the loaded module
load_bertopic_model = module.load_bertopic_model
load_sentence_dataframe = module.load_sentence_dataframe
load_excluded_ids = module.load_excluded_ids
infer_topic_ids_for_prob_columns = module.infer_topic_ids_for_prob_columns
transform_texts_in_batches = module.transform_texts_in_batches
normalize_id_series = module.normalize_id_series


def setup_logging(logs_dir: Path, log_file: str = "generate_tertile_topic_probs.log") -> logging.Logger:
    """Setup logging to file and console."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tertile_topic_probs")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(logs_dir / log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def split_book_into_tertiles(
    book_sentences: pd.DataFrame,
    text_col: str,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split a book's sentences into three tertiles (begin, middle, end).
    
    Args:
        book_sentences: DataFrame with sentences for one book (preserves order)
        text_col: Column name containing sentence text
        
    Returns:
        Tuple of (begin_texts, middle_texts, end_texts) - each is list of strings
    """
    n_sentences = len(book_sentences)
    if n_sentences == 0:
        return [], [], []
    
    # Calculate tertile boundaries
    third = n_sentences // 3
    begin_end = third
    middle_end = 2 * third
    
    # Split into tertiles
    begin_df = book_sentences.iloc[:begin_end]
    middle_df = book_sentences.iloc[begin_end:middle_end]
    end_df = book_sentences.iloc[middle_end:]
    
    # Concatenate sentences within each tertile (join with space)
    begin_text = " ".join(begin_df[text_col].astype(str).tolist())
    middle_text = " ".join(middle_df[text_col].astype(str).tolist())
    end_text = " ".join(end_df[text_col].astype(str).tolist())
    
    return [begin_text], [middle_text], [end_text]


def process_tertiles_for_all_books(
    sentence_df: pd.DataFrame,
    book_id_col: str,
    text_col: str,
    topic_model: BERTopic,
    topic_ids: List[int],
    logger: logging.Logger,
    batch_size: int = 0,
) -> pd.DataFrame:
    """
    Process all books: split into tertiles and infer topic probabilities.
    
    Returns:
        DataFrame with columns [book_id, tertile, topic_id, prob]
    """
    logger.info("Processing tertiles for all books...")
    
    rows = []
    books = sentence_df[book_id_col].unique()
    
    for book_id in tqdm(books, desc="Books", ncols=100):
        book_sentences = sentence_df[sentence_df[book_id_col] == book_id].copy()
        
        # Preserve original order (important for tertile split)
        # If there's a chapter_id or sentence index, use it; otherwise keep as-is
        if "chapter_id" in book_sentences.columns:
            book_sentences = book_sentences.sort_values(["chapter_id"])
        elif "sentence_index" in book_sentences.columns:
            book_sentences = book_sentences.sort_values(["sentence_index"])
        # Otherwise, assume order is already correct
        
        # Split into tertiles
        begin_texts, middle_texts, end_texts = split_book_into_tertiles(book_sentences, text_col)
        
        # Skip if any tertile is empty
        if not begin_texts[0] or not middle_texts[0] or not end_texts[0]:
            logger.warning(f"Book {book_id} has empty tertile(s), skipping")
            continue
        
        # Infer topic probabilities for each tertile
        tertile_texts = {
            "begin": begin_texts,
            "middle": middle_texts,
            "end": end_texts,
        }
        
        for tertile_name, texts in tertile_texts.items():
            # Transform using BERTopic
            _, probs = transform_texts_in_batches(topic_model, texts, batch_size, logger)
            
            if probs.size == 0:
                logger.warning(f"Book {book_id} tertile {tertile_name} produced empty probabilities")
                continue
            
            # probs shape: (1, n_topics) - single document
            tertile_probs = probs[0] if probs.shape[0] > 0 else np.zeros(len(topic_ids))
            
            # Normalize to ensure sum = 1.0
            prob_sum = tertile_probs.sum()
            if prob_sum > 0:
                tertile_probs = tertile_probs / prob_sum
            else:
                logger.warning(f"Book {book_id} tertile {tertile_name} has zero probability sum")
                continue
            
            # Emit one row per topic
            for col_i, topic_id in enumerate(topic_ids):
                rows.append((
                    str(book_id),
                    tertile_name,
                    int(topic_id),
                    float(tertile_probs[col_i])
                ))
    
    out = pd.DataFrame(rows, columns=["book_id", "tertile", "topic_id", "prob"])
    logger.info(f"✓ tertile_topic_probs: {out.shape[0]:,} rows "
                f"({out['book_id'].nunique():,} books × 3 tertiles × {out['topic_id'].nunique():,} topics)")
    
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate topic probabilities for book tertiles (begin/middle/end)"
    )
    parser.add_argument(
        "--sentence-df",
        type=Path,
        required=True,
        help="Path to sentence_df_with_topics.parquet (or .csv)"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to BERTopic model (directory or .pkl)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Directory for logs (default: <output-dir>/logs)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Batch size for BERTopic.transform (0 = all at once, rarely needed for tertiles)"
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default=None,
        help="Text column in sentence_df (auto-detect if omitted)"
    )
    
    # ID strategy (same as generate_topic_probabilities_final.py)
    parser.add_argument(
        "--book-id-source",
        choices=["goodreads", "existing", "author_title"],
        default="goodreads",
        help="Which identifier to output as book_id. Use 'goodreads' for reliable merges."
    )
    parser.add_argument(
        "--goodreads-id-col",
        type=str,
        default="ID",
        help="Column in sentence_df containing Goodreads id (e.g., 'ID'). Used when --book-id-source=goodreads."
    )
    parser.add_argument(
        "--author-col",
        type=str,
        default="Author",
        help="Author column for author_title fallback"
    )
    parser.add_argument(
        "--title-col",
        type=str,
        default="Book Title",
        help="Title column for author_title fallback"
    )
    
    # Cohort control
    parser.add_argument(
        "--exclude-book-ids",
        type=str,
        default=None,
        help="Comma-separated IDs or path to CSV with column 'book_id' to exclude from processing."
    )
    
    # Output options
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write CSV version of output (in addition to Parquet)."
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.logs_dir or (args.output_dir / "logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(logs_dir)
    logger.info("=== generate_tertile_topic_probs.py ===")
    logger.info(f"sentence_df: {args.sentence_df}")
    logger.info(f"model_path : {args.model_path}")
    logger.info(f"output_dir : {args.output_dir}")
    logger.info(f"book_id_source={args.book_id_source} | goodreads_id_col={args.goodreads_id_col}")
    logger.info(f"exclude_book_ids={args.exclude_book_ids}")
    
    exclude_ids = load_excluded_ids(args.exclude_book_ids)
    if exclude_ids:
        logger.info(f"Loaded {len(exclude_ids)} excluded IDs")
    
    # Load sentence dataframe
    sent = load_sentence_dataframe(
        args.sentence_df,
        book_id_source=args.book_id_source,
        goodreads_id_col=args.goodreads_id_col,
        author_col=args.author_col,
        title_col=args.title_col,
        text_col=args.text_col,
        exclude_ids=exclude_ids,
        logger=logger,
    )
    df = sent.df
    
    # Load model
    topic_model = load_bertopic_model(args.model_path, logger=logger)
    
    # Infer topic IDs
    # We need to know how many topics the model has
    # Do a dummy transform to get the number of topics
    dummy_text = ["dummy"]
    _, dummy_probs = topic_model.transform(dummy_text)
    n_topics = dummy_probs.shape[1]
    topic_ids = infer_topic_ids_for_prob_columns(topic_model, n_topics, logger=logger)
    logger.info(f"Topic id labeling: {topic_ids[:10]}{'...' if len(topic_ids)>10 else ''}")
    
    # Process tertiles
    tertile_topic_probs = process_tertiles_for_all_books(
        df,
        book_id_col=sent.book_id_col,
        text_col=sent.text_col,
        topic_model=topic_model,
        topic_ids=topic_ids,
        logger=logger,
        batch_size=args.batch_size,
    )
    
    # Write output
    topic_probs_dir = args.output_dir / "topic_probabilities"
    topic_probs_dir.mkdir(parents=True, exist_ok=True)
    out_file = topic_probs_dir / "tertile_topic_probs.parquet"
    tertile_topic_probs.to_parquet(out_file, index=False)
    logger.info(f"✓ Wrote: {out_file}")
    
    if args.write_csv:
        csv_file = topic_probs_dir / "tertile_topic_probs.csv"
        tertile_topic_probs.to_csv(csv_file, index=False)
        logger.info(f"✓ Wrote: {csv_file}")
    
    # Validation
    n_books = tertile_topic_probs["book_id"].nunique()
    n_tertiles = tertile_topic_probs["tertile"].nunique()
    n_topics = tertile_topic_probs["topic_id"].nunique()
    expected_rows = n_books * n_tertiles * n_topics
    actual_rows = len(tertile_topic_probs)
    
    logger.info(f"Validation: {n_books} books × {n_tertiles} tertiles × {n_topics} topics = {expected_rows} expected rows")
    logger.info(f"Actual rows: {actual_rows}")
    
    if actual_rows != expected_rows:
        logger.warning(f"⚠ Row count mismatch! Expected {expected_rows}, got {actual_rows}")
    else:
        logger.info("✓ Row count matches expected")
    
    # Check probability normalization per tertile
    prob_sums = tertile_topic_probs.groupby(["book_id", "tertile"])["prob"].sum()
    if not prob_sums.between(0.99, 1.01).all():
        logger.warning(f"⚠ Some tertiles don't sum to 1.0: {prob_sums[~prob_sums.between(0.99, 1.01)]}")
    else:
        logger.info("✓ All tertile probabilities sum to ~1.0")
    
    logger.info("DONE.")


if __name__ == "__main__":
    main()

