#!/usr/bin/env python3
"""
generate_tertile_topic_probs.py

Generates topic probabilities for begin/middle/end tertiles of each book by:
1. Loading sentence dataframe with text ordered by book
2. For each book, splitting ordered sentences into 3 tertiles (begin/middle/end)
3. Chunking each tertile into manageable docs (prevents embedding truncation)
4. Inferring topic mixtures for each chunk using BERTopic
5. Aggregating chunk probabilities within tertile (weighted mean)
6. Outputting tertile_topic_probs.parquet: [book_id, tertile, topic_id, prob]

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
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from bertopic import BERTopic
from tqdm import tqdm

# Import utilities from data_preparation module (same directory)
# Use importlib since module name starts with a number
import importlib.util
import sys
spec = importlib.util.spec_from_file_location(
    "generate_topic_probabilities_final",
    Path(__file__).parent / "03_generate_topic_probabilities_final.py"
)
module = importlib.util.module_from_spec(spec)
# Add to sys.modules before exec_module to fix dataclass issues
sys.modules["generate_topic_probabilities_final"] = module
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
    chunk_size_sentences: int = 40,
    min_sentences: int = 3,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split a book's ordered sentences into three tertiles (begin/middle/end) and return
    **lists of chunked documents** per tertile.

    Why chunk?
      BERTopic uses an embedding model with max sequence length; joining an entire tertile
      into one mega-string risks truncation. Instead we create many shorter chunks, run
      BERTopic.transform on each chunk, and later aggregate probabilities across chunks.

    Args:
        book_sentences: DataFrame with sentences for one book (order should already be correct)
        text_col: column containing sentence text
        chunk_size_sentences: how many sentences to concatenate per chunk doc
        min_sentences: minimum number of non-empty sentences required to compute tertiles

    Returns:
        (begin_docs, middle_docs, end_docs) where each element is a list[str] of chunk docs.
        Empty lists indicate the book should be skipped.
    """
    if text_col not in book_sentences.columns:
        raise KeyError(f"Missing text column: {text_col}")

    texts = book_sentences[text_col].astype(str).fillna("").tolist()
    texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    n = len(texts)
    if n < min_sentences:
        return [], [], []

    idx = np.arange(n)
    parts = np.array_split(idx, 3)

    def _chunk_from_indices(indices: np.ndarray) -> List[str]:
        if indices.size == 0:
            return []
        out_docs: List[str] = []
        buf: List[str] = []
        for i in indices.tolist():
            buf.append(texts[i])
            if len(buf) >= chunk_size_sentences:
                out_docs.append(" ".join(buf))
                buf = []
        if buf:
            out_docs.append(" ".join(buf))
        out_docs = [d.strip() for d in out_docs if d.strip()]
        return out_docs

    begin_docs = _chunk_from_indices(parts[0])
    middle_docs = _chunk_from_indices(parts[1])
    end_docs = _chunk_from_indices(parts[2])

    if not begin_docs or not middle_docs or not end_docs:
        return [], [], []

    return begin_docs, middle_docs, end_docs


def process_tertiles_for_all_books(
    sentence_df: pd.DataFrame,
    book_id_col: str,
    text_col: str,
    topic_model: BERTopic,
    topic_ids: List[int],
    logger: logging.Logger,
    batch_size: int = 0,
    chunk_size_sentences: int = 40,
    min_sentences: int = 3,
) -> pd.DataFrame:
    """
    Process all books: split into tertiles and infer topic probabilities.
    
    Returns:
        DataFrame with columns [book_id, tertile, topic_id, prob]
    """
    logger.info("Processing tertiles for all books...")
    
    rows = []
    n_books = sentence_df[book_id_col].nunique(dropna=True)
    for book_id, book_sentences in tqdm(
        sentence_df.groupby(book_id_col, sort=False),
        total=n_books,
        desc="Books",
        ncols=100
    ):
        book_sentences = book_sentences.copy()
        # Preserve original order (important for tertile split)
        # If both chapter_id and sentence_index exist, sort by both
        sort_cols = [c for c in ["chapter_id", "sentence_index"] if c in book_sentences.columns]
        if sort_cols:
            book_sentences = book_sentences.sort_values(sort_cols)
        else:
            # Without ordering columns, we assume sentence_df is already ordered within book_id.
            # This is risky if upstream data isn't stable-sorted.
            logger.warning(f"⚠️  No ordering columns found for book {book_id}; assuming existing row order is narrative order.")
        # Otherwise, assume order is already correct
        
        # Split into tertiles
        begin_texts, middle_texts, end_texts = split_book_into_tertiles(
            book_sentences,
            text_col,
            chunk_size_sentences=chunk_size_sentences,
            min_sentences=min_sentences,
        )
        
        # Skip if any tertile is empty
        if (len(begin_texts) == 0) or (len(middle_texts) == 0) or (len(end_texts) == 0):
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
            
            # probs shape: (n_chunks, n_topics). Aggregate across chunks.
            if probs.ndim != 2 or probs.shape[0] == 0:
                tertile_probs = np.zeros(len(topic_ids), dtype=float)
            else:
                # Weight each chunk by word count (proxy for content volume)
                weights = np.array([max(1, len(str(t).split())) for t in texts], dtype=float)
                if len(weights) != probs.shape[0]:
                    # fallback if transform filtered anything unexpectedly
                    weights = np.ones(probs.shape[0], dtype=float)
                tertile_probs = np.average(np.nan_to_num(probs, nan=0.0), axis=0, weights=weights)
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
    # Alias for downstream notebooks expecting a 'segment' column
    out["segment"] = out["tertile"]
    out = out[["book_id","segment","tertile","topic_id","prob"]]
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

    # CLI compatibility (some runners always pass --no-cache)
    # This script currently does not implement caching; these flags are accepted and ignored.
    parser.add_argument(
        "--cache",
        action="store_true",
        help="(Ignored) Present for CLI compatibility. This script does not cache."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="(Ignored) Present for CLI compatibility. This script does not cache."
    )

    parser.add_argument(
        "--chunk-size-sentences",
        type=int,
        default=40,
        help="How many sentences to join per chunk doc before BERTopic.transform (prevents long-text truncation)."
    )
    parser.add_argument(
        "--min-sentences",
        type=int,
        default=3,
        help="Minimum number of non-empty sentences required to compute begin/middle/end tertiles."
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
    if getattr(args, "cache", False) or getattr(args, "no_cache", False):
        logger.warning("⚠️  --cache/--no-cache flags are ignored in this script (kept for CLI compatibility).")
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
        chunk_size_sentences=args.chunk_size_sentences,
        min_sentences=args.min_sentences,
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

