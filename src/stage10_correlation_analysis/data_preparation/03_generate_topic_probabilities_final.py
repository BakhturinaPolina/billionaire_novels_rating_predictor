#!/usr/bin/env python3
"""
generate_topic_probabilities_final.py

Unified, "correct-by-construction" script to generate BERTopic topic probabilities at:
- book level (normalized per book)
- chapter level (normalized per chapter)

Key guarantees / fixes vs older versions:
1) Goodreads-first `book_id` (best for merging). You choose via --book-id-source.
2) Robust ID normalization (strip, handle "123.0" artifacts).
3) Optional cohort exclusion (e.g., books missing from sentence_df).
4) Safer cache fingerprinting (sentence_df file + model path mtime/size; supports directories).
5) Topic-id alignment: writes topic_id using BERTopic's topic IDs when we can infer ordering.
6) NEW: Normalizes mean probability vectors so sums are ~1 per book/chapter.
7) NEW: Logs min/median/max pre-normalization mass per book/chapter (like sanity check).

Outputs (Parquet by default, saved to <output-dir>/topic_probabilities/):
- book_topic_probs.parquet:  [book_id, topic_id, prob] (prob sums to ~1 per book)
- chapter_topic_probs.parquet: [book_id, chapter_id, topic_id, prob] (prob sums to ~1 per chapter)

Typical usage (Goodreads IDs, recommended):
  python generate_topic_probabilities_final.py \
    --sentence-df data/processed/sentence_df_with_topics.parquet \
    --model-path models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_radway_mappings \
    --output-dir results/stage10_correlation_analysis/data_preparation \
    --book-id-source goodreads \
    --goodreads-id-col ID \
    --exclude-book-ids notebooks/07_analysis/statistical_analysis/excluded_book_ids.csv \
    --no-cache
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm


# -------------------------
# Logging
# -------------------------
def setup_logging(logs_dir: Path, log_file: str = "generate_topic_probabilities.log") -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("topic_probs")
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


# -------------------------
# ID helpers
# -------------------------
_ID_FLOAT_FIX_RE = re.compile(r"^([0-9]+)\.0$")

def normalize_id_series(s: pd.Series, *, name: str, logger: Optional[logging.Logger] = None) -> pd.Series:
    """Normalize IDs for safe joins:
    - string dtype
    - strip whitespace
    - convert '', 'nan', 'none' -> NA
    - convert '123.0' -> '123' (common parquet/csv artifact)
    """
    if logger is None:
        logger = logging.getLogger("topic_probs")

    out = s.astype("string").str.strip()
    out = out.mask(out.str.lower().isin(["", "nan", "none"]))
    out = out.str.replace(_ID_FLOAT_FIX_RE, r"\1", regex=True)

    missing = out.isna()
    if missing.any():
        logger.warning(f"{missing.sum():,} rows have missing {name} values")
    return out


def slugify_text(x: str) -> str:
    """Conservative slugify used only for author_title fallback IDs."""
    if x is None:
        return ""
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"[^\w\s-]", "", x)  # drop punctuation
    x = x.replace(" ", "_")
    return x


def create_author_title_id(author: str, title: str) -> str:
    return f"{slugify_text(author)}__{slugify_text(title)}"


def load_excluded_ids(exclude_arg: Optional[str]) -> List[str]:
    """Load excluded ids from:
    - None
    - comma-separated string: '1,2,3'
    - path to CSV/TSV with a column named 'book_id' (or first column)
    """
    if not exclude_arg:
        return []

    p = Path(exclude_arg)
    if p.exists() and p.is_file():
        df = pd.read_csv(p)
        if "book_id" in df.columns:
            ids = df["book_id"].astype("string").str.strip().tolist()
        else:
            ids = df.iloc[:, 0].astype("string").str.strip().tolist()
        return [i for i in ids if i and i.lower() not in ["nan", "none"]]

    # otherwise treat as comma-separated list
    ids = [x.strip() for x in exclude_arg.split(",")]
    return [i for i in ids if i and i.lower() not in ["nan", "none"]]


def log_nan_diagnostics(
    probs: np.ndarray,
    df: pd.DataFrame,
    text_col: str,
    book_id_col: str,
    chapter_id_col: str,
    logger: logging.Logger,
    sample_size: int = 5,
) -> None:
    """Log detailed diagnostics for NaN probabilities."""
    row_nan_mask = np.isnan(probs).any(axis=1)
    n_rows_with_nan = row_nan_mask.sum()
    total_rows = len(probs)
    nan_rows_pct = n_rows_with_nan / max(total_rows, 1) * 100

    logger.warning(f"  Rows with any NaN: {n_rows_with_nan:,} / {total_rows:,} ({nan_rows_pct:.2f}%)")

    # Per-topic NaN counts (top 10)
    topic_nan_counts = np.isnan(probs).sum(axis=0)
    top_topic_idx = np.argsort(topic_nan_counts)[::-1]
    top_topics = [(int(i), int(topic_nan_counts[i])) for i in top_topic_idx[:10] if topic_nan_counts[i] > 0]
    if top_topics:
        logger.warning(f"  Topics with most NaNs (topic_id, nan_count): {top_topics}")

    # Per-book NaN counts (top 10)
    if book_id_col in df.columns:
        book_counts = df.loc[row_nan_mask, book_id_col].value_counts().head(10)
        if not book_counts.empty:
            logger.warning("  Books with NaNs (top 10):")
            logger.warning("\n" + book_counts.to_string())

    # Sample rows with NaNs for inspection
    sample_cols = [c for c in [book_id_col, chapter_id_col, text_col] if c in df.columns]
    if sample_cols:
        sample = df.loc[row_nan_mask, sample_cols].head(sample_size)
        if not sample.empty:
            logger.warning(f"  Sample rows with NaNs (first {len(sample)}):")
            logger.warning("\n" + sample.to_string(index=False))


# -------------------------
# Cache fingerprinting
# -------------------------
def _fingerprint_file(p: Path) -> str:
    st = p.stat()
    return f"FILE:{p.resolve()}:{st.st_size}:{st.st_mtime_ns}"


def _fingerprint_dir(p: Path, max_files: int = 5000) -> str:
    """Fingerprint a directory by (count, total size, max mtime_ns) of contained files.
    Avoids hashing full contents (fast enough, still captures changes).
    """
    count = 0
    total_size = 0
    max_mtime = 0
    # deterministic traversal
    for root, _, files in os.walk(p):
        for fn in sorted(files):
            fp = Path(root) / fn
            try:
                st = fp.stat()
            except OSError:
                continue
            count += 1
            total_size += st.st_size
            max_mtime = max(max_mtime, st.st_mtime_ns)
            if count >= max_files:
                # cap to avoid pathological directories
                break
        if count >= max_files:
            break
    return f"DIR:{p.resolve()}:{count}:{total_size}:{max_mtime}"


def make_cache_key(sentence_df_path: Path, model_path: Path, num_texts: int, extra: str = "") -> str:
    parts = [
        _fingerprint_file(sentence_df_path),
        _fingerprint_dir(model_path) if model_path.is_dir() else _fingerprint_file(model_path),
        f"N:{num_texts}",
        f"EXTRA:{extra}",
    ]
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


# -------------------------
# BERTopic utilities
# -------------------------
def load_bertopic_model(model_path: Path, logger: Optional[logging.Logger] = None):
    if logger is None:
        logger = logging.getLogger("topic_probs")

    logger.info(f"Loading BERTopic model from: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    from bertopic import BERTopic

    # BERTopic.load handles directories
    model = BERTopic.load(str(model_path))
    logger.info("✓ Loaded BERTopic model")
    return model


def infer_topic_ids_for_prob_columns(topic_model, n_topics: int, logger: Optional[logging.Logger] = None) -> List[int]:
    """Best-effort mapping from probability columns -> BERTopic topic IDs.

    If we can infer that BERTopic returns probability vectors aligned to get_topic_info() (excluding -1),
    we use those topic IDs. Otherwise we fall back to [0..n_topics-1] and warn loudly.

    This matters if you later join probs to topic lookup tables by topic_id.
    """
    if logger is None:
        logger = logging.getLogger("topic_probs")

    topic_ids = None
    if hasattr(topic_model, "get_topic_info"):
        info = topic_model.get_topic_info()
        if "Topic" in info.columns:
            ids = info.loc[info["Topic"] != -1, "Topic"].tolist()
            if len(ids) == n_topics:
                topic_ids = [int(x) for x in ids]

    if topic_ids is None:
        logger.warning(
            "⚠ Could not confidently infer BERTopic topic-id ordering for probability columns. "
            "Falling back to column indices 0..n_topics-1. "
            "If your topic IDs are non-contiguous, downstream joins may be wrong."
        )
        topic_ids = list(range(n_topics))

    return topic_ids


def transform_texts_in_batches(topic_model, texts: List[str], batch_size: int, logger: logging.Logger) -> Tuple[List[int], np.ndarray]:
    """Run BERTopic.transform with optional batching. Returns topics list and (n_docs, n_topics) probs array."""
    if batch_size <= 0 or batch_size >= len(texts):
        logger.info("Processing all documents at once...")
        topics, probs = topic_model.transform(texts)
        topics_list = topics.tolist() if isinstance(topics, np.ndarray) else list(topics)
        probs_array = probs if isinstance(probs, np.ndarray) else np.array(probs)
        return topics_list, probs_array

    logger.info(f"Processing {len(texts):,} documents in batches of {batch_size:,}...")
    all_topics: List[int] = []
    all_probs: List[np.ndarray] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    pbar = tqdm(total=len(texts), unit="doc", unit_scale=True, desc="Processing batches", ncols=100, mininterval=1.0)
    for b in range(total_batches):
        start = b * batch_size
        end = min(len(texts), start + batch_size)
        batch = texts[start:end]
        topics, probs = topic_model.transform(batch)
        topics_list = topics.tolist() if isinstance(topics, np.ndarray) else list(topics)
        probs_array = probs if isinstance(probs, np.ndarray) else np.array(probs)

        all_topics.extend(topics_list)
        all_probs.append(probs_array)

        pbar.update(len(batch))
        pbar.set_postfix({"batch": f"{b+1}/{total_batches}"})
    pbar.close()

    probs_full = np.vstack(all_probs) if all_probs else np.zeros((0, 0), dtype=float)
    return all_topics, probs_full


# -------------------------
# Data loading
# -------------------------
@dataclass
class SentenceDF:
    df: pd.DataFrame
    book_id_col: str
    chapter_id_col: str
    text_col: str


def autodetect_text_col(df: pd.DataFrame) -> str:
    for c in ["Sentence", "sentence", "text", "Text", "chunk", "content"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not autodetect a text column. Columns: {df.columns.tolist()}")


def ensure_chapter_id(df: pd.DataFrame) -> str:
    """Ensure df has chapter_id column; returns column name."""
    if "chapter_id" in df.columns:
        return "chapter_id"
    if "Chapter" in df.columns:
        df["chapter_id"] = df["Chapter"]
        return "chapter_id"
    if "chapter" in df.columns:
        df["chapter_id"] = df["chapter"]
        return "chapter_id"
    # fallback single chapter
    df["chapter_id"] = 0
    return "chapter_id"


def load_sentence_dataframe(
    sentence_df_path: Path,
    *,
    book_id_source: str,
    goodreads_id_col: str,
    author_col: str,
    title_col: str,
    text_col: Optional[str],
    exclude_ids: List[str],
    logger: logging.Logger,
) -> SentenceDF:
    logger.info(f"Loading sentence dataframe: {sentence_df_path}")
    if not sentence_df_path.exists():
        raise FileNotFoundError(f"Sentence dataframe not found: {sentence_df_path}")

    # Parquet is expected, but allow csv for convenience
    if sentence_df_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(sentence_df_path)
    else:
        df = pd.read_csv(sentence_df_path)

    logger.info(f"✓ Loaded sentence_df: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Text column
    if text_col is None:
        text_col = autodetect_text_col(df)
    if text_col not in df.columns:
        raise ValueError(f"text_col='{text_col}' not found. Columns: {df.columns.tolist()}")
    logger.info(f"Using text column: {text_col}")

    # Chapter id
    chapter_id_col = ensure_chapter_id(df)

    # Book id
    if book_id_source == "goodreads":
        if goodreads_id_col not in df.columns:
            raise ValueError(
                f"❌ REQUIRED: book_id_source=goodreads but column '{goodreads_id_col}' not found.\n"
                f"Available columns: {df.columns.tolist()}\n"
                f"Fix: add Goodreads ID to sentence_df upstream (merge with goodreads.csv), then re-run."
            )
        df["book_id"] = normalize_id_series(df[goodreads_id_col], name=goodreads_id_col, logger=logger)
        if df["book_id"].isna().any():
            raise ValueError(
                f"Found missing Goodreads IDs in column '{goodreads_id_col}'. "
                "Cannot produce reliable book-level outputs; fix upstream joins into sentence_df."
            )
        book_id_col = "book_id"
        logger.info(f"✓ Using Goodreads-based book_id from '{goodreads_id_col}'")

    elif book_id_source == "existing":
        if "book_id" not in df.columns:
            raise ValueError("book_id_source=existing but no 'book_id' column found in sentence_df")
        df["book_id"] = normalize_id_series(df["book_id"], name="book_id", logger=logger)
        book_id_col = "book_id"
        logger.info("✓ Using existing 'book_id' from sentence_df")

    elif book_id_source == "author_title":
        if author_col not in df.columns or title_col not in df.columns:
            raise ValueError(f"book_id_source=author_title requires '{author_col}' and '{title_col}'")
        df["book_id"] = df.apply(lambda r: create_author_title_id(r[author_col], r[title_col]), axis=1)
        df["book_id"] = normalize_id_series(df["book_id"], name="author_title_id", logger=logger)
        book_id_col = "book_id"
        logger.info("✓ Using Author+Title fallback book_id (NOT recommended for merging)")

    else:
        raise ValueError(f"Unknown --book-id-source: {book_id_source}")

    # Exclusions
    if exclude_ids:
        excl = set(pd.Series(exclude_ids, dtype="string").str.strip().tolist())
        before_books = df[book_id_col].nunique()
        df = df[~df[book_id_col].isin(excl)].copy()
        after_books = df[book_id_col].nunique()
        logger.info(f"✓ Applied exclusions: unique books {before_books} → {after_books} (excluded {before_books - after_books})")

    # Clean text (keep empty rows but warn)
    txt = df[text_col].astype("string")
    empty = txt.isna() | (txt.str.strip() == "")
    if empty.any():
        logger.warning(f"{empty.sum():,} rows have empty text in '{text_col}'. They will still be transformed; consider filtering upstream.")
        # Replace NA with empty string to keep alignment
        df.loc[txt.isna(), text_col] = ""

    logger.info(f"Books: {df[book_id_col].nunique():,} | Chapters: {df[chapter_id_col].nunique():,}")
    return SentenceDF(df=df, book_id_col=book_id_col, chapter_id_col=chapter_id_col, text_col=text_col)


# -------------------------
# Aggregation
# -------------------------
def aggregate_to_book_level(
    df: pd.DataFrame,
    probs: np.ndarray,
    *,
    book_id_col: str,
    topic_ids: List[int],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Average sentence probabilities per book, normalize, then emit long format [book_id, topic_id, prob]."""
    if probs.shape[0] != len(df):
        raise ValueError(f"probs rows ({probs.shape[0]}) != df rows ({len(df)})")

    logger.info("Aggregating to book level...")
    df_idx = df[[book_id_col]].copy()
    # group indices for each book
    groups = df_idx.groupby(book_id_col, sort=False).indices

    rows = []
    book_mass_sums = []  # pre-normalization probability mass per book
    for book_id, idxs in tqdm(groups.items(), total=len(groups), desc="Books", ncols=100):
        book_probs_raw = probs[list(idxs)]
        # Replace any NaN values with 0.0 BEFORE aggregation to ensure clean results
        if np.isnan(book_probs_raw).any():
            logger.warning(f"Book {book_id} has NaN probabilities, replacing with 0.0")
            book_probs_raw = np.nan_to_num(book_probs_raw, nan=0.0)
        # Now compute mean on clean data
        book_probs = book_probs_raw.mean(axis=0)
        # Normalize to a proper per-book distribution (aligns with tertile script)
        mass = float(np.sum(book_probs))
        book_mass_sums.append(mass)
        if mass > 0:
            book_probs = book_probs / mass
        else:
            logger.warning(f"Book {book_id}: total probability mass is 0; leaving vector as zeros")
        # Final safety check - should never be NaN after nan_to_num
        if np.isnan(book_probs).any():
            raise ValueError(f"Book {book_id} still has NaN after cleaning/normalization - this should not happen")
        for col_i, topic_id in enumerate(topic_ids):
            rows.append((book_id, int(topic_id), float(book_probs[col_i])))

    # Log pre-normalization mass stats
    if book_mass_sums:
        ms = pd.Series(book_mass_sums)
        logger.info(
            "Book-level pre-normalization prob mass (sum over topics): min=%.4f, median=%.4f, max=%.4f"
            % (ms.min(), ms.median(), ms.max())
        )
    else:
        logger.warning("No book mass sums were collected; check grouping logic")

    out = pd.DataFrame(rows, columns=["book_id", "topic_id", "prob"])
    logger.info(f"✓ book_topic_probs: {out.shape[0]:,} rows ({out['book_id'].nunique():,} books × {out['topic_id'].nunique():,} topics)")
    return out


def aggregate_to_chapter_level(
    df: pd.DataFrame,
    probs: np.ndarray,
    *,
    book_id_col: str,
    chapter_id_col: str,
    topic_ids: List[int],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Average sentence probabilities per chapter, normalize, then emit long format [book_id, chapter_id, topic_id, prob]."""
    if probs.shape[0] != len(df):
        raise ValueError(f"probs rows ({probs.shape[0]}) != df rows ({len(df)})")

    logger.info("Aggregating to chapter level...")
    keys = df[[book_id_col, chapter_id_col]].copy()
    groups = keys.groupby([book_id_col, chapter_id_col], sort=False).indices

    rows = []
    chapter_mass_sums = []  # pre-normalization probability mass per chapter
    for (book_id, chapter_id), idxs in tqdm(groups.items(), total=len(groups), desc="Chapters", ncols=100):
        chap_probs_raw = probs[list(idxs)]
        # Replace any NaN values with 0.0 BEFORE aggregation to ensure clean results
        if np.isnan(chap_probs_raw).any():
            logger.warning(f"Book {book_id}, Chapter {chapter_id} has NaN probabilities, replacing with 0.0")
            chap_probs_raw = np.nan_to_num(chap_probs_raw, nan=0.0)
        # Now compute mean on clean data
        chap_probs = chap_probs_raw.mean(axis=0)
        # Normalize to a proper per-chapter distribution (aligns with tertile script)
        mass = float(np.sum(chap_probs))
        chapter_mass_sums.append(mass)
        if mass > 0:
            chap_probs = chap_probs / mass
        else:
            logger.warning(f"Book {book_id}, Chapter {chapter_id}: total probability mass is 0; leaving vector as zeros")
        # Final safety check - should never be NaN after nan_to_num
        if np.isnan(chap_probs).any():
            raise ValueError(f"Book {book_id}, Chapter {chapter_id} still has NaN after cleaning/normalization - this should not happen")
        for col_i, topic_id in enumerate(topic_ids):
            rows.append((book_id, int(chapter_id), int(topic_id), float(chap_probs[col_i])))

    # Log pre-normalization mass stats
    if chapter_mass_sums:
        ms = pd.Series(chapter_mass_sums)
        logger.info(
            "Chapter-level pre-normalization prob mass (sum over topics): min=%.4f, median=%.4f, max=%.4f"
            % (ms.min(), ms.median(), ms.max())
        )
    else:
        logger.warning("No chapter mass sums were collected; check grouping logic")

    out = pd.DataFrame(rows, columns=["book_id", "chapter_id", "topic_id", "prob"])
    logger.info(f"✓ chapter_topic_probs: {out.shape[0]:,} rows ({out[['book_id','chapter_id']].drop_duplicates().shape[0]:,} book-chapters × {out['topic_id'].nunique():,} topics)")
    return out


# -------------------------
# Main
# -------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate book/chapter-level BERTopic probability tables (unified, Goodreads-first).")
    parser.add_argument("--sentence-df", type=Path, required=True, help="Path to sentence_df_with_topics.parquet (or .csv)")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to BERTopic model (directory or .pkl)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--logs-dir", type=Path, default=None, help="Directory for logs (default: <output-dir>/logs)")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Cache directory (default: <output-dir>/cache)")
    parser.add_argument("--no-cache", action="store_true", help="Disable loading/saving cache for transform outputs")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size for BERTopic.transform (0 = all at once)")
    parser.add_argument("--text-col", type=str, default=None, help="Text column in sentence_df (auto-detect if omitted)")

    # ID strategy
    parser.add_argument("--book-id-source", choices=["goodreads", "existing", "author_title"], default="goodreads",
                        help="Which identifier to output as book_id. Use 'goodreads' for reliable merges.")
    parser.add_argument("--goodreads-id-col", type=str, default="goodreads_book_id",
                        help="Column in sentence_df containing Goodreads id (e.g., 'ID'). Used when --book-id-source=goodreads.")
    parser.add_argument("--author-col", type=str, default="Author", help="Author column for author_title fallback")
    parser.add_argument("--title-col", type=str, default="Book Title", help="Title column for author_title fallback")

    # Cohort control
    parser.add_argument("--exclude-book-ids", type=str, default=None,
                        help="Comma-separated IDs or path to CSV with column 'book_id' to exclude from processing.")

    # Output options
    parser.add_argument("--write-csv", action="store_true", help="Also write CSV versions of outputs (in addition to Parquet).")
    parser.add_argument("--write-sentence-topics", action="store_true",
                        help="Write sentence-level topic assignments/probability max for debugging (can be large).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.logs_dir or (args.output_dir / "logs")
    cache_dir = args.cache_dir or (args.output_dir / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(logs_dir)
    logger.info("=== generate_topic_probabilities_final.py ===")
    logger.info(f"sentence_df: {args.sentence_df}")
    logger.info(f"model_path : {args.model_path}")
    logger.info(f"output_dir : {args.output_dir}")
    logger.info(f"book_id_source={args.book_id_source} | goodreads_id_col={args.goodreads_id_col}")
    logger.info(f"exclude_book_ids={args.exclude_book_ids}")

    exclude_ids = load_excluded_ids(args.exclude_book_ids)
    if exclude_ids:
        logger.info(f"Loaded {len(exclude_ids)} excluded IDs")

    # Load inputs
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
    texts = df[sent.text_col].astype(str).tolist()

    # Load model
    topic_model = load_bertopic_model(args.model_path, logger=logger)

    # Transform (with caching)
    cache_key = make_cache_key(args.sentence_df, args.model_path, len(texts), extra=f"bs={args.batch_size}|id={args.book_id_source}|gid={args.goodreads_id_col}")
    cache_file = cache_dir / f"transform_{cache_key}.pkl"

    if not args.no_cache and cache_file.exists():
        logger.info(f"✓ Loading cached transform outputs: {cache_file.name}")
        with cache_file.open("rb") as f:
            payload = pickle.load(f)
        topics = payload["topics"]
        probs = payload["probs"]
    else:
        topics, probs = transform_texts_in_batches(topic_model, texts, args.batch_size, logger)
        if not args.no_cache:
            with cache_file.open("wb") as f:
                pickle.dump({"topics": topics, "probs": probs}, f)
            logger.info(f"✓ Wrote cache: {cache_file.name}")

    probs = probs.astype(float)
    n_topics = probs.shape[1]
    logger.info(f"Probability matrix: {probs.shape[0]:,} docs × {n_topics:,} topics")
    
    # Check for NaN values in probability matrix and fix them immediately
    nan_count = np.isnan(probs).sum()
    total_values = probs.size
    if nan_count > 0:
        nan_pct = (nan_count / total_values) * 100
        logger.warning(f"⚠ Found {nan_count:,} NaN values in probability matrix ({nan_pct:.2f}%)")
        logger.warning(f"  Replacing NaN values with 0.0 to ensure clean aggregation")
        # Log detailed diagnostics about where NaNs appear
        log_nan_diagnostics(
            probs=probs,
            df=df,
            text_col=sent.text_col,
            book_id_col=sent.book_id_col,
            chapter_id_col=sent.chapter_id_col,
            logger=logger,
            sample_size=5,
        )
        # Replace NaN with 0.0 immediately - this ensures no NaNs propagate to aggregation
        probs = np.nan_to_num(probs, nan=0.0)
        logger.info(f"✓ Cleaned probability matrix - all NaN values replaced with 0.0")
    else:
        logger.info("✓ No NaN values found in probability matrix")

    topic_ids = infer_topic_ids_for_prob_columns(topic_model, n_topics, logger=logger)
    logger.info(f"Topic id labeling: {topic_ids[:10]}{'...' if len(topic_ids)>10 else ''}")

    # Aggregate
    book_topic_probs = aggregate_to_book_level(df, probs, book_id_col=sent.book_id_col, topic_ids=topic_ids, logger=logger)
    chapter_topic_probs = aggregate_to_chapter_level(df, probs, book_id_col=sent.book_id_col, chapter_id_col=sent.chapter_id_col, topic_ids=topic_ids, logger=logger)
    
    # Optional: quick post-write normalization check (in-memory)
    book_sums = book_topic_probs.groupby("book_id")["prob"].sum()
    logger.info(
        "Post-normalization check (book prob sums): min=%.4f, median=%.4f, max=%.4f"
        % (book_sums.min(), book_sums.median(), book_sums.max())
    )
    chapter_sums = chapter_topic_probs.groupby(["book_id", "chapter_id"])["prob"].sum()
    logger.info(
        "Post-normalization check (chapter prob sums): min=%.4f, median=%.4f, max=%.4f"
        % (chapter_sums.min(), chapter_sums.median(), chapter_sums.max())
    )
    
    # Final validation: ensure NO NaNs in output
    logger.info("\n" + "="*80)
    logger.info("Final validation: checking for NaN values")
    logger.info("="*80)
    
    book_nan_count = book_topic_probs["prob"].isna().sum()
    chapter_nan_count = chapter_topic_probs["prob"].isna().sum()
    
    if book_nan_count > 0:
        logger.error(f"❌ ERROR: Found {book_nan_count:,} NaN values in book_topic_probs!")
        logger.error("  This should not happen - all NaNs should have been replaced with 0.0")
        raise ValueError(f"book_topic_probs contains {book_nan_count:,} NaN values")
    else:
        logger.info(f"✓ book_topic_probs: No NaN values ({len(book_topic_probs):,} rows)")
    
    if chapter_nan_count > 0:
        logger.error(f"❌ ERROR: Found {chapter_nan_count:,} NaN values in chapter_topic_probs!")
        logger.error("  This should not happen - all NaNs should have been replaced with 0.0")
        raise ValueError(f"chapter_topic_probs contains {chapter_nan_count:,} NaN values")
    else:
        logger.info(f"✓ chapter_topic_probs: No NaN values ({len(chapter_topic_probs):,} rows)")

    # Write outputs to topic_probabilities subdirectory
    topic_probs_dir = args.output_dir / "topic_probabilities"
    topic_probs_dir.mkdir(parents=True, exist_ok=True)
    book_out = topic_probs_dir / "book_topic_probs.parquet"
    chap_out = topic_probs_dir / "chapter_topic_probs.parquet"
    book_topic_probs.to_parquet(book_out, index=False)
    chapter_topic_probs.to_parquet(chap_out, index=False)
    logger.info(f"✓ Wrote: {book_out}")
    logger.info(f"✓ Wrote: {chap_out}")

    if args.write_csv:
        book_csv = topic_probs_dir / "book_topic_probs.csv"
        chap_csv = topic_probs_dir / "chapter_topic_probs.csv"
        book_topic_probs.to_csv(book_csv, index=False)
        chapter_topic_probs.to_csv(chap_csv, index=False)
        logger.info(f"✓ Wrote: {book_csv}")
        logger.info(f"✓ Wrote: {chap_csv}")

    if args.write_sentence_topics:
        # lightweight debug output: predicted topic + max prob
        mx = probs.max(axis=1) if probs.size else np.array([])
        sent_out = df[[sent.book_id_col, sent.chapter_id_col]].copy()
        sent_out = sent_out.rename(columns={sent.book_id_col: "book_id", sent.chapter_id_col: "chapter_id"})
        sent_out["topic_pred"] = topics
        sent_out["topic_prob_max"] = mx
        dbg_out = topic_probs_dir / "sentence_topic_debug.parquet"
        sent_out.to_parquet(dbg_out, index=False)
        logger.info(f"✓ Wrote: {dbg_out}")

    logger.info("DONE.")


if __name__ == "__main__":
    main()
