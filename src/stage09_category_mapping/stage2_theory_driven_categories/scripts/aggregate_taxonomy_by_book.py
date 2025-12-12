"""Stage 09: Aggregate taxonomy mappings to book-level category proportions.

This module loads taxonomy mappings (from zeroshot_taxonomy_openrouter) and
sentence-level topic assignments, then computes per-book proportions of
each taxonomy category.
"""

import json
from pathlib import Path
from typing import Dict

import pandas as pd


def load_taxonomy_mapping(path: Path) -> pd.DataFrame:
    """
    Load taxonomy_mappings_*.json and return a small DataFrame
    with topic → main/secondary category ids.

    Parameters
    ----------
    path:
        Path to taxonomy mappings JSON file.

    Returns
    -------
    DataFrame with columns: topic, main_category_id, secondary_category_id,
    confidence, is_noise.
    """
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Dict] = json.load(f)

    rows = []
    for k, v in data.items():
        rows.append(
            {
                "topic": int(k),
                "main_category_id": v.get("main_category_id"),
                "secondary_category_id": v.get("secondary_category_id"),
                "confidence": v.get("confidence", None),
                "is_noise": bool(v.get("is_noise", False)),
            }
        )
    return pd.DataFrame(rows)


def build_book_category_proportions(
    sentence_df_path: Path,
    taxonomy_mapping_path: Path,
    min_sentences_per_book: int = 50,
) -> pd.DataFrame:
    """
    Join sentence-level topic assignments with taxonomy mappings
    and compute per-book proportions of main_category_id.

    Parameters
    ----------
    sentence_df_path:
        Path to sentence-level parquet file with at least:
        - 'book_id'
        - 'rating_class' (or similar)
        - 'topic' (BERTopic topic id)
    taxonomy_mapping_path:
        Path to taxonomy_mappings_*.json from Stage 2.
    min_sentences_per_book:
        Minimum number of sentences per book to include in analysis.

    Returns
    -------
    DataFrame with columns:
        - book_id
        - rating_class
        - main_category_id
        - n_sentences (count for this book + category)
        - total_sentences (total sentences for this book)
        - prop (proportion of sentences in this category for this book)
    """
    df = pd.read_parquet(sentence_df_path)

    # Drop outlier / unlabeled topic -1
    df = df[df["topic"] >= 0].copy()

    mapping_df = load_taxonomy_mapping(taxonomy_mapping_path)

    df = df.merge(mapping_df, on="topic", how="left")

    # Optional: drop sentences where we have no taxonomy mapping
    df = df[~df["main_category_id"].isna()].copy()

    # Optional: keep only books with enough sentences
    book_counts = df.groupby("book_id")["topic"].size()
    keep_books = book_counts[book_counts >= min_sentences_per_book].index
    df = df[df["book_id"].isin(keep_books)].copy()

    # Aggregate: count sentences per (book, rating_class, main_category_id)
    grouped = (
        df.groupby(["book_id", "rating_class", "main_category_id"])
        .size()
        .reset_index(name="n_sentences")
    )

    # Convert to proportions within each book
    grouped["total_sentences"] = grouped.groupby("book_id")["n_sentences"].transform("sum")
    grouped["prop"] = grouped["n_sentences"] / grouped["total_sentences"]

    return grouped


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate taxonomy mappings to book-level category proportions."
    )
    parser.add_argument(
        "--sentences",
        type=Path,
        required=True,
        help="Path to sentence_df_with_topics.parquet",
    )
    parser.add_argument(
        "--taxonomy-mapping",
        type=Path,
        required=True,
        help="Path to taxonomy_mappings_*.json from Stage 2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to save book-level category proportions",
    )
    parser.add_argument(
        "--min-sentences-per-book",
        type=int,
        default=50,
        help="Minimum sentences per book to include (default: 50)",
    )

    args = parser.parse_args()

    book_cat = build_book_category_proportions(
        sentence_df_path=args.sentences,
        taxonomy_mapping_path=args.taxonomy_mapping,
        min_sentences_per_book=args.min_sentences_per_book,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    book_cat.to_parquet(args.output, index=False)
    print(f"Saved book-level category proportions to {args.output}")

