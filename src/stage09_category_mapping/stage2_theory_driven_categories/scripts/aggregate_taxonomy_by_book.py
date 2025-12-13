"""Stage 09: Aggregate taxonomy mappings to book-level category proportions.

This module loads taxonomy mappings (from zeroshot_taxonomy_openrouter) and
sentence-level topic assignments, then computes per-book proportions of
each taxonomy category.

Supports loading taxonomy mappings from:
1. JSON file (taxonomy_mappings_*.json)
2. BERTopic model's embedded topic_metadata_ attribute (recommended)
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from bertopic import BERTopic


def load_taxonomy_mapping_from_json(path: Path) -> pd.DataFrame:
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


def load_taxonomy_mapping_from_model(model_path: Path) -> pd.DataFrame:
    """
    Load taxonomy mappings from BERTopic model's topic_metadata_ attribute.
    
    This follows the recommendation from MODEL_COMPARISON_REPORT.md to use
    models with embedded taxonomy metadata (e.g., model_1_with_llm_labels_and_metadata_disambiguated.pkl).

    Parameters
    ----------
    model_path:
        Path to BERTopic model (.pkl file or directory).

    Returns
    -------
    DataFrame with columns: topic, main_category_id, secondary_category_id,
    confidence, is_noise.
    
    Raises
    ------
    ValueError:
        If model doesn't have topic_metadata_ attribute or it's empty.
    """
    # Load model (handle both pickle wrapper and native format)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    if model_path.suffix == ".pkl":
        with open(model_path, "rb") as f:
            loaded_obj = pickle.load(f)
        
        # Check if it's a RetrainableBERTopicModel wrapper
        if hasattr(loaded_obj, "trained_topic_model") and loaded_obj.trained_topic_model is not None:
            model = loaded_obj.trained_topic_model
        elif isinstance(loaded_obj, BERTopic):
            model = loaded_obj
        else:
            model = BERTopic.load(str(model_path))
    else:
        model = BERTopic.load(str(model_path))
    
    # Extract taxonomy metadata
    if not hasattr(model, "topic_metadata_") or not model.topic_metadata_:
        raise ValueError(
            f"Model at {model_path} does not have topic_metadata_ attribute. "
            "Use a model with embedded taxonomy mappings (e.g., model_1_with_llm_labels_and_metadata_disambiguated.pkl)"
        )
    
    metadata = model.topic_metadata_
    
    rows = []
    for topic_id, topic_data in metadata.items():
        # Handle both int and str keys
        tid = int(topic_id) if isinstance(topic_id, str) else topic_id
        
        rows.append(
            {
                "topic": tid,
                "main_category_id": topic_data.get("main_category_id"),
                "secondary_category_id": topic_data.get("secondary_category_id"),
                "confidence": topic_data.get("confidence", None),
                "is_noise": bool(topic_data.get("is_noise", False)),
            }
        )
    
    return pd.DataFrame(rows)


def load_taxonomy_mapping(path: Path) -> pd.DataFrame:
    """
    Load taxonomy mappings from JSON file or BERTopic model.
    
    Automatically detects the source type:
    - If path ends with .json, loads from JSON file
    - If path ends with .pkl or is a directory, loads from model's topic_metadata_

    Parameters
    ----------
    path:
        Path to taxonomy mappings JSON file or BERTopic model.

    Returns
    -------
    DataFrame with columns: topic, main_category_id, secondary_category_id,
    confidence, is_noise.
    """
    if path.suffix == ".json":
        return load_taxonomy_mapping_from_json(path)
    else:
        return load_taxonomy_mapping_from_model(path)


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
        Path to taxonomy_mappings_*.json from Stage 2, or BERTopic model with embedded taxonomy metadata (e.g., model_1_with_llm_labels_and_metadata_disambiguated.pkl).
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
        help="Path to taxonomy_mappings_*.json from Stage 2, or BERTopic model with embedded taxonomy metadata (e.g., model_1_with_llm_labels_and_metadata_disambiguated.pkl)",
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

