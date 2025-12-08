"""Topic quality analysis and noisy topic detection for BERTopic models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

from src.stage06_topic_exploration.explore_retrained_model import (
    LOGGER,
    extract_all_topics,
    stage_timer,
)

# Use the same logger as stage06_topic_exploration for consistency
logger = LOGGER


def get_topic_distribution(topic_model: BERTopic, min_size: int = 30) -> pd.DataFrame:
    """
    Get topic counts and a 'keep_by_size' flag.
    
    Args:
        topic_model: BERTopic model instance
        min_size: Minimum number of documents per topic to be considered valid
        
    Returns:
        DataFrame with columns: Topic, Count, Name, Representation, keep_by_size
    """
    info = topic_model.get_topic_info()
    # Exclude outlier topic -1
    info = info[info["Topic"] != -1].copy()
    info["keep_by_size"] = info["Count"] >= min_size
    return info


def get_pos_representation_stats(
    topic_model: BERTopic, top_k: int = 10
) -> pd.DataFrame:
    """
    Extract POS representation topics and basic stats.
    
    Args:
        topic_model: BERTopic model instance
        top_k: Number of top words per topic to consider
        
    Returns:
        DataFrame with columns: Topic, n_pos_words, pos_words
    """
    all_topics = extract_all_topics(topic_model, top_k=top_k)
    pos_topics = all_topics.get("POS", {})

    rows = []
    for topic_id, word_list in pos_topics.items():
        n_words = len(word_list)
        rows.append(
            {
                "Topic": topic_id,
                "n_pos_words": n_words,
                "pos_words": [w["word"] for w in word_list],
            }
        )

    if not rows:
        logger.warning("No POS topics found in extracted aspects.")
        return pd.DataFrame(columns=["Topic", "n_pos_words", "pos_words"])

    df = pd.DataFrame(rows).sort_values("Topic").reset_index(drop=True)
    return df


def compute_pos_coherence_per_topic(
    topic_model: BERTopic,
    docs_tokens: list[list[str]],
    dictionary: Dictionary,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Compute c_v coherence per topic, using POS representation words.
    
    We convert words to ids to align with Stage 06's dictionary usage.
    
    Args:
        topic_model: BERTopic model instance
        docs_tokens: List of tokenized documents
        dictionary: Gensim dictionary for coherence computation
        top_k: Number of top words per topic to consider
        
    Returns:
        DataFrame with columns: Topic, coherence_c_v_pos
    """
    all_topics = extract_all_topics(topic_model, top_k=top_k)
    pos_topics = all_topics.get("POS", {})

    topic_ids = []
    topic_ids_lists = []

    for topic_id, word_dicts in pos_topics.items():
        words = [w["word"] for w in word_dicts]
        ids = [dictionary.token2id[w] for w in words if w in dictionary.token2id]
        if not ids:
            continue
        topic_ids.append(topic_id)
        topic_ids_lists.append(ids)

    if not topic_ids_lists:
        logger.warning("No POS topics with valid dictionary tokens for coherence.")
        return pd.DataFrame(columns=["Topic", "coherence_c_v_pos"])

    with stage_timer("Computing per-topic POS coherence (c_v)"):
        cm = CoherenceModel(
            topics=topic_ids_lists,
            texts=docs_tokens,
            dictionary=dictionary,
            coherence="c_v",
        )
        scores = cm.get_coherence_per_topic()

    df = pd.DataFrame(
        {
            "Topic": topic_ids,
            "coherence_c_v_pos": scores,
        }
    ).sort_values("Topic").reset_index(drop=True)

    return df


def build_topic_quality_table(
    topic_model: BERTopic,
    docs_tokens: list[list[str]],
    dictionary: Dictionary,
    min_size: int = 30,
    min_pos_words: int = 3,
    min_pos_coherence: float = 0.0,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Combine size, POS stats, and POS coherence; flag candidate noisy topics.
    
    This function does NOT remove topics from the model; it only flags them
    for manual inspection.
    
    Args:
        topic_model: BERTopic model instance
        docs_tokens: List of tokenized documents
        dictionary: Gensim dictionary for coherence computation
        min_size: Minimum number of documents per topic
        min_pos_words: Minimum number of POS words per topic
        min_pos_coherence: Minimum per-topic POS coherence threshold
        top_k: Number of top words per topic to consider
        
    Returns:
        DataFrame with topic quality metrics and noise candidate flags
    """
    with stage_timer("Building topic quality table"):
        topic_info = get_topic_distribution(topic_model, min_size=min_size)
        pos_stats = get_pos_representation_stats(topic_model, top_k=top_k)
        pos_coh = compute_pos_coherence_per_topic(
            topic_model,
            docs_tokens=docs_tokens,
            dictionary=dictionary,
            top_k=top_k,
        )

        df = (
            topic_info[["Topic", "Count", "Name", "Representation"]]
            .merge(
                pos_stats[["Topic", "n_pos_words", "pos_words"]],
                on="Topic",
                how="left",
            )
            .merge(
                pos_coh[["Topic", "coherence_c_v_pos"]],
                on="Topic",
                how="left",
            )
        )

        # Flag conditions
        df["flag_small"] = df["Count"] < min_size
        df["flag_few_pos"] = df["n_pos_words"].fillna(0) < min_pos_words
        df["flag_low_coh"] = df["coherence_c_v_pos"].fillna(-1.0) < min_pos_coherence

        # Aggregate into noise_candidate + reason
        def _noise_reason(row) -> str:
            reasons = []
            if row["flag_small"]:
                reasons.append(f"small<{min_size}")
            if row["flag_few_pos"]:
                reasons.append(f"few_pos<{min_pos_words}")
            if row["flag_low_coh"]:
                reasons.append(f"low_coh<{min_pos_coherence:.2f}")
            return ";".join(reasons)

        df["noise_reason"] = df.apply(_noise_reason, axis=1)
        df["noise_candidate"] = df["noise_reason"].str.len() > 0

        # Label for manual inspection: prepend reason to topic name
        def _inspection_label(row) -> str:
            base_name = str(row.get("Name", "") or "").strip()
            reasons = row["noise_reason"]
            if not reasons:
                return base_name
            return f"[NOISE_CANDIDATE:{reasons}] {base_name}"

        df["inspection_label"] = df.apply(_inspection_label, axis=1)

        # Sort for easier EDA
        df.sort_values(
            ["noise_candidate", "coherence_c_v_pos", "Count"],
            ascending=[False, True, True],
            inplace=True,
        )
        df.reset_index(drop=True, inplace=True)

    return df


def apply_noise_labels_to_model(
    topic_model: BERTopic,
    quality_df: pd.DataFrame,
    only_noise_candidates: bool = True,
) -> dict[int, str]:
    """
    Build a label dictionary for topics based on quality analysis.
    
    Args:
        topic_model: BERTopic model instance
        quality_df: DataFrame from build_topic_quality_table
        only_noise_candidates: If True, only label noisy topics; if False, label all
        
    Returns:
        Dictionary mapping topic_id -> label string
    """
    if only_noise_candidates:
        candidates_df = quality_df[quality_df["noise_candidate"]]
    else:
        candidates_df = quality_df

    labels = {
        int(row.Topic): str(row.inspection_label)
        for row in candidates_df.itertuples(index=False)
    }

    logger.info(
        "Prepared %d labels for %s",
        len(labels),
        "noise candidates" if only_noise_candidates else "all topics",
    )

    return labels

