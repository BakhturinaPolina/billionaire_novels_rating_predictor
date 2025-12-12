"""Explore hierarchical topics structure for Stage 1 natural clusters analysis.

This script:
1. Loads the matched-only dataframe with topic assignments from Step 3
2. Extracts sentences from the dataframe
3. Loads the BERTopic model (trained on all 105 books)
4. Builds hierarchical structure on matched docs only
5. Visualizes dendrogram
6. Prints text tree for inspection
7. Helps choose target number of meta-topics (typically 40-80)

Key principle:
- Model was trained on all 105 books (100% of sentences)
- This script builds hierarchy using only the 92 matched books (~612k sentences, ~90%)
- The unmatched 10% of sentences helped shape the topic space during training,
  but are excluded from hierarchy construction to ensure consistency with analysis.
"""

from __future__ import annotations

import argparse
import copy
import logging
import pickle
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from bertopic import BERTopic

from src.common.logging import setup_logging
from src.stage06_topic_exploration.explore_retrained_model import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    load_native_bertopic_model,
)


@dataclass(frozen=True)
class TopicIdMapping:
    """Mapping between original and remapped topic IDs."""
    old_to_new: dict[int, int]
    new_to_old: dict[int, int]


def load_dataframe(input_path: Path, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Load the matched-only dataframe with topic assignments from Step 3.
    
    Args:
        input_path: Path to sentence_df_with_topics.parquet
        logger: Logger instance
        
    Returns:
        DataFrame with matched sentences and topic assignments
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Loading dataframe from: {input_path}")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_parquet(input_path)
    
    logger.info(f"✓ Loaded {len(df):,} sentences from {df['book_id'].nunique()} books")
    logger.info(f"  Columns: {', '.join(df.columns.tolist())}")
    
    # Verify required columns
    required_cols = ["text", "book_id", "topic"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Log topic distribution
    topic_counts = df["topic"].value_counts()
    logger.info(f"  Unique topics: {df['topic'].nunique()}")
    logger.info(f"  Outlier topic (-1) count: {topic_counts.get(-1, 0):,}")
    logger.info(f"  Non-outlier topics: {df['topic'].nunique() - (1 if -1 in topic_counts else 0)}")
    
    return df


def extract_sentences(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> list[str]:
    """Extract sentences from dataframe.
    
    Args:
        df: DataFrame with 'text' column
        logger: Logger instance
        
    Returns:
        List of sentence strings
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Extracting sentences from dataframe...")
    
    # Check for missing values
    missing_count = df["text"].isna().sum()
    if missing_count > 0:
        logger.warning(f"Found {missing_count:,} missing sentences, will be handled by BERTopic")
    
    docs = df["text"].astype(str).tolist()
    
    logger.info(f"✓ Extracted {len(docs):,} sentences")
    
    return docs


def load_model(
    base_dir: Path,
    embedding_model: str,
    model_suffix: str,
    stage_subfolder: str | None = None,
    logger: Optional[logging.Logger] = None,
) -> BERTopic:
    """Load BERTopic model.
    
    Args:
        base_dir: Base directory for models
        embedding_model: Embedding model name
        model_suffix: Model suffix (e.g., "_with_llm_labels")
        stage_subfolder: Optional stage subfolder (e.g., "stage08_llm_labeling")
        logger: Logger instance
        
    Returns:
        Loaded BERTopic model
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Loading BERTopic model (suffix: {model_suffix}, stage: {stage_subfolder or 'base'})...")
    
    # Construct model path with optional stage subfolder
    if stage_subfolder:
        model_dir = base_dir / embedding_model / stage_subfolder / f"model_1{model_suffix}"
    else:
        model_dir = base_dir / embedding_model / f"model_1{model_suffix}"
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    logger.info(f"Loading model from: {model_dir}")
    
    # Try loading from directory first, then fall back to .pkl file if needed
    try:
        topic_model = BERTopic.load(str(model_dir))
    except (KeyError, pickle.UnpicklingError, Exception) as e:
        logger.warning(f"Failed to load from directory ({e}), trying .pkl file...")
        # Try .pkl file in the same directory
        pkl_path = model_dir.parent / f"model_1{model_suffix}.pkl"
        if pkl_path.exists():
            logger.info(f"Loading from .pkl file: {pkl_path}")
            loaded_obj = pickle.load(open(pkl_path, "rb"))
            
            # Check if it's a RetrainableBERTopicModel wrapper
            if hasattr(loaded_obj, "trained_topic_model") and loaded_obj.trained_topic_model is not None:
                logger.info("  Extracted BERTopic model from RetrainableBERTopicModel wrapper")
                topic_model = loaded_obj.trained_topic_model
            elif isinstance(loaded_obj, BERTopic):
                topic_model = loaded_obj
            else:
                # Try BERTopic.load() as fallback
                topic_model = BERTopic.load(str(pkl_path))
        else:
            raise FileNotFoundError(
                f"Could not load model from directory or .pkl file. "
                f"Tried: {model_dir} and {pkl_path}"
            ) from e
    
    # Log model info
    if hasattr(topic_model, "topic_representations_"):
        topic_ids = [
            tid for tid in topic_model.topic_representations_.keys() 
            if tid != -1
        ]
        logger.info(f"✓ Model loaded with {len(topic_ids)} topics (excluding outlier -1)")
    
    if hasattr(topic_model, "custom_labels_") and topic_model.custom_labels_:
        logger.info(f"✓ Model has {len(topic_model.custom_labels_)} custom labels")
    
    return topic_model


def custom_labels_as_dict(topic_model: BERTopic) -> dict[int, str]:
    """Normalize custom_labels_ to a dict, handling both list and dict formats.
    
    BERTopic stores custom_labels_ as a list[str] (not a dict), but some code
    may treat it as a dict. This helper normalizes it to a dict format.
    
    Args:
        topic_model: BERTopic model
        
    Returns:
        Dictionary mapping topic_id to label string
    """
    labels = getattr(topic_model, "custom_labels_", None)
    if not labels:
        return {}
    
    if isinstance(labels, dict):
        return {int(k): v for k, v in labels.items() if v is not None}
    
    # BERTopic stores custom_labels_ as list[str]
    # Need to get topic IDs from topics_ or topic_representations_
    if hasattr(topic_model, "topics_") and topic_model.topics_ is not None:
        unique_topics = sorted(set(topic_id for topic_id in topic_model.topics_ if topic_id != -1))
    elif hasattr(topic_model, "topic_representations_"):
        unique_topics = sorted([tid for tid in topic_model.topic_representations_.keys() if tid != -1])
    else:
        return {}
    
    # Map list indices to topic IDs
    # Note: This assumes labels list is in the same order as sorted unique topics
    # This may not always be true, but it's the best we can do without more context
    result = {}
    for i, topic_id in enumerate(unique_topics):
        if i < len(labels) and labels[i] is not None:
            result[topic_id] = str(labels[i])
    
    return result


def get_llm_label_map(topic_model: BERTopic) -> dict[int, str]:
    """Robustly fetch LLM/custom labels as {topic_id: label} using get_topic_info.
    
    Args:
        topic_model: BERTopic model
        
    Returns:
        Dictionary mapping topic_id to label string
    """
    try:
        info = topic_model.get_topic_info()
        # BERTopic stores custom labels for visualizations via set_topic_labels
        custom_cols = [c for c in info.columns if c.lower().startswith("custom")]
        if custom_cols:
            col = custom_cols[0]
            out = {}
            for tid, lbl in zip(info["Topic"].tolist(), info[col].tolist()):
                if int(tid) == -1:
                    continue
                if lbl is None or (isinstance(lbl, float) and np.isnan(lbl)):
                    continue
                out[int(tid)] = str(lbl)
            return out
    except (AttributeError, TypeError, Exception):
        pass
    
    # Fallback: best effort from custom_labels_ list/dict
    return custom_labels_as_dict(topic_model)


def identify_noise_topics(
    topic_model: BERTopic,
    logger: Optional[logging.Logger] = None,
) -> set[int]:
    """Identify noise topics from model metadata, labels, and word patterns.
    
    Noise topics are identified by (in priority order):
    1. topic_metadata_[tid]["is_noise"] == True (if metadata is available)
    2. Labels starting with [NOISE_CANDIDATE: or [NOISE:
    3. Word patterns indicating noise (contractions, common stopwords, etc.)
    
    Args:
        topic_model: Loaded BERTopic model
        logger: Logger instance
        
    Returns:
        Set of topic IDs that are noise topics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    noise_topics: set[int] = set()
    
    # (A) Prefer metadata if present (source of truth from labeling pipeline)
    meta = getattr(topic_model, "topic_metadata_", None)
    if isinstance(meta, dict):
        for tid, md in meta.items():
            try:
                tid_int = int(tid)
            except Exception:
                continue
            if tid_int == -1:
                continue
            if isinstance(md, dict) and md.get("is_noise") is True:
                noise_topics.add(tid_int)
            # Also honor explicit [NOISE...] label if stored in metadata
            label = (md or {}).get("label")
            if isinstance(label, str) and (label.startswith("[NOISE_CANDIDATE:") or label.startswith("[NOISE:")):
                noise_topics.add(tid_int)
    
    # (B) Fall back to label prefixes (use robust get_llm_label_map instead of custom_labels_as_dict)
    labels = get_llm_label_map(topic_model)
    for tid, lab in labels.items():
        if isinstance(lab, str) and (lab.startswith("[NOISE_CANDIDATE:") or lab.startswith("[NOISE:")):
            noise_topics.add(tid)
    
    # (C) Word pattern heuristic as final fallback (only for topics not already identified)
    noise_indicators = {
        # Contractions
        "didn", "wasn", "couldn", "doesn", "hadn", "wouldn", "isn", "aren", "won",
        "didna", "wasna", "couldna", "doesna", "hadna", "wouldna", "isna", "aren",
        # Common stopwords that dominate noise topics
        "therea", "youa", "hea", "shea", "wea", "theya", "ia", "ita",
        # Generic words
        "matter", "sense", "true", "thing", "things", "stuff",
    }
    
    try:
        if hasattr(topic_model, "topic_representations_"):
            for topic_id, words_list in topic_model.topic_representations_.items():
                if topic_id == -1:  # Skip outlier
                    continue
                
                if topic_id in noise_topics:  # Already identified
                    continue
                
                if not isinstance(words_list, list) or len(words_list) == 0:
                    continue
                
                # Get top words (first 5-10 words are most important)
                top_words = [word.lower().strip() for word, _ in words_list[:10]]
                
                # Filter out empty strings and very short words
                top_words = [w for w in top_words if len(w) > 1]
                
                # Check for empty or minimal content
                if len(top_words) == 0 or all(len(w) <= 2 for w in top_words[:3]):
                    noise_topics.add(topic_id)
                    logger.debug(
                        f"  Topic {topic_id} identified as noise: empty or minimal content"
                    )
                    continue
                
                # Check for topics with mostly underscores or special characters
                if any(all(c in "_" for c in w) for w in top_words[:3]):
                    noise_topics.add(topic_id)
                    logger.debug(
                        f"  Topic {topic_id} identified as noise: contains mostly underscores"
                    )
                    continue
                
                # Count how many top words are noise indicators
                noise_word_count = sum(1 for word in top_words if word in noise_indicators)
                
                # If majority of top words are noise indicators, mark as noise
                if noise_word_count >= 3:  # At least 3 out of top 10 words are noise
                    noise_topics.add(topic_id)
                    logger.debug(
                        f"  Topic {topic_id} identified as noise by word pattern: "
                        f"{noise_word_count}/{len(top_words)} top words are noise indicators"
                    )
                # Also check if topic name/representation is dominated by contractions
                elif len(top_words) >= 3:
                    # Check if first 3 words are all contractions
                    if all(word in noise_indicators for word in top_words[:3]):
                        noise_topics.add(topic_id)
                        logger.debug(
                            f"  Topic {topic_id} identified as noise: first 3 words are all noise indicators"
                        )
    
    except Exception as e:
        logger.warning(f"Could not identify noise topics from word patterns: {e}")
    
    if noise_topics:
        logger.info(f"  Identified {len(noise_topics)} noise topics to exclude: {sorted(noise_topics)}")
        # Log some examples
        if hasattr(topic_model, "topic_representations_"):
            examples = sorted(noise_topics)[:5]
            logger.info("  Example noise topics:")
            for tid in examples:
                if tid in topic_model.topic_representations_:
                    words = [w for w, _ in topic_model.topic_representations_[tid][:5]]
                    logger.info(f"    Topic {tid}: {', '.join(words)}")
    else:
        logger.info("  No noise topics identified")
    
    return noise_topics


@contextmanager
def temporary_reindexed_hierarchy_model(
    topic_model: BERTopic,
    docs: list[str],
    topics: list[int],
    logger: Optional[logging.Logger] = None,
) -> tuple[list[str], list[int], TopicIdMapping]:
    """Temporarily reindex the model so that c_tf_idf_ rows match ONLY the topics present in `topics`.
    
    This avoids mismatches and keeps hierarchy + plotting consistent. The model state is
    restored when exiting the context.
    
    Args:
        topic_model: BERTopic model (will be temporarily modified)
        docs: List of sentence strings (already filtered to exclude outliers/noise)
        topics: List of topic assignments (already filtered to exclude -1 and noise)
        logger: Logger instance
        
    Yields:
        Tuple of (docs, remapped_topics, TopicIdMapping)
        The model state remains modified (reindexed) while in context.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Temporarily reindexing model for hierarchy construction...")
    logger.info(f"  Using {len(docs):,} documents")
    logger.info(f"  Topics provided: {len(topics):,}")
    
    if len(docs) != len(topics):
        raise ValueError(
            f"Documents ({len(docs)}) and topics ({len(topics)}) must have the same length"
        )
    
    # Backup state (exclude read-only properties like topic_labels_)
    backup = {
        "topics_": getattr(topic_model, "topics_", None),
        "c_tf_idf_": getattr(topic_model, "c_tf_idf_", None),
        "topic_representations_": copy.deepcopy(getattr(topic_model, "topic_representations_", None)),
        "topic_sizes_": copy.deepcopy(getattr(topic_model, "topic_sizes_", None)),
        "custom_labels_": copy.deepcopy(getattr(topic_model, "custom_labels_", None)),
        "topic_embeddings_": getattr(topic_model, "topic_embeddings_", None),
        "representative_docs_": copy.deepcopy(getattr(topic_model, "representative_docs_", None)),
    }
    
    try:
        # Determine topics to keep (exclude -1; noise already filtered before calling this)
        old_ids = sorted(set([t for t in topics if t != -1]))
        old_to_new = {old: i for i, old in enumerate(old_ids)}
        new_to_old = {i: old for old, i in old_to_new.items()}
        mapping = TopicIdMapping(old_to_new=old_to_new, new_to_old=new_to_old)
        
        logger.info(f"  Reindexing {len(old_ids)} topics: {old_ids[:5]}... -> 0..{len(old_ids)-1}")
        
        # Remap per-doc topics
        new_topics = [old_to_new[t] if t != -1 else -1 for t in topics]
        topic_model.topics_ = new_topics
        
        # Validate: all topic IDs should be -1 or in range [0, len(old_ids)-1]
        unique_new_topics = set(new_topics)
        max_valid_id = len(old_ids) - 1
        invalid_topics = [t for t in unique_new_topics if t != -1 and (t < 0 or t > max_valid_id)]
        if invalid_topics:
            raise ValueError(
                f"Invalid topic IDs after remapping: {invalid_topics}. "
                f"Expected range: [-1] or [0, {max_valid_id}]"
            )
        
        # Important: clear custom labels so later set_topic_labels won't try to "zip" wrong lengths
        topic_model.custom_labels_ = None
        
        # Determine whether the provided docs/topics include outliers
        has_outliers_in_docs = any(t == -1 for t in topics)
        
        # Set _outliers attribute (BERTopic uses this to determine num topics)
        # _outliers = 1 means row 0 in c_tf_idf_ is the outlier, 0 means no outlier row
        try:
            if hasattr(topic_model, "_outliers"):
                topic_model._outliers = 1 if has_outliers_in_docs else 0
        except (AttributeError, TypeError):
            # _outliers is likely a read-only property, which is fine
            pass
        
        # Filter c_tf_idf_ rows: align with new topic IDs
        # Original trained model almost always has outlier row at index 0
        # We want c_tf_idf_ rows to align with new topic ids 0..N-1
        if getattr(topic_model, "c_tf_idf_", None) is not None:
            from scipy.sparse import csr_matrix
            
            if has_outliers_in_docs:
                # Keep outlier row (index 0) + topic rows (old_id + 1)
                rows_to_keep = [0] + [old + 1 for old in old_ids]
                logger.info(f"  Filtered c_tf_idf_ to {len(rows_to_keep)} rows (0 for outlier, 1..{len(old_ids)} for topics 0..{max_valid_id})")
            else:
                # Drop outlier row - only keep topic rows (old_id + 1)
                rows_to_keep = [old + 1 for old in old_ids]
                logger.info(f"  Filtered c_tf_idf_ to {len(rows_to_keep)} rows (no outlier, topics 0..{max_valid_id})")
            
            topic_model.c_tf_idf_ = topic_model.c_tf_idf_[rows_to_keep]
            
            # Validate c_tf_idf_ shape matches expected number of topics
            expected_rows = len(old_ids) + (1 if has_outliers_in_docs else 0)
            if topic_model.c_tf_idf_.shape[0] != expected_rows:
                raise ValueError(
                    f"c_tf_idf_ shape mismatch: expected {expected_rows} rows, got {topic_model.c_tf_idf_.shape[0]}"
                )
        
        # Reindex topic representations
        reps = getattr(topic_model, "topic_representations_", None)
        if isinstance(reps, dict):
            topic_model.topic_representations_ = {
                new: reps[old] for new, old in new_to_old.items() if old in reps
            }
        
        # Reindex topic sizes
        sizes = getattr(topic_model, "topic_sizes_", None)
        if isinstance(sizes, dict):
            topic_model.topic_sizes_ = {new: sizes.get(old, 0) for new, old in new_to_old.items()}
        
        # Reindex representative docs (optional)
        repdocs = getattr(topic_model, "representative_docs_", None)
        if isinstance(repdocs, dict):
            topic_model.representative_docs_ = {
                new: repdocs.get(old) for new, old in new_to_old.items() if old in repdocs
            }
        
        logger.info("  ✓ Model reindexed")
        
        yield docs, new_topics, mapping
    
    finally:
        # Restore state
        for k, v in backup.items():
            if hasattr(topic_model, k):
                setattr(topic_model, k, v)
        
        logger.info("  ✓ Model state restored to original")


def get_topic_words_labels(
    topic_model: BERTopic,
    num_words: int = 3,
    logger: Optional[logging.Logger] = None,
) -> dict[int, str]:
    """Get topic words labels formatted as "word1_word2_word3 (T{id})".
    
    This function ensures ALL topics in the model are covered by building labels
    from sorted(set(topic_model.topics_)) rather than just topic_representations_.keys().
    
    Args:
        topic_model: Loaded BERTopic model (may be subsetted)
        num_words: Number of top words to include
        logger: Logger instance
        
    Returns:
        Dictionary mapping topic_id to formatted label (covers all topics in model)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    labels = {}
    
    # Get all unique topics from the model (excluding outlier -1)
    if hasattr(topic_model, "topics_") and topic_model.topics_ is not None:
        unique_topics = sorted(set(topic_id for topic_id in topic_model.topics_ if topic_id != -1))
    else:
        # Fallback to topic_representations_ if topics_ is not available
        if hasattr(topic_model, "topic_representations_"):
            unique_topics = sorted([tid for tid in topic_model.topic_representations_.keys() if tid != -1])
        else:
            logger.warning("Model does not have topics_ or topic_representations_ attribute")
            return labels
    
    if not hasattr(topic_model, "topic_representations_"):
        logger.warning("Model does not have topic_representations_ attribute")
        # Still create labels for all topics, just with generic names
        for topic_id in unique_topics:
            labels[topic_id] = f"Topic_{topic_id} (T{topic_id})"
        return labels
    
    # Build labels for all topics
    for topic_id in unique_topics:
        if topic_id in topic_model.topic_representations_:
            words_list = topic_model.topic_representations_[topic_id]
            # Extract top words (words_list is list of (word, score) tuples)
            if isinstance(words_list, list) and len(words_list) > 0:
                top_words = [word for word, _ in words_list[:num_words]]
                words_str = "_".join(top_words)
                labels[topic_id] = f"{words_str} (T{topic_id})"
            else:
                labels[topic_id] = f"Topic_{topic_id} (T{topic_id})"
        else:
            # Topic exists in topics_ but not in representations_ (shouldn't happen, but handle it)
            labels[topic_id] = f"Topic_{topic_id} (T{topic_id})"
    
    logger.debug(f"Generated word labels for {len(labels)} topics")
    
    return labels


def visualize_dendrogram(
    topic_model: BERTopic,
    hierarchical_topics: pd.DataFrame,
    output_path: Path,
    *,
    label_mode: str,  # "llm" or "words"
    mapping: TopicIdMapping,
    llm_labels_old: Optional[dict[int, str]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Visualize hierarchical dendrogram with consistent topic IDs.
    
    This function works with a reindexed model state. It maps labels/words back to
    original topic IDs for display, ensuring consistency.
    
    Args:
        topic_model: BERTopic model (should be in reindexed state from context manager)
        hierarchical_topics: Hierarchical topics structure (uses remapped topic IDs)
        output_path: Path to save HTML visualization
        label_mode: "llm" to use LLM labels, "words" to use topic words
        mapping: TopicIdMapping between original and remapped IDs
        llm_labels_old: Dictionary mapping original topic IDs to LLM labels (required for label_mode="llm")
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Backup current (temporary) labels
    original_custom = copy.deepcopy(getattr(topic_model, "custom_labels_", None))
    
    try:
        labels_by_new: dict[int, str] = {}
        
        if label_mode == "llm":
            if not llm_labels_old:
                raise ValueError("label_mode='llm' requires llm_labels_old mapping (old_id -> label).")
            
            for new_id, old_id in mapping.new_to_old.items():
                base = llm_labels_old.get(old_id, f"Topic_{old_id}")
                # Enforce original ID in label (so you can cross-ref back to the full model)
                label = base if f"(T{old_id})" in base else f"{base} (T{old_id})"
                labels_by_new[new_id] = label
        
        elif label_mode == "words":
            reps = getattr(topic_model, "topic_representations_", {}) or {}
            for new_id, old_id in mapping.new_to_old.items():
                words_list = reps.get(new_id) or []
                top_words = [w for w, _ in words_list[:3]] if words_list else [f"Topic_{old_id}"]
                label = f"{'_'.join(top_words)} (T{old_id})"
                labels_by_new[new_id] = label
        
        else:
            raise ValueError("label_mode must be one of: 'llm', 'words'")
        
        # Set labels using a dict (stable mapping, no ordering assumptions)
        # BERTopic's set_topic_labels() accepts both dict and list, but dict is more robust
        topic_model.set_topic_labels(labels_by_new)
        
        # Do NOT pass `topics=` or `top_n_topics=` if you want hierarchical names rendered
        fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics, custom_labels=True)
        fig.write_html(str(output_path))
        
        logger.info(f"✓ Dendrogram saved: {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    finally:
        # Restore original custom labels
        topic_model.custom_labels_ = original_custom


def print_topic_tree(
    topic_model: BERTopic,
    hierarchical_topics: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Print text tree for inspection.
    
    Note: get_topic_tree() uses the Child_Left_Name/Child_Right_Name columns from
    the hierarchical_topics DataFrame, not custom_labels_. To use LLM labels in the
    tree, you would need to rewrite those columns in the DataFrame before calling
    get_topic_tree().
    
    Args:
        topic_model: Loaded BERTopic model
        hierarchical_topics: Hierarchical topics structure
        logger: Logger instance
        
    Returns:
        Text representation of the tree
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Generating text tree representation...")
    
    try:
        tree = topic_model.get_topic_tree(hierarchical_topics)
        
        logger.info("✓ Text tree generated")
        logger.info("\n" + "=" * 80)
        logger.info("Hierarchical Topic Tree")
        logger.info("=" * 80)
        logger.info("\n" + tree)
        logger.info("\n" + "=" * 80)
        
        return tree
    except Exception as e:
        logger.error(f"Failed to generate topic tree: {e}")
        raise


def save_tree_to_file(
    tree: str,
    output_path: Path,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Save text tree to file.
    
    Args:
        tree: Text representation of the tree
        output_path: Path to save text file
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Saving text tree to: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tree)
    
    logger.info(f"✓ Text tree saved")
    logger.info(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")


def analyze_hierarchy_for_meta_topics(
    hierarchical_topics: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Analyze hierarchy to suggest target number of meta-topics.
    
    Args:
        hierarchical_topics: Hierarchical topics structure
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 80)
    logger.info("Hierarchy Analysis for Meta-Topic Selection")
    logger.info("=" * 80)
    
    if len(hierarchical_topics) == 0:
        logger.warning("No hierarchical topics to analyze")
        return
    
    if "Distance" not in hierarchical_topics.columns:
        logger.warning("hierarchical_topics has no 'Distance' column; "
                       "cannot analyze distance percentiles.")
        return
    
    # Extract distances from DataFrame
    distances = hierarchical_topics["Distance"].astype(float).tolist()
    
    distances_sorted = sorted(distances)
    
    # Calculate percentiles
    percentiles = [10, 25, 50, 75, 90]
    logger.info("\nDistance percentiles:")
    for p in percentiles:
        idx = int(len(distances_sorted) * p / 100)
        if idx < len(distances_sorted):
            logger.info(f"  {p}th percentile: {distances_sorted[idx]:.4f}")
    
    # Suggest target numbers based on typical recommendations (40-80)
    logger.info("\nRecommended target numbers of meta-topics:")
    logger.info("  - 40 topics: More aggregated, fewer interpretable groups")
    logger.info("  - 60 topics: Balanced (recommended starting point)")
    logger.info("  - 80 topics: More granular, closer to original topics")
    logger.info("\nLook for natural breakpoints in the dendrogram visualization")
    logger.info("  where branches separate clearly (large distance jumps)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Explore hierarchical topics structure for Stage 1 analysis"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/sentence_df_with_topics.parquet"),
        help="Path to input parquet file (matched-only dataframe with topics from Step 3)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base directory for models",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name",
    )
    parser.add_argument(
        "--model-suffix",
        type=str,
        default="_with_llm_labels",
        help="Model suffix (default: _with_llm_labels - model from stage08_llm_labeling)",
    )
    parser.add_argument(
        "--model-stage",
        type=str,
        default="stage08_llm_labeling",
        help="Stage subfolder to load model from (default: 'stage08_llm_labeling' to use model from stage08)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage09_category_mapping/stage1_natural_clusters"),
        help="Output directory for visualizations and logs",
    )
    parser.add_argument(
        "--save-tree",
        action="store_true",
        help="Save text tree to file (in addition to logging)",
    )
    parser.add_argument(
        "--include-noise",
        action="store_true",
        help="Include noise topics in hierarchy (default: exclude noise topics)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"stage06_category_mapping_explore_hierarchy_{timestamp}.log"
    logger = setup_logging(log_dir, log_file=log_file)
    
    logger.info("=" * 80)
    logger.info("Stage 1 Natural Clusters: Explore Hierarchical Topics")
    logger.info("=" * 80)
    logger.info("This script builds hierarchy using only matched sentences")
    logger.info("Model was trained on all 105 books, but hierarchy uses matched docs only")
    logger.info("=" * 80)
    logger.info(f"Input file: {args.input}")
    logger.info(f"Model base directory: {args.base_dir}")
    logger.info(f"Embedding model: {args.embedding_model}")
    logger.info(f"Model suffix: {args.model_suffix}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Step 1: Load dataframe
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading Matched-Only DataFrame with Topics")
    logger.info("=" * 80)
    
    df = load_dataframe(args.input, logger=logger)
    
    # Step 2: Extract sentences and topics
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Extracting Sentences and Topics")
    logger.info("=" * 80)
    
    docs = extract_sentences(df, logger=logger)
    topics = df["topic"].tolist()
    
    logger.info(f"✓ Extracted {len(topics):,} topic assignments")
    
    # Step 3: Load model
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Loading BERTopic Model")
    logger.info("=" * 80)
    
    topic_model = load_model(
        base_dir=args.base_dir,
        embedding_model=args.embedding_model,
        model_suffix=args.model_suffix,
        stage_subfolder=args.model_stage,
        logger=logger,
    )
    
    # Step 4: Filter noise and outliers BEFORE building hierarchy
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Filtering Noise Topics and Outliers")
    logger.info("=" * 80)
    
    # Identify noise topics if exclusion is requested
    noise_topics = set()
    if not args.include_noise:
        noise_topics = identify_noise_topics(topic_model, logger=logger)
    
    # Filter out outlier topics (-1) and noise topics
    valid_mask = [(t != -1) and (t not in noise_topics) for t in topics]
    valid_docs = [d for d, ok in zip(docs, valid_mask) if ok]
    valid_topics = [t for t, ok in zip(topics, valid_mask) if ok]
    
    excluded_count = sum(1 for t in topics if t in noise_topics)
    if not args.include_noise and excluded_count > 0:
        logger.info(f"  Excluded {excluded_count:,} sentences from {len(noise_topics)} noise topics")
    
    logger.info(f"  Filtered to {len(valid_docs):,} non-outlier, non-noise sentences")
    logger.info(f"  Unique topics in filtered data: {len(set(valid_topics))}")
    
    if len(valid_docs) == 0:
        raise ValueError("No valid (non-outlier) topics found in the data")
    
    # Step 5: Capture ORIGINAL (full-model) LLM labels once, using old IDs
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Capturing Original LLM Labels")
    logger.info("=" * 80)
    
    llm_labels_old = get_llm_label_map(topic_model)
    logger.info(f"  Captured {len(llm_labels_old)} LLM labels from original model")
    
    # Step 6: Build hierarchy on filtered docs/topics WITH reindexing (contiguous IDs)
    # This avoids IndexError by ensuring topic IDs are contiguous 0..N-1
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Building Hierarchical Topics Structure")
    logger.info("=" * 80)
    logger.info("  Building hierarchy on filtered documents (noise excluded unless --include-noise)...")
    logger.info("  Topics will be reindexed to contiguous IDs (0..N-1) to avoid IndexError.")
    logger.info("  This may take several minutes...")
    
    # Initialize variables for summary
    hierarchical_topics = None
    tree = None
    tree_path = None
    dendrogram_labels_path = None
    dendrogram_words_path = None
    
    # Use filtered docs/topics (noise already excluded in Step 4, unless --include-noise)
    docs_for_hierarchy = valid_docs if not args.include_noise else [d for d, t in zip(docs, topics) if t != -1]
    topics_for_hierarchy = valid_topics if not args.include_noise else [t for t in topics if t != -1]
    
    logger.info(f"  Using {len(docs_for_hierarchy):,} documents for hierarchy building")
    logger.info(f"  Unique topics in documents: {len(set(topics_for_hierarchy))}")
    
    # Build hierarchy + plot INSIDE the reindexed context manager
    # This ensures topic IDs are contiguous and all operations stay aligned
    with temporary_reindexed_hierarchy_model(
        topic_model, docs_for_hierarchy, topics_for_hierarchy, logger=logger
    ) as (docs_tmp, topics_tmp, mapping):
        
        # Build hierarchical structure (returns DataFrame with remapped topic IDs)
        # Use use_ctfidf=True now that c_tf_idf_ is properly aligned
        hierarchical_topics = topic_model.hierarchical_topics(docs_tmp, use_ctfidf=True)
        logger.info("  Built hierarchy using c-TF-IDF (use_ctfidf=True)")
        
        logger.info(f"✓ Hierarchical structure built")
        logger.info(f"  Number of hierarchical links: {len(hierarchical_topics)}")
        logger.info(f"  Columns: {', '.join(hierarchical_topics.columns.tolist())}")
        
        # Log some statistics
        if len(hierarchical_topics) > 0:
            if "Distance" in hierarchical_topics.columns:
                distances = hierarchical_topics["Distance"].astype(float).tolist()
                logger.info(f"  Distance range: [{min(distances):.4f}, {max(distances):.4f}]")
                logger.info(f"  Mean distance: {sum(distances) / len(distances):.4f}")
            else:
                logger.warning("No 'Distance' column found in hierarchical_topics; "
                               "skipping distance statistics.")
        
        # Step 7: Visualize dendrograms (two versions)
        # These run while model state is reindexed (noise topics already excluded)
        # IMPORTANT: All visualization must happen INSIDE the context manager so topic IDs match
        logger.info("\n" + "=" * 80)
        logger.info("Step 7: Creating Dendrogram Visualizations")
        logger.info("=" * 80)
        
        viz_dir = args.output_dir / "visualizations"
        
        # Version a) Labels + ID
        dendrogram_labels_path = viz_dir / f"hierarchy_dendrogram_labels_{timestamp}.html"
        visualize_dendrogram(
            topic_model,
            hierarchical_topics,
            dendrogram_labels_path,
            label_mode="llm",
            mapping=mapping,
            llm_labels_old=llm_labels_old,
            logger=logger,
        )
        
        # Version b) Topic words + ID
        dendrogram_words_path = viz_dir / f"hierarchy_dendrogram_words_{timestamp}.html"
        visualize_dendrogram(
            topic_model,
            hierarchical_topics,
            dendrogram_words_path,
            label_mode="words",
            mapping=mapping,
            logger=logger,
        )
        
        # Step 8: Print text tree
        # This also runs while model state is reindexed
        logger.info("\n" + "=" * 80)
        logger.info("Step 8: Generating Text Tree")
        logger.info("=" * 80)
        
        tree = print_topic_tree(topic_model, hierarchical_topics, logger=logger)
        
        # Step 9: Save tree to file (if requested)
        if args.save_tree:
            logger.info("\n" + "=" * 80)
            logger.info("Step 9: Saving Text Tree to File")
            logger.info("=" * 80)
            
            tree_path = args.output_dir / f"hierarchy_tree_{timestamp}.txt"
            save_tree_to_file(tree, tree_path, logger=logger)
        
        # Step 10: Analyze hierarchy for meta-topic selection
        logger.info("\n" + "=" * 80)
        logger.info("Step 10: Analyzing Hierarchy for Meta-Topic Selection")
        logger.info("=" * 80)
        
        analyze_hierarchy_for_meta_topics(hierarchical_topics, logger=logger)
    
    # Model state is now restored to original (outside context manager)
    # All visualization and tree generation happened INSIDE the context manager
    # to ensure topic IDs in hierarchical_topics DataFrame match the reindexed model state
    
    # Final summary (hierarchical_topics, tree, tree_path, dendrogram paths are from context)
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"Input sentences: {len(df):,}")
    logger.info(f"Books: {df['book_id'].nunique()}")
    logger.info(f"Unique topics: {df['topic'].nunique()}")
    logger.info(f"Hierarchical links: {len(hierarchical_topics)}")
    logger.info(f"Dendrogram (labels): {dendrogram_labels_path}")
    logger.info(f"Dendrogram (words): {dendrogram_words_path}")
    if args.save_tree and tree_path is not None:
        logger.info(f"Text tree: {tree_path}")
    logger.info(f"Log file: {log_dir / log_file}")
    logger.info("\n✓ Hierarchical exploration complete!")
    logger.info("\nNext steps:")
    logger.info("  1. Review the dendrogram visualization to identify natural breakpoints")
    logger.info("  2. Choose target number of meta-topics (typically 40-80)")
    logger.info("  3. Run reduce_to_meta_topics.py to reduce topics to chosen level")


if __name__ == "__main__":
    main()

