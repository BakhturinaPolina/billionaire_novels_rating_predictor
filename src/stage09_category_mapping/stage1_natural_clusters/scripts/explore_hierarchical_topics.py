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
import logging
import pickle
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from bertopic import BERTopic

from src.common.logging import setup_logging
from src.stage06_topic_exploration.explore_retrained_model import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    load_native_bertopic_model,
)


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


def identify_noise_topics(
    topic_model: BERTopic,
    logger: Optional[logging.Logger] = None,
) -> set[int]:
    """Identify noise topics from model labels and word patterns.
    
    Noise topics are identified by:
    1. Labels starting with [NOISE_CANDIDATE: or [NOISE:
    2. LLM labels with is_noise=True (if available in structured format)
    3. Word patterns indicating noise (contractions, common stopwords, etc.)
    
    Args:
        topic_model: Loaded BERTopic model
        logger: Logger instance
        
    Returns:
        Set of topic IDs that are noise topics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    noise_topics = set()
    
    # Common contractions and noise indicators
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
        # Method 1: Check labels for noise prefixes
        # Try get_topic_info() first, but handle models that don't have it
        try:
            info = topic_model.get_topic_info()
            
            # Check for custom labels column
            label_cols = [c for c in info.columns if c.lower().startswith("custom")]
            if label_cols:
                label_col = label_cols[0]
                for _, row in info.iterrows():
                    topic_id = int(row["Topic"])
                    if topic_id == -1:  # Skip outlier
                        continue
                    
                    label = row[label_col]
                    if pd.notna(label):
                        label_str = str(label)
                        # Check for noise prefixes
                        if label_str.startswith("[NOISE_CANDIDATE:") or label_str.startswith("[NOISE:"):
                            noise_topics.add(topic_id)
        except (AttributeError, TypeError):
            # Model doesn't have get_topic_info(), skip this method
            pass
        
        # Also check custom_labels_ attribute if available
        if hasattr(topic_model, "custom_labels_") and topic_model.custom_labels_:
            if isinstance(topic_model.custom_labels_, dict):
                for topic_id, label in topic_model.custom_labels_.items():
                    if topic_id == -1:
                        continue
                    if isinstance(label, str):
                        if label.startswith("[NOISE_CANDIDATE:") or label.startswith("[NOISE:"):
                            noise_topics.add(topic_id)
        
        # Method 2: Check topic word representations for noise patterns
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
        logger.warning(f"Could not identify noise topics: {e}")
    
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
def build_hierarchical_topics_context(
    topic_model: BERTopic,
    docs: list[str],
    topics: list[int],
    exclude_noise: bool = True,
    logger: Optional[logging.Logger] = None,
):
    """Context manager that builds hierarchical structure and keeps model state active.
    
    This context manager:
    1. Filters out noise topics and outliers
    2. Subsets/remaps the model's c_tf_idf_ and topic_representations_ to match filtered data
    3. Builds hierarchical_topics DataFrame
    4. Yields the hierarchical_topics DataFrame while keeping model state modified
    5. Restores original model state when exiting
    
    Args:
        topic_model: Loaded BERTopic model (will be temporarily modified)
        docs: List of sentence strings (matched only)
        topics: List of topic assignments for the matched docs
        exclude_noise: If True, exclude topics labeled as noise
        logger: Logger instance
        
    Yields:
        Tuple of (hierarchical_topics DataFrame, valid_topics list, old_to_new_topic dict or None)
        The model state remains modified (subsetted) while in context.
        old_to_new_topic maps original topic IDs to remapped IDs (None if no remapping occurred).
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Building hierarchical topics structure...")
    logger.info(f"  Using {len(docs):,} matched sentences only")
    logger.info(f"  Topics provided: {len(topics):,}")
    
    if len(docs) != len(topics):
        raise ValueError(
            f"Documents ({len(docs)}) and topics ({len(topics)}) must have the same length"
        )
    
    logger.info("  This may take several minutes...")
    
    # Identify noise topics if exclusion is requested
    noise_topics = set()
    if exclude_noise:
        noise_topics = identify_noise_topics(topic_model, logger=logger)
    
    # Filter out outlier topics (-1) and noise topics
    # hierarchical_topics() requires valid topic IDs that exist in the model
    valid_mask = [(t != -1) and (t not in noise_topics) for t in topics]
    valid_docs = [doc for doc, valid in zip(docs, valid_mask) if valid]
    valid_topics_original = [t for t, valid in zip(topics, valid_mask) if valid]
    
    excluded_count = sum(1 for t in topics if t in noise_topics)
    if exclude_noise and excluded_count > 0:
        logger.info(f"  Excluded {excluded_count:,} sentences from {len(noise_topics)} noise topics")
    
    logger.info(f"  Filtered to {len(valid_docs):,} non-outlier, non-noise sentences")
    logger.info(f"  Unique topics in filtered data: {len(set(valid_topics_original))}")
    
    if len(valid_docs) == 0:
        raise ValueError("No valid (non-outlier) topics found in the data")
    
    # Check which topics from the model are present in our data
    # This helps diagnose potential mismatches
    unique_topics_in_data = set(valid_topics_original)
    if hasattr(topic_model, "topic_representations_"):
        model_topic_ids = set(topic_model.topic_representations_.keys()) - {-1} - noise_topics
        missing_in_data = model_topic_ids - unique_topics_in_data
        if missing_in_data:
            logger.info(f"  Topics in model but not in data: {len(missing_in_data)} topics")
            logger.warning(
                "  ⚠️  This can cause IndexError in hierarchical_topics()! "
                "The method assumes all topics in c_tf_idf_ appear in documents."
            )
            logger.info("  Attempting workaround: filtering c_tf_idf_ to match documents...")
    
    # Save original state for restoration
    original_topics = topic_model.topics_ if hasattr(topic_model, "topics_") else None
    original_c_tf_idf = None
    original_topic_representations = None
    original_custom_labels = None
    old_to_new_topic = None  # Will be set if remapping occurs - maps old_id -> new_id
    
    # Save original custom_labels_ if it exists
    if hasattr(topic_model, "custom_labels_") and topic_model.custom_labels_:
        if isinstance(topic_model.custom_labels_, dict):
            original_custom_labels = topic_model.custom_labels_.copy()
    
    try:
        # Set topics to match our matched documents (only non-outlier topics)
        topic_model.topics_ = valid_topics_original
        
        # Workaround for BERTopic bug: hierarchical_topics() fails when some topics
        # in c_tf_idf_ don't appear in documents. We need to filter c_tf_idf_ to only
        # include topics present in our documents.
        valid_topics = valid_topics_original  # Will be remapped if needed
        if hasattr(topic_model, "c_tf_idf_") and topic_model.c_tf_idf_ is not None:
            # Get sorted list of topics present in data (excluding -1)
            topics_in_data_sorted = sorted(unique_topics_in_data)
            
            # Check if we need to filter
            if hasattr(topic_model, "topic_representations_"):
                model_topic_ids = sorted([tid for tid in topic_model.topic_representations_.keys() if tid != -1 and tid not in noise_topics])
                if set(topics_in_data_sorted) != set(model_topic_ids):
                    logger.info(f"  Filtering c_tf_idf_ from {len(model_topic_ids)} to {len(topics_in_data_sorted)} topics")
                    
                    # Save original state
                    original_c_tf_idf = topic_model.c_tf_idf_
                    if hasattr(topic_model, "topic_representations_"):
                        original_topic_representations = topic_model.topic_representations_.copy()
                    
                    # Filter c_tf_idf_ to only include topics present in data
                    # c_tf_idf_ has shape (num_topics + 1, vocab_size) where +1 is for outlier -1
                    # Row 0 is outlier (-1), rows 1+ are topics 0, 1, 2, ...
                    from scipy.sparse import csr_matrix
                    import numpy as np
                    
                    # Create mapping: topic_id -> row_index in original c_tf_idf_
                    # Row 0 is outlier (-1), rows 1+ are topics 0, 1, 2, ...
                    topic_to_row = {tid: tid + 1 for tid in model_topic_ids}
                    topic_to_row[-1] = 0  # Outlier is at row 0
                    
                    # Get row indices we want to keep: outlier (0) + topics in data (sorted)
                    rows_to_keep = [0]  # Always keep outlier row
                    rows_to_keep.extend([topic_to_row[tid] for tid in topics_in_data_sorted])
                    
                    # Filter c_tf_idf_
                    topic_model.c_tf_idf_ = topic_model.c_tf_idf_[rows_to_keep]
                    
                    # Create new topic ID mapping: old_topic_id -> new_topic_id
                    # After filtering, row 0 is still outlier (-1), row 1 is the first topic
                    # BERTopic uses 0-indexed topic IDs (0, 1, 2, ...), so row 1 -> topic 0, row 2 -> topic 1, etc.
                    old_to_new_topic = {-1: -1}  # Outlier stays as -1
                    for new_row_idx, old_topic_id in enumerate(topics_in_data_sorted, start=1):
                        # Map row index to topic ID (row 1 -> topic 0, row 2 -> topic 1, etc.)
                        new_topic_id = new_row_idx - 1
                        old_to_new_topic[old_topic_id] = new_topic_id
                    
                    # Remap valid_topics to use new topic IDs (row indices)
                    valid_topics = [old_to_new_topic.get(tid, tid) for tid in valid_topics_original]
                    
                    # Filter and remap topic_representations_ to match new row indices
                    if hasattr(topic_model, "topic_representations_"):
                        filtered_representations = {}
                        # Keep outlier at -1
                        if -1 in topic_model.topic_representations_:
                            filtered_representations[-1] = topic_model.topic_representations_[-1]
                        # Remap other topics to new row indices
                        for old_topic_id in topics_in_data_sorted:
                            if old_topic_id in topic_model.topic_representations_:
                                new_topic_id = old_to_new_topic[old_topic_id]
                                filtered_representations[new_topic_id] = topic_model.topic_representations_[old_topic_id]
                        topic_model.topic_representations_ = filtered_representations
                    
                    # Also remap custom_labels_ if it exists
                    if hasattr(topic_model, "custom_labels_") and topic_model.custom_labels_:
                        if isinstance(topic_model.custom_labels_, dict):
                            remapped_labels = {}
                            if -1 in topic_model.custom_labels_:
                                remapped_labels[-1] = topic_model.custom_labels_[-1]
                            for old_topic_id in topics_in_data_sorted:
                                if old_topic_id in topic_model.custom_labels_:
                                    new_topic_id = old_to_new_topic[old_topic_id]
                                    remapped_labels[new_topic_id] = topic_model.custom_labels_[old_topic_id]
                            topic_model.custom_labels_ = remapped_labels
                    
                    # Update _outliers attribute if it exists (should be 1, meaning skip row 0)
                    # Note: _outliers is a read-only property, so we can't set it directly
                    # The value should already be correct from the model
                    try:
                        if hasattr(topic_model, "_outliers"):
                            # Try to set it, but it may be read-only
                            topic_model._outliers = 1
                    except (AttributeError, TypeError):
                        # _outliers is likely a read-only property, which is fine
                        # The model should already have the correct value
                        pass
                    
                    logger.info(f"  ✓ Filtered c_tf_idf_ from {len(model_topic_ids)} to {len(topics_in_data_sorted)} topics")
                    logger.info(f"  ✓ Remapped topic IDs to match new row indices")
                    
                    # Update topic_model.topics_ with remapped topic IDs
                    topic_model.topics_ = valid_topics
        
        # Build hierarchical structure (returns DataFrame)
        # Note: hierarchical_topics() is called AFTER remapping, so the DataFrame should
        # already have remapped topic IDs. Parent IDs (>= num_topics) are synthetic and
        # don't need remapping.
        hierarchical_topics = topic_model.hierarchical_topics(valid_docs)
        
        logger.info(f"✓ Hierarchical structure built")
        logger.info(f"  Number of hierarchical links: {len(hierarchical_topics)}")
        logger.info(f"  Columns: {', '.join(hierarchical_topics.columns.tolist())}")
        
        # Log some statistics
        if len(hierarchical_topics) > 0:
            # hierarchical_topics is a DataFrame, use the Distance column
            if "Distance" in hierarchical_topics.columns:
                distances = hierarchical_topics["Distance"].astype(float).tolist()
                logger.info(f"  Distance range: [{min(distances):.4f}, {max(distances):.4f}]")
                logger.info(f"  Mean distance: {sum(distances) / len(distances):.4f}")
            else:
                logger.warning("No 'Distance' column found in hierarchical_topics; "
                               "skipping distance statistics.")
            
            # Workaround: visualize_hierarchy() builds all_topics from self.topics_, but
            # hierarchical_topics DataFrame contains parent IDs (>= num_topics) that aren't
            # in self.topics_. We need to add parent IDs to self.topics_ temporarily so
            # visualize_hierarchy() can find them.
            if "Child_Left_ID" in hierarchical_topics.columns and "Child_Right_ID" in hierarchical_topics.columns:
                all_ids_in_df = set(
                    [int(x) for x in hierarchical_topics["Child_Left_ID"].tolist()] +
                    [int(x) for x in hierarchical_topics["Child_Right_ID"].tolist()]
                )
                if "Parent_ID" in hierarchical_topics.columns:
                    all_ids_in_df.update([int(x) for x in hierarchical_topics["Parent_ID"].tolist()])
                
                num_topics = len(set(valid_topics))
                parent_ids = [tid for tid in all_ids_in_df if tid >= num_topics]
                
                if parent_ids:
                    # Save state before modifying (if not already saved)
                    if original_topic_representations is None and hasattr(topic_model, "topic_representations_"):
                        original_topic_representations = topic_model.topic_representations_.copy()
                    if original_c_tf_idf is None and hasattr(topic_model, "c_tf_idf_") and topic_model.c_tf_idf_ is not None:
                        original_c_tf_idf = topic_model.c_tf_idf_
                    
                    # Add parent IDs to self.topics_ so visualize_hierarchy() can find them
                    # Parent IDs are synthetic and don't need document assignments
                    extended_topics = list(valid_topics) + parent_ids
                    topic_model.topics_ = extended_topics
                    
                    # Also add placeholder entries to topic_representations_ for parent IDs
                    # visualize_hierarchy() may use topic_representations_.keys() to build all_topics
                    if hasattr(topic_model, "topic_representations_"):
                        for parent_id in parent_ids:
                            if parent_id not in topic_model.topic_representations_:
                                # Add placeholder representation for parent ID
                                topic_model.topic_representations_[parent_id] = [("parent", 0.0)]
                    
                    # Add placeholder rows to c_tf_idf_ for parent IDs
                    # visualize_hierarchy() needs to access c_tf_idf_[parent_id] for embeddings
                    if hasattr(topic_model, "c_tf_idf_") and topic_model.c_tf_idf_ is not None:
                        from scipy.sparse import vstack, csr_matrix
                        import numpy as np
                        
                        # Get the shape of c_tf_idf_ (num_rows, vocab_size)
                        num_rows, vocab_size = topic_model.c_tf_idf_.shape
                        max_parent_id = max(parent_ids)
                        
                        # Create placeholder rows (zeros) for parent IDs
                        # c_tf_idf_ has row 0 for outlier (-1), rows 1+ for topics 0, 1, 2, ...
                        # Parent IDs start at num_topics, so we need rows at indices (parent_id + 1)
                        # But we need to add rows up to max_parent_id
                        num_rows_needed = max_parent_id + 1 - num_rows
                        if num_rows_needed > 0:
                            # Create zero rows for parent IDs
                            zero_rows = csr_matrix((num_rows_needed, vocab_size), dtype=topic_model.c_tf_idf_.dtype)
                            topic_model.c_tf_idf_ = vstack([topic_model.c_tf_idf_, zero_rows])
                            logger.debug(f"  Added {num_rows_needed} placeholder rows to c_tf_idf_ for parent IDs")
                    
                    logger.debug(f"  Added {len(parent_ids)} parent topic IDs to model.topics_ and topic_representations_ for visualization")
        
        # Yield hierarchical_topics, valid_topics, and old_to_new_topic mapping while model state is modified
        yield hierarchical_topics, valid_topics, old_to_new_topic
    
    finally:
        # Restore original state
        if original_topics is not None:
            topic_model.topics_ = original_topics
        elif hasattr(topic_model, "topics_"):
            # If there were no original topics, we can leave it as is or clear it
            # Leaving it as is should be fine since we're not modifying the model permanently
            pass
        
        # Restore original c_tf_idf_ and topic_representations_ if we modified them
        if original_c_tf_idf is not None:
            topic_model.c_tf_idf_ = original_c_tf_idf
        if original_topic_representations is not None:
            topic_model.topic_representations_ = original_topic_representations
        if original_custom_labels is not None:
            topic_model.custom_labels_ = original_custom_labels
        
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
    use_labels: bool = True,
    num_actual_topics: Optional[int] = None,
    old_to_new_topic: Optional[dict[int, int]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Visualize hierarchical dendrogram.
    
    Note: This function assumes the model has already been subsetted (noise topics removed)
    by the context manager. The model state should already exclude noise topics.
    
    Args:
        topic_model: Loaded BERTopic model (should be subsetted, noise already excluded)
        hierarchical_topics: Hierarchical topics structure (uses remapped topic IDs)
        output_path: Path to save HTML visualization
        use_labels: If True, use custom labels; if False, use topic words
        num_actual_topics: Number of actual topics (excluding parent IDs)
        old_to_new_topic: Optional mapping from original topic IDs to remapped IDs
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    label_type = "labels" if use_labels else "topic words"
    logger.info(f"Creating hierarchical dendrogram visualization with {label_type}...")
    
    # Get current labels from the model (already remapped if remapping occurred)
    # Since we're inside the context manager, custom_labels_ uses remapped topic IDs
    # which match the hierarchical_topics DataFrame
    original_labels = None
    try:
        # Prefer custom_labels_ directly since it's already remapped to match the hierarchy
        if hasattr(topic_model, "custom_labels_") and topic_model.custom_labels_:
            if isinstance(topic_model.custom_labels_, dict):
                original_labels = topic_model.custom_labels_.copy()
        else:
            # Fallback to get_topic_info() if custom_labels_ not available
            info = topic_model.get_topic_info()
            label_cols = [c for c in info.columns if c.lower().startswith("custom")]
            if label_cols:
                label_col = label_cols[0]
                original_labels = dict(zip(info["Topic"], info[label_col]))
    except Exception as e:
        logger.debug(f"Could not get original labels: {e}")
    
    try:
        if use_labels:
            # Use custom labels with IDs (they should already have IDs from disambiguation)
            # Since model is already subsetted, all topics in custom_labels_ are valid
            if original_labels:
                # Ensure all labels have IDs
                labels_with_ids = {}
                for topic_id, label in original_labels.items():
                    if topic_id == -1 or pd.isna(label):
                        continue
                    label_str = str(label)
                    # Skip if label starts with [NOISE: (shouldn't happen if subsetted correctly, but double-check)
                    if label_str.startswith("[NOISE_CANDIDATE:") or label_str.startswith("[NOISE:"):
                        continue
                    if f"(T{topic_id})" not in label_str:
                        labels_with_ids[topic_id] = f"{label_str} (T{topic_id})"
                    else:
                        labels_with_ids[topic_id] = label_str
                
                # Add placeholder labels for parent topic IDs (synthetic IDs >= num_topics)
                # These are needed because we added parent IDs to self.topics_ for visualize_hierarchy()
                if hasattr(topic_model, "topics_") and topic_model.topics_ is not None:
                    # Use provided num_actual_topics if available, otherwise calculate from topics_
                    if num_actual_topics is None:
                        # Calculate num_topics from topics_ by excluding very large IDs (parent IDs)
                        num_topics = len(set([t for t in topic_model.topics_ if t < 1000]))
                    else:
                        num_topics = num_actual_topics
                    for topic_id in set(topic_model.topics_):
                        if topic_id >= num_topics and topic_id not in labels_with_ids:
                            # Parent IDs are synthetic, use a generic label
                            labels_with_ids[topic_id] = f"Parent_Topic_{topic_id} (T{topic_id})"
                
                # Save extended labels for restoration
                current_labels = labels_with_ids.copy()
                
                # Set labels with IDs (all topics in model are already non-noise)
                topic_model.set_topic_labels(labels_with_ids)
            
            fig = topic_model.visualize_hierarchy(
                hierarchical_topics=hierarchical_topics,
                custom_labels=True,
            )
        else:
            # Use topic words with IDs
            # get_topic_words_labels() now covers all topics in the model
            words_labels = get_topic_words_labels(topic_model, num_words=3, logger=logger)
            
            # Add placeholder labels for parent topic IDs (synthetic IDs >= num_topics)
            # These are needed because we added parent IDs to self.topics_ for visualize_hierarchy()
            if hasattr(topic_model, "topics_") and topic_model.topics_ is not None:
                # Use provided num_actual_topics if available, otherwise calculate from topics_
                if num_actual_topics is None:
                    # Calculate num_topics from topics_ by excluding very large IDs (parent IDs)
                    num_topics = len(set([t for t in topic_model.topics_ if t < 1000]))
                else:
                    num_topics = num_actual_topics
                for topic_id in set(topic_model.topics_):
                    if topic_id >= num_topics and topic_id not in words_labels:
                        # Parent IDs are synthetic, use a generic label
                        words_labels[topic_id] = f"Parent_Topic_{topic_id} (T{topic_id})"
            
            # Save extended labels for restoration
            current_labels = words_labels.copy()
            
            topic_model.set_topic_labels(words_labels)
            
            fig = topic_model.visualize_hierarchy(
                hierarchical_topics=hierarchical_topics,
                custom_labels=True,
            )
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        
        logger.info(f"✓ Dendrogram saved to: {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    except Exception as e:
        logger.error(f"Failed to create dendrogram visualization: {e}")
        raise
    finally:
        # Restore original labels to avoid label leakage between "labels" and "words" dendrograms
        # Note: We're still in the context manager, so self.topics_ still has parent IDs.
        # We restore original_labels to clean up. When the context manager exits, it will restore
        # self.topics_ and custom_labels_ to their original state anyway.
        if original_labels is not None:
            topic_model.set_topic_labels(original_labels)


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
    
    # Step 4: Build hierarchical structure and keep model state active for visualization/tree
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Building Hierarchical Topics Structure")
    logger.info("=" * 80)
    
    # Initialize variables for summary
    hierarchical_topics = None
    tree = None
    tree_path = None
    dendrogram_labels_path = None
    dendrogram_words_path = None
    
    # Use context manager to keep subsetted model state active for all visualization steps
    with build_hierarchical_topics_context(
        topic_model, docs, topics, exclude_noise=not args.include_noise, logger=logger
    ) as (hierarchical_topics, valid_topics, old_to_new_topic):
        
        # Step 5: Visualize dendrograms (two versions)
        # These run while model state is subsetted (noise topics already excluded)
        # IMPORTANT: All visualization must happen INSIDE the context manager so topic IDs match
        logger.info("\n" + "=" * 80)
        logger.info("Step 5: Creating Dendrogram Visualizations")
        logger.info("=" * 80)
        
        viz_dir = args.output_dir / "visualizations"
        
        # Version a) Labels + ID
        dendrogram_labels_path = viz_dir / f"hierarchy_dendrogram_labels_{timestamp}.html"
        num_actual_topics = len(set(valid_topics))  # Number of actual topics (excluding parent IDs)
        visualize_dendrogram(
            topic_model, hierarchical_topics, dendrogram_labels_path, 
            use_labels=True, num_actual_topics=num_actual_topics, 
            old_to_new_topic=old_to_new_topic, logger=logger
        )
        
        # Version b) Topic words + ID
        dendrogram_words_path = viz_dir / f"hierarchy_dendrogram_words_{timestamp}.html"
        visualize_dendrogram(
            topic_model, hierarchical_topics, dendrogram_words_path,
            use_labels=False, num_actual_topics=num_actual_topics,
            old_to_new_topic=old_to_new_topic, logger=logger
        )
        
        # Step 6: Print text tree
        # This also runs while model state is subsetted (remapped topic IDs)
        logger.info("\n" + "=" * 80)
        logger.info("Step 6: Generating Text Tree")
        logger.info("=" * 80)
        
        tree = print_topic_tree(topic_model, hierarchical_topics, logger=logger)
        
        # Step 7: Save tree to file (if requested)
        if args.save_tree:
            logger.info("\n" + "=" * 80)
            logger.info("Step 7: Saving Text Tree to File")
            logger.info("=" * 80)
            
            tree_path = args.output_dir / f"hierarchy_tree_{timestamp}.txt"
            save_tree_to_file(tree, tree_path, logger=logger)
        
        # Step 8: Analyze hierarchy for meta-topic selection
        logger.info("\n" + "=" * 80)
        logger.info("Step 8: Analyzing Hierarchy for Meta-Topic Selection")
        logger.info("=" * 80)
        
        analyze_hierarchy_for_meta_topics(hierarchical_topics, logger=logger)
    
    # Model state is now restored to original (outside context manager)
    # All visualization and tree generation happened INSIDE the context manager
    # to ensure topic IDs in hierarchical_topics DataFrame match the remapped model state
    
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

