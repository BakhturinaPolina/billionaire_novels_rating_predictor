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
    logger: Optional[logging.Logger] = None,
) -> BERTopic:
    """Load BERTopic model.
    
    Args:
        base_dir: Base directory for models
        embedding_model: Embedding model name
        model_suffix: Model suffix (e.g., "_with_noise_labels")
        logger: Logger instance
        
    Returns:
        Loaded BERTopic model
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Loading BERTopic model (suffix: {model_suffix})...")
    
    topic_model = load_native_bertopic_model(
        base_dir=base_dir,
        embedding_model=embedding_model,
        pareto_rank=1,
        model_suffix=model_suffix,
    )
    
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


def build_hierarchical_topics(
    topic_model: BERTopic,
    docs: list[str],
    topics: list[int],
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Build hierarchical structure on matched docs.
    
    Args:
        topic_model: Loaded BERTopic model
        docs: List of sentence strings (matched only)
        topics: List of topic assignments for the matched docs
        logger: Logger instance
        
    Returns:
        Hierarchical topics DataFrame
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
    
    # Filter out outlier topics (-1) and ensure all topic IDs are valid
    # hierarchical_topics() requires valid topic IDs that exist in the model
    valid_mask = [t != -1 for t in topics]
    valid_docs = [doc for doc, valid in zip(docs, valid_mask) if valid]
    valid_topics = [t for t, valid in zip(topics, valid_mask) if valid]
    
    logger.info(f"  Filtered to {len(valid_docs):,} non-outlier sentences")
    logger.info(f"  Unique topics in filtered data: {len(set(valid_topics))}")
    
    if len(valid_docs) == 0:
        raise ValueError("No valid (non-outlier) topics found in the data")
    
    # Check which topics from the model are present in our data
    # This helps diagnose potential mismatches
    unique_topics_in_data = set(valid_topics)
    if hasattr(topic_model, "topic_representations_"):
        model_topic_ids = set(topic_model.topic_representations_.keys()) - {-1}
        missing_in_data = model_topic_ids - unique_topics_in_data
        if missing_in_data:
            logger.info(f"  Topics in model but not in data: {len(missing_in_data)} topics")
            logger.warning(
                "  ⚠️  This can cause IndexError in hierarchical_topics()! "
                "The method assumes all topics in c_tf_idf_ appear in documents."
            )
            logger.info("  Attempting workaround: filtering c_tf_idf_ to match documents...")
    
    # Temporarily set the model's topics_ to match our matched documents
    # This is necessary because hierarchical_topics() uses self.topics_ internally
    original_topics = topic_model.topics_ if hasattr(topic_model, "topics_") else None
    original_c_tf_idf = None
    original_topic_representations = None
    
    try:
        # Set topics to match our matched documents (only non-outlier topics)
        topic_model.topics_ = valid_topics
        
        # Workaround for BERTopic bug: hierarchical_topics() fails when some topics
        # in c_tf_idf_ don't appear in documents. We need to filter c_tf_idf_ to only
        # include topics present in our documents.
        if hasattr(topic_model, "c_tf_idf_") and topic_model.c_tf_idf_ is not None:
            # Get sorted list of topics present in data (excluding -1)
            topics_in_data_sorted = sorted(unique_topics_in_data)
            
            # Check if we need to filter
            if hasattr(topic_model, "topic_representations_"):
                model_topic_ids = sorted([tid for tid in topic_model.topic_representations_.keys() if tid != -1])
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
                    valid_topics = [old_to_new_topic.get(tid, tid) for tid in valid_topics]
                    
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
        
        return hierarchical_topics
    
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


def visualize_dendrogram(
    topic_model: BERTopic,
    hierarchical_topics: pd.DataFrame,
    output_path: Path,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Visualize hierarchical dendrogram.
    
    Args:
        topic_model: Loaded BERTopic model
        hierarchical_topics: Hierarchical topics structure
        output_path: Path to save HTML visualization
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Creating hierarchical dendrogram visualization...")
    
    try:
        fig = topic_model.visualize_hierarchy(
            hierarchical_topics=hierarchical_topics,
            custom_labels=True,  # uses LLM labels if set
        )
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        
        logger.info(f"✓ Dendrogram saved to: {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    except Exception as e:
        logger.error(f"Failed to create dendrogram visualization: {e}")
        raise


def print_topic_tree(
    topic_model: BERTopic,
    hierarchical_topics: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Print text tree for inspection.
    
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
        default="_with_noise_labels",
        help="Model suffix (default: _with_noise_labels - model from openrouter_experiments without category prefixes)",
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
        logger=logger,
    )
    
    # Step 4: Build hierarchical structure
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Building Hierarchical Topics Structure")
    logger.info("=" * 80)
    
    hierarchical_topics = build_hierarchical_topics(topic_model, docs, topics, logger=logger)
    
    # Step 5: Visualize dendrogram
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Creating Dendrogram Visualization")
    logger.info("=" * 80)
    
    viz_dir = args.output_dir / "visualizations"
    dendrogram_path = viz_dir / f"hierarchy_dendrogram_{timestamp}.html"
    visualize_dendrogram(topic_model, hierarchical_topics, dendrogram_path, logger=logger)
    
    # Step 6: Print text tree
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
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"Input sentences: {len(df):,}")
    logger.info(f"Books: {df['book_id'].nunique()}")
    logger.info(f"Unique topics: {df['topic'].nunique()}")
    logger.info(f"Hierarchical links: {len(hierarchical_topics)}")
    logger.info(f"Dendrogram: {dendrogram_path}")
    if args.save_tree:
        logger.info(f"Text tree: {tree_path}")
    logger.info(f"Log file: {log_dir / log_file}")
    logger.info("\n✓ Hierarchical exploration complete!")
    logger.info("\nNext steps:")
    logger.info("  1. Review the dendrogram visualization to identify natural breakpoints")
    logger.info("  2. Choose target number of meta-topics (typically 40-80)")
    logger.info("  3. Run reduce_to_meta_topics.py to reduce topics to chosen level")


if __name__ == "__main__":
    main()

