"""Assign topics to sentences using BERTopic model (inference only).

This script:
1. Loads the matched-only dataframe from Step 1 (sentence_df_with_ratings.parquet)
2. Extracts sentences from the dataframe
3. Loads the BERTopic model (trained on all 105 books)
4. Transforms only the matched sentences (inference only, no retraining)
5. Attaches topic assignments to the dataframe
6. Saves the output with topic assignments

Key principle:
- Model was trained on all 105 books (100% of sentences)
- This script transforms only the 92 matched books (~612k sentences, ~90%)
- The unmatched 10% of sentences helped shape the topic space during training,
  but are excluded from statistical analysis to avoid contamination.
"""

from __future__ import annotations

import argparse
import logging
import time
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


def load_dataframe(input_path: Path, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Load the matched-only dataframe from Step 1.
    
    Args:
        input_path: Path to sentence_df_with_ratings.parquet
        logger: Logger instance
        
    Returns:
        DataFrame with matched sentences only
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
    required_cols = ["text", "book_id", "rating_class"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
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


def assign_topics(
    topic_model: BERTopic,
    docs: list[str],
    batch_size: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[list[int], Optional[list[float]]]:
    """Transform sentences to get topic assignments (inference only).
    
    Args:
        topic_model: Loaded BERTopic model
        docs: List of sentence strings
        batch_size: Optional batch size for processing (None = process all at once)
        logger: Logger instance
        
    Returns:
        Tuple of (topics, probabilities)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Transforming sentences to get topic assignments (inference only)...")
    logger.info(f"  Processing {len(docs):,} sentences...")
    
    start_time = time.time()
    
    # Determine if we should use batch processing
    # BERTopic's transform can be slow for large datasets, but it doesn't natively support
    # batching. However, we can process in chunks and combine results.
    # For very large datasets (>100k), batch processing helps with memory and progress tracking.
    use_batching = batch_size is not None and len(docs) > 100000
    
    if use_batching and batch_size:
        logger.info(f"  Using batch processing (batch_size={batch_size:,})")
        logger.info(f"  Total batches: {(len(docs) + batch_size - 1) // batch_size}")
        
        all_topics = []
        all_probs = []
        
        num_batches = (len(docs) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(docs))
            batch_docs = docs[batch_start:batch_end]
            
            batch_start_time = time.time()
            logger.info(f"  Processing batch {batch_idx + 1}/{num_batches} "
                       f"(sentences {batch_start:,}-{batch_end:,})...")
            
            try:
                # Transform batch
                batch_topics, batch_probs = topic_model.transform(batch_docs)
                
                # Convert to lists if needed
                if isinstance(batch_topics, np.ndarray):
                    batch_topics = batch_topics.tolist()
                if isinstance(batch_probs, np.ndarray):
                    batch_probs = batch_probs.tolist()
                
                all_topics.extend(batch_topics)
                if batch_probs is not None:
                    all_probs.extend(batch_probs)
                
                batch_elapsed = time.time() - batch_start_time
                elapsed_total = time.time() - start_time
                avg_time_per_batch = elapsed_total / (batch_idx + 1)
                remaining_batches = num_batches - (batch_idx + 1)
                estimated_remaining = avg_time_per_batch * remaining_batches
                
                logger.info(f"    ✓ Batch {batch_idx + 1} complete in {batch_elapsed:.1f}s")
                logger.info(f"    Progress: {batch_end:,}/{len(docs):,} sentences "
                           f"({100*batch_end/len(docs):.1f}%)")
                logger.info(f"    Estimated time remaining: {estimated_remaining/60:.1f} minutes")
                
            except Exception as e:
                logger.error(f"    ✗ Error processing batch {batch_idx + 1}: {e}")
                logger.error(f"    Batch range: {batch_start:,}-{batch_end:,}")
                raise
        
        topics = all_topics
        probs = all_probs if all_probs else None
        
    else:
        # Process all at once (faster for smaller datasets or when batch_size not specified)
        if batch_size is None and len(docs) > 100000:
            logger.info("  Processing all sentences at once (this may take a while)...")
            logger.info("  Note: For very large datasets, consider using --batch-size")
        
        try:
            # Transform documents (inference only, no retraining)
            topics, probs = topic_model.transform(docs)
            
            # Convert to lists if they're numpy arrays
            if isinstance(topics, np.ndarray):
                topics = topics.tolist()
            if isinstance(probs, np.ndarray):
                probs = probs.tolist()
                
        except Exception as e:
            logger.error(f"Error during topic transformation: {e}")
            logger.error(f"Consider using --batch-size for large datasets")
            raise
    
    elapsed_time = time.time() - start_time
    logger.info(f"✓ Topic assignment complete in {elapsed_time/60:.1f} minutes ({elapsed_time:.1f}s)")
    logger.info(f"  Throughput: {len(docs)/elapsed_time:.0f} sentences/second")
    
    # Log topic distribution
    topics_array = np.array(topics)
    unique_topics, counts = np.unique(topics_array, return_counts=True)
    logger.info(f"  Assigned topics: {len(unique_topics)} unique topics")
    
    outlier_count = counts[unique_topics == -1][0] if -1 in unique_topics else 0
    logger.info(f"  Outlier topic (-1) count: {outlier_count:,} ({100*outlier_count/len(docs):.2f}%)")
    logger.info(f"  Non-outlier topics: {len(unique_topics) - (1 if -1 in unique_topics else 0)}")
    
    # Log top topics
    if len(unique_topics) > 0:
        top_10_indices = np.argsort(counts)[-10:][::-1]
        top_10_topics = unique_topics[top_10_indices]
        top_10_counts = counts[top_10_indices]
        logger.info("  Top 10 topics by frequency:")
        for topic_id, count in zip(top_10_topics, top_10_counts):
            logger.info(f"    Topic {topic_id}: {count:,} sentences ({100*count/len(docs):.2f}%)")
    
    return topics, probs


def attach_topics_to_dataframe(
    df: pd.DataFrame,
    topics: list[int],
    probs: Optional[list[float]] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Attach topic assignments to dataframe.
    
    Args:
        df: Original dataframe
        topics: List of topic assignments
        probs: Optional list of topic probabilities
        logger: Logger instance
        
    Returns:
        DataFrame with topic columns added
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Attaching topic assignments to dataframe...")
    
    if len(topics) != len(df):
        raise ValueError(
            f"Topic count ({len(topics)}) doesn't match dataframe length ({len(df)})"
        )
    
    df = df.copy()
    df["topic"] = topics
    
    if probs is not None:
        if len(probs) != len(df):
            logger.warning(
                f"Probability count ({len(probs)}) doesn't match dataframe length ({len(df)}), "
                "skipping probability assignment"
            )
        else:
            df["topic_prob"] = probs
            logger.info("✓ Attached topic probabilities")
    
    logger.info(f"✓ Attached topic assignments to {len(df):,} sentences")
    logger.info(f"  New columns: topic" + (", topic_prob" if probs is not None else ""))
    
    return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Assign topics to sentences using BERTopic model (inference only)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/sentence_df_with_ratings.parquet"),
        help="Path to input parquet file (matched-only dataframe from Step 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/sentence_df_with_topics.parquet"),
        help="Path to save output parquet file with topic assignments",
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
        "--include-probs",
        action="store_true",
        help="Include topic probabilities in output (slower, uses more memory)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for processing large datasets (default: None = process all at once). "
             "Recommended: 50000-100000 for datasets >100k sentences to enable progress tracking.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage09_category_mapping/stage1_natural_clusters"),
        help="Output directory for logs",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"stage06_category_mapping_assign_topics_{timestamp}.log"
    logger = setup_logging(log_dir, log_file=log_file)
    
    logger.info("=" * 80)
    logger.info("Stage 1 Natural Clusters: Assign Topics to Sentences")
    logger.info("=" * 80)
    logger.info("This script performs inference only (no retraining)")
    logger.info("Model was trained on all 105 books, but we transform only matched sentences")
    logger.info("=" * 80)
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Model base directory: {args.base_dir}")
    logger.info(f"Embedding model: {args.embedding_model}")
    logger.info(f"Model suffix: {args.model_suffix}")
    logger.info(f"Include probabilities: {args.include_probs}")
    logger.info(f"Batch size: {args.batch_size if args.batch_size else 'None (process all at once)'}")
    
    # Step 1: Load dataframe
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading Matched-Only DataFrame")
    logger.info("=" * 80)
    
    df = load_dataframe(args.input, logger=logger)
    
    # Step 2: Extract sentences
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Extracting Sentences")
    logger.info("=" * 80)
    
    docs = extract_sentences(df, logger=logger)
    
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
    
    # Step 4: Assign topics
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Assigning Topics (Inference Only)")
    logger.info("=" * 80)
    
    topics, probs = assign_topics(
        topic_model, 
        docs, 
        batch_size=args.batch_size,
        logger=logger
    )
    
    # Only use probabilities if requested
    if not args.include_probs:
        probs = None
    
    # Step 5: Attach to dataframe
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Attaching Topics to DataFrame")
    logger.info("=" * 80)
    
    df_with_topics = attach_topics_to_dataframe(df, topics, probs, logger=logger)
    
    # Step 6: Save output
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Saving Output")
    logger.info("=" * 80)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_with_topics.to_parquet(args.output, index=False)
    
    logger.info(f"✓ Saved {len(df_with_topics):,} sentences with topic assignments")
    logger.info(f"  Output file: {args.output}")
    logger.info(f"  File size: {args.output.stat().st_size / (1024*1024):.2f} MB")
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"Input sentences: {len(df):,}")
    logger.info(f"Output sentences: {len(df_with_topics):,}")
    logger.info(f"Books: {df_with_topics['book_id'].nunique()}")
    logger.info(f"Unique topics assigned: {df_with_topics['topic'].nunique()}")
    logger.info(f"Outlier sentences (topic -1): {(df_with_topics['topic'] == -1).sum():,}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Log file: {log_dir / log_file}")
    logger.info("\n✓ Topic assignment complete!")
    logger.info("\nNext step: Run explore_hierarchical_topics.py to build hierarchical structure")


if __name__ == "__main__":
    main()

