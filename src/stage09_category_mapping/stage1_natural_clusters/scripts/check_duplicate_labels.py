"""Check for duplicate labels in BERTopic model and optionally disambiguate them.

This script:
1. Loads a BERTopic model
2. Checks if different topic IDs share the same label text
3. Reports duplicate labels
4. Optionally disambiguates labels by appending topic IDs

Usage:
    python -m src.stage09_category_mapping.stage1_natural_clusters.check_duplicate_labels \
        --model-suffix _with_llm_labels \
        [--disambiguate] \
        [--save-model]
"""

from __future__ import annotations

import argparse
import logging
import shutil
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
    load_retrained_wrapper,
    backup_existing_file,
    stage_timer,
)
import pickle


def check_duplicate_labels(
    topic_model: BERTopic,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Check for duplicate labels in the model.
    
    Args:
        topic_model: Loaded BERTopic model
        logger: Logger instance
        
    Returns:
        DataFrame with duplicate labels (columns: CustomName, Topic list)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Checking for duplicate labels...")
    
    # Get topic info
    info = topic_model.get_topic_info()
    
    # Find the custom label column
    label_cols = [c for c in info.columns if c.lower().startswith("custom")]
    if not label_cols:
        logger.warning("No custom label column found in get_topic_info()")
        logger.info("Checking custom_labels_ attribute directly...")
        
        if hasattr(topic_model, "custom_labels_") and topic_model.custom_labels_:
            # Build DataFrame from custom_labels_ dict
            labels_dict = topic_model.custom_labels_
            info = pd.DataFrame([
                {"Topic": tid, "CustomName": label}
                for tid, label in labels_dict.items()
            ])
            label_cols = ["CustomName"]
        else:
            logger.error("No custom labels found in model")
            return pd.DataFrame()
    
    label_col = label_cols[0]
    logger.info(f"Using label column: {label_col}")
    
    # Filter to topics with labels (exclude NaN)
    labeled_topics = info.loc[info[label_col].notna(), ["Topic", label_col]].copy()
    
    if len(labeled_topics) == 0:
        logger.warning("No topics with labels found")
        return pd.DataFrame()
    
    logger.info(f"Found {len(labeled_topics)} topics with labels")
    
    # Group by label and collect topic IDs
    dups = (
        labeled_topics
        .groupby(label_col)
        .agg({"Topic": list})
        .reset_index()
    )
    
    # Show only labels used for more than one topic
    dups_multi = dups[dups["Topic"].map(len) > 1].copy()
    
    if len(dups_multi) == 0:
        logger.info("✓ No duplicate labels found - all labels are unique")
    else:
        logger.warning(f"⚠ Found {len(dups_multi)} labels used by multiple topics")
        logger.info("\nDuplicate labels:")
        logger.info("=" * 80)
        for _, row in dups_multi.iterrows():
            logger.info(f"  Label: '{row[label_col]}'")
            logger.info(f"    Topics: {row['Topic']}")
        logger.info("=" * 80)
    
    return dups_multi


def disambiguate_labels(
    topic_model: BERTopic,
    logger: Optional[logging.Logger] = None,
) -> dict[int, str]:
    """Disambiguate labels by appending topic IDs.
    
    Args:
        topic_model: Loaded BERTopic model
        logger: Logger instance
        
    Returns:
        Dictionary mapping topic_id to disambiguated label
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Disambiguating labels by appending topic IDs...")
    
    # Get current labels
    info = topic_model.get_topic_info()
    
    # Find the custom label column
    label_cols = [c for c in info.columns if c.lower().startswith("custom")]
    if not label_cols:
        if hasattr(topic_model, "custom_labels_") and topic_model.custom_labels_:
            labels_dict = topic_model.custom_labels_
            info = pd.DataFrame([
                {"Topic": tid, "CustomName": label}
                for tid, label in labels_dict.items()
            ])
            label_cols = ["CustomName"]
        else:
            raise ValueError("No custom labels found in model")
    
    label_col = label_cols[0]
    
    # Create disambiguated labels
    info["CustomName_v2"] = info.apply(
        lambda row: f"{row[label_col]} (T{row.Topic})" if pd.notna(row[label_col]) else None,
        axis=1
    )
    
    # Build new labels dict
    labels_v2 = dict(zip(info["Topic"], info["CustomName_v2"]))
    
    # Remove None values (topics without labels)
    labels_v2 = {tid: label for tid, label in labels_v2.items() if label is not None}
    
    logger.info(f"✓ Created {len(labels_v2)} disambiguated labels")
    
    return labels_v2


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check for duplicate labels in BERTopic model and optionally disambiguate them"
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
        help="Model suffix (default: _with_llm_labels)",
    )
    parser.add_argument(
        "--model-stage",
        type=str,
        default="stage08_llm_labeling",
        help="Stage subfolder to load model from",
    )
    parser.add_argument(
        "--disambiguate",
        action="store_true",
        help="Disambiguate duplicate labels by appending topic IDs",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save model with disambiguated labels (requires --disambiguate)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage09_category_mapping/stage1_natural_clusters"),
        help="Output directory for logs and reports",
    )
    
    args = parser.parse_args()
    
    if args.save_model and not args.disambiguate:
        parser.error("--save-model requires --disambiguate")
    
    # Setup logging
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"check_duplicate_labels_{timestamp}.log"
    logger = setup_logging(log_dir, log_file=log_file)
    
    logger.info("=" * 80)
    logger.info("Check Duplicate Labels")
    logger.info("=" * 80)
    logger.info(f"Model base directory: {args.base_dir}")
    logger.info(f"Embedding model: {args.embedding_model}")
    logger.info(f"Model suffix: {args.model_suffix}")
    logger.info(f"Model stage: {args.model_stage}")
    logger.info(f"Disambiguate: {args.disambiguate}")
    logger.info(f"Save model: {args.save_model}")
    
    # Load model
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading BERTopic Model")
    logger.info("=" * 80)
    
    wrapper = None
    try:
        # Construct model path with optional stage subfolder
        if args.model_stage:
            model_dir = args.base_dir / args.embedding_model / args.model_stage / f"model_1{args.model_suffix}"
        else:
            model_dir = args.base_dir / args.embedding_model / f"model_1{args.model_suffix}"
        
        logger.info(f"Looking for model at: {model_dir}")
        
        # Try to load wrapper first (for saving both formats later)
        wrapper_pickle_path = model_dir.parent / f"model_1{args.model_suffix}.pkl"
        if wrapper_pickle_path.exists():
            try:
                with open(wrapper_pickle_path, "rb") as f:
                    wrapper = pickle.load(f)
                topic_model = wrapper.trained_topic_model
                logger.info("✓ Loaded wrapper (will be able to save both formats)")
            except Exception as e:
                logger.warning(f"Failed to load wrapper: {e}, trying native format")
                wrapper = None
        
        # Fallback to native format if wrapper not available
        if wrapper is None:
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
            logger.info("Loading native BERTopic format")
            topic_model = BERTopic.load(str(model_dir))
        
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        raise
    
    # Check for duplicate labels
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Checking for Duplicate Labels")
    logger.info("=" * 80)
    
    dups_df = check_duplicate_labels(topic_model, logger=logger)
    
    # Disambiguate if requested
    if args.disambiguate:
        logger.info("\n" + "=" * 80)
        logger.info("Step 3: Disambiguating Labels")
        logger.info("=" * 80)
        
        labels_v2 = disambiguate_labels(topic_model, logger=logger)
        
        # Apply disambiguated labels
        topic_model.set_topic_labels(labels_v2)
        logger.info("✓ Applied disambiguated labels to model")
        
        # Verify no duplicates remain
        logger.info("\nVerifying no duplicates remain...")
        dups_after = check_duplicate_labels(topic_model, logger=logger)
        if len(dups_after) == 0:
            logger.info("✓ Confirmed: no duplicate labels remain")
        else:
            logger.warning("⚠ Some duplicates still remain (this should not happen)")
        
        # Update wrapper if available
        if wrapper is not None:
            wrapper.trained_topic_model = topic_model
            logger.info("✓ Updated wrapper.trained_topic_model reference")
        
        # Save model if requested
        if args.save_model:
            logger.info("\n" + "=" * 80)
            logger.info("Step 4: Saving Model with Disambiguated Labels")
            logger.info("=" * 80)
            
            try:
                # Create stage subfolder path
                stage_subfolder = args.base_dir / args.embedding_model / "stage09_category_mapping"
                stage_subfolder.mkdir(parents=True, exist_ok=True)
                
                # Create new suffix with disambiguated labels
                new_suffix = args.model_suffix + "_disambiguated"
                
                # 1. Save as native BERTopic model (directory format)
                native_model_dir = stage_subfolder / f"model_1{new_suffix}"
                if native_model_dir.exists() and native_model_dir.is_dir():
                    shutil.rmtree(native_model_dir)
                
                with stage_timer(f"Saving native BERTopic model with disambiguated labels to {native_model_dir}"):
                    topic_model.save(str(native_model_dir))
                    logger.info("Saved native BERTopic model with disambiguated labels to %s", native_model_dir)
                
                # 2. Save as wrapper pickle (file format) - only if wrapper was loaded
                if wrapper is not None:
                    wrapper_pickle_path = stage_subfolder / f"model_1{new_suffix}.pkl"
                    backup_existing_file(wrapper_pickle_path)
                    
                    with stage_timer(f"Saving wrapper with disambiguated labels to {wrapper_pickle_path.name}"):
                        with open(wrapper_pickle_path, "wb") as f:
                            pickle.dump(wrapper, f)
                        logger.info("Saved wrapper with disambiguated labels to %s", wrapper_pickle_path)
                else:
                    logger.info("Note: Only native BERTopic format saved (no wrapper available)")
                
                logger.info(f"✓ Saved model to {stage_subfolder}")
            except Exception as e:
                logger.error(f"✗ Failed to save model: {e}")
                raise
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"Duplicate labels found: {len(dups_df)}")
    if args.disambiguate:
        logger.info("Labels disambiguated: ✓")
        if args.save_model:
            logger.info("Model saved: ✓")
    else:
        logger.info("\nTo disambiguate labels, run with --disambiguate")
        logger.info("To save the model, run with --disambiguate --save-model")


if __name__ == "__main__":
    main()
