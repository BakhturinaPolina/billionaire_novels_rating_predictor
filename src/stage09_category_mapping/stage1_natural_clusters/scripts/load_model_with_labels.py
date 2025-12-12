"""Load BERTopic model and attach LLM labels for Stage 1 natural clusters analysis.

This script:
1. Loads the BERTopic model from model_1_with_llm_labels (from stage08_llm_labeling)
2. Checks if LLM labels are already integrated (via custom_labels_ attribute)
3. If not, loads labels from JSON file and attaches them
4. Verifies model has expected number of topics
5. Logs verification results

Note: According to MODEL_VERSIONING.md, model_1_with_llm_labels should already have
labels integrated and persisted. This script primarily verifies
the model state, but can reload labels if needed (e.g., with --force-reload-labels).

The JSON file may contain full structured data (label, scene_summary, primary_categories,
secondary_categories, is_noise, rationale) when using --use-improved-prompts, but only
the label field is extracted for topic assignment.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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


def load_labels_from_json(json_path: Path, logger: Optional[logging.Logger] = None) -> dict[int, str]:
    """Load topic labels from JSON file.
    
    Supports multiple JSON structures:
    - Structure 1: {topic_id: {label, keywords, scene_summary, primary_categories, secondary_categories, is_noise, rationale}}
      (Full JSON from --use-improved-prompts, extracts only label field)
    - Structure 2: {topic_id: "label"} (simple string format)
    - Structure 3: {topic_id: {label: "..."}} (nested dict with label key)
    
    Args:
        json_path: Path to labels JSON file
        logger: Logger instance
        
    Returns:
        Dictionary mapping topic_id to label string
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Loading labels from JSON: {json_path}")
    
    if not json_path.exists():
        raise FileNotFoundError(f"Labels JSON file not found: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle different JSON structures
    labels: dict[int, str] = {}
    
    # Structure 1: {topic_id: {label, keywords, scene_summary, primary_categories, secondary_categories, is_noise, rationale}}
    # (Full JSON structure from --use-improved-prompts, but we only extract the label field)
    if isinstance(data.get("0"), dict) and "label" in data.get("0", {}):
        for topic_id_str, topic_data in data.items():
            if isinstance(topic_data, dict) and "label" in topic_data:
                topic_id = int(topic_id_str)
                labels[topic_id] = topic_data["label"]
    
    # Structure 2: {topic_id: "label"}
    else:
        for topic_id_str, label in data.items():
            if isinstance(label, str):
                topic_id = int(topic_id_str)
                labels[topic_id] = label
            elif isinstance(label, dict) and "label" in label:
                topic_id = int(topic_id_str)
                labels[topic_id] = label["label"]
    
    logger.info(f"Loaded {len(labels)} topic labels from JSON")
    return labels


def load_full_metadata_from_json(json_path: Path, logger: Optional[logging.Logger] = None) -> dict[int, dict[str, Any]]:
    """Load full topic metadata from JSON file.
    
    Loads the complete JSON structure including label, keywords, categories, etc.
    
    Args:
        json_path: Path to labels JSON file
        logger: Logger instance
        
    Returns:
        Dictionary mapping topic_id to full metadata dict
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Loading full metadata from JSON: {json_path}")
    
    if not json_path.exists():
        raise FileNotFoundError(f"Labels JSON file not found: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Convert string keys to int and preserve full structure
    metadata: dict[int, dict[str, Any]] = {}
    
    for topic_id_str, topic_data in data.items():
        topic_id = int(topic_id_str)
        
        if isinstance(topic_data, dict):
            # Full structure with all fields
            metadata[topic_id] = topic_data.copy()
        elif isinstance(topic_data, str):
            # Simple label-only format, convert to dict
            metadata[topic_id] = {"label": topic_data}
    
    logger.info(f"Loaded full metadata for {len(metadata)} topics from JSON")
    return metadata


def verify_model_topics(
    topic_model: BERTopic, 
    expected_count: int = 361,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """Verify model has expected number of topics.
    
    Args:
        topic_model: Loaded BERTopic model
        expected_count: Expected number of topics (excluding -1)
        logger: Logger instance
        
    Returns:
        Dictionary with verification results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    results = {
        "expected_count": expected_count,
        "actual_count": None,
        "has_custom_labels": False,
        "custom_labels_count": 0,
        "topic_ids": [],
    }
    
    # Count topics (excluding -1)
    if hasattr(topic_model, "topic_representations_"):
        topic_ids = [
            tid for tid in topic_model.topic_representations_.keys() 
            if tid != -1
        ]
        results["topic_ids"] = sorted(topic_ids)
        results["actual_count"] = len(topic_ids)
    else:
        logger.warning("Model does not have topic_representations_ attribute")
        return results
    
    # Check for custom labels
    if hasattr(topic_model, "custom_labels_") and topic_model.custom_labels_:
        results["has_custom_labels"] = True
        results["custom_labels_count"] = len(topic_model.custom_labels_)
        logger.info(f"Found custom_labels_ with {results['custom_labels_count']} labels")
    
    # Also check if labels are accessible via get_topic_info() (as per MODEL_VERSIONING.md)
    try:
        topic_info = topic_model.get_topic_info()
        if "Name" in topic_info.columns:
            named_topics = topic_info[topic_info["Name"].notna()]
            if len(named_topics) > 0:
                logger.info(f"Found {len(named_topics)} topics with names in get_topic_info()")
    except Exception as e:
        logger.debug(f"Could not check get_topic_info(): {e}")
    
    # Verify count
    if results["actual_count"] == expected_count:
        logger.info(f"✓ Topic count verified: {results['actual_count']} topics (expected {expected_count})")
    else:
        logger.warning(
            f"⚠ Topic count mismatch: {results['actual_count']} topics (expected {expected_count})"
        )
    
    return results


def attach_labels_to_model(
    topic_model: BERTopic,
    labels: dict[int, str],
    logger: Optional[logging.Logger] = None,
) -> None:
    """Attach labels to BERTopic model.
    
    Args:
        topic_model: BERTopic model instance
        labels: Dictionary mapping topic_id to label
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Attaching {len(labels)} labels to model...")
    
    # Use BERTopic's set_topic_labels method
    topic_model.set_topic_labels(labels)
    
    # Verify labels were attached
    if hasattr(topic_model, "custom_labels_") and topic_model.custom_labels_:
        logger.info(f"✓ Labels attached successfully: {len(topic_model.custom_labels_)} labels")
    else:
        logger.warning("⚠ Labels may not have been attached correctly")


def attach_metadata_to_model(
    topic_model: BERTopic,
    metadata: dict[int, dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> None:
    """Attach full metadata to BERTopic model as custom attribute.
    
    Stores the full JSON structure (keywords, categories, etc.) in topic_metadata_
    attribute for easier access during analysis.
    
    Args:
        topic_model: BERTopic model instance
        metadata: Dictionary mapping topic_id to full metadata dict
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Attaching full metadata for {len(metadata)} topics to model...")
    
    # Store as custom attribute (BERTopic allows custom attributes)
    topic_model.topic_metadata_ = metadata
    
    # Verify metadata was attached
    if hasattr(topic_model, "topic_metadata_") and topic_model.topic_metadata_:
        logger.info(f"✓ Metadata attached successfully: {len(topic_model.topic_metadata_)} topics")
        
        # Log sample of what's stored
        sample_topic = list(metadata.keys())[0]
        sample_data = metadata[sample_topic]
        logger.info(f"  Sample metadata keys for topic {sample_topic}: {list(sample_data.keys())}")
    else:
        logger.warning("⚠ Metadata may not have been attached correctly")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Load BERTopic model and attach LLM labels for Stage 1 analysis"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to BERTopic model directory (default: uses model_1_with_noise_labels)",
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
        "--labels-json",
        type=Path,
        default=Path(
            "results/stage08_llm_labeling/"
            "labels_pos_openrouter_mistralai_Mistral-Nemo-Instruct-2407_romance_aware_paraphrase-MiniLM-L6-v2.json"
        ),
        help="Path to labels JSON file",
    )
    parser.add_argument(
        "--expected-topics",
        type=int,
        default=368,
        help="Expected number of topics excluding outlier -1 (default: 368, "
             "based on model_1_with_noise_labels). Use 361 for older models.",
    )
    parser.add_argument(
        "--force-reload-labels",
        action="store_true",
        help="Force reload labels even if model already has custom_labels_",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage09_category_mapping/stage1_natural_clusters"),
        help="Output directory for logs and reports",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save model with labels to stage09_category_mapping subfolder (both formats)",
    )
    parser.add_argument(
        "--save-stage",
        type=str,
        default=None,
        help="Override stage subfolder for saving (default: uses stage09_category_mapping or stage08_llm_labeling if --attach-metadata)",
    )
    parser.add_argument(
        "--attach-metadata",
        action="store_true",
        help="Also load and attach full metadata (keywords, categories, etc.) from JSON file",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"stage06_category_mapping_load_model_{timestamp}.log"
    logger = setup_logging(log_dir, log_file=log_file)
    
    logger.info("=" * 80)
    logger.info("Stage 1 Natural Clusters: Load Model with Labels")
    logger.info("=" * 80)
    logger.info("NOTE: model_1_with_noise_labels (from openrouter_experiments) should already have labels integrated")
    logger.info("      (stored in custom_labels_ and persisted in topics.json)")
    logger.info("      This script verifies the model state.")
    logger.info("=" * 80)
    logger.info(f"Model base directory: {args.base_dir}")
    logger.info(f"Embedding model: {args.embedding_model}")
    logger.info(f"Model suffix: {args.model_suffix}")
    logger.info(f"Model stage: {args.model_stage}")
    logger.info(f"Save model: {args.save_model}")
    logger.info(f"Labels JSON: {args.labels_json}")
    logger.info(f"Expected topics: {args.expected_topics}")
    
    # Load model
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading BERTopic Model")
    logger.info("=" * 80)
    
    load_start_time = time.time()
    wrapper = None
    
    try:
        if args.model_path:
            logger.info(f"Loading model from explicit path: {args.model_path}")
            if not args.model_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {args.model_path}")
            # Try loading as pickle first (might be a wrapper)
            if args.model_path.suffix == ".pkl":
                import pickle
                loaded_obj = pickle.load(open(args.model_path, "rb"))
                # Check if it's a RetrainableBERTopicModel wrapper
                if hasattr(loaded_obj, "trained_topic_model") and loaded_obj.trained_topic_model is not None:
                    logger.info("  Extracted BERTopic model from RetrainableBERTopicModel wrapper")
                    wrapper = loaded_obj
                    topic_model = loaded_obj.trained_topic_model
                elif isinstance(loaded_obj, BERTopic):
                    topic_model = loaded_obj
                    wrapper = None
                else:
                    # Try BERTopic.load() as fallback
                    topic_model = BERTopic.load(str(args.model_path))
                    wrapper = None
            else:
                topic_model = BERTopic.load(str(args.model_path))
                wrapper = None
        else:
            logger.info(f"Loading model using helper function (suffix: {args.model_suffix}, stage: {args.model_stage})")
            # Try to load wrapper first (for saving both formats later)
            try:
                wrapper, topic_model = load_retrained_wrapper(
                    base_dir=args.base_dir,
                    embedding_model=args.embedding_model,
                    pareto_rank=1,
                    model_suffix=args.model_suffix,
                    stage_subfolder=args.model_stage,
                )
                logger.info("✓ Loaded wrapper (will be able to save both formats)")
            except FileNotFoundError:
                # Fallback to native format if wrapper not available
                logger.info("Wrapper not found, loading native format only")
                topic_model = load_native_bertopic_model(
                    base_dir=args.base_dir,
                    embedding_model=args.embedding_model,
                    pareto_rank=1,
                    model_suffix=args.model_suffix,
                    stage_subfolder=args.model_stage,
                )
        
        load_elapsed = time.time() - load_start_time
        logger.info(f"✓ Model loaded successfully in {load_elapsed:.1f} seconds")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        logger.error(f"  Model path: {args.model_path if args.model_path else 'using helper function'}")
        raise
    
    # Verify model topics
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Verifying Model Topics")
    logger.info("=" * 80)
    
    verification = verify_model_topics(topic_model, expected_count=args.expected_topics, logger=logger)
    
    # Check if labels are already integrated
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Checking for Existing Labels and Metadata")
    logger.info("=" * 80)
    
    has_metadata = hasattr(topic_model, "topic_metadata_") and topic_model.topic_metadata_
    
    if verification["has_custom_labels"]:
        logger.info(f"✓ Model already has custom_labels_ with {verification['custom_labels_count']} labels")
        
        if has_metadata:
            logger.info(f"✓ Model already has topic_metadata_ with {len(topic_model.topic_metadata_)} topics")
            logger.info("  Full metadata is available (keywords, categories, etc.)")
        else:
            logger.info("  Full metadata not found - use --attach-metadata to load from JSON")
        
        if args.force_reload_labels:
            logger.info("Force reload requested, will reload labels from JSON")
        elif args.attach_metadata and not has_metadata:
            logger.info("Metadata attachment requested, will load from JSON")
        else:
            logger.info("Labels already integrated. Use --force-reload-labels to override.")
            # Don't return early if --save-model is requested or if we need to attach metadata
            if not args.save_model and not (args.attach_metadata and not has_metadata):
                logger.info("\n" + "=" * 80)
                logger.info("Summary")
                logger.info("=" * 80)
                logger.info(f"Model loaded: ✓")
                logger.info(f"Topic count: {verification['actual_count']} (expected {verification['expected_count']})")
                logger.info(f"Custom labels: ✓ ({verification['custom_labels_count']} labels)")
                if has_metadata:
                    logger.info(f"Full metadata: ✓ ({len(topic_model.topic_metadata_)} topics)")
                logger.info("\nModel is ready for Stage 1 analysis!")
                return
    
    # Load and attach labels
    if not verification["has_custom_labels"] or args.force_reload_labels:
        logger.info("\n" + "=" * 80)
        logger.info("Step 4: Loading and Attaching Labels")
        logger.info("=" * 80)
        
        try:
            labels = load_labels_from_json(args.labels_json, logger=logger)
            
            # Verify label count matches topic count
            if len(labels) != verification["actual_count"]:
                logger.warning(
                    f"Label count ({len(labels)}) doesn't match topic count "
                    f"({verification['actual_count']}). Some topics may not have labels."
                )
            
            attach_labels_to_model(topic_model, labels, logger=logger)
            
            # Re-verify after attaching
            verification = verify_model_topics(topic_model, expected_count=args.expected_topics, logger=logger)
        except Exception as e:
            logger.error(f"✗ Failed to load/attach labels: {e}")
            logger.error(f"  Labels JSON: {args.labels_json}")
            raise
    
    # Optionally attach full metadata
    if args.attach_metadata:
        logger.info("\n" + "=" * 80)
        logger.info("Step 4b: Loading and Attaching Full Metadata")
        logger.info("=" * 80)
        
        try:
            metadata = load_full_metadata_from_json(args.labels_json, logger=logger)
            attach_metadata_to_model(topic_model, metadata, logger=logger)
            
            # Update wrapper's trained_topic_model if wrapper exists
            if wrapper is not None and hasattr(wrapper, "trained_topic_model"):
                wrapper.trained_topic_model = topic_model
                logger.info("Updated wrapper's trained_topic_model with metadata")
        except Exception as e:
            logger.error(f"✗ Failed to load/attach metadata: {e}")
            logger.error(f"  Labels JSON: {args.labels_json}")
            raise
    
    # Save model with labels to stage09 subfolder (if requested)
    if args.save_model:
        logger.info("\n" + "=" * 80)
        logger.info("Step 5: Saving Model with Labels and Metadata")
        logger.info("=" * 80)
        
        try:
            # Determine save location
            if args.save_stage:
                # Use explicitly specified stage
                stage_subfolder = args.base_dir / args.embedding_model / args.save_stage
            elif args.attach_metadata:
                # If metadata is attached, save to new subfolder in stage08_llm_labeling
                stage_subfolder = args.base_dir / args.embedding_model / "stage08_llm_labeling" / "with_metadata"
            else:
                # Default: save to stage09_category_mapping
                stage_subfolder = args.base_dir / args.embedding_model / "stage09_category_mapping"
            
            stage_subfolder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving to: {stage_subfolder}")
            
            # Determine model suffix based on what's being saved
            if args.attach_metadata and hasattr(topic_model, "topic_metadata_"):
                model_suffix = "_with_llm_labels_and_metadata"
            else:
                model_suffix = "_with_categories"
            
            # 1. Save as native BERTopic model (directory format)
            native_model_dir = stage_subfolder / f"model_1{model_suffix}"
            if native_model_dir.exists() and native_model_dir.is_dir():
                shutil.rmtree(native_model_dir)
            
            with stage_timer(f"Saving native BERTopic model to {native_model_dir}"):
                topic_model.save(str(native_model_dir))
                logger.info("Saved native BERTopic model to %s", native_model_dir)
            
            # 2. Save as wrapper pickle (file format) - only if wrapper was loaded
            if wrapper is not None:
                wrapper_pickle_path = stage_subfolder / f"model_1{model_suffix}.pkl"
                backup_existing_file(wrapper_pickle_path)
                
                with stage_timer(f"Saving wrapper to {wrapper_pickle_path.name}"):
                    with open(wrapper_pickle_path, "wb") as f:
                        pickle.dump(wrapper, f)
                    logger.info("Saved wrapper to %s", wrapper_pickle_path)
            else:
                # If no wrapper, create a minimal wrapper with just the topic model
                logger.info("No wrapper available, creating minimal wrapper for pickle save...")
                wrapper_pickle_path = stage_subfolder / f"model_1{model_suffix}.pkl"
                backup_existing_file(wrapper_pickle_path)
                
                # Create a simple wrapper class to preserve the model
                class MinimalWrapper:
                    def __init__(self, topic_model):
                        self.trained_topic_model = topic_model
                
                minimal_wrapper = MinimalWrapper(topic_model)
                with stage_timer(f"Saving minimal wrapper to {wrapper_pickle_path.name}"):
                    with open(wrapper_pickle_path, "wb") as f:
                        pickle.dump(minimal_wrapper, f)
                    logger.info("Saved minimal wrapper to %s", wrapper_pickle_path)
            
            logger.info(f"✓ Saved model to {stage_subfolder}")
            logger.info(f"  Native format: {native_model_dir}")
            logger.info(f"  Pickle format: {wrapper_pickle_path}")
        except Exception as e:
            logger.error(f"✗ Failed to save model: {e}")
            logger.error("  Model has labels but was not saved")
            raise
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"Model loaded: ✓")
    logger.info(f"Topic count: {verification['actual_count']} (expected {verification['expected_count']})")
    if verification["has_custom_labels"]:
        logger.info(f"Custom labels: ✓ ({verification['custom_labels_count']} labels)")
    else:
        logger.warning("Custom labels: ✗ (not found)")
    
    # Check for metadata
    if hasattr(topic_model, "topic_metadata_") and topic_model.topic_metadata_:
        logger.info(f"Full metadata: ✓ ({len(topic_model.topic_metadata_)} topics)")
        logger.info("  Metadata includes: keywords, categories, scene_summary, rationale, etc.")
    else:
        logger.info("Full metadata: ✗ (not attached - use --attach-metadata to load from JSON)")
    
    logger.info("\nModel is ready for Stage 1 analysis!")
    logger.info("\nNext step: Run assign_topics_to_sentences.py to assign topics to matched sentences")


if __name__ == "__main__":
    main()

