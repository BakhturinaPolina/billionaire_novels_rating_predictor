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
            "labels_pos_openrouter_mistralai_mistral-nemo_romance_aware_paraphrase-MiniLM-L6-v2_reasoning_high.json"
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
            topic_model = BERTopic.load(str(args.model_path))
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
    logger.info("Step 3: Checking for Existing Labels")
    logger.info("=" * 80)
    
    if verification["has_custom_labels"]:
        logger.info(f"✓ Model already has custom_labels_ with {verification['custom_labels_count']} labels")
        
        if args.force_reload_labels:
            logger.info("Force reload requested, will reload labels from JSON")
        else:
            logger.info("Labels already integrated. Use --force-reload-labels to override.")
            logger.info("\n" + "=" * 80)
            logger.info("Summary")
            logger.info("=" * 80)
            logger.info(f"Model loaded: ✓")
            logger.info(f"Topic count: {verification['actual_count']} (expected {verification['expected_count']})")
            logger.info(f"Custom labels: ✓ ({verification['custom_labels_count']} labels)")
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
    
    # Save model with labels to stage09 subfolder (if requested)
    if args.save_model:
        logger.info("\n" + "=" * 80)
        logger.info("Step 5: Saving Model with Labels")
        logger.info("=" * 80)
        
        try:
            # Create stage subfolder path
            stage_subfolder = args.base_dir / args.embedding_model / "stage09_category_mapping"
            stage_subfolder.mkdir(parents=True, exist_ok=True)
            
            # 1. Save as native BERTopic model (directory format)
            native_model_dir = stage_subfolder / f"model_1_with_categories"
            if native_model_dir.exists() and native_model_dir.is_dir():
                shutil.rmtree(native_model_dir)
            
            with stage_timer(f"Saving native BERTopic model with categories to {native_model_dir}"):
                topic_model.save(str(native_model_dir))
                logger.info("Saved native BERTopic model with categories to %s", native_model_dir)
            
            # 2. Save as wrapper pickle (file format) - only if wrapper was loaded
            if wrapper is not None:
                wrapper_pickle_path = stage_subfolder / "model_1_with_categories.pkl"
                backup_existing_file(wrapper_pickle_path)
                
                with stage_timer(f"Saving wrapper with categories to {wrapper_pickle_path.name}"):
                    with open(wrapper_pickle_path, "wb") as f:
                        pickle.dump(wrapper, f)
                    logger.info("Saved wrapper with categories to %s", wrapper_pickle_path)
            else:
                logger.info("Note: Only native BERTopic format saved (no wrapper available)")
            
            logger.info(f"✓ Saved model to {stage_subfolder}")
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
    
    logger.info("\nModel is ready for Stage 1 analysis!")
    logger.info("\nNext step: Run assign_topics_to_sentences.py to assign topics to matched sentences")


if __name__ == "__main__":
    main()

