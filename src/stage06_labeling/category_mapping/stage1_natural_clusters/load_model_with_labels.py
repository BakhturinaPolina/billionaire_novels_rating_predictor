"""Load BERTopic model and attach LLM labels for Stage 1 natural clusters analysis.

This script:
1. Loads the BERTopic model from model_1_with_categories
2. Checks if LLM labels are already integrated (via custom_labels_ attribute)
3. If not, loads labels from JSON file and attaches them
4. Verifies model has expected number of topics (361)
5. Logs verification results

Note: According to MODEL_VERSIONING.md, model_1_with_categories should already have
labels integrated and persisted in topics.json. This script primarily verifies
the model state, but can reload labels if needed (e.g., with --force-reload-labels).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from bertopic import BERTopic

from src.common.logging import setup_logging
from src.stage06_exploration.explore_retrained_model import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    load_native_bertopic_model,
)


def load_labels_from_json(json_path: Path, logger: Optional[logging.Logger] = None) -> dict[int, str]:
    """Load topic labels from JSON file.
    
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
    
    # Structure 1: {topic_id: {label, keywords, scene_summary}}
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
        help="Path to BERTopic model directory (default: uses model_1_with_categories)",
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
        default="_with_categories",
        help="Model suffix (default: _with_categories)",
    )
    parser.add_argument(
        "--labels-json",
        type=Path,
        default=Path(
            "results/stage06_labeling_openrouter/"
            "labels_pos_openrouter_mistralai_mistral-nemo_romance_aware_paraphrase-MiniLM-L6-v2_reasoning_high.json"
        ),
        help="Path to labels JSON file",
    )
    parser.add_argument(
        "--expected-topics",
        type=int,
        default=361,
        help="Expected number of topics (default: 361)",
    )
    parser.add_argument(
        "--force-reload-labels",
        action="store_true",
        help="Force reload labels even if model already has custom_labels_",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage06_labeling/category_mapping/stage1_natural_clusters"),
        help="Output directory for logs and reports",
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
    logger.info("NOTE: model_1_with_categories should already have labels integrated")
    logger.info("      (stored in custom_labels_ and persisted in topics.json)")
    logger.info("      This script verifies the model state.")
    logger.info("=" * 80)
    logger.info(f"Model base directory: {args.base_dir}")
    logger.info(f"Embedding model: {args.embedding_model}")
    logger.info(f"Model suffix: {args.model_suffix}")
    logger.info(f"Labels JSON: {args.labels_json}")
    logger.info(f"Expected topics: {args.expected_topics}")
    
    # Load model
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading BERTopic Model")
    logger.info("=" * 80)
    
    if args.model_path:
        logger.info(f"Loading model from explicit path: {args.model_path}")
        topic_model = BERTopic.load(args.model_path)
    else:
        logger.info(f"Loading model using helper function (suffix: {args.model_suffix})")
        topic_model = load_native_bertopic_model(
            base_dir=args.base_dir,
            embedding_model=args.embedding_model,
            pareto_rank=1,
            model_suffix=args.model_suffix,
        )
    
    logger.info("✓ Model loaded successfully")
    
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
        
        labels = load_labels_from_json(args.labels_json, logger=logger)
        attach_labels_to_model(topic_model, labels, logger=logger)
        
        # Re-verify after attaching
        verification = verify_model_topics(topic_model, expected_count=args.expected_topics, logger=logger)
    
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

