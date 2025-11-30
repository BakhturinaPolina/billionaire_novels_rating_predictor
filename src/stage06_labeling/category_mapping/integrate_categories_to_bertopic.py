"""Integrate category mappings into BERTopic model for probability calculations per book/chapter."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

from bertopic import BERTopic

from src.stage06_BERTopic_topics_exploration.explore_retrained_model import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    load_native_bertopic_model,
    load_retrained_wrapper,
)
from src.stage06_labeling.generate_labels import integrate_labels_to_bertopic
LOGGER = logging.getLogger("stage06_category_mapping.bertopic_integration")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)


def load_category_mappings(
    category_probs_path: Path,
    labels_path: Path,
) -> Dict[str, Dict]:
    """
    Load category mappings and labels.
    
    Args:
        category_probs_path: Path to topic_to_category_probs.json
        labels_path: Path to labels JSON file (for topic labels)
    
    Returns:
        Dictionary mapping topic_id (str) to dict with:
        - categories: {cat: weight}
        - label: topic label
        - keywords: list of keywords
    """
    with open(category_probs_path, "r", encoding="utf-8") as f:
        category_data = json.load(f)
    
    with open(labels_path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)
    
    # Combine data
    combined = {}
    for topic_id in category_data.keys():
        cat_entry = category_data.get(topic_id, {})
        label_entry = labels_data.get(topic_id, {})
        
        # Handle both old format (direct dict) and new format (with "categories" key)
        if isinstance(cat_entry, dict) and "categories" in cat_entry:
            categories = cat_entry["categories"]
            label = cat_entry.get("label", label_entry.get("label", ""))
            keywords = cat_entry.get("keywords", label_entry.get("keywords", []))
        else:
            # Old format: cat_entry is directly the categories dict
            categories = cat_entry
            label = label_entry.get("label", "")
            keywords = label_entry.get("keywords", [])
        
        combined[topic_id] = {
            "categories": categories,
            "label": label,
            "keywords": keywords,
        }
    
    return combined


def create_enhanced_topic_labels(
    category_mappings: Dict[str, Dict],
    include_categories: bool = True,
    max_categories: int = 2,
) -> Dict[int, str]:
    """
    Create enhanced topic labels that include category information.
    
    Args:
        category_mappings: Dictionary from load_category_mappings()
        include_categories: If True, append category codes to labels
        max_categories: Maximum number of categories to include in label
    
    Returns:
        Dictionary mapping topic_id (int) to enhanced label string
    """
    enhanced_labels = {}
    
    for topic_id_str, data in category_mappings.items():
        try:
            topic_id = int(topic_id_str)
        except ValueError:
            continue
        
        if topic_id == -1:
            continue  # Skip outlier topic
        
        label = data.get("label", f"Topic {topic_id}")
        categories = data.get("categories", {})
        
        if include_categories and categories:
            # Get top categories by weight
            sorted_cats = sorted(
                categories.items(),
                key=lambda x: x[1],
                reverse=True
            )[:max_categories]
            
            # Format category codes (remove prefix if desired)
            cat_codes = [cat.replace("_", " ").title() for cat, _ in sorted_cats]
            cat_str = " | ".join(cat_codes)
            
            enhanced_label = f"{label} [{cat_str}]"
        else:
            enhanced_label = label
        
        enhanced_labels[topic_id] = enhanced_label
    
    return enhanced_labels


def update_topic_representations_with_categories(
    topic_model: BERTopic,
    category_mappings: Dict[str, Dict],
) -> None:
    """
    Update topic representations to include category tags.
    
    This adds category information to topic_representations_ so it appears
    in topic info and visualizations.
    
    Args:
        topic_model: BERTopic model instance
        category_mappings: Dictionary from load_category_mappings()
    """
    if not hasattr(topic_model, "topic_representations_"):
        LOGGER.warning("Model has no topic_representations_, skipping update")
        return
    
    updated_count = 0
    for topic_id_str, data in category_mappings.items():
        try:
            topic_id = int(topic_id_str)
        except ValueError:
            continue
        
        if topic_id == -1:
            continue
        
        if topic_id not in topic_model.topic_representations_:
            continue
        
        categories = data.get("categories", {})
        if not categories:
            continue
        
        # Get top category
        top_cat = max(categories.items(), key=lambda x: x[1])[0] if categories else None
        
        if top_cat:
            # Add category tag to topic representation
            # Format: "category: CAT_CODE" as a pseudo-keyword
            current_repr = topic_model.topic_representations_[topic_id]
            
            # Create new representation with category tag
            # BERTopic stores as list of (word, score) tuples
            if isinstance(current_repr, list):
                # Add category tag at the beginning
                cat_tag = (f"category:{top_cat}", 1.0)
                new_repr = [cat_tag] + current_repr
                topic_model.topic_representations_[topic_id] = new_repr
                updated_count += 1
    
    LOGGER.info(f"Updated {updated_count} topic representations with category tags")


def integrate_categories_to_bertopic(
    topic_model: BERTopic,
    category_mappings: Dict[str, Dict],
    update_labels: bool = True,
    update_representations: bool = True,
) -> None:
    """
    Integrate category mappings into BERTopic model.
    
    This function:
    1. Updates topic labels to include category information
    2. Optionally updates topic representations with category tags
    
    Args:
        topic_model: BERTopic model instance
        category_mappings: Dictionary from load_category_mappings()
        update_labels: If True, update topic labels with category info
        update_representations: If True, add category tags to representations
    """
    if update_labels:
        LOGGER.info("Creating enhanced topic labels with categories...")
        enhanced_labels = create_enhanced_topic_labels(
            category_mappings,
            include_categories=True,
            max_categories=2,
        )
        
        LOGGER.info(f"Integrating {len(enhanced_labels)} enhanced labels into BERTopic...")
        integrate_labels_to_bertopic(topic_model, enhanced_labels)
        LOGGER.info("✓ Enhanced labels integrated")
    
    if update_representations:
        LOGGER.info("Updating topic representations with category tags...")
        update_topic_representations_with_categories(topic_model, category_mappings)
        LOGGER.info("✓ Topic representations updated")


def save_bertopic_model(
    topic_model: BERTopic,
    output_dir: Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> None:
    """
    Save BERTopic model with updated category information.
    
    Args:
        topic_model: BERTopic model instance
        output_dir: Directory to save model
        embedding_model: Embedding model name for saving
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert embedding model name to sentence-transformers format if needed
    embedding_model_path = embedding_model
    if not embedding_model_path.startswith("sentence-transformers/"):
        embedding_model_path = f"sentence-transformers/{embedding_model}"
    
    LOGGER.info(f"Saving BERTopic model to {output_dir}...")
    topic_model.save(
        str(output_dir),
        serialization="safetensors",
        save_embedding_model=embedding_model_path,
        save_ctfidf=True,
    )
    LOGGER.info("✓ Model saved successfully")


def main():
    """Main entry point for integrating categories into BERTopic."""
    parser = argparse.ArgumentParser(
        description="Integrate category mappings into BERTopic model"
    )
    parser.add_argument(
        "--category-probs",
        type=Path,
        required=True,
        help="Path to topic_to_category_probs.json",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to labels JSON file (from openrouter experiments)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Path to BERTopic model directory (safetensors format)",
        default=None,
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
        "--pareto-rank",
        type=int,
        default=1,
        help="Model pareto rank",
    )
    parser.add_argument(
        "--use-native",
        action="store_true",
        help="Load native safetensors instead of pickle wrapper",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for saved model (default: overwrite original)",
        default=None,
    )
    parser.add_argument(
        "--no-update-labels",
        action="store_true",
        help="Skip updating topic labels",
    )
    parser.add_argument(
        "--no-update-representations",
        action="store_true",
        help="Skip updating topic representations",
    )
    
    args = parser.parse_args()
    
    # Load category mappings
    LOGGER.info("Loading category mappings...")
    category_mappings = load_category_mappings(
        args.category_probs,
        args.labels,
    )
    LOGGER.info(f"Loaded category mappings for {len(category_mappings)} topics")
    
    # Load BERTopic model
    if args.model_dir:
        LOGGER.info(f"Loading BERTopic model from {args.model_dir}...")
        topic_model = BERTopic.load(str(args.model_dir))
    elif args.use_native:
        LOGGER.info("Loading BERTopic model (native safetensors)...")
        topic_model = load_native_bertopic_model(
            base_dir=args.base_dir,
            embedding_model=args.embedding_model,
            pareto_rank=args.pareto_rank,
        )
    else:
        LOGGER.info("Loading BERTopic model (pickle wrapper)...")
        _, topic_model = load_retrained_wrapper(
            base_dir=args.base_dir,
            embedding_model=args.embedding_model,
            pareto_rank=args.pareto_rank,
        )
    
    LOGGER.info("✓ BERTopic model loaded")
    
    # Integrate categories
    integrate_categories_to_bertopic(
        topic_model,
        category_mappings,
        update_labels=not args.no_update_labels,
        update_representations=not args.no_update_representations,
    )
    
    # Save model
    if args.output_dir:
        save_bertopic_model(
            topic_model,
            args.output_dir,
            args.embedding_model,
        )
    else:
        # Overwrite original model
        if args.model_dir:
            output_dir = args.model_dir
        else:
            output_dir = (
                args.base_dir
                / args.embedding_model
                / f"model_{args.pareto_rank}"
            )
        
        LOGGER.info(f"Saving updated model to {output_dir}...")
        save_bertopic_model(
            topic_model,
            output_dir,
            args.embedding_model,
        )
    
    LOGGER.info("✓ Category integration complete!")


if __name__ == "__main__":
    main()

