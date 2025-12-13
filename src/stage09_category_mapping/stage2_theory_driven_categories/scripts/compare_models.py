"""Compare models from stage08_llm_labeling and stage09_category_mapping.

This script analyzes models from both stages to determine which is most complete
for use in taxonomy classification and statistical analysis.

It checks:
- Number of topics
- Presence of labels (custom_labels_)
- Presence of taxonomy mappings (topic_metadata_ or JSON files)
- Completeness of metadata
- Model file sizes and timestamps
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bertopic import BERTopic

from src.stage06_topic_exploration.explore_retrained_model import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDING_MODEL,
)

LOGGER = logging.getLogger("compare_models")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)


def load_model_safely(model_path: Path) -> Optional[BERTopic]:
    """Load BERTopic model from .pkl file or directory, handling wrappers."""
    if not model_path.exists():
        LOGGER.warning(f"Model path does not exist: {model_path}")
        return None

    try:
        # Try loading from directory first
        if model_path.is_dir():
            LOGGER.info(f"Loading model from directory: {model_path}")
            return BERTopic.load(str(model_path))
        
        # Try loading from .pkl file
        if model_path.suffix == ".pkl":
            LOGGER.info(f"Loading model from .pkl file: {model_path}")
            with open(model_path, "rb") as f:
                loaded_obj = pickle.load(f)
            
            # Check if it's a RetrainableBERTopicModel wrapper
            if hasattr(loaded_obj, "trained_topic_model") and loaded_obj.trained_topic_model is not None:
                LOGGER.info("  Extracted BERTopic model from RetrainableBERTopicModel wrapper")
                return loaded_obj.trained_topic_model
            elif isinstance(loaded_obj, BERTopic):
                return loaded_obj
            else:
                # Try BERTopic.load() as fallback
                return BERTopic.load(str(model_path))
        
        # Try loading as directory
        return BERTopic.load(str(model_path))
        
    except Exception as e:
        LOGGER.error(f"Failed to load model from {model_path}: {e}")
        return None


def get_model_info(model: BERTopic, model_path: Path) -> Dict[str, Any]:
    """Extract information about a BERTopic model."""
    info: Dict[str, Any] = {
        "path": str(model_path),
        "file_size_mb": model_path.stat().st_size / (1024 * 1024) if model_path.exists() else None,
        "num_topics": 0,
        "has_labels": False,
        "num_labels": 0,
        "has_taxonomy_metadata": False,
        "num_taxonomy_mappings": 0,
        "topic_ids": [],
    }
    
    # Count topics (excluding outlier -1)
    if hasattr(model, "topic_representations_"):
        topic_ids = [tid for tid in model.topic_representations_.keys() if tid != -1]
        info["num_topics"] = len(topic_ids)
        info["topic_ids"] = sorted(topic_ids)
    
    # Check for labels
    if hasattr(model, "custom_labels_") and model.custom_labels_:
        info["has_labels"] = True
        info["num_labels"] = len(model.custom_labels_)
    
    # Check for taxonomy metadata
    if hasattr(model, "topic_metadata_") and model.topic_metadata_:
        info["has_taxonomy_metadata"] = True
        info["num_taxonomy_mappings"] = len(model.topic_metadata_)
    
    return info


def find_models_in_stage(
    stage_dir: Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> List[Tuple[Path, str]]:
    """Find all model files in a stage directory."""
    models = []
    
    if not stage_dir.exists():
        return models
    
    # Look for .pkl files
    for pkl_file in stage_dir.glob("*.pkl"):
        # Skip backup files
        if "backup" in pkl_file.name.lower():
            continue
        models.append((pkl_file, pkl_file.name))
    
    # Look for model directories (without .pkl extension)
    for model_dir in stage_dir.iterdir():
        if model_dir.is_dir() and not model_dir.name.startswith("."):
            # Check if it's a BERTopic model directory
            if (model_dir / "config.json").exists() or (model_dir / "topic_representations.json").exists():
                models.append((model_dir, model_dir.name))
    
    return models


def load_taxonomy_mapping_json(json_path: Path) -> Optional[Dict[str, Any]]:
    """Load taxonomy mapping JSON file if it exists."""
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        LOGGER.warning(f"Failed to load taxonomy JSON from {json_path}: {e}")
        return None


def find_taxonomy_json_files(results_dir: Path) -> List[Tuple[Path, str]]:
    """Find taxonomy mapping JSON files in results directory."""
    json_files = []
    
    if not results_dir.exists():
        return json_files
    
    # Look for taxonomy_mappings_*.json files
    for json_file in results_dir.glob("**/taxonomy_mappings*.json"):
        json_files.append((json_file, json_file.name))
    
    return json_files


def compare_models(
    base_dir: Path = DEFAULT_BASE_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    results_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Compare models from stage08_llm_labeling and stage09_category_mapping.
    
    Parameters
    ----------
    base_dir:
        Base directory for models (default: models/retrained).
    embedding_model:
        Embedding model name (default: paraphrase-MiniLM-L6-v2).
    results_dir:
        Optional results directory to search for taxonomy JSON files.
        
    Returns
    -------
    Dict with comparison results and recommendations.
    """
    base_dir = Path(base_dir)
    stage08_dir = base_dir / embedding_model / "stage08_llm_labeling"
    stage09_dir = base_dir / embedding_model / "stage09_category_mapping"
    
    LOGGER.info("=" * 80)
    LOGGER.info("MODEL COMPARISON REPORT")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Base directory: {base_dir}")
    LOGGER.info(f"Embedding model: {embedding_model}")
    LOGGER.info("")
    
    # Find models in each stage
    LOGGER.info("Scanning stage08_llm_labeling...")
    stage08_models = find_models_in_stage(stage08_dir, embedding_model)
    LOGGER.info(f"  Found {len(stage08_models)} models")
    
    LOGGER.info("Scanning stage09_category_mapping...")
    stage09_models = find_models_in_stage(stage09_dir, embedding_model)
    LOGGER.info(f"  Found {len(stage09_models)} models")
    
    # Load and analyze models
    stage08_info: List[Dict[str, Any]] = []
    stage09_info: List[Dict[str, Any]] = []
    
    LOGGER.info("")
    LOGGER.info("Analyzing stage08_llm_labeling models...")
    for model_path, model_name in stage08_models:
        LOGGER.info(f"  Loading: {model_name}")
        model = load_model_safely(model_path)
        if model:
            info = get_model_info(model, model_path)
            info["model_name"] = model_name
            stage08_info.append(info)
            LOGGER.info(f"    Topics: {info['num_topics']}, Labels: {info['num_labels']}, Taxonomy: {info['num_taxonomy_mappings']}")
    
    LOGGER.info("")
    LOGGER.info("Analyzing stage09_category_mapping models...")
    for model_path, model_name in stage09_models:
        LOGGER.info(f"  Loading: {model_name}")
        model = load_model_safely(model_path)
        if model:
            info = get_model_info(model, model_path)
            info["model_name"] = model_name
            stage09_info.append(info)
            LOGGER.info(f"    Topics: {info['num_topics']}, Labels: {info['num_labels']}, Taxonomy: {info['num_taxonomy_mappings']}")
    
    # Find taxonomy JSON files
    taxonomy_jsons: List[Dict[str, Any]] = []
    if results_dir:
        LOGGER.info("")
        LOGGER.info("Scanning for taxonomy mapping JSON files...")
        json_files = find_taxonomy_json_files(results_dir)
        for json_path, json_name in json_files:
            data = load_taxonomy_mapping_json(json_path)
            if data:
                taxonomy_jsons.append({
                    "path": str(json_path),
                    "name": json_name,
                    "num_topics": len(data),
                })
                LOGGER.info(f"  Found: {json_name} ({len(data)} topics)")
    
    # Generate comparison report
    report = {
        "stage08_models": stage08_info,
        "stage09_models": stage09_info,
        "taxonomy_json_files": taxonomy_jsons,
        "recommendations": [],
    }
    
    # Determine best model for each stage
    best_stage08 = None
    best_stage09 = None
    
    if stage08_info:
        # Prefer models with labels and most topics
        best_stage08 = max(
            stage08_info,
            key=lambda x: (
                x["has_labels"],
                x["num_topics"],
                x.get("file_size_mb", 0) or 0,
            ),
        )
    
    if stage09_info:
        # Prefer models with taxonomy metadata and most topics
        best_stage09 = max(
            stage09_info,
            key=lambda x: (
                x["has_taxonomy_metadata"],
                x["has_labels"],
                x["num_topics"],
                x.get("file_size_mb", 0) or 0,
            ),
        )
    
    # Generate recommendations
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("RECOMMENDATIONS")
    LOGGER.info("=" * 80)
    
    if best_stage09 and best_stage09["has_taxonomy_metadata"]:
        LOGGER.info("✓ RECOMMENDED: Use stage09_category_mapping model")
        LOGGER.info(f"  Model: {best_stage09['model_name']}")
        LOGGER.info(f"  Path: {best_stage09['path']}")
        LOGGER.info(f"  Topics: {best_stage09['num_topics']}")
        LOGGER.info(f"  Has labels: {best_stage09['has_labels']}")
        LOGGER.info(f"  Has taxonomy mappings: {best_stage09['has_taxonomy_metadata']}")
        report["recommendations"].append({
            "stage": "stage09_category_mapping",
            "model": best_stage09["model_name"],
            "path": best_stage09["path"],
            "reason": "Has taxonomy metadata embedded in model",
        })
    elif best_stage08 and best_stage08["has_labels"]:
        LOGGER.info("⚠ RECOMMENDED: Use stage08_llm_labeling model + taxonomy JSON")
        LOGGER.info(f"  Model: {best_stage08['model_name']}")
        LOGGER.info(f"  Path: {best_stage08['path']}")
        LOGGER.info(f"  Topics: {best_stage08['num_topics']}")
        LOGGER.info(f"  Has labels: {best_stage08['has_labels']}")
        if taxonomy_jsons:
            LOGGER.info(f"  Taxonomy JSON: {taxonomy_jsons[0]['name']}")
            report["recommendations"].append({
                "stage": "stage08_llm_labeling",
                "model": best_stage08["model_name"],
                "path": best_stage08["path"],
                "taxonomy_json": taxonomy_jsons[0]["path"],
                "reason": "Has labels, use with separate taxonomy JSON file",
            })
        else:
            LOGGER.warning("  ⚠ No taxonomy JSON files found - need to run zeroshot_taxonomy_openrouter.py")
            report["recommendations"].append({
                "stage": "stage08_llm_labeling",
                "model": best_stage08["model_name"],
                "path": best_stage08["path"],
                "reason": "Has labels but no taxonomy mappings - need to run taxonomy classification",
            })
    else:
        LOGGER.warning("⚠ No suitable model found - check model files")
        report["recommendations"].append({
            "reason": "No complete model found - need to check model files",
        })
    
    # Check topic count consistency
    if best_stage08 and best_stage09:
        if best_stage08["num_topics"] != best_stage09["num_topics"]:
            LOGGER.warning("")
            LOGGER.warning("⚠ TOPIC COUNT MISMATCH:")
            LOGGER.warning(f"  Stage08: {best_stage08['num_topics']} topics")
            LOGGER.warning(f"  Stage09: {best_stage09['num_topics']} topics")
            report["warnings"] = [
                f"Topic count mismatch: stage08 has {best_stage08['num_topics']} topics, "
                f"stage09 has {best_stage09['num_topics']} topics"
            ]
    
    return report


def save_report(report: Dict[str, Any], output_path: Path) -> None:
    """Save comparison report to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Make report JSON-serializable
    serializable_report = {
        "stage08_models": [
            {k: v for k, v in m.items() if k != "topic_ids"}  # Exclude topic_ids for brevity
            for m in report["stage08_models"]
        ],
        "stage09_models": [
            {k: v for k, v in m.items() if k != "topic_ids"}
            for m in report["stage09_models"]
        ],
        "taxonomy_json_files": report["taxonomy_json_files"],
        "recommendations": report["recommendations"],
        "warnings": report.get("warnings", []),
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_report, f, indent=2, ensure_ascii=False)
    
    LOGGER.info("")
    LOGGER.info(f"Report saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare models from stage08_llm_labeling and stage09_category_mapping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base directory for models (default: models/retrained)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name (default: paraphrase-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Optional results directory to search for taxonomy JSON files",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional path to save JSON report",
    )
    
    args = parser.parse_args()
    
    # Default results directory
    if args.results_dir is None:
        project_root = Path(__file__).parent.parent.parent.parent.parent
        args.results_dir = project_root / "results" / "stage09_category_mapping" / "stage2_theory_driven_categories"
    
    report = compare_models(
        base_dir=args.base_dir,
        embedding_model=args.embedding_model,
        results_dir=args.results_dir,
    )
    
    if args.output_report:
        save_report(report, args.output_report)
    else:
        # Default output location
        project_root = Path(__file__).parent.parent.parent.parent.parent
        default_output = project_root / "results" / "stage09_category_mapping" / "stage2_theory_driven_categories" / "model_comparison_report.json"
        save_report(report, default_output)

