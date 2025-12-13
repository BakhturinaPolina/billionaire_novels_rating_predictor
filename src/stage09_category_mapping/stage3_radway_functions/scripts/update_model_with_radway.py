"""Update BERTopic model with Radway mappings from merged JSON file."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.stage09_category_mapping.stage3_radway_functions.scripts.zeroshot_radway_openrouter import (
    update_model_with_radway_mappings,
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDING_MODEL,
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Update BERTopic model with Radway mappings from merged JSON file."
    )

    parser.add_argument(
        "--merged-json",
        type=Path,
        required=True,
        help="Path to merged taxonomy + Radway mappings JSON file.",
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base directory for models.",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name.",
    )

    parser.add_argument(
        "--source-model-suffix",
        type=str,
        default="_with_taxonomy_mappings",
        help="Suffix of source model to load (default: _with_taxonomy_mappings, can also use _with_llm_labels_and_metadata_disambiguated).",
    )

    parser.add_argument(
        "--source-stage",
        type=str,
        default="stage09_category_mapping",
        help="Source stage subfolder.",
    )

    parser.add_argument(
        "--target-model-suffix",
        type=str,
        default="_with_radway_mappings",
        help="Suffix for the saved model.",
    )

    parser.add_argument(
        "--target-stage",
        type=str,
        default="stage09_category_mapping",
        help="Target stage subfolder.",
    )

    args = parser.parse_args()

    model_path = update_model_with_radway_mappings(
        merged_json_path=args.merged_json,
        base_dir=args.base_dir,
        embedding_model=args.embedding_model,
        model_suffix=args.source_model_suffix,
        source_stage_subfolder=args.source_stage,
        target_stage_subfolder=args.target_stage,
        target_model_suffix=args.target_model_suffix,
    )

    print(f"✓ Model updated and saved to: {model_path}")

