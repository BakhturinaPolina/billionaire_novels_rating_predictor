#!/usr/bin/env python
"""Inspect comparison of multiple OpenRouter models for a small set of topics.

Usage:
    python inspect_comparison_results.py \
        --comparison-json results/stage08_llm_labeling/comparison_models_2025....json \
        --topics 3 7 15

This will print, for each topic:
- top keywords
- then per model: label, scene_summary, categories, is_noise, rationale

The script helps you quickly compare how different models (Nemo-Instruct, Gutenberg, Celeste)
label the same topics, making it easier to evaluate which model works best for your research.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect multi-model comparison results for selected topics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--comparison-json",
        type=Path,
        required=True,
        help="Path to comparison JSON file produced by compare_models_openrouter.py",
    )
    parser.add_argument(
        "--topics",
        type=int,
        nargs="+",
        required=True,
        help="Topic IDs to inspect (e.g. 3 7 15).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.comparison_json.exists():
        raise FileNotFoundError(f"Comparison JSON not found: {args.comparison_json}")

    with open(args.comparison_json, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    metadata = data.get("metadata", {})
    models = metadata.get("models", [])
    model_descriptions = metadata.get("model_descriptions", {})
    topics = data.get("topics", {})

    print("=" * 80)
    print("INSPECTING TOPICS:")
    print("Requested topic IDs:", ", ".join(str(t) for t in args.topics))
    print("Models:", ", ".join(models))
    if model_descriptions:
        print("\nModel descriptions:")
        for model in models:
            desc = model_descriptions.get(model, "Custom model")
            print(f"  - {model}: {desc}")
    print("=" * 80)
    print()

    for topic_id in args.topics:
        topic_key = str(topic_id)
        topic_data = topics.get(topic_key)

        if not topic_data:
            print(f"[Topic {topic_id}] Not found in comparison JSON.")
            print("-" * 80)
            continue

        keywords = topic_data.get("keywords", [])
        print(f"[Topic {topic_id}]")
        print(f"  Keywords: {', '.join(keywords[:15])}")
        if len(keywords) > 15:
            print(f"            ... ({len(keywords) - 15} more)")
        print()

        # Check if we have the new structure (models dict) or old structure (labels dict)
        models_data = topic_data.get("models", {})
        labels_data = topic_data.get("labels", {})

        if models_data:
            # New structure: full model data
            for model_name in models:
                model_result = models_data.get(model_name, {})
                if not model_result:
                    print(f"  --- Model: {model_name} ---")
                    print(f"  (No data for this model)")
                    print()
                    continue

                label = model_result.get("label", "")
                scene_summary = model_result.get("scene_summary", "")
                primary = model_result.get("primary_categories", [])
                secondary = model_result.get("secondary_categories", [])
                is_noise = model_result.get("is_noise", False)
                rationale = model_result.get("rationale", "")

                print(f"  --- Model: {model_name} ---")
                print(f"  Label:          {label}")
                if scene_summary:
                    print(f"  Scene summary:  {scene_summary}")
                if primary:
                    print(f"  Primary cat.:   {', '.join(primary)}")
                if secondary:
                    print(f"  Secondary cat.: {', '.join(secondary)}")
                print(f"  Is noise:       {is_noise}")
                if rationale:
                    # Truncate very long rationales just for readability
                    short_rat = rationale.strip()
                    if len(short_rat) > 300:
                        short_rat = short_rat[:297] + "..."
                    print(f"  Rationale:      {short_rat}")
                print()
        elif labels_data:
            # Old structure: just labels
            for model_name in models:
                label = labels_data.get(model_name, "")
                print(f"  --- Model: {model_name} ---")
                print(f"  Label: {label}")
                print()
        else:
            print("  (No model data found for this topic)")
            print()

        print("-" * 80)
        print()


if __name__ == "__main__":
    main()
