"""Compare multiple OpenRouter models for topic labeling.

This script runs labeling with multiple literary/roleplay models and generates a comparison
output for manual inspection.

Why roleplay/story models excel at literary tasks:
Models fine-tuned for roleplay and story-writing are often excellent at literary analysis
because they've been trained extensively on fiction, dialogue, and narrative coherence,
not just "helpful assistant" style. They understand plot/scene patterns, genre tropes,
character roles, and fine-grained scene distinctions.

The key is selecting models that still obey instructions and JSON formatting.

Default models compared:
- mistralai/Mistral-Nemo-Instruct-2407: Baseline instruct model
- thedrummer/rocinante-12b: Literary/RP model designed for engaging storytelling and rich prose
- thedrummer/cydonia-24b-v4.1: 24B model
- thedrummer/anubis-70b-v1.1: 70B model
- thedrummer/skyfall-36b-v2: 36B model
- thedrummer/unslopnemo-12b: 12B model

See MODEL_AND_PROMPT_REASONING.md for detailed model information.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common.config import load_config, resolve_path
from src.common.logging import setup_logging
from src.stage06_topic_exploration.explore_retrained_model import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDING_MODEL,
)
from src.stage08_llm_labeling.generate_labels import (
    extract_pos_topics,
    extract_pos_topics_from_json,
    load_bertopic_model,
)
from src.stage08_llm_labeling.openrouter_experiments.core.generate_labels_openrouter import (
    DEFAULT_OPENROUTER_API_KEY,
    extract_representative_docs_per_topic,
    generate_labels_streaming,
    load_openrouter_client,
)

# Models to compare for literary / genre-aware labelling.
# All of these are accessed via OpenRouter, using a single API key.
#
# Why roleplay/story models excel at literary tasks:
# Models fine-tuned for roleplay and story-writing are often excellent at literary analysis
# because they've been trained extensively on fiction, dialogue, and narrative coherence,
# not just "helpful assistant" style. They understand:
# - Plot/scene patterns and narrative structure
# - Genre tropes and thematic coherence
# - Character roles and relationship dynamics
# - Scene-level distinctions (e.g., "Rough Angry Kisses" vs "Gentle Comforting Kisses")
#
# The key is selecting models that still obey instructions and JSON formatting,
# not just "sexy gremlin RP" models that ignore constraints.
ALL_MODELS_TO_COMPARE = [
    # Mistral Nemo: Baseline instruct model
    # Model: mistralai/Mistral-Nemo-Instruct-2407 (OpenRouter ID)
    "mistralai/Mistral-Nemo-Instruct-2407",
    # Rocinante: Literary/RP model designed for engaging storytelling and rich prose
    # Model: thedrummer/rocinante-12b (OpenRouter ID)
    "thedrummer/rocinante-12b",
    # Cydonia: 24B model
    # Model: thedrummer/cydonia-24b-v4.1 (OpenRouter ID)
    "thedrummer/cydonia-24b-v4.1",
    # Anubis: 70B model
    # Model: thedrummer/anubis-70b-v1.1 (OpenRouter ID)
    "thedrummer/anubis-70b-v1.1",
    # Skyfall: 36B model
    # Model: thedrummer/skyfall-36b-v2 (OpenRouter ID)
    "thedrummer/skyfall-36b-v2",
    # Unslopnemo: 12B model
    # Model: thedrummer/unslopnemo-12b (OpenRouter ID)
    "thedrummer/unslopnemo-12b",
]

DEFAULT_OUTPUT_DIR = Path("results/stage08_llm_labeling")
DEFAULT_NUM_KEYWORDS = 15
DEFAULT_MAX_TOKENS = 16  # Only need 2–6 words
DEFAULT_BATCH_SIZE = 50
DEFAULT_TEMPERATURE = 0.15  # Lower → less creativity, more deterministic
DEFAULT_TOPIC_LIMIT = 30  # For initial testing


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare multiple OpenRouter models for topic labeling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name (e.g., 'paraphrase-MiniLM-L6-v2')",
    )
    
    parser.add_argument(
        "--pareto-rank",
        type=int,
        default=1,
        help="Pareto rank of the model to load",
    )
    
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base directory containing retrained models",
    )
    
    parser.add_argument(
        "--use-native",
        action="store_true",
        help="Load native safetensors instead of pickle wrapper",
    )
    
    parser.add_argument(
        "--model-suffix",
        type=str,
        default="_with_noise_labels",
        help="Optional suffix to append to model filename/directory (default: '_with_noise_labels' to use model with noise labels)",
    )
    
    parser.add_argument(
        "--model-stage",
        type=str,
        default="stage07_topic_quality",
        help="Stage subfolder to load model from (default: 'stage07_topic_quality' to use model from stage07)",
    )
    
    parser.add_argument(
        "--num-keywords",
        type=int,
        default=DEFAULT_NUM_KEYWORDS,
        help=f"Number of top keywords per topic (default: {DEFAULT_NUM_KEYWORDS})",
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens per label (default: {DEFAULT_MAX_TOKENS})",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for comparison files",
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_OPENROUTER_API_KEY,
        help="OpenRouter API key (default: from OPENROUTER_API_KEY environment variable)",
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=ALL_MODELS_TO_COMPARE,
        help=f"List of models to compare (default: {ALL_MODELS_TO_COMPARE})",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of topics per progress log",
    )
    
    parser.add_argument(
        "--use-improved-prompts",
        action="store_true",
        help="Use JSON romance-aware prompts and structured JSON parsing for labels",
    )
    
    parser.add_argument(
        "--topics-json",
        type=Path,
        default=None,
        help="Path to topics JSON file (for streaming mode)",
    )
    
    parser.add_argument(
        "--limit-topics",
        type=int,
        default=DEFAULT_TOPIC_LIMIT,
        help=f"Limit number of topics to process per model (default: {DEFAULT_TOPIC_LIMIT})",
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paths.yaml"),
        help="Path to paths configuration file",
    )
    
    return parser.parse_args()


def generate_comparison_csv(
    all_results: dict[str, dict[int, dict[str, Any]]],
    output_path: Path,
) -> None:
    """Generate CSV comparison file with all models side-by-side."""
    # Collect all topic IDs
    all_topic_ids = set()
    for model_results in all_results.values():
        all_topic_ids.update(model_results.keys())
    all_topic_ids = sorted(all_topic_ids)
    
    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header row: Topic ID, Keywords, then one column per model
        header = ["Topic ID", "Keywords"]
        for model_name in all_results.keys():
            header.append(f"{model_name} Label")
        writer.writerow(header)
        
        # Data rows
        for topic_id in all_topic_ids:
            # Get keywords from first model (should be same for all)
            keywords = []
            for model_results in all_results.values():
                if topic_id in model_results:
                    keywords = model_results[topic_id].get("keywords", [])
                    break
            
            row = [
                topic_id,
                ", ".join(keywords[:10]),  # Limit to first 10 for readability
            ]
            
            # Add label from each model
            for model_name in all_results.keys():
                if topic_id in all_results[model_name]:
                    label = all_results[model_name][topic_id].get("label", "")
                    row.append(label)
                else:
                    row.append("")
            
            writer.writerow(row)
    
    print(f"✓ Comparison CSV saved to: {output_path}")


def generate_comparison_json(
    all_results: dict[str, dict[int, dict[str, Any]]],
    output_path: Path,
) -> None:
    """Generate JSON comparison file with structured format."""
    # Model descriptions for metadata
    model_descriptions = {
        "mistralai/Mistral-Nemo-Instruct-2407": "Baseline Nemo Instruct model",
        "thedrummer/rocinante-12b": "Rocinante: Literary/RP model designed for engaging storytelling and rich prose",
        "thedrummer/cydonia-24b-v4.1": "Cydonia: 24B model",
        "thedrummer/anubis-70b-v1.1": "Anubis: 70B model",
        "thedrummer/skyfall-36b-v2": "Skyfall: 36B model",
        "thedrummer/unslopnemo-12b": "Unslopnemo: 12B model",
    }
    
    comparison_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "models": list(all_results.keys()),
            "model_descriptions": {
                model: model_descriptions.get(model, "Custom model")
                for model in all_results.keys()
            },
            "total_topics": len(set().union(*[r.keys() for r in all_results.values()])),
            "note": (
                "Roleplay/story models excel at literary tasks because they're trained on "
                "fiction, dialogue, and narrative coherence. They understand scene patterns, "
                "genre tropes, and fine-grained distinctions better than general-purpose models."
            ),
        },
        "topics": {},
    }
    
    # Collect all topic IDs
    all_topic_ids = set()
    for model_results in all_results.values():
        all_topic_ids.update(model_results.keys())
    
    # Build comparison structure
    for topic_id in sorted(all_topic_ids):
        topic_data = {
            "topic_id": topic_id,
            "keywords": [],
            "models": {},  # Changed from "labels" to "models" to store full model results
        }
        
        # Get keywords from first model
        for model_results in all_results.values():
            if topic_id in model_results:
                topic_data["keywords"] = model_results[topic_id].get("keywords", [])
                break
        
        # Get full model results (label, scene_summary, categories, is_noise, rationale, etc.)
        for model_name, model_results in all_results.items():
            if topic_id in model_results:
                model_data = model_results[topic_id]
                # Store all fields from model result (label, scene_summary, categories, etc.)
                topic_data["models"][model_name] = {
                    "label": model_data.get("label", ""),
                    "scene_summary": model_data.get("scene_summary", ""),
                    "primary_categories": model_data.get("primary_categories", []),
                    "secondary_categories": model_data.get("secondary_categories", []),
                    "is_noise": model_data.get("is_noise", False),
                    "rationale": model_data.get("rationale", ""),
                }
        
        comparison_data["topics"][str(topic_id)] = topic_data
    
    # Write JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Comparison JSON saved to: {output_path}")


def main() -> None:
    """Main entry point for multi-model comparison."""
    args = parse_args()
    
    # Validate API key
    if not args.api_key or args.api_key.strip() == "":
        print("=" * 80)
        print("ERROR: OpenRouter API key is required")
        print("=" * 80)
        print("Please provide your OpenRouter API key in one of the following ways:")
        print("  1. Set environment variable: export OPENROUTER_API_KEY='your-key-here'")
        print("  2. Pass via command line: --api-key 'your-key-here'")
        print()
        print("You can get an API key from: https://openrouter.ai/keys")
        print("=" * 80)
        return
    
    # Load configuration
    try:
        paths_cfg = load_config(args.config)
        outputs = paths_cfg.get("outputs", {})
        logs_dir = resolve_path(Path(outputs.get("logs", "logs")))
    except Exception as e:
        logs_dir = Path("logs")
        print(f"Warning: Could not load config: {e}")
    
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"compare_models_openrouter_{timestamp}.log"
    logger = setup_logging(logs_dir, log_file=log_file)
    
    print("=" * 80)
    print("Multi-Model Comparison: OpenRouter Models")
    print("=" * 80)
    print(f"Models to compare: {len(args.models)}")
    print()
    
    # Model descriptions for user reference
    model_descriptions = {
        "mistralai/Mistral-Nemo-Instruct-2407": "Baseline Nemo Instruct model",
        "thedrummer/rocinante-12b": "Rocinante: Literary/RP model designed for engaging storytelling and rich prose",
        "thedrummer/cydonia-24b-v4.1": "Cydonia: 24B model",
        "thedrummer/anubis-70b-v1.1": "Anubis: 70B model",
        "thedrummer/skyfall-36b-v2": "Skyfall: 36B model",
        "thedrummer/unslopnemo-12b": "Unslopnemo: 12B model",
    }
    
    for i, model in enumerate(args.models, 1):
        description = model_descriptions.get(model, "Custom model")
        print(f"  {i}. {model}")
        print(f"     → {description}")
    print()
    print("Note: Roleplay/story models excel at literary tasks because they're trained on")
    print("      fiction, dialogue, and narrative coherence. They understand scene patterns,")
    print("      genre tropes, and fine-grained distinctions better than general-purpose models.")
    print()
    print(f"Topics per model: {args.limit_topics}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Step 1: Load BERTopic model
    print("Step 1: Loading BERTopic model...")
    wrapper, topic_model = load_bertopic_model(
        base_dir=args.base_dir,
        embedding_model=args.embedding_model,
        pareto_rank=args.pareto_rank,
        use_native=args.use_native,
        model_suffix=args.model_suffix,
        stage_subfolder=args.model_stage,
    )
    print(f"✓ Loaded BERTopic model")
    print()
    
    # Step 2: Extract topics
    print("Step 2: Extracting POS topics...")
    if args.topics_json and args.topics_json.exists():
        pos_topics_iter = extract_pos_topics_from_json(
            json_path=args.topics_json,
            top_k=args.num_keywords,
        )
        pos_topics_dict = dict(pos_topics_iter)
        pos_topics_iter = iter(pos_topics_dict.items())  # Recreate iterator
        print(f"✓ Loaded {len(pos_topics_dict)} topics from JSON")
    else:
        pos_topics_dict = extract_pos_topics(
            topic_model=topic_model,
            top_k=args.num_keywords,
        )
        pos_topics_iter = iter(pos_topics_dict.items())
        print(f"✓ Extracted {len(pos_topics_dict)} topics from BERTopic model")
    print()
    
    # Step 3: Extract representative docs
    print("Step 3: Extracting representative documents...")
    topic_to_snippets = extract_representative_docs_per_topic(topic_model)
    print(f"✓ Extracted snippets for {len([tid for tid, docs in topic_to_snippets.items() if docs])} topics")
    print()
    
    # Step 5: Generate labels for each model
    all_results: dict[str, dict[int, dict[str, Any]]] = {}
    model_name_safe = args.embedding_model.replace("/", "_").replace("\\", "_")
    
    for model_idx, model_name in enumerate(args.models, 1):
        print("=" * 80)
        print(f"Processing Model {model_idx}/{len(args.models)}: {model_name}")
        print("=" * 80)
        
        # Initialize OpenRouter client for THIS specific model (fresh client per model)
        print(f"Initializing OpenRouter client for {model_name}...")
        client, _ = load_openrouter_client(
            api_key=args.api_key,
            model_name=model_name,
        )
        print(f"✓ OpenRouter client initialized for {model_name}")
        print()
        
        # Create model-specific output path
        model_name_file = model_name.replace("/", "_").replace(":", "_")
        labels_filename = f"labels_pos_openrouter_{model_name_file}_{model_name_safe}"
        labels_path = args.output_dir / labels_filename
        
        # Recreate iterator for this model (fresh iterator per model)
        if args.topics_json and args.topics_json.exists():
            pos_topics_iter = extract_pos_topics_from_json(
                json_path=args.topics_json,
                top_k=args.num_keywords,
            )
        else:
            pos_topics_iter = iter(pos_topics_dict.items())
        
        # Generate labels (using romance-aware prompts with snippets if --use-improved-prompts is set)
        print(f"Generating labels (limit: {args.limit_topics} topics)...")
        topic_labels = generate_labels_streaming(
            pos_topics_iter=pos_topics_iter,
            client=client,
            model_name=model_name,
            output_path=labels_path,
            max_new_tokens=args.max_tokens,
            batch_size=args.batch_size,
            temperature=args.temperature,
            limit=args.limit_topics,
            use_improved_prompts=args.use_improved_prompts,
            topic_model=topic_model,
            topic_to_snippets=topic_to_snippets,
        )
        
        print(f"✓ Generated {len(topic_labels)} labels for {model_name}")
        all_results[model_name] = topic_labels
        print()
    
    # Step 6: Generate comparison files
    print("=" * 80)
    print("Step 6: Generating comparison files...")
    print("=" * 80)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV comparison
    csv_path = args.output_dir / f"comparison_models_{timestamp_str}.csv"
    generate_comparison_csv(all_results, csv_path)
    
    # JSON comparison
    json_path = args.output_dir / f"comparison_models_{timestamp_str}.json"
    generate_comparison_json(all_results, json_path)
    
    print()
    print("=" * 80)
    print("Comparison Complete")
    print("=" * 80)
    print(f"Models compared: {len(all_results)}")
    print(f"Topics per model: {args.limit_topics}")
    print(f"CSV output: {csv_path}")
    print(f"JSON output: {json_path}")
    print()
    print("Individual model outputs:")
    for model_name in all_results.keys():
        model_name_file = model_name.replace("/", "_").replace(":", "_")
        labels_filename = f"labels_pos_openrouter_{model_name_file}_{model_name_safe}.json"
        labels_path = args.output_dir / labels_filename
        print(f"  {model_name}: {labels_path}")


if __name__ == "__main__":
    main()

