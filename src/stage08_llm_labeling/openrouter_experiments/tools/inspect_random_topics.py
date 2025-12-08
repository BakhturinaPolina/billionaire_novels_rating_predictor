"""Standalone script to inspect random topics with labels, summaries, keywords, and representative sentences.

This script provides a "human in the loop" tool for manually evaluating topic labels
by showing:
- Topic ID
- Label
- Scene summary
- Keywords
- Exact sentences from representative documents

Usage:
    python -m src.stage08_llm_labeling.openrouter_experiments.tools.inspect_random_topics \
        --bertopic-model-path models/retrained/model_1_with_noise_labels \
        --labels-json results/stage08_llm_labeling/labels_pos_openrouter_mistralai_mistral-nemo_romance_aware_paraphrase-MiniLM-L6-v2.json \
        --num-topics 15 \
        --num-snippets-to-show 8
"""

import argparse
import json
import random
from pathlib import Path

from bertopic import BERTopic

from src.stage08_llm_labeling.generate_labels import load_bertopic_model
from src.stage06_topic_exploration.explore_retrained_model import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDING_MODEL,
)
from src.stage08_llm_labeling.openrouter_experiments.core.generate_labels_openrouter import (
    extract_representative_docs_per_topic,
    rerank_snippets_centrality,
)


def inspect_random_topics(
    bertopic_model_path: str | None = None,
    labels_json_path: str = "",
    base_dir: Path | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    pareto_rank: int = 1,
    use_native: bool = False,
    model_suffix: str = "_with_noise_labels",
    num_topics: int = 15,
    max_docs_per_topic: int = 40,
    num_snippets_to_show: int = 8,
    use_centrality: bool = True,
    random_seed: int = 42,
    topic_ids: list[int] | None = None,
) -> None:
    """
    Print a sample of topics with:
      - topic id
      - label
      - scene_summary
      - keywords
      - exact sentences from representative docs

    for manual evaluation.

    Args:
        bertopic_model_path: Direct path to BERTopic model file (if provided, uses BERTopic.load()).
                            If None, uses load_bertopic_model() with other parameters.
        labels_json_path: Path to labels JSON file
        base_dir: Base directory for retrained models (used if bertopic_model_path is None)
        embedding_model: Embedding model name (used if bertopic_model_path is None)
        pareto_rank: Pareto rank of model (used if bertopic_model_path is None)
        use_native: Use native safetensors (used if bertopic_model_path is None)
        model_suffix: Model suffix (used if bertopic_model_path is None)
        num_topics: Number of random topics to sample
        max_docs_per_topic: Maximum representative docs per topic to load
        num_snippets_to_show: Number of snippets to display per topic
        use_centrality: If True, rerank snippets by semantic centrality
        random_seed: Random seed for topic sampling
    """
    random.seed(random_seed)

    labels_path = Path(labels_json_path)
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels JSON not found: {labels_path}")

    # Load labels
    print(f"Loading labels from: {labels_path}")
    with labels_path.open("r", encoding="utf-8") as f:
        labels_data = json.load(f)

    # Load BERTopic model
    if bertopic_model_path:
        # Direct path provided - try to use project's load_bertopic_model
        # This handles both pickle wrappers and native formats
        model_path = Path(bertopic_model_path)
        
        # Try to infer parameters from path structure
        # models/retrained/{embedding_model}/model_X_suffix.pkl
        # or models/retrained/model_X_suffix (without .pkl)
        import re
        
        # Check if it's a file or directory
        if model_path.is_file() or model_path.suffix == ".pkl":
            # It's a file - extract model name
            model_name = model_path.stem if model_path.suffix == ".pkl" else model_path.name
            base_dir_path = model_path.parent
        else:
            # It's a directory or path without extension
            model_name = model_path.name
            base_dir_path = model_path.parent
        
        # Try to infer embedding model from path
        # models/retrained/{embedding_model}/model_X.pkl
        if len(base_dir_path.parts) >= 2 and base_dir_path.parts[-2] == "retrained":
            inferred_embedding_model = base_dir_path.parts[-1]
            inferred_base_dir = base_dir_path.parent
        else:
            # Use defaults
            inferred_embedding_model = embedding_model
            inferred_base_dir = base_dir if base_dir else DEFAULT_BASE_DIR
        
        # Extract pareto rank and suffix from model name
        # model_1_with_noise_labels -> rank=1, suffix=_with_noise_labels
        rank_match = re.search(r"model_(\d+)", model_name)
        inferred_rank = int(rank_match.group(1)) if rank_match else pareto_rank
        
        suffix_match = re.search(r"model_\d+(.+)", model_name)
        inferred_suffix = suffix_match.group(1) if suffix_match else model_suffix
        
        print(f"Loading BERTopic model from: {model_path}")
        print(f"  Using base_dir: {inferred_base_dir}")
        print(f"  Embedding model: {inferred_embedding_model}")
        print(f"  Pareto rank: {inferred_rank}")
        print(f"  Model suffix: {inferred_suffix}")
        _, topic_model = load_bertopic_model(
            base_dir=inferred_base_dir,
            embedding_model=inferred_embedding_model,
            pareto_rank=inferred_rank,
            use_native=use_native,
            model_suffix=inferred_suffix,
        )
    else:
        # Use project's load_bertopic_model function
        if base_dir is None:
            base_dir = DEFAULT_BASE_DIR
        print(f"Loading BERTopic model from base_dir: {base_dir}")
        print(f"  Embedding model: {embedding_model}")
        print(f"  Pareto rank: {pareto_rank}")
        print(f"  Model suffix: {model_suffix}")
        _, topic_model = load_bertopic_model(
            base_dir=base_dir,
            embedding_model=embedding_model,
            pareto_rank=pareto_rank,
            use_native=use_native,
            model_suffix=model_suffix,
        )

    # Get representative docs for each topic
    print(
        f"\nExtracting up to {max_docs_per_topic} representative docs per topic "
        f"for {len(labels_data)} labeled topics..."
    )
    topic_to_docs = extract_representative_docs_per_topic(
        topic_model,
        max_docs_per_topic=max_docs_per_topic,
    )

    # Determine which topics are both in labels and have docs
    available_topic_ids = [
        int(tid)
        for tid in labels_data.keys()
        if int(tid) in topic_to_docs and topic_to_docs[int(tid)]
    ]

    if not available_topic_ids:
        raise RuntimeError("No overlapping topics between labels JSON and representative docs.")

    # Check if specific topic IDs were requested
    if topic_ids:
        # Filter to only available topics
        sampled_ids = [tid for tid in topic_ids if tid in available_topic_ids]
        if len(sampled_ids) < len(topic_ids):
            missing = set(topic_ids) - set(sampled_ids)
            print(f"Warning: {len(missing)} requested topic IDs not available: {missing}")
        print(f"\nInspecting {len(sampled_ids)} specific topic(s): {sampled_ids}\n")
    else:
        sample_size = min(num_topics, len(available_topic_ids))
        sampled_ids = random.sample(available_topic_ids, k=sample_size)
        print(f"\nSampling {sample_size} topics for manual inspection...\n")

    for tid in sorted(sampled_ids):
        entry = labels_data[str(tid)]
        label = entry.get("label", "")
        scene_summary = entry.get("scene_summary", "")
        keywords = entry.get("keywords", [])

        docs = topic_to_docs.get(tid, []) or []

        if use_centrality and len(docs) > 0:
            docs = rerank_snippets_centrality(docs, top_k=min(max_docs_per_topic, len(docs)))

        # Take the top N docs for display (exact sentences as they appear)
        snippets = docs[:num_snippets_to_show]

        print("=" * 100)
        print(f"Topic ID: {tid}")
        print(f"Label: {label}")
        print(f"Scene summary: {scene_summary}")
        print(f"Keywords: {', '.join(keywords)}")
        print("-" * 100)
        print("Representative snippets (exact sentences):")
        if not snippets:
            print("  [No representative docs available for this topic]")
        else:
            for i, s in enumerate(snippets, start=1):
                # Clean up whitespace a bit, but keep the sentence content
                text = " ".join(s.split())
                print(f"{i:2d}. {text}")
        print()  # blank line between topics


def main():
    parser = argparse.ArgumentParser(
        description="Inspect random topics with labels, summaries, keywords, and representative sentences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model loading options - either direct path OR project parameters
    parser.add_argument(
        "--bertopic-model-path",
        type=str,
        default=None,
        help="Direct path to saved BERTopic model file (e.g. models/retrained/model_1_with_noise_labels.pkl). "
             "If provided, uses BERTopic.load() directly. If not provided, uses project's load_bertopic_model() "
             "with --base-dir, --embedding-model, etc.",
    )

    parser.add_argument(
        "--labels-json",
        type=str,
        required=True,
        help="Path to labels JSON produced by the OpenRouter labeling pipeline.",
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base directory for retrained models (used when --bertopic-model-path is not provided).",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name (used when --bertopic-model-path is not provided).",
    )

    parser.add_argument(
        "--pareto-rank",
        type=int,
        default=1,
        help="Pareto rank of model (used when --bertopic-model-path is not provided).",
    )

    parser.add_argument(
        "--use-native",
        action="store_true",
        help="Use native safetensors (used when --bertopic-model-path is not provided).",
    )

    parser.add_argument(
        "--model-suffix",
        type=str,
        default="_with_noise_labels",
        help="Model suffix (used when --bertopic-model-path is not provided).",
    )

    parser.add_argument(
        "--num-topics",
        type=int,
        default=15,
        help="Number of random topics to inspect (ignored if --topic-ids provided).",
    )
    
    parser.add_argument(
        "--topic-ids",
        type=str,
        default=None,
        help="Comma-separated list of specific topic IDs to inspect (e.g., '3,4,5,8'). Overrides --num-topics and --seed.",
    )

    parser.add_argument(
        "--max-docs-per-topic",
        type=int,
        default=40,
        help="Maximum representative docs per topic to load from BERTopic.",
    )

    parser.add_argument(
        "--num-snippets-to-show",
        type=int,
        default=8,
        help="Maximum number of representative sentences to print per topic.",
    )

    parser.add_argument(
        "--no-centrality",
        action="store_true",
        help="If set, do NOT rerank snippets by centrality; use BERTopic's original order.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for topic sampling.",
    )

    args = parser.parse_args()

    # Parse topic IDs if provided
    requested_topic_ids = None
    if args.topic_ids:
        try:
            requested_topic_ids = [int(tid.strip()) for tid in args.topic_ids.split(",")]
        except ValueError as e:
            raise ValueError(f"Invalid topic IDs format: {args.topic_ids}. Use comma-separated integers (e.g., '3,4,5,8')")
    
    # Determine model loading approach
    if args.bertopic_model_path:
        inspect_random_topics(
            bertopic_model_path=args.bertopic_model_path,
            labels_json_path=args.labels_json,
            num_topics=args.num_topics,
            max_docs_per_topic=args.max_docs_per_topic,
            num_snippets_to_show=args.num_snippets_to_show,
            use_centrality=not args.no_centrality,
            random_seed=args.seed,
            topic_ids=requested_topic_ids,
        )
    else:
        inspect_random_topics(
            bertopic_model_path=None,
            labels_json_path=args.labels_json,
            base_dir=args.base_dir,
            embedding_model=args.embedding_model,
            pareto_rank=args.pareto_rank,
            use_native=args.use_native,
            model_suffix=args.model_suffix,
            num_topics=args.num_topics,
            max_docs_per_topic=args.max_docs_per_topic,
            num_snippets_to_show=args.num_snippets_to_show,
            use_centrality=not args.no_centrality,
            random_seed=args.seed,
            topic_ids=requested_topic_ids,
        )


if __name__ == "__main__":
    main()

