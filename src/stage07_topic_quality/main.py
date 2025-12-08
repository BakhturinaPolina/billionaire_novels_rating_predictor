"""CLI entry point for topic quality analysis and noisy topic detection."""

from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

from bertopic import BERTopic

from src.stage06_topic_exploration.explore_retrained_model import (
    DEFAULT_BASE_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHAPTERS_CSV,
    DEFAULT_CHAPTERS_SUBSET_CSV,
    DEFAULT_CORPUS_PATH,
    DEFAULT_EMBEDDING_MODEL,
    LOGGER,
    load_dictionary_from_corpus,
    load_native_bertopic_model,
    load_retrained_wrapper,
    prepare_documents,
    resolve_dataset_path,
    stage_timer,
    backup_existing_file,
)
from src.stage07_topic_quality.topic_quality_analysis import (
    apply_noise_labels_to_model,
    build_topic_quality_table,
)

# Use the same logger as stage06_topic_exploration
logger = LOGGER


def parse_args() -> argparse.Namespace:
    """CLI argument parsing for topic quality analysis."""
    parser = argparse.ArgumentParser(
        description="EDA analysis and noisy topic detection for retrained BERTopic models."
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
        help="Pareto rank of the model to analyze",
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
        help="Load the native BERTopic safetensors directory instead of the pickle wrapper",
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=None,
        help="Optional dataset CSV (chapters.csv or subset) when wrapper docs are unavailable",
    )
    parser.add_argument(
        "--dictionary-path",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Path to the OCTIS corpus TSV used to build the gensim dictionary",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for document loading and dictionary building",
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Optionally limit the number of documents loaded from CSV sources",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top words per topic to consider",
    )
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=30,
        help="Minimum number of documents per topic to be considered valid",
    )
    parser.add_argument(
        "--min-pos-words",
        type=int,
        default=3,
        help="Minimum number of POS words per topic",
    )
    parser.add_argument(
        "--min-pos-coherence",
        type=float,
        default=0.0,
        help="Minimum per-topic POS coherence threshold",
    )
    parser.add_argument(
        "--fallback-dataset",
        type=str,
        default="chapters",
        choices=["chapters", "subset"],
        help="Which default CSV to use when wrapper data is missing",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage07_topic_quality"),
        help="Directory to save output files",
    )
    parser.add_argument(
        "--apply-labels",
        action="store_true",
        help="Apply inspection labels to noisy topics in the model (optional)",
    )
    parser.add_argument(
        "--save-model-with-labels",
        type=Path,
        default=None,
        help="If --apply-labels is used, optionally save the model with labels to this path",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for topic quality analysis and noisy topic detection."""
    args = parse_args()
    logger.info("=== Stage 06 EDA: Topic Quality Analysis ===")
    logger.info("Configuration: %s", vars(args))

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    wrapper = None
    topic_model: BERTopic

    with stage_timer("Loading retrained model"):
        if args.use_native:
            topic_model = load_native_bertopic_model(
                base_dir=args.base_dir,
                embedding_model=args.embedding_model,
                pareto_rank=args.pareto_rank,
            )
        else:
            wrapper, topic_model = load_retrained_wrapper(
                base_dir=args.base_dir,
                embedding_model=args.embedding_model,
                pareto_rank=args.pareto_rank,
            )

    logger.info("Wrapper loaded: %s", wrapper is not None)
    logger.info("topic_model type: %s", type(topic_model))

    # Diagnostics
    logger.info("=== Model Diagnostics (Loaded State) ===")
    if hasattr(topic_model, "topic_representations_") and topic_model.topic_representations_:
        topic_ids = [
            tid for tid in topic_model.topic_representations_.keys() if tid != -1
        ]
        logger.info("Topic IDs in topic_representations_: %s", sorted(topic_ids)[:20])
        logger.info("Total topics (excluding -1): %d", len(topic_ids))

    if hasattr(topic_model, "get_topic_info"):
        try:
            topic_info = topic_model.get_topic_info()
            logger.info("Topic info shape: %s", getattr(topic_info, "shape", "N/A"))
            logger.info(
                "Topic info columns: %s", getattr(topic_info, "columns", "N/A")
            )
        except Exception as e:
            logger.warning("Could not get topic_info: %s", e)

    # Load documents
    dataset_csv = None if wrapper else resolve_dataset_path(args)
    docs, docs_tokens = prepare_documents(
        wrapper,
        dataset_csv=dataset_csv,
        batch_size=args.batch_size,
        limit=args.limit_docs,
    )

    logger.info("Loaded documents: %d", len(docs))
    logger.info("Loaded tokens: %d", len(docs_tokens))

    # Load or build dictionary
    dictionary = load_dictionary_from_corpus(
        corpus_path=args.dictionary_path,
        batch_size=args.batch_size,
    )
    logger.info("Dictionary size: %d", len(dictionary))

    # Build topic quality table
    with stage_timer("Running topic EDA + noise candidate labeling"):
        quality_df = build_topic_quality_table(
            topic_model,
            docs_tokens=docs_tokens,
            dictionary=dictionary,
            min_size=args.min_topic_size,
            min_pos_words=args.min_pos_words,
            min_pos_coherence=args.min_pos_coherence,
            top_k=args.top_k,
        )

    logger.info("Total topics (excl. -1): %d", len(quality_df))
    logger.info("Candidate noisy topics: %d", int(quality_df["noise_candidate"].sum()))

    # Print summary statistics
    print("\n=== Topic Quality Summary ===")
    print(f"Total topics (excluding -1): {len(quality_df)}")
    print(f"Candidate noisy topics: {int(quality_df['noise_candidate'].sum())}")
    print(f"\nTopics flagged as small (<{args.min_topic_size} docs): {int(quality_df['flag_small'].sum())}")
    print(f"Topics flagged as few POS words (<{args.min_pos_words}): {int(quality_df['flag_few_pos'].sum())}")
    print(f"Topics flagged as low coherence (<{args.min_pos_coherence:.2f}): {int(quality_df['flag_low_coh'].sum())}")

    # Print worst topics by POS coherence
    print("\n=== 20 topics with lowest POS coherence ===")
    worst_topics = quality_df.sort_values("coherence_c_v_pos").head(20)[
        [
            "Topic",
            "Count",
            "coherence_c_v_pos",
            "n_pos_words",
            "noise_candidate",
            "noise_reason",
            "inspection_label",
        ]
    ]
    print(worst_topics.to_string(index=False))

    # Print candidate noisy topics
    print("\n=== Candidate noisy topics (for manual inspection) ===")
    noise_candidates = quality_df[quality_df["noise_candidate"]].head(20)[
        [
            "Topic",
            "Count",
            "coherence_c_v_pos",
            "n_pos_words",
            "noise_reason",
            "inspection_label",
        ]
    ]
    print(noise_candidates.to_string(index=False))

    # Save results
    model_name_safe = args.embedding_model.replace("/", "_").replace("\\", "_")
    quality_path_full = args.output_dir / f"topic_quality_{model_name_safe}.csv"
    quality_path_noise = args.output_dir / f"topic_noise_candidates_{model_name_safe}.csv"

    for path, df_to_save in [
        (quality_path_full, quality_df),
        (quality_path_noise, quality_df[quality_df["noise_candidate"]]),
    ]:
        backup_existing_file(path)
        with stage_timer(f"Saving {path.name}"):
            df_to_save.to_csv(path, index=False)
            logger.info("Saved %d rows to %s", len(df_to_save), path)

    print(f"\nSaved full quality table to: {quality_path_full}")
    print(f"Saved noise candidates to: {quality_path_noise}")

    # Optionally apply labels to model
    if args.apply_labels:
        with stage_timer("Applying inspection labels to noisy topics"):
            noise_labels = apply_noise_labels_to_model(
                topic_model, quality_df, only_noise_candidates=True
            )

            # Merge with existing labels if any
            existing_labels = getattr(topic_model, "custom_labels_", None)
            labels_dict = {}
            if isinstance(existing_labels, dict):
                labels_dict.update(existing_labels)
            labels_dict.update(noise_labels)

            topic_model.set_topic_labels(labels_dict)
            logger.info(
                "Updated labels for %d topics (noise candidates + existing)",
                len(labels_dict),
            )

        # Optionally save model with labels
        if args.save_model_with_labels:
            save_path = Path(args.save_model_with_labels)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with stage_timer(f"Saving BERTopic model with noise labels to {save_path}"):
                topic_model.save(str(save_path))
                logger.info("Saved model with noise labels to %s", save_path)


if __name__ == "__main__":
    main()

