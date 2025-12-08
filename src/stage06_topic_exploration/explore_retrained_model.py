"""Stage 06 utilities for exploring retrained BERTopic models with diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import re
import shutil
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Sequence, Tuple, Any

from bertopic import BERTopic
from bertopic.representation import (
    KeyBERTInspired,
    MaximalMarginalRelevance,
    PartOfSpeech,
)
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

from src.stage05_retraining.retrain_models import RetrainableBERTopicModel

LOGGER = logging.getLogger("stage06_topics_exploration")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)

DEFAULT_BASE_DIR = Path(
    "/home/polina/Documents/goodreads_romance_research_cursor/billionaire_novels_rating_predictor/models/retrained"
)
DEFAULT_EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"
DEFAULT_CORPUS_PATH = Path(
    "/home/polina/Documents/goodreads_romance_research_cursor/billionaire_novels_rating_predictor/data/interim/octis/corpus.tsv"
)
DEFAULT_CHAPTERS_CSV = Path(
    "/home/polina/Documents/goodreads_romance_research_cursor/billionaire_novels_rating_predictor/data/processed/chapters.csv"
)
DEFAULT_CHAPTERS_SUBSET_CSV = Path(
    "/home/polina/Documents/goodreads_romance_research_cursor/billionaire_novels_rating_predictor/data/processed/chapters_subset_10000.csv"
)
DEFAULT_BATCH_SIZE = 50_000
WHITESPACE_PATTERN = re.compile(r"\s+")


@contextmanager
def stage_timer(stage_name: str):
    """Context manager that logs start/end timestamps for each processing stage."""
    LOGGER.info("▶ %s | start", stage_name)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        LOGGER.info("■ %s | completed in %.2f s", stage_name, elapsed)


def normalize_sentence(sentence: str) -> str:
    """Normalize whitespace and casing for consistent downstream processing."""
    sentence = WHITESPACE_PATTERN.sub(" ", sentence.replace("\n", " ")).strip()
    return sentence.lower()


def log_batch_progress(stage: str, batch_idx: int, start: int, end: int, total: int):
    """Emit detailed progress logs for batched operations."""
    LOGGER.info(
        "%s | batch %d => rows %d-%d / %d",
        stage,
        batch_idx,
        start,
        end,
        total,
    )


def iter_corpus_tokens(corpus_path: Path) -> Iterator[list[str]]:
    """Stream tokenized sentences from a TSV corpus (column-0 sentence)."""
    with stage_timer(f"Dictionary corpus streaming: {corpus_path.name}"):
        with open(corpus_path, "r", encoding="latin1", errors="ignore") as handle:
            for row_idx, line in enumerate(handle, start=1):
                parts = line.rstrip("\n").split("\t")
                if not parts:
                    continue
                normalized = normalize_sentence(parts[0])
                if not normalized:
                    continue
                if row_idx % 50_000 == 0:
                    LOGGER.info("Dictionary stream | processed %d rows", row_idx)
                yield normalized.split()


def load_dictionary_from_corpus(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dictionary:
    """Load (or build) a gensim dictionary by streaming the pre-built corpus."""
    dictionary = Dictionary()
    buffer: list[list[str]] = []

    with stage_timer(f"Dictionary build: {corpus_path}"):
        for tokens in iter_corpus_tokens(corpus_path):
            buffer.append(tokens)
            if len(buffer) >= batch_size:
                dictionary.add_documents(buffer)
                LOGGER.info(
                    "Dictionary build | added %d docs (size=%d)",
                    len(buffer),
                    len(dictionary),
                )
                buffer.clear()

        if buffer:
            dictionary.add_documents(buffer)
            LOGGER.info(
                "Dictionary build | final flush %d docs (size=%d)",
                len(buffer),
                len(dictionary),
            )

        dictionary.compactify()
        LOGGER.info("Dictionary build | completed with %d tokens", len(dictionary))

    return dictionary


def load_metadata(
    base_dir: Path | str = DEFAULT_BASE_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    pareto_rank: int = 1,
) -> dict[str, Any] | None:
    """Load metadata JSON file if it exists."""
    base_dir = Path(base_dir)
    metadata_path = base_dir / embedding_model / f"model_{pareto_rank}_metadata.json"
    
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        LOGGER.warning("Could not load metadata: %s", e)
        return None


def load_retrained_wrapper(
    base_dir: Path | str = DEFAULT_BASE_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    pareto_rank: int = 1,
    model_suffix: str = "",
) -> Tuple[RetrainableBERTopicModel, BERTopic]:
    """
    Load the Stage 05 pickle wrapper and recover the BERTopic instance.
    
    Args:
        base_dir: Base directory for models
        embedding_model: Model name
        pareto_rank: Model rank
        model_suffix: Optional suffix to append to model filename (e.g., "_with_noise_labels")
    """
    base_dir = Path(base_dir)
    pickle_path = base_dir / embedding_model / f"model_{pareto_rank}{model_suffix}.pkl"

    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle model not found: {pickle_path}")

    with stage_timer(f"Loading pickle wrapper: {pickle_path.name}"):
        with open(pickle_path, "rb") as handle:
            wrapper = pickle.load(handle)

    if (
        not hasattr(wrapper, "trained_topic_model")
        or wrapper.trained_topic_model is None
    ):
        raise ValueError("Loaded wrapper has no trained_topic_model stored.")

    # Check metadata and warn if topic count doesn't match
    metadata = load_metadata(base_dir, embedding_model, pareto_rank)
    if metadata:
        expected_topics = metadata.get("num_topics")
        if expected_topics is not None:
            # Count actual topics in loaded model
            if hasattr(wrapper.trained_topic_model, "topic_representations_"):
                actual_topics = len([
                    tid for tid in wrapper.trained_topic_model.topic_representations_.keys()
                    if tid != -1
                ])
                if actual_topics != expected_topics:
                    LOGGER.warning(
                        "⚠️  TOPIC COUNT MISMATCH: Metadata says %d topics, but model has %d topics. "
                        "This may indicate the model was reduced during training or saving.",
                        expected_topics,
                        actual_topics,
                    )

    return wrapper, wrapper.trained_topic_model


def load_native_bertopic_model(
    base_dir: Path | str = DEFAULT_BASE_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    pareto_rank: int = 1,
    model_suffix: str = "",
) -> BERTopic:
    """
    Load a retrained BERTopic safetensors directory.
    
    Args:
        base_dir: Base directory for models
        embedding_model: Model name
        pareto_rank: Model rank
        model_suffix: Optional suffix to append to model directory name (e.g., "_with_noise_labels")
    """
    base_dir = Path(base_dir)
    model_dir = base_dir / embedding_model / f"model_{pareto_rank}{model_suffix}"

    if not model_dir.exists():
        raise FileNotFoundError(
            f"BERTopic native model directory not found: {model_dir}"
        )

    with stage_timer(f"Loading BERTopic safetensors: {model_dir}"):
        topic_model = BERTopic.load(model_dir)
    
    # Check metadata and warn if topic count doesn't match
    metadata = load_metadata(base_dir, embedding_model, pareto_rank)
    if metadata:
        expected_topics = metadata.get("num_topics")
        if expected_topics is not None:
            # Count actual topics in loaded model
            if hasattr(topic_model, "topic_representations_"):
                actual_topics = len([
                    tid for tid in topic_model.topic_representations_.keys()
                    if tid != -1
                ])
                if actual_topics != expected_topics:
                    LOGGER.warning(
                        "⚠️  TOPIC COUNT MISMATCH: Metadata says %d topics, but model has %d topics. "
                        "This may indicate the model was reduced during training or saving.",
                        expected_topics,
                        actual_topics,
                    )
    
    return topic_model


def log_wrapper_batches(total_docs: int, batch_size: int):
    """Emit synthetic batch logs for wrapper-loaded documents."""
    for batch_idx, start in enumerate(range(0, total_docs, batch_size), start=1):
        end = min(start + batch_size, total_docs)
        log_batch_progress("Wrapper docs in-memory", batch_idx, start + 1, end, total_docs)


def load_documents_from_csv(
    csv_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
) -> tuple[list[str], list[list[str]]]:
    """Load sentences + tokens from a Stage 05 CSV in streaming batches."""
    docs: list[str] = []
    tokens: list[list[str]] = []
    total_rows = 0

    with stage_timer(f"Loading dataset CSV: {csv_path.name}"):
        with open(csv_path, "r", encoding="latin1", errors="ignore") as handle:
            reader = csv.reader(
                handle,
                quotechar='"',
                delimiter=",",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True,
            )
            headers = next(reader, None)
            LOGGER.info("Dataset CSV | headers = %s", headers)

            batch_idx = 0
            batch_docs: list[str] = []
            batch_tokens: list[list[str]] = []

            for row in reader:
                if len(row) < 4:
                    continue

                sentence = normalize_sentence(row[3])
                if not sentence:
                    continue

                batch_docs.append(sentence)
                batch_tokens.append(sentence.split())
                total_rows += 1

                if len(batch_docs) >= batch_size:
                    batch_idx += 1
                    log_batch_progress(
                        "CSV streaming",
                        batch_idx,
                        total_rows - len(batch_docs) + 1,
                        total_rows,
                        limit or -1,
                    )
                    docs.extend(batch_docs)
                    tokens.extend(batch_tokens)
                    batch_docs.clear()
                    batch_tokens.clear()

                if limit and total_rows >= limit:
                    LOGGER.info("CSV streaming | reached limit=%d rows", limit)
                    break

            if batch_docs:
                batch_idx += 1
                log_batch_progress(
                    "CSV streaming",
                    batch_idx,
                    total_rows - len(batch_docs) + 1,
                    total_rows,
                    limit or total_rows,
                )
                docs.extend(batch_docs)
                tokens.extend(batch_tokens)

    LOGGER.info(
        "CSV streaming | loaded %d documents (tokens=%d)",
        len(docs),
        len(tokens),
    )
    return docs, tokens


def prepare_documents(
    wrapper: RetrainableBERTopicModel | None,
    dataset_csv: Path | None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
) -> tuple[list[str], list[list[str]]]:
    """
    Retrieve documents/token lists either from the wrapper or a CSV,
    emitting progress logs in batches.
    """
    if wrapper is not None:
        docs = wrapper.dataset_as_list_of_strings
        tokens = getattr(wrapper, "dataset_as_list_of_lists", None)
        if not tokens:
            tokens = [doc.split() for doc in docs]
        log_wrapper_batches(len(docs), batch_size)
        LOGGER.info(
            "Wrapper dataset | docs=%d, tokens=%d (limit=%s)",
            len(docs),
            len(tokens),
            limit,
        )
        return docs, tokens

    if dataset_csv is None:
        raise ValueError("dataset_csv must be provided when wrapper is not available.")

    return load_documents_from_csv(dataset_csv, batch_size=batch_size, limit=limit)


def build_representation_models(
    language_model: str = "en_core_web_sm",
) -> dict[str, object | None]:
    """Create representation bundle with KeyBERT, POS, and MMR variants."""
    pos_patterns = [
        [{"POS": "NOUN"}],
        [{"POS": "VERB"}],
        [{"POS": "ADJ"}],
    ]

    representation_models: dict[str, object | None] = {
        "Main": None,
        "KeyBERT": KeyBERTInspired(),
        "POS": PartOfSpeech(language_model, pos_patterns=pos_patterns),
        "MMR": MaximalMarginalRelevance(diversity=0.3),
    }

    return representation_models


def apply_representations_and_update(
    topic_model: BERTopic,
    docs: Sequence[str],
    representations: dict[str, object | None],
) -> None:
    """Attach new representations and refresh topic words."""
    with stage_timer("Updating topic representations"):
        # Check topic count before update
        if hasattr(topic_model, "topic_representations_") and topic_model.topic_representations_:
            topics_before = len([tid for tid in topic_model.topic_representations_.keys() if tid != -1])
            LOGGER.info("Topics before update_topics: %d", topics_before)
        else:
            LOGGER.warning("topic_representations_ not available before update")
        
        topic_model.representation_model = representations
        LOGGER.info(
            "Topic representations set: %s",
            ", ".join(representations.keys()),
        )
        topic_model.update_topics(docs)
        LOGGER.info("topic_model.update_topics completed for %d docs", len(docs))
        
        # Check topic count after update
        if hasattr(topic_model, "topic_representations_") and topic_model.topic_representations_:
            topics_after = len([tid for tid in topic_model.topic_representations_.keys() if tid != -1])
            LOGGER.info("Topics after update_topics: %d", topics_after)
            if topics_before != topics_after:
                LOGGER.warning(
                    "Topic count changed from %d to %d after update_topics!",
                    topics_before,
                    topics_after,
                )
        else:
            LOGGER.warning("topic_representations_ not available after update")


def compute_coherence(
    topics: Sequence[Sequence[str]],
    texts: Sequence[Sequence[str]],
    dictionary: Dictionary,
) -> float:
    """Compute c_v coherence for the supplied topics."""
    coherence_model = CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
    )
    return float(coherence_model.get_coherence())


def compute_topic_diversity(
    topics: Sequence[Sequence[str]],
    top_k: int = 10,
) -> float:
    """Topic diversity = unique words / total extracted terms."""
    if not topics:
        return 0.0

    truncated = [topic[:top_k] for topic in topics if topic]
    flattened = [word for topic in truncated for word in topic]
    if not flattened:
        return 0.0

    unique_words = len(set(flattened))
    total_terms = sum(len(topic) for topic in truncated)
    return unique_words / total_terms if total_terms else 0.0


def evaluate_representations(
    topic_model: BERTopic,
    docs_tokens: Sequence[Sequence[str]],
    dictionary: Dictionary,
    top_k: int = 10,
) -> list[dict[str, float | str | int]]:
    """Compute coherence/diversity for each available representation label."""
    results: list[dict[str, float | str | int]] = []

    def _process_representation(
        label: str, representation_dict: dict[int, list[tuple[str, float] | str]]
    ):
        """Helper to extract words and compute metrics for a single representation."""
        rep_topics: list[list[str]] = []
        
        for topic_id, topic_content in representation_dict.items():
            if topic_id == -1:
                continue
            
            words: list[str] = []
            for item in topic_content[:top_k]:
                if isinstance(item, tuple) and len(item) > 0:
                    words.append(str(item[0]))
                elif isinstance(item, str):
                    words.append(item)
            
            if words:
                rep_topics.append(words)

        if not rep_topics:
            LOGGER.warning("No valid topics found for representation=%s", label)
            return

        LOGGER.info("Evaluating representation='%s' with %d topics", label, len(rep_topics))
        
        try:
            # Convert tokens to IDs to bypass Gensim heuristic check
            rep_topic_ids = []
            for t in rep_topics:
                ids = [dictionary.token2id[w] for w in t if w in dictionary.token2id]
                if ids:
                    rep_topic_ids.append(ids)
            
            if not rep_topic_ids:
                LOGGER.warning("No topics with valid dictionary tokens for %s", label)
                return

            coherence = compute_coherence(
                rep_topic_ids,
                docs_tokens,
                dictionary=dictionary,
            )
            diversity = compute_topic_diversity(rep_topics, top_k=top_k)

            LOGGER.info(
                "Metrics | %s | topics=%d | c_v=%.4f | diversity=%.4f",
                label,
                len(rep_topics),
                coherence,
                diversity,
            )

            results.append(
                {
                    "representation": label,
                    "n_topics": len(rep_topics),
                    "coherence_c_v": coherence,
                    "topic_diversity": diversity,
                }
            )
        except Exception as e:
            LOGGER.error("Error computing metrics for representation='%s': %s", label, e)
            import traceback
            LOGGER.error(traceback.format_exc())

    with stage_timer("Evaluating representations"):
        # 1. Evaluate Main Representation
        if hasattr(topic_model, "topic_representations_"):
            LOGGER.info("Processing 'Main' representation from topic_representations_")
            _process_representation("Main", topic_model.topic_representations_)
        else:
            LOGGER.warning("topic_representations_ not found!")

        # 2. Evaluate Other Aspects
        if hasattr(topic_model, "topic_aspects_") and topic_model.topic_aspects_:
            LOGGER.info("Processing additional aspects from topic_aspects_")
            for aspect_name, aspect_content in topic_model.topic_aspects_.items():
                _process_representation(aspect_name, aspect_content)
        else:
            LOGGER.info("No additional topic_aspects_ found.")

    return results


def extract_all_topics(
    topic_model: BERTopic,
    top_k: int = 10,
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    """
    Extract all topics with word lists for all representations.
    
    Returns a nested dictionary:
    {
        "Main": {
            topic_id: [{"word": str, "score": float}, ...],
            ...
        },
        "KeyBERT": {
            topic_id: [{"word": str, "score": float}, ...],
            ...
        },
        ...
    }
    
    Based on BERTopic documentation:
    - topic_representations_: Main c-TF-IDF representation
    - topic_aspects_: Additional representations (KeyBERT, POS, MMR, etc.)
    """
    all_topics: dict[str, dict[int, list[dict[str, Any]]]] = {}
    
    def _extract_representation(
        label: str, 
        representation_dict: dict[int, list[tuple[str, float] | str]]
    ) -> dict[int, list[dict[str, Any]]]:
        """Extract words and scores for a single representation."""
        topics_dict: dict[int, list[dict[str, Any]]] = {}
        
        for topic_id, topic_content in representation_dict.items():
            if topic_id == -1:
                continue
            
            # Skip topics with empty or None content
            if not topic_content:
                LOGGER.debug("Skipping topic %d: empty content", topic_id)
                continue
            
            words_list: list[dict[str, Any]] = []
            for item in topic_content[:top_k]:
                word = None
                score = 0.0
                
                if isinstance(item, tuple) and len(item) >= 2:
                    word, score = item[0], item[1]
                elif isinstance(item, tuple) and len(item) == 1:
                    word = item[0]
                elif isinstance(item, str):
                    word = item
                
                # Filter out empty words and whitespace-only words
                if word is not None:
                    word_str = str(word).strip()
                    if word_str:  # Only add non-empty words
                        words_list.append({"word": word_str, "score": float(score)})
            
            # Only add topics that have at least one non-empty word
            if words_list:
                topics_dict[topic_id] = words_list
            else:
                LOGGER.debug("Skipping topic %d: no valid words after filtering", topic_id)
        
        return topics_dict
    
    with stage_timer("Extracting all topic representations"):
        # 1. Extract Main Representation
        if hasattr(topic_model, "topic_representations_") and topic_model.topic_representations_:
            LOGGER.info("Extracting 'Main' representation from topic_representations_")
            all_topics["Main"] = _extract_representation("Main", topic_model.topic_representations_)
            LOGGER.info("Extracted %d topics for 'Main' representation", len(all_topics["Main"]))
        else:
            LOGGER.warning("topic_representations_ not found or empty!")
            all_topics["Main"] = {}
        
        # 2. Extract Other Aspects
        if hasattr(topic_model, "topic_aspects_") and topic_model.topic_aspects_:
            LOGGER.info("Extracting additional aspects from topic_aspects_")
            for aspect_name, aspect_content in topic_model.topic_aspects_.items():
                all_topics[aspect_name] = _extract_representation(aspect_name, aspect_content)
                LOGGER.info(
                    "Extracted %d topics for '%s' representation",
                    len(all_topics[aspect_name]),
                    aspect_name,
                )
        else:
            LOGGER.info("No additional topic_aspects_ found.")
    
    return all_topics


def backup_existing_file(file_path: Path) -> None:
    """
    Backup an existing file by renaming it with a timestamp suffix.
    
    Args:
        file_path: Path to the file that may need backing up
    """
    if file_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.parent / f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
        shutil.copy2(file_path, backup_path)
        LOGGER.info("Backed up existing file: %s -> %s", file_path.name, backup_path.name)


def save_metrics(
    metrics: list[dict[str, float | str | int]],
    output_path: Path,
    format: str = "json",
) -> None:
    """
    Save metrics table to CSV or JSON file.
    Backs up existing files before overwriting.
    
    Args:
        metrics: List of metric dictionaries from evaluate_representations
        output_path: Path to save the file (without extension)
        format: Either "csv" or "json"
    """
    output_path = Path(output_path)
    
    if format.lower() == "json":
        json_path = output_path.with_suffix(".json")
        backup_existing_file(json_path)
        with stage_timer(f"Saving metrics to JSON: {json_path.name}"):
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            LOGGER.info("Saved metrics to %s", json_path)
    
    elif format.lower() == "csv":
        csv_path = output_path.with_suffix(".csv")
        backup_existing_file(csv_path)
        with stage_timer(f"Saving metrics to CSV: {csv_path.name}"):
            if not metrics:
                LOGGER.warning("No metrics to save")
                return
            
            fieldnames = list(metrics[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metrics)
            LOGGER.info("Saved metrics to %s", csv_path)
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'")


def save_topics(
    topics: dict[str, dict[int, list[dict[str, Any]]]],
    output_path: Path,
) -> None:
    """
    Save extracted topics with all representations to JSON for close reading.
    Backs up existing files before overwriting.
    
    The output JSON structure:
    {
        "Main": {
            "0": [{"word": "example", "score": 0.123}, ...],
            "1": [{"word": "another", "score": 0.456}, ...],
            ...
        },
        "KeyBERT": {
            "0": [{"word": "example", "score": 0.123}, ...],
            ...
        },
        ...
    }
    
    Args:
        topics: Dictionary from extract_all_topics
        output_path: Path to save the JSON file (without extension)
    """
    json_path = output_path.with_suffix(".json")
    backup_existing_file(json_path)
    
    with stage_timer(f"Saving topics to JSON: {json_path.name}"):
        # Convert topic IDs to strings for JSON serialization
        topics_serializable: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for rep_name, rep_topics in topics.items():
            topics_serializable[rep_name] = {
                str(topic_id): word_list for topic_id, word_list in rep_topics.items()
            }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(topics_serializable, f, indent=2, ensure_ascii=False)
        
        total_topics = sum(len(rep_topics) for rep_topics in topics.values())
        LOGGER.info(
            "Saved %d topics across %d representations to %s",
            total_topics,
            len(topics),
            json_path,
        )


def parse_args() -> argparse.Namespace:
    """CLI argument parsing for flexible experimentation."""
    parser = argparse.ArgumentParser(
        description="Explore retrained BERTopic models with rich logging."
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--pareto-rank", type=int, default=1)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument(
        "--use-native",
        action="store_true",
        help="Load the native BERTopic safetensors directory instead of the pickle wrapper.",
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        help="Optional dataset CSV (chapters.csv or subset) when wrapper docs are unavailable.",
    )
    parser.add_argument(
        "--dictionary-path",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Path to the OCTIS corpus TSV used to build the gensim dictionary.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Optionally limit the number of documents loaded from CSV sources.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--language-model",
        default="en_core_web_sm",
        help="spaCy model name for the POS representation.",
    )
    parser.add_argument(
        "--fallback-dataset",
        type=str,
        default="chapters",
        choices=["chapters", "subset"],
        help="Which default CSV to use when wrapper data is missing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage06_topic_exploration"),
        help="Directory to save output files (metrics and topics). Default: results/stage06_topic_exploration",
    )
    parser.add_argument(
        "--metrics-format",
        type=str,
        default="json",
        choices=["csv", "json"],
        help="Format for saving metrics table (default: json).",
    )
    parser.add_argument(
        "--save-topics",
        action="store_true",
        help="Extract and save all topics with all representations to JSON for close reading.",
    )
    return parser.parse_args()


def resolve_dataset_path(args: argparse.Namespace) -> Path | None:
    """Determine which dataset CSV to use when wrapper docs are unavailable."""
    if args.dataset_csv:
        return args.dataset_csv
    if args.fallback_dataset == "subset":
        return DEFAULT_CHAPTERS_SUBSET_CSV
    return DEFAULT_CHAPTERS_CSV


def main() -> None:
    """Entrypoint that ties together loading, instrumentation, and metric reporting."""
    args = parse_args()
    LOGGER.info("=== Stage 06 BERTopic Topics Exploration ===")
    LOGGER.info("Configuration: %s", vars(args))

    wrapper: RetrainableBERTopicModel | None = None

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

    dataset_csv = None if wrapper else resolve_dataset_path(args)
    docs, docs_tokens = prepare_documents(
        wrapper,
        dataset_csv=dataset_csv,
        batch_size=args.batch_size,
        limit=args.limit_docs,
    )

    # Diagnostic: Check model state before any updates
    LOGGER.info("=== Model Diagnostics (Before Update) ===")
    if hasattr(topic_model, "topic_representations_") and topic_model.topic_representations_:
        topic_ids = [tid for tid in topic_model.topic_representations_.keys() if tid != -1]
        LOGGER.info("Topic IDs in topic_representations_: %s", sorted(topic_ids)[:20])
        LOGGER.info("Total topics (excluding -1): %d", len(topic_ids))
    
    if hasattr(topic_model, "get_topic_info"):
        try:
            topic_info = topic_model.get_topic_info()
            LOGGER.info("Topic info shape: %s", topic_info.shape if hasattr(topic_info, 'shape') else 'N/A')
            LOGGER.info("Topic info columns: %s", list(topic_info.columns) if hasattr(topic_info, 'columns') else 'N/A')
            if hasattr(topic_info, 'shape'):
                LOGGER.info("Number of topics from get_topic_info(): %d", len(topic_info))
        except Exception as e:
            LOGGER.warning("Could not get topic_info: %s", e)
    
    dictionary = load_dictionary_from_corpus(
        corpus_path=args.dictionary_path,
        batch_size=args.batch_size,
    )

    representations = build_representation_models(language_model=args.language_model)
    apply_representations_and_update(topic_model, docs, representations)
    
    # Diagnostic: Check model state after updates
    LOGGER.info("=== Model Diagnostics (After Update) ===")
    if hasattr(topic_model, "topic_representations_") and topic_model.topic_representations_:
        topic_ids = [tid for tid in topic_model.topic_representations_.keys() if tid != -1]
        LOGGER.info("Topic IDs in topic_representations_: %s", sorted(topic_ids)[:20])
        LOGGER.info("Total topics (excluding -1): %d", len(topic_ids))

    metrics = evaluate_representations(
        topic_model,
        docs_tokens,
        dictionary=dictionary,
        top_k=args.top_k,
    )

    print("\n=== Representation Metrics ===")
    for row in metrics:
        print(
            f"{row['representation']:>10} | "
            f"topics={row['n_topics']:>4} | "
            f"c_v={row['coherence_c_v']:.4f} | "
            f"diversity={row['topic_diversity']:.4f}"
        )

    print(f"\nLoaded documents: {len(docs)}")
    
    # Create model-specific output filenames to preserve results for each model
    # Sanitize embedding model name for use in filenames
    model_name_safe = args.embedding_model.replace("/", "_").replace("\\", "_")
    metrics_filename = f"metrics_{model_name_safe}"
    topics_filename = f"topics_all_representations_{model_name_safe}"
    
    # Save metrics to file (always save)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / metrics_filename
    
    save_metrics(metrics, metrics_path, format=args.metrics_format)
    LOGGER.info("Metrics saved to %s", metrics_path.with_suffix(f".{args.metrics_format}"))
    
    # Extract and save topics for close reading (if requested)
    if args.save_topics:
        all_topics = extract_all_topics(topic_model, top_k=args.top_k)
        topics_path = output_dir / topics_filename
        save_topics(all_topics, topics_path)
        LOGGER.info("Topics extracted and saved for close reading evaluation")


if __name__ == "__main__":
    main()
