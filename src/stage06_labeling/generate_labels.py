"""Stage 06 utilities for generating topic labels using FLAN-T5 from POS representation keywords."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from bertopic import BERTopic
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.stage05_retraining.retrain_models import RetrainableBERTopicModel
from src.stage06_BERTopic_topics_exploration.explore_retrained_model import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    load_native_bertopic_model,
    load_retrained_wrapper,
)

LOGGER = logging.getLogger("stage06_labeling")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)


@contextmanager
def stage_timer_local(stage_name: str):
    """Context manager that logs start/end timestamps for each processing stage."""
    LOGGER.info("▶ %s | start", stage_name)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        LOGGER.info("■ %s | completed in %.2f s", stage_name, elapsed)


def load_bertopic_model(
    base_dir: Path | str = DEFAULT_BASE_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    pareto_rank: int = 1,
    use_native: bool = False,
) -> tuple[RetrainableBERTopicModel | None, BERTopic]:
    """
    Load the retrained BERTopic model from either pickle wrapper or native safetensors.
    
    Args:
        base_dir: Base directory for models
        embedding_model: Model name
        pareto_rank: Model rank
        use_native: If True, load native safetensors; if False, load pickle wrapper
        
    Returns:
        Tuple of (wrapper or None, BERTopic model)
    """
    if use_native:
        topic_model = load_native_bertopic_model(
            base_dir=base_dir,
            embedding_model=embedding_model,
            pareto_rank=pareto_rank,
        )
        return None, topic_model
    else:
        wrapper, topic_model = load_retrained_wrapper(
            base_dir=base_dir,
            embedding_model=embedding_model,
            pareto_rank=pareto_rank,
        )
        return wrapper, topic_model


def extract_pos_topics_from_json(
    json_path: Path,
    top_k: int = 8,
) -> Iterator[tuple[int, list[str]]]:
    """
    Extract POS topics from JSON file in a streaming fashion (memory-efficient).
    
    This function reads the JSON file and yields topics one at a time without
    loading the entire file into memory.
    
    Args:
        json_path: Path to topics JSON file
        top_k: Number of top keywords to extract per topic
        
    Yields:
        Tuples of (topic_id, keywords_list)
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Topics JSON file not found: {json_path}")
    
    with stage_timer_local(f"Streaming POS topics from JSON: {json_path.name}"):
        with open(json_path, "r", encoding="utf-8") as f:
            # Load JSON file (unavoidable for nested structure, but we yield immediately)
            # Note: This loads the JSON into memory, but JSON files are typically much smaller
            # than BERTopic models (e.g., 1.1MB vs 4.7GB). For truly large JSON files,
            # consider using ijson library for streaming JSON parsing.
            data = json.load(f)
        
        if "POS" not in data:
            raise ValueError(
                f"POS representation not found in JSON file: {json_path}. "
                "Please ensure the file contains POS representation."
            )
        
        pos_data = data["POS"]
        topic_count = 0
        
        for topic_id_str, topic_content in pos_data.items():
            try:
                topic_id = int(topic_id_str)
            except ValueError:
                continue  # Skip non-numeric topic IDs
            
            if topic_id == -1:
                continue  # Skip outlier topic
            
            keywords: list[str] = []
            for item in topic_content[:top_k]:
                if isinstance(item, dict) and "word" in item:
                    keywords.append(str(item["word"]))
                elif isinstance(item, str):
                    keywords.append(item)
            
            if keywords:
                topic_count += 1
                yield (topic_id, keywords)
        
        LOGGER.info(
            "Streamed POS keywords for %d topics (top_k=%d)",
            topic_count,
            top_k,
        )


def extract_pos_topics(
    topic_model: BERTopic,
    top_k: int = 8,
    limit: int | None = None,
) -> dict[int, list[str]]:
    """
    Extract top keywords from POS representation for each topic.
    
    WARNING: This loads all topics into memory. For memory efficiency,
    use extract_pos_topics_from_json() instead.
    
    Args:
        topic_model: Loaded BERTopic model
        top_k: Number of top keywords to extract per topic
        limit: Maximum number of topics to extract (None for all)
        
    Returns:
        Dictionary mapping topic_id to list of keyword strings
    """
    pos_topics: dict[int, list[str]] = {}
    
    # Check if POS representation exists
    if not hasattr(topic_model, "topic_aspects_") or not topic_model.topic_aspects_:
        raise ValueError(
            "Topic model does not have topic_aspects_. "
            "Please run explore_retrained_model.py with --save-topics first to generate representations."
        )
    
    if "POS" not in topic_model.topic_aspects_:
        raise ValueError(
            "POS representation not found in topic_aspects_. "
            "Please ensure POS representation was generated during topic exploration."
        )
    
    pos_representation = topic_model.topic_aspects_["POS"]
    
    with stage_timer_local("Extracting POS topics"):
        topic_count = 0
        for topic_id, topic_content in pos_representation.items():
            if topic_id == -1:
                continue  # Skip outlier topic
            
            if limit is not None and topic_count >= limit:
                break
            
            keywords: list[str] = []
            for item in topic_content[:top_k]:
                if isinstance(item, tuple) and len(item) > 0:
                    keywords.append(str(item[0]))
                elif isinstance(item, str):
                    keywords.append(item)
            
            if keywords:
                pos_topics[topic_id] = keywords
                topic_count += 1
        
        LOGGER.info(
            "Extracted POS keywords for %d topics (top_k=%d, limit=%s)",
            len(pos_topics),
            top_k,
            limit,
        )
    
    return pos_topics


def load_labeling_model(
    model_name: str = "google/flan-t5-small",
    device: str | None = None,
) -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    """
    Load FLAN-T5 tokenizer and model for label generation.
    
    Args:
        model_name: Hugging Face model name
        device: Device to use ('cuda', 'cpu', or None for auto-detection)
        
    Returns:
        Tuple of (tokenizer, model)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with stage_timer_local(f"Loading labeling model: {model_name}"):
        LOGGER.info("Loading tokenizer and model on device: %s", device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        LOGGER.info("Model loaded successfully")
    
    return tokenizer, model


def generate_label_from_keywords(
    keywords: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    max_new_tokens: int = 16,
    device: str | None = None,
) -> str:
    """
    Generate a topic label from keywords using FLAN-T5.
    
    Args:
        keywords: List of keyword strings
        tokenizer: FLAN-T5 tokenizer
        model: FLAN-T5 model
        max_new_tokens: Maximum number of tokens to generate
        device: Device to use ('cuda', 'cpu', or None for auto-detection)
        
    Returns:
        Generated label string
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create improved prompt for more descriptive labels
    kw_str = ", ".join(keywords)
    prompt = (
        "Given these keywords that describe a topic: "
        f"{kw_str}. "
        "Generate a concise and descriptive label that summarizes the main idea of the topic. "
        "The label should be a short noun phrase (2-4 words maximum)."
    )
    
    # Tokenize and generate
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
            )
        
        label = tokenizer.decode(outputs[0], skip_special_tokens=True)
        label = label.strip()
        
        # Post-process label to clean it up
        # Remove trailing commas, periods, and incomplete words
        label = label.rstrip(".,;")
        # If label is too long or seems incomplete, truncate at first comma/semicolon
        if "," in label or ";" in label:
            label = label.split(",")[0].split(";")[0].strip()
        # If label is just keywords repeated, try to extract a meaningful phrase
        if len(label.split()) > 6:  # Too many words, likely just keyword list
            # Take first few meaningful words
            words = label.split()[:4]
            label = " ".join(words)
        
        return label
    except Exception as e:
        LOGGER.warning("Error generating label for keywords %s: %s", keywords[:3], e)
        # Fallback: create a simple label from first few keywords
        return f"{keywords[0]}" if keywords else "Topic"


def log_batch_progress(stage: str, batch_idx: int, start: int, end: int, total: int):
    """Emit detailed progress logs for batched operations."""
    LOGGER.info(
        "%s | batch %d => topics %d-%d / %d",
        stage,
        batch_idx,
        start,
        end,
        total,
    )


def generate_labels_streaming(
    pos_topics_iter: Iterator[tuple[int, list[str]]],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    output_path: Path,
    max_new_tokens: int = 16,
    device: str | None = None,
    batch_size: int = 50,
) -> dict[int, str]:
    """
    Generate labels for topics in a streaming fashion and write incrementally to JSON.
    
    This function processes topics one at a time, generates labels, and writes them
    incrementally to avoid keeping all labels in memory at once.
    
    Args:
        pos_topics_iter: Iterator yielding (topic_id, keywords) tuples
        tokenizer: FLAN-T5 tokenizer
        model: FLAN-T5 model
        output_path: Path to save JSON file (without extension)
        max_new_tokens: Maximum number of tokens to generate per label
        device: Device to use ('cuda', 'cpu', or None for auto-detection)
        batch_size: Number of topics to process before logging progress
        
    Returns:
        Dictionary mapping topic_id to generated label (for integration if needed)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    json_path = output_path.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    topic_labels: dict[int, str] = {}
    batch_idx = 0
    processed_count = 0
    
    with stage_timer_local("Generating labels (streaming)"):
        # Open JSON file for incremental writing
        with open(json_path, "w", encoding="utf-8") as f:
            f.write("{\n")
            first_item = True
            
            for topic_id, keywords in pos_topics_iter:
                processed_count += 1
                
                # Generate label
                label = generate_label_from_keywords(
                    keywords=keywords,
                    tokenizer=tokenizer,
                    model=model,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                topic_labels[topic_id] = label
                
                # Write to JSON incrementally
                if not first_item:
                    f.write(",\n")
                else:
                    first_item = False
                
                # Write this topic's label
                f.write(f'  "{topic_id}": {json.dumps(label, ensure_ascii=False)}')
                f.flush()  # Ensure immediate write to disk
                
                # Log progress every batch_size topics
                if processed_count % batch_size == 0:
                    batch_idx += 1
                    start_idx = processed_count - batch_size + 1
                    end_idx = processed_count
                    log_batch_progress(
                        "Label generation (streaming)",
                        batch_idx,
                        start_idx,
                        end_idx,
                        -1,  # Total unknown for streaming
                    )
                    LOGGER.info(
                        "Progress: %d topics processed, %d labels written to disk",
                        processed_count,
                        processed_count,
                    )
            
            f.write("\n}\n")
            f.flush()
        
        LOGGER.info(
            "Successfully generated and saved %d labels to %s",
            processed_count,
            json_path,
        )
    
    return topic_labels


def generate_all_labels(
    pos_topics: dict[int, list[str]],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    max_new_tokens: int = 16,
    device: str | None = None,
    batch_size: int = 50,
) -> dict[int, str]:
    """
    Generate labels for all topics from POS keywords in batches.
    
    WARNING: This keeps all labels in memory. For memory efficiency,
    use generate_labels_streaming() instead.
    
    Args:
        pos_topics: Dictionary mapping topic_id to list of keywords
        tokenizer: FLAN-T5 tokenizer
        model: FLAN-T5 model
        max_new_tokens: Maximum number of tokens to generate per label
        device: Device to use ('cuda', 'cpu', or None for auto-detection)
        batch_size: Number of topics to process before logging progress
        
    Returns:
        Dictionary mapping topic_id to generated label
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    topic_labels: dict[int, str] = {}
    topic_items = list(pos_topics.items())
    total_topics = len(topic_items)
    
    with stage_timer_local("Generating labels for all topics"):
        LOGGER.info("Generating labels for %d topics (batch_size=%d)", total_topics, batch_size)
        
        # Process in batches
        batch_idx = 0
        batch_labels: dict[int, str] = {}
        
        for idx, (topic_id, keywords) in enumerate(topic_items, start=1):
            label = generate_label_from_keywords(
                keywords=keywords,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            batch_labels[topic_id] = label
            
            # Log progress every batch_size topics or at the end
            if idx % batch_size == 0 or idx == total_topics:
                batch_idx += 1
                start_idx = idx - len(batch_labels) + 1
                end_idx = idx
                log_batch_progress(
                    "Label generation",
                    batch_idx,
                    start_idx,
                    end_idx,
                    total_topics,
                )
                topic_labels.update(batch_labels)
                batch_labels.clear()
        
        # Flush any remaining labels
        if batch_labels:
            topic_labels.update(batch_labels)
        
        LOGGER.info("Successfully generated labels for %d topics", len(topic_labels))
    
    return topic_labels


def save_labels(
    topic_labels: dict[int, str],
    output_path: Path,
) -> None:
    """
    Save topic labels to JSON file.
    
    Args:
        topic_labels: Dictionary mapping topic_id to label
        output_path: Path to save JSON file (without extension)
    """
    json_path = output_path.with_suffix(".json")
    
    with stage_timer_local(f"Saving labels to JSON: {json_path.name}"):
        # Convert topic IDs to strings for JSON serialization
        labels_serializable: dict[str, str] = {
            str(topic_id): label for topic_id, label in topic_labels.items()
        }
        
        # Create parent directory if it doesn't exist
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(labels_serializable, f, indent=2, ensure_ascii=False)
        
        LOGGER.info(
            "Saved %d topic labels to %s",
            len(labels_serializable),
            json_path,
        )


def compare_topics_sources(
    bertopic_topics: dict[int, list[str]],
    json_topics: dict[int, list[str]],
) -> dict[str, Any]:
    """
    Compare topics extracted from BERTopic model vs JSON file for validation.
    
    Args:
        bertopic_topics: Topics from BERTopic model
        json_topics: Topics from JSON file (as dictionary)
        
    Returns:
        Dictionary with comparison statistics
    """
    
    bertopic_ids = set(bertopic_topics.keys())
    json_ids = set(json_topics.keys())
    
    common_ids = bertopic_ids & json_ids
    only_bertopic = bertopic_ids - json_ids
    only_json = json_ids - bertopic_ids
    
    # Compare keywords for common topics
    keyword_matches = 0
    keyword_differences = 0
    
    for topic_id in common_ids:
        bertopic_kw = set(bertopic_topics[topic_id])
        json_kw = set(json_topics[topic_id])
        if bertopic_kw == json_kw:
            keyword_matches += 1
        else:
            keyword_differences += 1
    
    comparison = {
        "bertopic_topics_count": len(bertopic_topics),
        "json_topics_count": len(json_topics),
        "common_topics": len(common_ids),
        "only_in_bertopic": len(only_bertopic),
        "only_in_json": len(only_json),
        "keyword_matches": keyword_matches,
        "keyword_differences": keyword_differences,
    }
    
    return comparison


def integrate_labels_to_bertopic(
    topic_model: BERTopic,
    topic_labels: dict[int, str],
) -> None:
    """
    Integrate generated labels back into BERTopic model.
    
    Args:
        topic_model: BERTopic model instance
        topic_labels: Dictionary mapping topic_id to label
    """
    with stage_timer_local("Integrating labels into BERTopic"):
        try:
            # Convert integer keys to match BERTopic's expected format
            # BERTopic's set_topic_labels expects a dict with int keys
            topic_model.set_topic_labels(topic_labels)
            LOGGER.info(
                "Successfully integrated %d labels into BERTopic model",
                len(topic_labels),
            )
        except Exception as e:
            LOGGER.error("Error integrating labels into BERTopic: %s", e)
            raise

