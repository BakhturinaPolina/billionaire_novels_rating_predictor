"""Stage 06 utilities for generating topic labels using OpenRouter API (mistralai/mistral-nemo) from POS representation keywords."""

from __future__ import annotations

import json
import os
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Import shared utilities from parent module
from src.stage06_labeling.generate_labels import (
    UNIVERSAL_SYSTEM_PROMPT,
    UNIVERSAL_USER_PROMPT,
    detect_domains,
    extract_pos_topics,
    extract_pos_topics_from_json,
    integrate_labels_to_bertopic,
    load_bertopic_model,
    log_batch_progress,
    make_context_hints,
    rerank_keywords_mmr,
    stage_timer_local,
)

LOGGER = logging.getLogger("stage06_labeling.openrouter")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)

# OpenRouter API configuration
DEFAULT_OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
DEFAULT_OPENROUTER_MODEL = "mistralai/mistral-nemo"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Module-level cache for MMR embedding model (loaded once, reused for all topics)
_MMR_EMBEDDING_MODEL: SentenceTransformer | None = None


def load_openrouter_client(
    api_key: str = DEFAULT_OPENROUTER_API_KEY,
    model_name: str = DEFAULT_OPENROUTER_MODEL,
    base_url: str = DEFAULT_OPENROUTER_BASE_URL,
    timeout: int = 60,
) -> tuple[OpenAI, str]:
    """
    Load OpenRouter API client for label generation.
    
    Args:
        api_key: OpenRouter API key
        model_name: Model name to use (e.g., 'mistralai/mistral-nemo')
        base_url: OpenRouter API base URL
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (OpenAI client, model_name)
    """
    with stage_timer_local(f"Initializing OpenRouter client: {model_name}"):
        LOGGER.info("Initializing OpenRouter API client")
        LOGGER.info("Model: %s", model_name)
        LOGGER.info("Base URL: %s", base_url)
        
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        
        LOGGER.info("✓ OpenRouter client initialized successfully")
    
    return client, model_name


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((Exception,)),
    reraise=True,
)
def generate_label_from_keywords_openrouter(
    keywords: list[str],
    client: OpenAI,
    model_name: str,
    max_new_tokens: int = 40,
    use_mmr_reranking: bool = False,
    mmr_diversity: float = 0.5,
    mmr_top_k: int | None = None,
    temperature: float = 0.3,
) -> str:
    """
    Generate a topic label from keywords using OpenRouter API.
    
    Args:
        keywords: List of keyword strings
        client: OpenRouter OpenAI client
        model_name: Model name to use
        max_new_tokens: Maximum number of tokens to generate
        use_mmr_reranking: If True, rerank keywords using MMR for diversity
        mmr_diversity: Diversity parameter for MMR (0.0-1.0)
        mmr_top_k: Maximum keywords to use after MMR reranking (None for all)
        temperature: Sampling temperature for generation
        
    Returns:
        Generated label string
    """
    # Apply MMR reranking for diversity if requested
    if use_mmr_reranking and len(keywords) > 1:
        keywords = rerank_keywords_mmr(
            keywords,
            embedding_model=None,  # Will create lightweight model on first call
            top_k=mmr_top_k,
            diversity=mmr_diversity,
        )
    
    # Build adaptive universal prompt
    kw_str = ", ".join(keywords)
    domains = detect_domains(keywords)
    hints = make_context_hints(domains)
    user_prompt = UNIVERSAL_USER_PROMPT.format(kw=kw_str, hints=("\n" + hints if hints else ""))
    messages = [
        {"role": "system", "content": UNIVERSAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    # Call OpenRouter API with timing
    try:
        api_start = time.perf_counter()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        api_elapsed = time.perf_counter() - api_start
        
        if not response.choices:
            raise ValueError("Empty API response")
        
        # Log API response details
        usage = getattr(response, 'usage', None)
        if usage:
            prompt_tokens = getattr(usage, 'prompt_tokens', 0)
            completion_tokens = getattr(usage, 'completion_tokens', 0)
            total_tokens = getattr(usage, 'total_tokens', 0)
            LOGGER.info(
                "API call | time=%.2fs | tokens: prompt=%d, completion=%d, total=%d",
                api_elapsed,
                prompt_tokens,
                completion_tokens,
                total_tokens,
            )
        else:
            LOGGER.info("API call | time=%.2fs", api_elapsed)
        
        label = response.choices[0].message.content.strip()
        LOGGER.debug("Raw API response: %s", label[:100] if len(label) > 100 else label)
        
        # Post-process label to clean it up
        # Strip leading/trailing quotation marks
        label = label.strip().strip('"').strip("'")
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
        
        LOGGER.info("Generated label: %s", label)
        return label
    except Exception as e:
        LOGGER.warning("Error generating label for keywords %s: %s", keywords[:3], e)
        LOGGER.exception("Full error traceback:")
        # Fallback: create a simple label from first few keywords
        fallback_label = f"{keywords[0]}" if keywords else "Topic"
        LOGGER.info("Using fallback label: %s", fallback_label)
        return fallback_label


def generate_labels_streaming(
    pos_topics_iter: Iterator[tuple[int, list[str]]],
    client: OpenAI,
    model_name: str,
    output_path: Path,
    max_new_tokens: int = 40,
    batch_size: int = 50,
    temperature: float = 0.3,
    limit: int | None = None,
) -> dict[int, str]:
    """
    Generate labels for topics in a streaming fashion and write incrementally to JSON.
    
    This function processes topics one at a time, generates labels, and writes them
    incrementally to avoid keeping all labels in memory at once.
    
    Args:
        pos_topics_iter: Iterator yielding (topic_id, keywords) tuples
        client: OpenRouter OpenAI client
        model_name: Model name to use
        output_path: Path to save JSON file (without extension)
        max_new_tokens: Maximum number of tokens to generate per label
        batch_size: Number of topics to process before logging progress
        temperature: Sampling temperature for generation
        limit: Maximum number of topics to process (None for all)
        
    Returns:
        Dictionary mapping topic_id to dict with 'label' and 'keywords' keys
    """
    json_path = output_path.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    topic_data: dict[int, dict[str, Any]] = {}
    batch_idx = 0
    processed_count = 0
    
    with stage_timer_local("Generating labels (streaming)"):
        # Open JSON file for incremental writing
        with open(json_path, "w", encoding="utf-8") as f:
            f.write("{\n")
            first_item = True
            
            for topic_id, keywords in pos_topics_iter:
                # Apply limit if specified
                if limit is not None and processed_count >= limit:
                    LOGGER.info("Reached topic limit (%d), stopping processing", limit)
                    break
                
                processed_count += 1
                
                # Telemetry: detect domains for this topic (one line)
                domains = detect_domains(keywords)
                LOGGER.info("topic %d | domains=%s | keywords=%s", 
                           topic_id, 
                           ",".join(domains) if domains else "None",
                           ", ".join(keywords[:5]) + ("..." if len(keywords) > 5 else ""))
                
                # Generate label with timing
                label_start = time.perf_counter()
                label = generate_label_from_keywords_openrouter(
                    keywords=keywords,
                    client=client,
                    model_name=model_name,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                label_elapsed = time.perf_counter() - label_start
                
                # Store both label and keywords
                topic_data[topic_id] = {
                    "label": label,
                    "keywords": keywords
                }
                
                LOGGER.info("topic %d | label='%s' | generation_time=%.2fs", 
                           topic_id, label, label_elapsed)
                
                # Write to JSON incrementally with new structure
                if not first_item:
                    f.write(",\n")
                else:
                    first_item = False
                
                # Write this topic's data (label and keywords)
                topic_entry = {
                    "label": label,
                    "keywords": keywords
                }
                # Format JSON entry (compact for incremental writing)
                entry_json = json.dumps(topic_entry, ensure_ascii=False)
                f.write(f'  "{topic_id}": {entry_json}')
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
                
                # Small delay to avoid rate limits
                time.sleep(0.5)
            
            f.write("\n}\n")
            f.flush()
        
        LOGGER.info(
            "Successfully generated and saved %d topic entries to %s",
            processed_count,
            json_path,
        )
    
    return topic_data


def generate_all_labels(
    pos_topics: dict[int, list[str]],
    client: OpenAI,
    model_name: str,
    max_new_tokens: int = 40,
    batch_size: int = 50,
    temperature: float = 0.3,
) -> dict[int, str]:
    """
    Generate labels for all topics from POS keywords in batches.
    
    WARNING: This keeps all labels in memory. For memory efficiency,
    use generate_labels_streaming() instead.
    
    Args:
        pos_topics: Dictionary mapping topic_id to list of keywords
        client: OpenRouter OpenAI client
        model_name: Model name to use
        max_new_tokens: Maximum number of tokens to generate per label
        batch_size: Number of topics to process before logging progress
        temperature: Sampling temperature for generation
        
    Returns:
        Dictionary mapping topic_id to dict with 'label' and 'keywords' keys
    """
    topic_data: dict[int, dict[str, Any]] = {}
    topic_items = list(pos_topics.items())
    total_topics = len(topic_items)
    
    with stage_timer_local("Generating labels for all topics"):
        LOGGER.info("Generating labels for %d topics (batch_size=%d)", total_topics, batch_size)
        
        # Process in batches
        batch_idx = 0
        batch_labels: dict[int, str] = {}
        
        for idx, (topic_id, keywords) in enumerate(topic_items, start=1):
            # Telemetry: detect domains for this topic (one line)
            domains = detect_domains(keywords)
            LOGGER.info("topic %d | domains=%s | keywords=%s", 
                       topic_id, 
                       ",".join(domains) if domains else "None",
                       ", ".join(keywords[:5]) + ("..." if len(keywords) > 5 else ""))
            
            # Generate label with timing
            label_start = time.perf_counter()
            label = generate_label_from_keywords_openrouter(
                keywords=keywords,
                client=client,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            label_elapsed = time.perf_counter() - label_start
            
            # Store both label and keywords
            batch_labels[topic_id] = {
                "label": label,
                "keywords": keywords
            }
            
            LOGGER.info("topic %d | label='%s' | generation_time=%.2fs", 
                       topic_id, label, label_elapsed)
            
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
                LOGGER.info(
                    "Batch %d completed: %d topics processed (%.1f%% complete)",
                    batch_idx,
                    idx,
                    (idx / total_topics * 100) if total_topics > 0 else 0,
                )
                topic_data.update(batch_labels)
                batch_labels.clear()
            
            # Small delay to avoid rate limits
            time.sleep(0.5)
        
        # Flush any remaining labels
        if batch_labels:
            topic_data.update(batch_labels)
        
        LOGGER.info("Successfully generated labels for %d topics", len(topic_data))
    
    return topic_data


def save_labels_openrouter(
    topic_data: dict[int, dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Save topic labels and keywords to JSON file with new structure.
    
    Args:
        topic_data: Dictionary mapping topic_id to dict with 'label' and 'keywords' keys
        output_path: Path to save JSON file (without extension)
    """
    json_path = output_path.with_suffix(".json")
    
    with stage_timer_local(f"Saving labels to JSON: {json_path.name}"):
        # Convert topic IDs to strings for JSON serialization
        data_serializable: dict[str, dict[str, Any]] = {
            str(topic_id): data for topic_id, data in topic_data.items()
        }
        
        # Create parent directory if it doesn't exist
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data_serializable, f, indent=2, ensure_ascii=False)
        
        LOGGER.info(
            "Saved %d topic entries to %s",
            len(data_serializable),
            json_path,
        )

