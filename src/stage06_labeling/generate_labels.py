"""Stage 06 utilities for generating topic labels using Mistral-7B-Instruct from POS representation keywords."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

# Universal prompt for topic labeling (works across any domain)
UNIVERSAL_SYSTEM_PROMPT = """You are a topic-labeling assistant.

Goal

- Produce a short, specific label for a cluster of keywords from a corpus.

- The label must help distinguish this topic from similar ones.

Style

- One short noun phrase, 2–6 words. No quotes. No trailing punctuation.

- Prefer specific entities or attributes over generic abstractions.

- If the keywords imply a facet (body part, activity, time period, object, place, emotion, event, relationship, profession, product, sport, etc.), name it explicitly.

- Use disambiguators only when clear (e.g., "Foreplay", "Explicit", "Wedding Planning", "Doorways & Elevators", "Wine Service").

- Avoid simply repeating a list of keywords; synthesize the core idea instead.

Consistency

- Use title casing where natural (not ALL CAPS).

- Avoid numbers unless essential (e.g., "24-Hour Timeline").

Output

- Return only the label, nothing else."""

UNIVERSAL_USER_PROMPT = """Topic keywords: {kw}{hints}

Label:"""

# Module-level cache for MMR embedding model (loaded once, reused for all topics)
_MMR_EMBEDDING_MODEL: SentenceTransformer | None = None


# Domain detection lexicon for adaptive hints
_DOMAIN_LEX = {
    "BodyParts": {
        "lip", "lips", "mouth", "tongue", "teeth", "cheek", "cheeks", "nose", "chin", "brows", "eyebrow", "eyebrows",
        "eye", "eyes", "neck", "nape", "shoulder", "shoulders", "arm", "arms", "hand", "hands", "finger", "fingers",
        "fist", "fists", "breast", "breasts", "nipples", "waist", "belly", "stomach", "chest", "spine", "back",
        "hip", "hips", "thigh", "thighs", "legs", "leg", "knee", "knees", "feet", "foot", "heels", "clit", "clitoris",
        "pussy", "genitals", "hair", "hairs", "saliva", "moan", "moans",
    },
    "FoodDrink": {"dinner", "lunch", "breakfast", "bakery", "chicken", "sandwich", "meal", "dessert", "hungry",
                  "wine", "bottle", "whiskey", "scotch", "cup", "drinks", "sip", "waiter", "waitress", "kitchen", "plate", "mug"},
    "TimeSpan":  {"minutes", "minute", "hours", "hour", "days", "day", "weeks", "week", "months", "month", "years", "year", "seconds", "second", "tonight", "night", "evening", "nights"},
    "Marriage":  {"wedding", "married", "marriage", "divorce", "divorced", "ceremony", "planner", "honeymoon", "aisle"},
    "Building":  {"door", "doors", "knock", "hallway", "stairs", "elevators", "floor", "lock", "button", "lobby", "knob"},
    "FacialExpr":{"smile", "smiles", "grin", "grins", "laugh", "laughs", "laughter", "giggles", "gaze", "stare", "stares", "glare", "brows", "eyebrows", "expression", "features"},
    "Clothing":  {"dress", "dresses", "panties", "skirt", "lace", "heels", "gown", "stilettos", "blouse", "underwear", "outfit", "clothes", "wardrobe", "clothing"},
    "Hygiene":   {"shower", "showers", "bath", "bathroom", "bathtub", "tub", "toilet", "restroom", "stall", "spray"},
    "Sports":    {"hockey", "puck", "players", "player", "teams", "goalie", "goal", "game", "games", "baseball"},
    "Business":  {"business", "company", "manager", "ceos", "bosses", "board", "profit", "income", "investment", "paycheck", "bank"},
    "Emotion":   {"sorry", "apology", "forgive", "forgiveness", "guilt", "guilty", "hurts", "painful", "agony", "ache", "ashamed", "scared", "proud", "ridiculous", "insane", "absurd"},
    "Phones":    {"cellphone", "phones", "pocket", "screen", "message", "messages", "ringing", "rang", "rings", "voicemail", "buzzes"},
}


def detect_domains(keywords):
    """Detect which domains are present in the keywords.
    
    Args:
        keywords: List of keyword strings
        
    Returns:
        List of detected domain names (empty if none detected)
    """
    toks = [re.sub(r"[^a-z]+", "", k.lower()) for k in keywords if k]
    hits = {name: 0 for name in _DOMAIN_LEX}
    for t in toks:
        for name, vocab in _DOMAIN_LEX.items():
            if t in vocab:
                hits[name] += 1
    
    # Keep domains with at least 2 hits or >25% coverage (whichever is easier)
    keep = []
    for name, count in hits.items():
        if count >= 2 or (toks and count / len(toks) >= 0.25):
            keep.append(name)
    
    return keep


def make_context_hints(domains):
    """Generate context hints text from detected domains.
    
    Args:
        domains: List of detected domain names
        
    Returns:
        Hint text string (empty if no domains detected)
    """
    if not domains:
        return ""
    
    # map domain → short hint text
    hints_map = {
        "BodyParts": "If body parts or intimacy are clear, name the exact parts; add 'Foreplay' or 'Explicit' only when unambiguous.",
        "FoodDrink": "If food or alcohol service is clear, name the item or service explicitly.",
        "TimeSpan":  "If temporal granularity is clear, name the unit (e.g., 'Minutes', 'Weeks').",
        "Marriage":  "If marriage lifecycle is clear, prefer 'Wedding Planning' vs 'Divorce' etc.",
        "Building":  "If access/transition areas are clear, prefer precise nouns (e.g., 'Doorways', 'Elevators').",
        "FacialExpr":"Prefer concrete expressions (e.g., 'Playful Laughter', 'Gaze & Brows').",
        "Clothing":  "If apparel is dominant, name the garment(s).",
        "Hygiene":   "If hygiene facilities dominate, use the specific facility.",
        "Sports":    "If a sport is evident, name the sport and object/role if clear.",
        "Business":  "If business context dominates, name the function (e.g., 'Company Management', 'Profit & Costs').",
        "Emotion":   "If emotion dominates, use the core affect (e.g., 'Apology & Forgiveness').",
        "Phones":    "If phones/notifications dominate, name the artifact or action (e.g., 'Phone Notifications').",
    }
    
    lines = [hints_map[d] for d in domains if d in hints_map]
    return "\nContext hints: " + " ".join(lines)


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
                word = None
                if isinstance(item, dict) and "word" in item:
                    word = str(item["word"]).strip()
                elif isinstance(item, str):
                    word = item.strip()
                
                # Filter out empty keywords
                if word:
                    keywords.append(word)
            
            # Skip topics with no valid keywords
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
                word = None
                if isinstance(item, tuple) and len(item) > 0:
                    word = str(item[0]).strip()
                elif isinstance(item, str):
                    word = item.strip()
                
                # Filter out empty keywords
                if word:
                    keywords.append(word)
            
            # Skip topics with no valid keywords
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
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    device: str | None = None,
    use_quantization: bool = True,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load Mistral-7B-Instruct tokenizer and model for label generation with 4-bit quantization.
    
    Enforces GPU usage when CUDA is available. Falls back to CPU only if CUDA is not available.
    
    Args:
        model_name: Hugging Face model name
        device: Device to use ('cuda', 'cpu', or None for auto-detection with GPU preference)
        use_quantization: If True, use 4-bit quantization (recommended for 12-16GB VRAM)
        
    Returns:
        Tuple of (tokenizer, model)
    """
    # Enforce GPU usage when available
    cuda_available = torch.cuda.is_available()
    if device is None:
        device = "cuda" if cuda_available else "cpu"
    elif device == "cpu" and cuda_available:
        LOGGER.warning(
            "CUDA is available but device='cpu' was specified. "
            "Using CPU may be very slow. Consider using 'cuda' for better performance."
        )
    
    if not cuda_available and device == "cuda":
        LOGGER.warning(
            "CUDA is not available but device='cuda' was requested. "
            "Falling back to CPU. This will be very slow."
        )
        device = "cpu"
        use_quantization = False  # Quantization requires CUDA
    
    with stage_timer_local(f"Loading labeling model: {model_name}"):
        LOGGER.info("Loading tokenizer and model on device: %s", device)
        if cuda_available and device == "cuda":
            LOGGER.info("GPU detected: %s (VRAM: %.2f GB)", 
                       torch.cuda.get_device_name(0),
                       torch.cuda.get_device_properties(0).total_memory / 1e9)
        
        # Check if model is cached
        is_cached, cache_path = check_model_cache_status(model_name)
        if is_cached and cache_path:
            # Quick size estimate (check a few key files instead of all files)
            try:
                # Estimate size from a few large files
                total_size = 0
                file_count = 0
                for f in cache_path.rglob('*.safetensors'):
                    total_size += f.stat().st_size
                    file_count += 1
                    if file_count >= 3:  # Sample first few files
                        break
                # Rough estimate: multiply by number of shards if we see multiple files
                if file_count > 0:
                    # Estimate total by checking if there are more files
                    all_files = list(cache_path.rglob('*.safetensors'))
                    if len(all_files) > file_count:
                        # Rough estimate: average size * total files
                        avg_size = total_size / file_count
                        total_size = avg_size * len(all_files)
                    cache_size_gb = total_size / (1024**3)
                    LOGGER.info("✓ Model found in cache: %s (~%.2f GB estimated)", cache_path, cache_size_gb)
                else:
                    LOGGER.info("✓ Model found in cache: %s", cache_path)
            except Exception as e:
                LOGGER.info("✓ Model found in cache: %s", cache_path)
            LOGGER.info("  Loading from local cache (no download needed)")
        else:
            LOGGER.info("⚠ Model not found in cache - will download from Hugging Face")
            LOGGER.info("  This may take several minutes depending on your internet connection")
            LOGGER.info("  Model will be cached for future use")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not present (Mistral doesn't have one by default)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if use_quantization and device == "cuda":
            # Configure 4-bit quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            LOGGER.info("Loading model with 4-bit quantization on GPU")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",  # Automatically uses GPU when available
                torch_dtype=torch.float16,
            )
            # Verify model is on GPU
            try:
                model_device = next(model.parameters()).device
                if model_device.type == "cuda":
                    LOGGER.info("Model successfully loaded on GPU: %s", model_device)
                else:
                    LOGGER.warning("Model loaded on %s instead of GPU", model_device)
            except StopIteration:
                LOGGER.warning("Could not verify model device placement")
        else:
            if use_quantization:
                LOGGER.warning("Quantization requested but CUDA not available. Loading without quantization.")
            LOGGER.info("Loading model without quantization on device: %s", device)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model = model.to(device)
            if device == "cuda":
                model = model.half()  # Use float16 on GPU
                LOGGER.info("Model loaded on GPU with float16 precision")
            else:
                LOGGER.warning("Model loaded on CPU - this will be very slow for inference")
        
        model.eval()  # Set to evaluation mode
        
        # Final confirmation
        if is_cached:
            LOGGER.info("✓ Model loaded successfully from cache")
        else:
            LOGGER.info("✓ Model downloaded and loaded successfully")
            LOGGER.info("  Model has been cached for future use")
    
    return tokenizer, model


def check_model_cache_status(model_name: str) -> tuple[bool, Path | None]:
    """
    Check if a Hugging Face model is cached locally.
    
    Args:
        model_name: Hugging Face model identifier (e.g., 'mistralai/Mistral-7B-Instruct-v0.2')
        
    Returns:
        Tuple of (is_cached, cache_path)
    """
    # Convert model name to cache directory format
    # e.g., "mistralai/Mistral-7B-Instruct-v0.2" -> "models--mistralai--Mistral-7B-Instruct-v0.2"
    cache_dir_name = f"models--{model_name.replace('/', '--')}"
    
    # Check default Hugging Face cache location
    hf_cache_base = Path.home() / ".cache" / "huggingface" / "hub"
    cache_path = hf_cache_base / cache_dir_name
    
    if cache_path.exists():
        # Check if it has actual model files (not just empty directory)
        # Look for common model files
        has_files = False
        for item in cache_path.rglob("*"):
            if item.is_file() and item.suffix in ['.json', '.bin', '.safetensors', '.model']:
                has_files = True
                break
        
        if has_files:
            return True, cache_path
    
    # Also check if HF_HOME or TRANSFORMERS_CACHE environment variables are set
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        alt_cache_path = Path(hf_home) / "hub" / cache_dir_name
        if alt_cache_path.exists():
            return True, alt_cache_path
    
    transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
    if transformers_cache:
        alt_cache_path = Path(transformers_cache) / cache_dir_name
        if alt_cache_path.exists():
            return True, alt_cache_path
    
    return False, None


def rerank_keywords_mmr(
    keywords: list[str],
    embedding_model: SentenceTransformer | None = None,
    top_k: int | None = None,
    diversity: float = 0.5,
) -> list[str]:
    """
    Rerank keywords using Maximal Marginal Relevance (MMR) for diversity.
    
    MMR balances relevance (keeping important keywords) with diversity
    (ensuring keywords are not too similar to each other).
    
    Args:
        keywords: List of keyword strings to rerank
        embedding_model: SentenceTransformer model for computing embeddings.
                        If None, creates a lightweight model on first call.
        top_k: Maximum number of keywords to return (None for all)
        diversity: Diversity parameter (0.0 = relevance only, 1.0 = diversity only)
                   Default 0.5 balances both
        
    Returns:
        Reranked list of keywords
    """
    if len(keywords) <= 1:
        return keywords
    
    # Use a lightweight embedding model for keyword reranking
    if embedding_model is None:
        # Use a small, fast model for keyword embeddings
        # This is separate from the main embedding model to avoid loading overhead
        try:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        except Exception as e:
            LOGGER.warning("Could not load embedding model for MMR reranking: %s. Skipping reranking.", e)
            return keywords[:top_k] if top_k else keywords
    
    # Compute embeddings for all keywords
    try:
        keyword_embeddings = embedding_model.encode(keywords, convert_to_numpy=True, show_progress_bar=False)
    except Exception as e:
        LOGGER.warning("Error computing embeddings for MMR reranking: %s. Skipping reranking.", e)
        return keywords[:top_k] if top_k else keywords
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(keyword_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    keyword_embeddings = keyword_embeddings / norms
    
    # MMR algorithm: iteratively select keywords that maximize diversity
    # For keyword reranking, we prioritize diversity over relevance
    # since all keywords are already relevant to the topic
    selected_indices: list[int] = []
    remaining_indices = list(range(len(keywords)))
    
    # Start with the first keyword
    if remaining_indices:
        selected_indices.append(remaining_indices.pop(0))
    
    # Determine how many to select
    max_select = min(top_k, len(keywords)) if top_k else len(keywords)
    
    while len(selected_indices) < max_select and remaining_indices:
        best_score = -float('inf')
        best_idx = None
        
        for candidate_idx in remaining_indices:
            candidate_emb = keyword_embeddings[candidate_idx:candidate_idx+1]
            
            # Relevance: average similarity to already selected keywords
            # (keywords similar to selected ones are less novel)
            if selected_indices:
                selected_embs = keyword_embeddings[selected_indices]
                similarities = np.dot(candidate_emb, selected_embs.T)[0]
                avg_relevance = np.mean(similarities)
            else:
                avg_relevance = 0.0
            
            # Diversity: minimum distance from selected keywords
            # (we want keywords that are different from what we already have)
            if selected_indices:
                max_similarity = np.max(similarities)
                diversity_score = 1.0 - max_similarity  # Higher = more diverse
            else:
                diversity_score = 1.0
            
            # MMR score: balance relevance (keeping related keywords) 
            # with diversity (ensuring variety)
            # Lower relevance + higher diversity = better for diverse keyword set
            mmr_score = (1 - diversity) * avg_relevance + diversity * diversity_score
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = candidate_idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        else:
            break
    
    # Return reranked keywords in selection order
    reranked = [keywords[i] for i in selected_indices]
    return reranked


def generate_label_from_keywords(
    keywords: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_new_tokens: int = 40,
    device: str | None = None,
    use_mmr_reranking: bool = False,
    mmr_diversity: float = 0.5,
    mmr_top_k: int | None = None,
) -> str:
    """
    Generate a topic label from keywords using Mistral-7B-Instruct.
    
    Args:
        keywords: List of keyword strings
        tokenizer: Mistral tokenizer
        model: Mistral model
        max_new_tokens: Maximum number of tokens to generate
        device: Device to use ('cuda', 'cpu', or None for auto-detection)
        use_mmr_reranking: If True, rerank keywords using MMR for diversity
        mmr_diversity: Diversity parameter for MMR (0.0-1.0)
        mmr_top_k: Maximum keywords to use after MMR reranking (None for all)
        
    Returns:
        Generated label string
    """
    # Enforce GPU usage when available
    cuda_available = torch.cuda.is_available()
    if device is None:
        device = "cuda" if cuda_available else "cpu"
    elif device == "cpu" and cuda_available:
        LOGGER.warning("CUDA is available but device='cpu' was specified for inference")
    
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
    
    # Tokenize and generate
    try:
        # Apply chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        # Move inputs to device
        # For quantized models with device_map="auto", find the device from model parameters
        try:
            # Try to get device from first model parameter (most reliable method)
            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
        except (StopIteration, AttributeError):
            # Fallback: use device_map or explicit device
            if hasattr(model, "hf_device_map") and model.hf_device_map:
                # Model is distributed across devices, use first device
                first_device = next(iter(model.hf_device_map.values()))
                if isinstance(first_device, (list, tuple)):
                    first_device = first_device[0] if first_device else device
                inputs = {k: v.to(first_device) for k, v in inputs.items()}
            elif device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the generated part (skip the prompt)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        label = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        label = label.strip()
        
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
    model: AutoModelForCausalLM,
    output_path: Path,
    max_new_tokens: int = 40,
    device: str | None = None,
    batch_size: int = 50,
) -> dict[int, str]:
    """
    Generate labels for topics in a streaming fashion and write incrementally to JSON.
    
    This function processes topics one at a time, generates labels, and writes them
    incrementally to avoid keeping all labels in memory at once.
    
    Args:
        pos_topics_iter: Iterator yielding (topic_id, keywords) tuples
        tokenizer: Mistral tokenizer
        model: Mistral model
        output_path: Path to save JSON file (without extension)
        max_new_tokens: Maximum number of tokens to generate per label
        device: Device to use ('cuda', 'cpu', or None for auto-detection)
        batch_size: Number of topics to process before logging progress
        
    Returns:
        Dictionary mapping topic_id to generated label (for integration if needed)
    """
    # Enforce GPU usage when available
    cuda_available = torch.cuda.is_available()
    if device is None:
        device = "cuda" if cuda_available else "cpu"
    elif device == "cpu" and cuda_available:
        LOGGER.warning("CUDA is available but device='cpu' was specified for inference")
    
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
                
                # Telemetry: detect domains for this topic (one line)
                domains = detect_domains(keywords)
                LOGGER.info("topic %d | domains=%s", topic_id, ",".join(domains) if domains else "None")
                
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
    model: AutoModelForCausalLM,
    max_new_tokens: int = 40,
    device: str | None = None,
    batch_size: int = 50,
) -> dict[int, str]:
    """
    Generate labels for all topics from POS keywords in batches.
    
    WARNING: This keeps all labels in memory. For memory efficiency,
    use generate_labels_streaming() instead.
    
    Args:
        pos_topics: Dictionary mapping topic_id to list of keywords
        tokenizer: Mistral tokenizer
        model: Mistral model
        max_new_tokens: Maximum number of tokens to generate per label
        device: Device to use ('cuda', 'cpu', or None for auto-detection)
        batch_size: Number of topics to process before logging progress
        
    Returns:
        Dictionary mapping topic_id to generated label
    """
    # Enforce GPU usage when available
    cuda_available = torch.cuda.is_available()
    if device is None:
        device = "cuda" if cuda_available else "cpu"
    elif device == "cpu" and cuda_available:
        LOGGER.warning("CUDA is available but device='cpu' was specified for inference")
    
    topic_labels: dict[int, str] = {}
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
            LOGGER.info("topic %d | domains=%s", topic_id, ",".join(domains) if domains else "None")
            
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

