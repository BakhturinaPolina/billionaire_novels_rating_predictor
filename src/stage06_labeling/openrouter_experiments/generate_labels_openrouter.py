"""Stage 06 utilities for generating topic labels using OpenRouter API (mistralai/mistral-nemo) from POS representation keywords."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from bertopic import BERTopic
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

# Import improved prompts
try:
    from src.stage06_labeling.prompts.prompts import BASE_LABELING_PROMPT
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False
    LOGGER.warning("Improved prompts module not available. Using default prompts.")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)

# Try to import spaCy for real POS tagging
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    LOGGER.warning("spaCy not available. POS cues will use simplified extraction.")

# Romance-aware prompts for modern romantic fiction
ROMANCE_AWARE_SYSTEM_PROMPT = """You are a topic-labeling assistant for modern romantic fiction.

Goal

- Produce one short, specific label (2–6 words) for a cluster of romance-novel keywords.

- The label must clearly distinguish this topic from similar ones and prioritize romance-relevant facets.

Romance Context

- Prefer facets common in romantic scenes and plots: physical intimacy (body parts, touches, foreplay, oral), emotional beats (jealousy, fear, apology, panic), relationship milestones (engagement, vows, wedding, breakup, reunion), communication moments (whispers, texts, calls), setting cues (bedroom, doorways/elevators, kitchen/meal, bar/night), family dynamics (parents, kids), and sensual aesthetics (scent, fabric, lingerie, makeup).

- If multiple facets appear, pick the single most distinctive one.

REPRESENTATIVE SNIPPETS

- You will be given several short snippets (usually 3–6 sentences) taken from documents in this topic.

- These are more informative than the keyword list for fine distinctions:

  - whether kisses are rough or gentle,

  - whether rage is emotional vs physical,

  - whether a date is a picnic in a park vs dinner in a restaurant,

  - whether a sexual act is a blowjob, fingering, clitoral stimulation, etc.

- When snippets and keywords disagree, trust the snippets.

- Use the snippets as primary evidence for:

  - scene type (kitchen argument, car makeout, office meeting, research discussion),

  - emotional tone (angry, tender, playful, humiliated),

  - explicit sexual acts (e.g., blowjob, fingering, breast play, anal sex, clitoral stimulation).

- Do NOT euphemize clearly explicit sexual acts. Label them factually:

  - "Blowjob in Car", "Clitoral Stimulation on Couch", "Breast Foreplay in Bed".

- Keep wording neutral and non-romanticized; this is for scientific analysis, not for readers' enjoyment.

Style

- Output: one short noun phrase, 2–6 words. No quotes. No trailing punctuation.

- Use Title Case where natural.

- Prefer concrete, scene-level romance phrasing over generic abstractions (e.g., "Whispered Promises," not "Communication").

- Use clear disambiguators only when obvious (e.g., "Elevator Encounter," "Wedding Vows," "Tender Foreplay").

- Do not list or repeat the keywords; synthesize the core idea.

Disambiguation Rules

- Body parts / touch verbs / sex acts → intimacy (e.g., "Tender Foreplay," "Breast Play," "Oral Intimacy").

- Eyes/gaze/face/brows → eye-contact or facial expression (e.g., "Lingering Gazes," "Playful Laughter").

- Wedding/engagement/marriage terms → relationship milestone (e.g., "Wedding Planning," "Wedding Vows," "Engagement Drama").

- Phones/rings/buzzes/screens → device-mediated moments (e.g., "Buzzing Phones," "Late-Night Texts").

- Bed/sheets/pillows/night → setting/scene (e.g., "Bedtime Routine," "Nighttime Cuddling").

- Food/kitchen/dinner/wine → meal/hosting scene (e.g., "Wine Service," "Meal Preparation").

- Emotions like fear/anger/jealousy/apology → emotional beat (e.g., "Fear and Anxiety," "Remorseful Apologies," "Jealous Girlfriend").

- Work/career/office: if any romantic cue appears → "Office Romance"; otherwise a compact neutral like "Work & Career."

- Off-genre clusters (e.g., sports with no romance cues): keep a short, literal label to avoid miscasting ("Ice Hockey Players").

Noise Handling

- If keywords are incoherent or overly generic, choose the most concrete facet present; otherwise use the clearest scene-level summary ("Awkward Silence," "Small Talk").

Guidance Inputs (Optional)

- If a line begins with "Context hints:" follow those nudges.

- If a line begins with "POS cues:" use nouns/verbs to anchor the facet (e.g., body-part nouns + tactile verbs → intimacy).

Output

- Return only the label."""

ROMANCE_AWARE_USER_PROMPT = """Topic keywords (most important first):

{kw}{hints}

{pos}{snippets}

Remember:

- Base your label primarily on the shared pattern in the snippets.

- Use the keywords to check you are not missing important body parts, actions, or settings.

- Ignore single outlier words (e.g. a random city) unless they appear in multiple snippets and keywords.

- Use precise, neutral, scene-level phrasing, 2–6 words only.

Label:"""

# OpenRouter API configuration
# Get API key from environment variable, fallback to empty string if not set
DEFAULT_OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
DEFAULT_OPENROUTER_MODEL = "mistralai/mistral-nemo"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Module-level cache for MMR embedding model (loaded once, reused for all topics)
_MMR_EMBEDDING_MODEL: SentenceTransformer | None = None

# Module-level cache for spaCy NLP model (loaded once, reused for all topics)
_SPACY_NLP = None


def _load_spacy_model():
    """Load spaCy model for POS tagging (cached)."""
    global _SPACY_NLP
    if _SPACY_NLP is None and SPACY_AVAILABLE:
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            LOGGER.info("Loaded spaCy model for POS tagging")
        except OSError:
            LOGGER.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            return None
    return _SPACY_NLP


def extract_pos_cues(keywords: list[str]) -> str:
    """
    Extract real POS cues from keywords using spaCy POS tagging for romance-aware labeling.
    
    Uses actual POS tagging from spaCy to categorize keywords into nouns, verbs, and adjectives.
    Falls back to simplified extraction if spaCy is not available.
    
    Args:
        keywords: List of keyword strings
        
    Returns:
        Formatted POS cues string (empty if no cues detected)
    """
    if not keywords:
        return ""
    
    nlp = _load_spacy_model()
    
    # Use real POS tagging if spaCy is available
    if nlp is not None:
        nouns = []
        verbs = []
        adjs = []
        
        # Process each keyword individually for better POS accuracy
        # (spaCy works better on individual words/phrases than joined text)
        for kw in keywords:
            if not kw or not kw.strip():
                continue
            
            # Process keyword with spaCy
            doc = nlp(kw)
            
            # Get POS tag from the first (and usually only) token
            # For multi-word keywords, use the head token or first significant token
            pos_tag = None
            for token in doc:
                # Skip punctuation and stop words for POS determination
                if not token.is_punct and not token.is_stop:
                    pos_tag = token.pos_
                    break
            
            # If no significant token found, use first token's POS
            if pos_tag is None and len(doc) > 0:
                pos_tag = doc[0].pos_
            
            # spaCy POS tags: NOUN, VERB, ADJ, etc.
            if pos_tag == "NOUN" or pos_tag == "PROPN":  # Noun or proper noun
                nouns.append(kw)
            elif pos_tag == "VERB":
                verbs.append(kw)
            elif pos_tag == "ADJ":  # Adjective
                adjs.append(kw)
        
        # Build POS cues string
        parts = []
        if nouns:
            parts.append(f"Nouns→{', '.join(nouns[:5])}")  # Limit to 5 for brevity
        if verbs:
            parts.append(f"Verbs→{', '.join(verbs[:5])}")
        if adjs:
            parts.append(f"Adjectives→{', '.join(adjs[:5])}")
        
        if parts:
            return "POS cues: " + "; ".join(parts) + "."
        return ""
    
    # Fallback: simplified extraction using domain knowledge
    # Common body part nouns (from domain lexicon)
    body_parts = {
        "lip", "lips", "mouth", "tongue", "teeth", "cheek", "cheeks", "nose", "chin",
        "brows", "eyebrow", "eyebrows", "eye", "eyes", "neck", "nape", "shoulder",
        "shoulders", "arm", "arms", "hand", "hands", "finger", "fingers", "fist",
        "fists", "breast", "breasts", "nipples", "waist", "belly", "stomach", "chest",
        "spine", "back", "hip", "hips", "thigh", "thighs", "legs", "leg", "knee",
        "knees", "feet", "foot", "heels", "clit", "clitoris", "pussy", "genitals",
    }
    
    # Common touch/intimacy verbs
    touch_verbs = {
        "kiss", "kissed", "kissing", "touch", "touched", "touching", "caress", "caressed",
        "cup", "cupped", "grab", "grabbed", "grasp", "grasped", "hold", "held", "hug",
        "hugged", "embrace", "embraced", "stroke", "stroked", "rub", "rubbed", "squeeze",
        "squeezed", "pinch", "pinched", "lick", "licked", "suck", "sucked", "bite",
        "bit", "nibble", "nibbled",
    }
    
    # Common emotional/adjective words
    adjectives = {
        "tender", "gentle", "soft", "hard", "rough", "smooth", "warm", "cold", "hot",
        "sweet", "bitter", "sour", "intense", "passionate", "romantic", "loving",
        "affectionate", "desperate", "urgent", "slow", "fast", "deep", "shallow",
    }
    
    nouns = []
    verbs = []
    adjs = []
    
    keywords_lower = [kw.lower() for kw in keywords]
    
    for kw in keywords_lower:
        if kw in body_parts:
            nouns.append(kw)
        elif kw in touch_verbs:
            verbs.append(kw)
        elif kw in adjectives:
            adjs.append(kw)
    
    # Build POS cues string
    parts = []
    if nouns:
        parts.append(f"Nouns→{', '.join(nouns[:5])}")  # Limit to 5 for brevity
    if verbs:
        parts.append(f"Verbs→{', '.join(verbs[:5])}")
    if adjs:
        parts.append(f"Adjectives→{', '.join(adjs[:5])}")
    
    if parts:
        return "POS cues: " + "; ".join(parts) + "."
    return ""


def format_snippets(
    docs: list[str],
    max_snippets: int = 6,
    max_chars: int = 200,
) -> str:
    """
    Convert a list of documents into a bullet-style snippets block for LLM prompts.
    
    Formats representative document snippets as numbered quotes, with truncation
    for long sentences. Designed for sentence-level documents (each doc is a sentence).
    
    Args:
        docs: List of document strings (sentences in this corpus)
        max_snippets: Maximum number of snippets to include (default: 6)
        max_chars: Maximum characters per snippet before truncation (default: 200)
        
    Returns:
        Formatted string with numbered snippets, or empty string if no docs provided
    """
    if not docs:
        return ""
    
    snippets = []
    for i, doc in enumerate(docs[:max_snippets], start=1):
        # Collapse whitespace
        s = " ".join(doc.split())
        
        # Truncate at word boundary if too long
        if len(s) > max_chars:
            s = s[:max_chars].rsplit(" ", 1)[0] + "..."
        
        snippets.append(f'{i}) "{s}"')
    
    if not snippets:
        return ""
    
    return "Representative snippets (short excerpts for this topic):\n" + "\n".join(snippets)


def extract_representative_docs_per_topic(
    topic_model: BERTopic,
    max_docs_per_topic: int = 10,
) -> dict[int, list[str]]:
    """
    Extract representative documents for each topic from BERTopic model.
    
    Tries get_representative_docs() method first, falls back to representative_docs_
    attribute. Handles both dict and method return formats.
    
    Args:
        topic_model: BERTopic model instance
        max_docs_per_topic: Maximum number of representative docs to extract per topic
        
    Returns:
        Dictionary mapping topic_id to list of representative document strings
    """
    topic_to_docs: dict[int, list[str]] = {}
    
    # Get all topic IDs (excluding outlier topic -1)
    if hasattr(topic_model, "topics_"):
        topic_ids = set(topic_model.topics_)
        topic_ids.discard(-1)  # Skip outlier topic
    elif hasattr(topic_model, "topic_representations_"):
        topic_ids = set(topic_model.topic_representations_.keys())
        topic_ids.discard(-1)
    else:
        LOGGER.warning("Cannot determine topic IDs from BERTopic model")
        return topic_to_docs
    
    LOGGER.info("Extracting representative documents for %d topics", len(topic_ids))
    
    # Try get_representative_docs() method first (newer BERTopic versions)
    if hasattr(topic_model, "get_representative_docs"):
        try:
            for topic_id in topic_ids:
                try:
                    rep_docs = topic_model.get_representative_docs(
                        topic=topic_id,
                        nr_docs=max_docs_per_topic,
                    )
                    # Handle both list and dict return types
                    if isinstance(rep_docs, dict):
                        # If dict, extract the list value
                        rep_docs = list(rep_docs.values())[0] if rep_docs else []
                    elif isinstance(rep_docs, list):
                        # Already a list
                        pass
                    else:
                        LOGGER.warning(
                            "Unexpected return type from get_representative_docs for topic %d: %s",
                            topic_id,
                            type(rep_docs),
                        )
                        rep_docs = []
                    
                    # Ensure all items are strings
                    rep_docs = [str(doc) for doc in rep_docs if doc]
                    topic_to_docs[topic_id] = rep_docs
                    
                except Exception as e:
                    LOGGER.warning(
                        "Error getting representative docs for topic %d via method: %s",
                        topic_id,
                        e,
                    )
                    topic_to_docs[topic_id] = []
            
            LOGGER.info(
                "Extracted representative docs via get_representative_docs() for %d topics",
                len([tid for tid, docs in topic_to_docs.items() if docs]),
            )
            return topic_to_docs
            
        except Exception as e:
            LOGGER.warning(
                "get_representative_docs() method failed, falling back to attribute: %s",
                e,
            )
    
    # Fallback to representative_docs_ attribute
    if hasattr(topic_model, "representative_docs_"):
        try:
            rep_docs_attr = topic_model.representative_docs_
            
            if isinstance(rep_docs_attr, dict):
                # Dict format: {topic_id: [doc1, doc2, ...]}
                for topic_id in topic_ids:
                    if topic_id in rep_docs_attr:
                        docs = rep_docs_attr[topic_id]
                        # Ensure it's a list and convert to strings
                        if isinstance(docs, list):
                            docs = [str(doc) for doc in docs[:max_docs_per_topic] if doc]
                        else:
                            docs = [str(docs)] if docs else []
                        topic_to_docs[topic_id] = docs
                    else:
                        topic_to_docs[topic_id] = []
            else:
                LOGGER.warning(
                    "representative_docs_ is not a dict, got type: %s",
                    type(rep_docs_attr),
                )
            
            LOGGER.info(
                "Extracted representative docs via representative_docs_ for %d topics",
                len([tid for tid, docs in topic_to_docs.items() if docs]),
            )
            return topic_to_docs
            
        except Exception as e:
            LOGGER.warning("Error accessing representative_docs_ attribute: %s", e)
    
    # If both methods failed, log warning and return empty dict
    LOGGER.warning(
        "Could not extract representative docs from BERTopic model. "
        "Labels will be generated from keywords only."
    )
    return topic_to_docs


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
    use_improved_prompts: bool = False,
    representative_docs: list[str] | None = None,
    max_snippets: int = 6,
    max_chars_per_snippet: int = 200,
) -> dict[str, Any]:
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
        use_improved_prompts: If True, use BASE_LABELING_PROMPT and parse JSON response
        representative_docs: Optional list of representative document strings (snippets)
        max_snippets: Maximum number of snippets to include in prompt (default: 6)
        max_chars_per_snippet: Maximum characters per snippet before truncation (default: 200)
        
    Returns:
        Dictionary with 'label' and optionally 'categories', 'is_noise', 'rationale'
        If use_improved_prompts is False, returns dict with just 'label' (backward compatible)
    """
    # Apply MMR reranking for diversity if requested
    if use_mmr_reranking and len(keywords) > 1:
        keywords = rerank_keywords_mmr(
            keywords,
            embedding_model=None,  # Will create lightweight model on first call
            top_k=mmr_top_k,
            diversity=mmr_diversity,
        )
    
    # Choose prompt type
    if use_improved_prompts and PROMPTS_AVAILABLE:
        # Use improved prompt with JSON output
        kw_str = ", ".join(keywords)
        user_prompt = f"Topic keywords: {kw_str}\n\nLabel:"
        messages = [
            {"role": "system", "content": BASE_LABELING_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        max_new_tokens = max(max_new_tokens, 200)  # Need more tokens for JSON
    else:
        # Use original romance-aware prompt
        kw_str = ", ".join(keywords)
        domains = detect_domains(keywords)
        hints = make_context_hints(domains)  # Already includes "Context hints:" prefix
        hints_str = hints if hints else ""
        
        # Extract POS cues for romance-aware labeling
        pos_cues = extract_pos_cues(keywords)
        pos_str = pos_cues if pos_cues else ""
        
        # Format snippets if representative_docs provided
        snippets_block = ""
        if representative_docs:
            snippets_block = "\n\n" + format_snippets(
                representative_docs,
                max_snippets=max_snippets,
                max_chars=max_chars_per_snippet,
            )
        
        user_prompt = ROMANCE_AWARE_USER_PROMPT.format(
            kw=kw_str,
            hints=hints_str,
            pos=pos_str,
            snippets=snippets_block
        )
        messages = [
            {"role": "system", "content": ROMANCE_AWARE_SYSTEM_PROMPT},
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
        
        content = response.choices[0].message.content.strip()
        LOGGER.debug("Raw API response: %s", content[:200] if len(content) > 200 else content)
        
        # Parse response based on prompt type
        if use_improved_prompts and PROMPTS_AVAILABLE:
            # Try to parse JSON response
            try:
                # Extract JSON from response (might have markdown code blocks)
                json_content = content
                if "```json" in json_content:
                    json_content = json_content.split("```json")[1].split("```")[0].strip()
                elif "```" in json_content:
                    json_content = json_content.split("```")[1].split("```")[0].strip()
                
                result = json.loads(json_content)
                
                # Extract fields
                label = result.get("label", "")
                primary_categories = result.get("primary_categories", [])
                secondary_categories = result.get("secondary_categories", [])
                is_noise = result.get("is_noise", False)
                rationale = result.get("rationale", "")
                
                # Clean label
                label = label.strip().strip('"').strip("'")
                
                # Build result dict
                result_dict = {
                    "label": label,
                    "primary_categories": primary_categories,
                    "secondary_categories": secondary_categories,
                    "is_noise": is_noise,
                }
                if rationale:
                    result_dict["rationale"] = rationale
                
                LOGGER.info("Generated label (improved prompt): %s | Categories: %s", 
                           label, primary_categories)
                return result_dict
                
            except (json.JSONDecodeError, KeyError) as e:
                LOGGER.warning("Failed to parse JSON response, falling back to text extraction: %s", e)
                # Fall through to text extraction
                label = content
        else:
            label = content
        
        # Post-process label to clean it up (for non-JSON responses)
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
        return {"label": label}  # Return dict for consistency
    except Exception as e:
        LOGGER.warning("Error generating label for keywords %s: %s", keywords[:3], e)
        LOGGER.exception("Full error traceback:")
        # Fallback: create a simple label from first few keywords
        fallback_label = f"{keywords[0]}" if keywords else "Topic"
        LOGGER.info("Using fallback label: %s", fallback_label)
        return {"label": fallback_label}


def generate_labels_streaming(
    pos_topics_iter: Iterator[tuple[int, list[str]]],
    client: OpenAI,
    model_name: str,
    output_path: Path,
    max_new_tokens: int = 40,
    batch_size: int = 50,
    temperature: float = 0.3,
    limit: int | None = None,
    use_improved_prompts: bool = False,
    topic_model: BERTopic | None = None,
    topic_to_snippets: dict[int, list[str]] | None = None,
    max_snippets: int = 6,
    max_chars_per_snippet: int = 200,
) -> dict[int, dict[str, Any]]:
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
        use_improved_prompts: If True, use BASE_LABELING_PROMPT and parse JSON response
        topic_model: Optional BERTopic model for extracting representative docs (if topic_to_snippets not provided)
        topic_to_snippets: Optional pre-extracted dict mapping topic_id to representative docs
        max_snippets: Maximum number of snippets to include per topic (default: 6)
        max_chars_per_snippet: Maximum characters per snippet before truncation (default: 200)
        
    Returns:
        Dictionary mapping topic_id to dict with 'label' and 'keywords' keys
    """
    json_path = output_path.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract representative docs if topic_model provided and topic_to_snippets not provided
    if topic_to_snippets is None and topic_model is not None:
        LOGGER.info("Extracting representative documents for snippets...")
        topic_to_snippets = extract_representative_docs_per_topic(topic_model)
        LOGGER.info(
            "Extracted snippets for %d topics",
            len([tid for tid, docs in topic_to_snippets.items() if docs]),
        )
    elif topic_to_snippets is None:
        topic_to_snippets = {}
        LOGGER.info("No topic_model or topic_to_snippets provided, labels will use keywords only")
    
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
                
                # Get representative docs for this topic
                rep_docs = topic_to_snippets.get(topic_id, []) if topic_to_snippets else []
                snippet_count = len(rep_docs)
                
                LOGGER.info(
                    "topic %d | domains=%s | keywords=%s | snippets=%d",
                    topic_id,
                    ",".join(domains) if domains else "None",
                    ", ".join(keywords[:5]) + ("..." if len(keywords) > 5 else ""),
                    snippet_count,
                )
                
                # Generate label with timing
                label_start = time.perf_counter()
                result = generate_label_from_keywords_openrouter(
                    keywords=keywords,
                    client=client,
                    model_name=model_name,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    use_improved_prompts=use_improved_prompts,
                    representative_docs=rep_docs,
                    max_snippets=max_snippets,
                    max_chars_per_snippet=max_chars_per_snippet,
                )
                label_elapsed = time.perf_counter() - label_start
                
                # Extract label (result is now a dict)
                label = result.get("label", "")
                
                # Store label, keywords, and any additional fields from improved prompts
                topic_data[topic_id] = {
                    "label": label,
                    "keywords": keywords,
                    **{k: v for k, v in result.items() if k != "label"}  # Add categories, is_noise, etc.
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
    use_improved_prompts: bool = False,
    topic_model: BERTopic | None = None,
    topic_to_snippets: dict[int, list[str]] | None = None,
    max_snippets: int = 6,
    max_chars_per_snippet: int = 200,
) -> dict[int, dict[str, Any]]:
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
        use_improved_prompts: If True, use BASE_LABELING_PROMPT and parse JSON response
        topic_model: Optional BERTopic model for extracting representative docs (if topic_to_snippets not provided)
        topic_to_snippets: Optional pre-extracted dict mapping topic_id to representative docs
        max_snippets: Maximum number of snippets to include per topic (default: 6)
        max_chars_per_snippet: Maximum characters per snippet before truncation (default: 200)
        
    Returns:
        Dictionary mapping topic_id to dict with 'label' and 'keywords' keys
    """
    # Extract representative docs if topic_model provided and topic_to_snippets not provided
    if topic_to_snippets is None and topic_model is not None:
        LOGGER.info("Extracting representative documents for snippets...")
        topic_to_snippets = extract_representative_docs_per_topic(topic_model)
        LOGGER.info(
            "Extracted snippets for %d topics",
            len([tid for tid, docs in topic_to_snippets.items() if docs]),
        )
    elif topic_to_snippets is None:
        topic_to_snippets = {}
        LOGGER.info("No topic_model or topic_to_snippets provided, labels will use keywords only")
    
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
            
            # Get representative docs for this topic
            rep_docs = topic_to_snippets.get(topic_id, []) if topic_to_snippets else []
            snippet_count = len(rep_docs)
            
            LOGGER.info(
                "topic %d | domains=%s | keywords=%s | snippets=%d",
                topic_id,
                ",".join(domains) if domains else "None",
                ", ".join(keywords[:5]) + ("..." if len(keywords) > 5 else ""),
                snippet_count,
            )
            
            # Generate label with timing
            label_start = time.perf_counter()
            result = generate_label_from_keywords_openrouter(
                keywords=keywords,
                client=client,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                use_improved_prompts=use_improved_prompts,
                representative_docs=rep_docs,
                max_snippets=max_snippets,
                max_chars_per_snippet=max_chars_per_snippet,
            )
            label_elapsed = time.perf_counter() - label_start
            
            # Extract label (result is now a dict)
            label = result.get("label", "")
            
            # Store label, keywords, and any additional fields from improved prompts
            batch_labels[topic_id] = {
                "label": label,
                "keywords": keywords,
                **{k: v for k, v in result.items() if k != "label"}  # Add categories, is_noise, etc.
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

