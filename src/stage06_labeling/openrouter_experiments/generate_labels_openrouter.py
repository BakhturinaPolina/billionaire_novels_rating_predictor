"""Stage 06 utilities for generating topic labels using OpenRouter API (mistralai/mistral-nemo) from POS representation keywords."""

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
ROMANCE_AWARE_SYSTEM_PROMPT = """You are a topic-labeling assistant for modern romantic and erotic fiction.

Your job is to assign precise, descriptive labels to topic clusters so that:
- A human reader can roughly guess the main words and scenes behind the topic.
- Different topics receive clearly distinguishable labels, even if they share some vocabulary.

GENERAL RULES
- Output exactly ONE short noun phrase of 2–6 words.
- No quotes, no numbering, no extra text.
- Use clear, neutral, descriptive language – not poetic titles.
- Prefer concrete scene-level descriptions over abstractions.
  - Good: "Rough Angry Kisses in Hallway"
  - Bad: "Intense Love", "Erotic Intimacy", "Romantic Moment"
- You may use explicit sexual terms found in the keywords (e.g., oral sex, blowjob, fingering, pussy, cock) but keep the tone factual, not arousing.

WHAT TO ENCODE IN THE LABEL (IN ORDER OF PRIORITY)
Focus on the strongest, most frequent signals in the keyword list:

1) MAIN ACTION OR SEX ACT
   - If keywords clearly describe a sex act, name it: e.g.,
     "Oral Sex on Him", "Fingering Her", "Breast Play", "Anal Sex".
   - If there is a conflict between generic intimacy ("touch", "kiss") and a specific act
     ("blowjob", "clit", "pussy", "erection"), prioritize the specific act.

2) ROLE / TARGET / BODY PART
   - Include who or what is involved if it clarifies the topic:
     "Rough Kisses Against Wall", "Gentle Kisses on Neck",
     "Clitoral Stimulation", "Handjob Under Table".

3) SETTING OR SITUATION (IF CLEAR)
   - Add a concise situational cue when obvious:
     "Kitchen Argument in Morning",
     "Elevator Makeout", "Picnic Date in City Park".

4) EMOTIONAL TONE OR PURPOSE
   - Only if clearly indicated and not speculative:
     "Comforting Hugs After Fight",
     "Jealous Rage and Yelling",
     "Playful Flirting at Bar".

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

DISAMBIGUATION REQUIREMENTS
- If two topics are both about similar themes (e.g., rage, kisses, dates)
  but differ in physical vs emotional focus, setting, or tone:
  - Encode that distinction explicitly:
    - "Physical Violence and Rage" vs "Silent Emotional Resentment"
    - "Gentle Comforting Kisses" vs "Rough Angry Kisses"
    - "First Date Picnic in Park" vs "Crowded City Restaurant Date".

- Do NOT reuse the same vague label (e.g., "Erotic Intimacy", "Erotic Encounter")
  for multiple distinct topics. Make each label specific to its keywords.

OUTLIERS AND NOISE
- Ignore single, isolated outliers (e.g., one city or place name)
  if most keywords point to a different core idea.
  - Example: If almost all words are about an unexpected meeting,
    and one keyword is a city name, do NOT put the city in the label
    unless multiple location words dominate the topic.

- If keywords are incoherent, choose the most concrete, frequent facet you see
  and describe it literally: "Random Small Talk", "Household Objects and Doors".

WORK / RESEARCH / META TOPICS
- Distinguish between:
  - "Work & Career Tasks" (job, office, boss, meeting, deadline)
  - "Researching Sexual Terminology" (research, terminology, user, giver, kink)
  - "Writing or Editing Scenes" (draft, editor, chapter, rewrite)

Sexual research in the text should not be mislabeled as "Preparing for Work".

SEXUAL CONTENT PRECISENESS
- Do NOT euphemize explicit sexual content:
  - If a topic is clearly about blowjob / oral sex, say so, e.g. "Public Blowjob in Alley",
    not "Erotic Intimacy".
  - If a topic is about physical foreplay to breasts, label it "Breast Foreplay" or similar.
  - If a topic is about clitoral stimulation and legs/hips, label it "Clitoral Stimulation Between Thighs" or similar.
- Always keep the phrasing clinical and non-romanticized.

OUTPUT FORMAT
- Return ONLY the label, as a short noun phrase of 2–6 words.
- No explanations, no extra sentences, no lists."""

ROMANCE_AWARE_USER_PROMPT = """Topic keywords (most important first):

{kw}{hints}

{pos}
{snippets}

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
    # According to official BERTopic docs: get_representative_docs(topic=None)
    # Only accepts 'topic' parameter, returns all representative docs for that topic
    if hasattr(topic_model, "get_representative_docs"):
        try:
            for topic_id in topic_ids:
                try:
                    # Call with only topic parameter (official API)
                    rep_docs = topic_model.get_representative_docs(topic=topic_id)
                    
                    # Handle both list and dict return types
                    if isinstance(rep_docs, dict):
                        # If dict, extract the list value (usually {topic_id: [docs]})
                        # Handle case where key might be different or multiple keys exist
                        if topic_id in rep_docs:
                            rep_docs = rep_docs[topic_id]
                        elif len(rep_docs) == 1:
                            # Single key, extract its value
                            rep_docs = list(rep_docs.values())[0]
                        else:
                            # Multiple topics in dict, try to find matching one
                            rep_docs = rep_docs.get(topic_id, [])
                    elif isinstance(rep_docs, list):
                        # Already a list - this is the expected format
                        pass
                    else:
                        LOGGER.warning(
                            "Unexpected return type from get_representative_docs for topic %d: %s",
                            topic_id,
                            type(rep_docs),
                        )
                        rep_docs = []
                    
                    # Ensure rep_docs is a list
                    if not isinstance(rep_docs, list):
                        rep_docs = [rep_docs] if rep_docs else []
                    
                    # Limit to max_docs_per_topic (BERTopic doesn't have limit parameter)
                    if len(rep_docs) > max_docs_per_topic:
                        rep_docs = rep_docs[:max_docs_per_topic]
                    
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


# Generic bad labels to avoid
GENERIC_BAD_LABELS = {
    "Erotic Intimacy",
    "Erotic Encounter",
    "Romantic Moment",
    "Intense Love",
}


def normalize_label(raw: str) -> str:
    """Normalize and clean a generated label.
    
    Args:
        raw: Raw label string from API response
        
    Returns:
        Cleaned, normalized label
    """
    # Basic cleanup
    label = raw.strip().strip('"').strip("'")
    
    # Remove special tokens and tags (e.g., <s>, </s>, [boss], etc.)
    label = re.sub(r"<s>|</s>|\[.*?\]", "", label)  # Remove <s>, </s>, and [tag] patterns
    label = re.sub(r"<[^>]+>", "", label)  # Remove any remaining HTML/XML tags
    
    # Clean up whitespace
    label = re.sub(r"\s+", " ", label)
    label = label.strip()
    
    # Remove trailing punctuation
    label = label.rstrip(".,;:!-")
    
    # If the model accidentally produced a sentence, keep first 6 words
    words = label.split()
    if len(words) > 6:
        words = words[:6]
    label = " ".join(words)

    # Title-case words, but keep short words lowercased where natural
    def smart_tc(w: str) -> str:
        return w if w.lower() in {"and", "or", "in", "on", "of", "at", "to"} else w.capitalize()
    label = " ".join(smart_tc(w) for w in label.split())

    # Avoid super-generic labels you know are useless
    if label in GENERIC_BAD_LABELS:
        # fallback to something slightly more specific using first 2 keywords
        label = label + " Topic"  # or just let it pass – or trigger a re-prompt in the future
    return label


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
            top_p=0.7,  # Slightly limits randomness
            presence_penalty=0,  # Leave penalties neutral
            frequency_penalty=0,
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
                
                # Normalize label
                label = normalize_label(label)
                
                # If normalization resulted in empty label, use fallback
                if not label or label.strip() == "":
                    LOGGER.warning("Normalized label is empty (improved prompt), using fallback for keywords: %s", keywords[:3])
                    fallback_label = f"{keywords[0]}" if keywords else "Topic"
                    LOGGER.info("Using fallback label: %s", fallback_label)
                    label = fallback_label
                
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
        
        # Normalize label using the improved normalizer
        label = normalize_label(label)
        
        # If normalization resulted in empty label, use fallback
        if not label or label.strip() == "":
            LOGGER.warning("Normalized label is empty, using fallback for keywords: %s", keywords[:3])
            fallback_label = f"{keywords[0]}" if keywords else "Topic"
            LOGGER.info("Using fallback label: %s", fallback_label)
            label = fallback_label
        
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
                
                # Delay to respect rate limits (16 requests/min = ~3.75s between requests)
                # Using 4 seconds to stay safely under the limit
                time.sleep(4.0)
            
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
            
            # Delay to respect rate limits (16 requests/min = ~3.75s between requests)
            # Using 4 seconds to stay safely under the limit
            time.sleep(4.0)
        
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

