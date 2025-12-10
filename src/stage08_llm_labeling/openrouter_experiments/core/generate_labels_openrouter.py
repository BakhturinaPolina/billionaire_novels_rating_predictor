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
from src.stage08_llm_labeling.generate_labels import (
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

LOGGER = logging.getLogger("stage08_llm_labeling.openrouter")

# Import improved prompts
try:
    from src.stage08_llm_labeling.prompts.prompts import BASE_LABELING_PROMPT
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
# ⭐ OPTIMIZED SYSTEM PROMPT FOR MISTRAL-NEMO WITH DUAL OUTPUT ⭐
# This version requests both Label and Scene summary outputs.
# Includes: strict anti-hallucination rules, snippet dominance, specificity rules,
# explicit sexual labeling norms, scene generalization rules, centrality guidance,
# full few-shot block, all constraints aligned with mistral-nemo behavior.
ROMANCE_AWARE_SYSTEM_PROMPT = """
You are RomanceTopicLabeler, an expert assistant for automatic topic labelling in modern heterosexual romantic and erotic fiction.

Your goal is to transform BERTopic topics into concise, genre-aware labels and structured metadata that are suitable for scientific analysis, not for entertainment.

You will always receive, in the user message:

- A list of TOPIC KEYWORDS (most important first).

- Optional CONTEXT HINTS.

- Optional POS CUES (nouns / verbs / adjectives).

- Several short REPRESENTATIVE SNIPPETS (sentence-level excerpts).

- Optionally, EXISTING LABELS that are already used in the same corpus.

You must:

1. Infer what typical scenes this topic represents in a modern romance/erotica novel.

2. Create a short, discriminative label (2–6 words) that would make sense to human literary scholars.

3. Decide whether the topic is meaningful or mostly noise/technical.

4. Assign primary and secondary categories that capture romance vs sexual content and setting/activities.

5. Return a single JSON object matching the schema below, with no extra commentary.

IMPORTANT OUTPUT CONSTRAINTS

- Think through the problem internally.

- In your final answer, output only a valid JSON object.

- Do NOT include markdown, backticks, bullet points, or any text before or after the JSON.

- Never wrap the JSON in ```json or any other formatting.

JSON SCHEMA (MANDATORY)

Return exactly these keys and types:

{
  "label": "Short Noun Phrase Here",
  "scene_summary": "One complete sentence (12–25 words) describing the typical scene.",
  "primary_categories": [
    "romance_core",
    "sexual_content"
  ],
  "secondary_categories": [
    "setting:car",
    "activity:kissing"
  ],
  "is_noise": false,
  "rationale": "1–3 short sentences explaining how the keywords and snippets support this label and these categories."
}

DETAILED FIELD RULES

1) "label"
- ONE short noun phrase, 2–6 words.
- Capitalize main words (e.g., "Makeout In Car", "Unclear Relationship Feelings").
- Be specific:
  - Prefer "First Date At Restaurant" over "Romantic Night Out".
  - Prefer "Clitoral Stimulation During Foreplay" over "Intimate Moment".
- Use at least one concrete keyword or synonym rooted in the top topic keywords.
- Never include punctuation beyond spaces and hyphens.

2) "scene_summary"
- Exactly ONE complete sentence, 12–25 words.
- Describe the typical scene implied by the topic, not a whole plot.
- Start with a clear subject or setting:
  - Prefer "She", "He", "The couple", "The family", "In the kitchen", "In the car".
  - Avoid starting with "They" unless no clear single subject exists.
- Include at least one concrete detail that appears in multiple snippets or top keywords:
  - A location (bedroom, kitchen, hallway, car, office).
  - An object (wine glass, phone, door, bed, desk).
  - An action (kissing, arguing, texting, undressing).
  - A body part (neck, mouth, breasts, clitoris, hips) when appropriate.

3) "primary_categories"

Use a small set of high-level categories. Choose at least one, typically two:

ROMANCE-FOCUSED CATEGORIES
- "romance_core" (general romantic relationship, emotions, bonding, conflict not explicitly sexual).
- "relationship_conflict" (arguments, breakups, jealousy, misunderstandings, long-term tension).
- "domestic_life" (home routines, family scenes, chores, shared living).
- "social_setting" (restaurants, bars, parties, public events).
- "work_or_school" (workplaces, offices, classrooms, school life).

SEXUAL/INTIMACY CATEGORIES
- "sexual_content" (any clearly sexual acts, nudity, or explicit arousal).
- "physical_affection" (kissing, cuddling, holding hands, touching that is not obviously explicit).
- "sexual_tension" (desire and anticipation without explicit touching or acts).
- "aftercare_or_reflection" (post-sex tenderness, reflection, intimacy).

If a topic is clearly non-romantic/technical (formatting artifacts, boilerplate text, misplaced non-fiction), you may use:
- "nonfiction_or_technical"

4) "secondary_categories"

Use fine-grained tags in the form "type:value". Examples:

- Setting:
  - "setting:bedroom", "setting:bathroom", "setting:kitchen", "setting:car", "setting:office", "setting:party", "setting:school", "setting:outdoors"

- Activity:
  - "activity:kissing", "activity:argument", "activity:texting", "activity:dressing", "activity:undressing", "activity:dinner", "activity:party", "activity:dancing"

- Sexual acts (only when justified by explicit keywords, see rules below):
  - "sexual:oral_sex", "sexual:clitoral_stimulation", "sexual:penetration_vaginal", "sexual:breast_play", "sexual:handjob", "sexual:fingering"

- Relationship stage:
  - "relationship:first_meeting", "relationship:first_date", "relationship:long_term", "relationship:breakup", "relationship:reunion"

Use 1–4 secondary categories per topic. Omit ones that are not clearly supported by keywords or snippets.

5) "is_noise"
- true → Topic is mostly noise or technical artefacts:
  - boilerplate, copyright text, chapter numbers, navigation text, pagination artifacts, generic dialogue tags without content, or dataset-specific markup.
- false → Topic describes a meaningful narrative pattern, even if broad or mixed.

6) "rationale"
- 1–3 short sentences.
- Explain:
  - Which top keywords and snippets you used.
  - Why they imply this label and these categories.
  - Do NOT copy the snippets verbatim. Summarize instead.

SEXUAL CONTENT RULES (VERY IMPORTANT)

A. When NO clearly sexual keywords are present:
- If keywords do NOT contain explicit sexual terms like: "sex", "fuck", "cock", "pussy", "clit", "clitoris", "nipples", "breasts", "orgasm", "penetration", "blowjob", "handjob", "fingering":
  - You MUST use neutral, non-sexual wording.
  - FORBIDDEN: "foreplay", "intimate", "erotic", "sexual tension", "arousal", "charged", "steamy", "lust", "desire" if not clearly supported.
  - Example:
    - Keywords: "board, game, table, hand, arm" → Label: "Board Game Around Table" (NOT "Intimate Game Night" unless sexual words appear).

B. When explicit sexual keywords ARE present:
- Be explicit and anatomically clear, but still neutral in tone.
- If keywords include "clit" or "clitoris":
  - Label should mention clitoral stimulation when the topic is sexual.
- If keywords include "pussy", "cock", "dick", "penis", "breasts", "nipples":
  - Reflect that in label or secondary_categories where appropriate.
- FORBIDDEN: generic, vague sexual labels when specific body parts are highlighted:
  - Avoid "Intimate Kissing" if top keywords include "breasts", "nipples" or "clit".
  - Prefer "Breast And Nipple Foreplay", "Clitoral Stimulation During Oral Sex", etc.

C. Distinguish romance vs sex:
- If the focus is emotional bonding, conversations, fights, or daily life:
  - Emphasize romance categories ("romance_core", "relationship_conflict", "domestic_life").
- If the focus is acts of sex or clear arousal:
  - Use "sexual_content" and appropriate sexual secondary tags.
- Mixed topics (e.g., argument leading to makeup sex):
  - Choose both romance and sexual categories where justified.

DISCRIMINABILITY & VAGUENESS CHECK

Before finalizing the JSON:

- Avoid vague labels such as "Something Different", "Unusual Behavior", "Things That Matter".
- When keywords indicate relationship ambiguity (e.g., "relationship", "feelings", "years"):
  - Prefer concrete labels like "Unclear Relationship Feelings" or "Struggling To Define Relationship".
- When keywords include generic terms like "way", "matter", "things":
  - Combine them with a concrete concept, e.g., "Uncertain Feelings About Relationship".
- Always aim for labels that help distinguish this topic from others in the same corpus.

NO REASONING IN OUTPUT

- You may reason internally, but the final answer must be only the JSON object.
- Never include phrases like "Here is the JSON" or "Based on the keywords".
- Never include analysis, bullet lists, or explanations outside of the "rationale" string inside the JSON.
""".strip()



ROMANCE_AWARE_USER_PROMPT = """
### TOPIC DATA

TOPIC KEYWORDS (most important first):
{kw}{hints}

POS CUES (optional, extracted from keywords):
{pos}

REPRESENTATIVE SNIPPETS (short excerpts from the corpus):
{snippets}

OPTIONAL EXISTING LABELS (used elsewhere in the same corpus, avoid reusing them exactly):
{existing_labels}

### TASK

Using ONLY the information above, generate a JSON object following the schema described in the system message.

You are labeling topics in a corpus of modern heterosexual romantic and erotic fiction. Topics may correspond to:
- Romantic situations (dates, conversations, arguments, daily life).
- Sexual situations (foreplay, explicit acts, aftercare).
- Mixed emotional and physical scenes.
- Non-romantic or technical noise (which should be marked as noise).

### SAFETY AND PRECISION CHECKS (APPLY IN THIS ORDER)

1. SEXUAL TERMS CHECK
- First, scan the keywords for explicit sexual terms like: "sex", "fuck", "cock", "dick", "penis", "pussy", "clit", "clitoris", "orgasm", "blowjob", "handjob", "fingering", "penetration", "nipples", "breasts".
- If NO such explicit sexual terms are present:
  - You MUST use neutral, non-sexual wording in both "label" and "scene_summary".
  - FORBIDDEN in that case: "foreplay", "intimate", "erotic", "sexual tension", "arousal", "charged", "lust", "steamy".
  - Focus instead on activities, locations, and emotions that are clearly supported (e.g., "Board Game Around Table").
- If explicit sexual terms ARE present:
  - Use anatomically clear, non-euphemistic language when describing the scene.
  - If "clit" or "clitoris" appears near the top of the keywords, mention clitoral stimulation in the label and/or secondary categories when appropriate.
  - If "breasts" / "nipples" appear, prefer labels like "Breast And Nipple Foreplay" over generic "Intimate Kissing".

2. RELATIONSHIP / EMOTION CHECK
- When keywords emphasize: "relationship", "feelings", "years", "marriage", "divorce", "jealousy", "trust", "breakup", "reunion", "family", "home":
  - Prioritize romance and emotional categories ("romance_core", "relationship_conflict", "domestic_life").
  - Use labels like "Unclear Relationship Feelings", "Slow-Burn Romantic Tension", or "Breakup And Emotional Fallout".
  - Ensure labels avoid vagueness:
    - FORBIDDEN vague labels: "Never Seen Before", "Things That Matter", "Something Different", "Unusual Behavior".
    - Combine abstract words ("way", "matter", "things") with a concrete concept (e.g., "Uncertain Feelings About Relationship").

3. SCENE TYPE AND SETTING CHECK
- Use POS cues and snippets to refine the label and scene_summary:
  - Look for locations: bedroom, kitchen, hallway, bathroom, office, car, restaurant, bar, party, school.
  - Look for repeated actions: kissing, arguing, texting, undressing, cooking, driving, working.
  - Prefer labels that highlight the central action + setting (e.g., "Argument In Kitchen", "Makeout In Parked Car").
- Your scene_summary should:
  - Pick one typical micro-scene that fits most snippets.
  - Include at least one concrete detail (location, object, action, or body part) that is repeated.
  - Stay at sentence-level scale (no book-level or chapter-level summaries).

4. NOISE CHECK
- Mark "is_noise": true if the topic appears to be:
  - Boilerplate (e.g., copyright text, TOC, pagination).
  - Generic formatting or navigation text.
  - Isolated character names with no clear shared scene or theme.
  - Technical artefacts from preprocessing or file conversion.
- Otherwise, "is_noise": false.

5. DISCRIMINABILITY CHECK
- Assume this label will be compared against labels of many other topics from the same corpus.
- Make the label as discriminative as possible:
  - Prefer "Morning Commute Through City" over "Busy Day".
  - Prefer "Family Dinner Around Table" over "Family Moment".
- If EXISTING LABELS are provided, avoid copying them exactly.
- If you must reuse one, modify it slightly to make it more specific to this topic.

### OUTPUT

Now, using all checks above, produce the final JSON object.

REMINDERS:
- Do NOT include explanations outside the JSON.
- Do NOT use markdown or backticks.
- The JSON must match the schema from the system message exactly.
""".strip()



# OpenRouter API configuration
# Preferred: set OPENROUTER_API_KEY in your environment.
# You can still override with --api-key on the CLI.
DEFAULT_OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
# OpenRouter uses an OpenAI-compatible /v1 endpoint
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Default model for production romance/erotica topic labelling.
# You can override this via --model-name on the CLI.
DEFAULT_OPENROUTER_MODEL = "mistralai/Mistral-Nemo-Instruct-2407"
# Curated set of literary / roleplay-oriented models you might want to try.
# These are all accessed via the same OpenRouter API key.
ROLEPLAY_LITERARY_MODELS: dict[str, str] = {
    # General-purpose Nemo Instruct (good reasoning + instruction following)
    "nemo_instruct_2407": "mistralai/Mistral-Nemo-Instruct-2407",
    # Story-writing / roleplay model based on Nemo (very "literary" prose)
    "nemo_celeste": "nothingiisreal/mn-celeste-12b",
    # Gutenberg-tuned Nemo trained on book-like data (strong for fiction)
    "nemo_gutenberg_v2": "nbeerbower/mistral-nemo-gutenberg-12B-v2",
    # Alias for clarity
    "nemo_instruct": "mistralai/Mistral-Nemo-Instruct-2407",
}

# Module-level cache for MMR embedding model (loaded once, reused for all topics)
_MMR_EMBEDDING_MODEL: SentenceTransformer | None = None

# Module-level cache for spaCy NLP model (loaded once, reused for all topics)
_SPACY_NLP = None


def _get_embedding_model() -> SentenceTransformer:
    """Load (or reuse) a sentence embedding model for snippet centrality."""
    global _MMR_EMBEDDING_MODEL
    if _MMR_EMBEDDING_MODEL is None:
        LOGGER.info("Loading SentenceTransformer model for snippet centrality...")
        _MMR_EMBEDDING_MODEL = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    return _MMR_EMBEDDING_MODEL


def rerank_snippets_centrality(
    docs: list[str],
    top_k: int,
) -> list[str]:
    """
    Rerank representative docs by semantic centrality.

    - Embed each sentence.
    - Compute centroid (mean embedding).
    - Rank by cosine similarity to centroid.
    - Return top_k most central sentences.
    """
    if not docs:
        return []
    model = _get_embedding_model()
    embeddings = model.encode(docs, normalize_embeddings=True)
    if len(embeddings) == 0:
        return docs[:top_k]
    centroid = embeddings.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0.0:
        return docs[:top_k]
    centroid = centroid / norm
    sims = embeddings @ centroid  # cosine similarity (normalized)
    indices = np.argsort(-sims)[:top_k]
    return [docs[i] for i in indices]


def _load_spacy_model(enable_ner: bool = False):
    """Load spaCy model for POS tagging or NER (cached).
    
    Note: If model was previously loaded without NER and NER is now requested,
    the model will be reloaded with NER enabled. This means:
    - First call (e.g., from extract_pos_cues) loads model without NER (faster for POS)
    - Later call (e.g., from format_snippets with anonymize=True) reloads with NER
    This is a reasonable trade-off for simplicity, but not "free" - the model loads twice.
    
    Args:
        enable_ner: If True, enable NER component (needed for name anonymization)
    """
    global _SPACY_NLP
    if _SPACY_NLP is None and SPACY_AVAILABLE:
        try:
            if enable_ner:
                # Load with NER enabled for name anonymization
                _SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser"])
            else:
                # Load with NER disabled for POS tagging (faster)
                _SPACY_NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            LOGGER.info("Loaded spaCy model for %s", "NER and POS tagging" if enable_ner else "POS tagging")
        except OSError:
            LOGGER.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            return None
    elif _SPACY_NLP is not None and enable_ner and "ner" not in _SPACY_NLP.pipe_names:
        # Reload model with NER enabled if it was previously loaded without NER
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser"])
            LOGGER.info("Reloaded spaCy model with NER enabled")
        except Exception as e:
            LOGGER.warning("Could not reload spaCy model with NER: %s", e)
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


def anonymize_names(text: str, nlp) -> str:
    """
    Anonymize person and pet names in text by replacing them with generic role tokens.
    
    Uses spaCy NER to detect PERSON entities and replaces them with "[NAME]".
    This helps prevent the model from overfitting on specific character names in snippets.
    
    Args:
        text: Input text string
        nlp: spaCy language model (must have NER enabled)
        
    Returns:
        Text with person/pet names replaced by generic tokens
    """
    if not text or not nlp:
        return text
    
    try:
        # Check if NER is enabled
        if "ner" not in nlp.pipe_names:
            LOGGER.debug("NER not enabled in spaCy model, skipping anonymization")
            return text
        
        doc = nlp(text)
        if not doc.ents:
            return text
        
        result = text
        # Process entities in reverse order to preserve character indices
        # (process from end to start to avoid index shifting issues)
        entities = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)
        
        for ent in entities:
            if ent.label_ in {"PERSON"}:
                # Replace with generic token
                # Could be fancier (e.g., detect role from context), but [NAME] is sufficient
                result = result[:ent.start_char] + "[NAME]" + result[ent.end_char:]
        
        return result
    except Exception as e:
        LOGGER.warning("Error anonymizing names in text: %s", e)
        return text  # Return original text on error


def format_snippets(
    docs: list[str],
    max_snippets: int = 15,
    max_chars: int = 1200,
    anonymize: bool = True,
) -> str:
    """
    Convert a list of documents into a bullet-style snippets block for LLM prompts.
    
    Formats representative document snippets as numbered quotes, with truncation
    for long sentences. Designed for sentence-level documents (each doc is a sentence).
    Optionally anonymizes person/pet names to prevent overfitting on specific characters.
    
    Args:
        docs: List of document strings (sentences in this corpus)
        max_snippets: Maximum number of snippets to include (default: 15)
        max_chars: Maximum characters per snippet before truncation (default: 1200)
        anonymize: If True, anonymize person/pet names using spaCy NER (default: True)
        
    Returns:
        Formatted string with numbered snippets, or empty string if no docs provided
    """
    if not docs:
        return ""
    
    # Load spaCy model for anonymization if requested
    nlp = None
    if anonymize and SPACY_AVAILABLE:
        nlp = _load_spacy_model(enable_ner=True)
    
    snippets = []
    for i, doc in enumerate(docs[:max_snippets], start=1):
        # Collapse whitespace
        s = " ".join(doc.split())
        
        # Anonymize names if requested and spaCy is available
        if anonymize and nlp is not None:
            s = anonymize_names(s, nlp)
        
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
    Uses the official OpenAI Python client pointed at OpenRouter's /v1 endpoint.
    We also set the recommended identification headers.
    """
    with stage_timer_local(f"Initializing OpenRouter client: {model_name}"):
        LOGGER.info("Initializing OpenRouter API client")
        LOGGER.info("Model: %s", model_name)
        LOGGER.info("Base URL: %s", base_url)
        # Log API key status (masked for security)
        if api_key:
            api_key_display = (
                f"{api_key[:10]}...{api_key[-4:]}" if len(api_key) > 14 else "***"
            )
            LOGGER.info("API key: %s (length: %d)", api_key_display, len(api_key))
        else:
            LOGGER.warning("API key is empty or None! Authentication WILL fail.")
        # OpenRouter-specific headers (recommended but not strictly required)
        default_headers = {
            # Replace this with the URL of your project/repo if you want to appear on leaderboards
            "HTTP-Referer": "https://example.com/your-bertopic-project",
            "X-Title": "bertopic-romance-llm-labeling",
        }
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            default_headers=default_headers,
        )
        LOGGER.info("✓ OpenRouter client initialized successfully")
    return client, model_name


def test_openrouter_authentication(
    client: OpenAI,
    model_name: str,
) -> bool:
    """
    Test OpenRouter API authentication with a simple test call.
    
    Args:
        client: OpenRouter OpenAI client
        model_name: Model name to test
        
    Returns:
        True if authentication succeeds, False otherwise
    """
    try:
        LOGGER.info("Testing OpenRouter API authentication...")
        # Make a minimal test call (1 token request)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "test"}
            ],
            max_tokens=1,
        )
        if response.choices and len(response.choices) > 0:
            LOGGER.info("✓ API authentication test successful")
            return True
        else:
            LOGGER.warning("API authentication test returned empty response")
            return False
    except Exception as e:
        error_msg = str(e)
        LOGGER.error("✗ API authentication test FAILED: %s", error_msg)
        
        if "401" in error_msg or "Unauthorized" in error_msg or "User not found" in error_msg:
            LOGGER.error("AUTHENTICATION ERROR: API key is invalid, expired, or account doesn't exist")
            LOGGER.error("Please verify:")
            LOGGER.error("  1. API key is correct and active (check at https://openrouter.ai/keys)")
            LOGGER.error("  2. Account has sufficient credits/billing enabled")
            LOGGER.error("  3. Model '%s' is accessible with your account tier", model_name)
        elif "403" in error_msg or "Forbidden" in error_msg:
            LOGGER.error("ACCESS DENIED: Account may not have access to model '%s'", model_name)
            LOGGER.error("Please check if your account tier allows access to this model")
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            LOGGER.warning("Rate limit hit during test - but authentication may be OK")
            return True  # Rate limit means auth worked
        else:
            LOGGER.error("Unexpected error during authentication test")
        
        return False


# Generic bad labels to avoid
GENERIC_BAD_LABELS = {
    "Erotic Intimacy",
    "Erotic Encounter",
    "Romantic Moment",
    "Intense Love",
}


def clean_scene_summary(summary: str, keywords: list[str]) -> str:
    """Clean scene summary to remove hallucinations that violate constraints.
    
    Removes:
    - Car repair mentions when repair keywords are not present
    - Family relations (son/father/mother) when not in keywords
    
    Args:
        summary: Raw scene summary string
        keywords: List of topic keywords to check against
        
    Returns:
        Cleaned scene summary
    """
    if not summary:
        return summary
    
    summary_lower = summary.lower()
    keywords_lower = [kw.lower() for kw in keywords]
    
    # Check for car repair hallucinations
    repair_keywords = {"repair", "fix", "mechanic", "garage", "engine", "broken", "broken down"}
    has_repair_keywords = any(kw in keywords_lower for kw in repair_keywords)
    
    if not has_repair_keywords:
        # Remove car repair mentions
        original_summary = summary
        if "car repair" in summary_lower or "fix" in summary_lower:
            # Replace common patterns
            summary = re.sub(
                r"discuss(?:ing)?\s+car\s+repairs?\s+(?:and\s+)?(?:other\s+)?topics?",
                "discuss various topics",
                summary,
                flags=re.IGNORECASE,
            )
            summary = re.sub(
                r"fix\s+(?:hers?|his|their|the)\s+car",
                "their vehicles",
                summary,
                flags=re.IGNORECASE,
            )
            summary = re.sub(
                r"car\s+repairs?",
                "their cars",
                summary,
                flags=re.IGNORECASE,
            )
            if summary != original_summary:  # Only log if we made changes
                LOGGER.info("Removed car repair hallucination from scene summary")
    
    # Check for family relation hallucinations
    family_keywords = {"son", "father", "mother", "daughter", "dad", "mom", "parent", "parents"}
    has_family_keywords = any(kw in keywords_lower for kw in family_keywords)
    
    if not has_family_keywords:
        # Remove family relation mentions
        family_patterns = [
            (r"\bhis\s+son\b", "the player"),
            (r"\bher\s+son\b", "the player"),
            (r"\bthe\s+son\b", "the player"),
            (r"\bhis\s+father\b", "the goalie"),
            (r"\bher\s+father\b", "the goalie"),
            (r"\bthe\s+father\b", "the goalie"),
        ]
        original_summary = summary
        for pattern, replacement in family_patterns:
            summary = re.sub(pattern, replacement, summary, flags=re.IGNORECASE)
        if summary != original_summary:
            LOGGER.info("Removed family relation hallucination from scene summary")
    
    return summary.strip()


def normalize_label(raw: str, keywords: list[str] | None = None) -> str:
    """Normalize and clean a generated label.
    
    Args:
        raw: Raw label string from API response
        keywords: Optional list of topic keywords to check for sexual wording violations
        
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

    # Code-level safety net: Fix forbidden technical labels
    low = label.lower()
    if low == "time units passing" or low == "time passing units":
        label = "Waiting And Watching Clock"
        LOGGER.info("Fixed forbidden technical label '%s' → 'Waiting And Watching Clock'", raw)
    elif "units" in low and ("time" in low or "passing" in low):
        # Catch variants like "Time Passing Units", "Time Units", etc.
        label = re.sub(r"\bunits\b", "", label, flags=re.IGNORECASE)
        label = re.sub(r"\s+", " ", label).strip()
        if not label or len(label.split()) < 2:
            label = "Waiting And Watching Clock"
        LOGGER.info("Removed 'units' from technical time label: '%s' → '%s'", raw, label)
    
    # Code-level safety net: Remove sexual wording when no sexual keywords present
    if keywords is not None:
        keywords_lower = [kw.lower() for kw in keywords]
        sexual_keywords = {
            "sex", "fuck", "cock", "dick", "pussy", "clit", "clitoris", "nipples", "breasts",
            "orgasm", "cum", "come", "blowjob", "handjob", "fingering", "anal", "penetration",
            "pounding", "thrusting", "erection", "hard", "wet", "moan", "groan", "pleasure"
        }
        has_sexual_keywords = any(kw in keywords_lower for kw in sexual_keywords)
        
        if not has_sexual_keywords:
            # Check for forbidden sexual wording in label
            forbidden_sexual_words = ["foreplay", "intimate", "arousal", "erotic", "charged", "tension"]
            label_lower = label.lower()
            for word in forbidden_sexual_words:
                if word in label_lower:
                    # Replace with neutral alternative or remove
                    if "foreplay" in label_lower:
                        # If "game" is already in the label, just remove "foreplay"
                        if "game" in label_lower:
                            label = re.sub(r"\bforeplay\b", "", label, flags=re.IGNORECASE)
                        else:
                            label = re.sub(r"\bforeplay\b", "game", label, flags=re.IGNORECASE)
                        label = re.sub(r"\s+", " ", label).strip()  # Clean up extra spaces
                        LOGGER.info("Removed 'foreplay' from label (no sexual keywords): %s", raw)
                    elif "intimate" in label_lower:
                        label = re.sub(r"\bintimate\b", "", label, flags=re.IGNORECASE)
                        label = re.sub(r"\s+", " ", label).strip()
                        LOGGER.info("Removed 'intimate' from label (no sexual keywords): %s", raw)
                    break  # Only fix the first match
            
            # Check for family relations when not in keywords
            family_keywords = {"son", "father", "mother", "daughter", "dad", "mom", "parent", "parents"}
            has_family_keywords = any(kw in keywords_lower for kw in family_keywords)
            
            if not has_family_keywords:
                # Remove family relations from label
                label_lower = label.lower()
                if "son" in label_lower:
                    label = re.sub(r"\bwith\s+son\b", "", label, flags=re.IGNORECASE)
                    label = re.sub(r"\bson\b", "player", label, flags=re.IGNORECASE)
                    label = re.sub(r"\s+", " ", label).strip()
                    LOGGER.info("Removed 'son' from label (not in keywords): %s", raw)
                elif "father" in label_lower or "dad" in label_lower:
                    label = re.sub(r"\bwith\s+(?:father|dad)\b", "", label, flags=re.IGNORECASE)
                    label = re.sub(r"\b(?:father|dad)\b", "goalie", label, flags=re.IGNORECASE)
                    label = re.sub(r"\s+", " ", label).strip()
                    LOGGER.info("Removed family relation from label (not in keywords): %s", raw)

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
    max_snippets: int = 15,
    max_chars_per_snippet: int = 1200,
    existing_labels: set[str] | None = None,
    reasoning_effort: str | None = None,
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
        use_improved_prompts: If True, use romance-aware prompt with JSON output (includes label, scene_summary, categories, is_noise, rationale)
        representative_docs: Optional list of representative document strings (snippets)
        max_snippets: Maximum number of snippets to include in prompt (default: 15)
        max_chars_per_snippet: Maximum characters per snippet before truncation (default: 1200)
        existing_labels: Optional set of existing labels to avoid reusing (for romance-aware prompts)
        reasoning_effort: Optional reasoning effort level ("low", "medium", "high") for supported models
        
    Returns:
        Dictionary with 'label' and optionally 'scene_summary', 'primary_categories', 'secondary_categories', 'is_noise', 'rationale'
        If use_improved_prompts is True, returns full JSON structure with all fields
        If use_improved_prompts is False, returns dict with 'label' and 'scene_summary' (backward compatible)
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
    if use_improved_prompts:
        # Romance-aware JSON prompt (reuse JSON parsing pipeline)
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
            central_docs = rerank_snippets_centrality(
                representative_docs,
                top_k=max_snippets,
            )
            snippets_block = "\n\n" + format_snippets(
                central_docs,
                max_snippets=max_snippets,
                max_chars=max_chars_per_snippet,
            )
        
        # Format existing labels for prompt
        existing_labels_str = ""
        if existing_labels:
            existing_labels_str = "\n\nExisting labels in this dataset (avoid reusing them exactly):\n" + ", ".join(sorted(existing_labels))
        else:
            existing_labels_str = ""
        
        user_prompt = ROMANCE_AWARE_USER_PROMPT.format(
            kw=kw_str,
            hints=hints_str,
            pos=pos_str,
            snippets=snippets_block,
            existing_labels=existing_labels_str
        )
        messages = [
            {"role": "system", "content": ROMANCE_AWARE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        max_new_tokens = max(max_new_tokens, 220)  # More room for JSON
    else:
        # Legacy text-only romance-aware prompt (backward compatibility)
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
            central_docs = rerank_snippets_centrality(
                representative_docs,
                top_k=max_snippets,
            )
            snippets_block = "\n\n" + format_snippets(
                central_docs,
                max_snippets=max_snippets,
                max_chars=max_chars_per_snippet,
            )
        
        # Format existing labels for prompt
        existing_labels_str = ""
        if existing_labels:
            existing_labels_str = "\n\nExisting labels in this dataset (avoid reusing them exactly):\n" + ", ".join(sorted(existing_labels))
        else:
            existing_labels_str = ""
        
        user_prompt = ROMANCE_AWARE_USER_PROMPT.format(
            kw=kw_str,
            hints=hints_str,
            pos=pos_str,
            snippets=snippets_block,
            existing_labels=existing_labels_str
        )
        messages = [
            {"role": "system", "content": ROMANCE_AWARE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        # ROMANCE_AWARE_SYSTEM_PROMPT requests JSON output, so we need more tokens
        max_new_tokens = max(max_new_tokens, 220)  # More room for JSON
    
    # Build optional reasoning config for OpenRouter
    extra_body: dict[str, Any] | None = None
    if reasoning_effort and reasoning_effort.lower() not in ("none", "off"):
        extra_body = {
            "reasoning": {
                "effort": reasoning_effort.lower(),
                # We do NOT request the chain-of-thought text,
                # we only let the model reason more internally.
                "exclude": True,
            }
        }
        LOGGER.info("Using reasoning effort: %s (exclude=True)", reasoning_effort.lower())
    
    # Call OpenRouter API with timing
    try:
        api_start = time.perf_counter()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,  # Allows more diverse token selection for natural phrasing
            presence_penalty=0.0,  # No presence penalty
            frequency_penalty=0.3,  # Frequency penalty to discourage repetitive patterns
            # Note: seed removed for model comparisons to allow different models to produce different outputs
            # seed=42,  # Fixed seed for reproducibility (disabled for comparisons)
            extra_body=extra_body,  # Pass reasoning config if provided
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
        if use_improved_prompts:
            # Try to parse JSON response (romance-aware prompt with JSON output)
            try:
                # Extract JSON from response (might have markdown code blocks)
                json_content = content
                if "```json" in json_content:
                    json_content = json_content.split("```json")[1].split("```")[0].strip()
                elif "```" in json_content:
                    json_content = json_content.split("```")[1].split("```")[0].strip()
                
                result = json.loads(json_content)
                
                # Extract fields
                label_text = result.get("label", "")
                primary_categories = result.get("primary_categories", [])
                secondary_categories = result.get("secondary_categories", [])
                is_noise = result.get("is_noise", False)
                rationale = result.get("rationale", "")
                scene_summary = result.get("scene_summary", "")
                
                # Normalize label
                label = normalize_label(label_text, keywords=keywords)
                
                # If normalization resulted in empty label, use fallback
                if not label or label.strip() == "":
                    LOGGER.warning("Normalized label is empty (JSON prompt), using fallback for keywords: %s", keywords[:3])
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
                if scene_summary:
                    # Clean scene summary to remove hallucinations
                    scene_summary = clean_scene_summary(scene_summary, keywords)
                    result_dict["scene_summary"] = scene_summary
                
                LOGGER.info("Generated label (JSON prompt): %s | Categories: %s", 
                           label, primary_categories)
                return result_dict
                
            except (json.JSONDecodeError, KeyError) as e:
                LOGGER.warning("Failed to parse JSON response, falling back to text extraction: %s", e)
                # Fall through to text extraction below
        
        # Romance-aware prompt path (label + scene summary)
        # Since ROMANCE_AWARE_SYSTEM_PROMPT requests JSON, try JSON parsing first even when use_improved_prompts=False
        # Then fall back to text format if JSON parsing fails
        label_text = content
        scene_summary = ""
        
        # Try JSON parsing first (since prompt requests JSON)
        try:
            json_content = content
            if "```json" in json_content:
                json_content = json_content.split("```json")[1].split("```")[0].strip()
            elif "```" in json_content:
                json_content = json_content.split("```")[1].split("```")[0].strip()
            
            # Try to parse complete JSON first
            try:
                result = json.loads(json_content)
            except json.JSONDecodeError:
                # If complete JSON fails, try to extract from incomplete JSON
                # Look for "label" field even in truncated JSON
                label_match = re.search(r'"label"\s*:\s*"([^"]+)"', json_content)
                scene_match = re.search(r'"scene_summary"\s*:\s*"([^"]+)"', json_content)
                
                if label_match:
                    label_text = label_match.group(1)
                    scene_summary = scene_match.group(1) if scene_match else ""
                    
                    # If we extracted from incomplete JSON, use it
                    if label_text:
                        label = normalize_label(label_text, keywords=keywords)
                        if not label or label.strip() == "":
                            LOGGER.warning("Normalized label is empty (incomplete JSON), using fallback for keywords: %s", keywords[:3])
                            fallback_label = f"{keywords[0]}" if keywords else "Topic"
                            label = fallback_label
                        
                        if scene_summary:
                            scene_summary = clean_scene_summary(scene_summary, keywords)
                        
                        LOGGER.info("Generated label (incomplete JSON): %s | Scene summary: %s", label, scene_summary[:50] if scene_summary else "(none)")
                        return {"label": label, "scene_summary": scene_summary}
                raise  # Re-raise if we couldn't extract from incomplete JSON
            
            # Extract fields from complete JSON
            label_text = result.get("label", "")
            scene_summary = result.get("scene_summary", "")
            
            # If we got valid JSON, use it and skip text parsing
            if label_text:
                label = normalize_label(label_text, keywords=keywords)
                if not label or label.strip() == "":
                    LOGGER.warning("Normalized label is empty (JSON fallback), using fallback for keywords: %s", keywords[:3])
                    fallback_label = f"{keywords[0]}" if keywords else "Topic"
                    label = fallback_label
                
                if scene_summary:
                    scene_summary = clean_scene_summary(scene_summary, keywords)
                
                LOGGER.info("Generated label (JSON fallback): %s | Scene summary: %s", label, scene_summary[:50] if scene_summary else "(none)")
                return {"label": label, "scene_summary": scene_summary}
        except (json.JSONDecodeError, KeyError, AttributeError, ValueError) as e:
            # JSON parsing failed, fall through to text extraction
            LOGGER.debug("JSON parsing failed, trying text extraction: %s", e)
        
        # Text format fallback (for backward compatibility)
        # Expect format:
        #   Label: <...>
        #   Scene summary: <...>
        
        m_label = re.search(r"Label:\s*(.+)", content, flags=re.IGNORECASE)
        if m_label:
            label_text = m_label.group(1).strip()
            # Remove "Scene summary:" if it appears on the same line
            if "Scene summary:" in label_text:
                label_text = label_text.split("Scene summary:")[0].strip()
        
        m_scene = re.search(r"Scene summary:\s*(.+?)(?:\n\n|\nLabel:|$)", content, flags=re.IGNORECASE | re.DOTALL)
        if m_scene:
            scene_summary = m_scene.group(1).strip()
            # Clean up any trailing content - take first sentence/line but preserve full sentence
            # Split on newlines but keep the full first line/sentence
            lines = scene_summary.split("\n")
            scene_summary = lines[0].strip()
            # If the summary ends without proper punctuation and seems incomplete, log a warning
            # This often happens when max_tokens cuts off the response
            if scene_summary and not scene_summary.rstrip().endswith(('.', '!', '?')):
                # Check if it looks like it was cut off mid-sentence
                # If it ends with a comma, semicolon, colon, or lowercase letter without punctuation, it's likely incomplete
                if scene_summary.rstrip().endswith((',', ';', ':')) or (len(scene_summary) > 10 and scene_summary[-1].islower() and not any(c in scene_summary[-5:] for c in '.!?')):
                    LOGGER.warning("Scene summary appears truncated (likely due to max_tokens limit): '%s'", scene_summary[:80])
                    LOGGER.warning("Consider increasing --max-tokens if summaries are consistently incomplete")
            # Clean scene summary to remove hallucinations (car repairs, family relations)
            scene_summary = clean_scene_summary(scene_summary, keywords)
        
        # Normalize label text
        label = normalize_label(label_text, keywords=keywords)
        
        # If normalization resulted in empty label, use fallback
        if not label or label.strip() == "":
            LOGGER.warning("Normalized label is empty, using fallback for keywords: %s", keywords[:3])
            fallback_label = f"{keywords[0]}" if keywords else "Topic"
            LOGGER.info("Using fallback label: %s", fallback_label)
            label = fallback_label
        
        LOGGER.info("Generated label: %s | Scene summary: %s", label, scene_summary[:50] if scene_summary else "(none)")
        return {"label": label, "scene_summary": scene_summary}
    except Exception as e:
        error_msg = str(e)
        LOGGER.warning("Error generating label for keywords %s: %s", keywords[:3], error_msg)
        
        # Check for authentication errors specifically
        if "401" in error_msg or "Unauthorized" in error_msg or "User not found" in error_msg:
            LOGGER.error("AUTHENTICATION ERROR: API key may be invalid, expired, or account may not have access to model %s", model_name)
            LOGGER.error("Please verify:")
            LOGGER.error("  1. API key is correct and active")
            LOGGER.error("  2. Account has sufficient credits/billing enabled")
            LOGGER.error("  3. Model %s is accessible with your account tier", model_name)
        
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
    max_snippets: int = 15,
    max_chars_per_snippet: int = 1200,
    reasoning_effort: str | None = None,
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
        reasoning_effort: Optional reasoning effort level ("low", "medium", "high") for supported models
        
    Returns:
        Dictionary mapping topic_id to dict with 'label' and 'keywords' keys
    """
    # Use manual string construction to avoid pathlib truncation issues with long filenames
    json_path_str = str(output_path.parent) + "/" + output_path.name + ".json"
    json_path = Path(json_path_str)
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
    existing_labels: set[str] = set()  # Track existing labels to avoid duplicates
    
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
                    existing_labels=existing_labels if existing_labels else None,
                    reasoning_effort=reasoning_effort,
                )
                label_elapsed = time.perf_counter() - label_start
                
                # Extract label and scene_summary (result is now a dict)
                label = result.get("label", "")
                scene_summary = result.get("scene_summary", "")
                
                # Add label to existing_labels set for future topics
                if label:
                    existing_labels.add(label)
                
                # Store label, keywords, and any additional fields from improved prompts
                topic_data[topic_id] = {
                    "label": label,
                    "keywords": keywords,
                    **{k: v for k, v in result.items() if k != "label"}  # Add scene_summary, categories, is_noise, etc.
                }
                
                LOGGER.info("topic %d | label='%s' | generation_time=%.2fs", 
                           topic_id, label, label_elapsed)
                
                # Write to JSON incrementally with new structure
                if not first_item:
                    f.write(",\n")
                else:
                    first_item = False
                
                # Write this topic's data (label, scene_summary, and keywords)
                topic_entry = {
                    "label": label,
                    "scene_summary": scene_summary,
                    "keywords": keywords,
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
    max_snippets: int = 15,
    max_chars_per_snippet: int = 1200,
    reasoning_effort: str | None = None,
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
        reasoning_effort: Optional reasoning effort level ("low", "medium", "high") for supported models
        
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
    existing_labels: set[str] = set()  # Track existing labels to avoid duplicates
    
    with stage_timer_local("Generating labels for all topics"):
        LOGGER.info("Generating labels for %d topics (batch_size=%d)", total_topics, batch_size)
        
        # Process in batches
        batch_idx = 0
        batch_labels: dict[int, dict[str, Any]] = {}
        
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
                existing_labels=existing_labels if existing_labels else None,
                reasoning_effort=reasoning_effort,
            )
            label_elapsed = time.perf_counter() - label_start
            
            # Extract label (result is now a dict)
            label = result.get("label", "")
            
            # Add label to existing_labels set for future topics
            if label:
                existing_labels.add(label)
            
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
    # Use manual string construction to avoid pathlib truncation issues with long filenames
    # pathlib's with_suffix() and path joining can truncate in some cases
    json_path_str = str(output_path.parent) + "/" + output_path.name + ".json"
    json_path = Path(json_path_str)
    
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

