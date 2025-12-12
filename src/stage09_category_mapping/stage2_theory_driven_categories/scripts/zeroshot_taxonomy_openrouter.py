"""Stage 09: Zero-shot taxonomy mapping of BERTopic topics using OpenRouter (Mistral-Nemo).

This module assumes you already ran Stage 08 LLM labeling and have a JSON file
with enriched topic metadata of the form:

{
  "33": {
    "label": "Business Discussion",
    "keywords": ["business", "company", "dollars", ...],
    "primary_categories": ["domestic_life", "work_or_school"],
    "secondary_categories": ["setting:dinner_table", "activity:discussion"],
    "scene_summary": "The couple discusses business matters at the dinner table.",
    "is_noise": false,
    "rationale": "...",
    ...
  },
  ...
}

We now map each topic to a fixed Romance Corpus Topic Taxonomy using zero-shot
classification with mistralai/Mistral-Nemo-Instruct-2407 via OpenRouter.

The output is a JSON mapping:

{
  "33": {
    "topic_id": 33,
    "main_category_id": "6.1",
    "secondary_category_id": "5.1",
    "other_plausible_ids": ["4.2"],
    "is_noise": false,
    "rationale": "..."
  },
  ...
}
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from bertopic import BERTopic

# Reuse existing OpenRouter client + helpers from Stage 08
from src.stage08_llm_labeling.openrouter_experiments.core.generate_labels_openrouter import (
    DEFAULT_OPENROUTER_API_KEY,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_MODEL,
    load_openrouter_client,
    rerank_snippets_centrality,
    format_snippets,
    extract_representative_docs_per_topic,
)

# Reuse model loading helpers from Stage 1
from src.stage06_topic_exploration.explore_retrained_model import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    load_native_bertopic_model,
)

LOGGER = logging.getLogger("stage09_zeroshot_taxonomy")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# 1. Romance Corpus Topic Taxonomy
#    (reconstructed from our earlier design with your adjustments)
# ---------------------------------------------------------------------------

TAXONOMY_NODES: List[Dict[str, str]] = [
    # 1. Embodied & Sensory Experience
    {
        "id": "1.1",
        "name": "Body Parts & Physical Reactions",
        "group": "Embodied & Sensory Experience",
        "description": "Body sensations and reactions (heartbeat, breath, trembling, blushing, sweating, etc.).",
    },
    {
        "id": "1.2",
        "name": "Pain, Injury & Vulnerability",
        "group": "Embodied & Sensory Experience",
        "description": "Non-lethal pain, minor injuries, physical vulnerability not framed as deliberate violence.",
    },
    {
        "id": "1.5",
        "name": "Exercise & Physical Activity",
        "group": "Embodied & Sensory Experience",
        "description": "Sport and non-violent physical activity, workouts, training, running, dancing, hiking, practice.",
    },
    # 2. Sexuality, Attraction & Intimacy
    {
        "id": "2.1",
        "name": "Attraction & Sexual Tension",
        "group": "Sexuality, Attraction & Intimacy",
        "description": "Desire, longing, flirtation, sexual tension without explicit acts.",
    },
    {
        "id": "2.2",
        "name": "Kissing & Non-Explicit Affection",
        "group": "Sexuality, Attraction & Intimacy",
        "description": "Kissing, cuddling, touching that is affectionate but not clearly explicit.",
    },
    {
        "id": "2.3",
        "name": "Explicit Sexual Acts",
        "group": "Sexuality, Attraction & Intimacy",
        "description": "Clearly sexual behavior, oral/vaginal/anal sex, explicit stimulation, orgasms.",
    },
    {
        "id": "2.4",
        "name": "Aftercare & Post-Sex Reflection",
        "group": "Sexuality, Attraction & Intimacy",
        "description": "Aftercare, cuddling, emotional processing immediately after sex.",
    },
    # 3. Emotions, Cognition & Inner Life
    {
        "id": "3.1",
        "name": "Positive Emotions & Security",
        "group": "Emotions, Cognition & Inner Life",
        "description": "Joy, comfort, emotional safety, feeling loved and accepted.",
    },
    {
        "id": "3.2",
        "name": "Negative Emotions & Distress",
        "group": "Emotions, Cognition & Inner Life",
        "description": "Sadness, fear, shame, anxiety, emotional turmoil.",
    },
    {
        "id": "3.3",
        "name": "Ambivalence & Internal Conflict",
        "group": "Emotions, Cognition & Inner Life",
        "description": "Mixed feelings, indecision, cognitive dissonance about relationship or life choices.",
    },
    {
        "id": "3.4",
        "name": "Beliefs, Values & Moral Reflection",
        "group": "Emotions, Cognition & Inner Life",
        "description": "Characters reflecting on norms, values, ethics, promises, duties.",
    },
    # 4. Relationship Trajectory (Main Couple Only)
    {
        "id": "4.1",
        "name": "Meeting, First Impressions & Setup",
        "group": "Relationship Trajectory (Main Couple)",
        "description": "First encounters, initial attraction or dislike, meet-cutes.",
    },
    {
        "id": "4.2",
        "name": "Bonding, Everyday Intimacy & Growth",
        "group": "Relationship Trajectory (Main Couple)",
        "description": "Dates, everyday closeness, trust building, deepening connection.",
    },
    {
        "id": "4.3",
        "name": "Secrets, Misunderstandings & Hidden Information",
        "group": "Relationship Trajectory (Main Couple)",
        "description": "Concealed facts, misunderstandings, withheld truths between the main couple.",
    },
    {
        "id": "4.4",
        "name": "Conflict, Distance & Breakup Threats",
        "group": "Relationship Trajectory (Main Couple)",
        "description": "Arguments, distancing, threats of breakup, serious relational strain.",
    },
    {
        "id": "4.5",
        "name": "Reconciliation, Commitments & HEA",
        "group": "Relationship Trajectory (Main Couple)",
        "description": "Apologies, reunions, proposals, explicit commitments, 'happily ever after' moves.",
    },
    # 5. Social World Outside Main Couple
    {
        "id": "5.1",
        "name": "Family & Kinship",
        "group": "Social World Outside Couple",
        "description": "Parents, children, siblings, in-laws, family obligations, pregnancy, parenthood.",
    },
    {
        "id": "5.2",
        "name": "Friends & Social Circles",
        "group": "Social World Outside Couple",
        "description": "Friends, colleagues as social figures, found family, everyday social support.",
    },
    {
        "id": "5.3",
        "name": "Community, Norms & Social Events",
        "group": "Social World Outside Couple",
        "description": "Parties, weddings, holidays, community judgment, gossip, public rituals.",
    },
    # 6. Work, Wealth, Status & Institutions
    {
        "id": "6.1",
        "name": "Hero's Elite Work & Business World",
        "group": "Work, Wealth, Status & Institutions",
        "description": "Billionaire/CEO work, deals, negotiations, high-status professional life of the hero.",
    },
    {
        "id": "6.2",
        "name": "Heroine's Work & Professional Identity",
        "group": "Work, Wealth, Status & Institutions",
        "description": "Heroine's job (e.g., teacher, doctor, assistant), often lower-paid or less prestigious.",
    },
    {
        "id": "6.3",
        "name": "Shared Workplaces & Professional Interaction",
        "group": "Work, Wealth, Status & Institutions",
        "description": "Scenes where main couple interacts within a shared work or institutional setting.",
    },
    {
        "id": "6.4",
        "name": "Money, Housing & Economic Security",
        "group": "Work, Wealth, Status & Institutions",
        "description": "Financial worries, rent, housing stability, debts, economic dependency.",
    },
    {
        "id": "6.5",
        "name": "Law, Medicine, Education & Formal Institutions",
        "group": "Work, Wealth, Status & Institutions",
        "description": "Courts, hospitals, schools, universities, state bureaucracy as institutions.",
    },
    # 7. Conflict, Risk & Harm (Non-sexual)
    {
        "id": "7.1",
        "name": "Interpersonal Non-Romantic Conflict",
        "group": "Conflict, Risk & Harm",
        "description": "Conflicts with bosses, family, friends or antagonists outside the main couple.",
    },
    {
        "id": "7.2",
        "name": "Violence, Threats & Coercion",
        "group": "Conflict, Risk & Harm",
        "description": "Physical violence, threats, coercive control, clear harm or menace.",
    },
    {
        "id": "7.3",
        "name": "Risk, Danger & External Crises",
        "group": "Conflict, Risk & Harm",
        "description": "Accidents, crime, disasters, illness as external threats.",
    },
    # 8. Spaces, Time, Activities & Objects
    {
        "id": "8.1",
        "name": "Domestic Spaces & Routines",
        "group": "Spaces, Time, Activities & Objects",
        "description": "Home, kitchen, bedroom as setting; chores, domestic routines.",
    },
    {
        "id": "8.2",
        "name": "Public & Leisure Spaces",
        "group": "Spaces, Time, Activities & Objects",
        "description": "Restaurants, bars, hotels, parks, travel, holidays, public outings.",
    },
    {
        "id": "8.3",
        "name": "Objects, Technology & Everyday Artefacts",
        "group": "Spaces, Time, Activities & Objects",
        "description": "Phones, cars, clothes, jewelry, gifts, symbolic objects.",
    },
    {
        "id": "8.4",
        "name": "Time, Seasons & Temporal Framing",
        "group": "Spaces, Time, Activities & Objects",
        "description": "Passage of time, schedules, delays, seasons, holidays as temporal structure.",
    },
    # Special "noise" category for junk topics
    {
        "id": "noise",
        "name": "Noise / Technical / Paratext",
        "group": "Special",
        "description": "Boilerplate, front/back matter, artefacts, non-story material.",
    },
]


def _taxonomy_block_for_prompt() -> str:
    """Render taxonomy into a compact text block for the system prompt."""
    lines = []
    for node in TAXONOMY_NODES:
        lines.append(
            f"- {node['id']} — {node['name']} "
            f"({node['group']}): {node['description']}"
        )
    return "\n".join(lines)


TAXONOMY_TEXT_BLOCK = _taxonomy_block_for_prompt()


# ---------------------------------------------------------------------------
# 2. System + user prompts for zero-shot taxonomy mapping
#    Optimized for Mistral-Nemo, JSON-only output, similar style to Stage 08.
# ---------------------------------------------------------------------------

TAXONOMY_ZEROSHOT_SYSTEM_PROMPT = f"""
You are RomanceTaxonomyMapper, an expert assistant for assigning topics from modern heterosexual romantic and erotic fiction to a fixed analytic taxonomy.

You will receive, for each topic:

- A topic_id

- TOPIC KEYWORDS from BERTopic

- An LLM-generated label and scene_summary from a previous stage

- Primary and secondary categories from the earlier labeler (e.g., "romance_core", "sexual_content", "setting:bedroom", "activity:kissing")

- Optional representative snippets from the corpus

Your task is to map this topic to one or two nodes in a fixed Romance Corpus Topic Taxonomy.

IMPORTANT: This is ZERO-SHOT classification.

- The taxonomy is fixed and must NOT be modified.

- You must select IDs only from the taxonomy list shown below.

AVAILABLE TAXONOMY NODES

(Use these IDs exactly; do NOT invent new ones):

{TAXONOMY_TEXT_BLOCK}

OUTPUT CONSTRAINTS

- Think through the mapping internally.

- In your final answer, output only a valid JSON object.

- Do NOT include markdown, backticks, or any explanation outside the JSON.

- Never wrap JSON in ```json or any other formatting.

- Use only taxonomy IDs listed above, or "noise" for junk topics.

JSON SCHEMA (MANDATORY)

Return exactly these keys and types:

{{
  "topic_id": 0,
  "main_category_id": "4.2",
  "secondary_category_id": "5.1",
  "other_plausible_ids": ["3.2", "6.4"],
  "is_noise": false,
  "confidence": "medium",
  "rationale": "1–3 short sentences explaining why these IDs fit this topic."
}}

FIELD RULES

1) "topic_id"

- Echo the integer topic id from the input.

2) "main_category_id"

- REQUIRED.

- One taxonomy ID that best captures the central function of the topic.

- Use:

  - 4.x for dynamics of the main romantic couple.

  - 5.x for social world outside the main couple (family, friends, community).

  - 6.x for work, money, heroine/hero jobs, institutional scenes.

  - 2.x for sexual attraction/acts/intimacy.

  - 1.5 for any non-violent sport or physical training, workouts, exercise.

  - 7.x for risk, harm, violence, coercion, non-romantic conflicts.

  - 8.x when the topic is primarily about spaces, time, or objects.

  - 3.x when the topic is mostly inner feelings, beliefs, cognitive states.

  - "noise" only if the topic is mostly boilerplate or paratext.

3) "secondary_category_id"

- OPTIONAL but recommended.

- A second taxonomy ID when the topic clearly blends two dimensions
  (e.g., argument in kitchen → 4.4 + 8.1; pregnancy conflict → 5.1 + 3.2).

- Use null if there is no meaningful second dimension.

4) "other_plausible_ids"

- OPTIONAL list (0–3 items) of other taxonomy IDs that are plausible,
  but clearly less central than main_category_id and secondary_category_id.

5) "is_noise"

- true only if the topic is mostly boilerplate, technical artefacts, or paratext.

- If true, main_category_id MUST be "noise" and secondary_category_id MUST be null.

- If false, main_category_id MUST NOT be "noise".

6) "confidence"

- REQUIRED.

- One of: "low", "medium", "high".

- "high" = strong, unambiguous match to one taxonomy node.

- "medium" = reasonably clear, a couple of plausible alternatives.

- "low" = noisy or ambiguous topic, mapping is uncertain.

7) "rationale"

- 1–3 short sentences.

- Refer to:

  - specific high-weight keywords,

  - the label and scene_summary,

  - and any primary/secondary categories in the input.

- Explain why these support the chosen taxonomy IDs.

- Do NOT quote long snippets verbatim; summarize them instead.

SPECIAL RULES ABOUT VIOLENCE VS EXERCISE

- Any physical activity that is NOT clearly harmful or coercive
  (e.g., workouts, sport, training, running, dance practice) should use 1.5.

- Only use 7.2 (Violence, Threats & Coercion) when the language or scenes clearly
  indicate harm, threat, assault, or coercion.

NO REASONING OUTSIDE JSON

- You may think step-by-step internally.

- The final answer must be only the JSON object described above.
""".strip()


TAXONOMY_ZEROSHOT_USER_PROMPT = """
### TOPIC DATA

topic_id: {topic_id}

TOPIC KEYWORDS (most important first):

{keywords}

PREVIOUS LLM LABEL:

{label}

PREVIOUS SCENE SUMMARY:

{scene_summary}

PREVIOUS PRIMARY CATEGORIES:

{primary_categories}

PREVIOUS SECONDARY CATEGORIES:

{secondary_categories}

REPRESENTATIVE SNIPPETS (optional):

{snippets}

### TASK

Using ONLY the information above and the taxonomy defined in the system message:

- Decide whether this topic is noise or meaningful.

- If meaningful, choose:

  - one main_category_id (required),

  - an optional secondary_category_id,

  - optional other_plausible_ids (0–3).

- If noise, set main_category_id to "noise", secondary_category_id to null, and is_noise to true.

Return a SINGLE JSON object following the schema in the system message.

Do NOT include explanations outside the JSON.
""".strip()


# ---------------------------------------------------------------------------
# 3. Core function: classify a single topic into the taxonomy
# ---------------------------------------------------------------------------

def classify_topic_to_taxonomy_openrouter(
    *,
    topic_id: int,
    topic_metadata: Dict[str, Any],
    client: OpenAI,
    model_name: str,
    temperature: float = 0.25,
    max_new_tokens: int = 220,
    representative_docs: Optional[List[str]] = None,
    max_snippets: int = 8,
    max_chars_per_snippet: int = 400,
) -> Dict[str, Any]:
    """
    Classify a single BERTopic topic into the Romance Corpus Topic Taxonomy
    using Mistral-Nemo via OpenRouter.

    Parameters
    ----------
    topic_id:
        Integer topic id.
    topic_metadata:
        Dict with at least:
        - "keywords": List[str]
        - "label": str (from Stage 08)
        - "scene_summary": str (optional but recommended)
        - "primary_categories": List[str] (optional)
        - "secondary_categories": List[str] (optional)
        - "is_noise": bool (optional hint)
    client:
        OpenRouter OpenAI-compatible client.
    model_name:
        Model name (e.g., "mistralai/Mistral-Nemo-Instruct-2407").
    temperature:
        Sampling temperature (low-ish for stable classification).
    max_new_tokens:
        Max tokens for JSON output.
    representative_docs:
        Optional list of representative doc snippets for this topic.
    max_snippets:
        Max snippets to include.
    max_chars_per_snippet:
        Max characters per snippet.

    Returns
    -------
    Dict with keys:
        "topic_id", "main_category_id", "secondary_category_id",
        "other_plausible_ids", "is_noise", "rationale"
    """
    keywords = topic_metadata.get("keywords", [])
    label = topic_metadata.get("label", "")
    scene_summary = topic_metadata.get("scene_summary", "")
    primary_categories = topic_metadata.get("primary_categories", [])
    secondary_categories = topic_metadata.get("secondary_categories", [])
    prev_is_noise = topic_metadata.get("is_noise", False)

    kw_str = ", ".join(keywords) if keywords else "(no keywords)"
    primary_str = ", ".join(primary_categories) if primary_categories else "(none)"
    secondary_str = ", ".join(secondary_categories) if secondary_categories else "(none)"

    # Format representative snippets using same helper as Stage 08
    snippets_block = "(none)"
    if representative_docs:
        central_docs = rerank_snippets_centrality(
            representative_docs,
            top_k=max_snippets,
        )
        formatted = format_snippets(
            central_docs,
            max_snippets=max_snippets,
            max_chars=max_chars_per_snippet,
            anonymize=True,
        )
        snippets_block = formatted if formatted else "(none)"

    user_prompt = TAXONOMY_ZEROSHOT_USER_PROMPT.format(
        topic_id=topic_id,
        keywords=kw_str,
        label=label or "(no label)",
        scene_summary=scene_summary or "(no scene summary)",
        primary_categories=primary_str,
        secondary_categories=secondary_str,
        snippets=snippets_block,
    )

    messages = [
        {"role": "system", "content": TAXONOMY_ZEROSHOT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    LOGGER.info("Classifying topic %d into taxonomy (Mistral-Nemo)...", topic_id)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        frequency_penalty=0.3,
        presence_penalty=0.0,
    )

    if not response.choices:
        raise ValueError("Empty API response for taxonomy classification")

    content = response.choices[0].message.content.strip()
    LOGGER.debug("Raw taxonomy response for topic %d: %s", topic_id, content[:300])

    # Extract JSON (strip optional code fences defensively)
    json_content = content
    if "```json" in json_content:
        json_content = json_content.split("```json")[1].split("```")[0].strip()
    elif "```" in json_content:
        json_content = json_content.split("```")[1].split("```")[0].strip()

    try:
        result = json.loads(json_content)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse taxonomy JSON for topic {topic_id}: {e}\nContent:\n{content}"
        ) from e

    # Minimal sanity checks + gentle corrections
    result_topic_id = result.get("topic_id", topic_id)
    if result_topic_id != topic_id:
        LOGGER.warning(
            "Model echoed different topic_id (%s) than input (%s); overriding with input.",
            result_topic_id,
            topic_id,
        )
        result["topic_id"] = topic_id

    main_id = result.get("main_category_id")
    sec_id = result.get("secondary_category_id", None)
    is_noise = bool(result.get("is_noise", False))

    valid_ids = {node["id"] for node in TAXONOMY_NODES}

    # If model says noise, enforce noise semantics
    if is_noise:
        result["main_category_id"] = "noise"
        result["secondary_category_id"] = None
    else:
        # Fix missing or invalid main_category_id
        if not main_id or main_id not in valid_ids or main_id == "noise":
            # If previous stage already marked this topic as noise, keep it noise
            if prev_is_noise:
                result["main_category_id"] = "noise"
                result["secondary_category_id"] = None
                result["is_noise"] = True
            else:
                # Fallback: guess from previous categories
                # Very lightweight heuristics
                fallback = "4.2"  # generic romantic bonding
                if "sexual_content" in primary_categories:
                    fallback = "2.3"
                elif "physical_affection" in primary_categories:
                    fallback = "2.2"
                elif "work_or_school" in primary_categories:
                    fallback = "6.1"
                elif "domestic_life" in primary_categories:
                    fallback = "8.1"
                elif "relationship_conflict" in primary_categories:
                    fallback = "4.4"
                result["main_category_id"] = fallback
                result["is_noise"] = False

        # Validate secondary ID
        if sec_id is not None and sec_id not in valid_ids:
            result["secondary_category_id"] = None

    # Normalize other_plausible_ids
    other_ids = result.get("other_plausible_ids", [])
    if not isinstance(other_ids, list):
        other_ids = []
    filtered_other = []
    for cid in other_ids:
        if (
            isinstance(cid, str)
            and cid in valid_ids
            and cid not in {result["main_category_id"], result.get("secondary_category_id")}
        ):
            filtered_other.append(cid)
    result["other_plausible_ids"] = filtered_other

    # Normalize confidence
    confidence = result.get("confidence", None)
    valid_conf = {"low", "medium", "high"}
    if not isinstance(confidence, str) or confidence.lower() not in valid_conf:
        # Simple heuristic: if we had to fall back or fix IDs, treat as low confidence
        if is_noise:
            confidence = "medium"
        else:
            confidence = "low"
    else:
        confidence = confidence.lower()
    result["confidence"] = confidence

    LOGGER.info(
        "Topic %d → main=%s, secondary=%s, noise=%s",
        topic_id,
        result.get("main_category_id"),
        result.get("secondary_category_id"),
        result.get("is_noise"),
    )

    return result


# ---------------------------------------------------------------------------
# 4. Batch mapping utility: map all topics from a labels JSON file
# ---------------------------------------------------------------------------

def load_topic_metadata(labels_json_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Load Stage 08 topic metadata JSON (labels + keywords + categories etc.)
    and convert keys to int.
    """
    with open(labels_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta: Dict[int, Dict[str, Any]] = {}
    for k, v in data.items():
        try:
            tid = int(k)
        except ValueError:
            continue
        meta[tid] = v
    return meta


def load_bertopic_model_for_snippets(
    *,
    base_dir: Path = DEFAULT_BASE_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    model_suffix: str = "_with_llm_labels",
    stage_subfolder: Optional[str] = "stage08_llm_labeling",
    max_docs_per_topic: int = 10,
) -> Optional[Dict[int, List[str]]]:
    """
    Load BERTopic model and extract representative documents for snippets.
    
    Parameters
    ----------
    base_dir:
        Base directory for models (default: models/retrained).
    embedding_model:
        Embedding model name (default: paraphrase-MiniLM-L6-v2).
    model_suffix:
        Model suffix (default: "_with_llm_labels").
    stage_subfolder:
        Optional stage subfolder (default: "stage08_llm_labeling").
    max_docs_per_topic:
        Maximum number of representative docs to extract per topic.
        
    Returns
    -------
    Dict mapping topic_id to list of representative document strings, or None if loading fails.
    """
    try:
        LOGGER.info("Loading BERTopic model for representative document extraction...")
        LOGGER.info("  Base dir: %s", base_dir)
        LOGGER.info("  Embedding model: %s", embedding_model)
        LOGGER.info("  Model suffix: %s", model_suffix)
        LOGGER.info("  Stage subfolder: %s", stage_subfolder)
        
        topic_model = load_native_bertopic_model(
            base_dir=base_dir,
            embedding_model=embedding_model,
            pareto_rank=1,
            model_suffix=model_suffix,
            stage_subfolder=stage_subfolder,
        )
        
        LOGGER.info("✓ BERTopic model loaded successfully")
        
        # Extract representative documents
        LOGGER.info("Extracting representative documents from model...")
        topic_to_snippets = extract_representative_docs_per_topic(
            topic_model,
            max_docs_per_topic=max_docs_per_topic,
        )
        
        snippets_count = len([tid for tid, docs in topic_to_snippets.items() if docs])
        avg_snippets = (
            sum(len(docs) for docs in topic_to_snippets.values()) / max(snippets_count, 1)
        )
        LOGGER.info(
            "✓ Extracted representative docs for %d topics (avg %.1f docs per topic)",
            snippets_count,
            avg_snippets,
        )
        
        return topic_to_snippets
        
    except Exception as e:
        LOGGER.warning(
            "Failed to load BERTopic model or extract snippets: %s. "
            "Taxonomy classification will proceed without representative snippets.",
            e,
        )
        return None


def map_all_topics_to_taxonomy(
    *,
    labels_json_path: Path,
    output_path: Path,
    client: Optional[OpenAI] = None,
    model_name: str = DEFAULT_OPENROUTER_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.25,
    max_new_tokens: int = 220,
    topic_to_snippets: Optional[Dict[int, List[str]]] = None,
    # Model loading parameters for snippet extraction
    load_model_for_snippets: bool = True,
    base_dir: Path = DEFAULT_BASE_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    model_suffix: str = "_with_llm_labels",
    stage_subfolder: Optional[str] = "stage08_llm_labeling",
    max_docs_per_topic: int = 10,
    limit_topics: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Run zero-shot taxonomy mapping for all topics that have LLM labels.

    Parameters
    ----------
    labels_json_path:
        Path to Stage 08 labels JSON (as produced by generate_labels_openrouter.save_labels_openrouter).
    output_path:
        Path to write taxonomy mapping JSON (topic_id → mapping dict).
    client:
        Optional existing OpenRouter client. If None, a new one is created via load_openrouter_client.
    model_name:
        Model name for classification.
    api_key:
        Optional API key (if client is None). If None, environment variable is used.
    temperature, max_new_tokens:
        Generation params.
    topic_to_snippets:
        Optional dict[topic_id → list[str]] of representative docs.
        If None and load_model_for_snippets=True, will attempt to load from BERTopic model.
    load_model_for_snippets:
        If True and topic_to_snippets is None, load BERTopic model and extract snippets.
    base_dir, embedding_model, model_suffix, stage_subfolder:
        Parameters for loading BERTopic model (used if load_model_for_snippets=True).
    max_docs_per_topic:
        Maximum number of representative docs to extract per topic.

    Returns
    -------
    Dict[int, Dict[str, Any]] mapping topic_id → taxonomy mapping.
    """
    # Initialize client if needed
    if client is None:
        client, _ = load_openrouter_client(
            api_key=api_key or "",
            model_name=model_name,
        )

    # Load topic metadata
    topic_meta = load_topic_metadata(labels_json_path)
    topic_ids = sorted(topic_meta.keys())
    total = len(topic_ids)
    LOGGER.info("Loaded metadata for %d topics from %s", total, labels_json_path)
    
    # Apply limit if specified (for testing)
    if limit_topics is not None and limit_topics > 0:
        topic_ids = topic_ids[:limit_topics]
        total = len(topic_ids)
        LOGGER.info("Limited to first %d topics for testing", limit_topics)

    # Load snippets from BERTopic model if requested and not provided
    if topic_to_snippets is None and load_model_for_snippets:
        LOGGER.info("Loading BERTopic model to extract representative documents...")
        topic_to_snippets = load_bertopic_model_for_snippets(
            base_dir=base_dir,
            embedding_model=embedding_model,
            model_suffix=model_suffix,
            stage_subfolder=stage_subfolder,
            max_docs_per_topic=max_docs_per_topic,
        )
        if topic_to_snippets:
            LOGGER.info(
                "Using representative snippets from BERTopic model for taxonomy classification"
            )
        else:
            LOGGER.info("Proceeding without representative snippets")
    elif topic_to_snippets is None:
        LOGGER.info("No representative snippets provided, using keywords and labels only")

    taxonomy_map: Dict[int, Dict[str, Any]] = {}

    for idx, tid in enumerate(topic_ids, start=1):
        tm = topic_meta[tid]
        snippets = topic_to_snippets.get(tid, []) if topic_to_snippets else None
        result = classify_topic_to_taxonomy_openrouter(
            topic_id=tid,
            topic_metadata=tm,
            client=client,
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            representative_docs=snippets,
        )
        taxonomy_map[tid] = result

        if idx % 10 == 0 or idx == total:
            LOGGER.info(
                "Processed %d/%d topics (%.1f%%)", idx, total, idx / total * 100.0
            )

    # Save to JSON (string keys for stability)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(k): v for k, v in taxonomy_map.items()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    LOGGER.info(
        "Saved taxonomy mappings for %d topics to %s", len(serializable), output_path
    )
    return taxonomy_map


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 09: Zero-shot taxonomy mapping of BERTopic topics using Mistral via OpenRouter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--labels-json",
        type=Path,
        required=True,
        help="Path to Stage 08 labels JSON (labels_pos_openrouter_..._romance_aware_*.json).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Where to save taxonomy mappings JSON.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_OPENROUTER_MODEL,
        help="OpenRouter model name to use.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="OpenRouter API key (optional; otherwise environment variable is used).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.25,
        help="Sampling temperature (low for stable classification).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=220,
        help="Maximum new tokens for JSON output.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base directory for BERTopic models (default: models/retrained).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name (default: paraphrase-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--model-suffix",
        type=str,
        default="_with_llm_labels",
        help="Model suffix (default: _with_llm_labels).",
    )
    parser.add_argument(
        "--model-stage",
        type=str,
        default="stage08_llm_labeling",
        help="Stage subfolder for model (default: stage08_llm_labeling).",
    )
    parser.add_argument(
        "--max-docs-per-topic",
        type=int,
        default=10,
        help="Maximum number of representative docs to extract per topic (default: 10).",
    )
    parser.add_argument(
        "--no-snippets",
        action="store_true",
        help="Skip loading BERTopic model and extracting representative snippets.",
    )
    parser.add_argument(
        "--limit-topics",
        type=int,
        default=None,
        help="Limit processing to first N topics (for testing, default: process all).",
    )

    args = parser.parse_args()

    client, _ = load_openrouter_client(
        api_key=args.api_key or "",
        model_name=args.model_name,
    )

    map_all_topics_to_taxonomy(
        labels_json_path=args.labels_json,
        output_path=args.output_json,
        client=client,
        model_name=args.model_name,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        load_model_for_snippets=not args.no_snippets,
        base_dir=args.base_dir,
        embedding_model=args.embedding_model,
        model_suffix=args.model_suffix,
        stage_subfolder=args.model_stage,
        max_docs_per_topic=args.max_docs_per_topic,
        limit_topics=args.limit_topics,
    )

