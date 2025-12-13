"""Stage 3: Zero-shot mapping of BERTopic topics to Radway's 13 narrative functions using Mistral-Nemo via OpenRouter.

This module uses the taxonomy JSON (from Stage 2) as the single source of truth, which already contains:
- Taxonomy mappings (main_category_id, secondary_category_id, etc.)
- Source metadata (label, keywords, scene_summary, primary/secondary categories)

The output merges Radway function mappings back into the taxonomy JSON under a "radway_functions" key,
preserving all existing fields.

OUTPUT: A merged JSON mapping with structure:
{
  "33": {
    "topic_id": 33,
    ... (all existing taxonomy fields) ...,
    "radway_functions": {
      "radway_main_id": "R8",
      "radway_secondary_id": "R9",
      "radway_other_plausible_ids": ["R10"],
      "radway_phase": "II",
      "radway_is_none": false,
      "radway_confidence": "medium",
      "radway_rationale": "...",
      "radway_main_name": "Hero treats heroine tenderly",
      "radway_phase_name": "Turning Point & Recognition"
    }
  },
  ...
}
"""

from __future__ import annotations

import json
import logging
import shutil
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

# Reuse taxonomy structure from Stage 2 for taxonomy name/group lookups
from src.stage09_category_mapping.stage2_theory_driven_categories.scripts.zeroshot_taxonomy_openrouter import (
    TAXONOMY_BY_ID,
)

# Reuse model loading helpers from Stage 6
from src.stage06_topic_exploration.explore_retrained_model import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    load_native_bertopic_model,
)

LOGGER = logging.getLogger("stage3_radway_functions")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# 1. Radway's 13 narrative functions + "none"
#    (structured with phases and natural-language descriptions)
# ---------------------------------------------------------------------------

RADWAY_FUNCTIONS: List[Dict[str, str]] = [
    # Phase I: Initial Conflict & Isolation (The Setup)
    {
        "id": "R1",
        "name": "Heroine's social identity is destroyed",
        "phase": "I",
        "phase_name": "Initial Conflict & Isolation",
        "description": (
            "The heroine experiences a major blow to her social identity or self-concept: "
            "public humiliation, career-damaging event, loss of status, shame, regret, or "
            "sudden vulnerability that leaves her isolated and distressed."
        ),
    },
    {
        "id": "R2",
        "name": "Heroine reacts antagonistically to the hero",
        "phase": "I",
        "phase_name": "Initial Conflict & Isolation",
        "description": (
            "The heroine pushes back against the hero with anger, sarcasm, or open hostility, "
            "often seeing him as arrogant, entitled, or threatening."
        ),
    },
    {
        "id": "R3",
        "name": "Hero responds ambiguously to heroine",
        "phase": "I",
        "phase_name": "Initial Conflict & Isolation",
        "description": (
            "The hero's behaviour is mixed, ambiguous, or hard to read: cold then kind, "
            "sending conflicting signals that create confusion and tension."
        ),
    },
    {
        "id": "R4",
        "name": "Heroine interprets hero's behaviour as purely sexual interest",
        "phase": "I",
        "phase_name": "Initial Conflict & Isolation",
        "description": (
            "The heroine reads the hero's attention primarily as sexual objectification "
            "or predatory desire, focusing on physical attraction and tension."
        ),
    },
    {
        "id": "R5",
        "name": "Heroine responds with anger or coldness",
        "phase": "I",
        "phase_name": "Initial Conflict & Isolation",
        "description": (
            "The heroine meets the hero's interest with emotional withdrawal, icy politeness, "
            "or snapping back in anger; guarded, defensive responses dominate."
        ),
    },
    {
        "id": "R6",
        "name": "Hero retaliates or punishes heroine",
        "phase": "I",
        "phase_name": "Initial Conflict & Isolation",
        "description": (
            "The hero uses his power or status to retaliate, punish, or control the heroine "
            "in response to her resistance, escalating conflict and inequality."
        ),
    },
    {
        "id": "R7",
        "name": "Hero and heroine are physically or emotionally separated",
        "phase": "I",
        "phase_name": "Initial Conflict & Isolation",
        "description": (
            "The couple is separated by distance, breakup, avoidance, or mistrust; "
            "they are together in name only or fully apart."
        ),
    },
    # Phase II: Turning Point & Recognition (Developing Empathy)
    {
        "id": "R8",
        "name": "Hero treats heroine tenderly",
        "phase": "II",
        "phase_name": "Turning Point & Recognition",
        "description": (
            "The hero offers care, protection, or tenderness: comforting gestures, "
            "gentle touch, emotional support, or protective behaviour."
        ),
    },
    {
        "id": "R9",
        "name": "Heroine responds warmly to hero's tenderness",
        "phase": "II",
        "phase_name": "Turning Point & Recognition",
        "description": (
            "The heroine softens towards the hero, responding with warmth, gratitude, "
            "laughter, or emotional openness to his care."
        ),
    },
    {
        "id": "R10",
        "name": "Heroine reinterprets hero's behaviour as result of previous hurt",
        "phase": "II",
        "phase_name": "Turning Point & Recognition",
        "description": (
            "The heroine comes to understand the hero's past wounds or trauma and "
            "reframes his earlier harshness as self-protection rather than malice."
        ),
    },
    # Phase III: Commitment & Restoration (The Happy Ending)
    {
        "id": "R11",
        "name": "Hero declares love and demonstrates commitment",
        "phase": "III",
        "phase_name": "Commitment & Restoration",
        "description": (
            "The hero clearly declares his love and/or makes a concrete commitment: "
            "marriage proposal, vow, promise of long-term partnership, or equivalent gesture."
        ),
    },
    {
        "id": "R12",
        "name": "Heroine responds sexually and emotionally",
        "phase": "III",
        "phase_name": "Commitment & Restoration",
        "description": (
            "The heroine reciprocates with emotional and often explicit sexual intimacy: "
            "union scenes that combine erotic fulfilment and emotional acceptance."
        ),
    },
    {
        "id": "R13",
        "name": "Heroine's identity is restored",
        "phase": "III",
        "phase_name": "Commitment & Restoration",
        "description": (
            "The heroine attains a stable, integrated identity: secure relationship, "
            "domestic fulfilment, new parenthood, or reconciled social role."
        ),
    },
    # Special "none" option
    {
        "id": "none",
        "name": "None of the above",
        "phase": "NA",
        "phase_name": "Not a narrative function",
        "description": (
            "The topic does not primarily realise any of Radway's 13 functions. "
            "It covers background, side-plots, minor characters, or thematic material "
            "not central to the heroine–hero narrative arc."
        ),
    },
]

RADWAY_BY_ID: Dict[str, Dict[str, str]] = {fn["id"]: fn for fn in RADWAY_FUNCTIONS}

VALID_RADWAY_IDS = set(RADWAY_BY_ID.keys())


def _radway_block_for_prompt() -> str:
    """Render Radway functions into a compact text block for the system prompt."""
    lines = []
    for fn in RADWAY_FUNCTIONS:
        lines.append(
            f"- {fn['id']} — {fn['name']} "
            f"(Phase {fn['phase']}: {fn['phase_name']}): {fn['description']}"
        )
    return "\n".join(lines)


RADWAY_TEXT_BLOCK = _radway_block_for_prompt()


# ---------------------------------------------------------------------------
# 2. System + user prompts for Radway mapping (topic-level)
#    Reuses Stage 2 taxonomy data as additional signals.
# ---------------------------------------------------------------------------

RADWAY_ZEROSHOT_SYSTEM_PROMPT = f"""
You are RomanceRadwayMapper, an expert in the narrative structure of modern heterosexual romantic fiction.

Your job: assign each BERTopic TOPIC to one of Radway's 13 narrative functions
(or "none of the above") based on how that topic tends to function in the
heroine–hero story arc.

You will be given, for each topic:

- A topic_id

- TOPIC KEYWORDS from BERTopic

- An LLM-generated label and scene_summary from a previous stage

- Stage 1 primary/secondary categories (e.g. "romance_core", "sexual_content")

- Stage 2 taxonomy classification:

  - main taxonomy ID (e.g. 4.4)

  - its human-readable name and group

Use ALL of this information to infer which Radway function this topic most
commonly realises in a typical romance narrative.

RADWAY FUNCTIONS (AVAILABLE LABELS)

Use ONLY these IDs (do NOT invent new ones):

{RADWAY_TEXT_BLOCK}

INTERPRETATION HINTS

- Functions R1–R7 belong to Phase I (setup, conflict, isolation).

- Functions R8–R10 belong to Phase II (turning point, growing empathy).

- Functions R11–R13 belong to Phase III (commitment, restoration, HEA).

- Topics whose Stage 2 taxonomy group is "Relationship Trajectory (Main Couple)"
  and which centre on conflict, misunderstanding, or distance are strong
  candidates for R1–R7.

- Topics whose taxonomy group is "Relationship Trajectory (Main Couple)" and
  express tenderness, comfort, apology, or reinterpretation are candidates for
  R8–R10.

- Topics whose taxonomy group is "Relationship Trajectory (Main Couple)" OR
  "Sexuality, Attraction & Intimacy" and involve explicit commitments,
  reconciliations, or union scenes are candidates for R11–R13.

- Many topics about work, money, social worlds, or background settings will be
  better labelled as "none" unless they clearly participate in one of the
  narrative functions above.

OUTPUT CONSTRAINTS

- Think through the mapping internally.

- In your final answer, output only a valid JSON object.

- Do NOT include markdown, backticks, or any explanation outside the JSON.

- Never wrap JSON in ```json or any other formatting.

JSON SCHEMA (MANDATORY)

Return exactly these keys and types:

{{
  "topic_id": 0,
  "radway_main_id": "R8",
  "radway_secondary_id": "R9",
  "radway_other_plausible_ids": ["R10"],
  "radway_phase": "II",
  "radway_is_none": false,
  "radway_confidence": "medium",
  "radway_rationale": "1–3 short sentences explaining why this Radway function fits this topic."
}}

FIELD RULES

1) "topic_id"

- Echo the integer topic id from the input.

2) "radway_main_id"

- REQUIRED.

- One of: "R1"..."R13" or "none".

- Choose the Radway function that best describes the narrative function this topic usually plays.

- If the topic does not clearly correspond to any of the 13 functions, use "none".

3) "radway_secondary_id"

- OPTIONAL.

- A second Radway function ID when the topic clearly mixes two adjacent functions
  (e.g., R2 antagonism + R3 ambiguous response; R8 tenderness + R9 warm response).

- Use null when there is no meaningful second function.

4) "radway_other_plausible_ids"

- OPTIONAL list (0–3 items) of other Radway IDs that are plausible but clearly
  less central than radway_main_id and radway_secondary_id.

5) "radway_phase"

- REQUIRED.

- Must be "I", "II", "III", or "NA".

- For R1–R7 use "I"; R8–R10 use "II"; R11–R13 use "III"; "none" uses "NA".

6) "radway_is_none"

- REQUIRED boolean.

- true iff radway_main_id == "none".

- If true, radway_secondary_id MUST be null and radway_other_plausible_ids MUST be [].

7) "radway_confidence"

- REQUIRED.

- One of: "low", "medium", "high".

8) "radway_rationale"

- 1–3 short sentences.

- Refer to:

  - topic keywords,

  - label and scene_summary,

  - Stage 2 taxonomy group and category,

  - and any relevant representative snippets.

- Explain why this topic fits the chosen Radway function (or why "none" is appropriate).

- Do NOT quote long snippets verbatim; summarise them instead.

NO REASONING OUTSIDE JSON

- You may think step-by-step internally.

- The final answer must be only the JSON object described above.
""".strip()


RADWAY_ZEROSHOT_USER_PROMPT = """
### TOPIC DATA

topic_id: {topic_id}

TOPIC KEYWORDS (most important first):

{keywords}

PREVIOUS LLM LABEL (Stage 8):

{label}

PREVIOUS SCENE SUMMARY (Stage 8):

{scene_summary}

STAGE 1 CATEGORY SIGNALS:

- primary_categories: {primary_categories}

- secondary_categories: {secondary_categories}

STAGE 2 THEORY-DRIVEN TAXONOMY:

- main_category: {taxonomy_main_id} — {taxonomy_main_name} ({taxonomy_main_group})

- secondary_category: {taxonomy_secondary_id} — {taxonomy_secondary_name} ({taxonomy_secondary_group})

REPRESENTATIVE SNIPPETS (optional):

{snippets}

### TASK

Using ONLY the information above and the Radway functions defined in the system message:

- Decide whether this topic typically enacts one of Radway's 13 narrative functions
  in the heroine–hero romance arc, or "none of the above".

- If meaningful, choose:

  - one radway_main_id (required),

  - an optional radway_secondary_id,

  - optional radway_other_plausible_ids (0–3),

  - and the correct radway_phase.

- If none, set radway_main_id = "none", radway_phase = "NA",
  radway_secondary_id = null, radway_other_plausible_ids = [] and radway_is_none = true.

Return a SINGLE JSON object following the schema in the system message.

Do NOT include explanations outside the JSON.
""".strip()


# ---------------------------------------------------------------------------
# 3. Helper: Load taxonomy + source metadata (from JSON or model)
# ---------------------------------------------------------------------------

def load_topics_with_taxonomy_from_json(taxonomy_json_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Load Stage 2 taxonomy JSON (like taxonomy_mappings_*.json)
    and keep the full structure for each topic.

    This is the single source of truth that contains:
    - Taxonomy mappings (main_category_id, secondary_category_id, etc.)
    - Source metadata (label, keywords, scene_summary, primary/secondary categories)

    Parameters
    ----------
    taxonomy_json_path:
        Path to taxonomy mappings JSON file.

    Returns
    -------
    Dict[int, Dict[str, Any]] mapping topic_id → full topic object with all fields.
    """
    with open(taxonomy_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    topics: Dict[int, Dict[str, Any]] = {}
    for k, v in raw.items():
        try:
            tid = int(k)
        except ValueError:
            continue
        topics[tid] = v

    return topics


def load_topics_with_taxonomy_from_model(model_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Load taxonomy + source metadata from BERTopic model's topic_metadata_ attribute.
    
    This follows the recommendation from MODEL_COMPARISON_REPORT.md to use
    models with embedded taxonomy metadata (e.g., model_1_with_llm_labels_and_metadata_disambiguated.pkl).
    
    The model's topic_metadata_ should contain both:
    - Taxonomy mappings (main_category_id, secondary_category_id, etc.)
    - Source metadata (label, keywords, scene_summary, primary/secondary categories)
    
    Parameters
    ----------
    model_path:
        Path to BERTopic model (.pkl file or directory).
    
    Returns
    -------
    Dict[int, Dict[str, Any]] mapping topic_id → full topic object with all fields.
    
    Raises
    ------
    ValueError:
        If model doesn't have topic_metadata_ attribute or it's empty.
    """
    import pickle
    
    # Load model (handle both pickle wrapper and native format)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    if model_path.suffix == ".pkl":
        with open(model_path, "rb") as f:
            loaded_obj = pickle.load(f)
        
        # Check if it's a RetrainableBERTopicModel wrapper
        if hasattr(loaded_obj, "trained_topic_model") and loaded_obj.trained_topic_model is not None:
            model = loaded_obj.trained_topic_model
        elif isinstance(loaded_obj, BERTopic):
            model = loaded_obj
        else:
            model = BERTopic.load(str(model_path))
    else:
        model = BERTopic.load(str(model_path))
    
    # Extract taxonomy + source metadata
    # Models can have taxonomy in either topic_metadata_ (merged) or topic_taxonomy_ (separate)
    # We need to merge both sources if available
    
    has_metadata = hasattr(model, "topic_metadata_") and model.topic_metadata_
    has_taxonomy = hasattr(model, "topic_taxonomy_") and model.topic_taxonomy_
    
    if not has_metadata and not has_taxonomy:
        raise ValueError(
            f"Model at {model_path} does not have topic_metadata_ or topic_taxonomy_ attributes. "
            "Use a model with embedded taxonomy mappings (e.g., model_1_with_taxonomy_mappings or "
            "model_1_with_llm_labels_and_metadata_disambiguated.pkl)"
        )
    
    # Start with topic_metadata_ if available (may already have taxonomy merged)
    if has_metadata:
        metadata = model.topic_metadata_
    else:
        metadata = {}
    
    # Merge taxonomy from topic_taxonomy_ if it exists separately
    if has_taxonomy:
        taxonomy = model.topic_taxonomy_
        LOGGER.info("Found taxonomy in topic_taxonomy_ attribute, merging with topic_metadata_")
        for tid, tax_data in taxonomy.items():
            tid_int = int(tid) if isinstance(tid, str) else tid
            if tid_int not in metadata:
                metadata[tid_int] = {}
            # Merge taxonomy fields into metadata
            metadata[tid_int].update(tax_data)
    
    # Convert to dict with int keys
    topics: Dict[int, Dict[str, Any]] = {}
    for topic_id, topic_data in metadata.items():
        # Handle both int and str keys
        tid = int(topic_id) if isinstance(topic_id, str) else topic_id
        topics[tid] = dict(topic_data)  # Make a copy
    
    LOGGER.info("Loaded taxonomy + source metadata for %d topics from model", len(topics))
    return topics


def load_topics_with_taxonomy(input_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Load taxonomy + source metadata from JSON file or BERTopic model.
    
    Automatically detects the source type:
    - If path ends with .json, loads from JSON file
    - If path ends with .pkl or is a directory, loads from model's topic_metadata_
    
    This is the single source of truth that contains:
    - Taxonomy mappings (main_category_id, secondary_category_id, etc.)
    - Source metadata (label, keywords, scene_summary, primary/secondary categories)
    
    Parameters
    ----------
    input_path:
        Path to taxonomy mappings JSON file or BERTopic model.
    
    Returns
    -------
    Dict[int, Dict[str, Any]] mapping topic_id → full topic object with all fields.
    """
    if input_path.suffix == ".json":
        return load_topics_with_taxonomy_from_json(input_path)
    else:
        return load_topics_with_taxonomy_from_model(input_path)


def _format_taxonomy_for_prompt(topic_entry: Dict[str, Any]) -> Dict[str, str]:
    """
    Turn Stage 2 taxonomy mapping for a topic into strings for the user prompt.

    Parameters
    ----------
    topic_entry:
        Full topic entry from taxonomy JSON.

    Returns
    -------
    Dict with formatted taxonomy strings for prompt.
    """
    main_id = topic_entry.get("main_category_id")
    sec_id = topic_entry.get("secondary_category_id")

    main_node = TAXONOMY_BY_ID.get(main_id, {}) if main_id else {}
    sec_node = TAXONOMY_BY_ID.get(sec_id, {}) if sec_id else {}

    return {
        "taxonomy_main_id": str(main_id) if main_id is not None else "(none)",
        "taxonomy_main_name": main_node.get("name", "(unknown)"),
        "taxonomy_main_group": main_node.get("group", "(unknown)"),
        "taxonomy_secondary_id": str(sec_id) if sec_id is not None else "(none)",
        "taxonomy_secondary_name": sec_node.get("name", "(none)"),
        "taxonomy_secondary_group": sec_node.get("group", "(none)"),
    }


# ---------------------------------------------------------------------------
# 4. Core function: classify a single topic into a Radway function
# ---------------------------------------------------------------------------

def classify_topic_to_radway_openrouter(
    *,
    topic_id: int,
    topic_entry: Dict[str, Any],
    client: OpenAI,
    model_name: str,
    temperature: float = 0.25,
    max_new_tokens: int = 220,
    representative_docs: Optional[List[str]] = None,
    max_snippets: int = 8,
    max_chars_per_snippet: int = 400,
) -> Dict[str, Any]:
    """
    Classify a single BERTopic topic into Radway's 13 narrative functions
    using Mistral-Nemo via OpenRouter.

    Parameters
    ----------
    topic_id:
        Integer topic id.
    topic_entry:
        Full topic entry from taxonomy JSON (contains taxonomy fields + source_metadata).
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
        "topic_id",
        "radway_main_id",
        "radway_secondary_id",
        "radway_other_plausible_ids",
        "radway_phase",
        "radway_is_none",
        "radway_confidence",
        "radway_rationale",
        "radway_main_name",
        "radway_phase_name"
    """
    # Extract source metadata (from Stage 8, embedded in taxonomy JSON)
    source_metadata = topic_entry.get("source_metadata", {})
    keywords = source_metadata.get("keywords", [])
    label = source_metadata.get("label", "")
    scene_summary = source_metadata.get("scene_summary", "")
    primary_categories = source_metadata.get("primary_categories", [])
    secondary_categories = source_metadata.get("secondary_categories", [])

    # Fallback to top-level fields if source_metadata is missing (for backwards compatibility)
    if not keywords:
        keywords = topic_entry.get("keywords", [])
    if not label:
        label = topic_entry.get("label", "")
    if not scene_summary:
        scene_summary = topic_entry.get("scene_summary", "")

    kw_str = ", ".join(keywords) if keywords else "(no keywords)"
    primary_str = ", ".join(primary_categories) if primary_categories else "(none)"
    secondary_str = ", ".join(secondary_categories) if secondary_categories else "(none)"

    tax_strings = _format_taxonomy_for_prompt(topic_entry)

    # Format representative snippets
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

    user_prompt = RADWAY_ZEROSHOT_USER_PROMPT.format(
        topic_id=topic_id,
        keywords=kw_str,
        label=label or "(no label)",
        scene_summary=scene_summary or "(no scene summary)",
        primary_categories=primary_str,
        secondary_categories=secondary_str,
        snippets=snippets_block,
        **tax_strings,
    )

    messages = [
        {"role": "system", "content": RADWAY_ZEROSHOT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    LOGGER.info("Classifying topic %d into Radway function (Mistral-Nemo)...", topic_id)

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
        raise ValueError("Empty API response for Radway classification")

    content = response.choices[0].message.content.strip()
    LOGGER.debug("Raw Radway response for topic %d: %s", topic_id, content[:300])

    # Extract JSON (strip optional code fences defensively)
    json_content = content
    if "```json" in json_content:
        json_content = json_content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in json_content:
        json_content = json_content.split("```", 1)[1].split("```", 1)[0].strip()

    # Fallback: try to grab the first {...} span
    if not json_content.strip().startswith("{"):
        first_brace = json_content.find("{")
        last_brace = json_content.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_content = json_content[first_brace : last_brace + 1]

    try:
        result = json.loads(json_content)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse Radway JSON for topic {topic_id}: {e}\nContent:\n{content}"
        ) from e

    # --- sanity checks + normalisation -----------------------------------
    result_topic_id = result.get("topic_id", topic_id)
    if result_topic_id != topic_id:
        LOGGER.warning(
            "Model echoed different topic_id (%s) than input (%s); overriding with input.",
            result_topic_id,
            topic_id,
        )
        result["topic_id"] = topic_id

    main_id = result.get("radway_main_id")
    sec_id = result.get("radway_secondary_id", None)
    is_none = bool(result.get("radway_is_none", False))

    # Normalise main id
    if main_id not in VALID_RADWAY_IDS:
        # If model wrote something weird, fall back to "none"
        main_id = "none"
        result["radway_main_id"] = "none"

    # Fix radway_is_none consistency
    is_none = (main_id == "none")
    result["radway_is_none"] = is_none

    if is_none:
        # force secondary + others empty
        result["radway_secondary_id"] = None
        result["radway_other_plausible_ids"] = []
        result["radway_phase"] = "NA"

    # Validate secondary id
    if sec_id is not None and sec_id not in VALID_RADWAY_IDS:
        result["radway_secondary_id"] = None

    # Normalise other_plausible_ids
    other_ids = result.get("radway_other_plausible_ids", [])
    if not isinstance(other_ids, list):
        other_ids = []
    filtered_other = []
    for rid in other_ids:
        if (
            isinstance(rid, str)
            and rid in VALID_RADWAY_IDS
            and rid not in {main_id, result.get("radway_secondary_id")}
            and rid != "none"
        ):
            filtered_other.append(rid)
    result["radway_other_plausible_ids"] = filtered_other

    # Normalise phase from id (override model if inconsistent)
    fn_info = RADWAY_BY_ID.get(main_id, RADWAY_BY_ID["none"])
    result["radway_phase"] = fn_info["phase"]

    # Normalise confidence
    conf = result.get("radway_confidence", None)
    valid_conf = {"low", "medium", "high"}
    if not isinstance(conf, str) or conf.lower() not in valid_conf:
        # Quick heuristic: if main_id is "none", treat as medium; else low.
        conf = "medium" if main_id == "none" else "low"
    else:
        conf = conf.lower()
    result["radway_confidence"] = conf

    # Attach human-readable name + phase name for convenience
    result["radway_main_name"] = fn_info["name"]
    result["radway_phase_name"] = fn_info["phase_name"]

    LOGGER.info(
        "Topic %d → Radway main=%s (%s, phase %s), none=%s, confidence=%s",
        topic_id,
        result["radway_main_id"],
        result["radway_main_name"],
        result["radway_phase"],
        result["radway_is_none"],
        result["radway_confidence"],
    )

    return result


# ---------------------------------------------------------------------------
# 5. Helpers: Load snippets from BERTopic model (reuse from Stage 2)
# ---------------------------------------------------------------------------

def load_bertopic_model_for_snippets(
    *,
    base_dir: Path = DEFAULT_BASE_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    model_suffix: str = "_with_taxonomy_mappings",
    stage_subfolder: Optional[str] = "stage09_category_mapping",
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
        Model suffix (default: "_with_taxonomy_mappings" - can also use "_with_llm_labels_and_metadata_disambiguated").
    stage_subfolder:
        Optional stage subfolder (default: "stage09_category_mapping").
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

        # Construct path with stage subfolder
        if stage_subfolder:
            source_base_dir = base_dir / embedding_model / stage_subfolder
        else:
            source_base_dir = base_dir / embedding_model

        topic_model = load_native_bertopic_model(
            base_dir=source_base_dir,
            embedding_model=".",  # Use "." to avoid path duplication
            pareto_rank=1,
            model_suffix=model_suffix,
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
            "Radway classification will proceed without representative snippets.",
            e,
        )
        return None


# ---------------------------------------------------------------------------
# 6. Batch mapping: map all topics to Radway functions
# ---------------------------------------------------------------------------

def map_all_topics_to_radway(
    *,
    taxonomy_json_path: Path,
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
    model_suffix: str = "_with_taxonomy_mappings",
    stage_subfolder: Optional[str] = "stage09_category_mapping",
    max_docs_per_topic: int = 10,
    limit_topics: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Run zero-shot Radway function mapping for all topics with taxonomy mappings.

    Parameters
    ----------
    taxonomy_json_path:
        Path to Stage 2 taxonomy mappings JSON file OR BERTopic model with embedded taxonomy metadata.
        If JSON: loads from file (e.g., taxonomy_mappings_*.json).
        If model (.pkl or directory): loads from model's topic_metadata_ attribute (recommended).
        The recommended model is: model_1_with_llm_labels_and_metadata_disambiguated.pkl
    output_path:
        Path to write merged JSON (taxonomy + Radway mappings).
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
    load_model_for_snippets:
        If True and topic_to_snippets is None, load BERTopic model and extract snippets.
    base_dir, embedding_model, model_suffix, stage_subfolder:
        Parameters for loading BERTopic model (used if load_model_for_snippets=True).
    max_docs_per_topic:
        Maximum number of representative docs to extract per topic.
    limit_topics:
        If not None, only process the first N topics (for quick testing).

    Returns
    -------
    Dict[int, Dict[str, Any]] mapping topic_id → Radway mapping (nested under "radway_functions").
    """
    # Initialize client if needed
    if client is None:
        client, _ = load_openrouter_client(
            api_key=api_key or "",
            model_name=model_name,
        )

    # Load taxonomy + source metadata (from JSON or model)
    topics = load_topics_with_taxonomy(taxonomy_json_path)
    topic_ids = sorted(topics.keys())
    total = len(topic_ids)

    input_type = "model" if taxonomy_json_path.suffix != ".json" else "JSON"
    LOGGER.info("Loaded taxonomy + source metadata for %d topics from %s (%s)", total, taxonomy_json_path, input_type)

    if limit_topics is not None and limit_topics > 0:
        topic_ids = topic_ids[:limit_topics]
        total = len(topic_ids)
        LOGGER.info("Limiting Radway mapping to first %d topics", total)

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
                "Using representative snippets from BERTopic model for Radway classification"
            )
        else:
            LOGGER.info("Proceeding without representative snippets")
    elif topic_to_snippets is None:
        LOGGER.info("No representative snippets provided, using keywords and labels only")

    # Process each topic
    radway_results: Dict[int, Dict[str, Any]] = {}

    for idx, tid in enumerate(topic_ids, start=1):
        topic_entry = topics[tid]
        snippets = topic_to_snippets.get(tid, []) if topic_to_snippets else None

        result = classify_topic_to_radway_openrouter(
            topic_id=tid,
            topic_entry=topic_entry,
            client=client,
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            representative_docs=snippets,
            max_snippets=8,
            max_chars_per_snippet=400,
        )

        radway_results[tid] = result

        if idx % 10 == 0 or idx == total:
            LOGGER.info(
                "Processed %d/%d topics for Radway mapping (%.1f%%)",
                idx,
                total,
                idx / total * 100.0,
            )

    # Merge Radway results into taxonomy data
    # Reload original data to preserve exact structure
    if taxonomy_json_path.suffix == ".json":
        # Load from JSON
        with open(taxonomy_json_path, "r", encoding="utf-8") as f:
            merged_data = json.load(f)
    else:
        # Load from model again (to get fresh copy)
        merged_data_raw = load_topics_with_taxonomy_from_model(taxonomy_json_path)
        # Convert to string-keyed dict for JSON serialization
        merged_data = {str(k): v for k, v in merged_data_raw.items()}

    # Add radway_functions to each topic
    for tid, radway_obj in radway_results.items():
        key = str(tid)
        if key in merged_data:
            merged_data[key]["radway_functions"] = radway_obj
        else:
            LOGGER.warning("Topic %d in Radway results but not in taxonomy data", tid)

    # Save merged JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    LOGGER.info(
        "Saved merged taxonomy + Radway mappings for %d topics to %s",
        len(radway_results),
        output_path,
    )

    return radway_results


# ---------------------------------------------------------------------------
# 7. Optional: Model attachment function
# ---------------------------------------------------------------------------

def update_model_with_radway_mappings(
    *,
    merged_json_path: Path,
    base_dir: Path = DEFAULT_BASE_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    model_suffix: str = "_with_taxonomy_mappings",
    source_stage_subfolder: str = "stage09_category_mapping",
    target_stage_subfolder: str = "stage09_category_mapping",
    target_model_suffix: str = "_with_radway_mappings",
) -> Path:
    """
    Load BERTopic model, attach Radway mappings, and save to new location.

    Parameters
    ----------
    merged_json_path:
        Path to merged taxonomy + Radway JSON file.
    base_dir:
        Base directory for models.
    embedding_model:
        Embedding model name.
    model_suffix:
        Suffix of source model to load (default: "_with_taxonomy_mappings" - can also use "_with_llm_labels_and_metadata_disambiguated").
    source_stage_subfolder:
        Stage subfolder where source model is located.
    target_stage_subfolder:
        Stage subfolder where updated model will be saved.
    target_model_suffix:
        Suffix for the updated model.

    Returns
    -------
    Path to the saved model directory.
    """
    LOGGER.info("Loading merged taxonomy + Radway mappings from %s", merged_json_path)
    with open(merged_json_path, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    # Extract Radway mappings
    radway_map: Dict[int, Dict[str, Any]] = {}
    for k, v in merged_data.items():
        try:
            tid = int(k)
            if "radway_functions" in v:
                radway_map[tid] = v["radway_functions"]
        except ValueError:
            continue

    LOGGER.info("Loaded Radway mappings for %d topics", len(radway_map))

    # Load source model
    LOGGER.info("Loading source BERTopic model...")
    LOGGER.info("  Base dir: %s", base_dir)
    LOGGER.info("  Embedding model: %s", embedding_model)
    LOGGER.info("  Model suffix: %s", model_suffix)
    LOGGER.info("  Stage subfolder: %s", source_stage_subfolder)

    source_base_dir = base_dir / embedding_model / source_stage_subfolder

    topic_model = load_native_bertopic_model(
        base_dir=source_base_dir,
        embedding_model=".",  # Use "." to avoid path duplication
        pareto_rank=1,
        model_suffix=model_suffix,
    )

    LOGGER.info("✓ Source model loaded successfully")

    # Attach Radway mappings to model
    LOGGER.info("Attaching Radway mappings to model...")
    topic_model.topic_radway_ = radway_map
    LOGGER.info("✓ Radway mappings attached to model.topic_radway_")

    # Also merge into topic_metadata_ if it exists (recommended approach)
    if hasattr(topic_model, "topic_metadata_") and topic_model.topic_metadata_:
        LOGGER.info("Merging Radway mappings into topic_metadata_...")
        for tid, radway_data in radway_map.items():
            if tid in topic_model.topic_metadata_:
                topic_model.topic_metadata_[tid]["radway_functions"] = radway_data
        LOGGER.info("✓ Radway mappings merged into topic_metadata_ for %d topics", len(radway_map))

    # Verify attachment
    if hasattr(topic_model, "topic_radway_") and topic_model.topic_radway_:
        LOGGER.info(
            "✓ Verified: Radway mappings attached for %d topics",
            len(topic_model.topic_radway_),
        )
        # Log sample
        sample_topic = list(radway_map.keys())[0]
        sample_data = radway_map[sample_topic]
        LOGGER.info(
            "  Sample Radway keys for topic %d: %s",
            sample_topic,
            list(sample_data.keys())[:10],  # First 10 keys
        )
    else:
        LOGGER.warning("⚠ Radway mappings may not have been attached correctly")

    # Save model to target location
    target_dir = base_dir / embedding_model / target_stage_subfolder
    target_dir.mkdir(parents=True, exist_ok=True)

    model_dir = target_dir / f"model_1{target_model_suffix}"

    # Remove existing directory if it exists
    if model_dir.exists() and model_dir.is_dir():
        LOGGER.info("Removing existing model directory: %s", model_dir)
        shutil.rmtree(model_dir)

    LOGGER.info("Saving updated model to %s", model_dir)
    topic_model.save(str(model_dir))
    LOGGER.info("✓ Model saved successfully to %s", model_dir)

    return model_dir


# ---------------------------------------------------------------------------
# 8. Command-line interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 3: Zero-shot Radway function mapping of BERTopic topics using Mistral via OpenRouter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--taxonomy-json",
        type=Path,
        required=True,
        help=(
            "Path to Stage 2 taxonomy mappings JSON file OR BERTopic model with embedded taxonomy metadata. "
            "If JSON: loads from file (e.g., taxonomy_mappings_*.json). "
            "If model (.pkl or directory): loads from model's topic_metadata_ attribute (recommended). "
            "Recommended model: model_1_with_llm_labels_and_metadata_disambiguated.pkl"
        ),
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Where to save merged taxonomy + Radway mappings JSON.",
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
        "--limit-topics",
        type=int,
        default=None,
        help="Limit processing to first N topics (for testing).",
    )

    parser.add_argument(
        "--no-snippets",
        action="store_true",
        help="Skip loading BERTopic model and extracting representative snippets.",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    client, _ = load_openrouter_client(
        api_key=args.api_key or "",
        model_name=args.model_name,
    )

    map_all_topics_to_radway(
        taxonomy_json_path=args.taxonomy_json,
        output_path=args.output_json,
        client=client,
        model_name=args.model_name,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        load_model_for_snippets=not args.no_snippets,
        limit_topics=args.limit_topics,
    )

