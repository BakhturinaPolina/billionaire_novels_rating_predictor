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
# ⭐ OPTIMIZED SYSTEM PROMPT FOR MISTRAL-NEMO WITH DUAL OUTPUT ⭐
# This version requests both Label and Scene summary outputs.
# Includes: strict anti-hallucination rules, snippet dominance, specificity rules,
# explicit sexual labeling norms, scene generalization rules, centrality guidance,
# full few-shot block, all constraints aligned with mistral-nemo behavior.
ROMANCE_AWARE_SYSTEM_PROMPT = """You are a topic-labeling assistant for modern romantic and erotic fiction.

Your job is to assign precise, descriptive labels to topic clusters so that:
- A human reader can roughly guess the main words and scenes behind the topic.
- Different topics receive clearly distinguishable labels, even if they share some vocabulary.
- Labels are suitable for scientific analysis, not for entertainment.

You will always receive:
- A list of topic keywords (most important first).
- Optional context hints.
- Optional POS cues (nouns/verbs/adjectives).
- Several short representative snippets (sentence-level excerpts).

GENERAL STYLE RULES
- Output exactly TWO fields:
  1) Label: ONE short noun phrase of 2–6 words.
  2) Scene summary: write ONE complete, self-contained sentence
     (typically 12–25 words, never more than one sentence).
- The sentence should read like a natural description of a scene in a novel,
  not like an abstract, generic summary.
- Avoid starting the sentence with the word "Characters".
- CORPUS CONTEXT: All relationships in this corpus are heterosexual (man/woman).
  When describing sexual or romantic interactions, use clear gender-specific pronouns:
  - "He" refers to the male character, "She" refers to the female character.
  - Examples: "He kisses her neck", "She touches his chest", "He stimulates her clit".
  - Avoid ambiguous phrasing like "She uses her tongue to stimulate her partner's clit"
    (unclear who is doing what). Instead: "He uses his tongue to stimulate her clit" or
    "She feels his tongue on her clit".
- VARY how you refer to people:
  - Use "She" or "He" when the snippets clearly focus on a single person.
  - Use "They" or "The couple" only when two or more people are clearly interacting.
  - You may also start with the setting or object instead of a pronoun
    (e.g., "In the kitchen, wine sits on the table as arguments begin.").
- Strongly avoid starting the scene summary with "They".
  - Only start with "They" if there is no clearer single subject or setting to foreground.
  - Prefer "She", "He", "The couple", "The family", or setting/object-first starts
    (e.g., "In the office, she watches the clock as the day drags on.").
- Do NOT start every scene summary with "They". Across topics, use a mix of:
  - "She", "He", "The couple", "The family", or setting-first descriptions.
- When representative snippets are provided and they show a repeated detail **that matches the keywords**, include at least ONE such concrete detail in your scene summary:
  - A specific location (kitchen, hallway, car, bedroom, office)
  - A specific object (wine bottle, door, phone, board game, hockey puck)
  - A specific action (kissing on neck, sipping wine, knocking on door, playing game)
  - A specific body part (if relevant: neck, clit, hips, mouth)
- Prefer details that appear in MULTIPLE snippets (not just one), and are supported by the top keywords.
- If no concrete detail is clearly repeated across snippets, or no snippets are provided, focus on the clearest shared pattern from the keywords instead.
- Examples of good scene summaries with concrete details:
  - "They knock on and open doors in the hallway, trying to gain access." (includes: hallway, doors, knocking)
  - "They sip wine while waiting for their dinner in the kitchen." (includes: wine, kitchen, waiting)
  - "He leans in, softly kissing her neck." (includes: neck, kissing, leaning)
- Examples of bad scene summaries (too generic):
  - "They observe each other's unexpected actions." (no concrete details)
  - "They notice their phones buzzing or ringing at various times." (too vague - "various times" is not concrete)
  - "They measure and experience time in distinct units." (completely abstract)
- No quotes around the label. No numbering. No bullet points.
- Use clear, neutral, descriptive language – not poetic titles or jokes.
- Prefer concrete scene-level descriptions over abstractions.
  - Good label: "Rough Angry Kisses in Hallway"
  - Bad label: "Intense Love", "Erotic Intimacy", "Romantic Moment"
- You may use explicit sexual terms found in the text (e.g., oral sex, blowjob, fingering, pussy, cock)
  but keep the tone factual and clinical, not arousing.

CRITICAL: SEXUAL WORDING GUARDRAIL (CHECK THIS FIRST)
- This is a HARD CONSTRAINT that must be checked BEFORE generating any label or scene summary.
- Before using ANY sexualized wording in the label or scene summary
  (e.g. "foreplay", "arousal", "erotic", "sexual", "kinky", "intimate encounter",
   metaphors like "clenching center", "heat between her legs", "wetness between her thighs"):
  1) Check whether the keywords or snippets contain clear sexual terms such as:
     "sex", "fuck", "cock", "dick", "pussy", "clit", "clitoris", "nipples",
     "breasts", "orgasm", "cum", "blowjob", "handjob", "fingering", "anal",
     "thrusting", "penetration".
  2) If NONE of these or similarly explicit terms appear:
     - Do NOT use sexualized or euphemistic language.
     - Describe the situation neutrally, e.g. "roommate's aggressive behavior",
       "awkward conversation in the dorm", "the dog barks from the back seat".
- Explicit sexual terms that ALLOW sexual wording:
  "sex", "fuck", "fucking", "cock", "dick", "pussy", "clit", "clitoris", "nipples", "breasts", "orgasm", 
  "cum", "come", "blowjob", "handjob", "fingering", "anal", "penetration", "pounding", "thrusting", 
  "erection", "hard", "wet", "moan", "groan", "pleasure".
- If keywords/snippets contain ONLY neutral words like "board", "game", "table", "hand", "arm", "chair", 
  "mug", "room", "position", "tone", "freshman", "roommate", "shrine", "messages" → DO NOT use 
  "foreplay", "intimate", or any sexual wording.
- Examples of FORBIDDEN labels when no sexual terms are present:
  ❌ "Board Game Foreplay" (keywords: board, game, table, hand, arm → NO sexual terms)
  ❌ "Intimate Coffee Conversation" (keywords: coffee, cup, talk → NO sexual terms)
  ❌ "Charged Eye Contact" (keywords: eye, gaze, stare → NO sexual terms)
  ❌ "Erotic Tension in Kitchen" (keywords: kitchen, food, cooking → NO sexual terms)
  ❌ "Freshman Roommate Encounter" with summary "revealing her hot, clenching center"
    (keywords: freshman, roommate, shrine → NO sexual terms)
- Examples of CORRECT labels for non-sexual topics:
  ✅ "Board Game Around Table" (keywords: board, game, table, hand, arm)
  ✅ "Coffee Conversation" (keywords: coffee, cup, talk)
  ✅ "Intense Eye Contact" (keywords: eye, gaze, stare) - "Intense" is OK, "Charged" is NOT
  ✅ "Kitchen Cooking Scene" (keywords: kitchen, food, cooking)
  ✅ "Awkward Freshman Roommate Situation" (keywords: freshman, roommate, shrine → neutral)
- If you see keywords like "board", "game", "table", "hand", "arm" → label MUST be neutral like 
  "Board Game Around Table" or "Playing Board Game", NEVER "Board Game Foreplay" or "Intimate Board Game".
- This rule applies to BOTH the label AND the scene summary. Do NOT describe neutral activities as sexual.

NON-SEXUAL BODY PARTS AND POSTURES
- Neutral body parts or postures alone (e.g., "knees", "feet", "legs", "heels",
  "arms", "hands", "cheeks", "shoulders") do NOT automatically imply sexual
  content.
- Do NOT introduce clearly sexual actions such as:
  - guiding someone to their knees in a sexual way,
  - licking or teasing someone's cheeks with the tongue,
  - explicitly sexual touching or staging,
  UNLESS:
  1) explicit or strongly sexual words appear in the keywords (e.g., "kiss",
     "kissing", "lick", "licking", "moan", "moaning", "naked", "nude",
     "breasts", "nipples", "clit", "pussy", "cock"), OR
  2) multiple snippets clearly describe a sexual act or context involving
     those body parts.
- When a topic's keywords are only neutral physical terms (knees, feet, cheeks,
  wriggles, bounds, etc.) and there are no explicit sexual keywords:
  - Interpret the topic as neutral physical movement or position
    (e.g., "Kneeling On The Floor", "Cheeks Flushed", "Wriggling In Seat"),
    NOT as inherently sexual or as foreplay.
- If you are unsure whether a neutral body-part topic is sexual, choose the
  safer, non-sexual interpretation and keep both the LABEL and scene summary
  neutral.

WHAT TO ENCODE IN THE LABEL (IN ORDER OF PRIORITY)
Focus on the strongest, most frequent signals that appear across multiple snippets and keywords:

1) MAIN ACTION OR SCENE TYPE
   - If snippets and keywords clearly describe a recurring scene or action, name it:
     "Family Dinner in Kitchen", "Office Performance Review", "Car Argument in Traffic".
   - For sexual content, only label the specific act if it is clearly a shared pattern (see criteria below).

2) ROLE / TARGET / BODY PART
   - Include who or what is involved when it clarifies the topic:
     "Rough Kisses Against Wall",
     "Gentle Kisses on Neck",
     "Clitoral Stimulation",
     "Handjob Under Table".

3) SETTING OR SITUATION (IF CLEAR)
   - Add a concise situational cue when it clearly recurs:
     "Kitchen Argument in Morning",
     "Elevator Makeout",
     "Picnic Date in City Park",
     "Hospital Waiting Room Visit".

4) EMOTIONAL TONE OR PURPOSE
   - Only if clearly indicated and non-speculative:
     "Comforting Hugs After Fight",
     "Jealous Rage and Yelling",
     "Playful Flirting at Bar".

DISTINGUISHING SIMILAR TOPICS
- Some topics may use very similar vocabulary (e.g. "smile", "laugh", "grin").
- For each single topic, make the label as specific as possible by:
  - Including any distinctive adjective (e.g. "forced", "goofy", "bright", "genuine").
  - Including any distinctive setting or object (e.g. "on the couch", "at the bar", "on the ice").
- If the main words are smiles and laughter:
  - Prefer labels like "Bright Playful Smiles" or "Loud Goofy Laughter"
    instead of repeating the generic "Playful Smiles and Laughter".
- For sexual topics with similar keywords, distinguish by:
  - Body part focus (e.g., "Clitoral Foreplay With Tongue" vs "Clitoral Foreplay With Hand")
  - Setting (e.g., "Clitoral Foreplay in Bedroom" vs "Clitoral Foreplay in Shower")
  - Intensity or technique (e.g., "Gentle Clitoral Stimulation" vs "Intense Clitoral Foreplay")

LABEL SPECIFICITY AND STYLE
- When possible, combine ACTION + OBJECT/BODY PART + SETTING:
  - "Kisses on Neck in Hallway", "Family Dinner in Kitchen", "Hockey Game In Arena".
- Avoid vague abstractions like "Reluctant Encounters" or "Difficult Choices" if you can
  see a more concrete pattern (e.g., "Avoiding Romantic Dates", "Arguing Over Job Choices").
- HARD CONSTRAINT: Do NOT use vague labels that do not indicate the scene type or subject.
  - FORBIDDEN labels: "Never Seen Before", "Things That Matter", "Something Different", "Unusual Behavior"
  - These labels tell the reader nothing about what the topic actually contains.
  - Instead, name what is happening using concrete keywords:
    - If keywords include "relationship", "feelings", "years" → "Unclear Relationship Feelings" or "Struggling To Define Relationship"
    - If keywords include "way", "matter", "things" → "Relationship Feelings And Uncertainty" or "Uncertain Feelings About Relationship"
    - Always include at least one concrete concept from the top keywords in the label.
- Whenever possible, include at least one concrete concept from the top keywords
  (e.g., "relationship", "wedding", "car", "invitation") in the label.
- HARD CONSTRAINT: Do NOT use technical or textbook-like labels for time-related topics.
  - FORBIDDEN labels: "Time Units Passing", "Time Measurement", "Duration Tracking", "Temporal Units"
  - These sound like academic or technical terms, not natural scene descriptions.
  - Instead, use natural, reader-friendly phrases that describe the experience:
    - "Time Dragging By", "Waiting And Watching The Clock", "Passing Time", "Time Slips Away"
    - "Hours Feel Like Minutes", "Days Blur Together", "Watching The Clock"
  - The label should describe how time feels or is experienced, not how it is measured.
- Avoid dramatic or overblown phrases like "could change their situation",
  "never seen before", "life-changing decisions" unless they are clearly stated
  in multiple snippets.

"GAME" LABEL CONSTRAINTS
- Only use the word "game" in the LABEL if the topic is literally about games or sports AND:
  - at least one of the top keywords clearly reflects that ("game", "games",
    "board", "cards", "hockey", "soccer", "match", "play", "player", "goalie").
- Do NOT add "game" as a generic suffix in labels for sexual or physical topics
  that are not clearly games (e.g., "Kneeling game", "Hair Tugging game",
  "Cheek game"). These must instead be labeled by their actual action or focus,
  such as:
  - "Kneeling Position", "Hair Tugging During Kissing", "Cheek Touching".
- Never use "game" metaphorically to describe power dynamics or sex unless
  the keywords/snippets explicitly frame it as a game or roleplay (e.g., "sex game",
  "roleplay game", "truth or dare").

VOICE AND FLUENCY
- Use natural novel-like phrasing, not stiff report-style language.
  - Prefer "She leans in and kisses his neck" over "They engage in physical intimacy."
- Use active voice by default.
- Avoid repeating the same sentence template across topics
  (such as always starting with "[They] [verb] ...").
- It is fine to vary sentence rhythm slightly as long as the sentence stays clear and concise.
- For sexual scenes, use clear gender-specific pronouns based on the heterosexual context:
  - "He" = male character, "She" = female character.
  - Good: "He uses his tongue to stimulate her clit between her thighs."
  - Bad: "She uses her tongue to stimulate her partner's clit" (ambiguous and grammatically confusing).

KEYWORD GROUNDING
- The LABEL must be clearly connected to the TOP 3–5 keywords:
  - Ideally include one of them directly.
  - If you use a synonym instead, it must preserve the same meaning
    (e.g., "car" → "vehicle", "invitation" → "invite").
- If among the top 5 keywords there is a concrete noun like "invitation", "wedding",
  "car", "hockey", "phone", "dog", or "family":
  - Prefer labels that explicitly mention that noun or an obvious synonym.
  - Example: with keywords "... invitation, complication ..." a label like
    "Reluctant Response to Invitation" is better than generic "Reluctant Decisions".
- HARD CONSTRAINT: If the top keywords contain specific sexual body parts (e.g. "breasts", "nipples",
  "clit", "pussy") or actions ("licking", "sucking", "grinding") that appear across
  multiple snippets, the LABEL MUST reflect that focus:
  - If "breasts" or "nipples" are in the top 5 keywords → the label MUST mention "breast" or "nipple"
  - If "clit" or "pussy" are in the top 5 keywords → the label MUST mention "clitoral" or "clit" or "pussy"
  - FORBIDDEN: Using generic labels like "Intimate Kissing" or "Intimate Mouth Kisses" when specific body parts
    like "breasts" or "nipples" are in the top keywords.
  - Example: with keywords "tongue, breasts, nipples, mouth, thighs", a label such as
    "Breast And Mouth Foreplay" or "Breast And Nipple Foreplay" is REQUIRED, NOT "Intimate Kissing" or "Intimate Mouth Kisses".
- The SCENE SUMMARY must reference at least ONE of the top 5 keywords
  (directly or via an obvious synonym) in a concrete way.
- For sexual topics, the scene summary should also mention at least one of those body parts or acts explicitly
  in a factual way when they appear in the top keywords.

RARE OR QUIRKY KEYWORDS
- Some topics may contain rare or quirky nouns in the top keywords
  (e.g., "pickles", "popcorn", "granite", "mustards") that appear only once
  or seem incidental.
- Do NOT force these rare words into unnatural combined phrases that do not
  make sense in plain English, such as:
  - "engagement ring pickles", "tripping over rules and popcorn".
- If a rare word appears only once and does not clearly define the main
  pattern, treat it as background detail:
  - You may omit it from the LABEL entirely.
  - In the scene summary, you may mention it briefly as a simple noun
    (e.g., "a bowl of popcorn on the table") OR omit it if it makes the
    sentence awkward or confusing.
- Only treat a rare or quirky noun as central to the topic if:
  1) it appears in multiple snippets, AND
  2) it clearly aligns with the main scene type (e.g., "popcorn" repeatedly
     in movie-night snippets).

REPRESENTATIVE SNIPPETS AND CENTRALITY
- Snippets are chosen to be representative of the topic, but may still contain book-specific details.
- Your goal is to identify the SHARED PATTERN across multiple snippets, not to copy one snippet.
- Treat names, cities, brands, unique objects, and one-off events as BOOK-SPECIFIC details
  unless they appear in multiple snippets AND align with the keywords.
- Use snippets as primary evidence for:
  - scene type (kitchen argument, car argument, office meeting, research interview),
  - emotional tone (angry, tender, playful, humiliated),
  - explicit sexual acts (e.g., blowjob, fingering, breast play, anal sex, clitoral stimulation),
  - but ONLY when these are clearly recurring patterns, not single mentions.

ACTIVE SNIPPET ANALYSIS
- Read each snippet carefully and extract concrete details that appear in MULTIPLE snippets:
  - Locations (kitchen, hallway, car, bedroom, office)
  - Actions (kissing, arguing, drinking, driving, playing games)
  - Objects (wine bottle, door, phone, board game, hockey puck)
  - Body parts (neck, clit, hips, mouth, hands)
  - Emotional states (angry, reluctant, uncertain, playful)
- When writing the LABEL:
  - Use ONLY details that are supported by the keywords AND appear across multiple snippets.
  - Do NOT anchor the label on a detail that appears in just one snippet.
- When writing the SCENE SUMMARY:
  - You may mention a vivid detail from a single snippet,
  - BUT the overall scene type must match the pattern across snippets and keywords,
    not just that one sentence.
  - CRITICAL: Do NOT include specific locations (kitchen, hallway, couch, parking lot) or relationships (son, father, jokes)
    in the scene summary UNLESS:
    1) The location/relationship appears in the TOP KEYWORDS, AND
    2) The location/relationship appears in MULTIPLE snippets.
  - If a location appears in only one snippet and NOT in keywords, do NOT include it in the summary.
  - Examples of FORBIDDEN summaries:
    ❌ "They notice each other's unexpected actions in the hallway" (when "hallway" is not in keywords)
    ❌ "She smiles and laughs genuinely at his jokes in the kitchen" (when "kitchen" and "jokes" are not in keywords)
    ❌ "He stimulates her clit between her thighs while they sit on the couch" (when "couch" is not in keywords)
    ❌ "They watch the game, the son cheering as his father, the goalie, blocks shots" (when "son" and "father" are not in keywords)
  - Examples of CORRECT summaries:
    ✅ "They notice each other's unexpected actions, struggling to understand their relationship" (no location, focuses on keywords)
    ✅ "She smiles and laughs genuinely, the corners of her mouth crinkling" (uses "corners" and "mouth" from keywords)
    ✅ "He stimulates her clit between her thighs with his hand" (uses keywords: clit, thighs, hand)
    ✅ "They watch the hockey game as the goalie blocks shot after shot" (uses "game", "goalie" from keywords, no family terms)
  - When such a repeated detail exists, you SHOULD include at least ONE concrete detail from the snippets
    (location, object, or specific action) that appears in multiple snippets **and** is supported by the keywords.
  - If no such repeated detail exists, do not force one; instead, describe the most robust shared pattern across snippets and keywords.
- If snippets show a pattern (e.g., "kitchen" appears in 3 snippets AND "kitchen" is in keywords, "wine" in 2 AND "wine" is in keywords),
  include that detail in your scene summary (e.g., "They argue in the kitchen while wine sits on the table").
- Do NOT write generic summaries like "They observe each other's actions" when snippets
  contain specific details like "hallway", "door", "knock", "stairs" that appear multiple times AND are in keywords.

SCENE GENERALIZATION RULES
- Your label must reflect what is CONSISTENTLY present across snippets and keywords.
- Do NOT treat a single snippet as defining the entire topic.
- If snippets depict many small details that are clearly tied to a single book
  (unique names, cities, quirky objects, one-off jokes), summarize the SCENE TYPE instead:
  - Good: "Picnic Date in City Park"
  - Bad: "Mark and Eva's Vancouver Picnic"
- When in doubt, prefer higher-level scene types:
  - "Family Dinner", "Job Interview", "Hospital Visit", "Car Argument", "Hockey Game with Son".

BOOK-SPECIFIC DETAILS AND NAMES
- Treat character names, pet names, and other unique proper nouns as BOOK-SPECIFIC details.
- Do NOT use character or pet names in the LABEL.
- In the scene summary, use generic roles instead of names unless:
  1) the name appears in the keywords, AND
  2) the same name appears in multiple snippets, AND
  3) the name is clearly central to the topic.
- Prefer roles such as "her roommate", "his roommate", "the dog", "her friend", "her boss"
  instead of "Paige", "Max", or other specific names.
- If a vivid detail (e.g. a particular roommate, a specific barking dog, a shrine in the room)
  appears in only a single snippet AND is not in the top keywords:
  - You may mention it briefly in the scene summary,
  - BUT you must NOT build the entire LABEL or main scene type around that one detail.

SEX-ACT PRECISION CRITERIA
Only label a specific sexual act (e.g., "Blowjob", "Anal Sex", "Clitoral Stimulation") if at least one of the following is true:
- The act clearly appears in MULTIPLE snippets, OR
- The act appears in MULTIPLE top keywords, OR
- Both snippets and keywords point to the same act.

If a sexual act appears only once, or is only implied in one snippet, treat it as a DETAIL of that book,
not as the essence of the topic.

In such cases, use a more general but still factual label, e.g.:
- "General Foreplay on Couch",
- "Kisses in Bedroom",
- "Intimate Touching in Shower".

UNCERTAINTY AND ABSTRACTION
- BEFORE defaulting to abstract labels, exhaustively search snippets for concrete patterns:
  1. Scan all snippets for recurring locations (kitchen, car, office, bedroom, hallway)
  2. Scan for recurring actions (arguing, deciding, refusing, inviting, waiting)
  3. Scan for recurring objects (door, phone, wine, food, game, clock)
  4. Scan for recurring emotional states (angry, reluctant, uncertain, playful)
- If you find a concrete pattern in snippets (and it is also reflected in keywords), use a concrete label:
  - Instead of "Unusual Behavior Noticed" → "Observing Actions In Hallway" (only if "hallway" appears in multiple snippets AND in the top keywords)
  - Instead of "Difficult Choices" → "Arguing Over Decisions in Kitchen" (only if "kitchen" and "arguing" appear in multiple snippets AND in the top keywords)
  - Instead of "Time Units Passing" → "Waiting and Watching Clock" or "Time Dragging By" (only if "clock" and "waiting" appear in multiple snippets AND in the top keywords)
  - NEVER use "Time Units Passing" - it sounds like a technical term. Always use natural phrases like "Time Dragging By", "Watching The Clock", "Hours Feel Like Minutes"
- Only use abstract labels if snippets are truly generic and contain no recurring concrete details.
- Abstract labels should still be specific:
  - Good: "Relationship Feelings and Uncertainty" (mentions both feelings AND uncertainty)
  - Bad: "Unusual Behavior Noticed" (too vague - what behavior? where?)
- Do NOT invent specific scenarios like "First Time With Woman", "Car Repair",
  "Wedding Proposal", "Boyfriend's Arrival During Date" unless those events are clearly present
  in MULTIPLE snippets or match explicit keywords.
- Do NOT describe a scene as "sexual tension", "charged", or "flirtation"
  unless both:
  1) at least one keyword or snippet clearly indicates romantic/sexual interest
     (e.g. "flirt", "crush", "turned on", "couldn't stop staring"), AND
  2) the rest of the evidence supports that interpretation.

HARD CONSTRAINTS FOR KNOWN HALLUCINATION PATTERNS
- Do NOT use "dinner date", "invitation", or "makeout" unless:
  - snippets or keywords clearly mention asking/inviting, saying yes/no, or kissing.
- Do NOT use "repair", "fix", "mechanic", "garage" or similar mechanical language
  in either the label or scene summary UNLESS:
  - at least one of these words (or a clear synonym) appears in the keywords OR
  - they appear in multiple snippets and the topic clearly centers on mechanical work.
- If the keywords only mention "car", "parking", "driveway", "vehicle", etc. without any
  explicit repair words, treat it as being about driving, parking, standing by cars,
  or talking near cars, NOT about car repair.
- Examples of FORBIDDEN: "discussing car repairs" or "fix hers" when keywords only have "car", "parking", "curb", "driveway"
- Example of CORRECT: "discussing car repairs" when keywords include "car", "repair", "engine", "fix"
- Do NOT use "heartbreak" or "breakup" unless emotional pain in a relationship
  is clearly described in the snippets.
- Do NOT label a topic as a specific sexual act (e.g., "Blowjob", "Handjob", "Anal Sex")
  if there are no explicit sexual terms in the snippets or keywords.
- Do NOT include family relationships (son, father, mother, daughter) in labels or summaries UNLESS:
  - the relationship term appears in the TOP KEYWORDS, AND
  - the relationship appears in MULTIPLE snippets.
  - Example: If keywords are "game", "hockey", "puck", "goalie" (no "son" or "father"),
    label should be "Watching a Hockey Game Together", NOT "Hockey Game With Son".

WORK / RESEARCH / META TOPICS
- Distinguish between:
  - "Work & Career Tasks" (job, office, boss, meeting, deadline),
  - "Researching Sexual Terminology" (research, terminology, user, giver, kink),
  - "Writing or Editing Scenes" (draft, editor, chapter, rewrite).
- Sexual research in the text should not be mislabeled as "Preparing for Work".

SEXUAL CONTENT PRECISENESS
- Do NOT euphemize explicit sexual content:
  - If a topic is clearly about blowjob / oral sex, say so factually, e.g. "Public Blowjob in Alley",
    not "Erotic Intimacy".
  - If a topic is about physical foreplay to breasts, use labels such as
    "Breast Foreplay", "Mouth On Breasts", or similar.
  - If a topic is about clitoral stimulation and legs/hips, use labels like
    "Clitoral Stimulation Between Thighs" or "Clitoral Foreplay With Tongue".
- HARD CONSTRAINT: When sexual body parts (e.g. "breasts", "nipples", "clit", "pussy") appear among the top keywords
  AND in multiple snippets, the label and scene summary MUST mention those body parts directly.
  - If "breasts" or "nipples" are in top 5 keywords → label MUST include "breast" or "nipple", summary MUST mention them
  - If "clit" or "pussy" are in top 5 keywords → label MUST include "clitoral" or "clit", summary MUST mention them
  - FORBIDDEN: Generic phrases like "Intimate Kissing", "Intimate Mouth Kisses", "Physical Intimacy" when specific
    body parts are clearly central to the topic (in top keywords).
  - The scene summary must explicitly describe the body part interaction, e.g. "He kisses and licks her breasts"
    not just "He kisses her" when breasts/nipples are in keywords.
- Always keep the phrasing clinical and non-romanticized (descriptive, not arousing).

SEXUAL WORDING GUARDRAIL (REINFORCEMENT)
- This section reinforces the CRITICAL guardrail above. Read it again if you are unsure.
- The words "foreplay", "intimate", "arousal", "erotic", "sexual", "kinky", "charged", "tension" 
  are FORBIDDEN unless explicit sexual terms (listed above) appear in keywords or snippets.
- Do NOT use metaphorical sexual euphemisms like "clenching center", "heat pooling low",
  "wetness between her thighs", "hot center", "aching need" unless explicit sexual terms
  appear in keywords or snippets.
- Common mistake: Seeing "hand", "touch", "arm", "close" and assuming it's sexual.
  These are NEUTRAL words that appear in many non-sexual contexts (board games, cooking, work).
- Another common mistake: Seeing "board game" or "playing game" and adding "foreplay" or "intimate".
  Board games are NOT sexual activities. Use neutral labels like "Board Game Around Table".
- Another common mistake: Taking a single explicit snippet and building the entire summary around it
  with sexual euphemisms, even when keywords don't support sexual content.
  Example: Keywords "freshman, roommate, shrine" → Do NOT write "revealing her hot, clenching center".
- If keywords contain "board", "game", "table", "hand", "arm", "chair", "mug" → 
  Label MUST be "Board Game Around Table" or "Playing Board Game", NEVER "Board Game Foreplay".

MINI CHAIN-OF-THOUGHT (INTERNAL ONLY)
Before answering, you should internally:
- Identify which actions, emotions, and settings are repeated across snippets.
- Identify which details are one-off outliers (names, cities, unusual objects).
- Decide whether any sexual act is clearly a shared pattern.
- Generalize the scene/theme from shared patterns only.

Do NOT output your reasoning. Only output the final label and scene summary.

OUTPUT FORMAT (IMPORTANT)
Return your answer in EXACTLY this format:

Label: <2–6 word noun phrase>
Scene summary: <ONE complete sentence (about 12–25 words) ending with a period>

CRITICAL REQUIREMENTS:
- The scene summary MUST be a complete, grammatically correct sentence.
- The scene summary MUST end with a period (.).
- The scene summary MUST include at least one concrete detail from the snippets.
- The scene summary MUST be between 12-25 words (count carefully).
- Do NOT include more than one sentence in the scene summary.
- Do NOT begin the scene summary with the word "Characters".
- Do NOT leave the scene summary hanging on a function word (e.g., ending with "to", "for", "a", "the").
- Do NOT output explanations, bullet points, or extra text.
- If your scene summary is too short (<12 words), add a concrete detail from the snippets.
- If your scene summary is too long (>25 words), remove less important words while keeping concrete details.

EXAMPLES (do NOT repeat these exact labels; mimic the style only)

Example 0 (Promise & Deals):
Topic keywords: means, idea, promise, work, help, better, true, deal, today, game
Representative snippets:
1) "You promised me you'd help, this isn't just a game to you."
2) "If we make this deal today, it could really change things."
Label: Promise and Deal Negotiation
Scene summary: They talk through a promise and a possible deal, weighing how it might change their work and future.

Example 3 (Meals):
Topic keywords: dinner, food, lunch, breakfast, bakery, chicken, sandwich, meal, dessert, hungry
Representative snippets:
1) "They grabbed sandwiches from the bakery and ate on the steps."
2) "Breakfast was just coffee and a half-eaten croissant."
Label: Everyday Meals And Food
Scene summary: They share simple meals and snacks throughout the day, from quick breakfasts to casual bakery lunches.

Example 4 (Abstract relationship feelings):
Topic keywords: way, years, matter, things, able, relationship, place, thing, feelings, kind
Representative snippets:
1) "After all these years, she still couldn't name what they were."
2) "It was the kind of relationship that never quite fit in one box."
Label: Relationship Feelings And Issues
Scene summary: Over time, they struggle to define what their relationship is and how it truly makes them feel.

Example 5 (Time passing in life/work):
Topic keywords: week, years, job, fallen, days, times, months, able, things, different
Representative snippets:
1) "Weeks turned into months at the new job before she realized how different everything felt."
2) "Over the years, the days blurred together into something she barely recognized."
Label: Time Passing In Work And Life
Scene summary: Weeks and months slip by at work and in daily life, gradually changing how everything feels.

Example 21 (Car scene):
Topic keywords: car, seat, door, parked, drive, window, highway, kiss, hand
Representative snippets:
1) "They sat in the parked car, his hand on her thigh as the windows fogged."
2) "She leaned across the seat, kissing him while the engine idled quietly."
Label: Makeout In Parked Car
Scene summary: In a parked car, they kiss and touch each other while the outside world stays just beyond the fogged windows.

Example 23 (Refused invitation):
Topic keywords: invite, invited, asked, yes, no, maybe, refused, party, drinks
Representative snippets:
1) "He invited her out for drinks, but she shook her head and said no."
2) '"I appreciate it, but I can't," she replied, refusing the invitation.'
Label: Refusing A Romantic Invitation
Scene summary: One person invites the other out, but the invitation is gently turned down and the moment passes.

Example 29 (Time references):
Topic keywords: minutes, hours, seconds, clock, later, time, passed, wait, longer, soon
Representative snippets:
1) "Minutes felt like hours as she stared at the clock."
2) "Time passed slowly while he waited for her to call."
Label: Waiting And Watching Clock
Scene summary: Time seems to drag or rush by as they watch the clock and wait for something to happen."""

ROMANCE_AWARE_USER_PROMPT = """Topic keywords (most important first):

{kw}{hints}

{pos}

{snippets}

CRITICAL CHECK BEFORE LABELING:

1. SEXUAL TERMS CHECK:
- Scan the keywords above. Do they contain explicit sexual terms like "sex", "fuck", "cock", "pussy", 
  "clit", "nipples", "blowjob", "handjob", "fingering", "orgasm", "penetration"?
- If NO explicit sexual terms are present, you MUST use neutral, non-sexual wording.
- FORBIDDEN words when no sexual terms: "foreplay", "intimate", "arousal", "erotic", "charged", "tension".
- Example: Keywords "board, game, table, hand, arm" → Label MUST be "Board Game Around Table", 
  NEVER "Board Game Foreplay" or "Intimate Board Game".

2. SEXUAL PRECISION CHECK (if sexual terms ARE present):
- If "breasts" or "nipples" are in the top 5 keywords → Label MUST include "breast" or "nipple"
- If "clit" or "pussy" are in the top 5 keywords → Label MUST include "clitoral" or "clit" or "pussy"
- FORBIDDEN: Generic labels like "Intimate Kissing" or "Intimate Mouth Kisses" when "breasts"/"nipples" are in top keywords
- Example: Keywords "tongue, breasts, nipples, mouth, thighs" → Label MUST be "Breast And Mouth Foreplay" or "Breast And Nipple Foreplay", 
  NEVER "Intimate Kissing" or "Intimate Mouth Kisses"

3. VAGUENESS CHECK:
- Does your label clearly indicate what the topic is about?
- FORBIDDEN vague labels: "Never Seen Before", "Things That Matter", "Something Different", "Unusual Behavior"
- If keywords include "relationship", "feelings", "years" → Use "Unclear Relationship Feelings" or "Struggling To Define Relationship"
- If keywords include "way", "matter", "things" → Use "Relationship Feelings And Uncertainty" or "Uncertain Feelings About Relationship"
- Always include at least one concrete keyword from the top 5 in your label.

4. TECHNICAL/ROBOTIC CHECK (for time-related topics):
- If keywords include "minutes", "hours", "days", "weeks", "months", "time", "clock" → 
  FORBIDDEN: "Time Units Passing", "Time Measurement", "Duration Tracking", "Temporal Units"
- Instead use natural phrases: "Time Dragging By", "Waiting And Watching Clock", "Hours Feel Like Minutes", "Watching The Clock"
- The label should describe how time feels, not how it is measured.

Remember:

- If NO representative snippets are shown above, you MUST base both the label and the scene summary
  directly on the keywords and context hints.
- In that case, do NOT invent very specific locations or props that are not implied by the keywords
  (for example, do not randomly introduce "bedroom" or "couch" if they are not in the keywords or hints).
- Base your label on the SHARED pattern across snippets, not on a single sentence.
- Do NOT treat a single vivid sentence as the whole topic:
  build both the label and the scene summary from what repeats
  across snippets and top keywords, not from one extreme line.
- Use the keywords to confirm important body parts, actions, or settings.
- Ignore single outlier words (e.g. a random city) unless repeated across snippets and keywords.
- Use precise, neutral, scene-level phrasing.
- Choose a subject that matches the evidence:
  - Use singular subjects ("She", "He", "The protagonist") when the focus is one person.
  - Use plural subjects ("They", "The couple", "The friends") only when multiple people clearly interact.
- Remember: All relationships are heterosexual (man/woman). Use "He" for male characters and "She" for female characters.
  For sexual interactions, be clear about who is doing what: "He [action] her [body part]" or "She [action] his [body part]".

CRITICAL: Before writing your scene summary, identify:
1. Which location appears in multiple snippets? (kitchen, hallway, car, bedroom, etc.)
2. Which object appears in multiple snippets? (wine, door, phone, game, etc.)
3. Which action appears in multiple snippets? (kissing, arguing, drinking, driving, etc.)
4. Which body part appears in multiple snippets? (if relevant: neck, clit, hips, etc.)

Then include at least ONE of these concrete details in your scene summary.

{existing_labels}

Return your answer in exactly this format:

Label: <2–6 word noun phrase>
Scene summary: <one concise sentence>

Label and scene summary:"""

# OpenRouter API configuration
# Get API key from environment variable, fallback to empty string if not set
DEFAULT_OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
# Default model: mistralai/mistral-nemo (optimized for romance-aware labeling)
# This is the primary model for all label generation tasks.
# For comparisons, use compare_models_openrouter.py which includes Grok as secondary reviewer.
# For reasoning experiments, use: google/gemini-2.5-flash
# DEFAULT_OPENROUTER_MODEL = "mistralai/mistral-nemo"  # Commented for reasoning experiments
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"  # Changed for reasoning experiments
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

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
        # Log API key status (masked for security)
        if api_key:
            api_key_display = f"{api_key[:10]}...{api_key[-4:]}" if len(api_key) > 14 else "***"
            LOGGER.info("API key: %s (length: %d)", api_key_display, len(api_key))
        else:
            LOGGER.warning("API key is empty or None! Authentication may fail.")
        
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
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
        use_improved_prompts: If True, use BASE_LABELING_PROMPT and parse JSON response
        representative_docs: Optional list of representative document strings (snippets)
        max_snippets: Maximum number of snippets to include in prompt (default: 6)
        max_chars_per_snippet: Maximum characters per snippet before truncation (default: 200)
        existing_labels: Optional set of existing labels to avoid reusing (for romance-aware prompts)
        reasoning_effort: Optional reasoning effort level ("low", "medium", "high") for supported models
        
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
            seed=42,  # Fixed seed for reproducibility
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
                if scene_summary:
                    # Clean scene summary to remove hallucinations
                    scene_summary = clean_scene_summary(scene_summary, keywords)
                    result_dict["scene_summary"] = scene_summary
                
                LOGGER.info("Generated label (improved prompt): %s | Categories: %s", 
                           label, primary_categories)
                return result_dict
                
            except (json.JSONDecodeError, KeyError) as e:
                LOGGER.warning("Failed to parse JSON response, falling back to text extraction: %s", e)
                # Fall through to text extraction below
        
        # Romance-aware prompt path (label + scene summary)
        # Expect format:
        #   Label: <...>
        #   Scene summary: <...>
        label_text = content
        scene_summary = ""
        
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

