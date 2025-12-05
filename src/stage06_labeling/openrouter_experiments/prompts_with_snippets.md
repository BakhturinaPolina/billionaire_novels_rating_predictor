# Prompt Templates with Representative Snippets

This document describes the enhanced prompt structure that includes representative document snippets alongside keywords for improved topic labeling precision.

## Overview

The snippets-enhanced prompts provide the LLM with actual scene-level context from the corpus, enabling more precise and neutral labels. Instead of relying solely on keyword lists, the model can see patterns in actual sentences, leading to better distinctions (e.g., "Blowjob in Car" vs "Erotic Intimacy").

## System Prompt

The system prompt includes a "REPRESENTATIVE SNIPPETS" section that instructs the model on how to use snippets:

```
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
```

For the complete system prompt, see `prompts.md`.

## User Prompt Template

The user prompt now includes a snippets section between POS cues and the label prompt:

```
Topic keywords (most important first):

{kw}{hints}

{pos}
{snippets}

Remember:

- Base your label primarily on the shared pattern in the snippets.

- Use the keywords to check you are not missing important body parts, actions, or settings.

- Ignore single outlier words (e.g. a random city) unless they appear in multiple snippets and keywords.

- Use precise, neutral, scene-level phrasing, 2–6 words only.

Label:
```

Where:
- `{kw}` is the comma-separated keyword list
- `{hints}` is optional context hints (empty if no domains detected)
- `{pos}` is optional POS cues (empty if none detected)
- `{snippets}` is the formatted snippets block (empty if no representative docs available)

## Snippet Formatting

Snippets are formatted as numbered quotes:

```
Representative snippets (short excerpts for this topic):

1) "She knelt between his thighs, taking him into her mouth."

2) "Her tongue teased him as he gripped the sheets."

3) "He groaned when she sucked harder, hips jerking."

4) "She moved her head up and down, increasing the pace."

5) "His hands tangled in her hair, guiding her rhythm."

6) "He came with a shout, spilling into her mouth."
```

### Formatting Rules

- Each snippet is numbered: `{i}) "{text}"`
- Header: "Representative snippets (short excerpts for this topic):"
- Snippets are truncated at word boundaries if they exceed `max_chars_per_snippet` (default: 200)
- Maximum `max_snippets` snippets per topic (default: 6)
- Empty string returned if no docs provided

## Example: Before and After

### Before (Keywords Only)

**Input:**
```
Topic keywords: mouth, tongue, suck, lips, head, hair

POS cues: Nouns→mouth, tongue, lips, head, hair; Verbs→suck.
```

**Output:** "Erotic Intimacy" or "Oral Intimacy"

### After (Keywords + Snippets)

**Input:**
```
Topic keywords (most important first):

mouth, tongue, suck, lips, head, hair

POS cues: Nouns→mouth, tongue, lips, head, hair; Verbs→suck.

Representative snippets (short excerpts for this topic):

1) "She knelt between his thighs, taking him into her mouth."

2) "Her tongue teased him as he gripped the sheets."

3) "He groaned when she sucked harder, hips jerking."

4) "She moved her head up and down, increasing the pace."

5) "His hands tangled in her hair, guiding her rhythm."

6) "He came with a shout, spilling into her mouth."

Remember:

- Base your label primarily on the shared pattern in the snippets.

- Use the keywords to check you are not missing important body parts, actions, or settings.

- Ignore single outlier words (e.g. a random city) unless they appear in multiple snippets and keywords.

- Use precise, neutral, scene-level phrasing, 2–6 words only.

Label:
```

**Output:** "Blowjob in Bed" or "Oral Sex Scene"

## Benefits

1. **Precision**: Snippets reveal exact acts (blowjob vs generic intimacy)
2. **Context**: Setting and tone are visible (kitchen argument vs general anger)
3. **Neutrality**: Explicit instructions prevent euphemisms
4. **Scene-level understanding**: Labels capture actual narrative patterns, not just keyword associations

## Implementation Details

- Snippets are extracted from BERTopic's representative documents
- Default: 6 snippets per topic, 200 chars max per snippet
- Fallback: If snippets unavailable, prompt works with keywords only (backward compatible)
- Token cost: ~72 tokens for 6 snippets (6 sentences × ~12 tokens each)

## See Also

- `prompts.md`: Complete prompt documentation including full system and user prompts
- `generate_labels_openrouter.py`: Implementation of the prompt system

