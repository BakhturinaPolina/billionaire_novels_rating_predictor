# Prompt Improvement Suggestions Based on Output Analysis

## Issues Identified in Current Output

1. **Duplicate Labels**: Topics 13 and 25 both labeled "Playful Smiles and Laughter"
2. **Similar Labels**: Topics 2 and 15 both labeled "Clitoral Foreplay" 
3. **Abstract Labels**: Topics 4, 7, 8, 28, 29 too generic
4. **Generic Scene Summaries**: Missing concrete details from snippets
5. **Underutilized Snippets**: Not extracting specific details effectively

## Suggested Prompt Improvements

### 1. Enhanced Snippet Utilization Section

**Current Issue**: The prompt mentions using snippets but doesn't emphasize active extraction of concrete details.

**Suggested Addition** (after line 146):

```python
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

ACTIVE SNIPPET ANALYSIS (NEW)
- Read each snippet carefully and extract concrete details that appear in MULTIPLE snippets:
  - Specific locations (kitchen, hallway, car, bedroom, office, restaurant)
  - Specific actions (kissing, arguing, drinking, driving, playing games)
  - Specific objects (wine bottle, door, phone, board game, hockey puck)
  - Specific body parts (neck, clit, hips, mouth, hands)
  - Specific emotional states (angry, playful, reluctant, intense)
- For the scene summary, you MUST include at least ONE concrete detail from the snippets
  (location, object, or specific action) that appears in multiple snippets.
- If snippets show a pattern (e.g., "kitchen" appears in 3 snippets, "wine" in 2),
  include that detail in your scene summary (e.g., "They argue in the kitchen while wine sits on the table").
- Do NOT write generic summaries like "They observe each other's actions" when snippets
  contain specific details like "hallway", "door", "knock", "stairs" that appear multiple times.
```

### 2. Distinguishing Similar Topics

**Current Issue**: Topics with similar keywords get identical labels.

**Suggested Addition** (after line 127):

```python
DISTINGUISHING SIMILAR TOPICS
- When keywords overlap significantly (e.g., both topics have "smile", "laugh", "grin"),
  you MUST find distinguishing features in the snippets or keyword order:
  - Compare the snippets: Are they in different settings? Different emotional contexts?
  - Check keyword order: Which words appear first? (e.g., "bright smile" vs "forced laugh")
  - Look for unique keywords: Does one topic have "eyebrows", "giggles" while the other has "burst", "goofy"?
- If topics are truly identical, use the same label, but if snippets show different contexts,
  create distinct labels:
  - Good: "Playful Smiles With Eyebrow Raises" vs "Loud Bursting Laughter"
  - Bad: "Playful Smiles and Laughter" for both
- For sexual topics with similar keywords, distinguish by:
  - Body part focus (e.g., "Clitoral Foreplay With Tongue" vs "Clitoral Foreplay With Hand")
  - Setting (e.g., "Clitoral Foreplay in Bedroom" vs "Clitoral Foreplay in Shower")
  - Intensity or technique (e.g., "Gentle Clitoral Stimulation" vs "Intense Clitoral Foreplay")
```

### 3. Making Abstract Topics More Concrete

**Current Issue**: Topics 4, 7, 8, 28, 29 are too abstract.

**Suggested Addition** (replace lines 172-178):

```python
UNCERTAINTY AND ABSTRACTION
- BEFORE defaulting to abstract labels, exhaustively search snippets for concrete patterns:
  1. Scan all snippets for recurring locations (kitchen, car, office, bedroom, hallway)
  2. Scan for recurring actions (arguing, deciding, refusing, inviting, waiting)
  3. Scan for recurring objects (door, phone, wine, food, game, clock)
  4. Scan for recurring emotional states (angry, reluctant, uncertain, playful)
- If you find a concrete pattern in snippets (even if keywords are abstract),
  use a concrete label:
  - Instead of "Unusual Behavior Noticed" → "Observing Actions in Hallway" (if hallway appears in snippets)
  - Instead of "Difficult Choices" → "Arguing Over Decisions in Kitchen" (if kitchen/arguing in snippets)
  - Instead of "Time Units Passing" → "Waiting and Watching Clock" (if clock/waiting in snippets)
- Only use abstract labels if snippets are truly generic and contain no recurring concrete details.
- Abstract labels should still be specific:
  - Good: "Relationship Feelings and Uncertainty" (mentions both feelings AND uncertainty)
  - Bad: "Unusual Behavior Noticed" (too vague - what behavior? where?)
```

### 4. Enhanced Scene Summary Requirements

**Current Issue**: Many scene summaries are generic and don't use snippet details.

**Suggested Addition** (replace lines 84-91):

```python
- The sentence should read like a natural description of a scene in a novel,
  not like an abstract, generic summary.
- Avoid starting the sentence with the word "Characters". Prefer concrete agents like
  "They", "The couple", "She", "He", "The family", or describe the scene directly
  when the agents are obvious (e.g., "In the kitchen, they argue about breakfast.").
- MANDATORY: Include at least ONE concrete detail from the snippets in your scene summary:
  - A specific location (kitchen, hallway, car, bedroom, office)
  - A specific object (wine bottle, door, phone, board game, hockey puck)
  - A specific action (kissing on neck, sipping wine, knocking on door, playing game)
  - A specific body part (if relevant: neck, clit, hips, mouth)
- The detail must appear in MULTIPLE snippets (not just one).
- Examples of good scene summaries with concrete details:
  - "They knock on and open doors in the hallway, trying to gain access." (includes: hallway, doors, knocking)
  - "They sip wine while waiting for their dinner in the kitchen." (includes: wine, kitchen, waiting)
  - "He leans in, softly kissing her neck." (includes: neck, kissing, leaning)
- Examples of bad scene summaries (too generic):
  - "They observe each other's unexpected actions." (no concrete details)
  - "They notice their phones buzzing or ringing at various times." (too vague - "various times" is not concrete)
  - "They measure and experience time in distinct units." (completely abstract)
```

### 5. Better Output Format Enforcement

**Current Issue**: Some scene summaries may be truncated or incomplete.

**Suggested Addition** (replace lines 214-223):

```python
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
```

### 6. Enhanced User Prompt Instructions

**Suggested Addition** (replace lines 291-299):

```python
Remember:

- Base your label on the SHARED pattern across snippets, not on a single sentence.
- Use the keywords to confirm important body parts, actions, or settings.
- Ignore single outlier words (e.g. a random city) unless repeated across snippets and keywords.
- Use precise, neutral, scene-level phrasing.

CRITICAL: Before writing your scene summary, identify:
1. Which location appears in multiple snippets? (kitchen, hallway, car, bedroom, etc.)
2. Which object appears in multiple snippets? (wine, door, phone, game, etc.)
3. Which action appears in multiple snippets? (kissing, arguing, drinking, driving, etc.)
4. Which body part appears in multiple snippets? (if relevant: neck, clit, hips, etc.)

Then include at least ONE of these concrete details in your scene summary.
```

## Implementation Priority

1. **High Priority**: Enhanced snippet utilization (#1) and scene summary requirements (#4)
2. **Medium Priority**: Distinguishing similar topics (#2) and making abstract topics concrete (#3)
3. **Low Priority**: Output format enforcement (#5) - current format is mostly working

## Testing Recommendations

After implementing these changes, test on:
- Topics 13 vs 25 (should get different labels)
- Topics 2 vs 15 (should get different labels)
- Topics 4, 7, 8, 28, 29 (should get more concrete labels)
- All scene summaries (should include concrete details from snippets)

