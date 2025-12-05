# Prompt Improvements: Code Snippets and Examples

## Analysis of Current Output vs. Expected Output

### Issue 1: Duplicate Labels (Topics 13 & 25)

**Current Output:**
- Topic 13: "Playful Smiles and Laughter" - keywords: smile, laugh, laughs, grin, eyebrows, giggles, laughter, small, eyebrow, bright
- Topic 25: "Playful Smiles and Laughter" - keywords: smile, laugh, laughter, grin, burst, forced, lip, loud, goofy, cheeks

**Problem**: Both topics have identical labels despite different keyword emphasis.

**Solution**: Add distinction logic in prompt:

```python
# In ROMANCE_AWARE_SYSTEM_PROMPT, add after line 127:

DISTINGUISHING SIMILAR TOPICS
- When topics share many keywords, compare snippet details and keyword order:
  - Topic 13 has "eyebrows", "giggles", "bright" → suggests subtle, bright expressions
  - Topic 25 has "burst", "loud", "goofy", "cheeks" → suggests loud, explosive laughter
- Create distinct labels:
  - Topic 13: "Bright Smiles With Eyebrow Expressions"
  - Topic 25: "Loud Bursting Laughter"
```

### Issue 2: Abstract Labels (Topic 4)

**Current Output:**
- Topic 4: "Unusual Behavior Noticed"
- Keywords: way, years, matter, things, able, relationship, place, thing, feelings, kind
- Scene summary: "They observe each other's unexpected actions."

**Problem**: Too abstract - no concrete details despite snippets likely containing specific actions/locations.

**Solution**: Force concrete detail extraction:

```python
# In ROMANCE_AWARE_SYSTEM_PROMPT, replace lines 172-178:

UNCERTAINTY AND ABSTRACTION
- BEFORE using abstract labels, scan snippets for:
  1. Recurring locations (appears in 2+ snippets)
  2. Recurring actions (appears in 2+ snippets)  
  3. Recurring objects (appears in 2+ snippets)
- If snippets contain "hallway" in 3 snippets + "door" in 2 snippets:
  → Use: "Observing Actions in Hallway" (concrete)
  → NOT: "Unusual Behavior Noticed" (abstract)
- Only use abstract if snippets are truly generic with no recurring concrete details.
```

### Issue 3: Generic Scene Summaries

**Current Output Examples:**
- Topic 4: "They observe each other's unexpected actions." (no concrete details)
- Topic 24: "They notice their phones buzzing or ringing at various times." (vague - "various times")
- Topic 29: "They measure and experience time in distinct units." (completely abstract)

**Good Examples (for comparison):**
- Topic 10: "They knock on and open doors in the hallway, trying to gain access." (concrete: hallway, doors, knocking)
- Topic 12: "He leans in, softly kissing her neck." (concrete: neck, kissing, leaning)

**Solution**: Mandatory concrete detail requirement:

```python
# In ROMANCE_AWARE_SYSTEM_PROMPT, replace lines 84-91:

- MANDATORY: Include at least ONE concrete detail from snippets in scene summary:
  - Location: kitchen, hallway, car, bedroom, office
  - Object: wine bottle, door, phone, board game
  - Action: kissing on neck, sipping wine, knocking on door
  - Body part: neck, clit, hips, mouth (if relevant)
- The detail MUST appear in MULTIPLE snippets (not just one).
- Examples:
  ✓ "They knock on and open doors in the hallway, trying to gain access." (includes: hallway, doors)
  ✗ "They observe each other's unexpected actions." (no concrete details)
```

### Issue 4: Underutilized Snippets

**Current Code** (lines 496-532):
```python
def format_snippets(
    docs: list[str],
    max_snippets: int = 15,
    max_chars: int = 1200,
) -> str:
    # ... formats snippets as numbered quotes
    return "Representative snippets (short excerpts for this topic):\n" + "\n".join(snippets)
```

**Problem**: Snippets are provided but prompt doesn't emphasize active extraction of details.

**Solution**: Enhance user prompt instructions:

```python
# In ROMANCE_AWARE_USER_PROMPT, replace lines 291-299:

Remember:

- Base your label on the SHARED pattern across snippets, not on a single sentence.
- Use the keywords to confirm important body parts, actions, or settings.
- Ignore single outlier words (e.g. a random city) unless repeated across snippets and keywords.
- Use precise, neutral, scene-level phrasing.

BEFORE writing your scene summary, identify in the snippets:
1. Which location appears in 2+ snippets? (kitchen, hallway, car, bedroom, etc.)
2. Which object appears in 2+ snippets? (wine, door, phone, game, etc.)
3. Which action appears in 2+ snippets? (kissing, arguing, drinking, driving, etc.)
4. Which body part appears in 2+ snippets? (if relevant: neck, clit, hips, etc.)

Then include at least ONE of these concrete details in your scene summary.
```

## Specific Code Changes

### Change 1: Add Active Snippet Analysis Section

**Location**: `generate_labels_openrouter.py`, after line 146

**Add**:
```python
ACTIVE SNIPPET ANALYSIS
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

### Change 2: Add Distinguishing Similar Topics Section

**Location**: `generate_labels_openrouter.py`, after line 127

**Add**:
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

### Change 3: Enhance Scene Summary Requirements

**Location**: `generate_labels_openrouter.py`, replace lines 84-91

**Replace with**:
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

### Change 4: Update User Prompt Instructions

**Location**: `generate_labels_openrouter.py`, replace lines 291-299

**Replace with**:
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

## Expected Improvements

After implementing these changes:

1. **Topics 13 & 25**: Should get distinct labels based on keyword differences
2. **Topic 4**: Should get more concrete label if snippets contain specific details
3. **All scene summaries**: Should include at least one concrete detail from snippets
4. **Abstract topics**: Should be more concrete when snippets provide details

## Testing Checklist

- [ ] Topics 13 and 25 get different labels
- [ ] Topics 2 and 15 get different labels (if they should be different)
- [ ] Topics 4, 7, 8, 28, 29 get more concrete labels
- [ ] All scene summaries include at least one concrete detail
- [ ] No scene summaries are completely abstract when snippets contain details

