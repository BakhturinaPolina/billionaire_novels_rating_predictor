# Prompt Templates Documentation

This document contains all versions of prompts used for topic labeling in the stage06_labeling module.

## Version 1: Universal Prompt (from generate_labels.py)

This is the prompt currently used in `generate_labels.py` for local LLM inference. It produces a single short label (2-6 words).

### System Prompt

```
You are a topic-labeling assistant.

Goal

- Produce a short, specific label for a cluster of keywords from a corpus.

- The label must help distinguish this topic from similar ones.

Style

- One short noun phrase, 4–8 words. No quotes. No trailing punctuation.

- Prefer specific entities or attributes over generic abstractions.

- If the keywords imply a facet (body part, activity, time period, object, place, emotion, event, relationship, profession, product, sport, etc.), name it explicitly.

- Use disambiguators only when clear (e.g., "Foreplay", "Explicit", "Wedding Planning", "Doorways & Elevators", "Wine Service").

- Avoid simply repeating a list of keywords; synthesize the core idea instead.

Consistency

- Use title casing where natural (not ALL CAPS).

- Avoid numbers unless essential (e.g., "24-Hour Timeline").

Output

- Return only the label, nothing else.
```

### User Prompt Template

```
Topic keywords: {kw}{hints}

Label:
```

Where:
- `{kw}` is replaced with a comma-separated list of keywords
- `{hints}` is replaced with optional context hints based on detected domains (empty string if no domains detected)

### Example Usage

**Input keywords:** `["smile", "laugh", "grin"]`

**Detected domains:** `["FacialExpr"]`

**Generated hints:** `"\nContext hints: Prefer concrete expressions (e.g., 'Playful Laughter', 'Gaze & Brows')."`

**Full prompt:**
```
Topic keywords: smile, laugh, grin
Context hints: Prefer concrete expressions (e.g., 'Playful Laughter', 'Gaze & Brows').

Label:
```

**Expected output:** A short label like "Playful Laughter" or "Joyful Expressions"

---

## Version 2: Enhanced 6-Component Prompt (from notebook)

This is the prompt template used in the archived notebook experiments. It produces 6 tab-separated components for more detailed topic analysis.

### Full Prompt Template

```
Generate a topic label with EXACTLY 6 TAB-SEPARATED components based on these detailed guidelines:

1. CHARACTER DYNAMICS: Relationships, interactions, or conflicts between characters
   • Example types: "Rivalry Turned Alliance", "Unrequited Love", "Mentor-Protégé Conflict"
   • If keywords like "friend", "love", "trust", "betray" appear, include this component
   • Use "None" if no clear character dynamics are present

2. THEMES: Abstract ideas representing moral, philosophical, or ideological concerns
   • Example types: "Duty vs. Desire", "The Power of Redemption", "Class Struggles"
   • ALWAYS include a theme, even if abstract or general
   • For objects/places, focus on what they represent (e.g., "doors" → "Transitions and Boundaries")

3. SETTINGS: Physical locations or conceptual environments
   • Example types: "Haunted Castle", "War-Torn Battlefield", "Urban Underworld"
   • Use "None" if no clear setting is indicated in keywords
   • Be specific but not overly speculative

4. ACTIONS & EVENTS: Plot-driven occurrences advancing the narrative
   • Example types: "Betrayal and Retribution", "Quest for Identity", "Escape from Captivity"
   • Focus on actions mentioned in keywords
   • Use "None" if no clear events are indicated

5. EMOTIONAL ATMOSPHERE: Overall mood and emotional impact
   • Format: "Mood - [emotion]" (e.g., "Mood - Cheerful", "Mood - Tense")
   • Use "Mood - Cheerful" for positive emotions (smile, laugh)
   • Use "Mood - Tense" for conflict/danger (war, fight)
   • Default to "Mood - Neutral" for neutral topics

6. REPRESENTATION: How the topic is portrayed
   • Format: "Portrayal - [tone]" (e.g., "Portrayal - Positive", "Portrayal - Serious")
   • Use "Portrayal - Positive" for topics portrayed favorably
   • Use "Portrayal - Serious" for grave or solemn topics
   • Default to "Portrayal - Neutral" for neutral topics

IMPORTANT EXAMPLES FROM EXISTING TOPICS:

Topic: smile, laugh, grin
Friendly Camaraderie\tJoy and Lightheartedness\tJoyous Social Gatherings\tFestive or Cheerful Gatherings\tMood - Cheerful\tPortrayal - Positive

Topic: war, combat, battle
Allies and Enemies in Battle\tConflict and Warfare\tWar-Torn Battlefield\tWar and Combat\tMood - Tense\tPortrayal - Serious

Topic: door, doorway, entrance
None\tTransitions and Boundaries\tEntrances & Corridors\tEntering and Exiting\tMood - Neutral\tPortrayal - Neutral

Topic: kiss, kissed, embrace
Romantic Intimacy\tLove and Affection\tPrivate Romantic Spaces\tKissing and Embracing\tMood - Affectionate\tPortrayal - Positive

RULES:
- Use exactly ONE tab (\\t) between components
- Be descriptive but concise for each component
- Use "None" when appropriate for dynamics, settings, or events
- ALWAYS provide a theme (never "None" for theme component)
- Follow the mood and portrayal formats exactly
- Return ONLY the 6 tab-separated components with no other text

YOUR TASK:
For the following keywords, provide 6 tab-separated components following the guidelines above.

TOPIC KEYWORDS: [KEYWORDS]
REPRESENTATIVE DOCUMENTS:
[DOCUMENTS]
```

### Example Usage

**Input keywords:** `["smile", "laugh", "grin"]`

**Representative documents:** `["help smile back, bit back smile, smiled smile..."]`

**Expected output:**
```
Friendly Camaraderie	Joy and Lightheartedness	Joyous Social Gatherings	Festive or Cheerful Gatherings	Mood - Cheerful	Portrayal - Positive
```

### Output Format

The output is a single line with 6 tab-separated values:
1. Character Dynamics
2. Themes
3. Settings
4. Actions & Events
5. Emotional Atmosphere (format: "Mood - [emotion]")
6. Representation (format: "Portrayal - [tone]")

---

## Version 3: Romance-Aware Prompt

This is a romance-specific prompt designed for modern romantic fiction. It produces a single short label (2-6 words) with enhanced romance-relevant disambiguation rules.

### System Prompt

```
You are a topic-labeling assistant for modern romantic fiction.

Goal

- Produce one short, specific label (2–6 words) for a cluster of romance-novel keywords.

- The label must clearly distinguish this topic from similar ones and prioritize romance-relevant facets.

Romance Context

- Prefer facets common in romantic scenes and plots: physical intimacy (body parts, touches, foreplay, oral), emotional beats (jealousy, fear, apology, panic), relationship milestones (engagement, vows, wedding, breakup, reunion), communication moments (whispers, texts, calls), setting cues (bedroom, doorways/elevators, kitchen/meal, bar/night), family dynamics (parents, kids), and sensual aesthetics (scent, fabric, lingerie, makeup).

- If multiple facets appear, pick the single most distinctive one.

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

- Return only the label.
```

### User Prompt Template

```
Topic keywords: {kw}{hints}

{pos}

Label:
```

Where:
- `{kw}` is replaced with a comma-separated list of keywords
- `{hints}` is an optional single line starting with `Context hints:` (e.g., `Context hints: If body parts and touch verbs dominate, prefer intimacy labels like "Tender Foreplay".`)
- `{pos}` is an optional single line starting with `POS cues:` (e.g., `POS cues: Nouns→breasts, lips; Verbs→kiss, cup; Adjectives→tender.`)

### Example Usage

**Input keywords:** `["kiss", "lips", "tender", "embrace"]`

**Context hints:** `Context hints: If body parts and touch verbs dominate, prefer intimacy labels like "Tender Foreplay".`

**POS cues:** `POS cues: Nouns→lips; Verbs→kiss, embrace; Adjectives→tender.`

**Full prompt:**
```
Topic keywords: kiss, lips, tender, embrace
Context hints: If body parts and touch verbs dominate, prefer intimacy labels like "Tender Foreplay".

POS cues: Nouns→lips; Verbs→kiss, embrace; Adjectives→tender.

Label:
```

**Expected output:** A short label like "Tender Foreplay" or "Intimate Kisses"

---

## Comparison

| Aspect | Version 1 (Universal) | Version 2 (6-Component) | Version 3 (Romance-Aware) |
|--------|----------------------|------------------------|--------------------------|
| **Output Format** | Single short label (2-6 words) | 6 tab-separated components | Single short label (2-6 words) |
| **Complexity** | Simple, concise | Detailed, structured | Romance-specific, detailed rules |
| **Use Case** | Quick topic identification | Comprehensive topic analysis | Romance novel topic labeling |
| **Domain Adaptation** | Uses adaptive hints | Fixed structure | Romance-specific disambiguation |
| **Current Usage** | `generate_labels.py` (local LLM) | Notebook experiments (OpenRouter) | Available for romance-specific labeling |
| **Integration** | Direct label string | Requires parsing tab-separated values | Direct label string |
| **Special Features** | Generic domain hints | Multi-component analysis | Romance context, POS cues support |

---

## Notes

- **Version 1** is currently used in the main `generate_labels.py` module for local LLM inference
- **Version 2** was used in experimental notebook workflows with OpenRouter API
- **Version 3** is a romance-specific prompt designed for modern romantic fiction with enhanced disambiguation rules and optional POS cues support
- All versions can be adapted for use with OpenRouter API or local LLMs
- The choice between versions depends on the desired level of detail in topic labels and domain specificity

