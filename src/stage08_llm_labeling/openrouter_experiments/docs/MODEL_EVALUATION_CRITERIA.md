# Model Evaluation Criteria for Digital Humanities Research

**Date:** December 2024  
**Purpose:** Criteria for evaluating which OpenRouter model (Nemo-Instruct, Gutenberg, Celeste) works best for BERTopic label generation in romance fiction research

---

## Overview

When comparing model outputs from `compare_models_openrouter.py`, use this checklist to evaluate which model aligns best with your **Digital Humanities research goals**. The goal is to select a model that produces labels that are:

1. **Research-reliable** (no hallucination, literal interpretation)
2. **Domain-appropriate** (romance/erotic fiction aware)
3. **Discriminative** (distinct topics get distinct labels)
4. **Analytically useful** (labels support your research questions)

---

## Evaluation Checklist

### 1. Label Quality

#### 1.1. Specificity
**Question:** Is the label specific enough to distinguish this topic from similar ones?

**Good examples:**
- ✅ "Argument In Kitchen About Money" (specific: location + topic)
- ✅ "Rough Angry Kisses Against Wall" (specific: action + tone + location)
- ✅ "Clitoral Stimulation During Foreplay" (specific: act + context)

**Bad examples:**
- ❌ "Relationship Conflict" (too vague, could apply to many topics)
- ❌ "Erotic Intimacy" (generic, not discriminative)
- ❌ "Romantic Moment" (too abstract, no concrete details)

**Evaluation:**
- [ ] Label includes at least one concrete detail (location, body part, specific act, object)
- [ ] Label is specific enough that you could identify this topic from the label alone
- [ ] Label distinguishes this topic from similar topics in your corpus

#### 1.2. Genre Awareness
**Question:** Does the label recognize romance/erotic fiction conventions?

**Good examples:**
- ✅ Distinguishes "First Time Sex Scene" from "Established Relationship Sex"
- ✅ Recognizes "BDSM Negotiation" as distinct from "Consensual BDSM Scene"
- ✅ Identifies "Emotional Reconciliation After Fight" vs "Physical Makeup Sex"

**Bad examples:**
- ❌ Uses generic relationship terms that ignore genre context
- ❌ Fails to distinguish between romantic vs. erotic vs. domestic scenes
- ❌ Misses genre-specific tropes (e.g., "enemies to lovers" tension)

**Evaluation:**
- [ ] Label reflects understanding of romance/erotic fiction conventions
- [ ] Label distinguishes romantic, erotic, and domestic/emotional content appropriately
- [ ] Label captures genre-specific nuances (tropes, power dynamics, relationship stages)

#### 1.3. Discriminative Power
**Question:** Do different topics get clearly distinguishable labels?

**Test:** Look at 5-10 similar topics (e.g., all about arguments, all about sex scenes, all about dates).

**Good examples:**
- ✅ Topic 3: "Kitchen Argument About Money"
- ✅ Topic 7: "Bedroom Argument About Jealousy"
- ✅ Topic 15: "Public Argument Leading to Breakup"

**Bad examples:**
- ❌ Multiple topics all labeled "Relationship Conflict"
- ❌ Topics collapse into same vague phrase despite different keywords
- ❌ Labels are too similar to distinguish topics in analysis

**Evaluation:**
- [ ] Similar topics receive distinct labels that capture their differences
- [ ] No more than 10% of topics share identical labels (unless they truly are the same)
- [ ] Labels enable you to group/analyze topics by theme without confusion

---

### 2. Scene Summary Quality

**Note:** Scene summaries are only available when using `--use-improved-prompts` flag.

#### 2.1. Micro-Scene Focus
**Question:** Does the summary focus on a specific scene, not a whole plot?

**Good examples:**
- ✅ "He grabs her roughly and kisses her against the kitchen wall, anger evident in his movements."
- ✅ "She sits at the table, placing board game pieces while they discuss the rules."

**Bad examples:**
- ❌ "This theme explores the dynamics of power in relationships." (too abstract, not a scene)
- ❌ "The entire story arc of their relationship from meeting to marriage." (too broad)

**Evaluation:**
- [ ] Summary describes a specific moment/scene, not a plot arc
- [ ] Summary is concrete and visual (you can "see" what's happening)
- [ ] Summary avoids abstract academic language ("explores", "represents", "signifies")

#### 2.2. Concrete Details
**Question:** Does the summary include at least one concrete detail?

**Good examples:**
- ✅ Includes location: "in the kitchen", "against the wall", "at the restaurant"
- ✅ Includes objects: "board game pieces", "car keys", "wine glass"
- ✅ Includes body parts: "on her neck", "his hands", "her clit"
- ✅ Includes specific actions: "grabs roughly", "whispers", "places carefully"

**Bad examples:**
- ❌ "They have a conversation about their relationship." (no concrete details)
- ❌ "Emotional intimacy occurs." (too abstract)

**Evaluation:**
- [ ] Summary includes at least one concrete detail (location, object, body part, or specific action)
- [ ] Summary provides enough context to understand the scene type
- [ ] Summary avoids vague language ("something happens", "they interact")

#### 2.3. Tone and Style
**Question:** Is the summary neutral and analytical, not dramatic or chatty?

**Good examples:**
- ✅ "He argues with her about money while they stand in the kitchen."
- ✅ "She performs oral sex on him in the bedroom."

**Bad examples:**
- ❌ "OMG they're totally fighting and it's so intense!!" (chatty, second-person)
- ❌ "You feel the tension building as they..." (second-person RP style)
- ❌ "This is a beautiful moment of reconciliation." (overly dramatic)

**Evaluation:**
- [ ] Summary uses third-person, neutral language
- [ ] Summary avoids second-person ("you") or first-person ("I") perspective
- [ ] Summary maintains analytical tone suitable for research
- [ ] Summary doesn't include chatty asides or dramatic flourishes

---

### 3. Categories & Noise Detection

**Note:** Categories are only available when using `--use-improved-prompts` flag.

#### 3.1. Category Consistency
**Question:** Do the primary and secondary categories match your own reading of the topic?

**Good examples:**
- Topic about explicit sex: `primary_categories: ["sexual_content"]` ✅
- Topic about arguments: `primary_categories: ["conflict", "emotional"]` ✅
- Topic about dates: `primary_categories: ["romantic", "social"]` ✅

**Bad examples:**
- Topic clearly about sex labeled as `["romantic", "emotional"]` (missing sexual_content)
- Topic about arguments labeled as `["sexual_content"]` (incorrect category)

**Evaluation:**
- [ ] Primary categories accurately reflect the main theme(s) of the topic
- [ ] Secondary categories capture additional relevant themes
- [ ] Categories align with your manual reading of keywords and snippets
- [ ] Categories are consistent across similar topics

#### 3.2. Noise Detection Accuracy
**Question:** Does `is_noise` correctly identify junk topics without over-flagging real ones?

**Good examples:**
- ✅ `is_noise: true` for topics with incoherent keywords (e.g., "door", "table", "chair", "wall")
- ✅ `is_noise: false` for topics with coherent themes (e.g., "kiss", "touch", "gentle", "tender")

**Bad examples:**
- ❌ `is_noise: true` for a real topic just because keywords are abstract
- ❌ `is_noise: false` for clearly incoherent topics (random word collections)

**Evaluation:**
- [ ] Noise flag correctly identifies topics with incoherent/random keywords
- [ ] Noise flag doesn't over-flag abstract but meaningful topics
- [ ] Noise detection helps filter out topics that aren't useful for analysis

---

### 4. Stability & Format Compliance

#### 4.1. JSON Schema Compliance
**Question:** Does the model respect your JSON schema and formatting requirements?

**Good examples:**
- ✅ All fields present and correctly formatted
- ✅ Labels are 2-6 words as specified
- ✅ Categories are lists of strings
- ✅ `is_noise` is boolean

**Bad examples:**
- ❌ JSON parsing fails (model adds extra text, markdown, or invalid JSON)
- ❌ Labels exceed word limit
- ❌ Missing required fields
- ❌ Wrong data types (e.g., `is_noise` as string instead of boolean)

**Evaluation:**
- [ ] JSON output is valid and parseable for all topics
- [ ] All required fields are present
- [ ] Data types match schema (strings, lists, booleans)
- [ ] Format constraints (word limits, etc.) are respected

#### 4.2. Tone Stability
**Question:** Does the model maintain neutral, analytical tone without drifting into RP or chatty style?

**Good examples:**
- ✅ Consistent third-person, neutral language
- ✅ No second-person ("you") or first-person ("I") perspective
- ✅ No chatty asides or dramatic flourishes

**Bad examples:**
- ❌ Occasional second-person RP style ("You feel...", "You notice...")
- ❌ Chatty asides ("OMG", "lol", "so intense!")
- ❌ Overly dramatic language ("This is a beautiful moment...")

**Evaluation:**
- [ ] Model maintains consistent analytical tone across all topics
- [ ] No drift into roleplay or chatty style
- [ ] Language is suitable for research/publication

#### 4.3. Consistency Across Similar Topics
**Question:** Does the model produce consistent labeling for similar topics?

**Test:** Look at 3-5 topics with similar keywords (e.g., all about "kiss", "touch", "gentle").

**Good examples:**
- ✅ Similar topics get similar but distinct labels
- ✅ Model recognizes patterns and applies them consistently

**Bad examples:**
- ❌ Wildly different labels for very similar topics
- ❌ Inconsistent application of labeling rules

**Evaluation:**
- [ ] Similar topics receive consistent labeling (similar structure, similar specificity)
- [ ] Model applies labeling rules uniformly
- [ ] Variations in labels reflect actual differences in topics, not model inconsistency

---

## Model-Specific Expectations

### Nemo-Instruct-2407 (Baseline)

**Expected Strengths:**
- ✅ Strongest JSON compliance and format obedience
- ✅ Most neutral, academic tone
- ✅ Lowest hallucination rate
- ✅ Most consistent across topics

**Potential Weaknesses:**
- ⚠️ May be slightly less expressive than literary models
- ⚠️ May miss some genre-specific nuances
- ⚠️ Scene summaries may be more literal/less narrative

**Best For:**
- Production labeling when reliability is paramount
- Topics requiring strict format compliance
- Research contexts where neutral tone is essential

### Nemo-Gutenberg (Literary Fine-Tuning)

**Expected Strengths:**
- ✅ Better thematic abstraction ("Unclear Relationship Feelings" vs generic fluff)
- ✅ Stronger scene summary quality (more "literary" phrasing)
- ✅ Better understanding of narrative structure
- ✅ More nuanced category assignment

**Potential Weaknesses:**
- ⚠️ May skew slightly formal/classical (less modern romance-aware)
- ⚠️ May be less literal than Nemo-Instruct (slightly more abstract)

**Best For:**
- Topics requiring strong literary context understanding
- Abstract/emotional topics that need thematic abstraction
- When you want more "literary" scene summaries

### Celeste (Story-Writing/RP Model)

**Expected Strengths:**
- ✅ Most expressive and genre-aware labels
- ✅ Excellent at fine-grained scene distinctions
- ✅ Strong narrative coherence understanding
- ✅ May produce brilliant labels for some topics

**Potential Weaknesses:**
- ⚠️ May occasionally over-dramatize or push boundaries
- ⚠️ Slightly less literal (may need lower temperature)
- ⚠️ May introduce stylistic variation
- ⚠️ Test JSON compliance (RP models sometimes fight strict formatting)

**Best For:**
- Fine-grained scene distinctions (e.g., "Kitchen Argument in Morning" vs "Bedroom Argument at Night")
- Genre-aware labeling that captures romance/erotic fiction nuances
- Topics requiring narrative structure understanding

---

## Decision Framework

### Step 1: Run Comparison
```bash
python compare_models_openrouter.py \
  --api-key "$OPENROUTER_API_KEY" \
  --limit-topics 40 \
  --max-tokens 16 \
  --num-keywords 15
```

### Step 2: Inspect Selected Topics
```bash
python inspect_comparison_results.py \
  --comparison-json results/stage08_llm_labeling/comparison_models_*.json \
  --topics 3 7 15 21 29
```

Pick 5-10 topics that represent:
- Explicit sexual content
- Emotional/relationship content
- Abstract/thematic content
- Everyday/domestic content
- Edge cases (ambiguous keywords, potential noise)

### Step 3: Evaluate Using Checklist

For each topic, score each model (1-5) on:
1. Label specificity
2. Genre awareness
3. Discriminative power
4. Scene summary quality (if using improved prompts)
5. Category accuracy (if using improved prompts)
6. Noise detection (if using improved prompts)
7. JSON compliance
8. Tone stability

### Step 4: Make Decision

**Choose Nemo-Instruct-2407 if:**
- JSON compliance and reliability are your top priorities
- You need consistent, neutral labels for publication
- You're willing to trade some expressiveness for reliability

**Choose Gutenberg if:**
- You need better thematic abstraction
- Scene summaries are important and you want more "literary" phrasing
- Abstract/emotional topics are common in your corpus

**Choose Celeste if:**
- Fine-grained scene distinctions are critical
- Genre-aware labeling is more important than strict format compliance
- You can tolerate occasional stylistic variation for better expressiveness

**Consider Ensemble Approach:**
- Use Nemo-Instruct for production (reliability)
- Use Celeste/Gutenberg for specific topic types that benefit from their strengths
- Compare outputs for edge cases

---

## Quick Reference: Red Flags

**Immediate disqualifiers (choose a different model if you see these):**
- ❌ JSON parsing failures > 5% of topics
- ❌ Second-person RP style ("you", "your") in labels/summaries
- ❌ Hallucination rate > 10% (inventing events not in keywords/snippets)
- ❌ > 20% of topics share identical vague labels

**Warning signs (monitor closely):**
- ⚠️ Occasional format violations (< 5%)
- ⚠️ Slight tone drift (chatty asides, dramatic language)
- ⚠️ Over-abstraction (labels too vague to be useful)
- ⚠️ Under-specificity (labels don't distinguish similar topics)

---

## Related Documentation

- **Model Selection:** `MODEL_SELECTION_RECOMMENDATION.md` - Empirical comparison of Nemo vs. Grok
- **Model Reasoning:** `MODEL_AND_PROMPT_REASONING.md` - Detailed rationale for model choices
- **Comparison Script:** `tools/compare_models_openrouter.py` - Script to generate comparisons
- **Inspection Script:** `tools/inspect_comparison_results.py` - Script to inspect selected topics
