# Theoretical Reasoning: Representative Snippets for Topic Labeling

This document explains the theoretical reasoning and design decisions behind using representative document snippets in the OpenRouter labeling pipeline.

## Problem Statement

Keyword-based topic labeling suffers from several limitations:

1. **Ambiguity**: Keywords like "mouth, tongue, suck" could mean many things (eating, talking, oral sex)
2. **Vagueness**: Without context, models default to generic labels like "Erotic Intimacy"
3. **Missing nuance**: Cannot distinguish "rough kiss" from "gentle kiss", "kitchen argument" from "general anger"
4. **Over-generalization**: Single outlier words can skew labels (e.g., random city name)

## Solution: Representative Snippets

By including actual sentence-level excerpts from the topic's documents, we provide the LLM with:

- **Scene-level context**: What is actually happening, not just what words appear
- **Tone and style**: Emotional valence, intensity, setting
- **Precise acts**: Specific sexual acts, not generic "intimacy"
- **Pattern recognition**: Shared narrative patterns across multiple examples

## Corpus Statistics

Our corpus characteristics inform the design:

- **680,822 sentences** (each is a BERTopic "document")
- **Mean sentence length ≈ 12.5 tokens**
- **Chapters ~200 sentences on average**

These statistics support using 6-8 short snippets:
- 6 sentences × 12.5 tokens = ~75 tokens (very manageable)
- Even 8 sentences ≈ 100 tokens of raw content
- Modern chat models (4k–8k context) can easily handle this

## Design Decisions

### Why 6 Snippets?

**Token Budget Analysis:**
- 6 sentences × ~12.5 tokens = ~75 tokens
- With formatting overhead: ~100-120 tokens total
- This is <3% of a 4k context window
- Leaves plenty of room for system prompt, keywords, and response

**Information Density:**
- 6 sentences provide enough examples to see a pattern
- Fewer than 6 risks missing important variations
- More than 6 provides diminishing returns and increases token cost
- 6 is the sweet spot for pattern recognition without overwhelming the model

**Empirical Justification:**
- In romance/erotica, scene-level distinctions are usually clear in 4-6 sentences
- Multiple examples help distinguish outliers from core patterns
- Enough context to see setting, tone, and specific acts

### Why 200 Characters Per Snippet?

**Sentence Length:**
- Average sentence: ~12.5 tokens ≈ 50-60 characters
- 200 characters ≈ 3-4 average sentences
- Most sentences fit comfortably within 200 chars
- Truncation at word boundaries prevents mid-word cuts

**Practical Considerations:**
- Long sentences (>200 chars) are rare in this corpus
- Truncation preserves readability
- Word-boundary truncation maintains sentence integrity

### Why Representative Documents?

**Centrality vs Randomness:**
- Representative docs are chosen by BERTopic for their centrality to the topic
- They are more informative than random documents
- They capture the "core" of what the topic is about
- Random docs might include edge cases or noise

**BERTopic's Selection:**
- BERTopic uses c-TF-IDF and similarity metrics to select representative docs
- These are documents that best exemplify the topic
- They are more likely to show the shared pattern we want to label

**Alternative Approaches Considered:**
- Random sampling: Less reliable, may miss core patterns
- All documents: Too many tokens, includes noise
- Top keywords only: Loses context and scene-level understanding

## How Snippets Improve Label Precision

### Example 1: Generic → Specific

**Without Snippets:**
- Keywords: "mouth, tongue, suck, lips"
- Label: "Erotic Intimacy" or "Oral Intimacy"

**With Snippets:**
- Snippets show: kneeling, taking into mouth, head movement, hair guidance
- Label: "Blowjob in Bed" or "Oral Sex Scene"

**Improvement:** Specific act identified, setting included

### Example 2: Ambiguity Resolution

**Without Snippets:**
- Keywords: "kitchen, angry, voice, raised"
- Label: "Angry Conversation" or "Conflict"

**With Snippets:**
- Snippets show: kitchen setting, argument about dinner, slammed dishes
- Label: "Kitchen Argument" or "Meal Conflict"

**Improvement:** Setting-specific, more precise

### Example 3: Tone Distinction

**Without Snippets:**
- Keywords: "kiss, lips, tender, soft"
- Label: "Tender Kiss" or "Intimate Kiss"

**With Snippets:**
- Snippets show: gentle, slow, lingering, eyes closed
- Label: "Gentle Kissing" or "Tender Foreplay"

**Improvement:** Tone captured more accurately

## Scene-Level vs Keyword-Level Understanding

### Keyword-Level (Before)
- Sees: "mouth, tongue, suck"
- Infers: Something involving mouth and tongue
- Labels: Generic "Oral Intimacy"

### Scene-Level (After)
- Sees: "She knelt between his thighs, taking him into her mouth"
- Understands: Specific act, position, context
- Labels: "Blowjob in Bed"

**Key Difference:** Snippets provide narrative context, not just word associations.

## Fallback Strategy

The implementation includes robust fallback:

1. **Try `get_representative_docs()` method** (newer BERTopic versions)
2. **Fallback to `representative_docs_` attribute** (older versions)
3. **Handle both dict and list return types**
4. **If snippets unavailable**: Works with keywords only (backward compatible)

This ensures the system works across BERTopic versions and gracefully degrades if representative docs are missing.

## Token Cost Analysis

**Per-Topic Cost:**
- System prompt: ~800 tokens
- Keywords: ~50 tokens
- POS cues: ~30 tokens
- Snippets (6 × 12.5): ~75 tokens
- User prompt overhead: ~100 tokens
- **Total: ~1055 tokens per topic**

**Cost Comparison:**
- Without snippets: ~980 tokens per topic
- With snippets: ~1055 tokens per topic
- **Increase: ~75 tokens (7.6%)**

**Justification:**
- Small token increase for significant label quality improvement
- Still well within context limits
- Cost is justified by precision gains

## Neutral, Scientific Labeling

The system prompt explicitly instructs:

- "Do NOT euphemize clearly explicit sexual acts"
- "Keep wording neutral and non-romanticized"
- "This is for scientific analysis, not for readers' enjoyment"

This addresses the problem of vague, poetic labels like "Erotic Intimacy" by forcing factual, scene-level descriptions.

## Limitations and Future Work

### Current Limitations

1. **Snippet Selection**: Relies on BERTopic's representative doc selection (may miss important variations)
2. **Truncation**: Long sentences are truncated, potentially losing context
3. **Token Cost**: Adds ~75 tokens per topic (acceptable but not free)

### Potential Improvements

1. **Diversity-Aware Selection**: Use MMR or similar to ensure snippet diversity
2. **Adaptive Snippet Count**: More snippets for complex topics, fewer for simple ones
3. **Snippet Reranking**: Prioritize snippets that best distinguish this topic from similar ones
4. **Multi-Representation**: Include snippets from different document types if available

## Conclusion

Representative snippets provide a cost-effective way to dramatically improve label precision by giving the LLM scene-level context. The design choices (6 snippets, 200 chars, representative docs) balance information density, token cost, and practical implementation constraints.

The theoretical foundation is sound: scene-level understanding beats keyword-level associations for precise, neutral labeling in narrative text.

## References

- BERTopic documentation on representative documents
- Sentence-level corpus statistics from EDA
- Token budget analysis for modern LLMs (4k-8k context windows)
- Information theory: pattern recognition requires multiple examples

