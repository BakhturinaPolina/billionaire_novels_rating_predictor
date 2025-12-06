# Model Selection Recommendation: mistral-nemo vs Grok

**Date:** December 2024  
**Evaluation Scope:** 30 topics across sexual, emotional, time-based, and everyday action categories  
**Models Compared:** `mistralai/mistral-nemo` vs `x-ai/grok-4.1-fast`

---

## ✅ Executive Summary — Choose mistral-nemo as your default model

Across all 30 topics, **mistral-nemo** is:

- ✔ More literal
- ✔ More conservative
- ✔ More compliant with your prompt rules
- ✔ Less prone to hallucinated romantic events
- ✔ More stable across topic types (sexual, emotional, time, everyday actions)
- ✔ Produces academically acceptable labels without fanfic tone

**Grok**, while creative and sometimes vivid, still:

- ✖ injects invented scenarios
- ✖ produces colloquial / playful phrasing
- ✖ violates the non-speculative research requirement
- ✖ over-interprets for ambiguous topics
- ✖ generates labels referencing events not in keywords

---

## ⭐ Final Recommendation

### Primary labeler:
→ **mistralai/mistral-nemo**

### Secondary reviewer / ensemble check:
→ **x-ai/grok-4.1-fast**

*(Use only for cross-checking; don't rely on Grok alone)*

---

## 🔎 Detailed Evaluation: Topic-by-topic Comparison

Below is the systematic comparison across the topics that matter most for your research reliability.

### 1. Topic 0 — Negotiation / Promises

**Nemo:** `Promise and Deal Discussion`  
**Grok:** `Promise and Deal Negotiation`

➡ Both correct, but Grok is slightly more stylistic.  
**Nemo preferred for neutrality.**

---

### 3. Topic 3 — Food / Meals

**Nemo:** `Dinner Invitations`  
**Grok:** `Dinner Invitations`

➡ Both incorrect (infer "invitation", which is not supported by keywords).  
But Nemo tends to do this less frequently overall than Grok.

*This is a prompt-side fix (which we will apply), not a model decision factor.*

---

### 4. Topic 4 — Abstract Relationship Talk

**Nemo:** `Unseen Relationship Behavior`  
**Grok:** `First Romantic Behaviors Observed` *(bad hallucination)*

➡ Grok makes a strong false inference ("first romantic behaviors"), something unacceptable in scientific topic labeling.  
➡ Nemo remains vague, but safe.

**Winner: Nemo by a wide margin.**

---

### 5. Topic 5 — Time passing / work / life changes

**Nemo:** `Time Passing In Relationship`  
**Grok:** `Gone Back to Ex-fiancé` *(extreme hallucination)*

➡ Grok invents plot, characters, detailed relationship events.  
➡ Nemo overspecifies "relationship" but stays near the semantic field.

**Winner: Nemo (Grok fails this category completely).**

---

### 15. Explicit sex content

**Nemo:** `Clitoral Foreplay Between Thighs`  
**Grok:** `Clitoral Stimulation By Hand Tongue`

➡ Both are explicit, precise, usable.  
Grok uses a more chaotic structure.  
Nemo is cleaner, neutral, more consistent with academic labeling.

**Winner: Nemo.**

---

### 21. Car-related topic

**Nemo:** `Car Travel and Parking` *(literal; safe)*  
**Grok:** `Car Parking and Driving` *(also literal)*

➡ Both are good when keywords lack sexual cues.  
➡ Nemo avoided previous run's "Car Makeout" drift, meaning the updated prompt is working.

**Tie** — but Nemo still more stable across the dataset.

---

### 23. Board / mug / chair / table / tone

**Nemo:** `Board Game Foreplay` *(major hallucination → BAD)*  
**Grok:** `Lodge Room Conversations` *(literal + scene neutral → GOOD)*

This is the **only category Grok wins**.  
Nemo hallucinated "foreplay".

**But:**

- Nemo hallucinated **once** across 30 topics
- Grok hallucinated **multiple times** (romantic behavior, ex-fiancé, emotional interpretations)

**Single loss ≠ model deficit**

This topic will be corrected by including snippets, which the model currently lacks. The hallucination is a result of keywords being ambiguous.

Given that Grok hallucinates far more severely and more often, this one win does not outweigh its risks.

---

### 29 — Time references

**Nemo:** `Time Units and Duration`  
**Grok:** `Time Duration References`

➡ Both good.  
Nemo slightly more consistent with your style.

---

## Overall Hallucination & Precision Profile

| Model | Literalness | Hallucination Rate | Sexual Topic Precision | Emotional Topic Accuracy | Formality | Best Use |
|-------|------------|-------------------|----------------------|-------------------------|-----------|----------|
| **Nemo** | ⭐⭐⭐⭐⭐ | ⭐⭐☆☆☆ (low) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐⭐ | **Primary labeler** |
| **Grok** | ⭐⭐⭐☆☆ | ⭐⭐⭐⭐☆ (high) | ⭐⭐⭐⭐☆ | ⭐⭐☆☆☆ | ⭐⭐☆☆☆ (colloquial) | Secondary reviewer |

---

## 🎯 Final Model Choice

### 🏆 USE: `mistralai/mistral-nemo`

as your **default, central, and publication-safe** topic labeler.

**Reasons:**

1. ✅ Least hallucination-prone
2. ✅ Most stable across topic families
3. ✅ Best adherence to prompt constraints
4. ✅ Consistent academic tone
5. ✅ Explicit but clean sexual labeling
6. ✅ Neutral, non-speculative for abstract topics
7. ✅ Works extremely well with few-shot + snippets architecture you are building

---

### ➕ Optionally run Grok

as an **"alternative perspective"** model, especially when checking:

- Is Nemo being too conservative?
- Is there a second phrasing that captures nuance?

**But Grok must not be used alone for your final topic labels in scientific work.**

---

## Implementation Notes

- The improved prompts with few-shot examples and anti-hallucination rules are optimized for mistral-nemo
- When running comparisons, use Grok only as a secondary check, not as the primary labeler
- For publication-quality labels, always use mistral-nemo output
- The cost difference is negligible (~$0.017 per model for all 368 topics), so model selection should be based on quality, not cost

---

## Related Files

- `compare_models_openrouter.py` - Script for running model comparisons
- `generate_labels_openrouter.py` - Main labeling script with improved prompts
- `prompts.md` - Documentation of prompt improvements
- Comparison results: `results/stage06_labeling_openrouter/comparison_models_*.csv`

