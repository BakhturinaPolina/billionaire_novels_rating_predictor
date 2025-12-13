# Model Comparison Report: 5-Topic Evaluation Run

**Date:** 2025-12-10  
**Comparison File:** `comparison_models_20251210_215446.json`  
**Models Evaluated:** 6 (mistralai/Mistral-Nemo-Instruct-2407, thedrummer/rocinante-12b, thedrummer/cydonia-24b-v4.1, thedrummer/anubis-70b-v1.1, thedrummer/skyfall-36b-v2, thedrummer/unslopnemo-12b)  
**Topics:** 5 topics (0-4)

---

## Executive Summary

**Short version:**

* **Nemo-Instruct is the safest "research default."**
* **Cydonia + Anubis look like the best literary/RP helpers** (for nuance and romance-awareness).
* **Skyfall and Unslopnemo are usable but a bit noisier.**
* **Rocinante shows one real red flag in this sample.**

---

## 1. Label Quality

### Specificity

Across topics 0–4:

* **mistralai/Mistral-Nemo-Instruct-2407**
  * Strong on the explicit / concrete stuff:
    * Topic 1: *"Intimate Breast And Nipple Foreplay"*
    * Topic 2: *"Clitoral Stimulation During Foreplay"*
  * Topic 0 label *"Negotiating Deal"* is slightly vague (doesn't say "relationship"), but categories fix it.
  * Overall: **high specificity**, occasionally a bit dry.

* **thedrummer/cydonia-24b-v4.1**
  * Topic 0: *"Negotiating Relationship Terms"* – better than Nemo's "Negotiating Deal" because it bakes the relationship into the label.
  * Topics 1–3 equally clear and concrete.
  * Overall: **excellent label specificity**, slightly more romance-aware phrasing than Nemo.

* **thedrummer/anubis-70b-v1.1**
  * Very similar to Cydonia, also *"Negotiating Relationship Terms"*, and very clear sex labels on topics 1–2.
  * Topic 1 is a bit softer (*"Passionate Mouth And Body Contact"*), but still clear.
  * Overall: **high specificity, mildly more euphemistic in one place.**

* **thedrummer/skyfall-36b-v2**
  * Often specific, but:
    * Topic 0: *"Negotiating A Deal"* with **setting:office** – that's likely wrong given the romance context and other models' agreement on relationship talk.
  * So: **good specificity, but one setting hallucination.**

* **thedrummer/unslopnemo-12b**
  * Topic 0: *"Uncertain Feelings About Relationship"* – focuses on emotion instead of the "deal/negotiation" that keywords suggest. Not wrong, but less aligned with the "deal" flavour.
  * Very explicit and specific on topics 1–2.
  * Overall: **fine, but sometimes tilts to emotional abstraction instead of concrete action.**

* **thedrummer/rocinante-12b**
  * Topics 0–3 have solid, specific labels.
  * Topic 4: *"Never Seen Before"* with **empty categories and rationale** → this is not specific at all and basically fails the label task for that topic.
  * Overall: **inconsistent**—some great labels, one real dud.

**Winner on specificity (in this sample):**

* **Cydonia & Anubis**, with **Nemo** very close behind (slightly less explicit about "relationship" in topic 0).

---

### Genre Awareness (romance/erotic fiction)

Looking for: distinction between romance core, sexual content, domestic scenes, emotional uncertainty.

* **Nemo**
  * Uses `romance_core` + `relationship_conflict` + `sexual_content` in reasonable ways.
  * Correctly marks the clit/pussy topic as explicit `sexual_content`.
  * Genre-aware enough, but not fancy.

* **Cydonia & Anubis**
  * Very clearly distinguish:
    * Negotiation about relationship vs. sex vs. dates.
    * Topic 0 explicitly framed as *relationship terms*, which is very romance-aware.
  * Handle explicit sexual content correctly (`sexual_content`, `sexual:oral_sex`, `sexual:clitoral_stimulation`).

* **Skyfall**
  * Labels and categories show good genre awareness, but that "office" setting in topic 0 feels like it's leaning into a trope that isn't supported by the keywords.

* **Unslopnemo**
  * Strong on sexual topics.
  * Topic 0 label highlights emotional uncertainty; that's romance-aware, but it downplays the "deal/negotiation" aspect.

* **Rocinante**
  * Topic 1: *"Intimate Kissing And Caressing"* with `romance_core` + `physical_affection` – nicely romance-coded.
  * But topic 4's label + missing categories means it basically drops the ball on an emotional/relationship topic—bad for genre consistency.

**Best here:** **Cydonia & Anubis**, then **Nemo**. Skyfall is okay-but-quirky, Unslopnemo fine, Rocinante uneven.

---

## 2. Scene Summary Quality

Per your criteria: micro-scene, concrete detail, neutral-ish tone.

* **Nemo**
  * Short, literal, and very scene-focused:
    * Topic 2: *"She spreads her legs as he uses his tongue to stimulate her clit."*
  * Concrete, zero drama. Great for research.

* **Cydonia & Anubis**
  * Slightly richer but still controlled:
    * Cydonia topic 2: *"He uses his tongue to stroke her clit while his fingers spread her lips."*
    * Anubis topic 2: *"She spreads her legs as he drags his tongue over her clit in one long stroke."*
  * Very visual, clear micro-scenes, still in third person and not chatty.

* **Skyfall & Unslopnemo**
  * Also do fine: very similar concrete, scene-level descriptions.
  * No obvious RP drift in this sample.

* **Rocinante**
  * Also scene-based; e.g. topic 2 includes *"licks and sucks her clit"* – quite explicit but still descriptive rather than RP.
  * The problem is not tone; it's the missing rationale/categories on topic 4, which makes that summary feel contextless.

Overall, **all models pass the "micro-scene / concrete detail / neutral grammar" tests**, with **Nemo a bit more clinical** and **Cydonia/Anubis more richly descriptive but still research-usable.**

---

## 3. Categories & Noise

You're using `primary_categories`, `secondary_categories`, and `is_noise`.

* **Noise detection**
  * All topics are correctly `is_noise: false`. These are clearly coherent topics, so that's fine.

* **Category accuracy & consistency**
  * **Nemo, Cydonia, Anubis, Skyfall, Unslopnemo**:
    * Sex topics: correctly `sexual_content` (plus clitoral/oral tags where appropriate).
    * Date topics: `romance_core` + `social_setting` + restaurant/dining categories.
    * Emotional-topic (4): they all give `romance_core`, plus some relationship_conflict / relationship:unclear style tags.
    * This is exactly the kind of stable category behaviour you want for analysis.

  * **Rocinante**:
    * Topics 0–3 have decent categories.
    * Topic 4: `primary_categories: []`, `secondary_categories: []`, empty rationale → category failure.

**So: Rocinante is the only one with a clear category-quality red flag.** The others look solid and consistent across similar topics.

---

## 4. Stability & Format Compliance

Per your criteria, disqualifiers would be JSON failures, second-person RP, etc.

From this sample:

* All six models:
  * Use valid JSON structure (strings, lists, booleans).
  * Maintain third-person, neutral-ish tone.
  * Do not drift into "you/your" RP narration.

* **Rocinante**:
  * The empty categories + empty rationale in topic 4 are **within schema**, but they *violate the spirit* of "all fields meaningfully filled in." It's not an outright JSON error, but it's a quality/stability issue.

Given your red-flag list, nothing here is an "immediate disqualifier," but Rocinante is flirting with the *"over-abstraction / under-specificity / missing metadata"* warning zone.

---

## 5. Overall Recommendation (for this comparison run)

Putting it all together:

### Best single model for **production research labeling**

* **🏆 mistralai/Mistral-Nemo-Instruct-2407**

Why:

* Most predictable, literal, and format-obedient.
* Very good label specificity on explicit/delicate content.
* Neutral tone that will look respectable when quoted in a methods appendix.
* No obvious genre mistakes; just occasionally a bit dull or slightly generic for abstract emotional topics.

This matches how your criteria doc imagines the "baseline instruct" role: boring but trustworthy.

### Best "literary / genre-aware assistant" models

* **🥈 thedrummer/cydonia-24b-v4.1**
* **🥈 thedrummer/anubis-70b-v1.1**

Why:

* Labels like *"Negotiating Relationship Terms"* show more romance-savvy abstraction than Nemo while staying concrete.
* Scene summaries are rich but still neutral enough for academic use.
* Categories are accurate and consistent in this sample.

I'd treat these as:

* Great for **fine-grained distinctions** and **nuanced emotional/sexual labeling**.
* Use them to refine or cross-check Nemo's labels on tricky topics.

### "Okay but watch them" models

* **thedrummer/skyfall-36b-v2**
  * Generally good, but that office-setting hallucination on topic 0 is a hint that it might lean into ungrounded tropes sometimes. I'd want to see more topics before trusting it fully.

* **thedrummer/unslopnemo-12b**
  * Totally usable. Slight tendency to emphasise emotions (topic 0) instead of the concrete frame ("deal/negotiation"), which you may or may not like depending on your analysis focus.

### Model with a clear warning flag

* **thedrummer/rocinante-12b**
  * Capable of nice labels, but topic 4's **"Never Seen Before" + missing categories + missing rationale** is exactly the kind of weird inconsistency that makes downstream analysis annoying.
  * I would *not* pick it as a primary research model based on this behaviour.

---

### Practical strategy

Given your own decision framework, the natural setup from this run is:

* Use **Nemo-Instruct** as the **primary, production** labeler.

* Use **Cydonia or Anubis** as a **secondary literary model** to:
  * Re-label a subset of topics where emotional nuance or relationship structure matters most.
  * Compare where they diverge from Nemo to identify particularly rich or ambiguous topics.

That way you get: stability for the quantitative work, and genre-sensitive nuance where your humanities brain wants to dig deeper.

---

## Appendix: Comparison Run Details

**Script:** `compare_models_openrouter.py`  
**Command:** Run with `--use-improved-prompts` flag  
**Topics Processed:** 5 (topics 0-4)  
**Output Files:**
- `comparison_models_20251210_215446.json` (full structured data)
- `comparison_models_20251210_215446.csv` (side-by-side comparison)

**Models Compared:**
1. `mistralai/Mistral-Nemo-Instruct-2407` - Baseline Nemo Instruct model
2. `thedrummer/rocinante-12b` - Literary/RP model designed for engaging storytelling and rich prose
3. `thedrummer/cydonia-24b-v4.1` - 24B model
4. `thedrummer/anubis-70b-v1.1` - 70B model
5. `thedrummer/skyfall-36b-v2` - 36B model
6. `thedrummer/unslopnemo-12b` - 12B model
