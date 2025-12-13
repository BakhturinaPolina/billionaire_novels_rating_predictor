# Model Selection and Prompt Structure: Design Rationale

**Date:** December 2024  
**Purpose:** Document the reasoning behind model choices and prompt architecture for BERTopic label generation in romance fiction research

---

## 1. Model Selection Rationale

### 1.1. Primary Model: Mistral-Nemo-Instruct-2407

**Choice:** `mistralai/Mistral-Nemo-Instruct-2407` as the default production model.

**Reasoning:**

1. **Instruction Following & Consistency**
   - Nemo-Instruct is specifically fine-tuned for instruction-following tasks
   - Demonstrates strong adherence to prompt constraints (length limits, format requirements, anti-hallucination rules)
   - Produces consistent, reproducible labels across similar topic inputs

2. **Literary Analysis Capability**
   - Trained on diverse text corpora including literary content
   - Handles genre-specific terminology (romance, erotic fiction) without over-interpreting
   - Balances literal interpretation with contextual understanding

3. **Research Reliability**
   - Low hallucination rate compared to creative/generative models
   - Conservative approach: prefers literal labels over speculative scenarios
   - Academic tone suitable for research publication
   - Does not invent plot elements, character relationships, or events not present in keywords/snippets

4. **Cost-Effectiveness via OpenRouter**
   - Accessible through OpenRouter API (single API key for multiple models)
   - Competitive pricing for production-scale labeling (368+ topics)
   - No local infrastructure required (pure API-based)

5. **Compatibility with Prompt Architecture**
   - Works well with structured JSON output requirements
   - Handles few-shot examples effectively
   - Responds appropriately to anti-hallucination constraints

### 1.2. Why Roleplay/Story Models Excel at Literary Tasks

**Core Insight:** Models fine-tuned for roleplay and story-writing are often excellent at literary analysis because they've been trained extensively on fiction, dialogue, and narrative coherence, not just "helpful assistant" style.

**What These Models Understand:**
- **Plot/scene patterns:** How narrative events connect and flow
- **Genre tropes:** Recognition of romance, erotic, dramatic conventions
- **Character roles and relationships:** Understanding of character dynamics
- **Scene-level distinctions:** Fine-grained differences (e.g., "Rough Angry Kisses" vs "Gentle Comforting Kisses")
- **Narrative coherence:** How scenes fit into larger story structures
- **Thematic abstraction:** Moving from concrete events to thematic labels

**The Key Constraint:** We need models that still **obey instructions and JSON formatting**, not just "sexy gremlin RP" models that ignore constraints. All models in our comparison set maintain instruction-following capabilities from their Nemo base.

### 1.3. Alternative Literary Models: Detailed Comparison

#### 1.3.1. Celeste (`nothingiisreal/mn-celeste-12b`)

**Model Family:** Nemo-based story-writing / roleplay model

**Training Data:**
- Reddit Writing Prompts (including NSFW/"Dirty & WritingPrompts" variants)
- Kalo's Opus 25k Instruct dataset
- Filtered "c2 logs" (roleplay conversation logs)

**Positioning:** Specifically engineered for story writing & RP, with strong narrative coherence and genre adaptation.

**Strengths:**
- **Narrative saturation:** Much more exposed to narrative fiction than vanilla Nemo
- **Scene pattern recognition:** Excellent at condensing plot/scene patterns into concise labels
- **Genre adaptation:** Understands romance/erotic fiction conventions
- **Character dynamics:** Maintains understanding of character roles and relationships
- **Instruct lineage:** Still has Opus Instruct 25k training, so respects JSON/formatting constraints

**Use Cases:**
- Fine-grained scene distinctions (e.g., distinguishing "Kitchen Argument in Morning" from "Bedroom Argument at Night")
- Genre-aware labeling that captures romance/erotic fiction nuances
- When you need labels that reflect narrative structure, not just keyword aggregation

**Trade-offs:**
- May introduce stylistic variation (more "literary" phrasing)
- Slightly less literal than vanilla Nemo (may need lower temperature)
- Test JSON compliance (RP models sometimes fight strict formatting)

**Recommendation:** Use for exploratory comparisons, especially for topics requiring narrative understanding. Monitor for format compliance.

#### 1.3.2. Gutenberg (`nbeerbower/mistral-nemo-gutenberg-12B-v2`)

**Model Family:** Nemo-based, DPO-tuned on Project Gutenberg corpus

**Training Data:**
- Based on `romulus-mistral-nemo-12b-simpo`
- Fine-tuned on `jondurbin/gutenberg-dpo-v0.1` (Project Gutenberg DPO dataset)
- DPO (Direct Preference Optimization) training emphasizes good narrative continuations

**Positioning:** Literary/book-like fine-tuning for digital literary studies.

**Strengths:**
- **Literary brain:** Training data is literally book-like / Gutenberg-style narrative
- **Thematic understanding:** Better sense of themes, motifs, and narrative focus
- **Abstract labels:** More "literary" abstractions (e.g., "Unclear Relationship Feelings" vs generic fluff)
- **Narrative coherence:** DPO training emphasizes coherent narrative progression
- **Stylistic alignment:** Learns to align with "good" literary continuations

**Use Cases:**
- Topics requiring strong literary context understanding
- When you need thematic abstraction (moving from concrete events to themes)
- Research contexts where "literary" tone is acceptable (scientific labels)

**Trade-offs:**
- May skew style slightly "classic" (more formal) than modern romance fiction
- Gutenberg corpus is classical literature, not modern romance/erotica
- Still relies on `ROMANCE_AWARE_SYSTEM_PROMPT` to force modern romance semantics

**Recommendation:** Test on subset of topics to evaluate if literary tuning improves label quality. Particularly useful for abstract/emotional topics.

**Alternative Variant:** `nbeerbower/Mistral-Nemo-Gutenberg-Doppel-12B-v2` (tuned on two Gutenberg DPO datasets) - may provide even stronger literary understanding.

#### 1.3.3. Starcannon (`aetherwiing/mn-starcannon-12b`) - Experimental

**Model Family:** Nemo → Celeste → merged with `mini-magnum` (another story model)

**Positioning:** Creative roleplay and story-writing model, effectively a "double-merged" story model.

**Strengths:**
- **Maximum genre flexibility:** More expressive than Celeste alone
- **Fine scene distinctions:** May excel at distinguishing subtle scene type differences
- **Natural phrasing:** Could produce more natural-sounding scene summaries

**Trade-offs:**
- **Reduced reliability:** Every extra merge tends to reduce reliability for strict tasks (JSON, deterministic behavior)
- **Format compliance risk:** Higher chance of ignoring JSON/formatting constraints
- **Experimental status:** Less tested than Celeste/Gutenberg

**Recommendation:** Second-wave experiment after testing Celeste/Gutenberg. Use only if those models show promise but need more expressiveness. Monitor closely for format compliance.

#### 1.3.4. Why Stay in the Nemo Family?

**Decision:** All comparison models are Nemo-based (Nemo-Instruct, Nemo-Celeste, Nemo-Gutenberg).

**Reasoning:**

1. **Code Compatibility**
   - Your code and prompt design already assume Nemo-like instruction behavior
   - Temperature ranges, context lengths, and response patterns are consistent
   - No need to retune prompts for different model families

2. **Infrastructure Simplicity**
   - Can swap models in OpenRouter just by changing `--model-name`
   - Keep all prompting infrastructure unchanged
   - Single API key, single codebase

3. **Instruction Following**
   - Nemo family maintains strong instruction-following even after fine-tuning
   - JSON compliance is more reliable than LLaMA/Qwen roleplay models
   - Format constraints are respected (critical for research reliability)

4. **Progressive Enhancement**
   - Start with vanilla Nemo-Instruct (baseline)
   - Add literary tuning (Gutenberg) for thematic understanding
   - Add narrative tuning (Celeste) for scene distinctions
   - All share same base, so improvements are additive, not conflicting

**Why Not Other Roleplay Models?**
- LLaMA/Qwen-based RP models may ignore JSON formatting more often
- Different context lengths require prompt adjustments
- Temperature ranges may need retuning
- Less tested for research/academic use cases

### 1.4. Model Selection Strategy

**Primary Production Model:** `mistralai/Mistral-Nemo-Instruct-2407`
- Reliable, low hallucination, format-compliant
- Use for all production labeling runs

**Comparison Models (for A/B testing):**
- `nbeerbower/mistral-nemo-gutenberg-12B-v2` - Test for abstract/thematic topics
- `nothingiisreal/mn-celeste-12b` - Test for fine-grained scene distinctions

**Experimental (future):**
- `aetherwiing/mn-starcannon-12b` - Only if Celeste shows promise but needs more expressiveness

**Workflow:**
1. Run comparison script with all three models on subset of topics (30-50 topics)
2. Inspect CSV/JSON comparison outputs
3. Identify which model performs best for which topic types
4. Optionally: Use ensemble approach (Nemo for reliability, Celeste/Gutenberg for specific topic types)

### 1.5. Why Not Local HuggingFace Models?

**Decision:** Stay API-based via OpenRouter rather than switching to local HF inference.

**Reasoning:**

1. **Infrastructure Simplicity**
   - No GPU requirements, model downloads, or memory management
   - Works on any machine with internet access
   - No version conflicts or dependency hell

2. **Model Access**
   - OpenRouter provides access to models not easily available via HF (e.g., Celeste, Gutenberg variants)
   - Single API key unlocks multiple models
   - No need to maintain separate model repositories

3. **Cost vs. Complexity Trade-off**
   - API costs are reasonable for research-scale labeling (~$0.017 per 368 topics)
   - Avoids infrastructure setup/maintenance overhead
   - Time saved on setup/debugging outweighs API costs for research use case

4. **Consistency with Existing Architecture**
   - Codebase already uses OpenAI-compatible client pattern
   - OpenRouter is OpenAI-compatible (just change `base_url`)
   - Minimal code changes required

---

## 2. Prompt Structure Rationale

### 2.1. Romance-Aware Prompt Design

**Core Principle:** The prompt must be **domain-specific** (romance/erotic fiction) while maintaining **research rigor** (no hallucination, literal interpretation).

#### 2.1.1. System Prompt Architecture

**Structure:**
```
1. Role Definition (domain context)
2. General Rules (format constraints)
3. Priority Hierarchy (what to encode in label)
4. Snippet Integration (how to use representative docs)
5. Disambiguation Requirements (avoiding label collisions)
6. Anti-Hallucination Constraints (hard rules for known patterns)
```

**Why This Structure:**

1. **Role Definition First**
   - Sets domain context immediately: "romantic and erotic fiction"
   - Prevents model from applying generic topic labeling heuristics
   - Establishes that explicit sexual terminology is acceptable (research context)

2. **Format Rules Before Content Rules**
   - Model processes constraints in order
   - Format violations are easier to catch than content violations
   - "2-6 words, no quotes, no markdown" is unambiguous

3. **Priority Hierarchy (Action → Role → Setting → Tone)**
   - Reflects how humans read romance fiction topics
   - Most important: what sexual/romantic act is happening
   - Least important: abstract emotional tone (only if clearly indicated)
   - Prevents over-interpretation of ambiguous keywords

4. **Snippet Integration as Primary Evidence**
   - Keywords are sparse and ambiguous
   - Representative document snippets provide rich context
   - Explicit instruction: "When snippets and keywords disagree, trust the snippets"
   - Prevents hallucination from keyword ambiguity (e.g., "board" + "table" → "Board Game Foreplay" hallucination)

5. **Disambiguation Requirements**
   - Prevents label collisions (multiple topics getting same vague label)
   - Forces model to encode distinguishing features explicitly
   - Example: "Physical Violence and Rage" vs "Silent Emotional Resentment"

6. **Anti-Hallucination Hard Constraints**
   - Learned from empirical testing (see `MODEL_SELECTION_RECOMMENDATION.md`)
   - Specific patterns that models hallucinate: "dinner date", "invitation", "repair", "heartbreak"
   - Explicit prohibition with conditions: "Do NOT use X unless Y is clearly present"

#### 2.1.2. User Prompt Template

**Structure:**
```
Topic keywords (most important first):
[list of keywords]

Representative snippets:
[snippet 1]
[snippet 2]
...

Generate a label following the system instructions.
```

**Why Keywords First, Then Snippets:**

1. **Keywords provide quick semantic overview**
   - Model can form initial hypothesis from keywords
   - Snippets then refine/confirm that hypothesis
   - Mimics human reading pattern: scan keywords → read snippets for detail

2. **Snippets as Refinement, Not Primary Input**
   - If snippets were first, model might over-weight single document
   - Keywords represent aggregated topic signal (from all documents)
   - Snippets provide disambiguation, not primary signal

3. **Explicit Instruction: "Generate a label following the system instructions"**
   - Reinforces that format/content rules apply
   - Prevents model from adding explanatory text
   - Ensures JSON output when using improved prompts

### 2.2. Improved Prompts: JSON Output Structure

**Extension:** When `--use-improved-prompts` is enabled, prompt requests structured JSON output:

```json
{
  "label": "2-6 word noun phrase",
  "scene_summary": "12-25 word scene description",
  "categories": ["category1", "category2"],
  "is_noise": false,
  "rationale": "brief explanation of label choice"
}
```

**Why JSON Output:**

1. **Structured Data for Downstream Analysis**
   - Enables programmatic filtering (e.g., `is_noise == true`)
   - Categories can be used for topic grouping/clustering
   - Rationale provides interpretability for research

2. **Enforced Format Compliance**
   - JSON parsing fails if model adds extra text
   - Forces model to follow exact structure
   - Reduces need for post-processing/regex extraction

3. **Multi-Field Information in Single API Call**
   - More efficient than separate calls for label + metadata
   - Consistent model "state" across all fields (no drift between calls)

**Trade-offs:**
- Some models may struggle with strict JSON formatting
- Requires robust JSON parsing with fallback to text extraction
- Slightly more complex prompt (but worth it for structured output)

### 2.3. Few-Shot Examples (When Used)

**Pattern:** Include 2-3 example topic → label pairs in prompt.

**Why Few-Shot:**

1. **Demonstrates Format Expectations**
   - Model sees concrete examples of "good" labels
   - Reduces need for verbose format instructions
   - Especially helpful for JSON output structure

2. **Domain-Specific Examples**
   - Shows how to handle explicit sexual terminology
   - Demonstrates literal vs. abstract interpretation balance
   - Illustrates disambiguation patterns

**When Not Used:**
- For very long prompts (token limit concerns)
- When prompt already has extensive rules (redundant)
- For models that don't benefit from examples (some instruction-tuned models)

### 2.4. Anti-Hallucination Constraints: Empirical Basis

**Source:** Testing on 30 diverse topics revealed consistent hallucination patterns.

**Patterns Identified:**

1. **"Dinner Date" / "Invitation" Hallucination**
   - **Trigger:** Keywords include "dinner", "restaurant", "food"
   - **Hallucination:** Model infers "invitation" or "date" even when keywords don't mention asking/inviting
   - **Fix:** "Do NOT use 'dinner date' or 'invitation' unless snippets/keywords explicitly mention asking/inviting"

2. **"Repair" Hallucination**
   - **Trigger:** Keywords include "car", "engine", "tool"
   - **Hallucination:** Model infers "repair" even when keywords are about car travel, not mechanics
   - **Fix:** "Do NOT use 'repair' unless keywords/snippets include mechanical terms like 'fix', 'mechanic', 'repair', 'tools'"

3. **"Heartbreak" / "Breakup" Hallucination**
   - **Trigger:** Keywords include emotional terms ("hurt", "pain", "cry")
   - **Hallucination:** Model infers relationship ending even when context is about ongoing conflict
   - **Fix:** "Do NOT use 'heartbreak' or 'breakup' unless emotional pain in relationship ending is clearly described"

**Why Hard Constraints Work Better Than Soft Guidance:**

- Soft guidance ("prefer not to...") is often ignored by creative models
- Hard constraints ("Do NOT use X unless Y") are more likely to be followed
- Explicit conditions (Y) prevent over-correction (model can still use X when appropriate)

---

## 3. Integration with BERTopic Workflow

### 3.1. Why POS-Filtered Keywords?

**Approach:** Extract only nouns, verbs, adjectives from topic keywords (via spaCy POS tagging).

**Reasoning:**

1. **Removes Noise**
   - Function words ("the", "and", "of") don't help with labeling
   - Pronouns ("he", "she", "it") are ambiguous without context
   - Prepositions/conjunctions add little semantic value

2. **Focuses on Content Words**
   - Nouns: entities, objects, body parts, actions (as nouns)
   - Verbs: actions, sexual acts, emotional states
   - Adjectives: descriptors, emotional tone

3. **Reduces Token Count**
   - Fewer tokens in prompt = lower API cost
   - Faster processing
   - Less chance of hitting context limits

**Trade-off:**
- May lose some context (e.g., "against wall" → "wall" loses spatial relationship)
- But snippets compensate for this loss

### 3.2. Representative Document Snippets

**Source:** BERTopic's `get_representative_docs()` method (typically 3-5 docs per topic).

**Why Snippets Are Critical:**

1. **Disambiguation**
   - Keywords: ["kiss", "rough", "angry"]
   - Without snippets: ambiguous (romantic? violent? playful?)
   - With snippets: "He grabbed her and kissed her roughly, anger evident in his movements" → "Rough Angry Kisses"

2. **Prevents Hallucination**
   - Keywords: ["board", "table", "chair"]
   - Without snippets: model might infer "Board Game Foreplay" (hallucination)
   - With snippets: "She sat at the table, placing the board game pieces" → "Board Game Setup" (literal)

3. **Scene Context**
   - Keywords don't encode setting, emotional tone, or explicit acts
   - Snippets provide this missing context
   - Enables more precise labels: "Kitchen Argument in Morning" vs. "Argument"

**Implementation:**
- Extract snippets during topic extraction (one-time cost)
- Include in prompt for each topic
- Model uses snippets as primary evidence (per prompt instruction)

### 3.3. Streaming vs. Batch Processing

**Streaming Mode:** When `--topics-json` is provided, process topics incrementally and write labels to disk as they're generated.

**Why Streaming:**

1. **Memory Efficiency**
   - For 368+ topics, storing all labels in memory can be large
   - Streaming writes incrementally, reducing peak memory

2. **Fault Tolerance**
   - If process crashes, already-written labels are preserved
   - Can resume from last written topic (with some modification)

3. **Progress Visibility**
   - Can monitor progress by checking JSON file size
   - Useful for long-running jobs

**Batch Mode:** Load all topics, generate all labels, save at once.

**When to Use:**
- Small topic sets (< 100 topics)
- When you need all labels in memory for post-processing
- Simpler code path (no file I/O during generation)

---

## 4. Model Parameters: Temperature and Reasoning Effort

### 4.1. Temperature Selection

**Default:** `temperature=0.35` (balanced)

**Reasoning:**

1. **Too Low (0.0-0.2):**
   - Overly deterministic, may miss valid alternative phrasings
   - Can get stuck in repetitive patterns
   - Good for consistency but may lack nuance

2. **Too High (0.7-1.0):**
   - Excessive variation, inconsistent labels for similar topics
   - May violate format constraints more often
   - Good for creativity but bad for research reproducibility

3. **Sweet Spot (0.3-0.4):**
   - Allows natural phrasing variation without excessive randomness
   - Maintains consistency for similar topics
   - Balances literal interpretation with natural language

**Model-Specific Tuning:**
- Nemo-Instruct: Works well at 0.35 (default)
- Celeste (creative model): May benefit from lower temp (0.2-0.3) to reduce stylistic drift
- Gutenberg: Test at 0.3-0.4 (similar to Nemo)

### 4.2. Reasoning Effort (Gemini Models)

**Parameter:** `extra_body['reasoning']['effort']` for models like `google/gemini-2.5-flash`.

**Options:** `"none"`, `"low"`, `"medium"`, `"high"`.

**When to Use:**

1. **None (Default):**
   - Fastest, lowest cost
   - Sufficient for straightforward topic labeling
   - Use for production runs

2. **Low/Medium:**
   - When topics are ambiguous or require deeper analysis
   - May improve label quality for abstract/emotional topics
   - Trade-off: slower, more expensive

3. **High:**
   - Rarely needed for topic labeling (overkill)
   - Use only for research experiments on reasoning quality

**Recommendation:** Start with `"none"`, only increase if label quality is insufficient.

---

## 5. Cost and Performance Considerations

### 5.1. API Cost Estimation

**Per-Topic Cost (approximate):**
- Nemo-Instruct-2407: ~$0.00005 per topic (varies by prompt length)
- For 368 topics: ~$0.018 total
- Celeste/Gutenberg: Similar pricing (check OpenRouter pricing page)

**Why Cost Is Acceptable:**
- One-time labeling cost (labels are saved to BERTopic model)
- No infrastructure maintenance overhead
- Research budget typically allows for API costs
- Time saved on setup/debugging > API cost

### 5.2. Rate Limiting

**OpenRouter Rate Limits:**
- Varies by model and account tier
- Default: ~4 requests/second (conservative)
- Can increase with higher tier or by model

**Implementation:**
- Current code uses 4.0s delay between API calls (conservative)
- Can reduce if rate limits allow
- Streaming mode helps with long-running jobs (can pause/resume)

### 5.3. Latency

**Per-Topic Latency:**
- Nemo-Instruct: ~1-3 seconds per topic (depends on prompt length, model load)
- For 368 topics: ~6-18 minutes total (sequential processing)

**Optimization Options:**
- Parallel processing (multiple topics concurrently) - not currently implemented
- Batch API calls (if OpenRouter supports) - check API docs
- Caching: Reuse labels for unchanged topics (implemented via `existing_labels` parameter)

---

## 6. Future Improvements

### 6.1. Model-Specific Prompt Tuning

**Opportunity:** Different models may benefit from slightly different prompt structures.

**Examples:**
- Celeste: Emphasize literal interpretation (reduce stylistic drift)
- Gutenberg: Leverage literary context understanding
- Gemini: Use reasoning effort for ambiguous topics

**Implementation:**
- Add model-specific prompt variants in `generate_labels_openrouter.py`
- Select based on `model_name` parameter

### 6.2. Few-Shot Example Selection

**Current:** Static few-shot examples (if used).

**Improvement:** Dynamic few-shot selection based on topic similarity.

**Approach:**
- For each topic, find 2-3 most similar labeled topics
- Include their (topic, label) pairs as examples
- Helps model understand context-specific labeling patterns

**Challenge:** Requires labeled topics first (chicken-and-egg), but can be iterative.

### 6.3. Label Quality Validation

**Current:** Manual inspection of comparison CSVs.

**Improvement:** Automated quality metrics.

**Metrics to Consider:**
- Label length distribution (should be 2-6 words)
- Label uniqueness (no duplicates for distinct topics)
- Keyword coverage (label should reference top keywords)
- Snippet alignment (label should match snippet content)

**Implementation:**
- Post-processing script that validates labels against these criteria
- Flags topics that may need manual review

---

## 7. References and Related Documentation

- **Prompt Templates:** `docs/prompts.md` - Detailed prompt structure and examples
- **Model Comparison:** `docs/MODEL_SELECTION_RECOMMENDATION.md` - Empirical evaluation of Nemo vs. Grok
- **Snippets Logic:** `docs/SNIPPETS_LOGIC.md` - How representative documents are extracted and used
- **Cost Analysis:** `docs/COST_EVALUATION_REPORT.md` - API cost breakdown and optimization
- **OpenRouter Docs:** https://openrouter.ai/docs - API reference and model information

---

## 8. Summary: Key Design Decisions

1. **Model:** Nemo-Instruct-2407 (default) for reliability, with Celeste/Gutenberg as alternatives
2. **Prompt:** Romance-aware, snippet-integrated, anti-hallucination constraints
3. **Format:** JSON output for structured data (when improved prompts enabled)
4. **Keywords:** POS-filtered (nouns, verbs, adjectives) to reduce noise
5. **Snippets:** Primary evidence source (keywords are secondary)
6. **Temperature:** 0.35 (balanced for consistency + natural phrasing)
7. **Infrastructure:** API-based (OpenRouter) for simplicity and model access
8. **Processing:** Streaming mode for large topic sets (memory efficiency)

These choices prioritize **research reliability** (low hallucination, literal interpretation) while maintaining **practical efficiency** (API-based, structured output, fault tolerance).
