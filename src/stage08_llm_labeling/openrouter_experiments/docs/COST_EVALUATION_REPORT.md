# Cost Evaluation Report: Mistral Nemo Labeling

You're absolutely right: the scene summaries are getting **cut off mid-sentence**, and we can see why directly from the logs + JSON:

* `max_tokens` (completion) is only **16**

* typical summaries look like:

  * `"Characters negotiate a"`

  * `"Characters engage"`

  * `"Character uses tongue to stimulate"`

  * `"Characters suggest or accept invitations for"` 

* logs show `completion=16` for almost every call.

So the model *is* trying to follow your instruction ("one concise sentence"), but we're chopping the sentence after ~16 tokens.

Let me answer your two questions:

---

## 1️⃣ Making scene descriptions longer

### What's happening now

From your logs:

* Example topic 0:

  `tokens: prompt=2663, completion=16`

  → label & scene summary must fit inside **16 tokens**.

Most scene summaries you see in the JSON stop mid-clause exactly because of that 16-token cap.

### What to change

You don't need to change the **prompt text** for length; Nemo is already starting proper sentences. You mainly need to:

1. **Increase `max_tokens`** for completions

2. (Optionally) slightly sharpen the instruction about length.

Concrete suggestions:

* In your CLI call you used: `--max-tokens 16`

* For a full sentence, I'd go to at least:

  ```bash
  --max-tokens 48
  ```

  or even 64 if you want some headroom.

* In code, that's the `max_new_tokens` / `max_tokens` argument you already pass into `generate_label_from_keywords_openrouter`.

With `max_tokens=48`, scene summaries like:

> "Characters negotiate a sexual boundary while trying to balance desire and emotional safety in the relationship."

will *fit*, instead of being chopped at "sexual boundary".

If you want to nudge the model a bit more, you can slightly tighten the line in the **system prompt**:

> *Scene summary: one concise sentence of about 12–25 words describing the scene/theme.*

But the **primary bottleneck is token limit**, not the wording of the instruction.

---

## 2️⃣ Approximate cost to run Nemo on ALL topics

We can estimate cost using OpenRouter's current pricing for `mistralai/mistral-nemo`:

* **$0.02 per 1M input tokens**

* **$0.04 per 1M output tokens**

From your logs for this run (30 topics):

* Typical prompt size: `prompt ≈ 2,630–2,700 tokens`

* Completion: `16 tokens` (because of `max_tokens=16`)

Let's approximate with **2,660 input tokens** and **16 output tokens** per topic.

### How many topics total?

The log says:

> `Metadata says 9936 topics, but model has 368 topics.`

> So BERTopic's *current* model actually has **368 topics**.

So "run it for all topics" = 368 Nemo calls.

---

### 🔢 Step-by-step cost estimate (current settings)

Per topic:

* Input tokens ≈ 2,660

* Output tokens ≈ 16

All topics:

* Input: 2,660 × 368 = 978,880 tokens ≈ **0.98M input tokens**

* Output: 16 × 368 = 5,888 tokens ≈ **0.006M output tokens**

Cost:

* Input: 0.98M / 1M × $0.02 ≈ **$0.0196**

* Output: 0.006M / 1M × $0.04 ≈ **$0.00024**

Total ≈ **$0.0198**, i.e. **about 2 cents** to label *all 368 topics* with your current config.

Your 30-topic test run cost on the order of **0.0016 USD** (~0.16 cents).

---

### 💡 What if we make summaries longer and snippets richer?

If you:

* Increase `max_tokens` to, say, **48**, and

* Increase snippets per topic (we've been targeting up to 15 central snippets),

your **prompt size** per call might rise to, say, ~3,500–4,000 tokens, and completion to ~40–50 tokens.

Even if we assume **4,000 input** + **50 output** tokens per topic:

* Input: 4,000 × 368 ≈ 1.47M tokens → ≈ $0.0294

* Output: 50 × 368 ≈ 18,400 tokens → ≈ $0.00074

Total ≈ **$0.03–0.04** for *all topics*.

So even in the "fatter prompt + longer summary" regime, you're still only spending **a few cents** per full labeling pass with Nemo.

---

### TL;DR

* **Scene summaries are truncated because `max_tokens=16` is too low.**

  → Bump to **48–64** to get full, informative one-sentence descriptions.

* **Total cost for labeling all 368 topics with Nemo is tiny**:

  → ~**$0.02** with current settings; maybe **$0.03–$0.05** after we make prompts and summaries richer.

If you'd like, next I can:

* Propose a slightly sharper **scene_summary wording** for the system prompt adapted to 48 tokens, and

* Suggest a quick **post-processing script** to flag summaries that still look truncated (e.g., ending with "a", "to", "for").

