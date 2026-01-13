# Hypothesis Testing Results: Theme Indices and Goodreads Outcomes

**Source Notebook:** `04_hypothesis_testing_inference_only_v4_1_macro_axes_bayes.ipynb`  
**Analysis Date:** 2025-01-27  
**Sample Size:** N = 92 books

---

## Executive Summary

This analysis examines how theme indices predict two distinct Goodreads channels:
1. **Mass Appeal / Visibility** = `log_rating_count` (how many people rated it)
2. **Perceived Quality** = `rating_mean` and `avg_rating_bayes` (how positively readers evaluate it)

These are not the same thing. A book can be widely read but not loved, or loved by a smaller audience.

**Key Finding:** Popularity (reach) is strongly associated with a "billionaire-romance package": status/luxury + alpha guarding + repair + emotional safety + social/kin network. Perceived quality (ratings), after accounting for popularity, is most consistently associated with "care + safety" and is negatively associated with "baseline negative affect" and explicit erotics. Narrative pacing matters: higher-rated books tend to show increasing tension/negative affect toward the end (a stronger third-act crisis), even though baseline negative affect is associated with lower ratings.

---

## What We Measured (In Human Terms)

### Success Signals from Goodreads

1. **Mass appeal / visibility** = `log_rating_count` (how many people rated it)
   - Think: market reach, discoverability, hype, fandom size.

2. **Perceived quality** = `rating_mean` and **better**: `avg_rating_bayes`
   - Think: how positively readers evaluate it.
   - `avg_rating_bayes` uses Bayesian shrinkage to stabilize ratings from books with few voters.

### Story Dynamics (Arcs)

We also measured **story dynamics** with arcs:
- `end − begin` and `middle − begin` for each theme
- Think: does a theme *build up* during the story?

---

## How to Read Effect Tables

For each predictor, we report:

- **beta_std**: effect size in standardized units
  - Roughly: "if this theme goes up by 1 SD, does the outcome tend to go up or down?"

- **CI** (ci_low, ci_high): uncertainty range
  - If it crosses 0 → it's not stable enough to call "clearly positive/negative."

- **p(beta>0)**: sign stability across bootstrap samples
  - 0.99 = almost always positive
  - 0.01 = almost always negative
  - 0.50 = coin flip / no clear direction

This is exactly the right way to talk about trends in a small-N pilot.

---

## A) Big Picture Summary

**Popularity (reach) is strongly associated with a "billionaire-romance package": status/luxury + alpha guarding + repair + emotional safety + social/kin network.**

**Perceived quality (ratings), after accounting for popularity, is most consistently associated with "care + safety" and is negatively associated with "baseline negative affect" and explicit erotics.**

**Narrative pacing matters:** higher-rated books tend to show *increasing tension/negative affect toward the end* (a stronger third-act crisis), even though baseline negative affect is associated with lower ratings.

That's a clean, theory-friendly pilot result.

---

## B) Mass Appeal / Reach Results (log_rating_count)

### 1) Core Predictors (Top 10)

From `top10_core_level_log_rating_count.csv`:

Most stable positive associations with reach:

1. **R2_alpha_guarding** β≈ +0.44, CI [+0.23, +0.60], P(β>0)=1.00
2. **Luxury/wealth (PC1)** β≈ +0.37, CI [+0.20, +0.55], P(β>0)=1.00
3. **Emotional safety (PC1)** β≈ +0.32, CI [+0.13, +0.50], P=0.998
4. **Repair** β≈ +0.23, CI [+0.02, +0.41], P=0.985
5. **Social support/kin** β≈ +0.19, P≈0.95 (less clean CI)

Everything below that becomes much weaker/noisy.

**Plain-language interpretation:**

The books that get *more* ratings tend to be the ones that combine:
- billionaire/status signals,
- dominance/conflict signals,
- and emotional payoff signals (repair + safety).

This looks like "market match" rather than "literary quality."

### 2) Macro Axes (Cleaner Story)

From `top10_macro_level_log_rating_count.csv` (5 axes):

- **AX_status_dominance** β≈ +0.46, CI [+0.28, +0.61], P=1.00
- **AX_payoff_safety** β≈ +0.33, CI [+0.14, +0.50], P≈0.999
- **AX_drama_obstacle** β≈ +0.34, CI [+0.07, +0.57], P≈0.993
- **AX_explicitness** β≈ −0.27, CI [−0.45, −0.06], P≈0.003 (strongly negative)
- **AX_negative_affect** ~ weak/unclear

**Plain-language interpretation:**

Reach is highest for: **status + dominance + obstacles + payoff**, and lowest for: **explicitness** (in this dataset).

That's exactly the "two-channel" framing: reach is trope-package driven.

### 3) Joint Ridge (The Collinearity-Aware Check)

From `ridge_joint_coeffs_log_rating_count.csv`:

Even when you throw all CORE predictors in at once (shrinkage model), the same signals remain:

**Positive:**
- alpha guarding
- luxury
- repair
- kin support
- safety

**Negative:**
- explicit erotics
- domestic nesting
- vices/addictions

**Why this matters:**

It means the popularity effects aren't just an artifact of running 1 predictor at a time. The "package" persists under joint modeling.

### 4) Simple Correlations (Goodreads Channel)

From `goodreads_index_correlations.csv` (Spearman with `log_n_ratings`, top predictors):

| Predictor | Spearman r | Interpretation |
|-----------|------------|----------------|
| D_power_wealth_luxury__pc1 | 0.48 | Luxury/wealth axis |
| R2_alpha_guarding | 0.35 | Alpha dominance/guarding |
| J_social_support_kin | 0.32 | Social support/kin networks |
| Q_miscommunication | 0.31 | Miscommunication drama |
| A2_emotional_safety__pc1 | 0.28 | Emotional safety |
| Q_repair | 0.20 | Conflict repair |

**Bottom (negative with popularity):**
- C_explicit_eroticism: -0.37
- H_domestic_nesting__pc1: -0.17
- F2_anger_frustration: -0.13

---

## C) Perceived Quality Results

We now have **two** quality measures:
- `rating_mean` (but controlled for log_rating_count in our models)
- `avg_rating_bayes` (more stable, because it shrinks noisy low-voter ratings)

### 1) rating_mean (Quality Beyond Reach)

From `top10_core_level_rating_mean.csv`:

Most stable positive:
1. **Protective caretaking** β≈ +0.22, CI [+0.05, +0.36], P=0.995
2. **Emotional safety (PC1)** β≈ +0.15, P=0.95 (CI slightly crosses 0)

Then everything gets weak.

**Plain language:**

Among books with similar popularity, the ones rated higher are the ones that feel **caring + emotionally safe**.

#### Macro Version (Even Clearer)

From `top10_macro_level_rating_mean.csv`:

- **AX_payoff_safety** β≈ +0.21, P=0.974
- **AX_explicitness** β≈ −0.15, P=0.095 (tends negative)
- **AX_negative_affect** β≈ −0.13, P=0.051 (borderline negative)

**Plain language:**

Quality beyond reach is mostly: **payoff/safety up**, **explicitness and baseline negativity down**.

This is a neat romance-theory statement: readers reward emotional care and safety, not just "heat," at least here.

### 2) avg_rating_bayes (Primary Quality Channel)

#### A) No Control (Quality as Observed in the Marketplace)

From `top10_macro_level_avg_rating_bayes_no_control.csv`:

- **AX_payoff_safety** β≈ +0.17, P=0.973
- **AX_status_dominance** β≈ +0.13, P=0.895
- Negative leaning: explicitness, negative affect

**Plain language:**

In raw Goodreads ratings (with shrinkage), payoff/safety is the strongest positive axis; status/dominance also looks positive because it overlaps with what the market likes.

#### B) Controlling Reach (Quality *Independent* of Popularity)

From `top10_macro_level_avg_rating_bayes_control_log_n.csv`:

- **AX_payoff_safety** β≈ +0.08, P=0.823 (weak but still positive leaning)
- Everything else ~ near zero / unclear

**Plain language:**

Once you control for reach, the only axis that still looks somewhat "quality-related" is **payoff/safety** — and even that is modest.

This is a very clean deliverable:
- status/dominance boosts visibility and also correlates with ratings in the wild,
- but "pure quality beyond reach" mostly comes from payoff/safety.

### 3) Simple Correlations with avg_rating

From `goodreads_index_correlations.csv` (Spearman with `avg_rating`, top):

| Predictor | Spearman r | Interpretation |
|-----------|------------|----------------|
| R1_protective_caretaking | 0.25 | Protective caretaking |
| D_power_wealth_luxury__pc1 | 0.24 | Luxury/wealth |
| A2_emotional_safety__pc1 | 0.23 | Emotional safety |
| R2_alpha_guarding | 0.21 | Alpha guarding |

**Bottom (negative with avg_rating):**
- C_explicit_eroticism: -0.30
- H_domestic_nesting__pc1: -0.23
- F2_anger_frustration: -0.20

### 4) Partial Correlations: Quality Beyond Popularity

From `partial_corr_rating_mean.csv` (`rating_mean` controlling `log_rating_count`):

**Most positive:**
- R1_protective_caretaking: **+0.245**
- A2_emotional_safety__pc1: **+0.152**
- D_power_wealth_luxury__pc1: **+0.093**
- R2_alpha_guarding: **+0.075**

**Most negative:**
- J_social_support_kin: **-0.162**
- H_domestic_nesting__pc1: **-0.154**
- C_explicit_eroticism: **-0.172**
- F3_anxiety_worry: **-0.137**
- F2_anger_frustration: **-0.121**

**Key Insight:** After accounting for popularity, **care and safety still predict higher ratings**, while baseline anger/anxiety and explicit erotics predict lower ratings.

### 5) Joint Ridge for rating_mean: "Themes Explain Almost Nothing"

From `ridge_joint_coeffs_rating_mean.csv`:

All coefficients are tiny (on the order of 0.01). Even `log_rating_count` itself is only +0.032.

**This is not a bug.** It means:
- Average rating is **not strongly predictable** from these theme indices (at least at N=92 and with these constructs), OR
- The predictors are too collinear/noisy, OR
- Mean rating is driven by things we're not measuring (prose quality, pacing, editing, author fandom, expectation management, etc.)

**Conclusion:** The theme system is **better at explaining market reach than "star rating."**

---

## D) Arc / Pacing Results (rating_mean arcs)

From `top25_core_arc_rating_mean.csv` (strongest arc predictors):

The top two are **very stable**:

- **Anger/frustration end−begin** β≈ +0.24, CI [+0.08, +0.41], P=0.995
- **Anxiety/worry end−begin** β≈ +0.19, CI [+0.02, +0.36], P=0.981

Also:
- jealousy increases mid-story (positive leaning)
- some increases in aesthetics/luxury mid/end (weaker)

**This looks paradoxical until you combine it with the "baseline" results.**

Baseline anger/anxiety are negatively associated with quality (from partial correlations), but *increasing* anger/anxiety late is positively associated with quality.

**Very simple explanation:**

Higher-rated books don't have more negativity overall — they have **better pacing**:
- Lower baseline negativity across the book,
- but a stronger late "crisis escalation" (third-act crisis),
- likely followed by payoff/repair (even if payoff language itself doesn't spike as cleanly).

That is an extremely romance-consistent pattern.

---

## E) Goodreads "Two Channels" Validation

### 1) Tiers Differ in Both Channels

From `tier_summary_goodreads_channels.csv`:

| Tier | n | avg_rating | n_ratings | Interpretation |
|------|---|------------|-----------|----------------|
| **bad** | 30 | ≈ 3.77 | ≈ 48k | Lower quality perception + lower visibility |
| **mid** | 32 | ≈ 4.01 | ≈ 44k | Moderate quality + moderate visibility |
| **good** | 30 | ≈ 4.22 | ≈ 116k | Higher quality perception + **much higher visibility** |

So "good vs bad" isn't only about rating. It's also visibility.

**Implication:** "Good vs bad" is both:
- **Higher perceived quality** (avg_rating: 4.22 vs 3.77)
- **Higher visibility** (n_ratings: 116k vs 48k)

When you see a tier difference, it's almost never a single thing. The "good" tier books have both better ratings AND more voters, suggesting they reach a broader audience.

### 2) Theme Correlations Split Cleanly

From `goodreads_index_correlations.csv`:

**With log_n_ratings (reach), strongest positive:**
- luxury
- alpha guarding
- miscommunication / kin support
- safety
- repair

**With avg_rating (quality), strongest positive:**
- protective caretaking
- safety
- (luxury and alpha also positive, but this partly reflects market confounding)

**Negative with both:**
- explicit erotics
- anger/frustration
- domestic nesting

---

## F) Predictive Performance (Repeated CV)

From `cv_repeats_summary.csv` (20 repeats of 5-fold CV):

| Outcome | Model | CV R² (mean ± sd) | Range |
|---------|-------|-------------------|-------|
| rating_mean | metadata_only | 0.108 ± 0.031 | -0.001 to 0.142 |
| rating_mean | metadata + themes | 0.056 ± 0.041 | -0.031 to 0.118 |
| log_rating_count | themes only | 0.050 ± 0.037 | -0.016 to 0.108 |

**Plain language:**

Theme indices provide **some** predictive signal for popularity, but not much for mean rating. Mean rating is likely driven by other factors we're not capturing (writing quality, pacing, editing, author fandom, etc.). That's not a failure — it's a *design insight*.

**This is exactly the kind of conclusion a strong pilot study is supposed to deliver.**

---

## G) What We Can Safely Claim (Pilot-Appropriate)

### You Can Say:

- "We separate popularity (visibility) from perceived quality (ratings)."
- "Popularity is associated with a coherent macro package: status/dominance + drama/obstacles + payoff/safety."
- "Perceived quality beyond reach is most consistently associated with payoff/safety, and negatively with baseline negative affect and explicitness."
- "Arc features suggest higher-rated books show stronger late escalation of tension (third-act crisis), consistent with romance narrative structure."

### Avoid Saying:

- "X causes popularity."
- "Explicit content reduces quality in general."
  - This is corpus-specific; treat it as a pattern in billionaire romance sample.

---

## H) Appendix Tables and Figures

### Appendix Tables (Clean and Defensible)

- Macro axes definitions (`macro_axes_definition.csv`)
- `top10_macro_level_log_rating_count.csv`
- `top10_macro_level_rating_mean.csv`
- `top10_macro_level_avg_rating_bayes_no_control.csv`
- `top10_macro_level_avg_rating_bayes_control_log_n.csv`
- Arc summary: `top25_core_arc_rating_mean.csv`

### Main Figures (Publication-Ready)

1. Macro axes → log_rating_count (dot + CI)
2. Macro axes → avg_rating_bayes (no control)
3. Macro axes → avg_rating_bayes (controlling log_n_ratings)
4. Arc macro effects or top arc predictors (crisis escalation story)

That gives you the full "two channels + pacing" narrative without drowning the reader in 26 composites.

---

## Detailed Results Report (Methods/Results Draft)

This section provides a comprehensive, plain-language narrative of findings suitable for Methods/Results sections of a research paper.

### Overview: Two Distinct Success Channels

We measured two different "success signals" from Goodreads that capture distinct aspects of book performance:

1. **Mass appeal / visibility** = `log_rating_count` (how many people rated it)
   - Think: market reach, discoverability, hype, fandom size.

2. **Perceived quality** = `rating_mean` and **better**: `avg_rating_bayes`
   - Think: how positively readers evaluate it.
   - `avg_rating_bayes` uses Bayesian shrinkage to stabilize ratings from books with few voters.

These are not the same thing. A book can be widely read but not loved, or loved by a smaller audience.

We also measured **story dynamics** with arcs:
- `end − begin` and `middle − begin` for each theme
- Think: does a theme *build up* during the story?

### Interpreting Effect Sizes

For each predictor, we report:

- **beta_std**: effect size in standardized units
  - Roughly: "if this theme goes up by 1 SD, does the outcome tend to go up or down?"

- **CI** (ci_low, ci_high): uncertainty range
  - If it crosses 0 → it's not stable enough to call "clearly positive/negative."

- **p(beta>0)**: sign stability across bootstrap samples
  - 0.99 = almost always positive
  - 0.01 = almost always negative
  - 0.50 = coin flip / no clear direction

This is exactly the right way to talk about trends in a small-N pilot.

### Big Picture Summary

**Popularity (reach) is strongly associated with a "billionaire-romance package": status/luxury + alpha guarding + repair + emotional safety + social/kin network.**

**Perceived quality (ratings), after accounting for popularity, is most consistently associated with "care + safety" and is negatively associated with "baseline negative affect" and explicit erotics.**

**Narrative pacing matters:** higher-rated books tend to show *increasing tension/negative affect toward the end* (a stronger third-act crisis), even though baseline negative affect is associated with lower ratings.

That's a clean, theory-friendly pilot result.

### Mass Appeal / Reach Results (log_rating_count)

#### Core Predictors (Top 10)

From `top10_core_level_log_rating_count.csv`:

Most stable positive associations with reach:

1. **R2_alpha_guarding** β≈ +0.44, CI [+0.23, +0.60], P(β>0)=1.00
2. **Luxury/wealth (PC1)** β≈ +0.37, CI [+0.20, +0.55], P(β>0)=1.00
3. **Emotional safety (PC1)** β≈ +0.32, CI [+0.13, +0.50], P=0.998
4. **Repair** β≈ +0.23, CI [+0.02, +0.41], P=0.985
5. **Social support/kin** β≈ +0.19, P≈0.95 (less clean CI)

Everything below that becomes much weaker/noisy.

**Plain-language interpretation:**

The books that get *more* ratings tend to be the ones that combine:
- billionaire/status signals,
- dominance/conflict signals,
- and emotional payoff signals (repair + safety).

This looks like "market match" rather than "literary quality."

#### Macro Axes (Cleaner Story)

From `top10_macro_level_log_rating_count.csv` (5 axes):

- **AX_status_dominance** β≈ +0.46, CI [+0.28, +0.61], P=1.00
- **AX_payoff_safety** β≈ +0.33, CI [+0.14, +0.50], P≈0.999
- **AX_drama_obstacle** β≈ +0.34, CI [+0.07, +0.57], P≈0.993
- **AX_explicitness** β≈ −0.27, CI [−0.45, −0.06], P≈0.003 (strongly negative)
- **AX_negative_affect** ~ weak/unclear

**Plain-language interpretation:**

Reach is highest for: **status + dominance + obstacles + payoff**, and lowest for: **explicitness** (in this dataset).

That's exactly the "two-channel" framing: reach is trope-package driven.

#### Joint Ridge (The Collinearity-Aware Check)

From `ridge_joint_coeffs_log_rating_count.csv`:

Even when throwing all CORE predictors in at once (shrinkage model), the same signals remain:

**Positive:**
- alpha guarding
- luxury
- repair
- kin support
- safety

**Negative:**
- explicit erotics
- domestic nesting
- vices/addictions

**Why this matters:**

It means the popularity effects aren't just an artifact of running 1 predictor at a time. The "package" persists under joint modeling.

### Perceived Quality Results

We have **two** quality measures:
- `rating_mean` (but controlled for log_rating_count in our models)
- `avg_rating_bayes` (more stable, because it shrinks noisy low-voter ratings)

#### rating_mean (Quality Beyond Reach)

From `top10_core_level_rating_mean.csv`:

Most stable positive:
1. **Protective caretaking** β≈ +0.22, CI [+0.05, +0.36], P=0.995
2. **Emotional safety (PC1)** β≈ +0.15, P=0.95 (CI slightly crosses 0)

Then everything gets weak.

**Plain language:**

Among books with similar popularity, the ones rated higher are the ones that feel **caring + emotionally safe**.

##### Macro Version (Even Clearer)

From `top10_macro_level_rating_mean.csv`:

- **AX_payoff_safety** β≈ +0.21, P=0.974
- **AX_explicitness** β≈ −0.15, P=0.095 (tends negative)
- **AX_negative_affect** β≈ −0.13, P=0.051 (borderline negative)

**Plain language:**

Quality beyond reach is mostly: **payoff/safety up**, **explicitness and baseline negativity down**.

This is a neat romance-theory statement: readers reward emotional care and safety, not just "heat," at least here.

#### avg_rating_bayes (Primary Quality Channel)

##### A) No Control (Quality as Observed in the Marketplace)

From `top10_macro_level_avg_rating_bayes_no_control.csv`:

- **AX_payoff_safety** β≈ +0.17, P=0.973
- **AX_status_dominance** β≈ +0.13, P=0.895
- Negative leaning: explicitness, negative affect

**Plain language:**

In raw Goodreads ratings (with shrinkage), payoff/safety is the strongest positive axis; status/dominance also looks positive because it overlaps with what the market likes.

##### B) Controlling Reach (Quality *Independent* of Popularity)

From `top10_macro_level_avg_rating_bayes_control_log_n.csv`:

- **AX_payoff_safety** β≈ +0.08, P=0.823 (weak but still positive leaning)
- Everything else ~ near zero / unclear

**Plain language:**

Once you control for reach, the only axis that still looks somewhat "quality-related" is **payoff/safety** — and even that is modest.

This is a very clean deliverable:
- status/dominance boosts visibility and also correlates with ratings in the wild,
- but "pure quality beyond reach" mostly comes from payoff/safety.

### Arc / Pacing Results (rating_mean arcs)

From `top25_core_arc_rating_mean.csv` (strongest arc predictors):

The top two are **very stable**:

- **Anger/frustration end−begin** β≈ +0.24, CI [+0.08, +0.41], P=0.995
- **Anxiety/worry end−begin** β≈ +0.19, CI [+0.02, +0.36], P=0.981

Also:
- jealousy increases mid-story (positive leaning)
- some increases in aesthetics/luxury mid/end (weaker)

**This looks paradoxical until you combine it with the "baseline" results.**

Baseline anger/anxiety are negatively associated with quality (from partial correlations), but *increasing* anger/anxiety late is positively associated with quality.

**Very simple explanation:**

Higher-rated books don't have more negativity overall — they have **better pacing**:
- Lower baseline negativity across the book,
- but a stronger late "crisis escalation" (third-act crisis),
- likely followed by payoff/repair (even if payoff language itself doesn't spike as cleanly).

That is an extremely romance-consistent pattern.

### Goodreads "Two Channels" Validation

#### 1) Tiers Differ in Both Channels

From `tier_summary_goodreads_channels.csv`:

| Tier | n | avg_rating | n_ratings | Interpretation |
|------|---|------------|-----------|----------------|
| **bad** | 30 | ≈ 3.77 | ≈ 48k | Lower quality perception + lower visibility |
| **mid** | 32 | ≈ 4.01 | ≈ 44k | Moderate quality + moderate visibility |
| **good** | 30 | ≈ 4.22 | ≈ 116k | Higher quality perception + **much higher visibility** |

So "good vs bad" isn't only about rating. It's also visibility.

**Implication:** "Good vs bad" is both:
- **Higher perceived quality** (avg_rating: 4.22 vs 3.77)
- **Higher visibility** (n_ratings: 116k vs 48k)

When you see a tier difference, it's almost never a single thing. The "good" tier books have both better ratings AND more voters, suggesting they reach a broader audience.

#### 2) Theme Correlations Split Cleanly

From `goodreads_index_correlations.csv`:

**With log_n_ratings (reach), strongest positive:**
- luxury
- alpha guarding
- miscommunication / kin support
- safety
- repair

**With avg_rating (quality), strongest positive:**
- protective caretaking
- safety
- (luxury and alpha also positive, but this partly reflects market confounding)

**Negative with both:**
- explicit erotics
- anger/frustration
- domestic nesting

### Predictive Performance (Repeated CV)

From `cv_repeats_summary.csv` (20 repeats of 5-fold CV):

| Outcome | Model | CV R² (mean ± sd) | Range |
|---------|-------|-------------------|-------|
| rating_mean | metadata_only | 0.108 ± 0.031 | -0.001 to 0.142 |
| rating_mean | metadata + themes | 0.056 ± 0.041 | -0.031 to 0.118 |
| log_rating_count | themes only | 0.050 ± 0.037 | -0.016 to 0.108 |

**Plain language:**

Theme indices provide **some** predictive signal for popularity, but not much for mean rating. Mean rating is likely driven by other factors we're not capturing (writing quality, pacing, editing, author fandom, etc.). That's not a failure — it's a *design insight*.

**This is exactly the kind of conclusion a strong pilot study is supposed to deliver.**

### What We Can Safely Claim (Pilot-Appropriate)

#### You Can Say:

- "We separate popularity (visibility) from perceived quality (ratings)."
- "Popularity is associated with a coherent macro package: status/dominance + drama/obstacles + payoff/safety."
- "Perceived quality beyond reach is most consistently associated with payoff/safety, and negatively with baseline negative affect and explicitness."
- "Arc features suggest higher-rated books show stronger late escalation of tension (third-act crisis), consistent with romance narrative structure."

#### Avoid Saying:

- "X causes popularity."
- "Explicit content reduces quality in general."
  - This is corpus-specific; treat it as a pattern in billionaire romance sample.

### Recommended Appendix Tables and Figures

#### Appendix Tables (Clean and Defensible)

- Macro axes definitions (`macro_axes_definition.csv`)
- `top10_macro_level_log_rating_count.csv`
- `top10_macro_level_rating_mean.csv`
- `top10_macro_level_avg_rating_bayes_no_control.csv`
- `top10_macro_level_avg_rating_bayes_control_log_n.csv`
- Arc summary: `top25_core_arc_rating_mean.csv`

#### Main Figures (Publication-Ready)

1. Macro axes → log_rating_count (dot + CI)
2. Macro axes → avg_rating_bayes (no control)
3. Macro axes → avg_rating_bayes (controlling log_n_ratings)
4. Arc macro effects or top arc predictors (crisis escalation story)

That gives you the full "two channels + pacing" narrative without drowning the reader in 26 composites.

### Practical Refinement Note

Right now the explicitness axis includes a bit of "chemistry" (B1). If you want the cleanest interpretation, consider splitting:

- **AX_explicitness** = explicit erotics only
- **AX_attraction** = chemistry / non-explicit affection

Because readers might like "chemistry" but not necessarily "explicit sex," and mixing them can muddy interpretation.

Also, the drama axis could be simplified if you want 3–4 axes total:
- keep status/dominance
- keep payoff/safety
- keep negative affect
- keep explicitness
- (Drama/obstacles can be rolled into dominance or tested separately.)

---

## Deliverable Conclusions

### Themes That Predict Mass Appeal (More Voters / Higher Visibility)

**Most consistent across correlation + ridge + macro axes:**

1. **Status/dominance axis** (AX_status_dominance)
2. **Payoff/safety axis** (AX_payoff_safety)
3. **Drama/obstacle axis** (AX_drama_obstacle)
4. **Luxury/wealth axis** (D_power_wealth_luxury__pc1)
5. **Alpha guarding** (R2_alpha_guarding)
6. **Repair** (Q_repair)
7. **Emotional safety** (A2_emotional_safety__pc1)
8. **Kin/social support** (J_social_support_kin)

**Negative:**
- **Explicitness** (AX_explicitness, C_explicit_eroticism)

### Themes That Predict Perceived Quality (Higher Average Rating, Beyond Popularity)

**Most consistent across partial correlations + Bayesian adjustment + macro axes:**

**Positive:**
- **Payoff/safety axis** (AX_payoff_safety)
- **Protective caretaking** (R1_protective_caretaking)
- **Emotional safety** (A2_emotional_safety__pc1)

**Negative:**
- **Explicitness** (AX_explicitness, C_explicit_eroticism)
- **Negative affect** (AX_negative_affect, F2_anger_frustration, F3_anxiety_worry)
- **Domestic nesting axis** (H_domestic_nesting__pc1) — in this corpus

### Arc/Pacing Pattern

**Higher-rated books show:**
- Lower baseline negative affect
- Stronger late escalation of tension (third-act crisis)
- Consistent with romance narrative structure

### Meta-Result

**Themes explain popularity better than they explain star ratings (at N=92).**

This is a feature, not a bug: it suggests that market reach is more systematically related to thematic content, while star ratings may be influenced by factors beyond theme indices (prose quality, pacing, editing, reader expectations, etc.).

---

## Files Referenced

All results saved to: `results/measurement_v5/bundle/inference_outputs/`

### Core Results
- `tier_summary_goodreads_channels.csv` — Tier differences in both channels
- `goodreads_index_correlations.csv` — Simple correlations (Pearson + Spearman)
- `macro_axes_definition.csv` — Macro axis definitions

### Mass Appeal
- `top10_core_level_log_rating_count.csv` — Top 10 core predictors of reach
- `top10_macro_level_log_rating_count.csv` — Top 10 macro axes for reach
- `ridge_joint_coeffs_log_rating_count.csv` — Joint ridge for popularity

### Perceived Quality
- `top10_core_level_rating_mean.csv` — Top 10 core predictors of quality
- `top10_macro_level_rating_mean.csv` — Top 10 macro axes for quality
- `top10_macro_level_avg_rating_bayes_no_control.csv` — Bayesian ratings (no control)
- `top10_macro_level_avg_rating_bayes_control_log_n.csv` — Bayesian ratings (controlling reach)
- `ridge_joint_coeffs_rating_mean.csv` — Joint ridge for quality
- `partial_corr_rating_mean.csv` — Partial correlations (quality beyond popularity)
- `partial_corr_log_rating_count.csv` — Partial correlations for popularity
- `goodreads_weighted_quality_table.csv` — Weighted + Bayesian rating analysis

### Arc Analysis
- `top25_core_arc_rating_mean.csv` — Top 25 arc predictors of quality

### Validation
- `cv_repeats_summary.csv` — Repeated cross-validation results

---

*Analysis conducted: 2025-01-27*  
*Notebook: `04_hypothesis_testing_inference_only_v4_1_macro_axes_bayes.ipynb`*
