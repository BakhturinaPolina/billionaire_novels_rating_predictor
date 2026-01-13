# Topic-Level Analysis: Micro-Scenes That Distinguish Top vs Trash Billionaire Romances

## Slide 1: Title Slide

**Topic-Level Landscape Across Popularity Tiers**

Micro-Scenes That Distinguish "Top" vs "Trash" Billionaire Romances

Stage 10: Correlation Analysis  
Topic-Level EDA

---

## Slide 2: Research Question

**What micro-scene patterns distinguish highly-rated from poorly-rated billionaire romance novels?**

- **Granularity:** Individual BERTopic probabilities (368 topics)
- **Comparison:** Top / Middle / Trash tiers (n=30/32/30 books)
- **Goal:** Identify specific scene types that differ across tiers

**Why topic-level?**
- Bridge between macro themes and finest narrative granularity
- Provides exemplars for macro-level patterns
- Reveals *which* specific scenes contribute to tier differences

---

## Slide 3: Data & Methods Overview

**Data:**
- 92 books × 368 topics = 33,856 book-topic probability pairs
- Normalized probabilities (sum to ~1.0 per book)
- Topic labels from Stage 08 LLM labeling

**Methods:**
- **Effect size:** Cliff's delta (Top vs Trash)
- **Significance:** Kruskal-Wallis, Mann-Whitney U (FDR-corrected)
- **Filtering:** Two-gate rule
  - Gate 1: |δ| ≥ 0.20 (effect)
  - Gate 2: mass ≥ 0.002 OR |mean diff| ≥ 0.001 (impact)

**Why two-gate?** With n=30 per tier, p-values are underpowered. Effect sizes + impact gates provide practical interpretability.

---

## Slide 4: Topic Health Landscape

**Key Finding: Most topics appear in most books**

- Median prevalence: **0.924** (92% of books have topic probability > 0.001)
- This is expected under soft topic assignment

**Implication:** 
- Naive "topic presence" is less informative
- Must use **effect sizes** and **mass thresholds**
- Discriminative analysis goes beyond simple frequency

**Top topics by mass:**
1. Intimate Breast And Nipple Kissing (0.0198)
2. Relationship Ambiguity Conversation (0.0172)
3. Work-related Presentation (0.0131)

---

## Slide 5: Two-Gate Filtering Results

**85 discriminative topics identified** (from 342 total)

**Two-tier structure:**

**Tier 1 (High Confidence):** 8 topics
- |δ| ≥ 0.35 AND raw p < 0.05 AND both gates
- 7 Top ↑, 1 Trash ↑

**Tier 2 (Exploratory):** 85 topics
- |δ| ≥ 0.20 AND both gates (no p-value filter)
- 70 Top ↑, 15 Trash ↑

**Filtering gates:**
- Gate 1 (Effect): 163 topics passed
- Gate 2 (Impact): 186 topics passed
- **Both gates: 85 topics** ← Final set

---

## Slide 6: Top-Associated Topics (Top ↑)

**Tier 1 High-Confidence Examples:**

| Topic | |Cliff's Δ| | Taxonomy | Radway Phase |
|-------|------------|----------|-----------|
| Married Couple's Affectionate Stares | 0.453 | Sexuality, Attraction & Intimacy | Turning Point & Recognition |
| Frightened Admissions | 0.420 | Emotions, Cognition & Inner Life | Initial Conflict & Isolation |
| Emotional Relationship Delusion | 0.404 | Emotions, Cognition & Inner Life | Initial Conflict & Isolation |
| Lip Biting During Intimacy | 0.364 | Sexuality, Attraction & Intimacy | Turning Point & Recognition |

**Pattern:** Two main types
1. **Psychological credibility** (fear, admissions, self-deception)
2. **Embodied intimacy cues** (affectionate stares, lip biting, shared joy)

---

## Slide 7: Trash-Associated Topics (Trash ↑)

**Tier 1 High-Confidence Example:**

| Topic | |Cliff's Δ| | Taxonomy | Radway Phase |
|-------|------------|----------|-----------|
| Dominatrix Session | 0.353 | Sexuality, Attraction & Intimacy | Commitment & Restoration |

**Top 5 Trash ↑ Topics:**
1. Dominatrix Session (δ = -0.353)
2. Protective Conversations By Firelight (δ = -0.349)
3. Deflecting Blame (δ = -0.333)
4. Work At Desk (δ = -0.329)
5. Awkward Questions At Door (δ = -0.322)

**Pattern:** Three main types
1. **Explicit sexual content** (BDSM, explicit erotics)
2. **Procedural/transition scenes** (doors, phones, desk work)
3. **Relationship-conflict mechanics** (deflecting blame, reluctant talk)

---

## Slide 8: Interpretation: What Kinds of Micro-Scenes Differ?

**Top books** are differentiated by:
- ✅ **Psychological credibility** (fear admissions, emotional delusion)
- ✅ **Embodied intimacy cues** (affectionate stares, lip biting)
- ✅ Often map to **Initial Conflict & Isolation** or **Turning Point & Recognition**

**Trash books** are differentiated by:
- ❌ **Explicit erotics** (dominatrix sessions, explicit sexual acts)
- ❌ **Procedural filler** (doors, phones, desk work)
- ❌ Often map to **Commitment & Restoration** or **Not a narrative function**

**Key insight:** Top books emphasize emotional truth and embodied intimacy; trash books emphasize explicit content and scene scaffolding.

---

## Slide 9: Topic-to-Hypothesis Mapping

**Hypothesis shortlists (from Tier 2 filtered set):**

- **H1_Explicit:** 12 topics (explicit sexual content)
- **H2_Commitment:** 25 topics (commitment/trust/relationship-definition)
- **H3_LuxuryWork:** 9 topics (luxury/work/business)
- **H4_Emotional:** 40 topics (emotional/feelings/conversation)
- **H5_Conflict:** 39 topics (argument/conflict/struggles)
- **S_Procedure:** 15 topics (scene scaffolding/structural)

**Overlap:** Many topics map to multiple hypotheses (e.g., "Frightened Admissions" appears in both H4_Emotional and H5_Conflict).

**Use case:** Provides micro-scene exemplars for macro-level hypothesis testing.

---

## Slide 10: Author Dominance Control

**Key Finding: Author style is a measurable confounder**

- **30 topics** show high author dominance (>50% from single author)
- **12 topics** show medium dominance (30-50%)

**Examples:**
- Doorway Questions And Answers (100% Meghan_Quinn)
- Confused About Relationship Feelings (100% Stella_Rhys)
- Parental Warning About Engagement (100% Ana_Huang)

**Methodological win:** Objective way to say "We controlled for author-style topics"

**Action items:**
- Filter author-dominant topics before tier interpretation
- Include author fixed/random effects in modeling
- Document separately as "author signature topics"

---

## Slide 11: Alignment with Macro-Level Findings

**Micro-scene patterns match macro-axis inference:**

✅ **Reach** → status/dominance packages  
✅ **Quality beyond reach** → payoff/safety  
✅ **Arcs** → "late crisis escalation" patterns

**Specific alignments:**
- Top-tier "psychological credibility" scenes → payoff/safety index
- Top-tier "embodied intimacy" → emotional coherence
- Trash-tier "explicit erotics" → explicitness separates popularity/quality channels
- Trash-tier "procedural filler" → pacing/structure differences

**Value:** Topic-level analysis provides **exemplars** of macro patterns at finest granularity.

---

## Slide 12: Limitations & Caveats

**Critical caveat:**

> "Certain micro-scenes are systematically more emphasized in one tier than another, with consistent directional effects under nonparametric comparisons."

**Not:** "Topic X causes popularity"

**Why?**
- BERTopic probabilities are soft and dense
- Differences are small in absolute magnitude
- Effect sizes are meaningful, but not causal

**Other limitations:**
- Statistical power: n=30 per tier → FDR-corrected p-values underpowered
- Author confounding: 30 topics show high author dominance
- Label interpretation: Topic labels are imperfect summaries

---

## Slide 13: Key Takeaways

**1. 85 discriminative topics** identified (two-gate filtering)

**2. Top-tier differentiation:**
- Psychological credibility scenes
- Embodied intimacy cues

**3. Trash-tier differentiation:**
- Explicit sexual content
- Procedural/transition scenes

**4. Author dominance:** 30 topics require control in modeling

**5. Hypothesis mapping:** Topics provide micro-scene exemplars for H1-H5

---

## Slide 14: Next Steps

**1. Subgroup analysis**
- Aggregate to taxonomy groups and Radway phases
- Compare topic-level vs category-level findings

**2. Modeling**
- Include author fixed/random effects
- Use topic-level predictors

**3. Time-course analysis (H6)**
- Examine topic probabilities across book tertiles (begin/middle/end)
- Test arc/pacing hypotheses

**4. Qualitative sampling
- Use procedure topics as scene anchors
- Sample exemplars for qualitative analysis

---

## Slide 15: Summary

**Topic-level EDA reveals:**

✅ **Specific micro-scenes** that distinguish tiers  
✅ **Patterns** that align with macro findings  
✅ **Author dominance** as measurable confounder  
✅ **Hypothesis exemplars** for subgroup analysis

**Methodological contributions:**
- Two-gate filtering rule for underpowered samples
- Two-tier structure (high confidence vs exploratory)
- Author dominance metrics for objective control

**Value for large-N study:**
- Design intelligence for modeling
- Micro-scene exemplars for interpretation
- Author control strategy

---

## Slide 16: Questions & Discussion

**Key questions:**
1. How do topic-level findings inform category-level aggregation?
2. Should author-dominant topics be excluded or controlled?
3. How to interpret "procedure topics" (scene scaffolding)?
4. What's the relationship between topic-level and index-level findings?

**Data & code available:**
- `results/stage10_correlation_analysis/01_topic_analysis/`
- `notebooks/07_analysis/01_topic_analysis/01_topic_analysis_v2_contract_normalized.ipynb`

---

## Appendix Slides (Optional)

### Slide A1: Topic Health Summary Statistics

**342 topics analyzed:**
- Median prevalence: 0.924
- Mean prevalence: 0.689
- Median mass: 0.0020
- Mean mass: 0.0027
- Median concentration ratio: 2.68

**Top 10 by prevalence:** All appear in 100% of books

**Top 10 by mass:**
1. Intimate Breast And Nipple Kissing (0.0198)
2. Relationship Ambiguity Conversation (0.0172)
3. Work-related Presentation (0.0131)
4. Wine Tasting At Dinner (0.0128)
5. Shoulder Burden (0.0113)
...

### Slide A2: Full Tier 1 Topics

**All 8 Tier 1 High-Confidence Topics:**

**Top ↑ (7 topics):**
1. Married Couple's Affectionate Stares (δ = 0.453)
2. Frightened Admissions (δ = 0.420)
3. Emotional Relationship Delusion (δ = 0.404)
4. Bluffing About Feelings (δ = 0.393)
5. Feminist Identity Affirmation (δ = 0.371)
6. Lip Biting During Intimacy (δ = 0.364)
7. Disco Night Adventure (δ = 0.351)

**Trash ↑ (1 topic):**
1. Dominatrix Session (δ = -0.353)

### Slide A3: Procedure Topics (Scene Scaffolding)

**15 procedure topics identified:**

These are scene anchors for qualitative sampling, not thematic content:
- Knocking On Door
- Phone Ringing And Answering
- Exiting Through Doorways
- Work At Desk
- Wine-drinking At Table
- Morning Grooming In Bathroom
- Sleeping Together
- Birdwatching In Park
...

**Use case:** Structural/pacing analysis, not thematic interpretation.

### Slide A4: Author Dominance Top 10

**Topics with 100% author dominance:**
1. Doorway Questions And Answers (Meghan_Quinn)
2. Confused About Relationship Feelings (Stella_Rhys)
3. Parental Warning About Engagement (Ana_Huang)
4. Serious Conversation About Relationship (Sam_Crescent)
5. Archery Practice (LJ_Shen)
6. Manager's Disappointed Moment (Jessica_Clare)
7. Breakup Reflection (Meghan_Quinn)
8. Incessant Complaints About Broccoli (Jessica_Clare)
9. Audience-focused Decision (Leslie_North)
10. Wheelchair Bonding (Lisa_Kleypas)

**Action:** Exclude from tier interpretation, control in modeling.

