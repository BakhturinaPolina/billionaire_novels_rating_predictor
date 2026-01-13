# Topic-Level Landscape Across Popularity Tiers: Micro-Scenes That Distinguish "Top" vs "Trash" Billionaire Romances

## Abstract

We analyzed BERTopic soft topic probabilities at the book level across three Goodreads rating tiers (Top / Middle / Trash) to identify micro-scene patterns that distinguish highly-rated from poorly-rated billionaire romance novels. Using a two-gate filtering approach (effect size ≥ 0.20 and meaningful impact), we identified 85 discriminative topics from 342 total topics. Top-tier books are characterized by psychological credibility scenes (fear admissions, emotional delusion) and embodied intimacy cues, while trash-tier books show more explicit sexual content and procedural/transition scenes. These micro-scene differences align with macro-level findings about payoff/safety and emotional coherence.

---

## 1. Introduction

### 1.1 Research Context

This analysis bridges the gap between macro-level thematic indices (taxonomy groups, Radway phases) and the finest granularity of narrative content: individual topic probabilities from BERTopic. By examining how specific micro-scenes vary across popularity tiers, we can identify not just *what* themes differ, but *which specific scene types* contribute to those differences.

### 1.2 Scope and Objectives

**Scope:** Topic-level analysis using BERTopic probabilities for all 368 topics. Statistical comparisons use all three tiers (Top / Middle / Trash); pairwise contrasts (Top vs Trash) are emphasized.

**Objectives:**
1. Characterize topic health (prevalence, mass, concentration) across the corpus
2. Identify topics with meaningful tier differences using effect sizes and impact gates
3. Map discriminative topics to hypotheses (H1-H5) and procedure categories
4. Control for author dominance as a potential confounder

---

## 2. Data and Methods

### 2.1 Data Sources

- **Book topic probabilities:** `results/stage10_correlation_analysis/00_data_preparation/topic_probabilities/book_topic_probs.parquet`
  - 33,856 rows (book-topic pairs)
  - Normalized probabilities (sum to ~1.0 per book)
  - 92 books × 368 topics

- **Chapter topic probabilities:** `results/stage10_correlation_analysis/00_data_preparation/topic_probabilities/chapter_topic_probs.parquet`
  - 1,089,280 rows (chapter-topic pairs)
  - Used for segment-level analysis (not primary focus here)

- **Topic lookup:** `results/stage10_correlation_analysis/00_data_preparation/taxonomy_radway_eda/topic_lookup.parquet`
  - 369 topics (368 + noise topic)
  - Contains labels, keywords, scene summaries, taxonomy/Radway mappings

- **Metadata:** Goodreads ratings, author information, rating class assignments

### 2.2 Preprocessing and Integrity Checks

**Probability normalization:** Verified that probability sums per book are ~1.0 (min: 0.999, max: 1.000). Missing probability mass was handled via an "OTHER" bucket approach for consistency.

**Rating tier mapping:**
- Rating classes (good/mid/bad) mapped to tiers: `good → top`, `mid → middle`, `bad → trash`
- Distribution: 30 top, 32 middle, 30 trash books

**Topic label merging:** All statistical outputs use topic labels (not topic IDs) for interpretability. Labels were merged from `topic_lookup.parquet`.

**Data quality checks:**
- Unique topics: 368 (as expected)
- Book ID overlap: 92/92 books matched between topic probabilities and metadata
- NaN probabilities: 0 (all probabilities valid)

### 2.3 Statistical Methods

**Topic health metrics:**
- **Prevalence:** Proportion of books where topic probability > 0.001
- **Mass:** Mean probability across all books
- **Concentration ratio:** Max probability / mean probability (flags single-book dominance)

**Tier comparisons:**
- **Summary statistics:** Median, mean, Q1, Q3, std per tier
- **Effect size:** Cliff's delta (Top vs Trash) as primary metric
- **Significance tests:**
  - Kruskal-Wallis (3-group comparison)
  - Mann-Whitney U (pairwise: Top vs Trash, Top vs Middle, Middle vs Trash)
  - FDR correction applied (Benjamini-Hochberg)

**Two-gate filtering rule:**
- **Gate 1 (Effect):** |Cliff's δ| ≥ 0.20 (meaningful effect size)
- **Gate 2 (Impact):** mass ≥ 0.002 OR |Top–Trash mean diff| ≥ 0.001 (meaningful impact)

**Two-tier structure:**
- **Tier 1 (High Confidence):** |δ| ≥ 0.35 AND raw p < 0.05 AND both gates passed (8 topics)
- **Tier 2 (Exploratory):** |δ| ≥ 0.20 AND both gates passed, no p-value filter (85 topics, includes Tier 1)

**Rationale for two-gate rule:** With n=30 books per tier, even large effect sizes (|δ| > 0.35) cannot survive FDR correction. The smallest adjusted p-value is ~0.20. The two-gate rule balances statistical rigor with practical interpretability for hypothesis-generating exploratory research.

---

## 3. Results

### 3.1 Topic Health Landscape

**Overall statistics (342 topics analyzed):**
- Median prevalence: 0.924 (most topics appear in most books)
- Mean prevalence: 0.689
- Median mass: 0.0020
- Mean mass: 0.0027
- Median concentration ratio: 2.68

**Key insight:** Under soft topic assignment, most topics receive small non-zero probability in most books. This makes naive "topic presence" less informative than effect sizes and mass thresholds.

**Top topics by prevalence (all appear in 100% of books):**
1. Admiring Handsome Man
2. Admiring Intelligence
3. Anger Management In Relationship
4. Angry Argument
5. Apologetic Excuse Me
6. Arrogant Man's Behavior
7. Awkward Questions At Door
8. Bedroom Encounter
9. Bedroom Intimacy
10. Belief Discussion

**Top topics by mass (highest mean probability):**
1. Intimate Breast And Nipple Kissing (0.0198)
2. Relationship Ambiguity Conversation (0.0172)
3. Work-related Presentation (0.0131)
4. Wine Tasting At Dinner (0.0128)
5. Shoulder Burden (0.0113)

### 3.2 Two-Gate Filtering Results

**Gate 1 (Effect |δ| ≥ 0.2):** 163 topics passed  
**Gate 2 (Impact: mass ≥ 0.002 OR |mean diff| ≥ 0.001):** 186 topics passed  
**Both gates passed:** 85 topics

**Final filtered set:**
- **Tier 1 (High Confidence):** 8 topics
  - Top-associated (δ > 0): 7
  - Trash-associated (δ < 0): 1
- **Tier 2 (Exploratory):** 85 topics
  - Top-associated (δ > 0): 70
  - Trash-associated (δ < 0): 15

### 3.3 Top-Associated Topics (Top ↑)

**Tier 1 High-Confidence Topics (Top ↑):**

| Topic ID | Label | |Cliff's Δ| | Top Median | Trash Median | Prevalence | Mass | Taxonomy Group | Radway Phase |
|----------|-------|------------|------------|-------------|--------------|------------|------|----------------|-------------|
| 67 | Married Couple's Affectionate Stares | 0.453 | 0.00037 | 0.00033 | 0.043 | 0.0010 | Sexuality, Attraction & Intimacy | Turning Point & Recognition |
| 82 | Frightened Admissions | 0.420 | 0.00229 | 0.00179 | 1.000 | 0.0022 | Emotions, Cognition & Inner Life | Initial Conflict & Isolation |
| 321 | Emotional Relationship Delusion | 0.404 | 0.00253 | 0.00206 | 0.978 | 0.0028 | Emotions, Cognition & Inner Life | Initial Conflict & Isolation |
| 66 | Bluffing About Feelings | 0.393 | 0.00095 | 0.00080 | 0.207 | 0.0017 | Emotions, Cognition & Inner Life | Initial Conflict & Isolation |
| 310 | Feminist Identity Affirmation | 0.371 | 0.00258 | 0.00222 | 1.000 | 0.0026 | Emotions, Cognition & Inner Life | Initial Conflict & Isolation |
| 39 | Lip Biting During Intimacy | 0.364 | 0.00282 | 0.00184 | 0.913 | 0.0023 | Sexuality, Attraction & Intimacy | Turning Point & Recognition |
| 226 | Disco Night Adventure | 0.351 | 0.00231 | 0.00188 | 0.978 | 0.0027 | Relationship Trajectory (Main Couple) | Turning Point & Recognition |

**Interpretation:**

Top-tier books are differentiated by two main micro-scene patterns:

1. **Inner-life and emotional truth moments:**
   - Fear admissions ("Frightened Admissions")
   - Self-deception about feelings ("Emotional Relationship Delusion")
   - Struggling to express feelings ("Bluffing About Feelings")
   - Identity affirmation ("Feminist Identity Affirmation")
   - These often map to **Initial Conflict & Isolation** (Radway Phase I)

2. **Embodied intimacy cues:**
   - Non-explicit but tactile/affective intimacy signals
   - "Married Couple's Affectionate Stares"
   - "Lip Biting During Intimacy"
   - "Disco Night Adventure" (shared joy/play)
   - These often map to **Turning Point & Recognition** (Radway Phase II)

**Top 10 Tier 2 Topics (by effect size, Top ↑):**

1. Engagement Surprise (δ = 0.573)
2. Argument With Glare (δ = 0.547)
3. Excited Diversion (δ = 0.487)
4. Married Couple's Affectionate Stares (δ = 0.453) [Tier 1]
5. Frightened Admissions (δ = 0.420) [Tier 1]
6. Compromising Photoshoot (δ = 0.404)
7. Emotional Relationship Delusion (δ = 0.404) [Tier 1]
8. Smiling And Laughing Together (δ = 0.396)
9. Bluffing About Feelings (δ = 0.393) [Tier 1]
10. Emotional Conversation Around Table (δ = 0.393)

### 3.4 Trash-Associated Topics (Trash ↑)

**Tier 1 High-Confidence Topics (Trash ↑):**

| Topic ID | Label | |Cliff's Δ| | Top Median | Trash Median | Prevalence | Mass | Taxonomy Group | Radway Phase |
|----------|-------|------------|------------|-------------|--------------|------------|------|----------------|-------------|
| 55 | Dominatrix Session | 0.353 | 0.00597 | 0.00834 | 1.000 | 0.0092 | Sexuality, Attraction & Intimacy | Commitment & Restoration |

**Interpretation:**

Trash-tier differentiation is dominated by:

1. **More explicit sexual content:**
   - "Dominatrix Session" (BDSM scenes)
   - "Intimate Breast And Nipple Kissing" (explicit erotics)
   - These map to **Explicit Sexual Acts** taxonomy and **Commitment & Restoration** (Radway Phase III)

2. **More procedural/transition scenes:**
   - "Work At Desk" (office routine)
   - "Exiting Through Doorways" (scene transitions)
   - "Phone Ringing And Answering" (pacing filler)
   - "Awkward Questions At Door" (structural scaffolding)
   - These map to **Domestic Spaces & Routines** or **Not a narrative function** (Radway)

3. **Relationship-conflict mechanics:**
   - "Deflecting Blame"
   - "Reluctant Relationship Talk"
   - "Protective Conversations By Firelight" (may reflect generic protector language patterns rather than genuine protective caretaking)

**Top 10 Tier 2 Topics (by effect size, Trash ↑):**

1. Dominatrix Session (δ = -0.353) [Tier 1]
2. Protective Conversations By Firelight (δ = -0.349)
3. Deflecting Blame (δ = -0.333)
4. Work At Desk (δ = -0.329)
5. Awkward Questions At Door (δ = -0.322)
6. Exiting Through Doorways (δ = -0.307)
7. Phone Ringing And Answering (δ = -0.304)
8. Reluctant Relationship Talk (δ = -0.304)
9. Intimate Breast And Nipple Kissing (δ = -0.300)
10. Admiring Intelligence (δ = -0.264)

**Important nuance:** A label like "Protective Conversations By Firelight" sounds positive, but topic labels are imperfect. This likely captures a *specific rhetorical pattern* (generic protector language / repetitive "protector" phrasing) rather than the broader "protective caretaking" construct that predicts quality in index-level models.

### 3.5 Topic-to-Hypothesis Mapping

**Hypothesis shortlists (from Tier 2 filtered set):**

- **H1_Explicit:** 12 topics (explicit sexual content)
  - Dominatrix Session, Intimate Breast And Nipple Kissing, Clitoral Stimulation During Foreplay, etc.
  
- **H2_Commitment:** 25 topics (commitment/trust/relationship-definition)
  - Marriage Ceremony Planning, Trust Assurance Conversation, Excited Diversion, etc.
  
- **H3_LuxuryWork:** 9 topics (luxury/work/business)
  - Work At Desk, Negotiating Business Deal, Luxury Hotel Suite, etc.
  
- **H4_Emotional:** 40 topics (emotional/feelings/conversation)
  - Emotional Relationship Delusion, Frightened Admissions, Smiling And Laughing Together, etc.
  
- **H5_Conflict:** 39 topics (argument/conflict/struggles)
  - Frightened Admissions, Insane Arguments, Deflecting Blame, etc.
  
- **S_Procedure:** 15 topics (scene scaffolding/structural)
  - Knocking On Door, Phone Ringing And Answering, Exiting Through Doorways, etc.

**Overlap note:** Many topics map to multiple hypotheses (e.g., "Frightened Admissions" appears in both H4_Emotional and H5_Conflict).

### 3.6 Author Dominance Analysis

**Author dominance metrics:**
- Topics analyzed: 330
- Topics with high author dominance (>50% from single author): 30
- Topics with medium author dominance (30-50%): 12

**Top 10 topics by author dominance:**
1. Doorway Questions And Answers (100% Meghan_Quinn)
2. Confused About Relationship Feelings (100% Stella_Rhys)
3. Parental Warning About Engagement (100% Ana_Huang)
4. Serious Conversation About Relationship (100% Sam_Crescent)
5. Archery Practice (100% LJ_Shen)
6. Manager's Disappointed Moment (100% Jessica_Clare)
7. Breakup Reflection (100% Meghan_Quinn)
8. Incessant Complaints About Broccoli (100% Jessica_Clare)
9. Audience-focused Decision (100% Leslie_North)
10. Wheelchair Bonding (100% Lisa_Kleypas)

**Filtered topics with author dominance flags:**
- Topics that are both significant AND author-driven: 6
- Examples:
  - Married Couple's Affectionate Stares (75% Catharina_Maura, δ = 0.453)
  - Voyeuristic Secretary (75% Sara_Cate, δ = 0.336)
  - Work-related Evening Call (60% Louise_Bay, δ = 0.289)

**Methodological implication:** Author-dominant topics should be:
- Excluded from tier interpretation (they reflect author style, not tier preferences)
- Treated as covariates in modeling
- Documented separately as "author signature topics"

---

## 4. Discussion

### 4.1 What Kinds of Micro-Scenes Differ?

**Top books** are more differentiated by:
- Scenes that carry **psychological credibility** (fear, admissions, self-deception)
- **Embodied intimacy cues** that function as "recognition/turning point" signals
- These align with the macro-level finding that **payoff/safety** and emotionally coherent arcs matter

**Trash books** are more differentiated by:
- **Explicit erotics** (aligns with macro finding that explicitness separates popularity/quality channels)
- **Procedural/transition scenes** (doors, phones, desk work) that often act as pacing filler
- This supports the planned "arc/pacing" focus: not just *what* themes exist, but *when* they dominate

### 4.2 Alignment with Macro-Level Findings

This micro-scene view matches the macro-axis inference:

- **Reach** is associated with status/dominance packages
- **Quality beyond reach** is associated with payoff/safety
- **Arcs** often show "late crisis escalation" patterns

The topic-level analysis provides **exemplars** of these patterns at the finest granularity.

### 4.3 Limitations and Caveats

**Critical caveat:** Because BERTopic probabilities are soft and dense, topic-level differences are often small in absolute magnitude even when effect sizes are non-trivial. The correct claim is not "Topic X causes popularity," but:

> "Certain micro-scenes are systematically more emphasized in one tier than another, with consistent directional effects under nonparametric comparisons."

**Statistical power:** With n=30 books per tier, FDR-corrected p-values are underpowered. We rely on effect sizes and two-gate filtering rather than significance alone.

**Author confounding:** 30 topics show high author dominance (>50% from single author). These should be controlled in modeling and interpreted cautiously.

**Label interpretation:** Topic labels are imperfect summaries. A label like "Protective Conversations By Firelight" may capture generic rhetorical patterns rather than genuine protective caretaking.

### 4.4 Implications for Large-N Study

This pilot EDA provides design intelligence:

1. Keep topic-level interpretation as **micro-scene exemplars**, not as "the cause"
2. The strongest "Trash ↑" signals align with macro result that **explicitness separates popularity/quality channels**
3. Procedural/transition topics support "arc/pacing" focus: not just *what* themes exist, but *when* they dominate
4. Author fixed effects or random effects should be included in modeling
5. Filter author-dominant topics before aggregation in subgroup analysis

---

## 5. Conclusions

### 5.1 Key Findings

1. **85 discriminative topics** identified from 342 total (two-gate filtering: effect size ≥ 0.20 and meaningful impact)

2. **Top-tier differentiation:**
   - Psychological credibility scenes (fear admissions, emotional delusion)
   - Embodied intimacy cues (affectionate stares, lip biting, shared joy)

3. **Trash-tier differentiation:**
   - Explicit sexual content (dominatrix sessions, explicit erotics)
   - Procedural/transition scenes (doors, phones, desk work)

4. **Author dominance:** 30 topics show high author dominance (>50% from single author), requiring control in modeling

5. **Hypothesis mapping:** Topics map to H1-H5 and procedure categories, providing micro-scene exemplars for macro-level patterns

### 5.2 Methodological Contributions

- **Two-gate filtering rule** balances statistical rigor with practical interpretability for underpowered samples
- **Two-tier structure** (high confidence vs exploratory) provides transparency about evidence strength
- **Author dominance metrics** provide objective way to control for author-style topics

### 5.3 Next Steps

1. **Subgroup analysis:** Aggregate topic-level findings to taxonomy groups and Radway phases
2. **Modeling:** Include author fixed/random effects and topic-level predictors
3. **Time-course analysis (H6):** Examine topic probabilities across book tertiles (begin/middle/end)
4. **Qualitative sampling:** Use procedure topics as scene anchors for qualitative analysis

---

## 6. Appendix

### 6.1 Tables

**Available in `results/stage10_correlation_analysis/01_topic_analysis/tables_csv/`:**

- `topic_health_table.csv` - Full topic health metrics (342 topics)
- `topic_leaderboard_all.csv` - Complete leaderboard with all statistics (342 topics)
- `topic_leaderboard_tier1_high_confidence.csv` - Tier 1 topics (8 topics)
- `topic_leaderboard_tier2_exploratory.csv` - Tier 2 topics (85 topics)
- `topic_leaderboard_filtered.csv` - Filtered set (85 topics, same as Tier 2)
- `topic_leaderboard_top_associated.csv` - Top 50 by top_median
- `topic_leaderboard_trash_associated.csv` - Top 50 by trash_median
- `topic_leaderboard_effect_sizes.csv` - Top 50 by |Cliff's δ|
- `topic_hypothesis_mapping.csv` - Topic-to-hypothesis mapping (133 topics)
- `procedure_topics.csv` - Procedure/structural topics (15 topics)
- `topic_author_dominance.csv` - Author dominance metrics (330 topics)

### 6.2 Figures

**Available in `results/stage10_correlation_analysis/01_topic_analysis/figures/`:**

- `topic_distributions/all_topics_violin_grid.html` - Grid visualization of 40 topics (top 30 + 10 random)
- `topic_distributions/top_20_topics_detailed/` - Detailed violin and box plots for top 20 topics by prevalence

**Recommended figures for publication:**
- Topic prevalence histogram (Figure 1)
- Effect size vs prevalence scatter (Figure 2)
- Bar plot of strongest Top ↑ topics (Figure 3)
- Bar plot of strongest Trash ↑ topics (Figure 4)

### 6.3 Data Availability

All data files, tables, and figures are available in:
- `results/stage10_correlation_analysis/01_topic_analysis/`

Code for reproducing this analysis:
- `notebooks/07_analysis/01_topic_analysis/01_topic_analysis_v2_contract_normalized.ipynb`

---

## References

(To be added: citations for BERTopic, statistical methods, romance genre studies, etc.)

