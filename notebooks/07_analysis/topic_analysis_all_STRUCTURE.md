# Notebook Structure: Analysis of All 368 Topics Across Top/Middle/Trash Tiers

**Purpose:** Comprehensive bottom-to-top analysis ("start from the atoms, then build molecules") of all 368 BERTopic topics across popularity tiers (Top/Middle/Trash), using human-tagged thematic taxonomy and theory-driven composites/indices to test H1–H6.

**Reference Documentation:**
- BERTopic API: https://maartengr.github.io/BERTopic/index.html#citation
- BERTopic approximate_distribution: https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.approximate_distribution

---

## 0. Analysis-Ready Structure (So Everything Downstream Behaves)

### 0.1 Setup & Imports
- Project root resolution
- Import libraries: pandas, numpy, matplotlib, seaborn, plotly, scipy.stats, statsmodels
- BERTopic import (for reference/documentation)
- Set plotting styles and output directories

### 0.2 Define Data Paths

**Note:** Notebooks use flexible path resolution with multiple fallback candidates to handle directory structure variations.

```python
PROJECT_ROOT = Path("/home/polina/Documents/goodreads_romance_research_cursor/billionaire_novels_rating_predictor")

# Data paths (from data preparation stage)
# Primary location: results/stage10_correlation_analysis/00_data_preparation/
# Fallbacks: results/stage10_correlation_analysis/data_preparation/, results/correlation_analysis/data_preparation/
BOOK_TOPIC_PROBS_PATH = PROJECT_ROOT / "results" / "stage10_correlation_analysis" / "00_data_preparation" / "topic_probabilities" / "book_topic_probs.parquet"
CHAPTER_TOPIC_PROBS_PATH = PROJECT_ROOT / "results" / "stage10_correlation_analysis" / "00_data_preparation" / "topic_probabilities" / "chapter_topic_probs.parquet"
TOPIC_LOOKUP_PATH = PROJECT_ROOT / "results" / "stage10_correlation_analysis" / "00_data_preparation" / "taxonomy_radway_eda" / "topic_lookup.parquet"
GOODREADS_PATH = PROJECT_ROOT / "data" / "processed" / "goodreads.csv"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "results" / "stage10_correlation_analysis" / "01_topic_analysis"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
```

### 0.3 Define Units of Analysis

Lock these in early—you'll switch units often:

| Unit | Structure | Notes |
|------|-----------|-------|
| **Book-level** | One row per book | Full topic mixture (sums to 1) |
| **Segment-level** | One row per (book × segment {begin, middle, end}) | Topic mixture per segment (each segment sums to 1) |
| **Topic-level** | Topic as "feature" | Outcomes are group differences / correlations |

### 0.4 Data Integrity Checks (Fast but Essential)

Using data contracts:

- `book_topic_probs`: confirm each book has (almost) all 368 topics; check `sum(prob) ≈ 1` per book
- `chapter_topic_probs`: for each (book, segment), check sums; confirm all segments exist
- `books_meta`: confirm group labels (rating_class: good/mid/bad), rating ranges, n_ratings, length, author counts
- **Record inference procedure**: sentence-level aggregation vs per-segment inference (affects comparability across segments)

### 0.5 Handle Compositional Data Reality

Topic probabilities are **compositional**: increasing one topic necessarily decreases others.

**Strategy:**
- Do **exploratory stats on raw probs** (informative for screening)
- Confirm key claims using **log-ratio** or **Dirichlet-style** modeling where feasible
- Document this limitation throughout analysis

### 0.6 Topic Prevalence Filters (Prevent 300+ Topic Chaos)

For each topic, compute:

| Metric | Definition |
|--------|------------|
| **Prevalence** | Fraction of books where prob > ε (e.g., > 0.001) |
| **Mass** | Mean prob across all books |
| **Concentration** | How "spiky" it is (Gini-ish: few books dominate it) |

Use these to:
- Separate "signal topics" from "dust topics"
- Avoid interpreting topics that appear in 3 books and vanish

Create "topic health table" with:
- topic_id, prevalence, mass, concentration
- manual label (from topic_lookup)
- taxonomy_main_name, taxonomy_main_group
- radway_phase_name (if available)

**Deliverable:** `topic_health_table.parquet` saved to TABLE_DIR

---

## 1. Bottom Layer: Individual Topic Distributions Across Top/Middle/Trash

Begin with **topic-by-topic probability distributions** across groups. With 300+ topics, this is a *screening + interpretation* workflow.

### 1.1 Merge Topic Probabilities with Book Metadata
- Merge book_topic_probs with books_meta to get rating_class
- Map rating_class: good → "Top", mid → "Middle", bad → "Trash"
- Verify all books have rating_class assigned
- **Use topic labels from topic_lookup.parquet** (not topic IDs) in all statistical outputs

### 1.2 Visual Exploration (Distribution-First, Not Mean-First)

For each topic, compare distributions across groups:

**Plot Types:**
- **Violin / ridge / density** plots (good for "shape" differences)
- **ECDF curves** (great when topic is mostly zero-ish and only sometimes spikes)
- **Boxplots + jitter** (good for seeing individual books)

**Key questions per topic:**
- Is the topic *present in all tiers but stronger in one*?
- Or *nearly absent in Top but common in Trash*?
- Does it show *bimodality* (two clusters) suggesting subtypes or author effects?

**Batch Visualization Strategy:**
- Summary visualizations for all topics (grid/facet plots)
- Individual detailed plots for top N topics (by prevalence/mass)
- Interactive Plotly figures saved to FIG_DIR

### 1.3 Quantify Differences Per Topic (Screen, Don't Overpromise)

For each topic, compute:

**1.3.1 Group-wise Central Tendency**
- Median (robust to outliers)
- Mean (for comparison)
- Q1, Q3 (quartiles)

**1.3.2 Effect Sizes**
- Top vs Trash (primary comparison)
- Top vs Medium, Medium vs Trash (optional)
- Use robust effect size: **Cliff's delta** or rank-biserial correlation (works well with non-normal, zero-inflated distributions)

**1.3.3 Significance Testing (Optional)**
- Non-parametric: Kruskal-Wallis for 3 groups
- Post-hoc: Mann-Whitney U tests
- Correction: FDR/Benjamini-Hochberg across 300+ topics

**1.3.4 Topic-Level Leaderboard**

Create tables:
- "Most Top-associated topics" (highest median/mean in Top)
- "Most Trash-associated topics" (highest in Trash)
- "Most Medium-peaked topics" (highest in Medium relative to others)
- "Topics with biggest Top–Trash separation" (largest effect size)

**Deliverables:**
- `topic_leaderboard_all.parquet` - full results for all 368 topics
- `topic_leaderboard_top_associated.parquet` - top N topics associated with Top tier
- `topic_leaderboard_trash_associated.parquet` - top N topics associated with Trash tier
- `topic_leaderboard_effect_sizes.parquet` - sorted by effect size

### 1.4 Tame Multiple Comparisons Problem (Don't Worship Noise)

With 300+ topics, you'll get "significant" stuff by rolling statistical dice.

**Sensible gates (must pass ALL):**
1. **Minimum prevalence**: appears in ≥ 15–20% of books
2. **Meaningful effect size**: not just p < .05, but |Cliff's delta| > threshold
3. **Survives FDR correction**: if testing, p_adj < 0.05
4. **Interpretable label**: topic actually coheres on inspection

**Deliverable:** `topic_leaderboard_filtered.parquet`

### 1.5 Author as "Shadow Confounder" (Early Check)

Romance authors can imprint topics strongly. Before interpreting a topic as "Top loves X":

- Check whether topic is dominated by 1–2 authors
- Quick diagnostic: compute topic prevalence per author
- Conceptually: topic leaderboard "leave-one-author-out"
- If topic disappears when one author is removed → "author signature," not "tier signature"
- **Author-signature filtering**: 41 high-dominant topics identified and excluded from main comparisons
- **Gate 3 filter**: prevalence >= 0.10 AND NOT author-dominant

**Deliverable:** `topic_author_dominance.parquet` with flags: **tier-stable vs author-driven**

---

## 2. Mid Layer: Distributions of Topic Groups (Taxonomy) Across Tiers

Aggregate topics into thematic tags and compare group-level shares.

### 2.1 Build the Taxonomy Mapping Table (The Spine)

Create a master mapping where each topic has:

| Field | Description |
|-------|-------------|
| `topic_id` | Unique topic identifier |
| `short_label` | Human-readable label |
| `main_group` | e.g., Embodied, Sexuality, Emotions, … |
| `subgroup_node` | e.g., 2.3 Explicit Sexual Acts |
| `theory_tags` | Radway phase/function, A–S composite tags (optional) |
| `confidence_score` | high/medium/low + notes for ambiguous topics (optional) |

**This table becomes the one truth source.** Every index is "just sums of this mapping."

### 2.2 Aggregation Rules (Don't Double-Count)

Decide whether each topic maps to:
- **Exactly one subgroup** (cleanest for statistics) — **RECOMMENDED**
- **Multiple subgroups with weights** (more expressive but harder to justify)

**Recommendation:** One primary subgroup per topic, plus optional secondary tags for qualitative interpretation only.

### 2.3 Main-Group Distribution Comparisons (Coarse Lens)

For each book:
- Main-group share = sum of probs of topics assigned to that main group

Compare across Top/Middle/Trash:
- Distribution plots
- Effect sizes (Cliff's Delta, Epsilon-squared)
- Group tests (Kruskal-Wallis with Holm correction for pairwise comparisons)

**Key Pipeline Truths:**
1. **OTHER bucket** (~0.32 mean mass) differs by tier - must normalize conditionally
2. **Author-signature topics** (41 high-dominant) excluded from main comparisons
3. **Gate 3 filter**: prevalence >= 0.10 AND NOT author-dominant
4. **Dual normalization**: absolute vs conditional shares to handle OTHER bucket variation

**Key questions:**
- Do Top books allocate more mass to **Emotions/Inner Life** and less to **Conflict/Risk**?
- Is **Work/Wealth** uniformly present (billionaire romance baseline), but *interacts* with relationship themes?

**Deliverable:** 7-8 bar chart per tier (with uncertainty), plus pairwise contrasts

### 2.4 Subgroup Distributions Per Main Group ("Not Too Messy" Middle Layer)

Analyze within each main group separately. This avoids "28 subgroups in one plot" nightmare.

For each main group (e.g., Sexuality, Emotions, Relationship Trajectory…):
- Compute subgroup shares for that group only
- Compare subgroup distributions across tiers **within that main group**

**Produces interpretable chapters like:**
- "Inside Sexuality: soft affection vs foreplay vs explicit acts vs courtship gestures vs romantic atmosphere"
- "Inside Emotions: positive safety vs vulnerability vs hostility vs shame vs reflection vs growth"

**Deliverable:** One figure panel per main group, with subgroup contrasts Top/Middle/Trash

---

## 3. Next Layer: Topic-Level Inside Each Subgroup (Targeted Drill-Down)

### 3.1 Within-Subgroup Ranking

For each subgroup:
- Rank topics by Top–Trash effect size (or by tier association)
- Keep top N (e.g., 5–15) to interpret

**Produces interpretable statements like:**
> "Explicit sex isn't monolithic: Topic 214 (condoms/explicit anatomy) spikes in Trash; Topic 37 (post-sex reflection/aftercare) aligns with Top."

### 3.2 Subgroup "Coherence Audit"

Within each subgroup, inspect whether topics actually belong together.

**Catch misassignments like:**
- A "jealousy" topic accidentally placed under "Emotions positive"
- A "law enforcement" topic that's actually "security detail in luxury setting"

**Deliverable:** Refined mapping table (topic → subgroup) with fewer ambiguities

---

## 4. Build Theory-Aligned Composites and Indices (Bridge to Hypotheses)

This is where taxonomy becomes hypothesis-testing machinery.

### 4.1 Basic Index Construction Pattern

Every index should specify:

| Component | Description |
|-----------|-------------|
| **Components** | Which subgroups/topics feed it |
| **Direction** | + or − |
| **Normalization** | Raw share, z-score, log-ratio, etc. |
| **Interpretation** | "higher = more X" |
| **Reliability check** | Does it behave consistently across books/segments? |

**Measurement Pipeline (v5.6):**
- **Reliability diagnostics**: Cronbach's alpha, McDonald's omega, PCA (PC1/PC2), stability metrics (leave-one-out, split-half, bootstrap-to-full)
- **Composite classification**: ATOMIC (<3 topics) vs COMPOSITE, CORE vs EXPLORATORY, UNIDIMENSIONAL vs MULTIDIMENSIONAL
- **Recommended scoring**: sum / pc1 / pc1+pc2 based on dimensionality
- **Coverage metrics**: topic membership, final_n after filtering

Compute indices at both:
- **Book** level (global theme emphasis): raw and z-scored, sum and max aggregation
- **Segment** level (begin/middle/end for arc hypotheses): raw and z-scored, sum and max aggregation
- **Arc contrasts**: end−begin, middle−begin deltas for trajectory analysis

### 4.2 Recommended Normalization Choices

Because these are proportions:

| Option | When to Use |
|--------|-------------|
| **Z-score standardized shares** | Simple, interpretable, good for regression and group comparisons |
| **Log-ratio contrasts** | Better for compositional logic; use when hypothesis is explicitly a balance (love vs sex; tenderness vs darkness). Form: `log((A + B + ε) / (C + ε))` |

**Strategy:** Use z-scored shares for overview; confirm headline hypotheses with log-ratio versions.

### 4.3 Measurement Pipeline v5.6: Reliability Diagnostics

**Composite construction includes comprehensive reliability diagnostics:**

| Diagnostic | Metric | Interpretation |
|-----------|--------|----------------|
| **Internal Consistency** | Cronbach's α, McDonald's ω | α/ω ≥ 0.55 for CORE composites |
| **Dimensionality** | PCA PC1/PC2 | PC1 < 50% → MULTIDIMENSIONAL, use PC1+PC2 |
| **Stability** | Leave-one-out, split-half, bootstrap-to-full | Stability ≥ 0.60 for CORE composites |
| **Coverage** | Final_n after filtering | Minimum topics per composite |

**Composite Classification:**
- **ATOMIC** (<3 topics) vs **COMPOSITE** (≥3 topics)
- **CORE** (operationalized, reliable) vs **EXPLORATORY** (needs validation)
- **UNIDIMENSIONAL** (PC1 ≥ 50%) vs **MULTIDIMENSIONAL** (PC1 < 50%)

**Scoring Strategy:**
- **UNIDIMENSIONAL**: Use sum (or PC1 if preferred)
- **MULTIDIMENSIONAL**: Use PC1+PC2 (captures both dimensions)
- **ATOMIC**: Use raw topic probability

**Key Decision:** For hypothesis testing, use **END segment indices** (final third of book) for cross-sectional tests; use **arc contrasts** (end−begin, middle−begin) for trajectory analysis.

### 4.4 A–S Composites Mapped to Taxonomy (Post-Split Structure)

Below is a clean, taxonomy-grounded mapping implementable with topic→subgroup table. **Note:** After validation, many composites split into subcomponents (A1/A2/A3, B1/B2, etc.) that enable more precise hypothesis testing.

---

#### A) Reassurance / Commitment (HEA Centrality) — SPLIT

**Post-split structure:**
- **A1_commitment_vows_END**: Commitment language, vows, marriage plot
- **A2_emotional_safety_END**: Trust, security, calm, comfort
- **A3_repair_reconciliation_END**: Apology, forgiveness, repair moments

| Type | Taxonomy Nodes |
|------|----------------|
| **Primary** | 4.2 Relationship Stage & Commitment, 4.6 Rupture/Separation/Reconciliation (apology/forgiveness/repair/commitment moments) |
| **Secondary** | 3.1 Positive Emotions & Safety (security, calm, comfort) |

**Interpretation:** "commitment + repair + safety language"

---

#### B) Mutual Intimacy (Non-Explicit) — SPLIT

**Post-split structure:**
- **B1_physical_chemistry_END**: Non-explicit attraction, kissing, anticipation
- **B2_emotional_intimacy_END**: Tender closeness, emotional connection

| Type | Taxonomy Nodes |
|------|----------------|
| **Primary** | 2.1 Soft Affection & Non-Sexual Touch, 2.2 Sexual Arousal & Foreplay (non-explicit/kissing/anticipation side; topic-level filtering helps) |
| **Secondary** | 3.1 Positive Emotions & Safety, 4.2 Commitment (everyday intimacy signals) |

**Interpretation:** "closeness without explicit act language"

---

#### C) Explicit Eroticism

| Type | Taxonomy Nodes |
|------|----------------|
| **Primary** | 2.3 Explicit Sexual Acts |

**Note:** Optionally exclude topics that are mostly "consent talk/aftercare" if they exist (those can belong to M or A depending on content).

---

#### D) Power / Wealth / Luxury

**Note:** Low PC1 (24%) indicates wealth, status, and settings are distinct. Test interactions rather than simple sums.

| Type | Taxonomy Nodes |
|------|----------------|
| **Primary** | 5.2 Money, Wealth & Economic Security, 5.3 Luxury Lifestyle & Status Performance |
| **Secondary** | 1.2 Appearance, Clothing & Grooming (status performance), 7.2 Public, Leisure & Travel Spaces (jets/hotels/high-end venues) |

**Interpretation:** "billionaire-world saturation"

---

#### E) Coercion / Brutality / Danger (Dark Themes)

| Type | Taxonomy Nodes |
|------|----------------|
| **Primary** | 6.2 Physical Threats & Violence, 6.3 Psychological Harm & Trauma |
| **Secondary** | 5.5 Social Roles & Power/Control (when it reflects coercion/manipulation), 6.1 Interpersonal Conflict & Betrayal (if threatening/dark rather than just quarrels) |

**Interpretation:** "threat + coercion + traumatic texture"

---

#### F) Angst / Negative Affect — SPLIT

**Post-split structure:**
- **F1_sadness_grief_END**: Trauma affect, grief, vulnerability
- **F2_anger_frustration_END**: Conflict affect, hostility, resentment
- **F3_anxiety_worry_END**: Suspense affect, fear, worry

| Type | Taxonomy Nodes |
|------|----------------|
| **Primary** | 3.2 Vulnerability, Sadness & Fear, 3.3 Anger, Resentment & Hostility, 3.4 Guilt, Shame & Moral Conflict |
| **Secondary** | 6.1 Interpersonal Conflict & Betrayal (emotional conflict content) |

**Interpretation:** "negative emotional load"

---

#### G) Courtship Rituals / Gifts (HEA Component)

**Note:** α = -0.03 indicates courtship topics don't co-occur. Test as interaction with A1 rather than simple sum.

| Type | Taxonomy Nodes |
|------|----------------|
| **Primary** | 2.4 Courtship Rituals & Romantic Gestures, 7.4 Time & Life Events (holidays, anniversaries, birthdays, festive rituals) |
| **Secondary** | 7.3 Food, Drink & Shared Meals (dates/dinners), 1.2 Appearance and 5.3 Luxury (if gifts/jewelry are central in those topics) |

**Interpretation:** "ritualized romance behaviors"

---

#### H) Domestic Nesting (Home-as-Refuge)

| Type | Taxonomy Nodes |
|------|----------------|
| **Primary** | 7.1 Domestic Spaces & Home Life, 4.2 Commitment (cohabitation/domestic routine terms) |
| **Secondary** | 5.2 Economic Security (home stability), 7.3 Meals (cooking, shared home meals) |

**Interpretation:** "nest-building and everyday shared life"

---

#### I) Humor / Lightness

| Type | Taxonomy Nodes |
|------|----------------|
| **Primary** | 3.1 Positive Emotions & Safety (laughter/joy) |
| **Secondary** | 7.6 Sports & Games (playful scenes) if topics are actually comedic/playful |

**Interpretation:** "comic relief / breezy tone proxies"

---

#### J) Social Support / Kin

| Type | Taxonomy Nodes |
|------|----------------|
| **Primary** | 4.3 Family & Kinship, 4.5 Friends, Colleagues & Community, 4.4 Children & Parenthood (optional) |

**Interpretation:** "stable social buffering around the couple"

---

#### K) Professional Intrusion (Office/Corporate Frame Share)

| Type | Taxonomy Nodes |
|------|----------------|
| **Primary** | 5.1 Work & Professional Life, 5.4 Institutions, Law & Authority (if workplace-structured authority matters) |
| **Secondary** | 4.5 Colleagues when it's workplace-social |

**Interpretation:** "workplace and institutional texture in the romance"

---

#### L) Vices / Addictions

| Type | Taxonomy Nodes |
|------|----------------|
| **Primary** | 6.4 Addictions & Risky Behaviours |
| **Secondary** | Some 7.2 bars/nightlife topics *only if* they're about substance/risk rather than leisure |

**Interpretation:** "substance and self-destructive risk"

---

#### M) Health / Recovery / Growth (Tender Care + Healing Arcs)

| Type | Taxonomy Nodes |
|------|----------------|
| **Primary** | 1.4 Health, Care & Recovery, 3.6 Memory, Learning & Personal Growth |
| **Secondary** | 6.3 Trauma (if framed as recovery rather than ongoing harm) |

**Interpretation:** "healing and protective caretaking"

---

#### N) Separation / Reunion (Arc Mechanics)

| Type | Taxonomy Nodes |
|------|----------------|
| **Primary** | 4.6 Rupture, Separation & Reconciliation |
| **Secondary** | 7.4 Time & Life Events (waiting, time passing, distance) |

**Interpretation:** "break → return → repair signals"

---

#### O) Aesthetics / Appearance (Visual/Cultural Cues)

| Type | Taxonomy Nodes |
|------|----------------|
| **Primary** | 1.2 Appearance, Clothing & Grooming |
| **Secondary** | 1.1 Body Parts & Physical Reactions (if topics are gaze/beauty-coded rather than physiology), 5.3 Status performance (if primarily appearance-coded) |

**Interpretation:** "look-and-status signaling"

---

#### Q) Miscommunication vs Repair (Balance Index)

Split into two subcomponents:

| Subcomponent | Source |
|--------------|--------|
| **Q_miscommunication** | 4.1 Communication & Miscommunication (secrets, misunderstandings, arguments, silence) |
| **Q_repair** | Subset of 4.6 (apologies, forgiveness, reconciliation) + A components |

**Index Definition:**
- **Miscommunication Balance** = (repair + commitment + tenderness) − miscommunication
- Or log-ratio: `log((repair + ε) / (miscomm + ε))`

---

#### R) Protectiveness vs Jealousy (Delta Index) — SPLIT

**Post-split structure:**
- **R1_protective_caretaking_END**: Tender care, health/care, safety
- **R2_alpha_guarding_END**: Possessive protection, guarding behavior
- **R_jealousy_possessiveness_END**: Negative control, jealousy, possessiveness

| Subcomponent | Likely Sources |
|--------------|----------------|
| **R1_protectiveness** | 1.4 Health/Care, 3.1 Safety, parts of 4.2, parts of 4.6 (supportive repair) |
| **R2_alpha_guarding** | Protective power in 5.5, parts of 4.2 (possessive commitment) |
| **R_jealousy/possessiveness** | Jealousy topics in 6.1, possessive power in 5.5, hostility 3.3 (if jealousy-coded) |

**Index Definition:**
- **Protective–Jealousy Delta** = protectiveness − jealousy
- Or log-ratio: `log((protect + ε) / (jealous + ε))`

**Note:** R2 (alpha) may correlate with ratings only when paired with R1 (tender care) → test R1 × R2 interaction.

---

#### S) Scene Anchors (For Qualitative Sampling)

Less an "index to test," more a **sampling tool**:
- High-load topics from 7.x (spaces/objects/time/tech/food)
- Plus 2.5 Romantic Atmosphere
- Plus setting-rich 5.3 Luxury Lifestyle

**Use:** Pick representative scenes for close reading tied to formulaic "scene kits."

---

## 5. Hypothesis Testing Plan (H1–H6) Using Indices — REVISED

### Part 1: Hypothesis Testing Implementation (v4.2)

**Analysis Approach:**
- **Bootstrap inference**: 800 iterations, 95% CI, P(β>0) for directional effects
- **Macro-axes model**: 5-axis PCA reduction (status/dominance, payoff/safety, drama/obstacle, explicitness, negative affect)
- **Arc trajectory tests**: Use exported deltas (end−begin, middle−begin) from measurement pipeline
- **Cross-validation**: 20 repeats of 5-fold CV for predictive performance
- **Two-channel analysis**: Separate mass appeal (`log_rating_count`) from perceived quality (`rating_mean`)

---

### 5.1 H1: Love-Over-Sex Balance (REVISED)

**Original Hypothesis:**
- Books with higher (love + intimacy) / explicit-sex ratios receive higher ratings

**Post-Split Refinements:**

| Test | Formula | Rationale |
|------|---------|-----------|
| **H1a: Commitment language** | `log(A1_END / C_END)` | Tests if marriage plot beats heat |
| **H1b: Emotional safety** | `log(A2_END / C_END)` | Tests if trust/security beats heat |
| **H1c: Physical chemistry** | `log(B1_END / C_END)` | Tests if non-explicit attraction beats explicit |
| **H1d: Emotional intimacy** | `log(B2_END / C_END)` | Tests if tender closeness beats heat |
| **H1e: Combined (original)** | `log((A1+A2+A3+B1+B2)_END / C_END)` | Overall love/sex balance |

**Key Decision:** Use **H1e** as primary test; report **H1b** and **H1d** as sub-hypotheses (safety and intimacy are theoretically cleanest contrasts to sex).

**Tests:**
- Compare index across Top/Middle/Trash (Kruskal–Wallis + pairwise)
- Regress avg_rating on this index controlling for length, year, author
- Optional: predict Top vs Trash with logistic regression

**Expected pattern if H1 holds:** Top > Medium > Trash on love-over-sex balance

---

### 5.2 H2: HEA Index Hypothesis (REVISED)

**Original:** HEA = A + G

**Problem:** G has α = -0.03 (courtship topics don't co-occur)

**Revised:**
- **H2 Primary:** A1_commitment_vows_END (commitment language = HEA signal)
- **H2 Secondary:** A1_END + G_courtship_rituals_END (test if gifts/rituals add predictive power)

**Test:**
```
Rating ~ A1_END + G_END + A1×G + controls
```

If interaction positive → courtship rituals amplify commitment's effect

**Additional Tests:**
- Group differences across tiers
- Predict rating and Top-vs-Trash
- Check whether effect holds after controlling for explicitness (C)

---

### 5.3 H3: Luxury × Love Interaction (REVISED)

**Problem:** D has PC1 = 24% (wealth, status, settings are distinct)

**Revised Tests:**

| Interaction | Interpretation |
|-------------|----------------|
| D_END × A1_END | Does luxury amplify commitment language? |
| D_END × B2_END | Does luxury amplify emotional intimacy? |
| D_END × (A1+B2)_END | Overall luxury × love depth |

**Model:**
```
Rating ~ D + (A1+B2) + D×(A1+B2) + C + controls
```

**Interpretation:** If luxury alone doesn't help but luxury *with love depth* does → interaction term should be positive and meaningful.

**Prediction:** Main effects weak, interaction positive (luxury only works with depth)

---

### 5.4 H4: Protectiveness vs Possessiveness (REVISED)

**Original:** R_protect − R_jealousy

**Post-Split:**

| Index | Formula | Hypothesis |
|-------|---------|------------|
| **H4a: Tender protection** | `log(R1_END / R_jealousy_END)` | Caretaking > jealousy → higher rating |
| **H4b: Alpha vs jealousy** | `log(R2_END / R_jealousy_END)` | Guarding vs possessiveness (may be nonlinear) |
| **H4c: Balance** | `log((R1+R2)_END / R_jealousy_END)` | Overall protective > jealous |

**Note:** R2 (alpha) may correlate with ratings only when paired with R1 (tender care) → test R1 × R2 interaction.

**Tests:**
- Tier differences and rating prediction
- Interaction with Explicitness or Conflict themes (optional) to see if "possessiveness" becomes acceptable under specific trope packages

---

### 5.5 H5: Darkness vs Tenderness (REVISED)

**Original:** (E + F) − B

**Post-Split:**
- F1_sadness_grief_END (trauma affect)
- F2_anger_frustration_END (conflict affect)
- F3_anxiety_worry_END (suspense affect)
- E_coercion_brutality_END (threat content)

**Revised Tests:**

| Test | Formula | Prediction |
|------|---------|------------|
| **H5a: Trauma vs safety** | `log((E+F1)_END / A2_END)` | Violence+grief vs emotional safety |
| **H5b: Anger vs intimacy** | `log(F2_END / B2_END)` | Conflict vs emotional closeness |
| **H5c: Darkness saturation** | `(E + F1 + F2)_END` | Quadratic effect? (Some ok, too much bad) |

**Key Test:** H5c with quadratic term:
```
Rating ~ darkness + darkness² + controls
```

If negative quadratic → inverted U (optimal darkness exists)

**Additional Tests:**
- Tier differences
- Does Darkness penalize ratings? Or is it nonlinear (some darkness ok, too much bad)?

---

### 5.6 H6: Narrative Arc (Time-Course) (REVISED)

**Use:** `arc_contrasts_sum.parquet` and `arc_contrasts_max.parquet` from measurement pipeline

**Key Insight:** Low begin-end correlation is GOOD → means metrics capture change

**Trajectory Tests Using Exported Deltas:**

| Metric | Expected Pattern | Test |
|--------|------------------|------|
| A1 (commitment) end−begin | Positive (↑) | Bootstrap β with 95% CI |
| C (explicit sex) end−begin | Peak middle? | Quadratic term or middle−begin |
| F1 (grief) end−begin | Negative (↓) | Negative slope |
| Q_repair end−begin | Positive (↑) | Positive slope |
| F2 (anger) end−begin | Positive (crisis escalation) | Positive slope for higher-rated books |
| F3 (anxiety) end−begin | Positive (crisis escalation) | Positive slope for higher-rated books |

**Analysis Approach:**
- **Arc contrasts**: end−begin, middle−begin deltas computed in measurement pipeline
- **Bootstrap inference**: Test if deltas predict `rating_mean` (controlling for `log_rating_count`)
- **Interpretation**: Higher-rated books show better pacing (lower baseline negativity, stronger late crisis escalation)

**Key Finding:**
- **F2_anger_frustration end−begin**: β≈ +0.24, CI [+0.08, +0.41], P=0.995
- **F3_anxiety_worry end−begin**: β≈ +0.19, CI [+0.02, +0.36], P=0.981
- **Interpretation**: Higher-rated books have lower baseline negativity but stronger third-act crisis escalation

**Deliverable:** Arc contrast results in `inference_arc_trajectory_tests.csv`

---

## 6. Goodreads Metadata Validation (Ratings + Number of Voters)

Two signal channels:
- **avg_rating** (quality perception)
- **n_ratings** (visibility/popularity/market reach)

Treat them differently.

### 6.1 Basic Checks

- Correlate indices with avg_rating
- Correlate indices with log(n_ratings) separately
- Check whether Top/Middle/Trash differs primarily by avg_rating, n_ratings, or both

### 6.2 Weighted Outcomes and Noise Control

avg_rating from 50 voters is noisier than from 50,000.

**Good practice:**
- Use **weights** based on n_ratings (or its sqrt/log) in rating regression, OR
- Use Bayesian-adjusted rating, then regress on that

**Deliverable:** "themes that predict perceived quality" vs "themes that predict mass appeal"

---

## 7. Modeling Strategy (Beyond Group Tests)

### 7.1 Predictive Models (Interpretability-First)

**Logistic Regression: Top vs Trash**
- Predictors: your indices + controls
- Controls: author fixed effects (or random effects), length, year

**OLS / Robust Regression: avg_rating**
- Predictors: indices + controls

**Why do this?**
- Tells you whether themes remain associated with ratings once you account for confounds
- Prevents over-interpreting single-topic effects that are actually proxies for author, length, or era

### 7.2 Multicollinearity and Index Redundancy

Your indices will correlate (e.g., A and G might travel together).

**Plan:**
- Examine correlation matrix among indices
- If two indices are near-duplicates: either combine them or pick one as primary
- Consider PCA as a descriptive tool (not as main theoretical claim)

---

## 8. Robustness and Credibility Checks

### 8.1 Sensitivity to Topic Assignment Ambiguity

- Recompute key indices with "ambiguous topics" removed
- Or do two versions: strict mapping vs generous mapping

If results hold → confidence jumps.

### 8.2 Alternative Aggregation

Compare:
- "Sum of probs" aggregation
- "Presence threshold" aggregation (topic present if prob > τ)

Helps when distributions are zero-inflated.

### 8.3 Leave-One-Author-Out Validation

- Recompute main hypothesis results while leaving out each author one at a time
- If H1–H6 persist → can credibly claim "not just an author artifact"

### 8.4 Bootstrapped Confidence Intervals

- Bootstrap books within tiers to get uncertainty bands for indices and arc trends

---

## 9. Mixed-Methods Integration (Making Quantitative Results Readable)

### 9.1 Quant → Qual Sampling Protocol

For each hypothesis index, pick:
- A few "high-index Top books"
- A few "low-index Top books"
- A few "high-index Trash books"
- A few "low-index Trash books"

Then sample passages from segments where relevant topics are strongest.

**Produces narrative evidence like:**
- What "commitment language" looks like in Top vs Trash
- How "repair" is enacted (dialogue style, apology structure)
- Whether explicitness in Top tends toward "aftercare + reflection" rather than purely mechanical depiction

### 9.2 "Scene Kits" Using S (Scene Anchors)

Use S category to locate recurring "set pieces":
- Luxury arrival scenes, boardroom scenes, penthouse domestic scenes, holiday ritual scenes

Then compare how Top vs Trash uses the same scene kit differently.

**This bridges distant reading and literary interpretation.**

---

## 10. Practical Chapter-by-Chapter Structure for Thesis/Paper

A clean narrative arc matching the bottom-to-top approach:

1. **Topic-level landscape**: what differs across tiers at the finest granularity
2. **Taxonomy-level structure**: main groups, then subgroup panels per main group
3. **Within-subgroup drivers**: the key topics that explain subgroup differences
4. **Theory composites and indices**: construction + validity checks
5. **Hypothesis tests**: H1–H6 with effect sizes, models, and arc analyses
6. **Goodreads validation**: quality vs popularity channels
7. **Qualitative triangulation**: close readings targeted by indices
8. **Robustness**: mapping sensitivity, author effects, bootstraps

---

## Index Blueprint Quick Reference

Compact "index → taxonomy nodes" cheat sheet:

| Index | Formula |
|-------|---------|
| **Love-over-Sex** | (4.2 + 4.6 + 2.1 + selected 2.2 + 3.1) − (2.3) |
| **HEA Index** | (4.2 + repair part of 4.6) + (2.4 + 7.4 + date-meal part of 7.3) |
| **Luxury Saturation** | (5.2 + 5.3) + luxury-coded (7.2) + appearance/status (1.2) |
| **Corporate Frame Share** | 5.1 (+ 5.4 optional) |
| **Family/Fertility Index** | 4.3 + 4.4 (+ supportive 4.5 if you want "village") |
| **Dark-vs-Tender** | (6.2 + 6.3 + coercive 5.5 + 3.2/3.3/3.4) − (2.1 + 3.1) |
| **Miscommunication Balance** | (repair subset 4.6 + commitment 4.2 + tenderness 2.1 + safety 3.1) − (4.1) |
| **Protective–Jealousy Delta** | (care 1.4 + safety 3.1 + supportive commitment 4.2/4.6) − (jealousy topics in 6.1 + possessive power in 5.5 + hostility 3.3) |
| **Growth/Recovery** | 1.4 + 3.6 (+ recovery-coded 6.3) |

---

## Helper Functions

### Cliff's Delta Implementation
```python
def cliffs_delta(x, y):
    """Compute Cliff's delta effect size."""
    # Implementation
    pass
```

### Topic Prevalence Metrics
```python
def compute_topic_prevalence(book_topic_probs, threshold=0.001):
    """Compute prevalence, mass, and concentration for each topic."""
    pass
```

### Distribution Comparison
```python
def compare_topic_distributions(topic_id, book_topic_probs, books_meta):
    """Compare topic distributions across rating classes."""
    pass
```

### Plotly Figure Helper
```python
def show_plotly_fig(fig, save_html=True, output_dir=FIG_DIR):
    """Display Plotly figure with fallback to HTML save."""
    pass
```

---

## Output Structure

```
results/stage10_correlation_analysis/
├── 00_data_preparation/          # Data preparation outputs
│   ├── topic_probabilities/
│   │   ├── book_topic_probs.parquet
│   │   └── chapter_topic_probs.parquet
│   ├── taxonomy_radway_eda/
│   │   └── topic_lookup.parquet
│   └── book_features/
│       ├── book_taxonomy_main_props_wide.parquet
│       └── book_taxonomy_main_props_long.parquet
├── 01_topic_analysis/            # Individual topic distributions
│   ├── figures/
│   │   ├── topic_distributions/
│   │   ├── topic_leaderboards/
│   │   └── ...
│   └── tables/
│       ├── topic_health_table.parquet
│       ├── topic_leaderboard_all.parquet
│       └── ...
├── 02_taxonomy_group_analysis/    # Taxonomy group comparisons
│   ├── figures/
│   └── tables/
└── measurement_v5/                # Composite indices & hypothesis testing
    ├── bundle/                     # Exported tables for inference
    │   ├── book_indices_raw.parquet
    │   ├── book_indices_z.parquet
    │   ├── segment_indices_raw.parquet
    │   ├── segment_indices_z.parquet
    │   ├── arc_contrasts_sum.parquet
    │   └── ...
    ├── audit/                      # Pipeline diagnostics
    │   ├── pipeline_audit.parquet
    │   ├── composite_registry.parquet
    │   ├── composite_diagnostics.parquet
    │   └── ...
    └── bundle/inference_outputs/  # Hypothesis testing results
        ├── inference_core_level_effects.csv
        ├── inference_macro_level_effects.csv
        └── ...
```

---

## Notes

1. **Compositional Data**: Topic probabilities sum to 1 per book. Raw comparisons are informative, but confirm key claims with log-ratio or Dirichlet modeling.

2. **Multiple Comparisons**: With 368 topics, expect many "significant" results by chance. Use FDR correction AND effect size thresholds.

3. **Author Effects**: Romance authors can imprint topics strongly. Always check author dominance before interpreting tier associations.

4. **Two Normalization Approaches**: Z-scored shares for overview; log-ratios for balance hypotheses. Use both and compare.

5. **Performance**: For 368 topics × 92 books, batch processing and vectorized operations are essential.

6. **BERTopic Reference**: Refer to `approximate_distribution` method documentation for understanding how probabilities are computed.
