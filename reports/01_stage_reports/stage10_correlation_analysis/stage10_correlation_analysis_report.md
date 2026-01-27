# Stage 10: Correlation Analysis Report

**Sample**: N=92 billionaire romance novels (pilot)  
**Analysis Type**: Effect-size-focused with bootstrap confidence intervals  
**Source Notebooks**: `notebooks/07_analysis/`, `04_hypothesis_testing_inference_only_v4_1_macro_axes_bayes.ipynb`

---

## 1. Research Objectives

Stage 10 examines the relationship between thematic content and book outcomes (reach and perceived quality) in billionaire romance novels. Analysis proceeds at three levels:

1. **Macro-axis level**: Weighted combinations of thematic predictors
2. **Topic-level**: Individual BERTopic topic probabilities (368 topics)
3. **Taxonomy-group level**: Aggregated probability mass across 8 main groups and 27 subgroups

**Key distinction**: We separate two outcome channels:
- **Reach/visibility**: `log_rating_count` (number of Goodreads ratings)
- **Perceived quality**: `rating_mean` and Bayesian-adjusted `avg_rating_bayes`

---

## 2. Methods

### 2.1 Data Preparation

- **Topic probabilities**: Sentence-level BERTopic assignments aggregated to book level
- **Normalization**: Probabilities sum to 1.0 per book (validated: 0% NaN, 100% coverage)
- **Tertile probabilities**: Book split into begin/middle/end segments for arc analysis
- **Rating tiers**: Books classified as Top/Middle/Trash based on Goodreads ratings (30/32/30 distribution)

### 2.2 Statistical Approach

| Analysis Level | Tests | Effect Size |
|---------------|-------|-------------|
| Macro-axis | Bootstrap regression | Standardized β with 95% CI, P(β>0) |
| Topic-level | Kruskal-Wallis, Mann-Whitney U | Cliff's delta (δ) |
| Taxonomy-group | Kruskal-Wallis, Mann-Whitney U | Cliff's delta (δ) |

**Two-gate filtering rule** (for topic/taxonomy analysis):
- Gate 1 (Effect): \|δ\| ≥ 0.20
- Gate 2 (Impact): mass ≥ 0.002 OR \|mean diff\| ≥ 0.001

**Rationale**: With N=30 per tier, FDR-corrected p-values are underpowered. Effect sizes provide practical interpretability.

---

## 3. Key Results

### 3.1 Macro-Axis Effects

**Macro-axes** are weighted combinations of CORE predictors:

| Axis | Key Components |
|------|---------------|
| AX_status_dominance | Power/wealth/luxury + alpha guarding |
| AX_payoff_safety | Emotional safety + repair + protective caretaking |
| AX_negative_affect | Anger/frustration + anxiety/worry + sadness/grief |
| AX_explicitness | Explicit sexual content |

#### Effects on Reach (log_rating_count)

| Predictor | β_std | 95% CI | P(β>0) |
|-----------|-------|--------|--------|
| AX_status_dominance | **0.461** | [0.284, 0.613] | 1.000 |
| AX_drama_obstacle | 0.335 | [0.074, 0.574] | 0.993 |
| AX_payoff_safety | 0.327 | [0.141, 0.500] | 0.999 |
| AX_explicitness | **-0.297** | [-0.503, -0.057] | 0.004 |

#### Effects on Quality Beyond Reach (rating_mean, controlling log_rating_count)

| Predictor | β_std | 95% CI | P(β>0) |
|-----------|-------|--------|--------|
| AX_payoff_safety | **0.207** | [-0.005, 0.413] | 0.974 |
| AX_status_dominance | 0.006 | [-0.184, 0.195] | 0.525 |
| AX_explicitness | -0.175 | [-0.356, 0.019] | 0.040 |

**Interpretation**: Status/dominance drives reach but not quality. Payoff/safety predicts both reach and quality (the clearest quality signal). Explicitness is negatively associated with both.

### 3.2 CORE Predictor-Level Effects

Individual CORE predictors (before aggregation into macro-axes):

#### Top Predictors for Reach (log_rating_count)

| Predictor | β_std | 95% CI | P(β>0) |
|-----------|-------|--------|--------|
| R2_alpha_guarding | **0.44** | [0.23, 0.60] | 1.00 |
| D_power_wealth_luxury__pc1 | **0.37** | [0.20, 0.55] | 1.00 |
| A2_emotional_safety__pc1 | **0.32** | [0.13, 0.50] | 0.998 |
| Q_repair | 0.23 | [0.02, 0.41] | 0.985 |
| J_social_support_kin | 0.19 | — | 0.95 |

#### Top Predictors for Quality Beyond Reach (rating_mean)

| Predictor | β_std | 95% CI | P(β>0) |
|-----------|-------|--------|--------|
| R1_protective_caretaking | **0.22** | [0.05, 0.36] | 0.995 |
| A2_emotional_safety__pc1 | 0.15 | — | 0.95 |

**Plain-language**: Books with more ratings combine billionaire/status signals, dominance cues, and emotional payoff signals (repair + safety). Books rated higher (controlling for popularity) feel **caring and emotionally safe**.

### 3.3 Ridge Regression Validation

Joint ridge regression (all CORE predictors simultaneously) confirms the same patterns persist under collinearity:

**Positive for reach**: Alpha guarding, luxury, repair, kin support, safety  
**Negative for reach**: Explicit erotics, domestic nesting, vices/addictions

**Key insight**: Popularity effects aren't artifacts of univariate analysis—the "billionaire-romance package" persists under joint modeling.

### 3.4 Arc Effects (Narrative Pacing)

Arc predictors measure topic change across book segments (`end_minus_begin`, `middle_minus_begin`).

**Positive arcs** (higher-rated books show increase toward end):

| Predictor | β_std | P(β>0) |
|-----------|-------|--------|
| F2_anger_frustration__end_minus_begin | **0.238** | 0.995 |
| F3_anxiety_worry__end_minus_begin | **0.190** | 0.981 |

**Negative arcs** (higher-rated books show decrease):

| Predictor | β_std | P(β>0) |
|-----------|-------|--------|
| J_social_support_kin__end_minus_begin | **-0.177** | 0.019 |
| C_explicit_eroticism__middle_minus_begin | **-0.112** | 0.089 |

**Interpretation**: Higher-rated books show late-story crisis escalation (anger/anxiety increases) rather than diffuse wrap-up. Increasing end-of-book kin emphasis or explicit content is associated with lower ratings.

### 3.5 Topic-Level Analysis

**85 discriminative topics** identified from 342 total (two-gate filtering).

#### Top-Tier Associated Topics (Tier 1, δ ≥ 0.35)

| Topic | Cliff's δ | Taxonomy Group |
|-------|-----------|----------------|
| Married Couple's Affectionate Stares | 0.453 | Sexuality, Attraction & Intimacy |
| Frightened Admissions | 0.420 | Emotions, Cognition & Inner Life |
| Emotional Relationship Delusion | 0.404 | Emotions, Cognition & Inner Life |
| Lip Biting During Intimacy | 0.364 | Sexuality, Attraction & Intimacy |

**Pattern**: Top-tier books emphasize (1) psychological credibility scenes (fear admissions, emotional delusion) and (2) embodied intimacy cues (non-explicit but tactile).

#### Trash-Tier Associated Topics

| Topic | Cliff's δ | Taxonomy Group |
|-------|-----------|----------------|
| Dominatrix Session | -0.353 | Sexuality, Attraction & Intimacy |
| Work At Desk | -0.329 | Work, Wealth, Status & Institutions |
| Exiting Through Doorways | -0.307 | Spaces, Time, Activities & Objects |

**Pattern**: Trash-tier books show more explicit sexual content and procedural/transition scenes (scene scaffolding, pacing filler).

### 3.6 Taxonomy-Group Analysis

#### Main-Group Effects (8 Macro Buckets)

| Group | Top vs Trash δ | Direction |
|-------|----------------|-----------|
| Relationship Trajectory (Main Couple) | +0.37 | Top higher |
| Social World Outside Couple | +0.30 | Top higher |
| Sexuality, Attraction & Intimacy | -0.25 | Trash higher |

**Interpretation**: Small but interpretable rebalancing—Top books allocate more to relationship dynamics and social context, less to pure sexuality mass.

#### Strongest Subgroup Effect

**Beliefs, Values & Moral Reflection**: δ = +0.46, adjusted p ≈ 0.008  
- Top books allocate more mass to value/identity/moral reflection moments
- Driver topics: "Feminist Identity Affirmation", "Sisters Reflecting on Childhood"

#### Other Subgroup Patterns

**Top-tier books lean into**:
- Values/identity reflection
- Professional/workplace interaction
- Time framing (temporal structuring)

**Trash-tier books lean into**:
- Distress/negative emotion intensity
- Threat/coercion/violence texture
- Procedural scene motion

### 3.7 Diversity Finding

**Higher-tier books are more thematically diverse**:

| Metric | Bad | Mid | Good |
|--------|-----|-----|------|
| Entropy | 5.33 | 5.43 | **5.48** |
| Effective topics | 207 | 229 | **240** |
| Richness | 247 | 258 | **265** |

**Interpretation**: Higher-quality books combine more thematic ingredients rather than over-concentrating on a few motifs. This is a structural property independent of specific romance tropes.

---

## 4. Author Dominance Control

**30 topics** show high author dominance (>50% from single author). These require control in modeling:
- Examples: "Doorway Questions And Answers" (100% Meghan_Quinn), "Archery Practice" (100% LJ_Shen)
- Recommendation: Exclude from tier interpretation, include author fixed/random effects in modeling

---

## 5. Summary of Findings

### Reach vs Quality Are Different Channels

| Predictor | Reach | Quality |
|-----------|-------|---------|
| Status/dominance | **Strong positive** | Null |
| Payoff/safety | Moderate positive | **Positive** |
| Explicitness | Negative | Negative |

### What Predicts Higher Quality (Beyond Reach)

1. **Payoff/safety content**: Emotional safety, repair, protective caretaking
2. **Psychological credibility**: Fear admissions, emotional processing
3. **Late-story crisis escalation**: Anger/anxiety increases toward end
4. **Thematic diversity**: More even distribution across topics
5. **Values/moral reflection**: Identity and belief exploration moments

### What Predicts Lower Quality

1. **Explicit erotics**: Particularly when increasing mid-to-end
2. **Procedural filler**: Transition scenes, scene scaffolding
3. **Distress intensity**: High-intensity panic, breakdown scenes
4. **Diffuse endings**: Kin/social support emphasis at end

---

## 6. Methodological Contributions

1. **Two-gate filtering**: Balances statistical rigor with practical interpretability for small-N exploratory research
2. **Two-tier structure**: Distinguishes high-confidence (δ ≥ 0.35, p < 0.05) from exploratory findings
3. **Reach/quality separation**: Distinguishes market visibility from reader evaluation
4. **Author dominance metrics**: Objective control for author-style topics

---

## 7. Tier Structure: Two-Channel Validation

Rating tiers differ in **both** channels, not just one:

| Tier | N | avg_rating | n_ratings | Interpretation |
|------|---|------------|-----------|----------------|
| **bad** | 30 | 3.77 | ~48k | Lower quality + lower visibility |
| **mid** | 32 | 4.01 | ~44k | Moderate quality + visibility |
| **good** | 30 | 4.22 | ~116k | Higher quality + **much higher visibility** |

**Implication**: "Good vs bad" is both higher perceived quality (4.22 vs 3.77) AND higher visibility (116k vs 48k ratings). Top-tier books reach a broader audience.

---

## 8. Claim Guidance

### What We Can Safely Claim (Pilot-Appropriate)

- "We separate popularity (visibility) from perceived quality (ratings)."
- "Popularity is associated with a coherent macro package: status/dominance + drama/obstacles + payoff/safety."
- "Perceived quality beyond reach is most consistently associated with payoff/safety, and negatively with baseline negative affect and explicitness."
- "Arc features suggest higher-rated books show stronger late escalation of tension (third-act crisis), consistent with romance narrative structure."

### Avoid Saying

- "X causes popularity." (Correlational, not causal)
- "Explicit content reduces quality in general." (Corpus-specific pattern in billionaire romance)

---

## 9. Limitations

- **Pilot sample**: N=92 (effect-size focused, not p-value focused)
- **Statistical power**: FDR-corrected p-values underpowered at this sample size
- **Causality**: Correlational findings, not causal claims
- **Topic labels**: LLM-generated labels are imperfect summaries of topic content
- **Genre specificity**: Findings specific to billionaire romance subgenre

---

## 10. Data and Code

**Results directories**:
- `results/stage10_correlation_analysis/` — Main analysis outputs
- `results/measurement_v5/bundle/inference_outputs/` — Hypothesis testing outputs

| Subdirectory | Contents |
|--------------|----------|
| `data_preparation/` | Topic probabilities, book features, diagnostics |
| `01_topic_analysis/` | Topic-level comparisons, tables, figures |
| `02_taxonomy_group_analysis/` | Group-level comparisons, diversity metrics |

**Key output files**:
- `tier_summary_goodreads_channels.csv` — Tier differences in both channels
- `goodreads_index_correlations.csv` — Simple correlations (Pearson + Spearman)
- `macro_axes_definition.csv` — Macro axis definitions
- `top10_macro_level_*.csv` — Top macro-axis effects for each outcome
- `ridge_joint_coeffs_*.csv` — Joint ridge coefficients
- `partial_corr_*.csv` — Partial correlations
- `cv_repeats_summary.csv` — Cross-validation results

**Notebooks**: `notebooks/07_analysis/`, `notebooks/08_power_analysis/`

---

*Report Generated: January 2025*
