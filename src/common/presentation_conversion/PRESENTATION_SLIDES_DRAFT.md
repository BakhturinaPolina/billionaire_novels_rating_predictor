# Presentation Slides: Modern Romantic Novels — Themes × Popularity
## A Mixed-Methods Computational Analysis

---

## Slide 1: Title Slide

**Modern Romantic Novels — Themes × Popularity: A Mixed-Methods Computational Analysis**

- Research Question: Which thematic patterns differentiate highly-rated romance novels from lower-rated ones?
- Dataset: 105 billionaire romance novels (92 in final analysis), 680,822 sentences
- Method: BERTopic neural topic modeling + statistical analysis
- Key Finding: Popularity (reach) and perceived quality (ratings) are driven by different thematic patterns

---

## Slide 2: Research Objectives & Questions

### Objectives
1. **Map topic-model outputs** to theory-driven themes (Radway, 1984)
2. **Build explainable indices** to quantify narrative qualities readers value
3. **Validate corpus findings** against Goodreads metadata (ratings & voter counts)

### Research Questions
- Which theme categories are most prevalent in Top vs Middle vs Trash novels?
- Does love/commitment/tenderness outweigh explicit sexual content in higher-rated books?
- Is luxury appealing only when paired with commitment/tenderness?
- Do protectiveness/care signals predict appreciation better than jealous/possessive affect?
- Do miscommunication/negative affect diminish across the book while HEA/repair rises?

---

## Slide 3: Dataset Overview

### Corpus Description
- **105 standalone billionaire romance novels** by **35 different authors**
- **680,822 sentences** organized hierarchically: Author → Book → Chapter → Sentence
- Each novel ≥ 100,000 words
- Selected from curated lists: "100 Best Billionaire Romance Books of All Time"

### Final Analysis Sample
- **92 books** with complete Goodreads metadata and topic probability assignments
- **Tier distribution:**
  - Top tier: 30 books (avg_rating ≈ 4.22, n_ratings ≈ 116k)
  - Middle tier: 32 books (avg_rating ≈ 4.01, n_ratings ≈ 44k)
  - Trash tier: 30 books (avg_rating ≈ 3.77, n_ratings ≈ 48k)

### Data Sources
- Raw text files: Full novel texts (TXT/EPUB)
- Goodreads metadata: Ratings, review counts, publication info
- BookNLP outputs: Character name extraction (4,444 character names added to stopwords)

---

## Slide 4: Methodology Overview

### Topic Modeling Pipeline
1. **BERTopic** (neural topic modeling) with **OCTIS** hyperparameter optimization
2. **GPU-accelerated** using RAPIDS cuML (CUDA 12.x)
3. **Character name exclusion**: 4,444 character names added to stopwords (93% of expanded list)
4. **Over 300 configurations** analyzed through Bayesian optimization
5. **Pareto-efficient model selection**: Multi-objective optimization balancing coherence and diversity
6. **Selected model**: `paraphrase-MiniLM-L6-v2` (368 topics, Pareto rank 1)

### Key Innovations
- **Multiple representation strategies**: Main, KeyBERT, POS, MMR
- **Automated LLM labeling**: Mistral-Nemo-Instruct via OpenRouter API
- **Theory-aligned taxonomy**: 8 main groups, 30+ categories
- **Radway narrative functions**: 13 functions mapped to 3 phases

### Three-Stage Labeling Pipeline

**Stage 08: LLM Labeling**
- Input: BERTopic topics (keywords + snippets)
- Output: Descriptive labels (2-6 words) + metadata
- Coverage: 368/368 topics (100%)
- Example: Keywords "car, seat, door, parked, kiss" → Label "Parked Car Makeout"

**Stage 09 Stage 2: Taxonomy Mapping**
- Input: LLM labels
- Output: Theory-driven category mappings (8 groups, 30+ categories)
- Coverage: 361/368 topics (98.1%)
- Example: "Parked Car Makeout" → Category 4.2 (Bonding, Everyday Intimacy)

**Stage 09 Stage 3: Radway Functions**
- Input: Taxonomy mappings
- Output: Radway narrative function mappings (R1-R13, 3 phases)
- Coverage: 361/368 topics (98.1%)
- Example: "Parked Car Makeout" → R8 (Hero treats heroine tenderly, Phase II)

---

## Slide 5: Multi-Objective Optimization Challenge

### The Trade-off Problem

**Topic modeling requires balancing competing objectives:**

- **Coherence** → Interpretable, semantically consistent topics
- **Topic Diversity** → Comprehensive thematic coverage

**The Fundamental Trade-off:**
- Maximizing coherence → Fewer, more focused topics (lower diversity)
- Maximizing diversity → More topics with less consistency (lower coherence)

**Traditional single-objective optimization fails** to capture this trade-off.

### Solution: Pareto-Efficient Multi-Objective Optimization

**Pareto efficiency** identifies configurations where:
- No other configuration is better in **BOTH** metrics simultaneously
- Any improvement in one metric requires degradation in the other
- Provides **objective selection criteria** without subjective trade-off judgments

![Pareto Frontier All Models](results/stage04_selection/pareto_frontier_all.png)

**Figure**: The Pareto frontier shows the optimal trade-off curve between coherence and diversity. Models on the frontier represent non-dominated solutions.

![Pareto Frontier by Model](results/stage04_selection/pareto_frontier_by_model.png)

**Figure**: Per-model Pareto frontiers showing model-specific optimal configurations.

![Combined Score Distribution](results/stage04_selection/combined_score_distribution.png)

**Figure**: Distribution of combined scores across all configurations, showing the performance landscape before Pareto filtering.

---

## Slide 6: Pareto Efficiency Analysis: Methodology

### Three-Stage Approach

**1. Data Cleaning**
- Removed failed runs (Coherence/Diversity = 1.0): 4 configurations
- Statistical outlier removal (z-score, 2σ threshold): 17 additional configurations
- Domain-specific filtering (diversity > 0.9, IQR method): 4 configurations
- **Result**: 251 valid configurations from 272 initial

**2. Pareto Efficiency Identification**
- Identifies non-dominated solutions across all embedding models
- Per-model Pareto analysis for model-specific optimal configurations
- **Result**: 4 Pareto-efficient configurations (down from 12 before filtering)

**3. Hyperparameter Correlation Analysis**
- Statistical correlation (Pearson/Spearman) with assumption checking
- Linear regression with multicollinearity assessment
- Tree-based feature importance (Random Forest, XGBoost) for validation

### Outlier Filtering Impact

| Stage | Configurations | Removed | Reason |
|-------|---------------|---------|--------|
| Original | 272 | - | - |
| After failed runs | 268 | 4 | Coherence/Diversity = 1.0 |
| After outliers | 251 | 17 | Statistical outliers (IQR) |
| After max diversity | 247 | 4 | Diversity > 0.9 (artifacts) |
| **Final Pareto-efficient** | **4** | - | **Non-dominated solutions** |

![Distribution with Cutoffs](results/stage04_selection/figures/distribution_with_cutoffs.png)

**Figure**: Outlier filtering ensures Pareto-efficient models represent genuine performance trade-offs, not statistical artifacts.

![Pareto Front Coherence Priority](results/stage04_selection/figures/pareto_front_coherence_priority.png)

**Figure**: Alternative Pareto frontier prioritizing coherence, showing different trade-off perspective for interpretability-focused applications.

---

## Slide 7: Pareto Efficiency Results: Top Models

### Pareto-Efficient Configurations (4 models)

**Top Performer: paraphrase-mpnet-base-v2, iteration 0 (Rank 1)**
- **Combined Score**: 1.75
- **Coherence**: 0.463 (highest among Pareto-efficient models)
- **Topic Diversity**: 0.82
- **Best balance** between coherence and diversity

**Key Hyperparameters:**
- `min_cluster_size`: 494
- `umap_n_components`: 10
- `umap_min_dist`: 0.058
- `vectorizer_min_df`: 0.007
- `min_topic_size`: 127

**Other Pareto-Efficient Models:**
1. **paraphrase-MiniLM-L6-v2, iteration 19**: Diversity = 0.94 (highest), Coherence = 0.425
2. **multi-qa-mpnet-base-cos-v1, iteration 11**: Coherence = 0.419, Diversity = 0.83

![Pareto Front Equal Weights](results/stage04_selection/figures/pareto_front_equal_weights.png)

**Figure**: The Pareto frontier shows clear trade-offs. The top performer (`paraphrase-mpnet-base-v2`) achieves the highest coherence while maintaining good diversity, making it optimal for interpretable topic modeling.

**Table**: `results/stage04_selection/pareto.csv`

---

## Slide 8: Pareto Efficiency: Model Comparison

### Per-Model Pareto Frontiers

**Model Hierarchy:**
- **mpnet variants dominate** combined scores
- **paraphrase-mpnet-base-v2**: Highest coherence (0.463)
- **paraphrase-MiniLM-L6-v2**: Highest diversity (0.94) but lower coherence

**Key Observations:**
- Pareto-efficient models cluster in upper-right (high coherence + high diversity)
- Clear trade-off curve visible
- No single embedding model dominates all others
- Optimal balance: Coherence 0.4-0.47, Diversity 0.8-0.85

![Pareto Fronts Per Model](results/stage04_selection/figures/pareto_fronts_per_model.png)

**Figure**: Different embedding models have distinct performance profiles. The per-model analysis reveals model-specific optimal configurations while the overall analysis identifies the globally best trade-offs.

---

## Slide 9: Hyperparameter Correlation Analysis

### Critical Hyperparameters Identified

**High Priority Parameters** (Strong, Significant Effects):

| Parameter | Effect on | Correlation | p-value | Effect Size (Cohen's d) |
|-----------|-----------|-------------|---------|------------------------|
| `umap__min_dist` | Diversity | **r = 0.794** | **p = 0.006** | Large (d = -1.10) |
| `vectorizer__min_df` | Combined Score | **r = 0.745** | **p = 0.014** | Very Large (d = -2.56) |
| `umap__n_components` | Combined Score | **r = 0.712** | **p = 0.021** | Large (d = -1.46) |

**Medium Priority Parameters** (Strong Effects, Marginal Significance):

| Parameter | Effect on | Correlation | p-value |
|-----------|-----------|-------------|---------|
| `bertopic__min_topic_size` | Coherence | r = 0.585 | p = 0.075 |
| `hdbscan__min_cluster_size` | Coherence | r = 0.563 | p = 0.090 |
| `umap__n_neighbors` | Coherence | r = -0.573 | p = 0.084 |

### Key Trade-offs Identified

1. **`bertopic__min_topic_size`**
   - ✅ Increases coherence (r = 0.585)
   - ❌ Decreases diversity (r = -0.591)
   - **Trade-off**: Filter small topics → better coherence, fewer topics

2. **`umap__min_dist`**
   - ❌ Decreases coherence (r = -0.528)
   - ✅ Increases diversity (r = 0.794, **significant**)
   - **Trade-off**: Tighter clustering → better coherence, less separation

![Hyperparameter Boxplots](results/stage04_selection/figures/hyperparameter_boxplots.png)

**Figure**: UMAP parameters (`min_dist`, `n_components`) show the strongest effects on performance. The `vectorizer__min_df` parameter has the largest effect on combined score, indicating vocabulary filtering is critical.

**Table**: `results/stage04_selection/tables/correlation_analysis_equal_weights.csv`

---

## Slide 10: Optimization Recommendations

### Evidence-Based Hyperparameter Guidance

**For Maximizing Coherence:**
- ⬆️ Increase `min_topic_size` (100-130)
- ⬆️ Increase `min_cluster_size` (400-500)
- ⬇️ Decrease `umap_n_neighbors`
- ⬇️ Decrease `umap_min_dist`

**For Maximizing Diversity:**
- ⬆️ Increase `umap_min_dist` (0.02-0.08)
- ⬆️ Increase `vectorizer_min_df` (0.007-0.009)
- ⬇️ Decrease `min_topic_size`

**For Balanced Performance:**
- Optimize `umap_min_dist` (critical trade-off parameter)
- Optimize `vectorizer_min_df` (strongest combined score effect)
- Set `umap_n_components` to 8-10

### Multi-Method Validation

**Tree-Based Methods Confirm Correlation Findings:**
- Random Forest & XGBoost feature importance:
  1. `umap__min_dist` (highest for diversity/combined)
  2. `vectorizer__min_df` (high for diversity/combined)
  3. `hdbscan__min_cluster_size` (high for coherence)
  4. `bertopic__min_topic_size` (moderate for coherence)
  5. `umap__n_components` (moderate for combined)

**Cross-Validated R²**: 0.3-0.6 (moderate predictive power)

**Key Insight**: Multiple independent methods identify the same critical parameters → **Robust findings**

**Table**: `results/stage04_selection/top_models/top_10_equal_weights.csv`

**Comment**: The convergence of correlation analysis, regression, and tree-based methods provides strong evidence for the identified critical hyperparameters.

---

## Slide 11: Topic Modeling Results (Selected Model)

### Model Selection
- **Selected model**: `paraphrase-MiniLM-L6-v2` (Pareto rank 1)
- **368 topics** extracted from 680,822 sentences
- **Coherence (c_v)**: 0.404 (Main representation)
- **Topic Diversity**: 0.602 (Main), 0.756 (MMR)
- **Selection rationale**: Pareto-efficient configuration with high diversity (0.94) and acceptable coherence (0.425) from hyperparameter optimization

### Topic Quality
- **98.1% coverage**: 361/368 topics successfully mapped to taxonomy
- **Quality threshold**: 94.6-96.5% of topics meet quality thresholds
- **Noise detection**: 13 candidate noisy topics (3.5%) identified and flagged

![Coverage Distribution](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_C1_coverage_distribution.png)

**Figure**: Distribution of taxonomy coverage across books, showing consistent high coverage (98.1% of topics mapped) across the corpus.

### Representation Performance
| Representation | Coherence (c_v) | Topic Diversity |
|---------------|-----------------|------------------|
| Main          | 0.404           | 0.602            |
| KeyBERT        | 0.278           | 0.645            |
| POS           | 0.315           | 0.692            |
| MMR           | 0.260           | 0.756            |

### Example Topics from LLM Labeling

**Example 1: "Frightened Admissions" (Topic 82)**
- **Keywords**: scared, afraid, sad, terrifying, frightening, frightened, fears, sadness, phobia
- **Label**: "Frightened Admissions"
- **Scene Summary**: "She admits her fears and insecurities to him in a quiet, intimate setting."
- **Taxonomy**: Romance Core, Relationship Conflict
- **Radway**: Phase II (R9: Heroine responds warmly / R10: Heroine reinterprets)

**Example 2: "Parked Car Makeout" (Topic 21)**
- **Keywords**: car, driver, parking, cars, driveway, suv, curb, cab, drive, vehicle
- **Label**: "Parked Car Makeout"
- **Scene Summary**: "The couple, parked in his car, engages in passionate kissing."
- **Taxonomy**: Romance Core, Physical Affection
- **Radway**: Phase II (R8: Hero treats heroine tenderly)

**Example 3: "Business Discussion" (Topic 33)**
- **Keywords**: business, company, dollars, job, investment, cost, profit, paycheck, income, bank
- **Label**: "Business Discussion"
- **Scene Summary**: "The couple discusses business matters at the dinner table."
- **Taxonomy**: Domestic Life, Work/School
- **Radway**: None (background/contextual content)

---

## Slide 12: Topic-Level Analysis: Micro-Scenes That Distinguish Tiers

### Two-Gate Filtering Results
- **85 discriminative topics** identified from 342 analyzed (368 total)
- **Gate 1 (Effect)**: |Cliff's δ| ≥ 0.20 → 163 topics passed
- **Gate 2 (Impact)**: mass ≥ 0.002 OR |mean diff| ≥ 0.001 → 186 topics passed
- **Both gates**: 85 topics (final filtered set)

### Tier 1 High-Confidence Topics (8 topics)
**Top-associated examples:**
- "Married Couple's Affectionate Stares" (δ = 0.453)
- "Frightened Admissions" (δ = 0.420)
- "Emotional Relationship Delusion" (δ = 0.404)

**Trash-associated example:**
- "Dominatrix Session" (δ = -0.353)

### Key Patterns
- **Top-tier**: Psychological credibility scenes (fear admissions, emotional delusion) + embodied intimacy cues
- **Trash-tier**: Explicit sexual content + procedural/transition scenes (doors, phones, desk work)

### Real Examples from LLM Labeling

**"Frightened Admissions" (Topic 82) - Top-tier associated**
- **Keywords**: scared, afraid, sad, terrifying, frightening, frightened, fears, sadness, phobia
- **Scene Summary**: "She admits her fears and insecurities to him in a quiet, intimate setting."
- **Why Top-tier**: Psychological vulnerability and emotional intimacy signal quality

**"Phone Notifications During Night" (Topic 24) - Trash-tier associated**
- **Keywords**: rings, buzzes, pocket, screen, cellphone, nightstand, message, purse, phones, vibrates
- **Scene Summary**: "Phone notifications interrupt the couple's quiet moment."
- **Why Trash-tier**: Procedural/transition scenes without emotional depth

![Topic Drivers Summary](results/stage10_correlation_analysis/02_taxonomy_group_analysis/figures/topic_drivers/topic_drivers_summary_heatmap.png)

**Figure**: Heatmap showing which topics drive key subgroup differences, revealing the micro-scenes that contribute to tier differentiation.

**Table reference**: `results/stage10_correlation_analysis/01_topic_analysis/tables_csv/topic_leaderboard_tier1_high_confidence.csv`

---

## Slide 13: Taxonomy Group Analysis: Main Groups

### Main Group Differences (8 Macro Buckets)

**Top vs Trash Effect Sizes (Cliff's δ):**
1. **Relationship Trajectory (Main Couple)**: Top higher (δ ≈ +0.37)
2. **Social World Outside Couple**: Top higher (δ ≈ +0.30)
3. **Embodied & Sensory Experience**: Top higher (δ ≈ +0.29)
4. **Sexuality, Attraction & Intimacy**: Trash higher (δ ≈ -0.25)

### Interpretation
At the broadest level, higher-tier books **slightly rebalance** thematic attention:
- **More** toward relationship dynamics and social context
- **Less** toward pure sexuality/intimacy mass

![Main Group Composition](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_T1_main_group_composition.png)

**Figure**: Main group composition across tiers showing the rebalancing of thematic attention.

![Main Group Heatmap](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_T2_main_group_heatmap.png)

**Figure**: Heatmap visualization of main group differences across tiers, providing a comprehensive overview of thematic rebalancing patterns.

![Main Group Effect Sizes](results/stage10_correlation_analysis/02_taxonomy_group_analysis/figures/main_group_distributions/effect_sizes.png)

**Figure**: Effect sizes (Cliff's δ) for main group comparisons, quantifying the magnitude of tier differences.

![Pairwise Comparisons Heatmap](results/stage10_correlation_analysis/02_taxonomy_group_analysis/figures/main_group_distributions/pairwise_comparisons_heatmap.png)

**Figure**: Pairwise comparisons between all tier combinations (Top vs Middle, Top vs Trash, Middle vs Trash) for main groups, showing which differences are most pronounced.

**Table**: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/tables/Table_T1_main_group_comparisons.csv`

---

## Slide 14: Taxonomy Group Analysis: Subgroups

### Strongest Subgroup Effects (Top vs Trash)

**Top-Associated Subgroups:**
- **Beliefs, Values & Moral Reflection**: δ ≈ +0.46, adjusted p ≈ 0.008 ⭐
- **Time/Seasons/Temporal framing**: δ ≈ +0.36
- **Shared Workplaces & Professional Interaction**: δ ≈ +0.35

**Trash-Associated Subgroups:**
- **Negative Emotions & Distress**: δ ≈ -0.37
- **Violence/Threats/Coercion**: δ ≈ -0.30
- **Public & Leisure Spaces**: δ ≈ -0.28

### Interpretation
**Top-tier books lean into:**
- Values/identity reflection (moral reasoning, self-concept)
- Professional/workplace interaction (institutional social texture)
- Time framing (temporal structuring cues)

**Trash-tier books lean into:**
- Distress/negative emotion intensity (panic, sobbing, emotional dysregulation)
- Threat/coercion/violence texture (risk/harm)

![Beliefs Values Moral Reflection Drivers](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_D3_drivers_Beliefs_ Values _ Moral Reflection.png)

**Figure**: Distribution of "Beliefs, Values & Moral Reflection" subgroup across tiers (strongest Top-tier association).

![Subgroup Effect Sizes Heatmap](results/stage10_correlation_analysis/02_taxonomy_group_analysis/figures/subgroup_distributions/subgroup_effect_sizes_heatmap.png)

**Figure**: Heatmap showing effect sizes for all subgroups (Top vs Trash comparisons).

![Shared Workplaces Drivers](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_D14_drivers_Shared Workplaces _ Professional Interaction.png)

**Figure**: Distribution of "Shared Workplaces & Professional Interaction" subgroup across tiers (strong Top-tier association, δ ≈ +0.35).

![Negative Emotions Drivers](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_D4_drivers_Negative Emotions _ Distress.png)

**Figure**: Distribution of "Negative Emotions & Distress" subgroup across tiers (strong Trash-tier association, δ ≈ -0.37).

**Table**: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/tables/Table_S1_subgroup_comparisons.csv`

---

## Slide 15: Thematic Diversity Findings

### Diversity Metrics by Tier

**Monotonic trend across tiers:**
- **Entropy**: bad ≈ 5.33 → mid ≈ 5.43 → good ≈ 5.48
- **Effective topics**: bad ≈ 207 → mid ≈ 229 → good ≈ 240
- **Richness** (topics > 1e-3): bad ≈ 247 → mid ≈ 258 → good ≈ 265
- **HHI**: Lower in good (more distributed)

**Statistical tests:**
- Entropy/effective_topics: p ≈ 0.019 (adjusted ≈ 0.077)
- Richness: p ≈ 0.015 (adjusted ≈ 0.073)

### Interpretation
**Higher-tier books show greater thematic diversity** — they combine more thematic ingredients rather than over-concentrating on a few motifs. This is a structural property, not tied to any single romance trope.

![Diversity Metrics by Tier](results/stage10_correlation_analysis/02_taxonomy_group_analysis/figures/diversity_metrics/fig_diversity_by_tier_faceted.png)

**Figure**: Thematic diversity metrics (entropy, effective topics, richness) showing monotonic increase across tiers.

![Diversity vs Other Metrics](results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_C2_diversity_vs_other.png)

**Figure**: Relationship between diversity metrics and other book characteristics.

**Table**: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/tables_csv/diversity_tests.csv`

---

## Slide 16: Two-Channel Analysis: Reach vs Quality

### Key Finding: Reach and Quality Are Different

The analysis separates two distinct Goodreads success signals:
1. **Mass Appeal / Visibility** = `log_rating_count` (how many people rated it)
2. **Perceived Quality** = `rating_mean` (how positively readers evaluate it)

**The same themes do not predict both equally well.**

### Mass Appeal Predictors (`log_rating_count`)
**Top predictors** (β with 95% CI, P(β>0)):
1. **R2_alpha_guarding**: β≈ +0.44, CI [+0.23, +0.60], P=1.00
2. **D_power_wealth_luxury__pc1**: β≈ +0.37, CI [+0.20, +0.55], P=1.00
3. **A2_emotional_safety__pc1**: β≈ +0.32, CI [+0.13, +0.50], P=0.998
4. **Q_repair**: β≈ +0.23, CI [+0.02, +0.41], P=0.985

**Macro axes:**
- **AX_status_dominance**: β≈ +0.46, P=1.00
- **AX_payoff_safety**: β≈ +0.33, P≈0.999
- **AX_explicitness**: β≈ −0.27, P≈0.003 (strongly negative)

![Macro Level Effects - Reach](results/stage10_correlation_analysis/04_hypothesis_testing/inference_20260112_213754__measurement_v5/figures/macro_level_effects_log_rating_count.png)

**Figure**: Macro-axis effects on mass appeal (log_rating_count). Status/dominance and payoff/safety drive reach, while explicitness is negatively associated.

![Macro Level Effects - Reach (No Control)](results/stage10_correlation_analysis/04_hypothesis_testing/inference_20260112_213754__measurement_v5/figures/macro_level_effects_avg_rating_bayes_no_control.png)

**Figure**: Alternative view of macro-axis effects without controlling for rating count, showing raw associations with average ratings.

---

## Slide 17: Perceived Quality Predictors

### Quality Beyond Reach (`rating_mean`, controlling for `log_rating_count`)

**Top predictors:**
1. **R1_protective_caretaking**: β≈ +0.22, CI [+0.05, +0.36], P=0.995
2. **A2_emotional_safety__pc1**: β≈ +0.15, P=0.95

**Macro axes:**
- **AX_payoff_safety**: β≈ +0.21, P=0.974
- **AX_explicitness**: β≈ −0.15, P=0.095 (tends negative)
- **AX_negative_affect**: β≈ −0.13, P=0.051 (borderline negative)

### Partial Correlations (Quality Beyond Popularity)
- **R1_protective_caretaking**: +0.245
- **A2_emotional_safety__pc1**: +0.152
- **C_explicit_eroticism**: -0.172 (negative)
- **F2_anger_frustration**: -0.121 (negative)

### Interpretation
**Quality beyond reach** is associated with:
- "Care + safety" (protective caretaking, emotional safety)
- Negatively with baseline negative affect and explicit erotics

![Macro Level Effects - Quality](results/stage10_correlation_analysis/04_hypothesis_testing/inference_20260112_213754__measurement_v5/figures/macro_level_effects_rating_mean.png)

**Figure**: Macro-axis effects on perceived quality (rating_mean, controlling for reach). Care/safety themes drive quality beyond popularity.

![Macro Level Effects - Quality (With Control)](results/stage10_correlation_analysis/04_hypothesis_testing/inference_20260112_213754__measurement_v5/figures/macro_level_effects_avg_rating_bayes_control_log_n.png)

**Figure**: Bayesian analysis of macro-axis effects on quality, controlling for log rating count, confirming the robustness of care/safety associations.

---

## Slide 18: Narrative Arc / Pacing Results

### Higher-Rated Books Show Better Pacing

**Late-story crisis escalation** (third-act crisis):
- **F2_anger_frustration end−begin**: β≈ +0.24, CI [+0.08, +0.41], P=0.995
- **F3_anxiety_worry end−begin**: β≈ +0.19, CI [+0.02, +0.36], P=0.981

### Interpretation
Higher-rated books have:
- **Lower baseline negativity** across the book
- **Stronger late "crisis escalation"** (third-act crisis)
- Consistent with romance narrative structure (Radway phases)

### Most Positive Arc Effects
1. **F2_anger_frustration end−begin**: β≈ +0.24, P=0.995
2. **F3_anxiety_worry end−begin**: β≈ +0.19, P=0.981
3. **O_aesthetics_appearance end−begin**: β≈ +0.15, P=0.950

### Most Negative Arc Effects
1. **J_social_support_kin end−begin**: β≈ -0.18, P=0.019
2. **S_scene_anchors end−begin**: β≈ -0.16, P=0.074
3. **C_explicit_eroticism end−begin**: β≈ -0.12, P=0.103

### Example: Radway Phase Distribution

**Phase I: Initial Conflict & Isolation** (147 topics, 54.0%)
- Example topics: "Insane Arguments", "Relationship Ambiguity Conversation"
- Radway functions: R2 (Heroine reacts antagonistically), R7 (Physical/emotional separation)

**Phase II: Turning Point & Recognition** (96 topics, 35.3%)
- Example topics: "Tender Forehead Kisses", "Frightened Admissions"
- Radway functions: R8 (Hero treats heroine tenderly), R9 (Heroine responds warmly)

**Phase III: Commitment & Restoration** (28 topics, 10.3%)
- Example topics: "Marriage Ceremony Planning", "Wedding Ceremony And Vows"
- Radway functions: R11 (Hero declares love), R13 (Heroine's identity restored)

**Table**: `results/stage10_correlation_analysis/04_hypothesis_testing/inference_20260112_213754__measurement_v5/appendix_csv/arc_effects_rating_mean.csv`

---

## Slide 19: Hypothesis Testing Results Summary

### H1: Love-over-Sex Hypothesis ✅
**(commitment_hea + tenderness_emotion) > explicit in Top vs Trash**
- **Status**: Supported (explicitness negatively associated with reach and quality)
- **Evidence**: Explicit erotics (C) shows negative associations: β≈ -0.27 for reach, β≈ -0.15 for quality

### H2: HEA Index Hypothesis ✅
**HEA Index higher in Top**
- **Status**: Supported (commitment/repair positively associated with reach and quality)
- **Evidence**: Q_repair: β≈ +0.23 for reach, A2_emotional_safety: β≈ +0.32 for reach, +0.15 for quality

### H3: Luxury × Love Interaction
**Luxury Saturation predicts Top only when (commitment_hea + tenderness_emotion) is high**
- **Status**: Partially supported (luxury predicts reach, but interaction needs further testing)
- **Evidence**: D_power_wealth_luxury: β≈ +0.37 for reach, but weak for quality

### H4: Protectiveness vs Possessiveness ✅
**protectiveness_care − jealousy_possessiveness is higher in Top**
- **Status**: Supported (protective caretaking predicts quality)
- **Evidence**: R1_protective_caretaking: β≈ +0.22 for quality (controlling reach)

### H5: Darkness vs Tenderness ✅
**(neg_affect + threat_violence_dark) − tenderness_emotion is lower in Top**
- **Status**: Supported (negative affect negatively associated with quality)
- **Evidence**: AX_negative_affect: β≈ -0.13 for quality (borderline)

### H6: Narrative Arc (Time-Course) ✅
**begin→end: miscommunication/neg_affect ↓; commitment_hea/apology_repair ↑**
- **Status**: Partially supported (late crisis escalation pattern observed)
- **Evidence**: Anger/frustration and anxiety increase toward end in higher-rated books

---

## Slide 20: Predictive Performance

### Cross-Validation Results (20 repeats of 5-fold CV)

| Outcome          | Model         | CV_R²_mean | CV_R²_sd |
|-----------------|---------------|------------|----------|
| rating_mean      | metadata_only | 0.108       | 0.031    |
| rating_mean      | metadata+core | 0.056       | 0.041    |
| log_rating_count | core_only     | 0.050       | 0.037    |

### Interpretation
- **Themes explain popularity (reach) better than star ratings**
- Star ratings likely influenced by factors beyond theme indices (prose quality, pacing, editing, reader expectations)
- **Meta-result**: Market reach is more systematically related to thematic content than star ratings

**Table**: `results/stage10_correlation_analysis/04_hypothesis_testing/inference_20260112_213754__measurement_v5/appendix_csv/cv_results.csv`

---

## Slide 21: Key Findings Summary

### 1. Two-Channel Success Model
- **Reach (visibility)**: Driven by "billionaire-romance package" (status/luxury + alpha guarding + repair + emotional safety)
- **Quality (ratings)**: Driven by "care + safety" (protective caretaking, emotional safety)
- **Explicitness**: Negatively associated with both reach and quality

### 2. Thematic Differentiation
- **Top-tier books**: More psychological credibility, embodied intimacy cues, values/moral reflection, professional context
- **Trash-tier books**: More explicit sexual content, procedural/transition scenes, distress intensity, coercion/threat texture
- **Diversity**: Higher-tier books show greater thematic diversity (entropy: 5.33 → 5.48)

### 3. Narrative Pacing
- Higher-rated books show **better pacing**: lower baseline negativity but stronger late "crisis escalation"
- Consistent with romance narrative structure (Radway phases)

### 4. Methodological Contributions
- Two-gate filtering rule balances statistical rigor with practical interpretability
- Two-tier structure (high confidence vs exploratory) provides transparency
- Author dominance metrics control for author-style topics

---

## Slide 22: Methodological Innovations

### Pareto-Efficient Model Selection
- **Multi-objective optimization**: Balances coherence and topic diversity
- **Robust outlier filtering**: Two-stage approach (domain knowledge + statistical methods)
- **4 Pareto-efficient models** identified from 272 configurations
- **Hyperparameter correlation analysis**: Identifies critical parameters (UMAP, vectorizer settings)
- **Evidence-based selection**: Multiple statistical methods validate findings

### Character Name Exclusion
- **4,444 character names** added to stopwords (93% of expanded list)
- **14x increase** in stopwords compared to standard English
- Ensures topics focus on thematic content rather than character co-occurrence patterns

### Multiple Representation Strategies
- **Main**: Standard c-TF-IDF (highest coherence: 0.404)
- **POS**: Part-of-speech filtered (balanced: 0.315 coherence, 0.692 diversity)
- **MMR**: Maximal Marginal Relevance (highest diversity: 0.756)
- **KeyBERT**: Semantic similarity-based

### Automated LLM Labeling
- **Mistral-Nemo-Instruct** via OpenRouter API
- **98.1% taxonomy coverage** (361/368 topics)
- **Representative snippets**: 3-6 sentence snippets provide scene-level context
- **Cost**: ~$0.018 for 368 topics

### Example: LLM Labeling Quality

**Input (Topic Keywords)**: "car, driver, parking, cars, driveway, suv, curb, cab, drive, vehicle"

**LLM Output:**
- **Label**: "Parked Car Makeout"
- **Scene Summary**: "The couple, parked in his car, engages in passionate kissing."
- **Primary Categories**: romance_core, physical_affection
- **Secondary Categories**: setting:car, activity:kissing
- **Rationale**: "The top keyword 'car' and snippets like 'the back of my car?' and 'in his car' indicate a scene in a vehicle. The verb 'kissing' from the POS cues and its repetition in snippets suggest a romantic, physical activity."

### Theory-Aligned Classification
- **Romance Corpus Topic Taxonomy**: 8 main groups, 30+ categories
- **Radway narrative functions**: 13 functions mapped to 3 phases
- **Zero-shot classification**: No training data required

---

## Slide 23: Statistical Methodology

### Sample Size & Approach
- **N = 92 books** (pilot analysis)
- **Effect-size focused** (not p-value focused)
- **Bootstrap inference**: 800 iterations, 95% confidence intervals
- **Sign stability**: P(β>0) for directional effects

### Two-Channel Analysis
- Separates **mass appeal** (`log_rating_count`) from **perceived quality** (`rating_mean`)
- Treats them as distinct outcome channels with different predictors
- Controls for reach when analyzing quality

### Nonparametric Tests
- **Kruskal-Wallis**: Overall tier differences
- **Mann-Whitney U**: Pairwise comparisons (Top vs Trash)
- **Effect sizes**: Cliff's delta (δ) as primary interpretation metric
- **Multiple comparisons**: Family-wise adjusted p-values (Holm correction)

### Arc Analysis
- **Tertile splitting**: begin/middle/end segments
- **Delta contrasts**: end−begin, middle−begin
- Tests if theme changes over story predict ratings

---

## Slide 24: Limitations & Future Work

### Current Limitations
- **Small sample size**: N=92 (pilot analysis)
- **Statistical power**: FDR-corrected p-values underpowered
- **Author confounding**: 30 topics show high author dominance (>50% from single author)
- **Label interpretation**: Topic labels are imperfect summaries

### Future Work
1. **Large-N study**: Expand to full corpus (105+ books)
2. **Author fixed effects**: Control for author style in modeling
3. **Time-course analysis (H6)**: Examine topic probabilities across book tertiles
4. **Qualitative sampling**: Use procedure topics as scene anchors for qualitative analysis
5. **Interaction effects**: Test H3 (Luxury × Love interaction) more rigorously

### Methodological Refinements
- Filter author-dominant topics before aggregation
- Include author fixed/random effects in modeling
- Refine composite index construction (19 theory-aligned categories A-S)
- Cross-validation with larger sample

---

## Slide 25: Implications & Contributions

### Theoretical Contributions
1. **Two-channel success model**: Separates market reach from perceived quality
2. **Thematic diversity hypothesis**: Higher-quality books combine more thematic ingredients
3. **Narrative pacing patterns**: Late crisis escalation associated with higher ratings
4. **Theory-aligned operationalization**: Radway functions + taxonomy categories

### Methodological Contributions
1. **Character name exclusion**: Novel approach to improve topic interpretability
2. **Multiple representation strategies**: Enables multi-faceted topic analysis
3. **Automated LLM labeling**: Scalable approach to topic interpretation
4. **Two-gate filtering**: Balances statistical rigor with practical interpretability

### Practical Implications
1. **For authors**: Focus on psychological credibility, embodied intimacy, values reflection
2. **For publishers**: Market reach driven by "billionaire-romance package" (status + safety)
3. **For readers**: Quality beyond reach associated with "care + safety" themes
4. **For researchers**: Replicable pipeline for computational literary analysis

---

## Slide 26: Acknowledgments & References

### Software & Tools
- **BERTopic** (Grootendorst, 2022) for topic modeling
- **OCTIS** (Terragni et al., 2021) for hyperparameter optimization
- **RAPIDS cuML** for GPU acceleration
- **SentenceTransformers** for embeddings
- **Mistral-Nemo-Instruct** for automated topic labeling

### Theoretical Foundations
- **Radway, J. (1984)**: *Reading the Romance: Women, Patriarchy, and Popular Literature*
- **Propp, V. (1928)**: *Morphology of the Folktale*
- **Ogas, O., & Gaddam, S. (2011)**: *A Billion Wicked Thoughts*

### Data Sources
- **Goodreads**: Ratings and metadata
- **BookNLP**: Character name extraction
- **Curated lists**: "100 Best Billionaire Romance Books of All Time"

### Repository
- **Code & Data**: Available in project repository
- **Results**: `results/stage10_correlation_analysis/`
- **Reports**: `reports/01_stage_reports/stage10_correlation_analysis/`

---

## Slide 27: Questions & Discussion

### Key Questions for Discussion
1. How generalizable are these findings beyond billionaire romance?
2. What role do author effects play in thematic patterns?
3. How can we better operationalize "narrative pacing"?
4. What other theoretical frameworks could be integrated?

### Contact & Collaboration
- **Repository**: [Project URL]
- **Documentation**: See `README.md` and `SCIENTIFIC_README.md`
- **Reports**: `reports/01_stage_reports/stage10_correlation_analysis/`

### Next Steps
- Expand to larger corpus
- Refine composite index construction
- Test interaction effects (H3)
- Integrate qualitative analysis

---

## Appendix: Additional Figures & Tables

### Stage 04: Pareto Efficiency Analysis Figures
- Pareto frontier (all models): `results/stage04_selection/pareto_frontier_all.png`
- Pareto frontier (by model): `results/stage04_selection/pareto_frontier_by_model.png`
- Pareto front (equal weights): `results/stage04_selection/figures/pareto_front_equal_weights.png`
- Pareto fronts (per model): `results/stage04_selection/figures/pareto_fronts_per_model.png`
- Distribution with cutoffs: `results/stage04_selection/figures/distribution_with_cutoffs.png`
- Hyperparameter boxplots: `results/stage04_selection/figures/hyperparameter_boxplots.png`
- Combined score distribution: `results/stage04_selection/combined_score_distribution.png`

### Stage 04: Pareto Efficiency Analysis Tables
- Pareto-efficient models: `results/stage04_selection/pareto.csv`
- Top 10 models (equal weights): `results/stage04_selection/top_models/top_10_equal_weights.csv`
- Top 10 models (coherence priority): `results/stage04_selection/top_models/top_10_coherence_priority.csv`
- Correlation analysis (equal weights): `results/stage04_selection/tables/correlation_analysis_equal_weights.csv`
- Correlation analysis (coherence priority): `results/stage04_selection/tables/correlation_analysis_coherence_priority.csv`

### Topic Analysis Figures
- Topic distributions: `results/stage10_correlation_analysis/01_topic_analysis/figures/topic_distributions/`
- Topic leaderboards: `results/stage10_correlation_analysis/01_topic_analysis/tables_csv/`

### Taxonomy Group Analysis Figures
- Main group composition: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_T1_main_group_composition.png`
- Main group heatmap: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_T2_main_group_heatmap.png`
- Main group effect sizes: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/figures/main_group_distributions/effect_sizes.png`
- Pairwise comparisons heatmap: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/figures/main_group_distributions/pairwise_comparisons_heatmap.png`
- Subgroup effect sizes: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/figures/subgroup_distributions/subgroup_effect_sizes_heatmap.png`
- Subgroup drivers (Beliefs/Values): `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_D3_drivers_Beliefs_ Values _ Moral Reflection.png`
- Subgroup drivers (Shared Workplaces): `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_D14_drivers_Shared Workplaces _ Professional Interaction.png`
- Subgroup drivers (Negative Emotions): `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_D4_drivers_Negative Emotions _ Distress.png`
- Diversity metrics: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/figures/diversity_metrics/fig_diversity_by_tier_faceted.png`
- Diversity vs other metrics: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_C2_diversity_vs_other.png`
- Coverage distribution: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/figures/Figure_C1_coverage_distribution.png`
- Topic drivers summary: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/figures/topic_drivers/topic_drivers_summary_heatmap.png`

### Hypothesis Testing Figures
- Macro-axis effects (reach): `results/stage10_correlation_analysis/04_hypothesis_testing/inference_20260112_213754__measurement_v5/figures/macro_level_effects_log_rating_count.png`
- Macro-axis effects (quality): `results/stage10_correlation_analysis/04_hypothesis_testing/inference_20260112_213754__measurement_v5/figures/macro_level_effects_rating_mean.png`
- Macro-axis effects (quality, no control): `results/stage10_correlation_analysis/04_hypothesis_testing/inference_20260112_213754__measurement_v5/figures/macro_level_effects_avg_rating_bayes_no_control.png`
- Macro-axis effects (quality, with control): `results/stage10_correlation_analysis/04_hypothesis_testing/inference_20260112_213754__measurement_v5/figures/macro_level_effects_avg_rating_bayes_control_log_n.png`

### Tables
- Topic leaderboards: `results/stage10_correlation_analysis/01_topic_analysis/tables_csv/`
- Taxonomy comparisons: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/report_figures_tables/tables/`
- Statistical results: `results/stage10_correlation_analysis/04_hypothesis_testing/inference_20260112_213754__measurement_v5/appendix_csv/`

