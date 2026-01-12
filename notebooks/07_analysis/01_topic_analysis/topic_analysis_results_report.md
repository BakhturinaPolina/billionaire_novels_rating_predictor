# Statistical Analysis and Results: Topic-Level Landscape

**Analysis Scope:** Topic-level analysis of 368 BERTopic topics across popularity tiers (Top, Middle, Trash) using book-level topic probability mixtures.

**Output Location:** `results/correlation_analysis/01_topic_analysis/`

---

## Statistical Analysis (Topic-Level Landscape)

### Overview and Unit of Analysis

We quantify thematic differences across popularity tiers (Top, Middle, Trash) at the finest granularity available: individual BERTopic topics (≈300–370 topics, depending on downstream filtering). The primary analytic unit is the **book-level topic mixture** (one probability distribution per book), enabling comparisons of topic prevalence and topic mass across tiers. For topic-level inferential checks, we focus on **Top vs Trash** as the cleanest contrast; three-tier comparisons (Top/Middle/Trash) are used as a descriptive robustness check.

A technical constraint of the exported topic-probability tables is that raw per-book topic probability sums can be < 1.0 when only a subset of topics (or truncated probability vectors) is stored. To preserve comparability, the pipeline introduces an **OTHER** bucket that captures missing probability mass at the book level, restoring book-level mixture sums to 1.0. This OTHER bucket is treated as a *coverage diagnostic* (how much of the book's distribution is outside the stored topic rows), not a substantive theme.

### Topic Health Metrics: Prevalence, Mass, and Concentration

Before interpreting tier differences, each topic is characterized by three "health" measures:

1. **Prevalence**: share of books where the topic has non-zero mass (prob > 0.001 threshold).
2. **Mass**: average probability mass of the topic across books.
3. **Concentration ratio**: how "spiky" a topic is (max share / mean share per topic; high values indicate that a small number of books contribute disproportionate mass).

These measures are used to distinguish stable, corpus-wide themes from rare or idiosyncratic topics that may produce fragile tier differences.

**Table 1. Topic health summary (N = 343 topics)**

|       | prevalence |       mass | concentration_ratio |
| ----: | ---------: | ---------: | ------------------: |
| count | 343.000000 | 343.000000 |          343.000000 |
|  mean |   0.575538 |   0.002734 |            8.077546 |
|   std |   0.401927 |   0.017250 |           11.432865 |
|   min |   0.000000 |   0.000003 |            1.187682 |
|   25% |   0.097826 |   0.000829 |            2.068178 |
|   50% |   0.728261 |   0.001368 |            2.663997 |
|   75% |   0.967391 |   0.002315 |            7.636642 |
|   max |   1.000000 |   0.319823 |           63.008440 |

*Source: `results/correlation_analysis/01_topic_analysis/tables/topic_health_table.parquet`*

Two features of this landscape are important for statistical strategy. First, prevalence is highly heterogeneous, ranging from topics that appear in almost all books to topics present in only a handful. Second, concentration ratios can be extremely large, indicating topics that are effectively "owned" by a small number of books (or authors). Figure 1 (prevalence histogram) and Figure 2 (concentration distribution) visualize these regimes.

**Figure 1: Prevalence Distribution**
*Location: `results/correlation_analysis/01_topic_analysis/figures/topic_distributions/`*

**Figure 2: Concentration Ratio Distribution (Log Scale)**
*Location: `results/correlation_analysis/01_topic_analysis/figures/topic_distributions/`*

For interpretability, Table 2 bins topic prevalence into broad ranges:

**Table 2. Prevalence bins (share of books)**

| Prevalence bin (share of books) | Number of topics |
| :------------------------------ | ---------------: |
| 0–5%                            |               65 |
| 5–10%                           |               22 |
| 10–20%                          |               18 |
| 20–50%                          |               40 |
| 50–80%                          |               37 |
| 80–95%                          |               50 |
| 95–100%                         |              111 |

This distribution implies that a large portion of topics are either (i) nearly ubiquitous "genre backbone" elements, or (ii) rare, potentially unstable topics. This is exactly the scenario where naïve p-value ranking becomes unreliable.

### Why Effect Sizes Are Prioritized at Topic Level

Topic-level inference involves hundreds of correlated tests (one per topic) with only ~30 books per tier (Top vs Trash). Under these conditions, multiple-testing correction is extremely conservative, and even meaningful effects often fail to survive FDR adjustment. Consequently, topic-level interpretation emphasizes **effect sizes** and **impact**, rather than adjusted p-values.

We quantify Top vs Trash differences using:

* **Cliff's delta (δ)**: a nonparametric effect size indicating how often Top values exceed Trash values (δ > 0 implies Top-associated; δ < 0 implies Trash-associated).
* **Impact metrics**: mass and mean difference (TopMean − TrashMean), ensuring we do not overinterpret effects driven by vanishingly small probability mass.
* **Nonparametric tests** (as diagnostics): Mann–Whitney U for Top vs Trash; Kruskal–Wallis for three-tier comparisons. These tests are reported, but not used as the primary ranking criterion once FDR correction is applied.

### "Two-Gate" Topic Filtering (Plus a Stability Gate for Interpretation)

To identify interpretable topic-level differences, we use a two-gate rule (effect + impact), then apply a stability constraint for narrative claims:

* **Gate 1 (Effect):** |δ| ≥ 0.20
* **Gate 2 (Impact):** mass ≥ 0.002 OR |TopMean − TrashMean| ≥ 0.001
* **Stability (interpretation-only):** prevalence ≥ 0.10

Table 3 shows how these filters reduce hundreds of topics to a manageable signal set.

**Table 3. Topic gating counts**

| Criterion                                                 | Count |
| :-------------------------------------------------------- | ----: |
| Total topics analyzed                                     |   343 |
| Effect gate \|δ\| ≥ 0.20                                    |   136 |
| Impact gate (mass ≥ 0.002 OR \|Δmean\| ≥ 0.001)             |   114 |
| Both gates                                                |    45 |
| Both gates + prevalence ≥ 0.10                            |    41 |
| Robust set (both + prevalence, excluding author-dominant) |    41 |

*Source: Computed from `topic_leaderboard_all.parquet` and `topic_author_dominance.parquet`*

The final line anticipates a key robustness concern: author style can dominate topic distributions. We therefore add an explicit author-dominance filter before interpreting tier differences.

### Author Dominance and Robustness Filtering

To reduce the risk that "tier differences" reflect author idiosyncrasies, we compute author dominance for each topic: the share of a topic's total mass accounted for by the most dominant author, plus categorical flags ("low/medium/high dominance"). Table 4 summarizes these flags:

**Table 4. Author dominance summary**

| Author dominance flag | Number of topics |
| :-------------------- | ---------------: |
| low                   |              266 |
| high                  |               41 |
| medium                |               17 |

*Source: `results/correlation_analysis/01_topic_analysis/tables/topic_author_dominance.parquet`*

A non-trivial subset of topics are author-driven; these are either excluded from interpretation or reported as sensitivity checks. In subsequent modeling stages (tier prediction, rating regression), author controls (fixed effects or random intercepts) are justified directly by this dominance pattern.

---

## Results (Topic-Level Tier Differences)

### Tier-Associated Topic Leaderboards: "High Confidence" vs "Exploratory"

Given limited statistical power under FDR correction, we report two complementary tier-difference views:

1. **Tier 1 (high confidence)**: very large effect sizes, non-adjusted significance as a diagnostic, and strong impact.
2. **Tier 2 / robust set**: topics passing effect+impact gates (and stability/author filters) that serve as the interpretable micro-scene signal bank to be aggregated into subgroups and composites.

#### Tier 1: High Confidence Topics

The strictest filtering yields a small set of "high confidence" topics. Table 5 reproduces these as the most defensible topic-level differences.

**Table 5. Tier 1 topics (Top vs Trash)**

| topic_id | label                                 | direction | cliffs_top_trash | mw_top_trash_p |     mass | prevalence |
| -------: | :------------------------------------ | :-------: | ---------------: | -------------: | -------: | ---------: |
|       -1 | OTHER (Missing/Noise Mass)            |   Top ↑   |            0.422 |       0.005084 | 0.319823 |      1.000 |
|      250 | Protective Conversations By Firelight |  Trash ↑  |           -0.362 |       0.016285 | 0.003958 |      0.978 |
|       55 | Dominatrix Session                    |  Trash ↑  |           -0.356 |       0.018368 | 0.006298 |      0.989 |

*Source: `results/correlation_analysis/01_topic_analysis/tables/topic_leaderboard_tier1_high_confidence.parquet`*

**Interpretation of Tier 1 signals:**

* **Dominatrix Session (Trash ↑)** reflects a specific explicit/BDSM-coded micro-scene signature that is more prominent in lower-rated books within this corpus. This is consistent with a *directional* reading of the Love-over-Sex framing (H1), with the important caveat that the signal pertains to a particular explicitness profile, not "sexual content" as a whole.

* **Protective Conversations by Firelight (Trash ↑)** underscores a central modeling limitation: topic labels can conflate tenderness with dominance-coded "protector" language. This topic motivates the later composite split that separates caring protectiveness from jealousy/control dynamics (H4 operationalization).

* **OTHER (Top ↑)** is a diagnostic: Top books contain more probability mass outside the stored topic rows. This may indicate greater thematic diversity (more long-tail topics), lower model certainty for less formulaic prose, or differences in how probability vectors were captured. OTHER is retained for normalization but excluded from thematic interpretation and indices.

### Robust Tier 2: Interpretable Micro-Scene Differences (After Gates + Stability + Author Filtering)

To characterize "what differs across tiers" in a way that is stable enough to aggregate, we interpret the **robust gated set** (both gates + prevalence ≥ 0.10 + excluding author-dominant topics). These topics represent the best candidates for subgroup-level drivers.

**Figure 3: Effect Size vs Mass (Robust Topics Highlighted)**
*Location: `results/correlation_analysis/01_topic_analysis/figures/topic_leaderboards/`*

**Figure 4: Top Robust Effects by Topic ID**
*Location: `results/correlation_analysis/01_topic_analysis/figures/topic_leaderboards/`*

Figure 3 plots topic **effect size** against **mass**, with annotated robust topics to highlight high-impact differences. Figure 4 visualizes the largest robust effects (topic IDs) to show that tier differentiation is not dominated by a single outlier.

Tables 6a and 6b list the most Top-associated and most Trash-associated topics within the robust set.

**Table 6a. Robust Top-associated topics (Top > Trash)**
(δ > 0; sorted by δ then mass)

| topic_id | label                                | cliffs_top_trash | mean_diff_top_trash |     mass | prevalence |
| -------: | :----------------------------------- | ---------------: | ------------------: | -------: | ---------: |
|       36 | Kneeling For Intimacy                |            0.331 |            0.000598 | 0.002175 |      0.815 |
|      226 | Disco Night Adventure                |            0.320 |            0.001453 | 0.001809 |      0.859 |
|       75 | Bedroom Encounter                    |            0.291 |            0.000423 | 0.002318 |      0.989 |
|        9 | Marriage Ceremony Planning           |            0.287 |            0.000798 | 0.004331 |      0.989 |
|       44 | Mother-player Relationship Struggles |            0.278 |            0.000479 | 0.002004 |      0.946 |
|       37 | Insane Arguments                     |            0.276 |            0.000396 | 0.002551 |      0.989 |
|       19 | Adrenaline-fueled game               |            0.276 |            0.000713 | 0.002893 |      0.815 |
|      264 | Schoolyard Argument                  |            0.271 |            0.000436 | 0.004377 |      1.000 |
|      159 | Intergenerational Family Bonding     |            0.240 |            0.000399 | 0.002001 |      0.826 |
|       84 | Trust Assurance Conversation         |            0.258 |            0.000302 | 0.002391 |      0.978 |
|       10 | Knocking On Door                     |            0.244 |            0.000606 | 0.005003 |      1.000 |
|       28 | Growling During Intimacy             |            0.242 |            0.000287 | 0.002710 |      0.978 |

**Table 6b. Robust Trash-associated topics (Trash > Top)**
(δ < 0; sorted by δ then mass)

| topic_id | label                                 | cliffs_top_trash | mean_diff_top_trash |     mass | prevalence |
| -------: | :------------------------------------ | ---------------: | ------------------: | -------: | ---------: |
|      250 | Protective Conversations By Firelight |           -0.362 |           -0.001922 | 0.003958 |      0.978 |
|       55 | Dominatrix Session                    |           -0.356 |           -0.002595 | 0.006298 |      0.989 |
|      139 | Awkward Questions At Door             |           -0.324 |           -0.001013 | 0.003079 |      0.957 |
|       27 | Exiting Through Doorways              |           -0.320 |           -0.000879 | 0.002992 |      0.989 |
|       54 | Phone Ringing And Answering           |           -0.307 |           -0.001348 | 0.004372 |      0.978 |
|        1 | Intimate Breast And Nipple Kissing    |           -0.298 |           -0.012035 | 0.013795 |      0.989 |
|      202 | Hatred-fueled Argument                |           -0.291 |           -0.000313 | 0.004430 |      1.000 |
|       33 | Business Discussion                   |           -0.284 |           -0.000487 | 0.002513 |      0.989 |
|       98 | Pride In Accomplishments              |           -0.278 |           -0.000561 | 0.006179 |      1.000 |
|      125 | Emotional Panic Attack                |           -0.278 |           -0.001031 | 0.002961 |      0.935 |
|      121 | Admiring Intelligence                 |           -0.278 |           -0.000648 | 0.002519 |      0.946 |
|       73 | Fascinated Gaze                       |           -0.273 |           -0.000992 | 0.004162 |      0.957 |

*Source: `results/correlation_analysis/01_topic_analysis/tables/topic_leaderboard_tier2_exploratory.parquet` (filtered for robust set)*

### Qualitative Interpretation: What Micro-Scenes Differ Across Tiers?

The robust topic set reveals three interpretable patterns that can be carried forward to subgroup-level analysis and composite indices.

**1) Explicitness markers skew Trash, especially high-impact explicit micro-scenes.**

The strongest Trash-leaning signal with substantial mass is **Intimate Breast and Nipple Kissing** (high mass and large mean difference), alongside **Dominatrix Session**. Importantly, these are not generic "sex topics"; they are specific, legible micro-scene scripts that likely sit within your Sexuality/Explicit Acts subgroup. This provides bottom-layer motivation for operationalizing H1 using an explicitness ratio that contrasts explicit erotics (2.3) against commitment/tenderness clusters.

**2) "Procedural scaffolding" micro-scenes skew Trash (doors, phones, transitions).**

Trash-associated topics include **doorway exits**, **awkward door questions**, and **phone ringing/answering**. These topics are thematically "thin" but narratively functional: they anchor scene transitions, blocking, and logistical movement. A conservative interpretation is that lower-rated books contain more high-frequency scaffolding units, which may reflect more formulaic pacing or a greater reliance on transitional action. Methodologically, these topics are valuable as (i) candidates for your "Scene Anchors" sampling set, and (ii) potential covariates when modeling rating outcomes to avoid conflating pacing with thematic content.

**3) Commitment infrastructure and reassurance cues skew Top.**

Top-associated topics include **Marriage Ceremony Planning** and **Trust Assurance Conversation**, as well as family-bonding micro-scenes (intergenerational bonding; parent/child contexts). This cluster aligns naturally with your commitment/HEA and social support hypotheses (H2, and parts of H6 once segmented time-course is introduced). At topic level, these signals do not prove an arc; they indicate that the building blocks of commitment and reassurance are measurably more prominent in Top-tier books.

### Robustness: Author Dominance and Rarity Controls

Author dominance analysis shows that many topics are partially or fully driven by a single author; without filtering, these can masquerade as tier differences. Applying dominance filtering (excluding medium/high dominance topics) and a prevalence threshold (≥ 0.10) yields a robust interpretive set of 41 topics. This approach supports a transparent claim: the reported tier differences are not dominated by one author's stylistic signature or by topics present in only one or two books.

**Table 7. Topics that are both significant AND author-driven (excluded from robust set)**

| label                                 | top_author  | top_author_share | cliffs_top_trash |
| :------------------------------------ | :---------- | ---------------: | ---------------: |
| Confused About Relationship Feelings  | stella rhys |               1.0 |          0.235556 |
| Serious Conversation About Relationship | sam crescent |               1.0 |          0.217778 |
| Business Meeting With Client          | leslie north |               1.0 |         -0.200000 |

*Source: `results/correlation_analysis/01_topic_analysis/tables/topic_author_dominance.parquet` merged with `topic_leaderboard_tier2_exploratory.parquet`*

These topics may reflect author style rather than tier preferences and are therefore excluded from the robust interpretive set.

---

## Practical Takeaway for the Next Stage (Subgroups and Within-Subgroup Drivers)

The topic-level results provide a defensible "signal bank" for the targeted drill-down you want next:

* **A high-impact explicitness subset** (explicit acts + BDSM-coded scenes) → maps to Sexuality 2.3 and to H1/H5 composites.
* **A scaffolding subset** (doors/phones/transitions) → maps to Spaces/Objects 7.x and Scene Anchors (S); treat as style/pacing controls or sampling scaffolds.
* **A commitment/reassurance subset** (wedding planning, trust assurance) → maps to Relationship Stage/Commitment 4.2 and to HEA/Commitment composites (A/G/Q).

These themes are now ready to be aggregated into your 28 subgroups to (i) increase power, (ii) reduce multiple testing, and (iii) identify within-subgroup driver topics that actually shift group-level proportions across tiers.

---

## Data Files Referenced

All output tables and figures are located in:
`results/correlation_analysis/01_topic_analysis/`

### Key Tables:
- `tables/topic_health_table.parquet` - Topic health metrics (prevalence, mass, concentration)
- `tables/topic_leaderboard_all.parquet` - Full topic statistics across tiers
- `tables/topic_leaderboard_tier1_high_confidence.parquet` - Tier 1 high-confidence topics
- `tables/topic_leaderboard_tier2_exploratory.parquet` - Tier 2 exploratory topics (robust set)
- `tables/topic_leaderboard_filtered.parquet` - Filtered topics (Tier 2 set)
- `tables/topic_author_dominance.parquet` - Author dominance analysis per topic
- `tables/topic_hypothesis_mapping.parquet` - Topic-to-hypothesis mapping (if generated)

### Key Figures:
- `figures/topic_distributions/` - Individual topic distribution plots (violin, box plots)
- `figures/topic_distributions/all_topics_violin_grid.html` - Grid visualization of topic distributions
- `figures/topic_leaderboards/` - Effect size visualizations (if generated)

---

*Report generated from: `notebooks/07_analysis/topic_analysis/topic_analysis_topics.ipynb`*

