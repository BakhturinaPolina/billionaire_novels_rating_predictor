# Scientific Methodology: Modern Romantic Novels — Themes × Popularity

**A Mixed-Methods Computational Analysis**

## Research Objectives

1. **Map topic-model outputs** from modern romance novels to theory-driven themes and test which themes differentiate Top / Medium / Trash popularity tiers.

2. **Build explainable indices** (e.g., Love-over-Sex, HEA Index) to quantify narrative qualities readers value.

3. **Validate corpus findings** against Goodreads metadata (rating and number of voters).

## Research Questions

1. Which theme categories are most prevalent in Top vs Trash novels?

2. Does love/commitment/tenderness outweigh explicit sexual content in higher-rated books?

3. Is luxury appealing only when paired with commitment/tenderness?

4. Do protectiveness/care signals predict appreciation better than jealous/possessive affect?

5. Do miscommunication/negative affect diminish across the book while HEA/repair rises (time-course)?

## Research Hypotheses

### H1: Love-over-Sex Hypothesis
**(commitment_hea + tenderness_emotion) > explicit in Top vs Trash**

Higher-rated novels emphasize emotional connection and commitment over explicit sexual content.

### H2: HEA Index Hypothesis
**HEA Index higher in Top**

Novels with higher Happily Ever After (HEA) indicators (commitment, symbolic gifts, festive rituals) are more appreciated.

### H3: Luxury × Love Interaction
**Luxury Saturation predicts Top only when (commitment_hea + tenderness_emotion) is high (positive interaction)**

Luxury settings and wealth are appealing only when combined with emotional depth and commitment.

### H4: Protectiveness vs Possessiveness
**protectiveness_care − jealousy_possessiveness is higher in Top**

Caring protectiveness is valued more than jealous possessiveness in highly-rated novels.

### H5: Darkness vs Tenderness
**(neg_affect + threat_violence_dark) − tenderness_emotion is lower in Top**

Top-rated novels favor tenderness over dark themes and negative affect.

### H6: Narrative Arc (Time-Course)
**begin→end: miscommunication/neg_affect ↓; commitment_hea/apology_repair ↑**

Successful romance novels show progression from conflict to resolution, with commitment and repair increasing while miscommunication and negative affect decrease.

## Dataset

### Corpus Description

The dataset includes **105 standalone billionaire romance novels** by **35 different authors**, selected from curated lists such as "100 Best Billionaire Romance Books of All Time". Each novel contains at least **100,000 words**, resulting in a dataset of **680,822 sentences** organized hierarchically by:

- **Author** → **Book** → **Chapter** → **Sentence**

This structure facilitates multi-level analyses, such as tracking topic evolution within a chapter or comparing thematic progression across multiple books.

### Data Contracts (Inputs)

#### Required Inputs

1. **`topics.json`**
   - Format: `{topic_id: [top_words]}`
   - Description: List of topics with top words per topic

2. **`book_topic_probs.csv`**
   - Columns: `book_id`, `topic_id`, `prob`
   - Description: Per-book topic mixture probabilities

3. **`books_meta.csv`**
   - Columns: `book_id`, `author_id`, `group` ∈ {Top, Medium, Trash}, `avg_rating`, `n_ratings`, `length_tokens|words`, optional `year`
   - Description: Book metadata with popularity grouping

4. **`chapter_topic_probs.csv`**
   - Columns: `book_id`, `chapter_id/segment` ∈ {begin, middle, end}, `topic_id`, `prob`
   - Description: Per-chapter topic mixtures
   - **Note**: If not provided, derive begin/middle/end by splitting each book's token stream into tertiles and re-infer topic mixtures per tertile

## Methodology

### Topic Modeling: BERTopic with OCTIS Optimization

#### Why BERTopic?

To overcome the limitations of traditional topic models (e.g., LDA), this study employs **BERTopic** (Grootendorst, 2022), a neural network-based method that uses BERT embeddings to understand word meanings in context. Unlike Latent Dirichlet Allocation (LDA), BERTopic captures more complex relationships between words, resulting in clearer and more meaningful topics (Egger & Yu, 2022; Gan et al., 2024; Sy, 2024; Liu, 2024).

#### Hyperparameter Optimization

Since BERTopic performance strongly depends on hyperparameters and the quality of sentence embeddings, we use the **OCTIS framework** (Terragni, Fersini, Galuzzi, Tropeano, & Candelieri, 2021) for Bayesian optimization.

**Hyperparameters Optimized:**
- **UMAP settings**: `n_neighbors`, `n_components`, `min_dist`
- **HDBSCAN settings**: `min_cluster_size`, `min_samples`
- **Vectorizer settings**: `min_df`, `stop_words`
- **BERTopic-specific**: `top_n_words`, `min_topic_size`, `n_gram_range`

**Embedding Models Evaluated:**
Six pre-trained sentence embedding models from SentenceTransformers:
- `all-MiniLM-L12-v2`
- `multi-qa-mpnet-base-cos-v1`
- `paraphrase-distilroberta-base-v1`
- `paraphrase-MiniLM-L6-v2`
- `paraphrase-mpnet-base-v2`
- `whaleloops-phrase-bert`

**Total Models Analyzed:** Over 300 different BERTopic configurations

#### Model Evaluation

Each BERTopic model is evaluated on two main criteria (Röder, Both, & Hinneburg, 2015):

1. **Coherence**: Measures how logically consistent and interpretable the topics are
2. **Diversity**: Evaluates the variety of topics generated to minimize redundancy

#### Pareto Efficiency Analysis

Once models are created and evaluated, a **Pareto efficiency analysis** (Liu et al., 2022) is performed to identify the best-performing models that have an optimal balance between coherence and diversity.

**Weighting Schemes:**
1. **Equal weights**: 50% coherence, 50% diversity
2. **Coherence priority**: 70% coherence, 30% diversity

Based on this analysis, the optimal model is selected from the top 10 models.

### Thematic Mapping: Topic → Category

Topics are mapped to **16 thematic composites** (A-P) using a semi-supervised approach:

- **Manual mapping**: Markdown files with researcher-defined topic-to-cluster mappings
- **Codebook**: CSV file with structured category definitions
- **Automated assignment**: Weighted topic-to-composite assignments based on keyword matching

**Composites:**
- **A**: Reassurance/Commitment
- **B**: Mutual Intimacy
- **C**: Explicit Eroticism
- **D**: Power/Wealth/Luxury
- **E**: Coercion/Brutality/Danger
- **F**: Angst/Negative Affect
- **G**: Courtship Rituals/Gifts
- **H**: Domestic Nesting
- **I**: Humor/Lightness
- **J**: Social Support/Kin
- **K**: Professional Intrusion
- **L**: Vices/Addictions
- **M**: Health/Recovery/Growth
- **N**: Separation/Reunion
- **O**: Aesthetics/Appearance
- **P**: Tech/Media Presence

### Derived Indices (per book & per segment)

#### Love-over-Sex
```
(commitment_hea + tenderness_emotion) − explicit
```
Measures the balance between emotional connection and explicit content.

#### HEA Index
```
commitment_hea + symbolic_gifts_jewelry + festive_rituals
```
Quantifies Happily Ever After indicators.

#### Explicitness Ratio
```
explicit / (explicit + commitment_hea + tenderness_emotion + 1e-9)
```
Proportion of explicit content relative to emotional themes.

#### Luxury Saturation
```
luxury_wealth + luxury_mobility_settings + luxury_consumption_style + nightlife_party_glamour
```
Measures the presence of luxury and wealth themes.

#### Corporate Frame Share
```
corporate_power + office_space + meetings_board
```
Proportion of corporate/professional themes.

#### Family/Fertility Index
```
family + fertility_pregnancy_baby + domestic_staff_childcare
```
Family and fertility-related themes.

#### Comms Density
```
comms + public_image_scandal
```
Communication and public image themes.

#### Dark-vs-Tender
```
(neg_affect + threat_violence_dark) − tenderness_emotion
```
Balance between dark themes and tenderness.

#### Miscommunication Balance
```
(commitment_hea + tenderness_emotion + apology_repair) − miscommunication
```
Resolution vs. conflict themes.

#### Protective–Jealousy Delta
```
protectiveness_care − jealousy_possessiveness
```
Caring protectiveness vs. jealous possessiveness.

## Statistical Analysis Plan

### 1. Validation & Preparation

- Schema checks
- Normalize topic probabilities (sum to 1 per book)
- Merge metadata
- Create segments (begin/middle/end) if not supplied

### 2. Map & Aggregate

- Run mapping pipeline → `topic_to_category_probs.json`
- Roll up to book and segment category proportions
- Compute all indices

### 3. Descriptives & Visualization

- Heatmaps of category proportions by group
- Group means ± 95% CI for indices
- UMAP / clustering on `book_category_props` (color by group)

### 4. Group Comparisons

**ANOVA** (or **Kruskal–Wallis** if non-normal) on indices across Top/Medium/Trash:
- Post-hoc tests with **Holm correction**
- Effect sizes: **Cohen's d** for continuous, **Cramér's V** for categorical

**χ² tests** on category presence/absence (or GLMs on proportions)

### 5. Modeling

#### Logistic Regression
**Outcome**: Top (1) vs Trash (0)  
**Predictors**: All indices  
**Controls**: `author_id` (fixed effects), `length`, `year`

#### OLS Regression
**Outcome**: `avg_rating`  
**Predictors**: Indices + controls

#### Key Interactions
- **Luxury × (Commitment+Tenderness)**: Tests H3
- **Contractual × Tenderness**
- **PublicImage × Commitment**
- **Protective–Jealousy**: Tests H4

### 6. Time-Course Analysis (Arc)

**Repeated-measures ANOVA** or **mixed-effects models** with:
- **Segment** (begin/middle/end) as within-subject factor
- **Category proportions** or **indices** as outcomes
- Tests H6 trends: commitment_hea/apology_repair ↑; miscommunication/neg_affect ↓

### 7. Robustness Checks

- Sensitivity to alternative thresholds
- Bootstrapped confidence intervals
- Leave-one-author-out validation

## Acceptance Criteria

- ✅ `topic_to_category_probs.json` & `topic_to_category_final.csv` (F1 ≥ target on small gold set)
- ✅ `book_category_props.csv` + `chapter_category_props.csv`
- ✅ Indices computed for all books (and segments)
- ✅ Group comparisons + effect sizes; models with coefficients & CIs
- ✅ Figures saved; one report notebook summarizing findings

## Deliverables

1. **Mapping Files**
   - `topic_to_category_probs.json`
   - `topic_to_category_final.csv`

2. **Aggregated Data**
   - `book_category_props.csv`
   - `chapter_category_props.csv` (if applicable)

3. **Indices Table**
   - All derived indices per book and per segment

4. **Statistical Tables**
   - Group comparison tests
   - Model coefficients with confidence intervals
   - Interaction effects

5. **Figures** (PNG/SVG)
   - Heatmaps
   - Group comparison plots
   - UMAP visualizations
   - Time-course plots

6. **Report Notebook**
   - `report.ipynb` summarizing all findings

## Technical Infrastructure

### GPU Acceleration

**Mandatory RAPIDS cuML** (CUDA 12.x) for:
- GPU-accelerated UMAP (`cuml.manifold.UMAP`)
- GPU-accelerated HDBSCAN (`cuml.cluster.HDBSCAN`)

**No CPU fallback** - system requires CUDA-compatible GPU.

### Software Stack

- **Python 3.12+**
- **BERTopic** (Grootendorst, 2022)
- **OCTIS** (Terragni et al., 2021)
- **RAPIDS cuML** for GPU acceleration
- **SentenceTransformers** for embeddings
- **scikit-learn** for statistical analysis
- **pandas**, **numpy** for data manipulation

## References

- Egger, R., & Yu, J. (2022). A topic modeling comparison between LDA, NMF, Top2Vec, and BERTopic to demystify Twitter posts. *Frontiers in Sociology*, 7, 886498.

- Gan, J., Qi, Z., Li, Z., & Zhang, Y. (2024). BERTopic for short texts. *arXiv preprint arXiv:2401.00724*.

- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv preprint arXiv:2203.05794*.

- Liu, Y., et al. (2022). Pareto efficiency in multi-objective optimization. [Reference details]

- Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. *Proceedings of WSDM*.

- Sy, K. (2024). [Reference details on BERTopic]

- Terragni, S., Fersini, E., Galuzzi, B. G., Tropeano, P., & Candelieri, A. (2021). OCTIS: Comparing and optimizing topic models is simple! *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics*.

---

For technical implementation details, see [docs/METHODOLOGY.md](docs/METHODOLOGY.md).  
For data contract specifications, see [docs/DATA_CONTRACTS.md](docs/DATA_CONTRACTS.md).  
For index definitions, see [docs/INDICES.md](docs/INDICES.md).

