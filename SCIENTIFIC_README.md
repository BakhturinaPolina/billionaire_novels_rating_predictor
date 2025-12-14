# Scientific Methodology: Modern Romantic Novels — Themes × Popularity

**A Mixed-Methods Computational Analysis**

## Research Objectives

1. **Map topic-model outputs** from modern romance novels to theory-driven themes and test which themes differentiate Top / Medium / Trash popularity tiers.

2. **Build explainable indices** to quantify narrative qualities readers value.

3. **Validate corpus findings** against Goodreads metadata (rating and number of voters).

## Research Questions

1. Which theme categories are most prevalent in Top vs Middle vs Trash novels?

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

### Topic Exploration & Evaluation

After retraining the top-performing models, we conduct comprehensive topic exploration and evaluation:

#### Multiple Representation Strategies

Each retrained BERTopic model is enriched with multiple representation strategies to capture different aspects of topics:

1. **Main Representation**: Standard c-TF-IDF representation (BERTopic default)
2. **KeyBERT**: Keyword extraction using KeyBERT-inspired representation
3. **POS (Part-of-Speech)**: Filters keywords by part-of-speech patterns (nouns, verbs, adjectives)
4. **MMR (Maximal Marginal Relevance)**: Balances keyword relevance with diversity (diversity=0.3)

These representations are attached to the model using BERTopic's `update_topics()` method, allowing for multi-faceted topic analysis.

#### Coherence & Diversity Evaluation

For each representation, we compute:

- **c_v Coherence** (Röder, Both, & Hinneburg, 2015): Measures semantic consistency of topics using a sliding window approach. Higher values indicate more interpretable topics.
- **Topic Diversity**: Ratio of unique words to total extracted terms. Higher values indicate less redundancy across topics.

Metrics are computed using the same gensim dictionary built from the OCTIS corpus used during retraining, ensuring consistency with the training vocabulary.

#### Topic Extraction for Close Reading

All topics with all representations are extracted and saved to JSON format for qualitative evaluation. This enables researchers to:
- Compare topic quality across different representations
- Identify the most interpretable representation for each topic
- Conduct close reading of topic keywords for thematic analysis

#### Topic Quality Analysis & Noisy Topic Detection

Before automated labeling, we conduct quality analysis to identify candidate noisy topics that may not be suitable for LLM labeling or downstream analysis. This process:

1. **Computes Quality Metrics**:
   - Topic size (number of documents assigned to each topic)
   - POS representation statistics (count of POS-filtered keywords per topic)
   - Per-topic POS coherence (c_v coherence computed on POS-filtered keywords using the same gensim dictionary as training)

2. **Flags Noisy Candidates**:
   - Topics with few POS words (< 3): Indicates topics that may lack interpretable keywords
   - Topics with low or missing POS coherence (< 0.0): Suggests semantically incoherent topics
   - Topics below minimum size threshold (< 30 documents): May represent outliers or noise

3. **Labels for Manual Inspection**:
   - Noisy topics are labeled with inspection tags (e.g., `[NOISE:few_pos<3]`, `[NOISE:low_coh<0.00]`)
   - Labels are applied to both wrapper pickle and native BERTopic model formats
   - Quality tables are saved to CSV for review (`topic_quality_{model}.csv`, `topic_noise_candidates_{model}.csv`)

This quality control step ensures that automated labeling and category mapping focus on interpretable, coherent topics, improving the reliability of downstream analyses.

### Stage 08: Automated Topic Labeling

#### Label Generation

To generate human-readable labels for topics, we employ two approaches:

**1. OpenRouter API (Recommended)**
- **Model**: `mistralai/mistral-nemo` via OpenRouter API
- **Advantages**: No local GPU required, faster iteration, cloud-based inference
- **Workflow**: Same prompt structure and domain detection as local inference
- See `src/stage08_llm_labeling/openrouter_experiments/` for details

**2. Local Mistral-7B-Instruct**
- **Model**: Mistral-7B-Instruct-v0.2 (Jiang et al., 2023) with 4-bit quantization
- **Memory Requirements**: ~6GB VRAM with quantization (vs. ~14GB without)
- **Device**: GPU-accelerated when available, with CPU fallback

**Label Generation Process**:
1. **Keyword Extraction**: Extract top keywords from POS representation (default: 15 keywords per topic)
2. **MMR Reranking**: Apply Maximal Marginal Relevance (MMR) reranking to balance keyword relevance with diversity, ensuring the model receives a diverse set of representative keywords
3. **Domain Detection**: Automatically detect semantic domains (e.g., BodyParts, FoodDrink, TimeSpan, Marriage) from keywords to provide context-aware hints
4. **Representative Snippets**: Extract 3-6 sentence snippets from documents in each topic to provide rich contextual evidence
5. **Label Generation**: Use Mistral-7B-Instruct with romance-aware prompts that include domain-specific hints and representative snippets for more accurate labeling
6. **Integration**: Automatically integrate generated labels back into BERTopic models for use in visualizations

#### Romance-Aware Prompt Design

**Core Principle:** The prompt must be **domain-specific** (romance/erotic fiction) while maintaining **research rigor** (no hallucination, literal interpretation).

**System Prompt Architecture:**
1. **Role Definition**: Establishes domain context (romantic and erotic fiction) and research acceptability of explicit terminology
2. **General Rules**: Format constraints (2-6 words, no quotes, no markdown, concrete scene-level descriptions)
3. **Priority Hierarchy**: Action → Role → Setting → Tone (reflects how humans read romance fiction topics)
4. **Snippet Integration**: Representative document snippets serve as primary evidence; "When snippets and keywords disagree, trust the snippets"
5. **Disambiguation Requirements**: Prevents label collisions by encoding distinguishing features explicitly
6. **Anti-Hallucination Constraints**: Hard rules for known hallucination patterns (e.g., "dinner date", "invitation", "repair", "heartbreak")

**Key Design Features:**
- **Snippet-Based Evidence**: Keywords are sparse and ambiguous; snippets provide rich context for fine distinctions (rough vs gentle kisses, emotional vs physical rage, specific sexual acts)
- **Priority Hierarchy**: Most important is what sexual/romantic act is happening; least important is abstract emotional tone (only if clearly indicated)
- **Explicit Sexual Terminology**: Clinical, non-romanticized phrasing is acceptable and encouraged (e.g., "Oral Sex on Him", "Clitoral Stimulation", "Anal Sex")
- **Anti-Hallucination Rules**: Explicit prohibitions prevent common model hallucinations (e.g., "Do NOT use 'dinner date' unless snippets clearly mention asking/inviting")

#### JSON Output Format

When using romance-aware prompts (`--use-improved-prompts`), the system produces structured JSON output with the following fields:

- **`label`**: Short noun phrase (2-6 words) describing the topic
- **`scene_summary`**: One complete sentence (12-25 words) describing a typical scene
- **`primary_categories`**: 1-3 high-level tags (e.g., "romance_core", "sexual_content", "work_life", "daily_routine")
- **`secondary_categories`**: 0-5 specific tags with dimension:value format (e.g., "setting:car", "activity:kissing", "emotion:anticipation", "temporal:waiting")
- **`is_noise`**: Boolean indicating if the topic is a technical artifact or meaningless
- **`rationale`**: 1-3 sentences explaining how keywords and snippets support the label

The JSON parsing pipeline automatically extracts all these fields and includes them in the output JSON file. This provides richer metadata for downstream analysis while maintaining backward compatibility with the label-only format.

**Category Taxonomy:**
- **Primary Categories**: High-level thematic tags including `romance_core`, `sexual_content`, `work_life`, `daily_routine`, `dating_ritual`
- **Secondary Categories**: Dimension:value pairs capturing specific aspects:
  - **Setting**: `setting:car`, `setting:casual`, `setting:office`
  - **Activity**: `activity:kissing`, `activity:negotiation`, `activity:eating`, `activity:invitation`
  - **Emotion**: `emotion:uncertainty`, `emotion:reluctance`, `emotion:anticipation`, `emotion:change`
  - **Temporal**: `temporal:waiting`, `temporal:passing`
  - **Stage**: `stage:undefined`

#### Label Quality Features

- **Romance-Aware Prompting**: Domain-specific system prompt designed for modern romantic and erotic fiction
- **Representative Snippets**: Uses actual document snippets (3-6 sentences) as primary evidence for label generation
- **Adaptive Context Hints**: Domain-specific hints generated from keyword analysis (e.g., "If body parts or intimacy are clear, name the exact parts")
- **Post-processing**: Automatic cleanup of labels (removes quotes, trailing punctuation, incomplete phrases)
- **Streaming Support**: Memory-efficient processing for large topic sets

#### Technical Specifications

- **Default Parameters**: 15 keywords per topic, 40 max tokens per label
- **Output Format**: JSON file with structured fields: `{"topic_id": {"label": "...", "scene_summary": "...", "primary_categories": [...], "secondary_categories": [...], "is_noise": false, "rationale": "...", "keywords": [...]}}`

### Stage 09: Category Mapping: Theory-Aligned Tagging

#### Overview

After generating human-readable topic labels (from Stage 08), we map topics to theory-aligned categories using a **three-stage zero-shot classification approach**. This operationalizes theoretical constructs from Radway (1984), Propp functions, and Ogas & Gaddam (2011), enabling quantitative hypothesis testing. The final goal is to construct **19 theory-aligned composite categories (A-S)** from the detailed classifications produced in Stages 2 and 3.

#### Three-Stage Classification Pipeline

**Stage 1: Natural Clusters** (optional)
- Data-driven topic groupings using BERTopic's hierarchical topics
- Reduces topics to interpretable meta-topics (40-80 topics)
- Provides baseline for comparison with theory-driven approaches

**Stage 2: Theory-Driven Taxonomy Classification** ✅ **Implemented**
- **Method**: Zero-shot classification to **Romance Corpus Topic Taxonomy** using Mistral-Nemo via OpenRouter
- **Output**: Each topic mapped to taxonomy nodes (30+ nodes across 8 groups)
- **Taxonomy Structure**:
  1. **Embodied & Sensory Experience** (1.1, 1.2, 1.5): Body parts, pain/vulnerability, physical activity
  2. **Sexuality, Attraction & Intimacy** (2.1, 2.2, 2.3, 2.4): Attraction, kissing, explicit acts, aftercare
  3. **Emotions, Cognition & Inner Life** (3.1, 3.2, 3.3, 3.4): Positive emotions, negative emotions, ambivalence, beliefs/values
  4. **Relationship Trajectory (Main Couple)** (4.1, 4.2, 4.3, 4.4, 4.5): Meeting, bonding, secrets/misunderstandings, conflict/breakup, reconciliation/HEA
  5. **Social World Outside Couple** (5.1, 5.2, 5.3): Family/kinship, friends/social circles, community/norms
  6. **Work, Wealth, Status & Institutions** (6.1, 6.2, 6.3, 6.4, 6.5): Hero's work, heroine's work, shared workplaces, money/housing, formal institutions
  7. **Conflict, Risk & Harm** (7.1, 7.2, 7.3): Interpersonal conflict, violence/coercion, external crises
  8. **Spaces, Time, Activities & Objects** (8.1, 8.2, 8.3, 8.4): Domestic spaces, public/leisure, objects/technology, temporal framing
- **Special Category**: `noise` for boilerplate/technical artifacts
- Uses topic keywords, LLM-generated labels, scene summaries, and representative document snippets
- Output: JSON with `main_category_id`, `secondary_category_id`, `other_plausible_ids`, `confidence`, `rationale`

**Stage 3: Radway Narrative Functions** ✅ **Implemented**
- **Method**: Zero-shot classification to **Radway's 13 narrative functions** using Mistral-Nemo via OpenRouter
- **Output**: Each topic mapped to Radway functions (R1-R13) organized into three phases
- **Radway Functions by Phase**:
  - **Phase I: Initial Conflict & Isolation**: R1 (identity destroyed), R2 (antagonistic reaction), R3 (ambiguous response), R4 (sexual interest interpretation), R5 (anger/coldness), R6 (retaliation), R7 (separation)
  - **Phase II: Turning Point & Recognition**: R8 (tenderness), R9 (warm response), R10 (reinterpretation)
  - **Phase III: Commitment & Restoration**: R11 (love declaration/commitment), R12 (sexual/emotional response), R13 (identity restored)
- Uses Stage 2 taxonomy classifications, topic keywords, labels, scene summaries, and representative snippets
- Includes heuristic overrides for systematic errors (e.g., explicit sex scenes 2.3 → R12, commitment cues → R11/R13)
- Output: Merged JSON preserving all Stage 2 fields plus `radway_functions` object with `radway_main_id`, `radway_secondary_id`, `radway_phase`, `radway_confidence`, `radway_rationale`

#### Target: Theory-Aligned Composite Categories (A-S)

**Next Urgent Step**: Build the **19 theory-aligned composite categories** from Stage 2 taxonomy nodes and Stage 3 Radway functions. These composites operationalize the research framework:

**Core Composites (A-P)**: 16 thematic categories:
- **A**: Reassurance/Commitment (HEA centrality, Propp functions #8–#11) ← *From taxonomy 4.5 (Reconciliation/HEA), Radway R11/R13*
- **B**: Mutual Intimacy (non-explicit; love-over-sex preference) ← *From taxonomy 2.2 (Kissing/Non-Explicit), Radway R8/R9*
- **C**: Explicit Eroticism (contrast against B; explicitness ratio) ← *From taxonomy 2.3 (Explicit Sexual Acts), Radway R12*
- **D**: Power/Wealth/Luxury (therapeutic safety; luxury × love interaction) ← *From taxonomy 6.1/6.4 (Elite Work, Money/Housing)*
- **E**: Coercion/Brutality/Danger (dark themes; Dark-vs-Tender) ← *From taxonomy 7.2/7.3 (Violence/Coercion, External Crises), Radway R6*
- **F**: Angst/Negative Affect (emotional escape vs angst; Radway's conflict arc) ← *From taxonomy 3.2 (Negative Emotions), 4.4 (Conflict/Breakup), Radway R1-R7*
- **G**: Courtship Rituals/Gifts (romantic rituals; HEA Index component) ← *From taxonomy 4.2 (Bonding/Intimacy), 8.3 (Objects/Technology)*
- **H**: Domestic Nesting (compensatory safety; home as refuge) ← *From taxonomy 8.1 (Domestic Spaces), 4.2 (Bonding)*
- **I**: Humor/Lightness (binge-readability; escape via lightness) ← *From taxonomy 3.1 (Positive Emotions), 8.2 (Leisure Spaces)*
- **J**: Social Support/Kin (stable social buffers; Family/Fertility Index) ← *From taxonomy 5.1/5.2 (Family, Friends)*
- **K**: Professional Intrusion (office romance trope; Corporate Frame Share) ← *From taxonomy 6.2/6.3 (Heroine's Work, Shared Workplaces)*
- **L**: Vices/Addictions (escape contrast; may reduce appeal) ← *From taxonomy 1.2 (Pain/Vulnerability), 3.3 (Ambivalence)*
- **M**: Health/Recovery/Growth (protective care; vulnerability → tenderness) ← *From taxonomy 1.2 (Pain/Vulnerability), 3.4 (Beliefs/Values), Radway R8-R10*
- **N**: Separation/Reunion (Propp/Radway arc; time-course H6) ← *From taxonomy 4.4/4.5 (Breakup/Reconciliation), Radway R7/R11*
- **O**: Aesthetics/Appearance ("detective agency"; physical/cultural cues) ← *From taxonomy 1.1 (Body Parts), 8.3 (Objects)*
- **P**: Tech/Media Presence (modern courtship infrastructure; Comms Density) ← *From taxonomy 8.3 (Technology), 6.5 (Institutions)*

**Cross-Cutting Categories**:
- **Q**: Miscommunication vs Repair (Radway's mid-arc; Miscommunication Balance) ← *From taxonomy 4.3 (Secrets/Misunderstandings), Radway R10*
- **R**: Protectiveness vs Jealousy ("strong but gentle"; Protective–Jealousy Delta, H4) ← *From taxonomy 3.1/3.2 (Positive/Negative Emotions), Radway R8/R6*

**Auxiliary Categories**:
- **S**: Scene Anchors (formulaic scene kits; qualitative sampling) ← *From taxonomy noise + formulaic patterns*

#### Operationalization of Hypotheses (Target)

Once composite categories (A-S) are built from Stage 2 and Stage 3 classifications, they will directly operationalize all research hypotheses:

- **H1 (Love-over-Sex)**: `(A_commitment_hea + B_mutual_intimacy) > C_explicit`
- **H2 (HEA Index)**: `A_commitment_hea + G_rituals_gifts`
- **H3 (Luxury × Love)**: `D_luxury_wealth_status × (A_commitment_hea + B_mutual_intimacy)`
- **H4 (Protectiveness vs Jealousy)**: `R_protectiveness - R_jealousy` (from R split 50/50)
- **H5 (Darkness vs Tenderness)**: `(F_negative_affect + E_threat_danger) - B_mutual_intimacy`
- **H6 (Narrative Arc)**: Time-course analysis with `Q_miscomm ↓`, `Q_repair ↑`, `F_negative_affect ↓`, `A_commitment_hea ↑`

#### Current Output Files

**Stage 2 Output**:
- **`taxonomy_mappings_*.json`**: Per-topic taxonomy classifications with main/secondary/other plausible IDs

**Stage 3 Output**:
- **`taxonomy_with_radway.json`**: Merged JSON with taxonomy + Radway function mappings
- **BERTopic models**: `model_1_with_radway_mappings` (Radway functions attached)

**Target Output** (to be implemented):
- **`topic_to_category_probs.json`**: Per-topic soft composite category assignments (A-S)
- **`topic_to_category_final.csv`**: Flat table format for inspection
- **`book_category_props.csv`**: Book-level category proportions
- **`indices_book.csv`**: All derived indices per book (Love-over-Sex, HEA Index, etc.)

See `src/stage09_category_mapping/README.md` for detailed documentation.

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

### Stage 10: Statistical Analysis & Correlation Analysis

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
- **BERTopic** (Grootendorst, 2022) for topic modeling
- **OCTIS** (Terragni et al., 2021) for hyperparameter optimization
- **RAPIDS cuML** for GPU acceleration
- **SentenceTransformers** for embeddings
- **Transformers** (Hugging Face) for Mistral-7B-Instruct label generation
- **bitsandbytes** for 4-bit model quantization
- **gensim** for coherence evaluation
- **scikit-learn** for statistical analysis
- **pandas**, **numpy** for data manipulation

## References

- Egger, R., & Yu, J. (2022). A topic modeling comparison between LDA, NMF, Top2Vec, and BERTopic to demystify Twitter posts. *Frontiers in Sociology*, 7, 886498.

- Gan, J., Qi, Z., Li, Z., & Zhang, Y. (2024). BERTopic for short texts. *arXiv preprint arXiv:2401.00724*.

- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv preprint arXiv:2203.05794*.

- Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. d. l., ... & Lample, G. (2023). Mistral 7B. *arXiv preprint arXiv:2310.06825*.

- Liu, Y., et al. (2022). Pareto efficiency in multi-objective optimization. [Reference details]

- Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. *Proceedings of WSDM*.

- Sy, K. (2024). [Reference details on BERTopic]

- Terragni, S., Fersini, E., Galuzzi, B. G., Tropeano, P., & Candelieri, A. (2021). OCTIS: Comparing and optimizing topic models is simple! *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics*.

---

For technical implementation details, see [docs/METHODOLOGY.md](docs/METHODOLOGY.md).  
For data contract specifications, see [docs/DATA_CONTRACTS.md](docs/DATA_CONTRACTS.md).  
For index definitions, see [docs/INDICES.md](docs/INDICES.md).

