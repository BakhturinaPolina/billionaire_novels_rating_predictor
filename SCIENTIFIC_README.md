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

**Final Analysis Sample**: After data preparation and quality filtering, the final statistical analysis uses **92 books** (30 top-tier, 32 middle-tier, 30 trash-tier) with complete Goodreads metadata and topic probability assignments. Five books were excluded during preprocessing (19561986, 19619918, 25781538, 52061964, 53491034) due to missing sentence-level data.

### Data Sources

#### Raw Text Files
- **Location**: `data/raw/Billionaire_Full_Novels_TXT/`
- **Format**: Plain text files (`.txt`) or EPUB files
- **Content**: Full novel texts, one file per book
- **Encoding**: UTF-8 (with handling for encoding issues and mojibake correction)

#### Goodreads Metadata
- **Location**: `data/processed/goodreads.csv`
- **Format**: CSV with columns: `ID`, `Author`, `Title`, `Score`, `RatingsCount`, `ReviewsCount`, `Pages`
- **Purpose**: Provides popularity metrics and quality indicators for grouping books into Top/Middle/Trash tiers
- **Statistics**: 
  - 97-98 books with complete metadata (92 books in final analysis after exclusions)
  - Rating distribution: Mean 3.99, Std 0.21, Range 3.26-4.42
  - All books have ≥100 ratings (minimum: 146, mean: 65,849)
  - Final tier distribution: 30 top-tier (avg_rating ≈ 4.22, n_ratings ≈ 116k), 32 middle-tier (avg_rating ≈ 4.01, n_ratings ≈ 44k), 30 trash-tier (avg_rating ≈ 3.77, n_ratings ≈ 48k)
- **Matching Strategy**: Fuzzy matching with configurable threshold (default: 0.85 similarity) to handle author/title format differences between text files and Goodreads metadata

#### BookNLP Outputs (Optional)
- **Location**: `data/interim/booknlp/`
- **Format**: BookNLP processing outputs (character entities, quotes, etc.)
- **Purpose**: Provides character name extraction for stopword generation (used in Stage 02 preprocessing)
- **Usage**: Character names extracted from 7,525 lines, resulting in 4,444 unique character name tokens added to stopwords list

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

**Character Name Exclusion:**
To improve topic interpretability by focusing on thematic content rather than character co-occurrence patterns, we implement character name exclusion during preprocessing (Stage 02). The pipeline:

**Processing Pipeline:**
- Processes 7,525 character name lines from romance novel texts
- Extracts 4,497 unique name tokens (4,444 after overlap removal with standard stopwords)
- Adds character names to stopwords list, resulting in 4,762 total stopwords (93% character names, 7% standard English stopwords)
- This represents a **14x increase** in stopwords compared to standard English stopword lists

**Character Name Processing Steps:**
- **Cleaning**: Removes leading prefixes ("A ", "Mr.", "Miss ", "the ", "AKA ", "#"), leading numbers, quotes and punctuation; converts to lowercase
- **Filtering**: Removes empty lines, very long lines (>50 characters), common phrases, descriptive patterns, geographic locations
- **Name Extraction**: Splits multi-word names (e.g., "Alex Crane" → extracts both "alex" and "crane") to ensure both first and last names are filtered

**Processing Statistics:**
- Total lines processed: 7,525
- Lines filtered out: 254 (3.4%)
- Lines with valid names: 7,271
- Multi-word names processed: 3,313
- Unique name tokens extracted: 4,497
- Final character names added to stopwords: 4,444

**Precision Trade-offs:**
The preprocessing retained some words that are not character names (estimated 1-2% of tokens), but we prioritize coverage over precision to ensure comprehensive character name exclusion. This approach ensures topics reflect thematic relationships rather than character references, aligning with computational literary analysis best practices (Bamman et al., 2013; Jockers, 2013).

**GPU Acceleration:**
All modeling stages use **RAPIDS cuML** (CUDA 12.x) for mandatory GPU acceleration:
- `cuml.manifold.UMAP` for dimensionality reduction
- `cuml.cluster.HDBSCAN` for clustering
- No CPU fallback - requires CUDA-compatible GPU

**Embedding Caching:**
Embeddings are cached to avoid recomputation across model training iterations, significantly reducing processing time.

#### Model Evaluation

Each BERTopic model is evaluated on two main criteria (Röder, Both, & Hinneburg, 2015):

1. **Coherence**: Measures how logically consistent and interpretable the topics are
2. **Diversity**: Evaluates the variety of topics generated to minimize redundancy

#### Pareto Efficiency Analysis

Once models are created and evaluated, a **Pareto efficiency analysis** (Liu et al., 2022) is performed to identify the best-performing models that have an optimal balance between coherence and diversity.

**Data Cleaning Pipeline:**
Before Pareto analysis, a two-stage cleaning process removes invalid configurations:
1. **Failed Run Removal**: Removes configurations where `Coherence = 1.0` or `Topic_Diversity = 1.0` (indicating model failures)
2. **Statistical Outlier Removal**: Applies z-score method (2 standard deviations) and IQR method (1.5× multiplier) to remove extreme outliers
3. **Domain-Specific Filtering**: Removes models with artificially high diversity (>0.9) that likely resulted from too few topics

**Final Results:**
After outlier filtering, **4 Pareto-efficient configurations** were identified (down from 12 before filtering):
- **Top Performer**: `paraphrase-mpnet-base-v2`, iteration 0
  - Coherence: 0.463 (highest among Pareto-efficient models)
  - Topic Diversity: 0.82
  - Combined Score: 1.75
  - Best balance between coherence and diversity

**Weighting Schemes:**
1. **Equal weights**: 50% coherence, 50% diversity
2. **Coherence priority**: 70% coherence, 30% diversity

**Hyperparameter Correlation Analysis:**
Statistical analysis of hyperparameter effects reveals:
- **UMAP parameters** (`umap__min_dist`, `umap__n_components`) show strong, significant effects on diversity and combined scores
- **Vectorizer parameter** (`vectorizer__min_df`) is crucial for overall performance (r = 0.745, p = 0.014 for combined score)
- **Cluster size parameters** (`bertopic__min_topic_size`, `hdbscan__min_cluster_size`) primarily affect coherence
- **Trade-offs identified**: Coherence and diversity show opposing relationships with several parameters (e.g., `bertopic__min_topic_size` improves coherence but reduces diversity)

The analysis employs multiple statistical methods (correlation analysis, linear regression, multicollinearity assessment, tree-based feature importance) with explicit assumption checking to ensure valid inference.

### Stage 05: Model Retraining

After Pareto analysis identifies optimal configurations, the top N Pareto-efficient models are retrained with their exact hyperparameters for final deployment. This stage:

**Key Features:**
- **Direct retraining** from Pareto-efficient model configurations (no OCTIS optimization)
- **Multiple output formats**: Pickle (full wrapper), BERTopic native format (safetensors), and metadata JSON
- **Independent model training**: Failures in one model don't stop others
- **Embedding caching**: Reuses embeddings from Stage 03 to avoid recomputation
- **Character name exclusion**: Same preprocessing pipeline as Stage 03
- **GPU acceleration**: Uses RAPIDS (cuML) - same as Stage 03

**Output Formats:**
1. **Pickle format** (`.pkl`): Full `RetrainableBERTopicModel` instance for direct Python loading
2. **BERTopic native format** (directory): Standard BERTopic format using safetensors for production deployment
3. **Metadata JSON**: Comprehensive training metadata including hyperparameters, evaluation scores, topic counts, and timestamps

**Model Statistics:**
Each retrained model includes topic count, full hyperparameter configuration, evaluation scores (coherence, diversity, combined score), and training metadata.

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

**Results for Selected Model** (`paraphrase-MiniLM-L6-v2`, Pareto rank 1, 368 topics):

| Representation | Coherence (c_v) | Topic Diversity |
|---------------|-----------------|------------------|
| Main          | 0.404           | 0.602            |
| KeyBERT       | 0.278           | 0.645            |
| POS           | 0.315           | 0.692            |
| MMR           | 0.260           | 0.756            |

**Representation Selection Recommendations:**
- **For LLM labeling (Stage 08)**: POS representation recommended (interpretable content words, balanced coherence and diversity)
- **For exploratory analysis**: MMR representation recommended (highest diversity, 0.756)
- **For statistical validation**: Main representation recommended (highest coherence, 0.404)

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
   - Noisy topics are labeled with inspection tags (e.g., `[NOISE_CANDIDATE:few_pos<3]`, `[NOISE_CANDIDATE:low_coh<0.00]`)
   - Labels are applied to both wrapper pickle and native BERTopic model formats
   - Quality tables are saved to CSV for review (`topic_quality_{model}.csv`, `topic_noise_candidates_{model}.csv`)

**Results for Selected Model** (`paraphrase-MiniLM-L6-v2`, 368 topics):
- **Total topics analyzed**: 368 (excluding outlier topic -1)
- **Candidate noisy topics identified**: 13 (3.5% of all topics)
- **Topics with POS words < 10**: 20 (5.4% of all topics)
- **Topics with valid coherence scores**: 361 (98.1% of all topics)
- **Topics with NaN coherence**: 7 (1.9% - all are noise candidates)

**Noise Detection Patterns:**
- **Large but noisy topics**: Topics 17 and 18 are among the largest (2,062-2,064 documents) but have no valid POS words and cannot compute coherence, suggesting catch-all clusters for unclassifiable content
- **Empty representations**: Topics with empty or near-empty word lists (17, 18, 132, 186, 224) capture uninterpretable content
- **Single-word topics**: Topics with only 1-2 POS words (141, 183, 241, 262, 347) lack semantic richness

This quality control step ensures that automated labeling and category mapping focus on interpretable, coherent topics, improving the reliability of downstream analyses. The analysis reveals that 94.6-96.5% of topics meet quality thresholds, with noise concentrated in a small number of topics requiring special attention.

### Stage 08: Automated Topic Labeling

#### Theoretical Foundations

**The Challenge of Topic Interpretation**: Topic modeling algorithms like BERTopic identify clusters of semantically similar text segments, but they do not provide interpretable labels. The output consists of keyword lists (e.g., "mouth, tongue, suck, lips") that require human interpretation to understand what the topic represents. For large-scale analysis of hundreds of topics across thousands of documents, manual labeling is impractical.

**Why Large Language Models for Labeling?**: LLMs offer a solution by combining:
- **Semantic understanding**: Ability to synthesize meaning from keyword lists
- **Domain knowledge**: Training on diverse text corpora including literary fiction
- **Consistency**: Reproducible labeling across similar topics
- **Scalability**: Can process hundreds of topics automatically

However, LLMs also present challenges:
- **Hallucination**: Tendency to infer details not present in the input
- **Vagueness**: May produce generic labels like "Erotic Intimacy" instead of specific scene descriptions
- **Format compliance**: Must follow strict output requirements (2-6 word labels, JSON structure)

**Zero-Shot Classification**: Zero-shot classification allows mapping topics to predefined categories without training data. This approach is ideal for theory-driven analysis, consistency across topics, and interpretability through clear category definitions.

#### Label Generation

To generate human-readable labels for topics, we employ two approaches:

**1. OpenRouter API (Recommended)**
- **Primary Model**: `mistralai/Mistral-Nemo-Instruct-2407` via OpenRouter API
- **Advantages**: No local GPU required, faster iteration, cloud-based inference, reliable instruction-following
- **Model Selection Rationale**: Nemo-Instruct chosen for research reliability (low hallucination, format compliance, academic tone). Demonstrates strong adherence to prompt constraints and produces consistent, reproducible labels.
- **Comparison Models**: Alternative literary models evaluated include `thedrummer/cydonia-24b-v4.1` and `thedrummer/anubis-70b-v1.1` for enhanced genre awareness and scene distinctions. See model comparison reports for detailed evaluation.
- **Workflow**: Same prompt structure and domain detection as local inference
- **Cost**: ~$0.00005 per topic (~$0.018 for 368 topics), one-time labeling cost
- See `src/stage08_llm_labeling/openrouter_experiments/` for details

**2. Local Mistral-7B-Instruct**
- **Model**: Mistral-7B-Instruct-v0.2 (Jiang et al., 2023) with 4-bit quantization
- **Memory Requirements**: ~6GB VRAM with quantization (vs. ~14GB without)
- **Device**: GPU-accelerated when available, with CPU fallback

**Label Generation Process**:
1. **Keyword Extraction**: Extract top keywords from POS representation (default: 15 keywords per topic)
2. **MMR Reranking**: Apply Maximal Marginal Relevance (MMR) reranking to balance keyword relevance with diversity, ensuring the model receives a diverse set of representative keywords
3. **Domain Detection**: Automatically detect semantic domains (e.g., BodyParts, FoodDrink, TimeSpan, Marriage) from keywords to provide context-aware hints
4. **Representative Snippets**: Extract 3-6 sentence snippets (default: 6 snippets, 200 chars max per snippet) from BERTopic's representative documents. Snippets provide scene-level context, enabling fine distinctions (rough vs gentle kisses, emotional vs physical rage, specific sexual acts). Snippets serve as primary evidence; "When snippets and keywords disagree, trust the snippets"
5. **Label Generation**: Use Mistral-Nemo-Instruct with romance-aware prompts that include domain-specific hints and representative snippets for more accurate labeling
6. **Integration**: Automatically integrate generated labels back into BERTopic models for use in visualizations

#### Representative Snippets: Design and Implementation

**Purpose**: Representative document snippets provide the LLM with actual scene-level context from the corpus, enabling more precise and neutral labels. Instead of relying solely on keyword lists, the model can see patterns in actual sentences, leading to better distinctions (e.g., "Blowjob in Car" vs "Erotic Intimacy").

**Design Decisions**:
- **6 Snippets**: Sweet spot for pattern recognition without overwhelming the model (~75 tokens, <3% of 4k context window)
- **200 Characters Per Snippet**: Average sentence length (~12.5 tokens ≈ 50-60 characters), 200 chars ≈ 3-4 sentences, truncation at word boundaries
- **Representative Documents**: Chosen by BERTopic for their centrality to the topic (using c-TF-IDF and similarity metrics), more informative than random documents

**How Snippets Improve Label Precision**:
- **Disambiguation**: Keywords like "mouth, tongue, suck" are ambiguous; snippets show specific acts (e.g., kneeling, taking into mouth) → "Blowjob in Bed"
- **Prevents Hallucination**: Keywords "board, table, chair" might infer "Board Game Foreplay"; snippets show literal setup → "Board Game Setup"
- **Scene Context**: Encodes setting, emotional tone, explicit acts that keywords miss → "Kitchen Argument in Morning" vs generic "Argument"

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
- **Anti-Hallucination Constraints**: Hard rules for known hallucination patterns identified through empirical testing:
  - **"Dinner Date" / "Invitation"**: Do NOT use unless snippets/keywords explicitly mention asking/inviting
  - **"Repair"**: Do NOT use unless keywords/snippets include mechanical terms like "fix", "mechanic", "repair", "tools"
  - **"Heartbreak" / "Breakup"**: Do NOT use unless emotional pain in relationship ending is clearly described
  - Hard constraints work better than soft guidance for creative models, with explicit conditions preventing over-correction

#### Model Evaluation Criteria

Labels are evaluated on research reliability criteria:

1. **Label Quality**:
   - **Specificity**: Labels include concrete details (location, body part, specific act, object) and distinguish topics clearly
   - **Genre Awareness**: Recognizes romance/erotic fiction conventions and distinguishes romantic, erotic, and domestic/emotional content
   - **Discriminative Power**: Different topics receive clearly distinguishable labels (no more than 10% duplicates unless truly identical)

2. **Scene Summary Quality** (when using improved prompts):
   - **Micro-Scene Focus**: Describes specific moments/scenes, not plot arcs
   - **Concrete Details**: Includes at least one concrete detail (location, object, body part, specific action)
   - **Neutral Tone**: Third-person, analytical language suitable for research

3. **Categories & Noise Detection**:
   - **Category Consistency**: Primary/secondary categories accurately reflect topic themes
   - **Noise Detection Accuracy**: Correctly identifies incoherent topics without over-flagging real ones

4. **Stability & Format Compliance**:
   - **JSON Schema Compliance**: Valid, parseable JSON with all required fields
   - **Tone Stability**: Consistent neutral, analytical tone without RP or chatty style drift
   - **Consistency**: Similar topics receive consistent labeling patterns

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

#### Computational Tools and Strategies

**Infrastructure: OpenRouter API**
- **Why OpenRouter**: Single API key for multiple models, no local infrastructure required, access to specialized models (e.g., Celeste, Gutenberg), cost-effective (~$0.017 per 368 topics), OpenAI-compatible API
- **Rate Limiting**: Conservative 4.0 second delay between API calls, robust retry logic with exponential backoff
- **Model Parameters**:
  - **Temperature**: 0.35 (balanced for consistency + natural phrasing). Too low (0.0-0.2) is overly deterministic; too high (0.7-1.0) causes excessive variation
  - **Max Tokens**: 40 for labeling, 220 for taxonomy/Radway mapping
  - **Sampling**: Deterministic for taxonomy/Radway mapping to ensure reproducibility

**Processing Strategies**:
- **Streaming Mode**: Process topics incrementally, write to disk as generated (memory-efficient, fault-tolerant, progress visibility)
- **Caching and Resumption**: Load existing labels, skip processed topics, only process new/updated topics (enables incremental updates, cost savings)
- **Snippet Reranking**: Maximal Marginal Relevance (MMR) for diverse, informative snippets when many representative documents available

**Integration with BERTopic**:
- **Model Loading**: Supports both pickle format (wrapped `RetrainableBERTopicModel`) and native BERTopic safetensors format
- **Metadata Storage**: All topic metadata (labels, taxonomy mappings, Radway functions) stored in BERTopic's `topic_metadata_` attribute, creating a single source of truth
- **Topic Assignment**: Labeled and categorized topics used for sentence-level assignment, book-level aggregation, and category-level aggregation for statistical analysis

#### Technical Specifications

- **Default Parameters**: 15 keywords per topic, 40 max tokens per label, temperature=0.35 (balanced for consistency + natural phrasing)
- **Output Format**: JSON file with structured fields: `{"topic_id": {"label": "...", "scene_summary": "...", "primary_categories": [...], "secondary_categories": [...], "is_noise": false, "rationale": "...", "keywords": [...]}}`
- **Processing Mode**: Streaming support for large topic sets (memory-efficient, fault-tolerant)
- **Token Cost**: ~1055 tokens per topic (system prompt ~800, keywords ~50, POS cues ~30, snippets ~75, overhead ~100). Snippets add ~75 tokens (7.6% increase) for significant quality improvement

#### Quality Assurance and Validation

**Model Comparison Results** (evaluated on 30 topics):
- **mistralai/Mistral-Nemo-Instruct-2407**: 100% success rate, 2.30 avg words per label, 0% keyword copying ✅ **BEST**
- **mistralai/mistral-7b-instruct:free**: 93.3% success rate, 2.80 avg words, 6.7% keyword copying ✅ **EXCELLENT**
- **venice/uncensored:free**: 76.7% success rate, 2.73 avg words, 23.3% keyword copying ✅ **GOOD**
- **x-ai/grok-4.1-fast**: 0% success rate, 1.00 avg words, 100% keyword copying ❌ **FAILED**
- **deepseek/deepseek-chat-v3-0324**: 0% success rate, 1.00 avg words, 100% keyword copying ❌ **FAILED**

**Coverage Metrics**:
- **Taxonomy Coverage**: 361 out of 368 topics (98.1%) successfully mapped to taxonomy categories
- **Radway Coverage**: Varies by topic type; topics in "Relationship Trajectory (Main Couple)" group should have near-100% coverage (not "none")

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
- **Taxonomy Structure**: **Romance Corpus Topic Taxonomy** with 8 main groups and 30+ categories:
  1. **Embodied & Sensory Experience** (3 categories): Body parts, pain/injury, physical activity
  2. **Sexuality, Attraction & Intimacy** (4 categories): Attraction, kissing, explicit sexual acts, aftercare
  3. **Emotions, Cognition & Inner Life** (4 categories): Positive emotions, negative emotions, ambivalence, moral reflection
  4. **Relationship Trajectory (Main Couple)** (5 categories): Meeting, bonding, secrets, conflict, reconciliation
  5. **Social World Outside Couple** (3 categories): Family, friends, community
  6. **Work, Wealth, Status & Institutions** (5 categories): Hero's work, heroine's work, shared workplaces, money/housing, formal institutions
  7. **Conflict, Risk & Harm** (3 categories): Interpersonal conflict, violence/coercion, external crises
  8. **Spaces, Time, Activities & Objects** (4 categories): Domestic spaces, public/leisure, objects/technology, temporal framing
- Each category has hierarchical ID (e.g., "4.2" = Relationship Trajectory, Bonding), name, group, and detailed description for LLM classification
- **Special Category**: `noise` for boilerplate/technical artifacts
- **Input**: Topic keywords, LLM-generated labels, scene summaries, primary/secondary categories, optional representative document snippets
- **Classification Task**: Map each topic to main category ID, secondary category ID, other plausible IDs, confidence (low/medium/high), and rationale
- **Output**: JSON with `main_category_id`, `secondary_category_id`, `other_plausible_ids`, `confidence`, `rationale`
- **Coverage**: 361 out of 368 topics (98.1%) successfully mapped to taxonomy categories
- **Statistical Analysis Results**: Kruskal-Wallis tests identified 3 categories with statistically significant differences (p < 0.05) across rating classes:
  - **5.3: Community, Norms & Social Events** (p = 0.029, η² = 0.070 - medium effect)
  - **6.2: Heroine's Work & Professional Identity** (p = 0.047, η² = 0.057 - small-medium effect)
  - **3.4: Beliefs, Values & Moral Reflection** (p = 0.048, η² = 0.048 - small effect)
- **Model Integration**: Taxonomy mappings embedded in BERTopic model's `topic_metadata_` attribute (recommended model: `model_1_with_llm_labels_and_metadata_disambiguated.pkl` with 361 taxonomy mappings, 98.1% coverage)

**Stage 3: Radway Narrative Functions** ✅ **Implemented**
- **Method**: Zero-shot classification to **Radway's 13 narrative functions** (Radway, 1984) using Mistral-Nemo via OpenRouter
- **Theoretical Foundation**: Janice Radway's analysis of romance fiction identifies 13 narrative functions that structure the heroine-hero relationship arc
- **Radway Functions by Phase**:
  - **Phase I: Initial Conflict & Isolation** (R1-R7): R1 (heroine's social identity destroyed), R2 (heroine reacts antagonistically), R3 (hero responds ambiguously), R4 (heroine interprets as purely sexual interest), R5 (heroine responds with anger/coldness), R6 (hero retaliates/punishes), R7 (physical/emotional separation)
  - **Phase II: Turning Point & Recognition** (R8-R10): R8 (hero treats heroine tenderly), R9 (heroine responds warmly), R10 (heroine reinterprets hero's behavior as result of previous hurt)
  - **Phase III: Commitment & Restoration** (R11-R13): R11 (hero declares love and demonstrates commitment), R12 (heroine responds sexually and emotionally), R13 (heroine's social identity restored - HEA)
- **Input**: Uses Stage 2 taxonomy JSON as single source of truth (includes taxonomy mappings, labels, keywords, scene summaries, representative snippets)
- **Classification Task**: Map each topic to radway_main_id (R1-R13 or "none"), radway_secondary_id, radway_other_plausible_ids, radway_phase (I, II, III, or NA), radway_is_none (boolean), radway_confidence (low/medium/high), radway_rationale
- **Disambiguation Rules**: 
  - R4 vs R12: R4 for attraction without explicit acts, R12 for described sex acts
  - R7 (separation) is narrow: only for actual breakup/separation, not arguments
  - Commitment overrides: wedding/marriage/proposal → R11/R13
  - Gated "none" decision: Only for topics clearly about background context (work, wealth, side characters) not the heroine-hero relationship
- **Post-LLM Heuristic Overrides**: Conservative rule-based corrections for systematic errors (explicit sex scenes 2.3 → R12, commitment cues → R11/R13, R7 sanity checks)
- **Output**: Merged JSON preserving all Stage 2 fields plus `radway_functions` object with all Radway mapping fields
- **Classification Results**: Successfully classified 361 topics (98.1% coverage):
  - **272 topics** mapped to specific Radway functions (R1-R13)
  - **96 topics** classified as "none" (background/contextual content)
  - **Distribution by Phase**: Phase I (147 topics, 54.0%), Phase II (96 topics, 35.3%), Phase III (28 topics, 10.3%)
  - **All 13 Radway functions** represented in the classification
  - **130 topics** classified with high confidence (36% of classified topics)
- **Key Finding**: Phase I (conflict and isolation) dominates the narrative function distribution, representing over half of all function-mapped topics, suggesting conflict and tension are central to romance narrative structure

#### Target: Theory-Aligned Composite Categories (A-S)

Build the **theory-aligned composite categories** from Stage 2 taxonomy nodes and Stage 3 Radway functions. These composites operationalize the research framework:

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
- **`book_category_proportions.parquet`**: Book-level category proportions aggregated from sentence-level topic assignments
- **BERTopic models**: `model_1_with_llm_labels_and_metadata_disambiguated.pkl` (recommended, with 361 taxonomy mappings embedded in `topic_metadata_`)

**Stage 3 Output**:
- **`taxonomy_with_radway.json`**: Merged JSON with taxonomy + Radway function mappings
- **BERTopic models**: `model_1_with_radway_mappings` (Radway functions attached)
- **EDA files**: Distribution analysis by taxonomy groups, narrative phases, and high-confidence classifications

**Statistical Analysis Outputs** (Stage 2):
- **Kruskal-Wallis test results**: Statistical significance and effect sizes (η²) for all taxonomy categories
- **Pairwise comparison results**: Post-hoc tests identifying which rating classes differ for significant categories
- **Visualization outputs**: Volcano plots, effect size charts, p-value heatmaps, enhanced violin plots, pairwise comparison plots

**Target Output** (to be implemented):
- **`topic_to_category_probs.json`**: Per-topic soft composite category assignments (A-S)
- **`topic_to_category_final.csv`**: Flat table format for inspection
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

#### Data Preparation Pipeline

The Stage 10 pipeline consists of four sequential scripts plus analysis notebooks that transform sentence-level topic assignments into book-level and segment-level features for statistical analysis:

**Script 03: Generate Topic Probabilities** (`03_generate_topic_probabilities_final.py`)
- Generates normalized topic probabilities at book and chapter levels from sentence-level BERTopic assignments
- **Key Features**: Goodreads-first book IDs, robust ID normalization, cohort exclusion (5 books excluded: 19561986, 19619918, 25781538, 52061964, 53491034), caching (~2 hours saved), NaN replacement (critical fix)
- **Outputs**: 
  - `book_topic_probs.parquet`: (book_id, topic_id, prob) - 33,856 rows for 92 books × 368 topics
  - `chapter_topic_probs.parquet`: (book_id, chapter_id, topic_id, prob) - 1,089,280 rows for 2,960 chapters × 368 topics
- **Output Location**: `results/stage10_correlation_analysis/00_data_preparation/topic_probabilities/`
- **Validation**: 0% NaN values (NaN replaced with 0.0 before aggregation), probabilities sum to ~1.0 per book/chapter (min: 0.999, max: 1.000)

**Script 04: Generate Tertile Topic Probabilities** (`04_generate_tertile_topic_probs_patched_v3.py`)
- Generates topic probabilities for begin/middle/end tertiles of each book, enabling narrative arc analysis
- **Key Features**: Tertile splitting, chunking (40 sentences per chunk), weighted aggregation, book ordering preservation
- **Outputs**: `tertile_topic_probs.parquet`: (book_id, segment, topic_id, prob) - ~101,568 rows for 92 books × 3 segments × 368 topics
- **Output Location**: `results/stage10_correlation_analysis/00_data_preparation/topic_probabilities/`

**Script 01: Data Validation & Extraction** (`01_data_validation_extraction.py`)
- Entry point for Stage 10 analysis. Loads final BERTopic model, merges Stage 08 label metadata, exports topic-level lookup table
- **Key Features**: Model loading with taxonomy & Radway mappings, label merging, QA checks, ID alignment diagnostics, fallback CSV support
- **Outputs**: 
  - `topic_lookup.parquet`: (369, 21) - one row per topic (368 topics + noise topic) with labels, keywords, taxonomy/Radway mappings
  - `full_model_data.csv` / `.parquet`: Full topic metadata in CSV/Parquet format (fallback when model unavailable)
  - `summary_statistics.json`: QA summary (topic counts, mapping coverage, keyword quality)
  - `topics_needs_review.csv`: Topics requiring manual review (missing mappings, poor keywords)
  - Diagnostic reports: `id_alignment_report.csv`, `missing_books_in_outputs.csv`
- **Output Location**: `results/stage10_correlation_analysis/00_data_preparation/taxonomy_radway_eda/`

**Script 02: Book Aggregation** (`02_book_aggregation.py`)
- Joins topic-level lookup to book topic mixture data to produce book-level taxonomy proportions and derived indices
- **Key Features**: Taxonomy aggregation, multiple formats (long/wide), segment-level support, derived indices computation, ID normalization
- **Outputs**: 
  - `book_taxonomy_main_props_long.parquet`: ~2,484 rows (92 books × 27 categories) in long format
  - `book_taxonomy_main_props_wide.parquet`: (92, 27+ categories) - one row per book, columns are taxonomy categories
  - `indices_book_taxonomy_proxy.parquet`: Derived indices aligned to research hypotheses (love_over_sex, hea_index, explicitness_ratio, dark_vs_tender, miscommunication_balance, luxury_saturation_proxy)
  - `segment_taxonomy_main_props_long.parquet`: Segment-level proportions in long format (if segment data available)
- **Output Location**: `results/stage10_correlation_analysis/00_data_preparation/book_features/`

**Analysis Notebooks** (`notebooks/07_analysis/`):

**01_topic_analysis** (`01_topic_analysis_v2_contract_normalized.ipynb`):
- Individual topic distributions across Top/Middle/Trash tiers
- Topic-level leaderboards, effect sizes (Cliff's Delta), FDR-corrected tests
- Two-gate filtering rule: effect size |Cliff's δ| ≥ 0.20 AND (mass ≥ 0.002 OR |mean diff| ≥ 0.001)
- **Results**: 85 discriminative topics identified from 342 analyzed topics (368 total, excluding noise/outlier)
  - **Tier 1 (High Confidence)**: 8 topics (7 Top-associated, 1 Trash-associated) with |δ| ≥ 0.35 AND raw p < 0.05
  - **Tier 2 (Exploratory)**: 85 topics (70 Top-associated, 15 Trash-associated) with |δ| ≥ 0.20
- **Top-tier differentiation**: Psychological credibility scenes (fear admissions, emotional delusion, bluffing about feelings) and embodied intimacy cues (affectionate stares, lip biting, shared joy)
- **Trash-tier differentiation**: Explicit sexual content (dominatrix sessions, explicit erotics) and procedural/transition scenes (doors, phones, desk work)
- **Author dominance**: 30 topics show high author dominance (>50% from single author), requiring control in modeling
- **Outputs**: `results/stage10_correlation_analysis/01_topic_analysis/`

**02_taxonomy_group_analysis** (`02_taxonomy_group_analysis_v2_contract_normalized.ipynb`):
- Taxonomy group-level distribution comparisons
- Dual normalization (absolute vs conditional shares) to handle OTHER bucket variation
- Kruskal-Wallis tests with Holm correction, Epsilon-squared effect sizes
- **Outputs**: `results/stage10_correlation_analysis/02_taxonomy_group_analysis/`

**03_composite_index_construction** (`05_build_theory_aligned_composites_indices_v5_6_measurement_pipeline.ipynb`):
- Theory-aligned composite indices (A-S) construction with measurement pipeline v5.6
- Reliability diagnostics: Cronbach's alpha, McDonald's omega, PCA (PC1/PC2), stability metrics (leave-one-out, split-half, bootstrap-to-full)
- Composite classification: ATOMIC vs COMPOSITE, CORE vs EXPLORATORY, UNIDIMENSIONAL vs MULTIDIMENSIONAL
- Book-level and segment-level indices (raw and z-scored, sum and max aggregation)
- Arc contrasts: end−begin, middle−begin deltas
- **Outputs**: `results/measurement_v5/` (bundle exports, audit diagnostics)

**04_hypothesis_testing** (`04_hypothesis_testing_inference_only_v4_2_macro_axes_tight.ipynb`):
- Hypothesis testing (H1-H6) using composite indices
- Macro-axes analysis (5-axis model: status/dominance, payoff/safety, drama/obstacle, explicitness, negative affect)
- Bootstrap inference (800 iterations) with 95% CI and P(β>0)
- Arc trajectory tests using exported deltas (end−begin, middle−begin)
- Cross-validation performance (20 repeats of 5-fold CV)
- **Outputs**: `results/measurement_v5/bundle/inference_outputs/`

#### Hypothesis Testing Results

**Sample Size**: N = 92 books

**Two-Channel Analysis**: The analysis separates two distinct Goodreads success signals:
1. **Mass Appeal / Visibility** = `log_rating_count` (how many people rated it)
2. **Perceived Quality** = `rating_mean` and `avg_rating_bayes` (how positively readers evaluate it)

**Key Finding**: Popularity (reach) is strongly associated with a "billionaire-romance package": status/luxury + alpha guarding + repair + emotional safety + social/kin network. Perceived quality (ratings), after accounting for popularity, is most consistently associated with "care + safety" and is negatively associated with "baseline negative affect" and explicit erotics.

**Mass Appeal Predictors** (`log_rating_count`):
- **Top predictors** (β with 95% CI, P(β>0)):
  1. **R2_alpha_guarding**: β≈ +0.44, CI [+0.23, +0.60], P=1.00
  2. **D_power_wealth_luxury__pc1**: β≈ +0.37, CI [+0.20, +0.55], P=1.00
  3. **A2_emotional_safety__pc1**: β≈ +0.32, CI [+0.13, +0.50], P=0.998
  4. **Q_repair**: β≈ +0.23, CI [+0.02, +0.41], P=0.985
  5. **J_social_support_kin**: β≈ +0.19, P≈0.95
- **Macro axes** (5-axis model):
  - **AX_status_dominance**: β≈ +0.46, CI [+0.28, +0.61], P=1.00
  - **AX_payoff_safety**: β≈ +0.33, CI [+0.14, +0.50], P≈0.999
  - **AX_drama_obstacle**: β≈ +0.34, CI [+0.07, +0.57], P≈0.993
  - **AX_explicitness**: β≈ −0.27, CI [−0.45, −0.06], P≈0.003 (strongly negative)

**Perceived Quality Predictors** (`rating_mean`, controlling for `log_rating_count`):
- **Top predictors**:
  1. **R1_protective_caretaking**: β≈ +0.22, CI [+0.05, +0.36], P=0.995
  2. **A2_emotional_safety__pc1**: β≈ +0.15, P=0.95
- **Macro axes**:
  - **AX_payoff_safety**: β≈ +0.21, P=0.974
  - **AX_explicitness**: β≈ −0.15, P=0.095 (tends negative)
  - **AX_negative_affect**: β≈ −0.13, P=0.051 (borderline negative)
- **Partial correlations** (quality beyond popularity):
  - **R1_protective_caretaking**: +0.245
  - **A2_emotional_safety__pc1**: +0.152
  - **C_explicit_eroticism**: -0.172 (negative)
  - **F2_anger_frustration**: -0.121 (negative)

**Narrative Arc / Pacing Results**:
- **Higher-rated books show**:
  - **F2_anger_frustration end−begin**: β≈ +0.24, CI [+0.08, +0.41], P=0.995
  - **F3_anxiety_worry end−begin**: β≈ +0.19, CI [+0.02, +0.36], P=0.981
- **Interpretation**: Higher-rated books have **better pacing**: lower baseline negativity across the book, but stronger late "crisis escalation" (third-act crisis), consistent with romance narrative structure.

**Topic-Level Analysis Results**:
- **Sample**: 92 books (30 top, 32 middle, 30 trash) × 342 analyzed topics (368 total, excluding noise/outlier)
- **Discriminative topics**: 85 topics identified via two-gate filtering (effect size |Cliff's δ| ≥ 0.20 AND meaningful impact)
- **Top-associated topics** (70 topics): Emphasize psychological credibility (fear admissions, emotional delusion, identity affirmation) and embodied intimacy cues (affectionate stares, lip biting, shared joy). Top Tier 1 examples: "Married Couple's Affectionate Stares" (δ = 0.453), "Frightened Admissions" (δ = 0.420), "Emotional Relationship Delusion" (δ = 0.404)
- **Trash-associated topics** (15 topics): Emphasize explicit sexual content ("Dominatrix Session", δ = -0.353) and procedural/transition scenes ("Work At Desk", "Exiting Through Doorways", "Phone Ringing And Answering")
- **Author dominance**: 30 topics show high author dominance (>50% from single author), requiring control in modeling
- **Topic health**: Median prevalence = 0.924 (most topics appear in most books), median mass = 0.0020, median concentration ratio = 2.68

**Tier Differences** (Top/Middle/Trash):
- **Top tier** (n=30): avg_rating ≈ 4.22, n_ratings ≈ 116k (higher quality perception + much higher visibility)
- **Middle tier** (n=32): avg_rating ≈ 4.01, n_ratings ≈ 44k (moderate quality + moderate visibility)
- **Trash tier** (n=30): avg_rating ≈ 3.77, n_ratings ≈ 48k (lower quality perception + lower visibility)

**Predictive Performance** (20 repeats of 5-fold CV):
- **rating_mean**: CV R² = 0.056 ± 0.041 (themes) vs 0.108 ± 0.031 (metadata only)
- **log_rating_count**: CV R² = 0.050 ± 0.037 (themes only)
- **Conclusion**: Themes explain popularity better than star ratings (at N=92). Star ratings likely influenced by factors beyond theme indices (prose quality, pacing, editing, reader expectations, etc.).

**Meta-Result**: The theme system is better at explaining market reach than "star rating," suggesting that market reach is more systematically related to thematic content, while star ratings may be influenced by factors beyond theme indices.

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

- Bamman, D., Underwood, T., & Smith, N. A. (2013). A Bayesian Mixed Effects Model of Literary Character. *Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics*.

- Jockers, M. L. (2013). *Macroanalysis: Digital Methods and Literary History*. University of Illinois Press.

---

For technical implementation details, see [docs/METHODOLOGY.md](docs/METHODOLOGY.md).  
For data contract specifications, see [docs/DATA_CONTRACTS.md](docs/DATA_CONTRACTS.md).  
For index definitions, see [docs/INDICES.md](docs/INDICES.md).

