# Scientific Methodology: Romance Novels — Themes × Reader Appreciation

**A Mixed-Methods Computational Analysis**

This document provides an overview of the research methodology and findings. For implementation details, see the [`reports/`](reports/) directory and stage-specific documentation in [`src/`](src/).

---

## Research Objectives

1. **Map topic-model outputs** from modern romance novels to theory-driven themes and test which themes differentiate higher-rated from lower-rated books.

2. **Build explainable indices** to quantify narrative qualities readers value.

3. **Validate findings** against Goodreads metadata (ratings and rating counts).

---

## Research Questions

1. Which theme categories are most prevalent in higher-rated vs lower-rated novels?
2. Does love/commitment/tenderness outweigh explicit sexual content in higher-rated books?
3. Is luxury appealing only when paired with commitment/tenderness?
4. Do protectiveness/care signals predict appreciation better than jealousy/possessiveness?
5. Do miscommunication/negative affect diminish across the book while HEA/repair rises?

---

## Research Hypotheses

### H1: Love-over-Sex
> (commitment + tenderness) > explicit sexual content in higher-rated books

Higher-rated novels emphasize emotional connection over explicit content.

### H2: HEA Index
> HEA indicators higher in top-rated books

Novels with stronger Happily Ever After signals (commitment, symbolic gifts, rituals) are more appreciated.

### H3: Luxury × Love Interaction
> Luxury predicts ratings only when combined with emotional depth

Wealth/luxury settings appeal to readers only when paired with commitment and tenderness.

### H4: Protectiveness vs Possessiveness
> protectiveness − jealousy is higher in top-rated books

Caring protectiveness is valued more than jealous possessiveness.

### H5: Darkness vs Tenderness
> (negative affect + threat/violence) − tenderness is lower in top-rated books

Top-rated novels favor tenderness over dark themes.

### H6: Narrative Arc
> begin→end: miscommunication ↓, negative affect ↓; commitment ↑, repair ↑

Successful romances show progression from conflict to resolution.

---

## Dataset

### Corpus

- **105 standalone billionaire romance novels** by 35 authors
- Selected from curated "Best Billionaire Romance" lists
- Each novel ≥100,000 words
- **680,822 sentences** organized: Author → Book → Chapter → Sentence

### Goodreads Metadata

- **92 books** in final analysis (after quality filtering)
- Rating distribution: Mean 3.99, Range 3.26–4.42
- All books have ≥100 ratings (mean: 65,849)

### Tier Distribution

| Tier | N | Avg Rating | Avg Ratings Count |
|------|---|------------|-------------------|
| Top | 30 | 4.22 | ~116k |
| Middle | 32 | 4.01 | ~44k |
| Trash | 30 | 3.77 | ~48k |

---

## Methodology Overview

### 1. Topic Modeling: BERTopic + OCTIS

**Why BERTopic?** Unlike traditional LDA, BERTopic uses BERT embeddings to capture contextual word meanings, producing more interpretable topics for literary analysis.

**Optimization**: Bayesian hyperparameter optimization via OCTIS framework across:
- 6 embedding models (SentenceTransformers)
- UMAP, HDBSCAN, and vectorizer parameters
- 300+ configurations evaluated

**Model Selection**: Pareto efficiency analysis balancing coherence (topic interpretability) and diversity (topic variety). Final model: 368 topics.

**Character Name Exclusion**: 4,444 character names added to stopwords to ensure topics reflect thematic content rather than character co-occurrence.

### 2. LLM-Based Topic Labeling

**Challenge**: Topic models produce keyword lists requiring human interpretation. Manual labeling is impractical at scale (368 topics).

**Solution**: Zero-shot labeling via Mistral-Nemo-Instruct through OpenRouter API.

**Key Design Decisions**:
- **Representative snippets**: Actual document excerpts provide scene-level context beyond keywords
- **Romance-aware prompts**: Domain-specific instructions for accurate labeling of romantic/erotic content
- **Anti-hallucination constraints**: Hard rules preventing common LLM inference errors

**Result**: 98.1% of topics successfully labeled (361/368).

### 3. Theory-Aligned Category Mapping

Topics are mapped to two theoretical frameworks via zero-shot classification:

**Romance Corpus Taxonomy** (8 groups, 30+ categories):
1. Embodied & Sensory Experience
2. Sexuality, Attraction & Intimacy
3. Emotions, Cognition & Inner Life
4. Relationship Trajectory (Main Couple)
5. Social World Outside Couple
6. Work, Wealth, Status & Institutions
7. Conflict, Risk & Harm
8. Spaces, Time, Activities & Objects

**Radway's 13 Narrative Functions** (Radway, 1984):
- Phase I (R1–R7): Initial Conflict & Isolation
- Phase II (R8–R10): Turning Point & Recognition
- Phase III (R11–R13): Commitment & Restoration

**Coverage**: 272 topics mapped to Radway functions; 96 classified as background/contextual.

### 4. Statistical Analysis

**Two-Channel Approach**: Separates analysis of:
- **Mass Appeal** (log rating count): What makes books popular/visible?
- **Perceived Quality** (rating mean): What makes readers rate books highly?

**Methods**:
- Bootstrap inference (800 iterations, 95% CI)
- Cross-validation (20 × 5-fold CV)
- Effect sizes (Cliff's δ) for tier comparisons
- Narrative arc analysis via tertile comparisons (begin/middle/end)

---

## Pipeline Overview

| Stage | Description |
|-------|-------------|
| 01 Ingestion | Load novels and Goodreads metadata |
| 02 Preprocessing | Text cleaning, sentence segmentation, character name removal |
| 03 Modeling | BERTopic training with OCTIS optimization |
| 04 Selection | Pareto-efficient model selection |
| 05 Retraining | Retrain selected models |
| 06 Topic Exploration | Multi-representation analysis |
| 07 Topic Quality | Noisy topic detection |
| 08 LLM Labeling | Automated topic labeling |
| 09 Category Mapping | Theory-aligned classification |
| 10 Correlation Analysis | Statistical hypothesis testing |

See [`reports/01_stage_reports/`](reports/01_stage_reports/) for detailed methodology per stage.

---

## Derived Indices

Composite indices operationalize the research hypotheses:

### Love-over-Sex
```
(commitment_hea + tenderness) − explicit
```

### HEA Index
```
commitment_hea + symbolic_gifts + festive_rituals
```

### Explicitness Ratio
```
explicit / (explicit + commitment + tenderness)
```

### Luxury Saturation
```
luxury_wealth + luxury_settings + luxury_consumption + nightlife_glamour
```

### Dark-vs-Tender
```
(negative_affect + threat_violence) − tenderness
```

### Miscommunication Balance
```
(commitment + tenderness + repair) − miscommunication
```

### Protective–Jealousy Delta
```
protectiveness_care − jealousy_possessiveness
```

---

## Key Findings

### Mass Appeal (Popularity)

Books with higher rating counts emphasize:
- **Status/dominance themes** (wealth, power, alpha behavior)
- **Emotional safety** (protective care, repair after conflict)
- **Social support** (family, friends, community)

Books with lower rating counts emphasize:
- **Explicit sexual content** (negative association with popularity)

### Perceived Quality (Ratings)

After controlling for popularity, higher-rated books show:
- **More protective caretaking** (positive)
- **More emotional safety** (positive)
- **Less explicit erotica** (negative)
- **Less baseline negative affect** (negative)

### Narrative Arc

Higher-rated books demonstrate better pacing:
- Lower baseline negativity throughout
- Stronger late-story "crisis escalation" (third-act tension)
- Anger/frustration increases toward ending (then resolves)

### Topic-Level Patterns

- **85 discriminative topics** identified (of 342 analyzed)
- **Top-associated themes**: Psychological credibility (fear admissions, emotional delusion), embodied intimacy (affectionate stares, lip biting)
- **Trash-associated themes**: Explicit sexual content, procedural/transition scenes

### Meta-Finding

> Thematic content better explains **popularity** (market reach) than **star ratings** (reader evaluation).

Star ratings likely influenced by factors beyond theme indices (prose quality, pacing, editing, reader expectations).

See [`reports/02_findings/hypothesis_testing/`](reports/02_findings/hypothesis_testing/) for detailed statistical results.

---

## Theoretical Framework

This research draws on:

- **Radway (1984)**: Narrative function analysis of romance fiction
- **Propp**: Narrative functions and story structure
- **Ogas & Gaddam (2011)**: Reader psychology and genre preferences

The category mapping operationalizes these theoretical constructs for quantitative analysis.

---

## References

Bamman, D., Underwood, T., & Smith, N. A. (2013). A Bayesian Mixed Effects Model of Literary Character. *Proceedings of ACL*.

Egger, R., & Yu, J. (2022). A topic modeling comparison between LDA, NMF, Top2Vec, and BERTopic. *Frontiers in Sociology*, 7, 886498.

Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv:2203.05794*.

Jiang, A. Q., et al. (2023). Mistral 7B. *arXiv:2310.06825*.

Jockers, M. L. (2013). *Macroanalysis: Digital Methods and Literary History*. University of Illinois Press.

Radway, J. A. (1984). *Reading the Romance: Women, Patriarchy, and Popular Literature*. University of North Carolina Press.

Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. *WSDM*.

Terragni, S., et al. (2021). OCTIS: Comparing and optimizing topic models is simple! *EACL*.

---

## Further Reading

| Topic | Location |
|-------|----------|
| Stage methodology | [`reports/01_stage_reports/`](reports/01_stage_reports/) |
| Hypothesis testing results | [`reports/02_findings/hypothesis_testing/`](reports/02_findings/hypothesis_testing/) |
| LLM labeling methodology | [`reports/02_findings/methodology_llm_labeling_and_taxonomy/`](reports/02_findings/methodology_llm_labeling_and_taxonomy/) |
| Implementation details | [`src/`](src/) stage READMEs |
