# LLM-Based Topic Labeling and Taxonomy Mapping: Presentation Summary

**Date:** December 2024  
**Purpose:** Essential information for presenting Stage 08 (LLM Labeling) and Stage 09 (Category Mapping) methodology and results

---

## Slide 1: Title Slide

**Title:** Automating Topic Interpretation: LLM-Based Labeling and Taxonomy Mapping for Computational Literary Analysis

**Subtitle:** From Keywords to Categories: A Three-Stage Pipeline for Romance Fiction Analysis

---

## Slide 2: The Challenge

### Problem Statement

**BERTopic Output:**
- 368 topics identified
- Each topic = list of keywords (e.g., "mouth, tongue, suck, lips, breath")
- **No interpretable labels** - requires human interpretation
- **No theoretical framework** - topics not connected to literary theory

**Manual Labeling:**
- Impractical for 368 topics
- Inconsistent across researchers
- Time-consuming and expensive

**Solution:** Automated LLM-based labeling + zero-shot classification

---

## Slide 3: Three-Stage Pipeline Overview

```
Stage 08: LLM Labeling
  ↓
  Input: BERTopic topics (keywords + snippets)
  Output: Descriptive labels (2-6 words) + metadata
  Coverage: 368/368 topics (100%)
  
Stage 09 Stage 2: Taxonomy Mapping
  ↓
  Input: LLM labels
  Output: Theory-driven category mappings (6 groups, 20 categories)
  Coverage: 361/368 topics (98.1%)
  
Stage 09 Stage 3: Radway Functions
  ↓
  Input: Taxonomy mappings
  Output: Radway narrative function mappings (R1-R13, 3 phases)
  Coverage: 361/368 topics (98.1%)
```

**Key Achievement:** Fully automated pipeline from raw topics to theoretical categories

---

## Slide 4: Stage 08 - LLM-Based Labeling

### What We Did

**Input:**
- Topic keywords (POS-filtered: nouns, verbs, adjectives)
- Representative document snippets (6 per topic, ~200 chars each)

**Process:**
- Romance-aware prompt design
- Anti-hallucination constraints
- JSON-structured output

**Output:**
- Descriptive labels (2-6 words)
- Scene summaries (12-25 words)
- Categories (primary/secondary)
- Noise detection flags

### Example

**Keywords:** "car, seat, door, parked, kiss, hand"

**Snippets:**
1. "They sat in the parked car, his hand on her thigh as the windows fogged."
2. "She leaned across the seat, kissing him while the engine idled quietly."

**Label:** "Makeout In Parked Car"

---

## Slide 5: Model Selection

### Why Mistral-Nemo-Instruct-2407?

**Tested 6 models on 5-topic subset:**

| Model | Success Rate | Avg Words | Keyword Copy | Status |
|-------|--------------|-----------|--------------|--------|
| **Mistral-Nemo-Instruct** | **100%** | **2.30** | **0%** | ✅ **BEST** |
| Mistral-7B-Instruct | 93.3% | 2.80 | 6.7% | ✅ Good |
| Cydonia-24B | High | 2.50 | 0% | ✅ Literary |
| Grok-4.1-Fast | 0% | 1.00 | 100% | ❌ Failed |
| DeepSeek-Chat | 0% | 1.00 | 100% | ❌ Failed |

**Key Finding:** Instruction-tuned models outperform creative models for structured research tasks

**Selection Criteria:**
- ✅ Instruction following
- ✅ Literary analysis capability
- ✅ Research reliability (low hallucination)
- ✅ Cost-effectiveness (~$0.017 per 368 topics)

---

## Slide 6: Prompt Design Innovation

### Romance-Aware Prompt Architecture

**Key Components:**

1. **Domain Context**: "You are a topic-labeling assistant for modern romantic and erotic fiction"

2. **Format Rules**: "Output exactly ONE short noun phrase of 2-6 words"

3. **Priority Hierarchy**: Action → Role → Setting → Tone

4. **Snippet Integration**: "When snippets and keywords disagree, trust the snippets"

5. **Anti-Hallucination Constraints**: Hard rules based on empirical testing
   - "Do NOT use 'dinner date' unless snippets explicitly mention asking/inviting"
   - "Do NOT use 'repair' unless keywords include mechanical terms"

### Why Snippets Matter

**Without Snippets:**
- Keywords: "mouth, tongue, suck, lips"
- Label: "Erotic Intimacy" (too generic)

**With Snippets:**
- Snippets show: kneeling, taking into mouth, head movement
- Label: "Blowjob in Bed" (specific and accurate)

**Token Cost:** +75 tokens per topic (7.6% increase) for significant quality improvement

---

## Slide 7: Stage 09 Stage 2 - Taxonomy Mapping

### Romance Corpus Topic Taxonomy

**6 Main Groups, 20 Categories:**

1. **Embodied & Sensory Experience** (3 categories)
2. **Sexuality, Attraction & Intimacy** (4 categories)
3. **Emotions, Cognition & Inner Life** (4 categories)
4. **Relationship Trajectory (Main Couple)** (5 categories) ← Largest (158 topics)
5. **Social World Outside Couple** (3 categories)
6. **Luxury Lifestyle & Material World** (1 category)

**Coverage:** 361/368 topics (98.1%)

**Method:** Zero-shot classification using LLM-generated labels + scene summaries

**Example Mapping:**
- Topic: "Makeout In Parked Car"
- Main Category: 4.2 (Bonding, Everyday Intimacy & Growth)
- Secondary Category: 2.1 (Kissing & Non-Explicit Affection)

---

## Slide 8: Stage 09 Stage 3 - Radway Functions

### Radway's 13 Narrative Functions

**Phase I: Initial Conflict & Isolation** (R1-R7)
- R1: Heroine's identity destroyed
- R2: Heroine reacts antagonistically
- R4: Heroine interprets hero as purely sexual
- R7: Physical/emotional separation

**Phase II: Turning Point & Recognition** (R8-R10)
- R8: Hero treats heroine tenderly
- R9: Heroine responds warmly
- R10: Heroine reinterprets hero's behavior

**Phase III: Commitment & Restoration** (R11-R13)
- R11: Hero declares love
- R12: Heroine responds sexually/emotionally
- R13: Heroine's identity restored (HEA)

**Coverage:** 361/368 topics (98.1%)

**Method:** Zero-shot classification with heuristic overrides for systematic errors

---

## Slide 9: Key Results - Distribution

### Taxonomy Group Distribution

![Taxonomy Group Distribution](figures/taxonomy_group_distribution.png)

**Largest Groups:**
- Relationship Trajectory (Main Couple): 158 topics (43.8%)
- Emotions, Cognition & Inner Life: 59 topics (16.3%)
- Sexuality, Attraction & Intimacy: 55 topics (15.2%)

### Radway Phase Distribution

![Radway Phase Distribution](figures/radway_phase_distribution.png)

**Distribution:**
- Phase I (Conflict & Isolation): 147 topics (54.0%)
- Phase II (Turning Point): 96 topics (35.3%)
- Phase III (Commitment): 28 topics (10.3%)
- "None" (Not narrative functions): 96 topics (26.6%)

**Key Finding:** Conflict and tension occupy more textual space than resolution

---

## Slide 10: Key Results - Mapping Patterns

### Taxonomy-to-Radway Mapping

![Taxonomy-Radway Heatmap](figures/taxonomy_radway_heatmap.png)

**Strong Mappings:**
- Explicit Sexual Acts (2.3) → R12 (Phase III)
- Reconciliation & HEA (4.5) → R11/R13 (Phase III)
- Conflict & Breakup (4.4) → R2/R7 (Phase I)
- Bonding & Intimacy (4.2) → R8/R9 (Phase II)

**Weak/No Mappings:**
- Social World (5.x) → Mostly "none"
- Work & Wealth (6.x) → Mostly "none"
- Spaces & Time (8.x) → Mostly "none"

**Interpretation:** Background/contextual content doesn't map to narrative functions (as expected)

---

## Slide 11: Quality Assurance

### Coverage Metrics

| Metric | Coverage | Percentage |
|--------|----------|------------|
| **Labels** | 368/368 | 100% |
| **Taxonomy** | 361/368 | 98.1% |
| **Radway** | 361/368 | 98.1% |
| **Radway Functions** | 272/361 | 75.3% |

### Classification Confidence

- **High confidence**: 130 topics (36% of classified)
- **Medium confidence**: Majority of remaining
- **Low confidence**: Small number of edge cases

### Model Comparison Results

- **Nemo-Instruct**: 100% success rate, 0% keyword copying
- **Failed models**: 0% success rate, 100% keyword copying

**Quality Control:**
- Heuristic overrides for systematic errors
- Deterministic decoding (temperature=0.0) for reproducibility
- Manual review of flagged topics

---

## Slide 12: Computational Implementation

### Infrastructure

**OpenRouter API:**
- Single API key for multiple models
- No local infrastructure (no GPU, no downloads)
- Cost: ~$0.017 per 368 topics
- Rate limiting: 4.0s delay between calls

**Processing:**
- Streaming mode for memory efficiency
- Caching and resumption for fault tolerance
- Snippet reranking (MMR) for diversity

**Integration:**
- All metadata stored in BERTopic model (`topic_metadata_` attribute)
- Single source of truth
- No separate JSON files required

### Model Parameters

- **Temperature**: 0.35 (labeling), 0.0 (taxonomy/Radway)
- **Max tokens**: 40 (labeling), 220 (taxonomy/Radway)
- **Processing time**: ~6-18 minutes for 368 topics

---

## Slide 13: Example Labels

### Label Quality Examples

| Topic Keywords | Label | Category | Radway |
|---------------|-------|----------|--------|
| car, seat, door, parked, kiss | Makeout In Parked Car | 4.2 (Bonding) | R8 (Tenderness) |
| mouth, tongue, suck, lips, clit | Clitoral Stimulation During Foreplay | 2.3 (Explicit Sex) | R12 (Sexual Response) |
| kitchen, angry, voice, raised | Kitchen Argument About Money | 4.4 (Conflict) | R2 (Antagonistic Reaction) |
| wedding, marriage, ceremony, aisle | Wedding Ceremony And Vows | 4.5 (Reconciliation) | R13 (Identity Restored) |

**Key Features:**
- ✅ Specific and concrete (not generic)
- ✅ Scene-level (not abstract)
- ✅ Genre-aware (romance/erotic fiction)
- ✅ Discriminative (distinct labels for similar topics)

---

## Slide 14: Limitations and Future Work

### Current Limitations

1. **Snippet Selection**: Relies on BERTopic's selection, may miss variations
2. **Hallucination**: Despite constraints, models may infer details not present
3. **Coverage**: 98.1% coverage (7 topics unmapped)
4. **Cost**: API costs scale with number of topics
5. **Latency**: Sequential processing (~6-18 minutes)

### Future Improvements

1. **Diversity-Aware Snippet Selection**: MMR or similar for snippet diversity
2. **Adaptive Snippet Count**: More snippets for complex topics
3. **Parallel Processing**: Concurrent topic processing
4. **Dynamic Few-Shot**: Example selection based on topic similarity
5. **Automated Quality Metrics**: Label length, uniqueness, coverage
6. **Model-Specific Tuning**: Optimize prompts for different model families

---

## Slide 15: Impact and Applications

### Research Applications

1. **Narrative Structure Analysis**: Track Radway functions across books
2. **Quality Comparison**: Compare taxonomy distributions between high/low-rated books
3. **Genre Studies**: Analyze romance fiction conventions at scale
4. **Computational Literary Analysis**: Automated interpretation of topic models

### Methodological Contributions

1. **Snippet-Integrated Prompting**: Scene-level understanding for better labels
2. **Anti-Hallucination Constraints**: Empirical approach to reducing model errors
3. **Theory-Driven Classification**: Connecting computational methods to literary theory
4. **Reproducible Pipeline**: Fully automated, documented, and shareable

### Broader Implications

- **Scalability**: Can process hundreds of topics automatically
- **Consistency**: Same criteria applied across all topics
- **Interpretability**: Labels and categories enable human understanding
- **Theory Integration**: Connects computational methods to established frameworks

---

## Slide 16: Summary

### Key Achievements

✅ **100% label coverage** (368/368 topics)  
✅ **98.1% taxonomy coverage** (361/368 topics)  
✅ **98.1% Radway coverage** (361/368 topics)  
✅ **Fully automated pipeline** from keywords to theoretical categories  
✅ **Reproducible methodology** with documented design decisions

### Key Innovations

1. **Romance-aware prompt design** with anti-hallucination constraints
2. **Snippet-integrated prompting** for scene-level understanding
3. **Zero-shot classification** to theory-driven taxonomies
4. **Heuristic overrides** for systematic error correction

### Next Steps

- Statistical analysis comparing taxonomy/Radway distributions across book ratings
- Hypothesis testing: Do well-rated books follow Radway's structure more closely?
- Visualization of narrative arcs across books
- Publication of methodology and results

---

## Slide 17: Questions?

### Contact Information

**Project Repository:** [GitHub link]  
**Documentation:** `reports/01_stage_reports/stage08_llm_labeling/`  
**Code:** `src/stage08_llm_labeling/` and `src/stage09_category_mapping/`

### Key References

- Radway, J. (1984). *Reading the Romance*
- BERTopic Documentation
- OpenRouter API Documentation

---

## Appendix: Figure Placeholders

### Figures to Create

1. **Pipeline Diagram** (`figures/pipeline_diagram.png`)
   - Three-stage flowchart with inputs/outputs

2. **Model Comparison Chart** (`figures/model_comparison_results.png`)
   - Bar chart: success rates, avg words, keyword copying

3. **Taxonomy Group Distribution** (`figures/taxonomy_group_distribution.png`)
   - Bar chart or pie chart: 6 groups with topic counts

4. **Radway Phase Distribution** (`figures/radway_phase_distribution.png`)
   - Bar chart: Phase I, II, III, "none" with counts

5. **Taxonomy-Radway Heatmap** (`figures/taxonomy_radway_heatmap.png`)
   - Heatmap: taxonomy categories × Radway functions

6. **Label Quality Examples Table** (`figures/label_quality_examples.png`)
   - Table: keywords, labels, categories, Radway functions

7. **Prompt Architecture Diagram** (`figures/prompt_architecture.png`)
   - System prompt components and flow

### Figure Directory Structure

```
reports/01_stage_reports/stage08_llm_labeling/
├── stage08_presentation_summary.md (this file)
└── figures/
    ├── pipeline_diagram.png
    ├── model_comparison_results.png
    ├── taxonomy_group_distribution.png
    ├── radway_phase_distribution.png
    ├── taxonomy_radway_heatmap.png
    ├── label_quality_examples.png
    └── prompt_architecture.png
```

---

**Note:** This presentation summary focuses on essential information for presenting the methodology and results. For detailed methodology, see:
- `reports/01_stage_reports/stage08_llm_labeling/stage08_research_article_draft_summary.md`
- `reports/02_findings/methodology_llm_labeling_and_taxonomy/llm_labeling_and_taxonomy_mapping_methodology.md`

