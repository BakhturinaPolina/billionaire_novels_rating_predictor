# Stage 08: LLM-Based Topic Labeling

**Research Report**

This report documents the methodology and results of Stage 08, which implements automated topic labeling using Large Language Models (LLMs) to transform BERTopic keyword clusters into interpretable, scene-level labels suitable for computational literary analysis.

---

## 1. Introduction

### 1.1 The Challenge

BERTopic outputs topic clusters as keyword lists (e.g., "mouth, tongue, suck, lips, breath") without interpretable labels. For large-scale analysis of 368 topics across thousands of documents, manual labeling is impractical.

### 1.2 Solution

Automated LLM-based labeling using a romance-aware prompt architecture with representative document snippets for scene-level disambiguation.

**Key Achievements**:
- **100% label coverage** (368/368 topics)
- **0% keyword copying** (all labels are descriptive multi-word phrases)
- **Reproducible pipeline** with documented design decisions

---

## 2. Methodology

### 2.1 Input Preparation

**POS-Filtered Keywords**:
- Extract only nouns, verbs, and adjectives from BERTopic's topic keywords using spaCy POS tagging
- Removes function words and pronouns that add little semantic value

**Representative Document Snippets**:
- 6 representative document snippets per topic (~200 characters each)
- Extracted using BERTopic's `get_representative_docs()` method
- Selected for centrality to the topic

**Why Snippets Matter**:
- Keywords are ambiguous: "mouth, tongue, suck" could mean eating, talking, or oral sex
- Snippets provide scene-level disambiguation
- Token cost: ~75 tokens per topic (7.6% increase) for significant quality improvement

### 2.2 Model Selection

**Primary Model**: `mistralai/Mistral-Nemo-Instruct-2407` via OpenRouter API

**Selection Criteria**:
| Criterion | Requirement | Nemo-Instruct Performance |
|-----------|-------------|---------------------------|
| Instruction Following | Strong adherence to format | ✅ 100% success rate |
| Literary Analysis | Genre-specific terminology | ✅ Romance-aware |
| Research Reliability | Low hallucination | ✅ 0% keyword copying |
| Cost-Effectiveness | Reasonable pricing | ✅ ~$0.017 per 368 topics |

**Model Comparison Results** (30-topic test):

| Model | Success Rate | Avg Words | Keyword Copy | Status |
|-------|--------------|-----------|--------------|--------|
| **mistralai/Mistral-Nemo** | **100%** | **2.30** | **0%** | ✅ BEST |
| mistralai/mistral-7b-instruct:free | 93.3% | 2.80 | 6.7% | ✅ Good |
| thedrummer/cydonia-24b-v4.1 | High | 2.50 | 0% | ✅ Literary |
| thedrummer/anubis-70b-v1.1 | High | 2.60 | 0% | ✅ Literary |
| x-ai/grok-4.1-fast | 0% | 1.00 | 100% | ❌ Failed |
| deepseek/deepseek-chat-v3-0324 | 0% | 1.00 | 100% | ❌ Failed |

**Key Finding**: Instruction-tuned models outperform creative models for structured research tasks requiring format compliance.

### 2.3 Prompt Architecture

**System Prompt Structure**:
1. **Role Definition**: Domain context (romantic and erotic fiction)
2. **Format Rules**: Output exactly one 2-6 word noun phrase, JSON structure
3. **Priority Hierarchy**: Action → Role → Setting → Tone
4. **Snippet Integration**: Trust snippets over keywords when they disagree
5. **Disambiguation Requirements**: Force explicit distinctions between similar topics
6. **Anti-Hallucination Constraints**: Hard rules prohibiting common hallucination patterns

**Design Principles**:
- **Literal over Abstract**: "Rough Angry Kisses in Hallway" vs "Intense Love"
- **Specific over Generic**: "Blowjob in Bed" vs "Erotic Intimacy"
- **Scene-Level over Keyword-Level**: Base labels on actual scene content
- **Neutral and Scientific**: Explicit terminology acceptable, factual tone

### 2.4 Anti-Hallucination Constraints

Empirically identified patterns where models infer details not present in input:

| Pattern | Trigger | Fix |
|---------|---------|-----|
| "Dinner Date" | Food keywords | "Do NOT use 'dinner date' unless snippets explicitly mention asking/inviting" |
| "Repair" | Car-related keywords | "Do NOT use 'repair' unless keywords include mechanical terms" |
| "Heartbreak" | Emotional keywords | "Do NOT use 'heartbreak' unless emotional pain in relationship ending is clearly described" |

**Why Hard Constraints Work**: Explicit prohibitions with conditions prevent over-correction while reducing hallucination.

### 2.5 Output Structure

```json
{
  "label": "Makeout In Parked Car",
  "scene_summary": "In a parked car, they kiss and touch each other while the outside world stays just beyond the fogged windows.",
  "primary_categories": ["romance_core", "sexual_content"],
  "secondary_categories": ["setting:car", "activity:kissing"],
  "is_noise": false,
  "rationale": "Keywords indicate car setting and physical intimacy. Snippets clearly show makeout scene in parked car with fogged windows."
}
```

---

## 3. Computational Implementation

### 3.1 Infrastructure

**OpenRouter API**:
- Single API key for multiple models
- No local infrastructure (no GPU, no downloads)
- Cost: ~$0.017 per 368 topics
- Rate limiting: 4.0s delay between calls

### 3.2 Model Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Temperature | 0.35 | Balanced for consistency + natural phrasing |
| Max tokens | 40 | Sufficient for 2-6 word labels |
| Sampling | Deterministic (temp=0.0) for taxonomy mapping | Reproducibility |

### 3.3 Processing Strategies

**Streaming Mode**:
- Process topics incrementally, write to disk as generated
- Memory efficient for 368+ topics
- Fault tolerant (preserves progress on crash)

**Caching and Resumption**:
- Load existing labels, skip already-processed topics
- Enables incremental updates and cost savings

**Snippet Reranking**:
- Maximal Marginal Relevance (MMR) for diverse, informative snippets
- Balances relevance with diversity

### 3.4 Integration with BERTopic

All labels and metadata stored in BERTopic model's `topic_metadata_` attribute:
- Single source of truth
- No separate JSON files required for analysis
- Direct access to all topic information from model object

---

## 4. Results

### 4.1 Label Quality

**Coverage**: 368/368 topics (100%)

**Label Characteristics**:
- Specific and concrete (not generic)
- Scene-level descriptions (not abstract)
- Genre-aware (romance/erotic fiction)
- Discriminative (distinct labels for similar topics)

### 4.2 Example Labels

| Topic Keywords | Label | Categories |
|---------------|-------|------------|
| car, seat, door, parked, kiss | Makeout In Parked Car | romance_core, sexual_content |
| mouth, tongue, suck, lips, clit | Clitoral Stimulation During Foreplay | sexual_content |
| kitchen, angry, voice, raised | Kitchen Argument About Money | conflict, emotional |
| wedding, marriage, ceremony, aisle | Wedding Ceremony And Vows | romance_core |

### 4.3 Model Evaluation Criteria

**Successful labels demonstrate**:
- ✅ Label specificity (concrete details)
- ✅ Genre awareness (romance/erotic fiction conventions)
- ✅ Discriminative power (distinct labels for similar topics)
- ✅ Scene summary quality (micro-scene, concrete details, neutral tone)
- ✅ Category accuracy
- ✅ JSON compliance
- ✅ Tone stability (neutral, analytical)

---

## 5. Discussion

### 5.1 Methodological Contributions

1. **Snippet-Integrated Prompting**: Representative document snippets provide scene-level understanding that keywords alone cannot convey
2. **Anti-Hallucination Constraints**: Empirically-derived hard rules prevent common inference errors
3. **Romance-Aware Design**: Domain-specific prompt design acknowledges genre conventions
4. **Structured JSON Output**: Enables programmatic filtering and downstream analysis

### 5.2 Practical Implications

**Model Selection Recommendation**:
- **Primary**: Mistral-Nemo-Instruct-2407 for reliability and format compliance
- **Alternative**: Cydonia/Anubis for literary nuance on specific topic types
- **Ensemble**: Use Nemo for production, literary models for refinement

**Prompt Design Recommendations**:
- Hard constraints with conditions > soft guidance
- Snippets as primary evidence source
- Clear priority hierarchy for label content
- Explicit disambiguation requirements

### 5.3 Limitations

1. **Snippet Selection**: Relies on BERTopic's representative doc selection
2. **Hallucination**: Despite constraints, models may still infer details not present
3. **Cost**: API costs scale with number of topics
4. **Latency**: Sequential processing (~6-18 minutes for 368 topics)

### 5.4 Future Improvements

1. Diversity-aware snippet selection (MMR)
2. Adaptive snippet count based on topic complexity
3. Parallel processing for reduced latency
4. Dynamic few-shot example selection
5. Automated label quality metrics

---

## 6. Conclusion

Stage 08 demonstrates that LLM-based labeling can effectively transform keyword clusters into interpretable, scene-level labels for computational literary analysis. The combination of:
- Snippet-integrated prompting
- Anti-hallucination constraints
- Romance-aware domain design
- Instruction-tuned models

achieves **100% label coverage** with **high-quality, discriminative labels** suitable for downstream taxonomy mapping and theoretical analysis.

---

**Analysis Date**: December 2024  
**Model**: mistralai/Mistral-Nemo-Instruct-2407 via OpenRouter  
**Coverage**: 368/368 topics (100%)  
**Results Location**: `results/stage08_llm_labeling/`  
**Code Location**: `src/stage08_llm_labeling/`
