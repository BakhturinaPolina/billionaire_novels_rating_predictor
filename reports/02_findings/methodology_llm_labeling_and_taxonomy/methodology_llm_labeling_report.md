# LLM-Based Topic Labeling and Taxonomy Mapping: Methodology Report

**Research Methodology Summary**

This report documents the three-stage pipeline for automated topic interpretation and theoretical mapping in computational literary analysis of romance fiction.

---

## 1. Overview

### 1.1 Pipeline Structure

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

### 1.2 Key Achievements

- **100% label coverage** (368/368 topics)
- **98.1% taxonomy coverage** (361/368 topics)
- **98.1% Radway coverage** (361/368 topics)
- **Fully automated pipeline** from keywords to theoretical categories
- **Reproducible methodology** with documented design decisions

---

## 2. Stage 08: LLM-Based Topic Labeling

### 2.1 Input Preparation

- **POS-Filtered Keywords**: Nouns, verbs, adjectives only (via spaCy)
- **Representative Snippets**: 6 snippets per topic (~200 chars each)
- **Token Cost**: ~75 tokens per topic for snippets (7.6% increase)

### 2.2 Model Selection

**Primary Model**: `mistralai/Mistral-Nemo-Instruct-2407`

| Criterion | Result |
|-----------|--------|
| Success Rate | 100% |
| Keyword Copying | 0% |
| Avg Words/Label | 2.30 |
| Cost | ~$0.017 per 368 topics |

### 2.3 Prompt Design

**Romance-Aware Architecture**:
1. Role Definition (domain context)
2. Format Rules (2-6 word noun phrase)
3. Priority Hierarchy (Action → Role → Setting → Tone)
4. Snippet Integration (trust snippets over keywords)
5. Anti-Hallucination Constraints

**Anti-Hallucination Examples**:
- "Do NOT use 'dinner date' unless snippets explicitly mention asking/inviting"
- "Do NOT use 'repair' unless keywords include mechanical terms"

---

## 3. Stage 09 Stage 2: Taxonomy Mapping

### 3.1 Taxonomy Structure

**6 Main Groups, 20 Categories**:

1. **Embodied & Sensory Experience** (3 categories)
   - Body parts, pain/injury, physical activity

2. **Sexuality, Attraction & Intimacy** (4 categories)
   - Attraction, kissing, explicit sexual acts, aftercare

3. **Emotions, Cognition & Inner Life** (4 categories)
   - Positive emotions, negative emotions, ambivalence, moral reflection

4. **Relationship Trajectory (Main Couple)** (5 categories) ← Largest group
   - Meeting, bonding, secrets, conflict, reconciliation

5. **Social World Outside Couple** (3 categories)
   - Family, friends, community

6. **Luxury Lifestyle & Material World** (1 category)
   - Wealth, luxury, material objects

### 3.2 Zero-Shot Classification

**Input**: Topic keywords + LLM label + scene summary
**Output**: Main category ID, secondary category ID, confidence, rationale
**Coverage**: 361/368 topics (98.1%)

---

## 4. Stage 09 Stage 3: Radway Narrative Functions

### 4.1 Radway's 13 Functions

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

### 4.2 Distribution Results

| Phase | Topics | Percentage |
|-------|--------|------------|
| Phase I (Conflict) | 147 | 54.0% |
| Phase II (Turning Point) | 96 | 35.3% |
| Phase III (Commitment) | 28 | 10.3% |
| "None" (Background) | 96 | 26.6% |

**Key Finding**: Conflict and tension occupy more textual space than resolution.

### 4.3 Taxonomy-to-Radway Mappings

**Strong Mappings**:
- Explicit Sexual Acts (2.3) → R12 (Phase III)
- Reconciliation & HEA (4.5) → R11/R13 (Phase III)
- Conflict & Breakup (4.4) → R2/R7 (Phase I)
- Bonding & Intimacy (4.2) → R8/R9 (Phase II)

**Weak/No Mappings** (Background content):
- Social World (5.x) → Mostly "none"
- Work & Wealth (6.x) → Mostly "none"

---

## 5. Computational Implementation

### 5.1 Infrastructure

- **API**: OpenRouter (single API key, multiple models)
- **Cost**: ~$0.017 per 368 topics
- **Rate Limiting**: 4.0s delay between calls
- **Processing Time**: ~6-18 minutes for 368 topics

### 5.2 Quality Assurance

**Heuristic Overrides** for systematic errors:
- Explicit sex scenes (taxonomy 2.3) → R12 (not R4)
- Wedding/commitment cues → R11/R13 (not none)
- Deterministic decoding (temperature=0.0) for reproducibility

### 5.3 Integration

All metadata stored in BERTopic's `topic_metadata_` attribute:
- Single source of truth
- Labels, taxonomy, Radway mappings unified
- No separate JSON files needed

---

## 6. Limitations and Future Work

### 6.1 Limitations

1. Snippet selection relies on BERTopic's representative doc selection
2. Hallucination possible despite constraints
3. 98.1% coverage (7 topics unmapped)
4. API costs scale with topic count
5. Sequential processing (~6-18 min)

### 6.2 Future Improvements

1. Diversity-aware snippet selection (MMR)
2. Adaptive snippet count
3. Parallel processing
4. Dynamic few-shot selection
5. Automated quality metrics
6. Model-specific prompt tuning

---

## 7. References

- Radway, J. (1984). *Reading the Romance: Women, Patriarchy, and Popular Literature*. University of North Carolina Press.
- BERTopic Documentation: https://maartengr.github.io/BERTopic/
- OpenRouter API Documentation: https://openrouter.ai/docs

---

**Code Location**: `src/stage08_llm_labeling/`, `src/stage09_category_mapping/`  
**Results Location**: `results/stage08_llm_labeling/`, `results/stage09_category_mapping/`
