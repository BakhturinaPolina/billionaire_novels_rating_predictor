# LLM-Based Topic Labeling and Taxonomy Mapping: Research Article Draft Summary

**Date:** December 2024  
**Purpose:** Essential information for research article methodology and results sections

---

## Executive Summary

This document summarizes the essential information from Stage 08 (LLM Labeling) and Stage 09 (Category Mapping) for inclusion in a research article. The methodology combines large language models (LLMs) with zero-shot classification to automatically label and categorize 368 BERTopic topics extracted from a corpus of billionaire romance novels.

**Key Achievements:**
- Successfully labeled 368 topics using LLM-based generation (100% coverage)
- Mapped 361 topics to theory-driven taxonomy categories (98.1% coverage)
- Mapped 361 topics to Radway's 13 narrative functions (98.1% coverage)
- Established reproducible pipeline for computational literary analysis

---

## 1. Methodology Overview

### 1.1 Three-Stage Pipeline

The methodology consists of three sequential stages:

1. **Stage 08: LLM-Based Topic Labeling**
   - Input: BERTopic topics (keywords + representative document snippets)
   - Output: Descriptive labels (2-6 words) + scene summaries + metadata
   - Model: `mistralai/Mistral-Nemo-Instruct-2407` via OpenRouter API
   - Coverage: 368/368 topics (100%)

2. **Stage 09 Stage 2: Theory-Driven Taxonomy Mapping**
   - Input: LLM-generated labels from Stage 08
   - Output: Taxonomy category mappings (6 groups, 20 categories)
   - Method: Zero-shot classification
   - Coverage: 361/368 topics (98.1%)

3. **Stage 09 Stage 3: Radway Narrative Functions Mapping**
   - Input: Taxonomy mappings from Stage 2
   - Output: Radway function mappings (R1-R13, 3 phases)
   - Method: Zero-shot classification with heuristic overrides
   - Coverage: 361/368 topics (98.1%)

### 1.2 Model Selection and Rationale

**Primary Model:** `mistralai/Mistral-Nemo-Instruct-2407`

**Selection Criteria:**
- **Instruction Following**: Strong adherence to prompt constraints and JSON formatting
- **Literary Analysis Capability**: Handles genre-specific terminology (romance/erotic fiction)
- **Research Reliability**: Low hallucination rate, conservative inference approach
- **Cost-Effectiveness**: ~$0.017 per 368 topics via OpenRouter API

**Model Comparison Results:**
- Tested 6 models on 5-topic subset
- Nemo-Instruct: Best balance of specificity, genre awareness, and format compliance
- Alternative models (Cydonia, Anubis) showed better literary nuance but slightly less consistency
- Failed models (Grok, DeepSeek): 100% keyword copying, 0% success rate

**Key Finding:** Instruction-tuned models (Nemo family) outperform creative/generative models for structured research tasks requiring format compliance and literal interpretation.

---

## 2. Prompt Design and Anti-Hallucination Strategies

### 2.1 Romance-Aware Prompt Architecture

**System Prompt Components:**
1. **Role Definition**: Establishes domain context (romantic and erotic fiction)
2. **Format Rules**: Output exactly one 2-6 word noun phrase, JSON structure
3. **Priority Hierarchy**: Action → Role → Setting → Tone
4. **Snippet Integration**: Trust snippets over keywords when they disagree
5. **Disambiguation Requirements**: Force explicit distinctions between similar topics
6. **Anti-Hallucination Constraints**: Hard rules prohibiting common hallucination patterns

**Key Design Principles:**
- **Literal over Abstract**: "Rough Angry Kisses in Hallway" vs "Intense Love"
- **Specific over Generic**: "Blowjob in Bed" vs "Erotic Intimacy"
- **Scene-Level over Keyword-Level**: Base labels on actual scene content
- **Neutral and Scientific**: Explicit terminology acceptable, but factual tone required

### 2.2 Representative Document Snippets

**Implementation:**
- Extract 6 representative document snippets per topic (BERTopic's `get_representative_docs()`)
- ~200 characters per snippet (typically 3-4 sentences)
- Selected for centrality to topic (not random)

**Why Snippets Matter:**
- **Disambiguation**: Keywords "mouth, tongue, suck" ambiguous; snippets show specific sexual act
- **Scene Context**: Capture setting, tone, explicit acts that keywords miss
- **Hallucination Prevention**: Prevent model from inferring "Board Game Foreplay" from "board, table, chair" keywords

**Token Cost Analysis:**
- Snippets add ~75 tokens per topic (7.6% increase)
- Justified by significant label quality improvement

### 2.3 Anti-Hallucination Constraints

**Empirically Identified Patterns:**
1. **"Dinner Date" Hallucination**: Models infer "invitation" or "date" from food keywords
   - **Fix**: "Do NOT use 'dinner date' unless snippets/keywords explicitly mention asking/inviting"

2. **"Repair" Hallucination**: Models infer mechanical repair from car-related keywords
   - **Fix**: "Do NOT use 'repair' unless keywords/snippets include mechanical terms"

3. **"Heartbreak" Hallucination**: Models infer relationship ending from emotional keywords
   - **Fix**: "Do NOT use 'heartbreak' unless emotional pain in relationship ending is clearly described"

**Effectiveness:** Hard constraints with explicit conditions prevent over-correction while reducing hallucination.

---

## 3. Taxonomy Structure

### 3.1 Romance Corpus Topic Taxonomy

**6 Main Groups, 20 Categories:**

1. **Embodied & Sensory Experience** (3 categories)
   - Body parts, pain/injury, physical activity

2. **Sexuality, Attraction & Intimacy** (4 categories)
   - Attraction, kissing, explicit sexual acts, aftercare

3. **Emotions, Cognition & Inner Life** (4 categories)
   - Positive emotions, negative emotions, ambivalence, moral reflection

4. **Relationship Trajectory (Main Couple)** (5 categories)
   - Meeting, bonding, secrets, conflict, reconciliation

5. **Social World Outside Couple** (3 categories)
   - Family, friends, community

6. **Luxury Lifestyle & Material World** (1 category)
   - Wealth, luxury, material objects

**Coverage:** 361/368 topics (98.1%) successfully mapped to taxonomy categories.

### 3.2 Radway's 13 Narrative Functions

**Phase I: Initial Conflict & Isolation** (R1-R7)
- R1: Heroine's social identity destroyed
- R2: Heroine reacts antagonistically
- R3: Hero responds ambiguously
- R4: Heroine interprets hero's behavior as purely sexual
- R5: Heroine responds with anger/coldness
- R6: Hero retaliates/punishes
- R7: Physical/emotional separation

**Phase II: Turning Point & Recognition** (R8-R10)
- R8: Hero treats heroine tenderly
- R9: Heroine responds warmly
- R10: Heroine reinterprets hero's behavior

**Phase III: Commitment & Restoration** (R11-R13)
- R11: Hero declares love and commitment
- R12: Heroine responds sexually and emotionally
- R13: Heroine's identity restored (HEA)

**Distribution Results:**
- Phase I: 147 topics (54.0% of function-mapped topics)
- Phase II: 96 topics (35.3%)
- Phase III: 28 topics (10.3%)
- "None" (not narrative functions): 96 topics (26.6% of classified)

**Key Finding:** Phase I (conflict and isolation) dominates narrative function distribution, suggesting conflict and tension are central to romance narrative structure.

---

## 4. Quality Assurance and Validation

### 4.1 Model Comparison Metrics

**Evaluation Criteria:**
- **Success Rate**: Percentage of topics receiving multi-word labels (vs single-word keyword copies)
- **Average Words per Label**: Should be 2-6 words
- **Keyword Copying**: Percentage of topics where model just copied first keyword
- **Label Quality**: Manual inspection of descriptive, meaningful labels

**Results (30-topic test set):**
- `mistralai/Mistral-Nemo-Instruct-2407`: 100% success rate, 2.30 avg words, 0% keyword copying ✅
- `mistralai/mistral-7b-instruct:free`: 93.3% success rate, 2.80 avg words, 6.7% keyword copying ✅
- `x-ai/grok-4.1-fast`: 0% success rate, 1.00 avg words, 100% keyword copying ❌
- `deepseek/deepseek-chat-v3-0324`: 0% success rate, 1.00 avg words, 100% keyword copying ❌

### 4.2 Coverage Metrics

- **Label Coverage**: 368/368 topics (100%)
- **Taxonomy Coverage**: 361/368 topics (98.1%)
- **Radway Coverage**: 361/368 topics (98.1%)
- **Radway Function Mapping**: 272/361 topics (75.3% of classified topics)

### 4.3 Classification Quality

**Radway Mapping Confidence:**
- High confidence: 130 topics (36% of classified)
- Medium confidence: Majority of remaining function topics
- Low confidence: Small number of edge cases

**Heuristic Override System:**
- Corrected systematic errors (explicit sex → R12, commitment → R11/R13)
- Deterministic decoding (temperature=0.0) for reproducibility
- Gated "none" decision process to prevent false negatives

---

## 5. Computational Implementation

### 5.1 Infrastructure

**OpenRouter API:**
- Single API key for multiple models
- No local infrastructure required (no GPU, no model downloads)
- Cost: ~$0.017 per 368 topics
- Rate limiting: 4.0 second delay between calls (conservative)

**Model Parameters:**
- Temperature: 0.35 (balanced for consistency + natural phrasing)
- Max tokens: 40 (labeling), 220 (taxonomy/Radway mapping)
- Sampling: Deterministic for taxonomy/Radway (temperature=0.0)

### 5.2 Processing Strategies

**Streaming Mode:**
- Process topics incrementally, write to disk as generated
- Memory efficient for 368+ topics
- Fault tolerant (preserves progress on crash)

**Caching and Resumption:**
- Load existing labels, skip already-processed topics
- Enables incremental updates and cost savings

**Snippet Reranking:**
- Maximal Marginal Relevance (MMR) for diverse, informative snippets
- Balances relevance with diversity

### 5.3 Integration with BERTopic

**Metadata Storage:**
- All labels, taxonomy, and Radway mappings stored in `topic_metadata_` attribute
- Single source of truth in BERTopic model object
- No separate JSON files required for analysis

**Data Structure:**
```python
{
  topic_id: {
    "label": "Makeout In Parked Car",
    "keywords": ["car", "seat", "door", "parked", "kiss"],
    "scene_summary": "...",
    "primary_categories": ["romance_core", "sexual_content"],
    "main_category_id": "4.2",
    "radway_functions": {
      "radway_main_id": "R8",
      "radway_phase": "II",
      ...
    }
  }
}
```

---

## 6. Key Findings for Results Section

### 6.1 Label Quality

- **Specificity**: Labels are concrete and scene-level (e.g., "Rough Angry Kisses in Hallway" vs "Intense Love")
- **Genre Awareness**: Distinguishes romance core, sexual content, domestic scenes, emotional uncertainty
- **Discriminative Power**: Similar topics receive distinct labels capturing their differences

### 6.2 Taxonomy Distribution

- **Largest Group**: Relationship Trajectory (Main Couple) - 158 topics
- **Second Largest**: Emotions, Cognition & Inner Life - 59 topics
- **Third Largest**: Sexuality, Attraction & Intimacy - 55 topics

### 6.3 Radway Narrative Structure

- **Phase I Dominance**: 54% of narrative function topics belong to Phase I (conflict and isolation)
- **Phase II Transition**: 35% map to Phase II (turning point and recognition)
- **Phase III Resolution**: Only 10% map to Phase III (commitment and restoration)

**Interpretation:** Conflict and tension occupy more textual space than resolution, despite resolution's narrative importance.

### 6.4 Taxonomy-to-Radway Mapping Patterns

**Strong Mappings:**
- Taxonomy 2.3 (Explicit Sexual Acts) → R12 (Phase III)
- Taxonomy 4.5 (Reconciliation, Commitments & HEA) → R11/R13 (Phase III)
- Taxonomy 4.4 (Conflict, Distance & Breakup Threats) → R2/R7 (Phase I)
- Taxonomy 4.2 (Bonding, Everyday Intimacy & Growth) → R8/R9 (Phase II)

**Weak/No Mappings:**
- Taxonomy 5.x (Social World Outside Couple) → Mostly "none"
- Taxonomy 6.x (Work, Wealth, Status) → Mostly "none"
- Taxonomy 8.x (Spaces, Time, Activities) → Mostly "none"

---

## 7. Limitations and Future Directions

### 7.1 Current Limitations

1. **Snippet Selection**: Relies on BERTopic's representative doc selection, may miss variations
2. **Hallucination**: Despite constraints, models may still infer details not present in input
3. **Coverage**: Not all topics successfully map to taxonomy/Radway (98.1% coverage)
4. **Cost**: API costs scale with number of topics (acceptable for research but not free)
5. **Latency**: Sequential processing means 368 topics take ~6-18 minutes

### 7.2 Future Improvements

1. **Diversity-Aware Snippet Selection**: Use MMR or similar to ensure snippet diversity
2. **Adaptive Snippet Count**: More snippets for complex topics, fewer for simple ones
3. **Parallel Processing**: Process multiple topics concurrently to reduce latency
4. **Few-Shot Example Selection**: Dynamic few-shot selection based on topic similarity
5. **Automated Quality Metrics**: Label length, uniqueness, keyword coverage, snippet alignment
6. **Model-Specific Prompt Tuning**: Optimize prompts for different model families

---

## 8. Figures and Tables (Placeholders)

### 8.1 Methodology Figures

**Figure 1: Three-Stage Pipeline Diagram**
- Location: `figures/pipeline_diagram.png`
- Description: Flowchart showing Stage 08 → Stage 09 Stage 2 → Stage 09 Stage 3

**Figure 2: Prompt Architecture**
- Location: `figures/prompt_architecture.png`
- Description: System prompt components and user prompt template structure

**Figure 3: Model Comparison Results**
- Location: `figures/model_comparison_results.png`
- Description: Bar chart comparing success rates, avg words, keyword copying across models

### 8.2 Results Figures

**Figure 4: Taxonomy Group Distribution**
- Location: `figures/taxonomy_group_distribution.png`
- Description: Bar chart or pie chart showing distribution of topics across 6 taxonomy groups

**Figure 5: Radway Phase Distribution**
- Location: `figures/radway_phase_distribution.png`
- Description: Bar chart showing distribution of topics across Phase I, II, III, and "none"

**Figure 6: Taxonomy-to-Radway Mapping Heatmap**
- Location: `figures/taxonomy_radway_heatmap.png`
- Description: Heatmap showing which taxonomy categories map to which Radway functions

**Figure 7: Label Quality Examples**
- Location: `figures/label_quality_examples.png`
- Description: Table or visualization showing example labels for different topic types

### 8.3 Tables

**Table 1: Model Comparison Summary**
- Success rates, avg words, keyword copying for all tested models

**Table 2: Taxonomy Category Definitions**
- Full taxonomy structure with IDs, names, groups, descriptions

**Table 3: Radway Function Definitions**
- All 13 functions with phases and descriptions

**Table 4: Coverage Statistics**
- Label coverage, taxonomy coverage, Radway coverage by topic type

---

## 9. References to Include

- Radway, J. (1984). *Reading the Romance: Women, Patriarchy, and Popular Literature*. University of North Carolina Press.
- BERTopic Documentation: https://maartengr.github.io/BERTopic/
- OpenRouter API Documentation: https://openrouter.ai/docs
- Mistral AI Models: https://mistral.ai/

---

## 10. Code and Data Availability

**Code Repository:**
- Stage 08: `src/stage08_llm_labeling/`
- Stage 09: `src/stage09_category_mapping/`

**Output Files:**
- Labels JSON: `results/stage08_llm_labeling/labels_pos_openrouter_*.json`
- Taxonomy Mappings: `results/stage09_category_mapping/stage2_theory_driven_categories/taxonomy_mappings_*.json`
- Radway Mappings: `results/stage09_category_mapping/stage3_radway_functions/radway_classification_all_topics.json`
- BERTopic Model with Metadata: `models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_llm_labels_and_metadata_disambiguated.pkl`

---

## Appendix: Key Statistics for Results Section

- **Total Topics**: 368
- **Topics with Labels**: 368 (100%)
- **Topics with Taxonomy**: 361 (98.1%)
- **Topics with Radway**: 361 (98.1%)
- **Topics with Radway Function**: 272 (75.3% of classified)
- **Topics with Radway "None"**: 96 (26.6% of classified)
- **Unique Taxonomy Categories**: 28
- **Unique Taxonomy Groups**: 9
- **Unique Radway Functions**: 13 (all functions represented)
- **Unique Radway Phases**: 4 (I, II, III, NA)
- **High Confidence Classifications**: 130 (36% of classified)

---

**Note:** This document serves as a summary for research article drafting. For detailed methodology, see:
- `reports/02_findings/methodology_llm_labeling_and_taxonomy/llm_labeling_and_taxonomy_mapping_methodology.md`
- `reports/01_stage_reports/stage08_llm_labeling/stage08_model_selection_and_prompt_design_rationale.md`
- `reports/01_stage_reports/stage09_category_mapping/03_stage3_radway_functions/stage09_stage3_radway_narrative_functions_research_report.md`

