# LLM-Based Topic Labeling and Taxonomy Mapping: Methodology and Implementation

**Draft Research Article Section**

This document describes the theoretical foundations, methodological approach, and computational implementation of our LLM-based topic labeling and taxonomy mapping pipeline for analyzing romance fiction. The methodology consists of three main stages: (1) generating descriptive labels for BERTopic topics using large language models, (2) mapping topics to a theory-driven taxonomy using zero-shot classification, and (3) mapping topics to Radway's narrative functions for structural analysis.

---

## 1. Theoretical Foundations

### 1.1 The Challenge of Topic Interpretation

Topic modeling algorithms like BERTopic identify clusters of semantically similar text segments, but they do not provide interpretable labels. The output consists of keyword lists (e.g., "mouth, tongue, suck, lips, breath") that require human interpretation to understand what the topic represents. For large-scale analysis of hundreds of topics across thousands of documents, manual labeling is impractical.

### 1.2 Why Large Language Models for Labeling?

Large language models (LLMs) offer a solution by combining:
- **Semantic understanding**: Ability to synthesize meaning from keyword lists
- **Domain knowledge**: Training on diverse text corpora including literary fiction
- **Consistency**: Reproducible labeling across similar topics
- **Scalability**: Can process hundreds of topics automatically

However, LLMs also present challenges:
- **Hallucination**: Tendency to infer details not present in the input
- **Vagueness**: May produce generic labels like "Erotic Intimacy" instead of specific scene descriptions
- **Format compliance**: Must follow strict output requirements (2-6 word labels, JSON structure)

### 1.3 Zero-Shot Classification for Taxonomy Mapping

Zero-shot classification allows mapping topics to predefined categories without training data. This approach is ideal for:
- **Theory-driven analysis**: Mapping to predefined theoretical frameworks (e.g., Radway's narrative functions)
- **Consistency**: Using the same classification criteria across all topics
- **Interpretability**: Categories have clear definitions and theoretical grounding

The key insight is that LLMs can understand category definitions and apply them to new topics, even if they haven't seen training examples of that specific mapping.

---

## 2. Methodological Approach

### 2.1 Stage 1: LLM-Based Topic Labeling

#### 2.1.1 Input Preparation

**POS-Filtered Keywords**: We extract only nouns, verbs, and adjectives from BERTopic's topic keywords using spaCy part-of-speech tagging. This removes function words ("the", "and", "of") and pronouns that add little semantic value, focusing the model's attention on content words.

**Representative Document Snippets**: For each topic, we extract 6 representative document snippets (typically 3-4 sentences each, ~200 characters) using BERTopic's `get_representative_docs()` method. These snippets are selected for their centrality to the topic, providing scene-level context that keywords alone cannot convey.

**Why Snippets Matter**: 
- Keywords are ambiguous: "mouth, tongue, suck" could mean eating, talking, or oral sex
- Snippets provide disambiguation: "She knelt between his thighs, taking him into her mouth" → clearly indicates a specific sexual act
- Scene-level understanding: Snippets capture setting, tone, and specific actions that keywords miss

#### 2.1.2 Prompt Architecture

Our prompt system uses a **romance-aware, snippet-integrated design** with the following components:

**System Prompt Structure**:
1. **Role Definition**: Establishes domain context (romantic and erotic fiction)
2. **Format Rules**: Output exactly one 2-6 word noun phrase, no quotes, no markdown
3. **Priority Hierarchy**: What to encode in the label (Action → Role → Setting → Tone)
4. **Snippet Integration**: Instructions to trust snippets over keywords when they disagree
5. **Disambiguation Requirements**: Force explicit distinctions between similar topics
6. **Anti-Hallucination Constraints**: Hard rules prohibiting common hallucination patterns

**Key Design Principles**:

- **Literal over Abstract**: Prefer "Rough Angry Kisses in Hallway" over "Intense Love"
- **Specific over Generic**: Prefer "Blowjob in Bed" over "Erotic Intimacy"
- **Scene-Level over Keyword-Level**: Base labels on what actually happens in snippets, not just keyword associations
- **Neutral and Scientific**: Use explicit sexual terminology when appropriate, but keep tone factual and non-romanticized

**Anti-Hallucination Constraints**: Based on empirical testing, we identified patterns where models infer details not present in the input:
- "Dinner Date" hallucination: Models infer "invitation" or "date" from food keywords
- "Repair" hallucination: Models infer mechanical repair from car-related keywords
- "Heartbreak" hallucination: Models infer relationship ending from emotional keywords

We address these with explicit prohibitions: "Do NOT use X unless Y is clearly present in snippets/keywords."

#### 2.1.3 Model Selection

**Primary Model**: `mistralai/Mistral-Nemo-Instruct-2407`

**Selection Criteria**:
- **Instruction Following**: Strong adherence to prompt constraints and format requirements
- **Literary Analysis Capability**: Handles genre-specific terminology without over-interpreting
- **Research Reliability**: Low hallucination rate, conservative approach to inference
- **Cost-Effectiveness**: Accessible via OpenRouter API at reasonable pricing

**Why Nemo Family Models**: All comparison models are Nemo-based (Nemo-Instruct, Nemo-Celeste, Nemo-Gutenberg) because:
- Code compatibility: Consistent instruction behavior across models
- Infrastructure simplicity: Single API key, single codebase
- Instruction following: Maintains format compliance even after fine-tuning
- Progressive enhancement: Can add literary tuning (Gutenberg) or narrative tuning (Celeste) as needed

**Alternative Models Tested**:
- `nbeerbower/mistral-nemo-gutenberg-12B-v2`: Literary fine-tuning on Project Gutenberg corpus, better for thematic abstraction
- `nothingiisreal/mn-celeste-12b`: Story-writing/roleplay fine-tuning, better for fine-grained scene distinctions
- `x-ai/grok-4.1-fast`: Failed completely (only returned single-word keyword copies)
- `deepseek/deepseek-chat-v3-0324`: Failed completely (only returned single-word keyword copies)

#### 2.1.4 Output Structure

When using improved prompts (`--use-improved-prompts`), the model outputs structured JSON:

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

This structured output enables:
- Programmatic filtering (e.g., `is_noise == true`)
- Category-based grouping for downstream analysis
- Interpretability through rationale fields

### 2.2 Stage 2: Theory-Driven Taxonomy Mapping

#### 2.2.1 Taxonomy Structure

We developed a **Romance Corpus Topic Taxonomy** with 6 main groups and 20 categories:

1. **Embodied & Sensory Experience** (3 categories): Body parts, pain/injury, physical activity
2. **Sexuality, Attraction & Intimacy** (4 categories): Attraction, kissing, explicit sexual acts, aftercare
3. **Emotions, Cognition & Inner Life** (4 categories): Positive emotions, negative emotions, ambivalence, moral reflection
4. **Relationship Trajectory (Main Couple)** (5 categories): Meeting, bonding, secrets, conflict, reconciliation
5. **Social World Outside Couple** (3 categories): Family, friends, community
6. **Luxury Lifestyle & Material World** (1 category): Wealth, luxury, material objects

Each category has:
- **ID**: Hierarchical identifier (e.g., "4.2" = Relationship Trajectory, Bonding)
- **Name**: Human-readable label
- **Group**: Parent category group
- **Description**: Detailed definition for LLM classification

#### 2.2.2 Zero-Shot Classification Approach

**Input**: For each topic, we provide:
- Topic keywords from BERTopic
- LLM-generated label and scene summary (from Stage 1)
- Primary/secondary categories (from Stage 1 metadata)
- Optional representative document snippets

**Classification Task**: Map each topic to:
- **Main category ID**: Primary taxonomy category (required)
- **Secondary category ID**: Optional second category if topic spans multiple categories
- **Other plausible IDs**: List of alternative categories that could fit
- **Confidence**: Low/medium/high confidence in the mapping
- **Rationale**: Explanation of why this category fits

**Prompt Design**: The system prompt includes:
- Full taxonomy structure with IDs, names, groups, and descriptions
- Interpretation hints linking topic characteristics to taxonomy groups
- Disambiguation rules for common confusions (e.g., distinguishing "Explicit Sexual Acts" from "Kissing & Non-Explicit Affection")
- Examples showing correct classification patterns

**Model**: Uses the same `mistralai/Mistral-Nemo-Instruct-2407` model via OpenRouter for consistency with Stage 1.

#### 2.2.3 Integration with BERTopic

The taxonomy mappings are stored in the BERTopic model's `topic_metadata_` attribute, creating a single source of truth that combines:
- Topic keywords (from BERTopic)
- LLM-generated labels (from Stage 1)
- Taxonomy mappings (from Stage 2)
- Source metadata (scene summaries, categories, rationale)

This integration enables:
- Direct access to all topic information from the model object
- No need for separate JSON files during analysis
- Consistent data structure across the pipeline

### 2.3 Stage 3: Radway Narrative Functions Mapping

#### 2.3.1 Radway's 13 Narrative Functions

Janice Radway's (1984) analysis of romance fiction identifies 13 narrative functions that structure the heroine-hero relationship arc:

**Phase I: Initial Conflict & Isolation** (R1-R7)
- R1: Heroine's social identity is destroyed
- R2: Heroine reacts antagonistically to the hero
- R3: Hero responds ambiguously to heroine
- R4: Heroine interprets hero's behaviour as purely sexual interest
- R5: Heroine responds with anger or coldness
- R6: Hero retaliates or punishes heroine
- R7: Hero and heroine are physically or emotionally separated

**Phase II: Turning Point & Recognition** (R8-R10)
- R8: Hero treats heroine tenderly
- R9: Heroine responds warmly to hero's tenderness
- R10: Heroine reinterprets hero's behaviour as result of previous hurt

**Phase III: Commitment & Restoration** (R11-R13)
- R11: Hero declares love and demonstrates commitment
- R12: Heroine responds sexually and emotionally
- R13: Heroine's social identity is restored (HEA)

#### 2.3.2 Zero-Shot Classification to Radway Functions

**Input**: Uses the taxonomy JSON (from Stage 2) as the single source of truth, which already contains:
- Taxonomy mappings (main_category_id, secondary_category_id)
- Source metadata (label, keywords, scene_summary, primary/secondary categories)
- Optional representative document snippets

**Classification Task**: Map each topic to:
- **Radway main ID**: Primary Radway function (R1-R13 or "none")
- **Radway secondary ID**: Optional second function if topic spans multiple functions
- **Radway other plausible IDs**: List of alternative functions
- **Radway phase**: I, II, III, or NA (for "none")
- **Radway is none**: Boolean indicating if topic doesn't fit any function
- **Radway confidence**: Low/medium/high confidence
- **Radway rationale**: Explanation of the mapping

**Prompt Design**: The system prompt includes:
- Full Radway function definitions with phases and descriptions
- Interpretation hints linking taxonomy groups to Radway functions:
  - Topics in "Relationship Trajectory (Main Couple)" group → almost always map to R1-R13 (not "none")
  - Topics in "Sexuality, Attraction & Intimacy" group → R4 (early attraction) or R12 (post-commitment union)
  - Topics in "Emotions, Cognition & Inner Life" with relationship conflict → R1-R7
- Disambiguation rules:
  - R4 vs R12: R4 for attraction without explicit acts, R12 for described sex acts
  - R7 (separation) is narrow: only for actual breakup/separation, not arguments
  - Commitment overrides: wedding/marriage/proposal → R11/R13
- Gated "none" decision: Only for topics clearly about background context (work, wealth, side characters) not the heroine-hero relationship

**Post-LLM Heuristic Overrides**: Conservative rule-based corrections for systematic errors:
- Explicit sex scenes (taxonomy_main_id = 2.3) → R12 (not R4)
- Wedding/marriage/commitment cues → R11/R13 (not none/R8)
- R7 only when actual breakup/separation cues exist
- R4 sanity checks for non-sexual contexts

**Output**: Radway mappings are merged back into the taxonomy JSON under a `"radway_functions"` key, preserving all existing fields. This creates a unified data structure with taxonomy + Radway mappings.

---

## 3. Computational Tools and Strategies

### 3.1 Infrastructure: OpenRouter API

**Why OpenRouter**: 
- **Single API Key**: Access to multiple LLM models through one interface
- **No Local Infrastructure**: No GPU requirements, model downloads, or memory management
- **Model Access**: Provides access to models not easily available via Hugging Face (e.g., Celeste, Gutenberg variants)
- **Cost-Effectiveness**: Reasonable pricing for research-scale labeling (~$0.017 per 368 topics)
- **Consistency**: OpenAI-compatible API, minimal code changes required

**Rate Limiting**: Conservative 4.0 second delay between API calls to respect rate limits. Can be adjusted based on account tier.

**Error Handling**: Robust retry logic with exponential backoff for transient API failures.

### 3.2 Model Parameters

**Temperature**: Default `0.35` (balanced)
- Too low (0.0-0.2): Overly deterministic, may miss valid alternative phrasings
- Too high (0.7-1.0): Excessive variation, inconsistent labels
- Sweet spot (0.3-0.4): Allows natural phrasing variation while maintaining consistency

**Max Tokens**: 
- Labeling: `40` tokens (sufficient for 2-6 word labels)
- Taxonomy mapping: `220` tokens (allows for JSON structure and rationale)
- Radway mapping: `220` tokens (allows for JSON structure and rationale)

**Sampling**: Deterministic sampling (`do_sample=False`, `temperature=0.0`) for taxonomy and Radway mapping to ensure reproducibility.

### 3.3 Processing Strategies

#### 3.3.1 Streaming vs Batch Processing

**Streaming Mode**: Process topics incrementally and write labels to disk as they're generated.
- **Memory Efficiency**: For 368+ topics, avoids keeping all labels in memory
- **Fault Tolerance**: If process crashes, already-written labels are preserved
- **Progress Visibility**: Can monitor progress by checking JSON file size

**Batch Mode**: Load all topics, generate all labels, save at once.
- **Use Case**: Small topic sets (< 100 topics) or when all labels needed in memory for post-processing

#### 3.3.2 Caching and Resumption

**Existing Labels**: If a labels JSON file already exists, the system can:
- Load existing labels
- Skip topics that already have labels
- Only process new or updated topics

This enables:
- Incremental updates when adding new topics
- Resumption after interruptions
- Cost savings by avoiding redundant API calls

#### 3.3.3 Snippet Reranking

**Maximal Marginal Relevance (MMR)**: For topics with many representative documents, we use MMR to select diverse, informative snippets:
- Balances relevance (snippets central to the topic) with diversity (snippets that differ from each other)
- Ensures snippets provide complementary information rather than redundant examples
- Uses lightweight sentence transformer model (`all-MiniLM-L6-v2`) for embedding-based similarity

**Centrality-Based Selection**: Default approach uses BERTopic's centrality metrics to select the most representative documents.

### 3.4 Integration with BERTopic

#### 3.4.1 Model Loading

The pipeline supports loading BERTopic models from:
- **Pickle format**: Wrapped `RetrainableBERTopicModel` objects (`.pkl` files)
- **Native format**: BERTopic safetensors models (directory structure)

#### 3.4.2 Metadata Storage

Topic metadata is stored in BERTopic's `topic_metadata_` attribute with structure:

```python
{
  topic_id: {
    "label": "Makeout In Parked Car",
    "keywords": ["car", "seat", "door", "parked", "kiss"],
    "scene_summary": "...",
    "primary_categories": ["romance_core", "sexual_content"],
    "secondary_categories": ["setting:car", "activity:kissing"],
    "main_category_id": "4.2",
    "main_category_name": "Bonding, Everyday Intimacy & Growth",
    "main_category_group": "Relationship Trajectory (Main Couple)",
    "radway_functions": {
      "radway_main_id": "R8",
      "radway_phase": "II",
      ...
    },
    "is_noise": false,
    "rationale": "..."
  }
}
```

#### 3.4.3 Topic Assignment Integration

The labeled and categorized topics are used for:
- **Sentence-level topic assignment**: Each sentence in the corpus is assigned to a topic
- **Book-level aggregation**: Topic proportions aggregated by book for statistical analysis
- **Category-level aggregation**: Taxonomy categories aggregated by book for hypothesis testing

---

## 4. Prompt Concepts and Design Patterns

### 4.1 Domain Adaptation

**Romance-Aware Prompting**: The system prompt explicitly establishes the domain (romantic and erotic fiction) to:
- Prevent generic topic labeling heuristics
- Accept explicit sexual terminology in research context
- Understand genre-specific patterns (e.g., "HEA" = happily ever after)

**Context Hints**: Adaptive hints based on detected domains (body parts, food/drink, time spans, etc.) guide the model toward appropriate label specificity.

### 4.2 Constraint-Based Design

**Hard Constraints**: Explicit prohibitions ("Do NOT use X unless Y") work better than soft guidance ("prefer not to...") because:
- Creative models often ignore soft guidance
- Hard constraints with conditions prevent over-correction
- Model can still use X when condition Y is met

**Format Enforcement**: JSON output structure with strict schema validation ensures:
- Consistent data structure across all topics
- Programmatic parsing without regex extraction
- Error detection when models violate format

### 4.3 Evidence Hierarchy

**Snippets > Keywords**: Explicit instruction to "trust snippets over keywords when they disagree" prevents hallucination from keyword ambiguity.

**Priority Encoding**: Clear hierarchy (Action → Role → Setting → Tone) guides what information to include in labels, preventing information overload.

### 4.4 Disambiguation Strategies

**Explicit Distinctions**: Force model to encode distinguishing features:
- "Physical Violence and Rage" vs "Silent Emotional Resentment"
- "Gentle Comforting Kisses" vs "Rough Angry Kisses"

**Label Uniqueness**: Prevent reuse of vague labels across distinct topics.

### 4.5 Uncertainty Handling

**Confidence Scores**: Low/medium/high confidence ratings for taxonomy and Radway mappings enable:
- Quality filtering in downstream analysis
- Identification of ambiguous topics for manual review
- Weighted analysis based on confidence

**"None" as Last Resort**: For taxonomy and Radway mappings, "none" is explicitly framed as a last resort, with clear criteria for when it's appropriate.

---

## 5. Quality Assurance and Validation

### 5.1 Model Comparison

We compared multiple models on a subset of 30 topics to evaluate:
- **Success Rate**: Percentage of topics receiving multi-word labels (vs single-word keyword copies)
- **Average Words per Label**: Should be 2-6 words
- **Keyword Copying**: Percentage of topics where model just copied the first keyword
- **Label Quality**: Manual inspection of descriptive, meaningful labels

**Results**: 
- `mistralai/mistral-nemo`: 100% success rate, 2.30 avg words, 0% keyword copying ✅
- `mistralai/mistral-7b-instruct:free`: 93.3% success rate, 2.80 avg words, 6.7% keyword copying ✅
- `venice/uncensored:free`: 76.7% success rate, 2.73 avg words, 23.3% keyword copying ✅
- `x-ai/grok-4.1-fast`: 0% success rate, 1.00 avg words, 100% keyword copying ❌
- `deepseek/deepseek-chat-v3-0324`: 0% success rate, 1.00 avg words, 100% keyword copying ❌

### 5.2 Coverage Metrics

**Taxonomy Coverage**: 361 out of 368 topics (98.1%) successfully mapped to taxonomy categories.

**Radway Coverage**: Varies by topic type; topics in "Relationship Trajectory (Main Couple)" group should have near-100% coverage (not "none").

### 5.3 Manual Review Process

**Flagged Topics**: Topics with low confidence, "none" mappings, or unusual patterns are flagged for manual review.

**Validation Examples**: Representative examples from each category are reviewed to ensure mappings align with theoretical definitions.

---

## 6. Limitations and Future Directions

### 6.1 Current Limitations

1. **Snippet Selection**: Relies on BERTopic's representative doc selection, which may miss important variations
2. **Hallucination**: Despite constraints, models may still infer details not present in input
3. **Coverage**: Not all topics successfully map to taxonomy/Radway (98.1% taxonomy coverage)
4. **Cost**: API costs scale with number of topics (acceptable for research but not free)
5. **Latency**: Sequential processing means 368 topics take ~6-18 minutes

### 6.2 Future Improvements

1. **Diversity-Aware Snippet Selection**: Use MMR or similar to ensure snippet diversity
2. **Adaptive Snippet Count**: More snippets for complex topics, fewer for simple ones
3. **Parallel Processing**: Process multiple topics concurrently to reduce latency
4. **Few-Shot Example Selection**: Dynamic few-shot selection based on topic similarity
5. **Automated Quality Metrics**: Label length, uniqueness, keyword coverage, snippet alignment
6. **Model-Specific Prompt Tuning**: Optimize prompts for different model families

---

## 7. Conclusion

This methodology demonstrates how LLM-based labeling and zero-shot classification can scale topic interpretation and theoretical mapping for large-scale text analysis. Key innovations include:

- **Snippet-integrated prompting** for scene-level understanding
- **Anti-hallucination constraints** based on empirical testing
- **Structured JSON output** for programmatic analysis
- **Unified data structure** combining labels, taxonomy, and Radway mappings
- **Theory-driven classification** enabling connection to established frameworks

The approach balances **research reliability** (low hallucination, literal interpretation) with **practical efficiency** (API-based, structured output, fault tolerance), making it suitable for large-scale computational literary analysis.

---

## References

- Radway, J. (1984). *Reading the Romance: Women, Patriarchy, and Popular Literature*. University of North Carolina Press.
- BERTopic Documentation: https://maartengr.github.io/BERTopic/
- OpenRouter API Documentation: https://openrouter.ai/docs
- Mistral AI Models: https://mistral.ai/

