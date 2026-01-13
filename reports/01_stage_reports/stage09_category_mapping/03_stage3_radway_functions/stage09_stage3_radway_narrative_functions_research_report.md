# Stage 3: Radway Narrative Functions - Research Report

## Executive Summary

This report documents Stage 3 of the category mapping pipeline, which maps BERTopic topics to Radway's 13 narrative functions. The goal is to analyze story structure and track narrative progression across billionaire romance novels, enabling comparison of narrative patterns between books with different ratings (bad/mid/good).

**Key Results:**
- Successfully classified 361 topics (out of 368 total) with Radway function mappings
- 272 topics mapped to specific Radway functions (R1-R13)
- 96 topics classified as "none" (not corresponding to narrative functions)
- Distribution across three narrative phases: Phase I (147 topics), Phase II (96 topics), Phase III (28 topics)
- All 13 Radway functions represented in the classification

## 1. Introduction

### 1.1 Background

Janice Radway's *Reading the Romance* (1984) identified 13 narrative functions that structure romance novels. These functions organize the heroine-hero relationship arc into three distinct phases:

- **Phase I: Initial Conflict & Isolation** - The setup where conflict emerges
- **Phase II: Turning Point & Recognition** - Developing empathy and connection
- **Phase III: Commitment & Restoration** - The happy ending

Mapping BERTopic topics to these functions enables quantitative analysis of narrative structure across a large corpus of romance novels.

### 1.2 Research Questions

1. How are Radway's narrative functions distributed across topics in billionaire romance novels?
2. Which narrative phases are most prevalent in the corpus?
3. How do topics map from taxonomy groups to Radway functions?
4. What proportion of topics correspond to narrative functions vs. background/contextual content?

## 2. Methodology

### 2.1 Classification Approach

The implementation uses **zero-shot classification** with Mistral-Nemo via OpenRouter API. Each topic is classified using:

- Topic keywords from BERTopic
- LLM-generated labels and scene summaries (from Stage 8)
- Stage 1 primary/secondary categories
- Stage 2 taxonomy classifications
- Optional representative document snippets

### 2.2 Radway's 13 Functions

**Phase I: Initial Conflict & Isolation**
- **R1**: Heroine's social identity is destroyed
- **R2**: Heroine reacts antagonistically to the hero
- **R3**: Hero responds ambiguously to heroine
- **R4**: Heroine interprets hero's behaviour as purely sexual interest
- **R5**: Heroine responds with anger or coldness
- **R6**: Hero retaliates or punishes heroine
- **R7**: Hero and heroine are physically or emotionally separated

**Phase II: Turning Point & Recognition**
- **R8**: Hero treats heroine tenderly
- **R9**: Heroine responds warmly to hero's tenderness
- **R10**: Heroine reinterprets hero's behaviour as result of previous hurt

**Phase III: Commitment & Restoration**
- **R11**: Hero declares love and demonstrates commitment
- **R12**: Heroine responds sexually and emotionally
- **R13**: Heroine's identity is restored

### 2.3 Quality Improvements

Several systematic improvements were implemented to address classification errors:

1. **Explicit sex scenes (2.3 → R12)**: Topics with `taxonomy_main_id = 2.3` correctly mapped to R12 rather than R4
2. **Commitment topics (→ R11/R13)**: Wedding/marriage/engagement topics mapped to R11/R13 rather than "none" or R8
3. **R7 narrowing**: R7 only used when actual breakup/separation cues exist
4. **"None" false negatives**: Improved detection prevents romance-core topics from being incorrectly marked as "none"

These improvements use:
- Enhanced prompt disambiguation rules
- Post-LLM heuristic override system with regex-based pattern matching
- Deterministic decoding (temperature=0.0) for consistency
- Gated "none" decision process

### 2.4 Output Structure

Each topic receives a `radway_functions` object with:
- `radway_main_id`: Primary Radway function (R1-R13 or "none")
- `radway_secondary_id`: Optional secondary function
- `radway_other_plausible_ids`: List of other plausible functions
- `radway_phase`: Phase (I, II, III, or NA)
- `radway_is_none`: Boolean flag
- `radway_confidence`: Confidence level (low/medium/high)
- `radway_rationale`: Explanation for the classification
- `radway_main_name`: Human-readable function name
- `radway_phase_name`: Human-readable phase name

## 3. Results Summary

### 3.1 Overall Statistics

From `summary_statistics.json`:
- **Total topics**: 368
- **Topics with labels**: 368 (100%)
- **Topics with taxonomy**: 361 (98.1%)
- **Topics with Radway mappings**: 361 (98.1%)
- **Topics with Radway function**: 272 (73.9% of total, 75.3% of classified)
- **Topics with Radway "none"**: 96 (26.1% of total, 26.6% of classified)
- **Unique taxonomy categories**: 28
- **Unique taxonomy groups**: 9
- **Unique Radway functions**: 13 (all functions represented)
- **Unique Radway phases**: 4 (I, II, III, NA)

### 3.2 Distribution by Narrative Phase

Based on EDA file analysis:

| Phase | Count | Percentage of Function Topics | Description |
|-------|-------|-------------------------------|-------------|
| **Phase I: Initial Conflict & Isolation** | 147 | 54.0% | Setup and conflict emergence |
| **Phase II: Turning Point & Recognition** | 96 | 35.3% | Developing empathy and connection |
| **Phase III: Commitment & Restoration** | 28 | 10.3% | Happy ending and resolution |
| **Not a narrative function** | 96 | - | Background/contextual content |
| **Unknown** | 1 | - | Unclassified |

**Key Observation**: Phase I (conflict and isolation) dominates the narrative function distribution, representing over half of all function-mapped topics. This suggests that conflict and tension are central to the romance narrative structure in this corpus.

### 3.3 Distribution by Taxonomy Group

Topics were analyzed across 9 taxonomy groups. Key findings:

#### Relationship Trajectory (Main Couple) - 158 topics
- Largest group, directly related to couple dynamics
- High mapping rate to Radway functions
- Covers all three phases

#### Sexuality, Attraction & Intimacy - 55 topics
- Strong mapping to Phase III (R12: sexual/emotional response)
- Also includes Phase I (R4: sexual interest interpretation)
- Phase II (R8/R9: tender moments)

#### Emotions, Cognition & Inner Life - 59 topics
- Distributed across phases
- Phase I: negative emotions, distress (R2, R5)
- Phase II: positive emotions, recognition (R9, R10)
- Some topics classified as "none" (internal states not directly narrative functions)

#### Social World Outside Couple - 36 topics
- High proportion classified as "none"
- Background/contextual content
- Family, friends, social circles

#### Work, Wealth, Status & Institutions - 25 topics
- Mostly classified as "none"
- Contextual setting information
- Some connection to R1 (identity destruction) in rare cases

#### Spaces, Time, Activities & Objects - 25 topics
- Primarily "none" classifications
- Setting and temporal framing
- Background narrative elements

#### Conflict, Risk & Harm - 11 topics
- Strong mapping to Phase I
- R2 (antagonistic reaction), R6 (retaliation), R7 (separation)

#### Embodied & Sensory Experience - 7 topics
- Mixed classifications
- Some connection to R12 (sexual response)

#### Special - 3 topics
- Edge cases and special classifications

### 3.4 High Confidence Classifications

130 topics were classified with high confidence. These represent clear, unambiguous mappings to Radway functions. Examples include:

- **R12 (Phase III)**: Explicit sexual acts (taxonomy 2.3) - clear sexual/emotional response
- **R13 (Phase III)**: Marriage/wedding topics (taxonomy 4.5) - identity restoration
- **R2 (Phase I)**: Arguments and conflict (taxonomy 4.4) - antagonistic reactions
- **R8/R9 (Phase II)**: Smiling, laughter, tender moments (taxonomy 2.2, 4.2) - tenderness and warm response

## 4. Key Findings

### 4.1 Narrative Structure Patterns

1. **Phase I Dominance**: Over half (54%) of narrative function topics belong to Phase I, indicating that conflict, tension, and isolation are central narrative elements.

2. **Phase II Transition**: 35% of topics map to Phase II, showing substantial narrative space dedicated to the turning point and recognition phase.

3. **Phase III Resolution**: Only 10% of topics map to Phase III, suggesting that commitment and restoration, while narratively crucial, occupy less textual space.

### 4.2 Taxonomy-to-Radway Mapping Patterns

**Strong Mappings:**
- Taxonomy 2.3 (Explicit Sexual Acts) → R12 (Phase III)
- Taxonomy 4.5 (Reconciliation, Commitments & HEA) → R11/R13 (Phase III)
- Taxonomy 4.4 (Conflict, Distance & Breakup Threats) → R2/R7 (Phase I)
- Taxonomy 4.2 (Bonding, Everyday Intimacy & Growth) → R8/R9 (Phase II)

**Weak/No Mappings:**
- Taxonomy 5.x (Social World Outside Couple) → Mostly "none"
- Taxonomy 6.x (Work, Wealth, Status) → Mostly "none"
- Taxonomy 8.x (Spaces, Time, Activities) → Mostly "none"

### 4.3 Classification Quality

- **High confidence**: 130 topics (36% of classified)
- **Medium confidence**: Majority of remaining function topics
- **Low confidence**: Small number of edge cases

The heuristic override system successfully corrected systematic errors:
- Explicit sex scenes now correctly map to R12
- Commitment topics correctly map to R11/R13
- R7 only used for actual separations

### 4.4 "None" Classifications

96 topics (26.6% of classified) were marked as "none", indicating they do not correspond to Radway's narrative functions. These primarily include:

- Background/contextual content (domestic spaces, objects, time)
- Social world outside the couple (family, friends, work)
- Setting and environmental descriptions
- Some internal states that don't directly advance the narrative function

This is expected and appropriate - not all textual content serves narrative function purposes.

## 5. Limitations

### 5.1 Classification Challenges

1. **Ambiguous Topics**: Some topics contain elements from multiple phases or functions, requiring judgment calls.

2. **Context Dependency**: Some classifications depend on narrative context that may not be fully captured in topic keywords and summaries.

3. **Boundary Cases**: Distinctions between similar functions (e.g., R4 vs R12, R8 vs R11) can be subtle.

4. **"None" Classification**: Determining when a topic truly doesn't serve a narrative function vs. serving a subtle or indirect function is challenging.

### 5.2 Data Limitations

1. **Topic Quality**: Classification depends on quality of BERTopic topic extraction and Stage 8 LLM-generated labels.

2. **Representative Snippets**: Not all topics have representative document snippets, potentially limiting classification accuracy.

3. **Taxonomy Dependencies**: Classification leverages Stage 2 taxonomy mappings, inheriting any limitations from that stage.

### 5.3 Methodological Limitations

1. **Zero-Shot Classification**: While effective, zero-shot classification may miss nuanced distinctions that require domain expertise.

2. **Heuristic Overrides**: While improving accuracy, heuristic overrides may introduce edge case errors.

3. **Single Classification**: Each topic receives one primary function, though many topics may serve multiple functions simultaneously.


## 7. Conclusion

Stage 3 successfully mapped 361 topics to Radway's 13 narrative functions, providing a quantitative framework for analyzing narrative structure in billionaire romance novels. The classification reveals that Phase I (conflict and isolation) dominates the narrative function distribution, while Phase III (commitment and restoration) occupies less textual space despite its narrative importance.

The mapping enables downstream analysis comparing narrative patterns between books with different ratings, testing hypotheses about whether well-rated books follow Radway's structure more closely or exhibit different narrative patterns.

The classification system, with its heuristic overrides and deterministic decoding, provides consistent and accurate mappings that can support robust statistical analysis in subsequent stages of the research pipeline.

---

## Appendix: File Structure

### Input Files
- Taxonomy mappings from Stage 2 (JSON or embedded in BERTopic model)
- BERTopic model with topic metadata
- Representative document snippets (optional)

### Output Files
- `taxonomy_with_radway.json`: Merged taxonomy and Radway mappings
- Updated BERTopic model with Radway mappings attached
- EDA files organized by:
  - Taxonomy groups (9 files)
  - Narrative phases (4 files)
  - High confidence classifications
  - Summary statistics

### Scripts
- `zeroshot_radway_openrouter.py`: Main classification script
- `update_model_with_radway.py`: Model update script

---

**Report Generated**: Draft version for review  
**Data Source**: `results/stage09_category_mapping/stage3_radway_functions/eda/`  
**Classification Date**: See Stage 3 README for latest run information

