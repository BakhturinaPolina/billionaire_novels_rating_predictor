# Stage 3: Radway Narrative Functions — Research Report

## Purpose

Map BERTopic topics to Radway's 13 narrative functions to analyze story structure and compare narrative patterns between books with different ratings.

## Research Rationale

### Why Radway's Narrative Functions?

Janice Radway's *Reading the Romance* (1984) identified 13 narrative functions that structure romance novels into three phases:

- **Phase I**: Initial Conflict & Isolation (setup, tension)
- **Phase II**: Turning Point & Recognition (empathy, connection)
- **Phase III**: Commitment & Restoration (happy ending)

Mapping topics to these functions enables:
1. Quantitative analysis of narrative structure across a large corpus
2. Comparison of narrative patterns between rating tiers
3. Testing hypotheses about what makes romance novels successful

---

## Radway's 13 Functions

### Phase I: Initial Conflict & Isolation

| ID | Function |
|----|----------|
| R1 | Heroine's social identity is destroyed |
| R2 | Heroine reacts antagonistically to the hero |
| R3 | Hero responds ambiguously to heroine |
| R4 | Heroine interprets hero's behaviour as purely sexual interest |
| R5 | Heroine responds with anger or coldness |
| R6 | Hero retaliates or punishes heroine |
| R7 | Hero and heroine are physically or emotionally separated |

### Phase II: Turning Point & Recognition

| ID | Function |
|----|----------|
| R8 | Hero treats heroine tenderly |
| R9 | Heroine responds warmly to hero's tenderness |
| R10 | Heroine reinterprets hero's behaviour as result of previous hurt |

### Phase III: Commitment & Restoration

| ID | Function |
|----|----------|
| R11 | Hero declares love and demonstrates commitment |
| R12 | Heroine responds sexually and emotionally |
| R13 | Heroine's identity is restored |

---

## Methodology

### Zero-Shot Classification

Each topic is classified using:
- Topic keywords from BERTopic
- LLM-generated labels and scene summaries (Stage 08)
- Stage 2 taxonomy classifications
- Optional representative document snippets

The LLM prompt includes:
- Interpretation hints linking taxonomy groups to Radway functions
- Disambiguation rules for common confusions
- Gated "none" decision process

### Quality Improvements

**Heuristic overrides** correct systematic LLM errors:

| Issue | Correction |
|-------|------------|
| Explicit sex (2.3) → R4 | Override to R12 |
| Wedding/marriage → none | Override to R11/R13 |
| Arguments → R7 | Only use R7 for actual separation |
| Romance-core → none | Prevent false negatives |

**Deterministic decoding** (temperature=0.0) ensures consistent results.

---

## Results

### Overall Statistics

| Metric | Value |
|--------|-------|
| Total topics | 368 |
| Topics with Radway mappings | 361 (98.1%) |
| Topics mapped to functions (R1-R13) | 272 (75.3%) |
| Topics classified as "none" | 96 (26.6%) |
| Unique Radway functions used | 13 (all represented) |
| High confidence classifications | 130 (36%) |

### Distribution by Phase

| Phase | Topics | % of Function Topics |
|-------|--------|---------------------|
| Phase I: Conflict & Isolation | 147 | 54.0% |
| Phase II: Turning Point | 96 | 35.3% |
| Phase III: Commitment | 28 | 10.3% |

**Key Finding**: Phase I (conflict, tension) dominates, representing over half of narrative function topics. Phase III (commitment, resolution) occupies least textual space despite narrative importance.

### Taxonomy-to-Radway Mapping Patterns

**Strong mappings:**

| Taxonomy | Radway | Pattern |
|----------|--------|---------|
| 2.3 (Explicit Sex) | R12 | Phase III sexual/emotional response |
| 4.5 (HEA, Commitments) | R11/R13 | Phase III resolution |
| 4.4 (Conflict, Breakup) | R2/R7 | Phase I tension |
| 4.2 (Bonding, Intimacy) | R8/R9 | Phase II tenderness |

**Weak mappings (mostly "none"):**
- 5.x: Social World Outside Couple
- 6.x: Work, Wealth, Status
- 8.x: Spaces, Time, Activities

### "None" Classifications

96 topics (26.6%) were classified as "none" — not corresponding to Radway's narrative functions:
- Background/contextual content
- Social world outside the couple
- Setting and environmental descriptions
- Internal states not advancing narrative function

This is expected: not all textual content serves narrative function purposes.

---

## Key Research Questions

The Radway mappings enable testing:

1. **Do highly-rated books follow Radway's structure more closely?**
2. **Which narrative phases are most associated with high ratings?**
3. **How does narrative arc differ by quality tier?**

---

## Limitations

1. **Ambiguous topics**: Some topics span multiple phases or functions
2. **Context dependency**: Classifications may miss narrative context
3. **Single classification**: Each topic gets one primary function, though many serve multiple
4. **Zero-shot**: May miss nuanced distinctions requiring domain expertise

---

## Integration with Pipeline

| Stage | Relationship |
|-------|--------------|
| ← Stage 2 | Uses taxonomy classifications as context |
| → Stage 10 | Radway function proportions used in correlation analysis |

---

## References

- Radway, J. A. (1984). *Reading the Romance: Women, Patriarchy, and Popular Literature*. University of North Carolina Press.
