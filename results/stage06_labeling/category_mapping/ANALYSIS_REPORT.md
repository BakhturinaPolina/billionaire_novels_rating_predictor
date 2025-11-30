# Category Mapping Output Analysis Report

**Generated:** Analysis of topic-to-category mapping outputs  
**Files Analyzed:**
- `topic_to_category_probs.json` (361 topics)
- `topic_to_category_final.csv` (449 topic-category pairs)
- `topic_to_category_summary.csv` (361 topics)

---

## Executive Summary

✅ **Overall Quality: GOOD**
- All 361 topics successfully mapped to categories
- Weight normalization working correctly (all sums = 1.0)
- Good separation between explicit (C) and mutual intimacy (B) categories
- 20.5% of topics have multi-category assignments (appropriate soft classification)

⚠️ **Areas of Concern:**
- 32.7% of topics mapped to `Z_noise_oog` (noise category) - may need pattern refinement
- Some topics have very low primary weights (< 0.5) indicating ambiguous classification
- R category (Protectiveness vs Jealousy) has very low representation (6 topics total)

---

## 1. Basic Statistics

- **Total Topics:** 361
- **Total Topic-Category Pairs:** 449
- **Average Categories per Topic:** 1.24
- **Multi-category Topics:** 74 (20.5%)
- **Single-category Topics:** 287 (79.5%)

### Distribution of Category Counts:
- 1 category: 287 topics (79.5%)
- 2 categories: 61 topics (16.9%)
- 3 categories: 12 topics (3.3%)
- 4 categories: 1 topic (0.3%)

---

## 2. Category Distribution

### Primary Category Frequency (Top 15)

| Category | Count | Percentage |
|----------|-------|------------|
| Z_noise_oog | 118 | 32.7% |
| B_mutual_intimacy | 26 | 7.2% |
| J_family_support | 25 | 6.9% |
| O_appearance_aesthetics | 24 | 6.6% |
| K_work_corporate | 23 | 6.4% |
| S_scene_anchor | 17 | 4.7% |
| F_negative_affect | 17 | 4.7% |
| P_tech_media | 15 | 4.2% |
| H_domestic_nesting | 14 | 3.9% |
| I_humor_lightness | 13 | 3.6% |
| Q_miscommunication | 12 | 3.3% |
| A_commitment_hea | 11 | 3.0% |
| D_luxury_wealth_status | 8 | 2.2% |
| E_threat_danger | 8 | 2.2% |
| G_rituals_gifts | 8 | 2.2% |

### Total Weight Distribution

The total weight distribution shows similar patterns, with Z_noise_oog dominating at 32.7% of total weight. This suggests that many topics couldn't be matched to specific theoretical categories.

---

## 3. Multi-Category Mappings

**74 topics (20.5%)** have multiple category assignments, which is appropriate for soft classification. Examples:

- **Topic 0** "True Promise": `A_commitment_hea(0.50) + K_work_corporate(0.50)`
- **Topic 4** "Long-Awaited Reunion": `A_commitment_hea(0.50) + N_separation_reunion(0.50)`
- **Topic 8** "Wrong Thoughts": `Q_miscommunication(0.50) + F_negative_affect(0.50)`
- **Topic 9** "Wedding Planning": `A_commitment_hea(0.50) + G_rituals_gifts(0.50)`
- **Topic 11** "Wine Service": `H_domestic_nesting(0.50) + D_luxury_wealth_status(0.50)`

### Most Common Category Co-occurrences:
1. `K_work_corporate + P_tech_media`: 5 topics
2. `A_commitment_hea + N_separation_reunion`: 4 topics
3. `I_humor_lightness + O_appearance_aesthetics`: 4 topics
4. `Q_miscommunication + S_scene_anchor`: 3 topics
5. `J_family_support + M_health_recovery`: 3 topics

---

## 4. Quality Validation

### ✅ Weight Normalization
- All topic weights sum to exactly 1.0
- Min weight: 0.25, Max weight: 1.0, Mean: 0.80
- Proper soft classification implementation

### ✅ Category Separation
- **Explicit (C) vs Mutual Intimacy (B):** No overlap (0 topics with both > 0.5)
  - Explicit topics (C > 0.5): 7 topics
  - Mutual intimacy topics (B > 0.5): 23 topics
- Good separation validates the Love-over-Sex hypothesis operationalization

### ✅ Topic ID Consistency
- All three output files have consistent topic IDs (361 topics each)

---

## 5. Noise Category Analysis

**32.7% of topics** (118 topics) mapped to `Z_noise_oog`, indicating:
- Many topics don't match existing category patterns
- May need pattern refinement or new category definitions
- Some examples of Z_noise topics:
  - "True Choice"
  - "Nighttime Stand"
  - "Guacamole Dilemma"
  - "Ice Hockey Players"
  - "Tomorrow's Outfit"
  - "Hesitant Intercourse"
  - "Moist Mouth Play"
  - "Grand Century Knock"
  - "Purring Rescue Kittens"
  - "Animal Encounter"

**Recommendation:** Review Z_noise topics to identify:
1. Patterns that could be captured by existing categories (pattern refinement needed)
2. New category definitions needed
3. Truly noise topics that should remain unclassified

---

## 6. Low-Confidence Mappings

**13 topics** have primary category weights < 0.5, indicating ambiguous classification:

Examples:
- **Topic 67** "Aisle Stares": Primary `I_humor_lightness(0.25)` with 4 categories total
- **Topic 59** "Playful Banter": Primary `I_humor_lightness(0.33)` with 3 categories
- **Topic 93** "Shameless Apologies": Primary `J_family_support(0.33)` with 3 categories
- **Topic 113** "Tabloid Scandal": Primary `J_family_support(0.33)` with 3 categories
- **Topic 123** "Police Encounter": Primary `K_work_corporate(0.33)` with 3 categories

**Recommendation:** Review these topics manually to ensure appropriate categorization.

---

## 7. Category-Specific Analysis

### R Category (Protectiveness vs Jealousy)
- **R1_protectiveness:** 5 topics (1.4%)
- **R2_jealousy:** 1 topic (0.3%)
- **Total R representation:** Very low (6 topics, 1.7%)

**Observation:** The R category is under-represented. This may indicate:
- Protectiveness/jealousy themes are less common in topic labels
- Pattern matching may need refinement for these categories
- These themes may be embedded in other categories

### Q Category (Miscommunication vs Repair)
- **Q_miscommunication:** 13 topics (3.6%)
- **T_repair_apology:** 5 topics (1.4%)
- **Ratio:** 2.6:1 (miscommunication to repair)

**Observation:** More miscommunication than repair topics, which aligns with narrative arc expectations (conflict before resolution).

---

## 8. Major Category Coverage

All major categories (A-P, Q) have at least some representation:

| Category | Topics | Percentage | Status |
|----------|--------|------------|--------|
| A_commitment_hea | 15 | 4.2% | ✓ |
| B_mutual_intimacy | 34 | 9.4% | ✓ |
| C_explicit | 10 | 2.8% | ✓ |
| D_luxury_wealth_status | 10 | 2.8% | ✓ |
| E_threat_danger | 13 | 3.6% | ✓ |
| F_negative_affect | 27 | 7.5% | ✓ |
| G_rituals_gifts | 12 | 3.3% | ✓ |
| H_domestic_nesting | 15 | 4.2% | ✓ |
| I_humor_lightness | 13 | 3.6% | ✓ |
| J_family_support | 27 | 7.5% | ✓ |
| K_work_corporate | 39 | 10.8% | ✓ |
| Q_miscommunication | 13 | 3.6% | ✓ |

**Note:** These counts include topics where the category appears as primary OR secondary.

---

## 9. Potential Issues

### Topics with Very Short Labels/Keywords
9 topics have very short labels (< 10 chars) or few keywords (< 3):
- Topic 141: "Rush of Passion" (1 keyword) → Z_noise_oog
- Topic 183: "Divided Hearts" (1 keyword) → Z_noise_oog
- Topic 241: "First Touch" (1 keyword) → Z_noise_oog
- Topic 262: "Clever Banter" (1 keyword) → I_humor_lightness
- Topic 347: "Passionate Affirmation" (1 keyword) → Z_noise_oog

**Recommendation:** These may benefit from manual review or pattern refinement.

---

## 10. Recommendations

### Immediate Actions
1. **Review Z_noise_oog topics:** 32.7% is high - identify patterns for refinement
2. **Review low-confidence mappings:** 13 topics with primary weight < 0.5
3. **Review short-label topics:** 9 topics may need pattern refinement

### Pattern Refinement
1. **Add patterns for common Z_noise topics** that could map to existing categories
2. **Refine R category patterns** (Protectiveness/Jealousy) - currently under-represented
3. **Consider splitting Z_noise** into subcategories if patterns emerge

### Validation
1. **Manual review sample** of multi-category topics to validate co-occurrence logic
2. **Cross-check** with theoretical framework to ensure category definitions are appropriate
3. **Validate** that Love-over-Sex hypothesis can be properly tested with current mappings

---

## 11. Data Quality Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Topics mapped | 361/361 (100%) | ✅ |
| Weight normalization | 100% (all sum to 1.0) | ✅ |
| Category separation (B vs C) | 0 overlap | ✅ |
| File consistency | All files match | ✅ |
| Noise rate | 32.7% | ⚠️ |
| Low-confidence mappings | 3.6% | ⚠️ |
| Multi-category rate | 20.5% | ✅ |

---

## Conclusion

The category mapping output is **functionally correct** with proper weight normalization and good separation between key categories (B vs C). However, the high noise rate (32.7%) suggests that pattern matching could be refined to capture more topics in theory-aligned categories.

The mapping successfully operationalizes the theoretical framework with appropriate soft classification (20.5% multi-category topics), enabling downstream hypothesis testing while maintaining semantic nuance.

**Next Steps:**
1. Review and refine patterns for Z_noise topics
2. Validate low-confidence mappings
3. Proceed to book-level aggregation and index computation

