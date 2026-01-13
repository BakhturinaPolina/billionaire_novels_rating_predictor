# Duplicate Labels Detection and Disambiguation Report

**Date:** December 12, 2025  
**Model:** `model_1_with_llm_labels_and_metadata`  
**Stage:** Stage 1 Natural Clusters - Pre-hierarchical Analysis

## Executive Summary

During validation of the topic assignment pipeline, we detected **19 duplicate labels** across 369 labeled topics in the BERTopic model. All duplicates were successfully disambiguated by appending topic IDs to ensure unique label identification.

## Duplicate Labels Found

### Statistics
- **Total topics with labels:** 369
- **Duplicate labels:** 19
- **Most problematic:** 'Unclear Relationship Feelings' used by 7 topics

### Complete List of Duplicates

| Label | Number of Topics | Topic IDs |
|-------|------------------|-----------|
| Unclear Relationship Feelings | 7 | [4, 8, 50, 52, 87, 107, 172] |
| Deep Breaths During Emotional Moment | 3 | [48, 213, 291] |
| Sister's Confusing Admissions | 3 | [104, 130, 134] |
| Argument About Saying No | 2 | [32, 300] |
| Breast And Nipple Play | 2 | [133, 290] |
| Business Deal Negotiation | 2 | [195, 277] |
| Casual Conversations | 2 | [143, 159] |
| Clitoral Stimulation During Foreplay | 2 | [2, 15] |
| Eye Contact And Staring | 2 | [20, 38] |
| Hockey Game Action | 2 | [30, 72] |
| Hurried Conversations | 2 | [105, 141] |
| Intense Argument With Glare | 2 | [315, 335] |
| Office Work At Desk | 2 | [212, 316] |
| Playful Laughter And Smiles | 2 | [13, 25] |
| Scent-induced Arousal | 2 | [207, 257] |
| Shower Hygiene | 2 | [57, 292] |
| Silent Staring Conversation | 2 | [59, 140] |
| Tender Forehead Kisses | 2 | [12, 78] |
| Whispered Secrets Between Lovers | 2 | [119, 155] |

## Disambiguation Process

### Method
Labels were disambiguated by appending topic IDs in the format: `"Original Label (T{topic_id})"`

**Example:**
- Before: `"Unclear Relationship Feelings"` (used by topics 4, 8, 50, 52, 87, 107, 172)
- After:
  - `"Unclear Relationship Feelings (T4)"`
  - `"Unclear Relationship Feelings (T8)"`
  - `"Unclear Relationship Feelings (T50)"`
  - `"Unclear Relationship Feelings (T52)"`
  - `"Unclear Relationship Feelings (T87)"`
  - `"Unclear Relationship Feelings (T107)"`
  - `"Unclear Relationship Feelings (T172)"`

### Verification
After disambiguation:
- ✓ All 369 labels are now unique
- ✓ No duplicate labels remain
- ✓ Model saved with disambiguated labels

## Saved Model

**Location:** `models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/`

**Files:**
- Native BERTopic format: `model_1_with_llm_labels_and_metadata_disambiguated/`
- Pickle wrapper: `model_1_with_llm_labels_and_metadata_disambiguated.pkl`

## Impact on Analysis

### Why This Matters
1. **Hierarchical Visualization:** Duplicate labels can cause confusion in dendrogram visualizations where multiple topics share the same name
2. **Topic Identification:** Unique labels ensure clear topic identification in downstream analysis
3. **Metadata Consistency:** Disambiguated labels maintain consistency with topic metadata

### No Impact on Topic Assignments
- Topic assignments to sentences remain unchanged
- Only label text was modified for clarity
- Topic IDs and embeddings are unaffected

## Recommendations

1. **Use Disambiguated Model:** The disambiguated model should be used for all hierarchical and visualization tasks
2. **Review High-Duplication Topics:** Consider investigating why 'Unclear Relationship Feelings' appears in 7 different topics - this may indicate:
   - Overly broad label definition
   - Topics that could potentially be merged
   - Need for more specific labeling

## Scripts Used

- **Detection:** `check_duplicate_labels.py --model-suffix _with_llm_labels_and_metadata --model-stage stage08_llm_labeling/with_metadata`
- **Disambiguation:** `check_duplicate_labels.py --model-suffix _with_llm_labels_and_metadata --model-stage stage08_llm_labeling/with_metadata --disambiguate --save-model`

## Next Steps

The disambiguated model is ready for:
1. Hierarchical topic exploration (`explore_hierarchical_topics.py`)
2. Meta-topic reduction
3. Category mapping analysis

---

**Status:** ✓ Complete - All duplicates resolved and model saved

