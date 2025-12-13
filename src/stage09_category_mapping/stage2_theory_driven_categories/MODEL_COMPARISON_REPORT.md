# Model Comparison Report

**Generated:** December 13, 2025  
**Purpose:** Compare models from Stage 08 (LLM Labeling) and Stage 09 (Category Mapping) to determine the most complete model for taxonomy classification and statistical analysis.

---

## Executive Summary

This report compares BERTopic models from two pipeline stages to identify the most comprehensive model for downstream analysis. The comparison evaluates:

- **Number of topics** (should be consistent across models)
- **Presence of LLM-generated labels** (from Stage 08)
- **Presence of taxonomy mappings** (from Stage 09)
- **Model completeness** for statistical analysis

### Key Finding

✅ **Recommended Model:** `model_1_with_llm_labels_and_metadata_disambiguated.pkl` from `stage09_category_mapping`

This model is the **most complete** because it contains:
- 368 topics (full coverage)
- 369 LLM-generated labels
- 361 taxonomy mappings embedded in the model

---

## Stage 08 Models (LLM Labeling)

Stage 08 models contain BERTopic topics with LLM-generated labels but **no taxonomy mappings**.

| Model Name | File Size (MB) | Topics | Labels | Taxonomy Mappings | Status |
|------------|---------------|--------|--------|-------------------|--------|
| `model_1_with_llm_labels.pkl` | 4,795.2 | 368 | 369 | 0 | ⚠️ Incomplete |
| `model_1_with_llm_labels_thedrummer_cydonia-24b-v4.1.pkl` | 4,795.2 | 368 | 369 | 0 | ⚠️ Incomplete |

### Stage 08 Summary

- **Total models analyzed:** 2
- **Topics per model:** 368 (consistent)
- **Labels per model:** 369 (includes outlier topic -1)
- **Taxonomy mappings:** None (requires Stage 09 processing)

**Limitation:** These models require separate taxonomy mapping JSON files for category-based analysis.

---

## Stage 09 Models (Category Mapping)

Stage 09 models contain BERTopic topics with labels and potentially taxonomy mappings.

| Model Name | File Size (MB) | Topics | Labels | Taxonomy Mappings | Status |
|------------|---------------|--------|--------|-------------------|--------|
| `model_1_with_llm_labels_and_metadata_disambiguated.pkl` | 4,795.7 | 368 | 369 | **361** | ✅ **RECOMMENDED** |
| `model_1_with_llm_labels_disambiguated.pkl` | 4,795.4 | 368 | 369 | 0 | ⚠️ Incomplete |
| `model_1_with_categories.pkl` | 4,795.4 | 368 | 369 | 0 | ⚠️ Incomplete |

### Stage 09 Summary

- **Total models analyzed:** 3
- **Topics per model:** 368 (consistent)
- **Labels per model:** 369 (consistent)
- **Models with taxonomy mappings:** 1 out of 3

**Key Finding:** Only `model_1_with_llm_labels_and_metadata_disambiguated.pkl` has taxonomy mappings embedded (361 out of 368 topics, ~98% coverage).

---

## Taxonomy Mapping Files

The following taxonomy mapping JSON files were found:

| File Name | Topics Mapped | Status |
|-----------|---------------|--------|
| `taxonomy_mappings_test_top10.json` | 10 | 🧪 Test file only |

**Note:** This is a test file with only 10 topics. For full analysis, a complete taxonomy mapping file covering all 368 topics should be generated using `zeroshot_taxonomy_openrouter.py`.

---

## Detailed Model Analysis

### Recommended Model: `model_1_with_llm_labels_and_metadata_disambiguated.pkl`

**Full Path:**
```
models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_llm_labels_and_metadata_disambiguated.pkl
```

**Specifications:**
- **File Size:** 4,795.7 MB (~4.8 GB)
- **Topics:** 368
- **Labels:** 369 (includes outlier topic -1)
- **Taxonomy Mappings:** 361 (98.1% coverage)
- **Taxonomy Metadata:** ✅ Embedded in model (`topic_metadata_` attribute)

**Advantages:**
1. ✅ **Complete taxonomy mappings** embedded directly in the model
2. ✅ **No separate JSON file required** for taxonomy lookups
3. ✅ **Ready for statistical analysis** with `aggregate_taxonomy_by_book.py`
4. ✅ **Compatible with visualization** in `taxonomy_category_analysis.ipynb`

**Coverage Analysis:**
- 361 topics have taxonomy mappings (98.1%)
- 7 topics missing taxonomy mappings (1.9%)
- All 368 topics have labels

---

## Comparison Matrix

| Feature | Stage 08 Models | Stage 09 (Recommended) |
|---------|----------------|------------------------|
| **Topics** | 368 | 368 |
| **Labels** | ✅ 369 | ✅ 369 |
| **Taxonomy Mappings** | ❌ 0 | ✅ 361 (embedded) |
| **Ready for Analysis** | ⚠️ Requires JSON | ✅ Ready |
| **Model Size** | ~4.8 GB | ~4.8 GB |

---

## Recommendations

### Primary Recommendation

**Use:** `model_1_with_llm_labels_and_metadata_disambiguated.pkl` from `stage09_category_mapping`

**Reason:** This model has taxonomy metadata embedded directly, making it the most complete and convenient option for downstream analysis.

### Usage Instructions

#### For `aggregate_taxonomy_by_book.py`:

The recommended model can be used directly. The taxonomy mappings are embedded in the model's `topic_metadata_` attribute, so you can:

1. Load the model
2. Access taxonomy mappings via `model.topic_metadata_`
3. Use with sentence-level topic assignments

#### For `taxonomy_category_analysis.ipynb`:

The model is ready for statistical analysis. The embedded taxonomy mappings provide:
- Main category IDs for each topic
- Secondary category IDs (where applicable)
- Confidence scores
- Rationale for each mapping

### Alternative Workflow (If Needed)

If you need to use a Stage 08 model instead:

1. Use `model_1_with_llm_labels.pkl` from Stage 08
2. Generate taxonomy mappings using `zeroshot_taxonomy_openrouter.py`:
   ```bash
   python -m src.stage09_category_mapping.stage2_theory_driven_categories.scripts.zeroshot_taxonomy_openrouter \
     --labels-json results/stage08_llm_labeling/labels_pos_openrouter_*.json \
     --output-json results/stage09_category_mapping/stage2_theory_driven_categories/taxonomy_mappings_full.json
   ```
3. Use the taxonomy JSON file with `aggregate_taxonomy_by_book.py`

---

## Next Steps

### Immediate Actions

1. ✅ **Use recommended model** for all downstream analysis
2. ⚠️ **Consider generating full taxonomy mappings** if you need 100% coverage (currently 98.1%)
3. ✅ **Proceed with statistical analysis** using `taxonomy_category_analysis.ipynb`

### Future Improvements

1. **Complete taxonomy coverage:** Generate mappings for the remaining 7 topics
2. **Validation:** Manually review taxonomy mappings for accuracy
3. **Documentation:** Document any topics that couldn't be mapped to taxonomy categories

---

## Technical Details

### Model Loading

All models use the `RetrainableBERTopicModel` wrapper format. The comparison script handles:
- Loading from `.pkl` files (wrapper format)
- Extracting `BERTopic` model from wrapper
- Loading from directory format (safetensors)

### Taxonomy Metadata Structure

The recommended model stores taxonomy mappings in `topic_metadata_` attribute with the following structure:

```python
{
  topic_id: {
    "main_category_id": "4.2",
    "secondary_category_id": "5.1",
    "other_plausible_ids": ["3.2"],
    "is_noise": false,
    "confidence": "medium",
    "rationale": "..."
  }
}
```

### Topic Count Consistency

✅ **All models have consistent topic counts:**
- 368 topics (excluding outlier -1)
- 369 labels (including outlier -1)
- No topic count mismatches detected

---

## Warnings and Notes

- ⚠️ **No warnings** detected in the comparison
- ℹ️ Only one test taxonomy JSON file found (10 topics) - full mapping file may need to be generated
- ℹ️ 7 topics (1.9%) in recommended model lack taxonomy mappings - may need manual review

---

## Report Generation

This report was generated by:
- **Script:** `scripts/compare_models.py`
- **JSON Report:** `results/stage09_category_mapping/stage2_theory_driven_categories/model_comparison_report.json`
- **Date:** December 13, 2025

To regenerate this report, run:
```bash
python -m src.stage09_category_mapping.stage2_theory_driven_categories.scripts.compare_models
```

---

## Conclusion

The comparison clearly identifies **`model_1_with_llm_labels_and_metadata_disambiguated.pkl`** as the most complete model for taxonomy-based statistical analysis. This model provides:

- ✅ Complete topic coverage (368 topics)
- ✅ LLM-generated labels (369 labels)
- ✅ Embedded taxonomy mappings (361 mappings, 98.1% coverage)
- ✅ Ready for immediate use in downstream analysis

**Status:** ✅ **Ready for production use** in statistical analysis pipeline.

