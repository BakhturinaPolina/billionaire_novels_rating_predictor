# Category Mapping Validation Summary

**Date**: 2025-11-30  
**Status**: ✅ Validation Complete - Issues Identified

## Executive Summary

Validation of category assignments identified **7 Z misclassifications** and **1 explicit/intimacy boundary issue**. The root cause is that category mapping was run before regex patterns for `intercourse`, `mouth play`, and `outfit` were added. The patterns now work correctly, but the existing mappings need to be updated.

## Key Findings

### ✅ Pattern Updates Applied

1. **Added missing patterns to `O_appearance_aesthetics`**:
   - `outfit(s)`, `clothing`, `wardrobe`, `fashion`, `clothes`
   - **Impact**: "Tomorrow's Outfit" will now correctly map to `O_appearance_aesthetics`

2. **Existing patterns verified**:
   - `intercourse` → `C_explicit` ✓
   - `mouth play` → `C_explicit` ✓
   - These patterns were added in recent commits but mappings weren't regenerated

### ⚠️ Issues Identified

#### High Priority (7 topics)

1. **Topic 32: "Hesitant Intercourse"**
   - Current: `Z_noise_oog`
   - Should be: `C_explicit`
   - Keywords contain "intercourse"
   - **Fix**: Pattern exists, needs re-mapping

2. **Topic 39: "Moist Mouth Play"**
   - Current: `Z_noise_oog`
   - Should be: `C_explicit` (or split `B_mutual_intimacy`/`C_explicit`)
   - Keywords: lip, teeth, nose, tongue, mouth, moan
   - **Fix**: Pattern exists, needs re-mapping

3. **Topic 31: "Tomorrow's Outfit"**
   - Current: `Z_noise_oog`
   - Should be: `O_appearance_aesthetics`
   - Keywords: clothes, outfit, wardrobe, clothing, fashion
   - **Fix**: Pattern now added, needs re-mapping

#### Medium Priority (4 topics)

4. **Topic 26: "Nighttime Stand"**
   - Current: `Z_noise_oog`
   - Should be: `B_mutual_intimacy` or `H_domestic_nesting`
   - Keywords: night, nighttime, evening
   - **Note**: May need LLM review (could be legitimate Z if "stand" means something else)

5. **Topic 106: "Nighttime Confession"**
   - Current: `Z_noise_oog`
   - Should be: `B_mutual_intimacy` or `H_domestic_nesting`
   - Keywords: night
   - **Note**: May need LLM review

6. **Topic 241: "First Touch"**
   - Current: `Z_noise_oog`
   - Should be: `B_mutual_intimacy`
   - Keywords: start (very sparse)
   - **Note**: May need LLM review - keywords are minimal

7. **Topic 274: "Wild Night Out"**
   - Current: `Z_noise_oog`
   - Should be: `B_mutual_intimacy` or `H_domestic_nesting`
   - Keywords: clubs, strip, whores (may be legitimate Z)
   - **Note**: Needs LLM review - context suggests nightlife, not romance intimacy

#### Explicit/Intimacy Boundary (1 topic)

8. **Topic 15: "Intimate Touch"**
   - Current: `B_mutual_intimacy` (0.5) + `C_explicit` (0.5)
   - Issue: Contains explicit keywords (clit, pussy, erection) but split assignment may be correct
   - **Status**: ✅ Current assignment is actually correct (split B/C)

## Category Distribution

| Category | Count | Percentage | Status |
|----------|-------|------------|--------|
| Z_noise_oog | 118 | 32.7% | ⚠️ High - needs review |
| J_family_support | 25 | 6.9% | ✅ Normal |
| B_mutual_intimacy | 26 | 7.2% | ✅ Normal |
| O_appearance_aesthetics | 24 | 6.6% | ✅ Normal |
| K_work_corporate | 23 | 6.4% | ✅ Normal |
| F_negative_affect | 17 | 4.7% | ✅ Normal |
| S_scene_anchor | 17 | 4.7% | ✅ Normal |
| P_tech_media | 15 | 4.2% | ✅ Normal |
| H_domestic_nesting | 14 | 3.9% | ✅ Normal |
| I_humor_lightness | 13 | 3.6% | ✅ Normal |
| A_commitment_hea | 11 | 3.0% | ✅ Normal |
| Q_miscommunication | 12 | 3.3% | ✅ Normal |
| C_explicit | 7 | 1.9% | ⚠️ Low - may increase after fixes |
| D_luxury_wealth_status | 8 | 2.2% | ✅ Normal |
| E_threat_danger | 8 | 2.2% | ✅ Normal |
| G_rituals_gifts | 8 | 2.2% | ✅ Normal |
| T_repair_apology | 4 | 1.1% | ✅ Normal |
| M_health_recovery | 5 | 1.4% | ✅ Normal |
| N_separation_reunion | 2 | 0.6% | ✅ Normal |
| R1_protectiveness | 2 | 0.6% | ✅ Normal |
| L_vices_addictions | 1 | 0.3% | ✅ Normal |
| R2_jealousy | 1 | 0.3% | ✅ Normal |

## Root Cause Analysis

1. **Patterns were added after initial mapping**: The regex patterns for `intercourse`, `mouth play`, and `outfit` were added in recent commits, but the category mapping wasn't regenerated.

2. **Z_noise_oog is fallback category**: When no patterns match, topics default to `Z_noise_oog`. This is correct behavior, but means topics that should match need their patterns to be present.

3. **Some topics need LLM review**: Topics like "Nighttime Stand" and "Wild Night Out" may be legitimate Z if the context is non-romantic (e.g., nightlife, sports).

## Recommendations

### Immediate Actions

1. ✅ **Update regex patterns** (COMPLETED)
   - Added `outfit`, `clothing`, `wardrobe`, `fashion`, `clothes` to `O_appearance_aesthetics`

2. **Run fix Z script** to reclassify identified misclassifications:
   ```bash
   python -m src.stage06_labeling.category_mapping.fix_z_topics \
       --labels results/stage06_labeling_openrouter/labels_pos_openrouter_romance_aware_paraphrase-MiniLM-L6-v2.json \
       --category-probs results/stage06_labeling/category_mapping/topic_to_category_probs.json \
       --outdir results/stage06_labeling/category_mapping
   ```

3. **Alternative: Re-run full category mapping** (if you want to regenerate everything):
   ```bash
   python -m src.stage06_labeling.category_mapping.main_category_mapping \
       --labels results/stage06_labeling_openrouter/labels_pos_openrouter_romance_aware_paraphrase-MiniLM-L6-v2.json \
       --outdir results/stage06_labeling/category_mapping
   ```

### Validation Steps

1. **After fix Z, compare results**:
   ```bash
   python -m src.stage06_labeling.category_mapping.compare_fix_z_results \
       --original results/stage06_labeling/category_mapping/topic_to_category_probs.json \
       --fixed results/stage06_labeling/category_mapping/topic_to_category_probs_fixed.json \
       --fix-results results/stage06_labeling/category_mapping/fix_z_results.json \
       --labels results/stage06_labeling_openrouter/labels_pos_openrouter_romance_aware_paraphrase-MiniLM-L6-v2.json \
       --output results/stage06_labeling/category_mapping/fix_z_comparison_report.md
   ```

2. **Re-run validation**:
   ```bash
   python -m src.stage06_labeling.category_mapping.validate_assignments \
       --summary results/stage06_labeling/category_mapping/topic_to_category_summary.csv \
       --probs results/stage06_labeling/category_mapping/topic_to_category_probs_fixed.json \
       --labels results/stage06_labeling_openrouter/labels_pos_openrouter_romance_aware_paraphrase-MiniLM-L6-v2.json
   ```

### Theoretical Framework Validation

✅ **Category assignments align with theoretical framework**:
- A-P categories correctly operationalize Radway (1984) and Propp functions
- Cross-cutting categories (Q, T, R) properly separate miscommunication, repair, and protectiveness/jealousy
- Z_noise_oog correctly captures out-of-domain topics (sports, animals, etc.)

⚠️ **Edge cases identified**:
- Some topics with minimal keywords may need manual review
- Nighttime-related topics may be ambiguous (romance intimacy vs. nightlife)

## Next Steps

1. ✅ Push to remote (COMPLETED)
2. ✅ Create validation script (COMPLETED)
3. ✅ Update regex patterns (COMPLETED)
4. ✅ Re-run category mapping with updated patterns (COMPLETED)
5. ✅ Validate updated assignments (COMPLETED)
6. ✅ Run OpenRouter labeling with POS keywords from JSON (COMPLETED - with fallback labels)
7. ✅ Map OpenRouter labels to categories (COMPLETED)
8. ✅ Integrate categories into BERTopic model (COMPLETED)
9. ✅ Update validation summary with final results (COMPLETED)

## Update: Category Mapping Re-run Results (2025-12-01)

### ✅ High-Priority Fixes Applied

After re-running category mapping with updated regex patterns:

1. **Topic 32: "Hesitant Intercourse"**
   - ✅ Fixed: Now correctly classified as `C_explicit` (was `Z_noise_oog`)

2. **Topic 39: "Moist Mouth Play"**
   - ✅ Fixed: Now correctly classified as `C_explicit` (was `Z_noise_oog`)

3. **Topic 31: "Tomorrow's Outfit"**
   - ✅ Fixed: Now correctly classified as `O_appearance_aesthetics` (was `Z_noise_oog`)

### Impact Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Z_noise_oog | 118 (32.7%) | 115 (31.9%) | -3 (-0.8%) |
| C_explicit | 7 (1.9%) | 12 (3.3%) | +5 (+1.4%) |
| O_appearance_aesthetics | 24 (6.6%) | 25 (6.9%) | +1 (+0.3%) |
| B_mutual_intimacy | 26 (7.2%) | 32 (8.9%) | +6 (+1.7%) |

### Remaining Edge Cases (4 topics)

These topics remain classified as `Z_noise_oog` and may need LLM review:

1. **Topic 26: "Nighttime Stand"** - Ambiguous context
2. **Topic 106: "Nighttime Confession"** - Minimal keywords
3. **Topic 241: "First Touch"** - Very sparse keywords ("start" only)
4. **Topic 274: "Wild Night Out"** - Context suggests nightlife, not romance intimacy

**Status**: All high-priority misclassifications have been fixed. Remaining topics are edge cases that may be legitimate Z classifications.

## Final Pipeline Execution (2025-12-01)

### ✅ OpenRouter API - Successfully Regenerated Labels

**Initial Issue**: First API key returned 401 authentication errors.  
**Resolution**: New API key (`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`) tested and validated successfully.

**Final Status**:
- ✅ **361 topics labeled** with LLM-generated romance-aware labels
- ✅ **Improved prompts** used (BASE_LABELING_PROMPT with JSON output)
- ✅ **Categories included** in LLM responses (validated against regex mapping)
- ✅ **Category mapping** completed successfully
- ✅ **BERTopic integration** completed with enhanced labels and category tags
- ✅ **Final validation** shows only 3 Z misclassifications (down from 7)

**Label Quality Improvement**:
- **Before (fallback)**: Generic keywords like "cheeks", "mirror", "privacy"
- **After (LLM)**: Descriptive phrases like "Self-Reflection in the Bathroom", "Protective Guardian", "Reluctant Reunion"

**Example Labels Generated**:
- Topic 365: "Self-Reflection in the Bathroom" (Categories: F_negative_affect, O_appearance_aesthetics)
- Topic 367: "Protective Guardian" (Categories: R1_protectiveness)
- Topic 364: "Reluctant Reunion" (Categories: N_separation_reunion)
- Topic 363: "Prophetic Love Connection" (Categories: Q_miscommunication, N_separation_reunion)

### Pipeline Steps Completed

1. **OpenRouter Labeling** (with LLM-generated labels)
   - Input: `results/stage06_exploration/topics_all_representations_paraphrase-MiniLM-L6-v2.json`
   - Output: `results/stage06_labeling_openrouter/labels_pos_openrouter_romance_aware_paraphrase-MiniLM-L6-v2.json`
   - Status: ✅ Completed (361 topics processed with LLM-generated labels)
   - Time: ~9.5 minutes (568 seconds)
   - API: All requests successful (200 OK)

2. **Category Mapping**
   - Input: Labels JSON from step 1 (with LLM-generated labels)
   - Output: `results/stage06_labeling/category_mapping/topic_to_category_probs.json`
   - Status: ✅ Completed (361 topics mapped)

3. **BERTopic Integration**
   - Input: Category mappings + LLM labels
   - Output: `models/retrained/paraphrase-MiniLM-L6-v2/model_1_with_categories/`
   - Status: ✅ Completed (361 topics integrated with enhanced labels)

4. **Final Validation**
   - Output: `results/stage06_labeling/category_mapping/validation_report_final_with_llm_labels.md`
   - Status: ✅ Completed
   - Results: **3 Z misclassifications** (down from 7), 0 explicit/intimacy issues, 0 regex pattern issues

### Final Status

✅ **All pipeline steps completed successfully**
- Categories integrated into BERTopic model for statistical analysis
- Model ready for Stage 07 analysis
- Validation shows high-priority issues resolved

## Full Pipeline Workflow

### Step 1: Generate Labels with OpenRouter API (POS Keywords)

```bash
# Set API key
export OPENROUTER_API_KEY=your_key_here

# Run OpenRouter labeling with POS representation from JSON
source venv/bin/activate
python -m src.stage06_labeling.openrouter_experiments.main_openrouter \
    --embedding-model paraphrase-MiniLM-L6-v2 \
    --topics-json results/stage06_exploration/topics_all_representations_paraphrase-MiniLM-L6-v2.json \
    --num-keywords 15 \
    --use-improved-prompts
```

**Output**: `results/stage06_labeling_openrouter/labels_pos_openrouter_romance_aware_paraphrase-MiniLM-L6-v2.json`

### Step 2: Map Labels to Categories

```bash
python -m src.stage06_labeling.category_mapping.main_category_mapping \
    --labels results/stage06_labeling_openrouter/labels_pos_openrouter_romance_aware_paraphrase-MiniLM-L6-v2.json \
    --outdir results/stage06_labeling/category_mapping
```

**Output**: 
- `topic_to_category_probs.json`
- `topic_to_category_final.csv`
- `topic_to_category_summary.csv`

### Step 3: Integrate Categories into BERTopic Model

```bash
python -m src.stage06_labeling.category_mapping.integrate_categories_to_bertopic \
    --model-path models/retrained/paraphrase-MiniLM-L6-v2/model_1.pkl \
    --category-probs results/stage06_labeling/category_mapping/topic_to_category_probs.json \
    --labels results/stage06_labeling_openrouter/labels_pos_openrouter_romance_aware_paraphrase-MiniLM-L6-v2.json \
    --output-path models/retrained/paraphrase-MiniLM-L6-v2/model_1_with_categories.pkl
```

**Output**: Updated BERTopic model with category mappings integrated

### Step 4: Validate Final Results

```bash
python -m src.stage06_labeling.category_mapping.validate_assignments \
    --summary results/stage06_labeling/category_mapping/topic_to_category_summary.csv \
    --probs results/stage06_labeling/category_mapping/topic_to_category_probs.json \
    --labels results/stage06_labeling_openrouter/labels_pos_openrouter_romance_aware_paraphrase-MiniLM-L6-v2.json
```

## Files Modified

- `src/stage06_labeling/category_mapping/map_topics_to_categories.py`: Added outfit/clothing patterns
- `src/stage06_labeling/category_mapping/validate_assignments.py`: New validation script
- `results/stage06_labeling/category_mapping/validation_report.md`: Validation results
- `results/stage06_labeling/category_mapping/VALIDATION_SUMMARY.md`: This summary

