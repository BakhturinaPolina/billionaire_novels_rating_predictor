# Max Tokens Update Summary

## Current Status

### 1. Current Setting (BEFORE)
- **`DEFAULT_MAX_TOKENS = 16`** in `main_openrouter.py`
- Comment: "Only need 2–6 words" (outdated)
- **Result**: All 15 topics analyzed have truncated scene summaries

### 2. Updated Setting (AFTER)
- **`DEFAULT_MAX_TOKENS = 60`** in `main_openrouter.py`
- Comment: "Need tokens for label (2–6 words) + scene summary (12–25 words, ~48 tokens)"
- **Expected Result**: Complete scene summaries with proper punctuation

## Analysis of Current Summaries (max_tokens=16)

### Statistics (First 15 Topics)
- **Total topics**: 15
- **Complete summaries**: 0 (0%)
- **Truncated summaries**: 15 (100%)
- **Missing punctuation**: 14 (93%)
- **Ends on function word**: 6 (40%)
- **Average tokens per summary**: 4.5 tokens

### Common Issues
1. **Truncated sentences** ending on function words:
   - `'Characters negotiate a'` (ends on "a")
   - `'Characters suggest or accept invitations for'` (ends on "for")
   - `'Characters reflect on time passing and'` (ends on "and")

2. **Missing terminal punctuation** (14/15 summaries):
   - Only 1 summary has a period (Topic 6: `'Characters drink wine together.'`)
   - All others are incomplete fragments

3. **Too short** (average 4.5 tokens vs target 12–25 tokens):
   - Shortest: 2 tokens (`'Characters engage'`)
   - Longest: 7 tokens (`'Characters discuss and question each other about'`)

## Recommendations

### 1. Increase max_tokens to 60
**Rationale:**
- Label: 2–6 words ≈ 8–12 tokens
- Scene summary: 12–25 words ≈ 48 tokens
- Buffer: ~4 tokens for formatting
- **Total: ~60 tokens**

### 2. Updated Prompt (Already Implemented)
The system prompt has been updated with:
- Explicit requirement for complete sentences ending with periods
- Guidance on 12–25 word target length
- Constraints against truncation on function words

### 3. Post-Processing Validation
Use `scripts/check_truncated_summaries.py` to flag suspicious summaries:
```bash
python scripts/check_truncated_summaries.py \
  --labels-json results/stage06_labeling_openrouter/labels_pos_openrouter_mistralai_mistral-nemo_romance_aware_paraphrase-MiniLM-L6-v2.json \
  --limit-topics 15
```

## Expected Improvements (After max_tokens=60)

### Before (max_tokens=16)
```
Topic 0: 'Characters negotiate a'
Topic 1: 'Characters engage'
Topic 2: 'Character uses tongue to stimulate'
```

### After (max_tokens=60) - Expected
```
Topic 0: 'Characters negotiate deals and promises that could change their situation.'
Topic 1: 'Characters engage in intimate kissing involving breasts, mouth, and body contact.'
Topic 2: 'Character uses tongue to stimulate clitoris during intimate encounters.'
```

## Implementation

### Files Updated
1. **`src/stage06_labeling/openrouter_experiments/main_openrouter.py`**
   - Changed `DEFAULT_MAX_TOKENS` from 16 to 60
   - Updated comment to reflect dual output (label + scene summary)

2. **`src/stage06_labeling/openrouter_experiments/generate_labels_openrouter.py`**
   - Updated `ROMANCE_AWARE_SYSTEM_PROMPT` with sharper scene_summary guidance
   - Added explicit constraints against truncation

3. **`scripts/check_truncated_summaries.py`** (New)
   - Flags suspicious summaries for manual review

4. **`scripts/compare_summaries_before_after.py`** (New)
   - Compares summaries before/after changes

## Next Steps

1. **Re-run labeling** with `max_tokens=60`:
   ```bash
   python -m src.stage06_labeling.openrouter_experiments.main_openrouter \
     --max-tokens 60 \
     --limit-topics 15
   ```

2. **Validate results** using the comparison script:
   ```bash
   python scripts/compare_summaries_before_after.py \
     --before-json results/stage06_labeling_openrouter/labels_pos_openrouter_mistralai_mistral-nemo_romance_aware_paraphrase-MiniLM-L6-v2.json \
     --after-json results/stage06_labeling_openrouter/labels_pos_openrouter_mistralai_mistral-nemo_romance_aware_paraphrase-MiniLM-L6-v2.json \
     --limit-topics 15
   ```

3. **Check for truncation**:
   ```bash
   python scripts/check_truncated_summaries.py \
     --labels-json <new_labels_file>.json \
     --limit-topics 15
   ```

## Cost Impact

Increasing `max_tokens` from 16 to 60:
- **Token increase**: ~3.75x per completion
- **Cost increase**: ~3.75x per label (completion tokens are typically more expensive)
- **Quality improvement**: Complete, usable scene summaries vs truncated fragments

**Trade-off**: Higher cost but significantly better quality for scientific analysis.

