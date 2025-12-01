# OpenRouter API Key Troubleshooting

## Issue: 401 Authentication Errors

During pipeline execution, OpenRouter API returned 401 errors with message "User not found" for all API calls.

### Symptoms

- All API calls return: `Error code: 401 - {'error': {'message': 'User not found.', 'code': 401}}`
- Pipeline falls back to using first keyword as label for each topic
- Pipeline completes successfully but with fallback labels instead of LLM-generated labels

### Root Cause

The API key `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` appears to be:
- Invalid (key doesn't exist in OpenRouter system)
- Expired (key was valid but has expired)
- Revoked (key was disabled by account owner)
- Incorrect format (though format looks correct: `sk-or-v1-...`)

### Verification Steps

1. **Check API key format**:
   ```bash
   echo $OPENROUTER_API_KEY | head -c 20
   # Should start with: sk-or-v1-
   ```

2. **Test API key directly**:
   ```bash
   curl https://openrouter.ai/api/v1/auth/key \
     -H "Authorization: Bearer $OPENROUTER_API_KEY"
   ```

3. **Check OpenRouter dashboard**:
   - Visit https://openrouter.ai/keys
   - Verify key exists and is active
   - Check usage limits and credits

### Solutions

#### Option 1: Get a New API Key

1. Go to https://openrouter.ai/keys
2. Create a new API key
3. Update environment variable:
   ```bash
   export OPENROUTER_API_KEY=sk-or-v1-YOUR_NEW_KEY_HERE
   ```

#### Option 2: Verify Existing Key

1. Log into OpenRouter account
2. Check if key is still active
3. Verify account has credits/balance
4. Check if key has usage restrictions

#### Option 3: Use Fallback Labels (Current State)

The pipeline successfully completed using fallback labels (first keyword from each topic). While not ideal, this allows:
- ✅ Category mapping to work correctly
- ✅ Categories to be integrated into BERTopic model
- ✅ Statistical analysis to proceed

**Note**: Fallback labels are less descriptive than LLM-generated labels, but category mapping still functions correctly since it uses keywords, not labels.

### Impact on Results

**Current State** (with fallback labels):
- ✅ 361 topics processed
- ✅ Categories mapped correctly
- ✅ Model integration completed
- ⚠️ Labels are generic (first keyword) instead of descriptive phrases

**With Valid API Key** (regenerated labels):
- ✅ More descriptive, romance-aware labels
- ✅ Better topic interpretation
- ✅ Same category mappings (since categories use keywords)

### Regenerating Labels (After Fixing API Key)

Once you have a valid API key:

```bash
# Set new API key
export OPENROUTER_API_KEY=sk-or-v1-YOUR_NEW_KEY_HERE

# Regenerate labels
source venv/bin/activate
python -m src.stage06_labeling.openrouter_experiments.main_openrouter \
    --embedding-model paraphrase-MiniLM-L6-v2 \
    --topics-json results/stage06_exploration/topics_all_representations_paraphrase-MiniLM-L6-v2.json \
    --num-keywords 15 \
    --use-improved-prompts

# Re-run category mapping (optional - categories should be same)
python -m src.stage06_labeling.category_mapping.main_category_mapping \
    --labels results/stage06_labeling_openrouter/labels_pos_openrouter_romance_aware_paraphrase-MiniLM-L6-v2.json \
    --outdir results/stage06_labeling/category_mapping
```

### Current Pipeline Status

Despite API authentication failures, the pipeline completed successfully:

1. ✅ **OpenRouter Labeling**: Completed with fallback labels
2. ✅ **Category Mapping**: Completed successfully (uses keywords, not labels)
3. ✅ **BERTopic Integration**: Categories integrated into model
4. ✅ **Validation**: Final validation completed

**The model is ready for Stage 07 statistical analysis**, even with fallback labels.

