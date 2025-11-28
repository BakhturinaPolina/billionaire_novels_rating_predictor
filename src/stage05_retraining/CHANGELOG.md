# Stage 05 Retraining - Changelog

## 2024-11-28: Preprocessing Fixes and Diagnostics

### Fixed Issues

1. **OCTIS Dataset Creation - Use Cleaned DataFrame**
   - **Problem**: `create_octis_dataset()` was re-reading raw CSV instead of using cleaned DataFrame, causing preprocessing mismatch between training text and OCTIS corpus
   - **Solution**: Modified function to use cleaned DataFrame passed as parameter
   - **Impact**: Ensures consistency - both training and OCTIS corpus now use same cleaned text (mojibake fixed, unicode normalized, lowercase)
   - **Files**: `retrain_models.py` (lines 238-360)

2. **Diagnostic Tool Bug Fixes**
   - **Problem**: `diagnose_data.py` referenced undefined `subset_path` variable
   - **Solution**: Replaced all `subset_path` references with `dataset_path` parameter
   - **Added**: CLI integration with config loading for easier usage
   - **Added**: Enhanced logging with step markers and emoji indicators
   - **Files**: `diagnose_data.py`

### Enhancements

1. **Consistency Verification**
   - Added automatic verification that compares first 5 sentences from training data and OCTIS corpus
   - Reports match/mismatch status with detailed logging
   - Helps catch preprocessing inconsistencies early
   - **Files**: `retrain_models.py` (lines 312-360)

2. **Improved Logging**
   - Enhanced diagnostic tool with step-by-step progress indicators
   - Added consistency check logging in OCTIS dataset creation
   - Better error messages and status reporting

### Testing

- ✅ Verified consistency check passes (5/5 sample sentences match)
- ✅ Tested with subset dataset (10K rows)
- ✅ Confirmed cleaned text is used in both training and OCTIS paths

### Next Steps

- [x] Verify character names preprocessing in stoplist loading
- [x] Add character name extraction logic if needed
- [ ] Test full retraining pipeline with all fixes

## 2024-11-28: Character Names Preprocessing

### Fixed Issues

1. **Character Names Not Preprocessed in Stoplist Loading**
   - **Problem**: `load_custom_stopwords()` was loading raw character name entries (e.g., "4 STELLA", "A Fitzpatrick", "A voice") without preprocessing
   - **Solution**: Added `preprocess_character_name()` function that:
     - Removes prefixes (A, Mr., Miss, the, AKA, #, numbers)
     - Removes quotes and punctuation
     - Splits multi-word names into individual tokens
     - Filters out non-name patterns (long lines, common phrases)
     - Extracts clean name tokens (e.g., "alex", "stella", "crane")
   - **Impact**: Character names are now properly extracted and added to stopwords, matching the preprocessing described in CHARACTER_NAMES_ANALYSIS.md
   - **Files**: `retrain_models.py` (lines 88-180)

### Verification Results

- ✅ Extracted 4,687 unique character name tokens from 7,525 raw entries
- ✅ Successfully filters raw entries like "4 STELLA" → extracts "stella"
- ✅ Successfully extracts names from entries like "A Fitzpatrick" → extracts "fitzpatrick"
- ✅ Multi-word names split correctly: "Alex Crane" → extracts both "alex" and "crane"
- ✅ Common character names verified: alex, stella, weston, fitzpatrick, crane, hunter, sebastian
- ✅ Total stopwords: ~5,000 (318 English + 4,687 character names)

### Notes

- Some false positives remain (e.g., "aardvarks", "about", "accustomed") - this is acceptable per CHARACTER_NAMES_ANALYSIS.md (1-2% false positive rate)
- Preprocessing matches the documented approach in README_CHARACTER_NAMES.md

