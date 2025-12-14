---
name: Fix Radway Classification Errors
overview: Fix systematic Radway classification errors by improving prompts with disambiguation rules, adding heuristic override layer for common confusions, setting deterministic decoding, and fixing CSV export issues.
todos:
  - id: prompt-disambiguation
    content: Add disambiguation rules block to RADWAY_ZEROSHOT_SYSTEM_PROMPT (R4 vs R12, R7 narrowing, commitment overrides)
    status: completed
  - id: prompt-gated-none
    content: Add gated 'none' decision process to RADWAY_ZEROSHOT_SYSTEM_PROMPT
    status: completed
  - id: prompt-examples
    content: Add micro-examples section to RADWAY_ZEROSHOT_SYSTEM_PROMPT
    status: completed
  - id: deterministic-decoding
    content: Change OpenRouter API call parameters to deterministic (temperature=0.0, top_p=1.0, frequency_penalty=0.0)
    status: completed
  - id: add-regex-import
    content: Add import re to zeroshot_radway_openrouter.py
    status: completed
  - id: heuristic-override-function
    content: Add override_radway_by_cues() function with regex patterns for commitment, sex acts, breakup, argument, apology cues
    status: completed
  - id: integrate-override
    content: Call override_radway_by_cues() in map_all_topics_to_radway() after apply_radway_fallback_heuristics()
    status: completed
  - id: snippet-logging
    content: Add warning logging when snippets are missing for critical taxonomy groups
    status: completed
  - id: csv-export-fix
    content: Ensure radway_phase='NA' is preserved as string in CSV exports (check notebook/export scripts)
    status: completed
---

# Fix Radway Classification Errors

## Problem Summary

The Radway classification system has four systematic error patterns:

1. **Explicit sex scenes (taxonomy_main_id = 2.3)** mislabeled as R4 instead of R12
2. **Wedding/marriage/commitment cues** treated as "none" or Phase II tenderness instead of R11/R13
3. **R7 ("separated") overused** for conflict conversations that should be R2/R5/R10
4. **Fallback heuristics only fix "none"** - don't address other high-frequency confusions

## Implementation Plan

### 1. Prompt Improvements (`zeroshot_radway_openrouter.py`)

**File**: `src/stage09_category_mapping/stage3_radway_functions/scripts/zeroshot_radway_openrouter.py`

#### 1.1 Add Disambiguation Rules Block

Insert after line 321 (after "INTERPRETATION HINTS" section, before "OUTPUT CONSTRAINTS"):

```python
DISAMBIGUATION RULES (apply strictly):

1) R4 vs R12:
                                                                                                                                                                                                                                                   - Choose R4 ONLY for sexual tension/attraction/flirting/interpretation WITHOUT a described sex act.
                                                                                                                                                                                                                                                   - Choose R12 if the topic describes sex acts or foreplay (undressing, oral, penetration, BDSM session, condom, "in bed", nipple/breast play, orgasm).
                                                                                                                                                                                                                                                   - If taxonomy_main_id == 2.3 → default to R12 unless the text is ONLY about attraction (no act).

2) R7 (separation) is NARROW:
                                                                                                                                                                                                                                                   - Use R7 only if there is breakup/leaving/physical separation/no-contact/moved out/"we can't be together".
                                                                                                                                                                                                                                                   - If it's mainly an argument/confrontation/jealousy conversation → prefer R2/R5.
                                                                                                                                                                                                                                                   - If it's mainly apology/forgiveness/regret/amends → prefer R10.

3) Commitment overrides taxonomy:
                                                                                                                                                                                                                                                   - If label/summary mentions wedding/marriage/engagement/proposal/vows/husband/wife → choose R11 (or R13 if "settled HEA/family/home/baby/forever").
```

#### 1.2 Add Gated "None" Decision

Insert before "JSON SCHEMA (MANDATORY)" section (around line 333):

```python
DECISION PROCESS

First decide: radway_is_none (true/false).
- If true: set radway_main_id="none".
- If false: radway_main_id MUST be one of R1..R13 (never "none").
```

#### 1.3 Add Micro-Examples

Insert after "RADWAY FUNCTIONS (AVAILABLE LABELS)" section (around line 280):

```python
EXAMPLES (very short):
- "BDSM session / condom / foreplay / nipple play" → R12 (not R4)
- "Wedding planning / proposal / vows" → R11
- "Argument / accusation / jealousy talk" → R2 (not R7 unless they separate)
- "Apology + forgiveness + regret" → R10
```

### 2. Code Changes

#### 2.1 Set Deterministic Decoding Parameters

**File**: `zeroshot_radway_openrouter.py`, function `classify_topic_to_radway_openrouter()` (around line 782)

Change the OpenRouter API call parameters:

- `temperature=0.0` (currently 0.25)
- `top_p=1.0` (currently 0.9)
- `frequency_penalty=0.0` (currently 0.3)
- `presence_penalty=0.0` (already 0.0)

#### 2.2 Add Heuristic Override Function

**File**: `zeroshot_radway_openrouter.py`

Add import at top (after line 37):

```python
import re
```

Add new function `override_radway_by_cues()` after `apply_radway_fallback_heuristics()` (after line 1083). This function implements the regex-based override logic provided by the user, handling:

- Commitment cues (wedding/marriage) → R11/R13
- Explicit sex (2.3) → R12 (not R4)
- R7 sanity check (only if breakup/separation cues exist)
- R4 sanity check (remap if not actually sexual context)

#### 2.3 Integrate Override Function

**File**: `zeroshot_radway_openrouter.py`, function `map_all_topics_to_radway()` (around line 1203)

After the existing fallback call:

```python
result = apply_radway_fallback_heuristics(result, topic_entry)
```

Add:

```python
result = override_radway_by_cues(result, topic_entry)
```

#### 2.4 Improve Snippet Handling and Logging

**File**: `zeroshot_radway_openrouter.py`, function `classify_topic_to_radway_openrouter()` (around line 750)

Add warning when snippets are missing for critical taxonomy groups:

```python
if not representative_docs:
    tax_group = topic_entry.get("main_category_group", "")
    if tax_group in {"Relationship Trajectory (Main Couple)", "Sexuality, Attraction & Intimacy"}:
        LOGGER.warning(
            "Topic %d (taxonomy_group=%s) has no representative snippets - classification may be less accurate",
            topic_id,
            tax_group
        )
```

**File**: `zeroshot_radway_openrouter.py`, function `load_bertopic_model_for_snippets()` (around line 910)

Increase `max_docs_per_topic` for critical taxonomy groups (optional enhancement - can be done via parameter tuning).

### 3. CSV Export Fix

**Issue**: `radway_phase` shows as blank/NaN for "none" topics even though classifier sets it to "NA".

**File**: `notebooks/07_analysis/radway_model_interactive_eda.ipynb` (or wherever CSV export happens)

Ensure that when exporting to CSV:

- `radway_phase` is preserved as string "NA" (not coerced to NaN)
- Use `df["radway_phase"] = df["radway_phase"].fillna("NA")` before export
- Or ensure JSON loading preserves "NA" as string

**Note**: This may require checking the notebook or any Python scripts that export Radway results to CSV.

### 4. Optional: Retry-on-Violation (Future Enhancement)

Add a retry mechanism when contradictions are detected (e.g., 2.3 + R4). This can be added later as a follow-up enhancement.

## Files to Modify

1. **`src/stage09_category_mapping/stage3_radway_functions/scripts/zeroshot_radway_openrouter.py`**

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Add `import re`
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Update `RADWAY_ZEROSHOT_SYSTEM_PROMPT` with disambiguation rules, gated "none", and micro-examples
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Change decoding parameters in `classify_topic_to_radway_openrouter()`
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Add `override_radway_by_cues()` function
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Integrate override in `map_all_topics_to_radway()`
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Add snippet warning logging

2. **CSV export code** (location TBD - may be in notebook or separate script)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Ensure "NA" is preserved as string for `radway_phase`

## Expected Impact

- **2.3 → R4 errors**: Should drop significantly due to explicit R12 rule + heuristic override
- **Wedding/marriage → none/R8**: Should be fixed by commitment override rules
- **R7 overuse**: Should be reduced by narrowing R7 definition + heuristic check
- **Overall accuracy**: Deterministic decoding + heuristics should improve consistency

## Testing Recommendations

1. Run classification on a small subset (e.g., `--limit-topics 30`)
2. Compare before/after for topics with taxonomy_main_id = 2.3
3. Check wedding/marriage topics are correctly mapped to R11/R13
4. Verify R7 is only used when breakup/separation cues exist
5. Confirm "none" false negatives are reduced