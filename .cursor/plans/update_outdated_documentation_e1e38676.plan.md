---
name: Update Outdated Documentation
overview: Update README.md, METHODOLOGY.md, MODEL_VERSIONING.md, SCIENTIFIC_README.md, and docs/README.md to reflect the current 10-stage pipeline structure and correct stage descriptions.
todos:
  - id: update-main-readme
    content: "Update main README.md: change 'seven-stage' to 'ten-stage' and verify all 10 stages are correctly described"
    status: completed
  - id: update-methodology
    content: "Update METHODOLOGY.md: fix Stage 06/07 descriptions, add missing Stage 08/09/10 sections, update data flow diagram"
    status: completed
  - id: update-scientific-readme
    content: "Update SCIENTIFIC_README.md: fix all stage number references (06→08, 07→10), update section references"
    status: completed
  - id: update-docs-readme
    content: "Update docs/README.md: fix stage references in Stage-Specific section, add missing stages"
    status: completed
  - id: verify-model-versioning
    content: Verify MODEL_VERSIONING.md stage references are correct
    status: completed
---

# Documentation Update Plan

## Issues Identified

### 1. Main README.md (`README.md`)

- **Line 24**: Says "seven-stage" but pipeline has 10 stages
- **Lines 214-246**: Pipeline stages section is mostly correct but needs verification
- **Line 24**: Overview mentions "seven-stage" incorrectly

### 2. METHODOLOGY.md (`docs/METHODOLOGY.md`)

- **Lines 111-136**: Stage 06 described as "Thematic Mapping" but it's actually "Topic Exploration"
- **Lines 137-168**: Stage 07 described as "Statistical Analysis" but it's actually "Topic Quality Analysis"
- **Missing**: Stage 08 (LLM Labeling), Stage 09 (Category Mapping), Stage 10 (Correlation Analysis)
- **Lines 192-217**: Data flow diagram is outdated - shows old stage numbers and missing stages

### 3. SCIENTIFIC_README.md (`SCIENTIFIC_README.md`)

- **Line 186**: Mentions "Stage 06: Labeling" but labeling is actually Stage 08
- **Line 222**: Mentions "Stage 07: Analysis" but analysis is actually Stage 10
- **Lines 281-305**: Thematic Mapping section references old stage structure
- **Lines 186-184**: Automated Topic Labeling section needs stage number updates

### 4. docs/README.md (`docs/README.md`)

- **Lines 56-57**: References outdated stages:
- "Stage 06: Labeling" should be "Stage 08: LLM Labeling"
- "Stage 07: Analysis" should be "Stage 10: Correlation Analysis"
- **Missing**: References to Stage 06 (Topic Exploration), Stage 07 (Topic Quality), Stage 09 (Category Mapping)

### 5. MODEL_VERSIONING.md (`docs/MODEL_VERSIONING.md`)

- Appears mostly up-to-date, but should verify stage references are correct

## Updates Required

### Update 1: Main README.md

- Change "seven-stage" to "ten-stage" in overview
- Verify all 10 stages are correctly described
- Ensure stage numbers match actual implementation

### Update 2: METHODOLOGY.md

- Update Stage 06 description from "Thematic Mapping" to "Topic Exploration"
- Update Stage 07 description from "Statistical Analysis" to "Topic Quality Analysis"
- Add Stage 08: LLM Labeling section
- Add Stage 09: Category Mapping section
- Add Stage 10: Correlation Analysis section
- Update data flow diagram to show all 10 stages correctly

### Update 3: SCIENTIFIC_README.md

- Update all stage references:
- "Stage 06: Labeling" → "Stage 08: LLM Labeling"
- "Stage 07: Analysis" → "Stage 10: Correlation Analysis"
- Update Thematic Mapping section to reference Stage 09
- Update Automated Topic Labeling section to reference Stage 08
- Update Category Mapping section to reference Stage 09

### Update 4: docs/README.md

- Update stage references in "Stage-Specific" section:
- Add Stage 06: Topic Exploration
- Add Stage 07: Topic Quality Analysis
- Update Stage 06 reference to Stage 08: LLM Labeling
- Update Stage 07 reference to Stage 10: Correlation Analysis
- Add Stage 09: Category Mapping

### Update 5: MODEL_VERSIONING.md

- Verify all stage references are correct (appears mostly correct but double-check)

## Files to Update

1. `README.md` - Main project README
2. `docs/METHODOLOGY.md` - Technical methodology
3. `SCIENTIFIC_README.md` - Scientific methodology
4. `docs/README.md` - Documentation index
5. `docs/MODEL_VERSIONING.md` - Model versioning (verify only)

## Verification

After updates, verify:

- All stage numbers match actual directory structure (stage01 through stage10)
- Stage descriptions match their README.md files
- Data flow diagrams show correct stage progression
- Cross-references between documents are consistent