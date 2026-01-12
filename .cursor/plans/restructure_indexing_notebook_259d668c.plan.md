---
name: Restructure indexing notebook
overview: Restructure the indexing_hypothesis_testing notebook to eliminate duplicate definitions, remove/disable deprecated code, and create a clean single-pipeline structure following the provided 13-cell skeleton. This will ensure deterministic execution and prevent silent overwrites.
todos:
  - id: analyze_structure
    content: Analyze current notebook structure to identify all duplicate definitions and deprecated sections
    status: completed
  - id: create_cell1
    content: "Create Cell 1: Title + global switches (PROJECT_DIR, OUT_DIR, RANDOM_SEED, NOTEBOOK_VERSION, imports)"
    status: completed
    dependencies:
      - analyze_structure
  - id: create_cell2
    content: "Create Cell 2: Input mapping - map book_topic_probs (rename prob→topic_prob), topic_lookup, optional book_wide"
    status: completed
    dependencies:
      - create_cell1
  - id: create_cell3
    content: "Create Cell 3: Column contract + assertions - validate required columns, data types, fail fast"
    status: completed
    dependencies:
      - create_cell2
  - id: create_cell4
    content: "Create Cell 4: Single definition of ALL helpers (ensure_cols, safe_bool_series, normalize_tax_node, export_vocab, build_topic_spine, add_weights, aggregate_membership, membership_wide, compute_book_indexes, split_csv_tags, parse_secondary_dim, compute_w_eff_safe, build_mask_logic, taxonomy_first_mask)"
    status: completed
    dependencies:
      - create_cell3
  - id: create_cell5
    content: "Create Cell 5: Build frozen topic_spine once (merge lookup+taxonomy, parse tags, normalize taxonomy, compute weights) - NEVER REBUILD AFTER THIS"
    status: completed
    dependencies:
      - create_cell4
  - id: create_cell6
    content: "Create Cell 6: Stage 0 Diagnostics (print counts, export vocabularies for taxonomy/tags/radway)"
    status: completed
    dependencies:
      - create_cell5
  - id: create_cell7
    content: "Create Cell 7: Stage 1 Membership (taxonomy-only) - use CORE_STAGE1 mapping, aggregate, compute indexes, export"
    status: completed
    dependencies:
      - create_cell6
  - id: create_cell8
    content: "Create Cell 8: Stage 2 Feature builder design (set STAGE2_MODE, define TAG_USE from secondary_dim keys)"
    status: completed
    dependencies:
      - create_cell7
  - id: create_cell9
    content: "Create Cell 9: Stage 2 Membership build (taxonomy + tags refinement using CORE_STAGE2 + REFINE_STAGE2)"
    status: completed
    dependencies:
      - create_cell8
  - id: create_cell10
    content: "Create Cell 10: Stage 3 Design (set RADWAY_COL, optionally RADWAY_TARGET_TAX_NODES)"
    status: completed
    dependencies:
      - create_cell9
  - id: create_cell11
    content: "Create Cell 11: Stage 3 Membership build (Stage 2 + Radway overlay using CORE_STAGE3 + REFINE_STAGE3)"
    status: completed
    dependencies:
      - create_cell10
  - id: create_cell12
    content: "Create Cell 12: Comparability table (merge Stage 1/2/3 indexes, export combined)"
    status: completed
    dependencies:
      - create_cell11
  - id: create_cell13
    content: "Create Cell 13: Optional merge with book_wide metadata"
    status: completed
    dependencies:
      - create_cell12
  - id: disable_deprecated
    content: "Disable DEPRECATED section (cell ~1539) by wrapping in if False: or deleting"
    status: completed
    dependencies:
      - create_cell13
  - id: remove_duplicates
    content: Remove all duplicate function/definition cells (keep only Cell 4 versions and Stage 1-3 definitions)
    status: completed
    dependencies:
      - disable_deprecated
  - id: remove_old_loaders
    content: Remove old data loading and topic_spine building cells (keep only Cells 2 and 5)
    status: completed
    dependencies:
      - remove_duplicates
  - id: test_execution
    content: Verify notebook runs top-to-bottom without errors and produces consistent outputs
    status: pending
    dependencies:
      - remove_old_loaders
---

# Restructure Indexing Hypothesis Testing Notebook

## Problem Summary

The notebook has critical issues causing non-deterministic behavior:

1. **Silent overwrites**: Functions (`build_mask_logic`, `taxonomy_first_mask`, `resolve_subgroups`, `split_tags`, `parse_secondary_dim`, `compute_w_eff_safe`) and dictionaries (`CORE`, `COMPOSITES`, `available_subgroups`) are defined multiple times
2. **Multiple competing pipelines**: Layered builder (cells 4-15), taxonomy-first builder (cells 23-28), and deprecated code (cell 1539+) all coexist
3. **DEPRECATED section is executable**: Can overwrite functions if cells run out of order

## Solution: Clean 13-Cell Structure

Replace the current structure with a single deterministic pipeline following the provided skeleton, adapted to actual column names found in the notebook.

### Column Name Mappings (from notebook analysis)

- **Book-topic probabilities**: `book_topic_probs` with columns `book_id`, `topic_id`, `prob` (not `topic_prob`)
- **Tag columns**: `primary_categories` → parsed to `primary_set`, `secondary_categories` → parsed to `secondary_set` and `secondary_dim`
- **Radway columns**: `radway_main_name`, `radway_phase_name`
- **Taxonomy columns**: `taxonomy_main_name` → `tax_node_name`, `taxonomy_main_group` → `tax_group_name`

### Implementation Steps

#### 1. Create New Clean Structure (Cells 1-13)

**Cell 1**: Title + global switches

- Set `PROJECT_DIR`, `OUT_DIR`, `RANDOM_SEED`, `NOTEBOOK_VERSION`
- Import statements

**Cell 2**: Input mapping (ONE place)

- Map `book_topic_probs` (columns: `book_id`, `topic_id`, `prob`)
- Map `topic_lookup` 
- Map optional `book_wide` if exists
- Rename `prob` → `topic_prob` for consistency with skeleton

**Cell 3**: Column contract + assertions

- Validate required columns exist
- Check data types
- Fail fast on missing data

**Cell 4**: Helpers (ONLY place they are defined)

- `ensure_cols()` - ensure columns exist with defaults
- `safe_bool_series()` - safe boolean conversion (handles string "True"/"False")
- `normalize_tax_node()` - unified taxonomy node creation
- `export_vocab()` - export vocabularies
- `build_topic_spine()` - build frozen spine
- `add_weights()` - compute w_eff with safe boolean handling
- `aggregate_membership()` - aggregate to long format
- `membership_wide()` - pivot to wide format
- `compute_book_indexes()` - compute book-level statistics
- `split_csv_tags()` - parse CSV tags (single definition)
- `parse_secondary_dim()` - parse secondary dimensions (single definition)
- `compute_w_eff_safe()` - compute effective weights (single definition)
- `build_mask_logic()` - rule engine (single definition)
- `taxonomy_first_mask()` - taxonomy-first mask builder (single definition)

**Cell 5**: Build frozen spine (NEVER REBUILD AFTER THIS)

- Call `build_topic_spine()` once
- Add weights using `add_weights()`
- Parse tags using `split_csv_tags()` and `parse_secondary_dim()`
- Normalize taxonomy using `normalize_tax_node()`
- This creates the immutable `topic_spine` used by all stages

**Cell 6**: Stage 0 Diagnostics

- Print counts (books, topics, tax nodes)
- Export vocabularies for taxonomy, primary tags, secondary dimensions
- Detect and export Radway columns

**Cell 7**: Stage 1 Membership (Taxonomy-only)

- Use `tax_node_name` only
- Aggregate membership
- Compute book indexes
- Export outputs

**Cell 8**: Stage 2 Feature builder design

- Set `STAGE2_MODE` ("within_node_subfeatures" or "augmented_node_defs")
- Define `TAG_USE` columns (detected from `secondary_dim` keys: `activity`, `setting`, `sexual`, etc.)

**Cell 9**: Stage 2 Membership build

- Build features from taxonomy + tags based on `STAGE2_MODE`
- Aggregate and export

**Cell 10**: Stage 3 Design (Radway augmentation)

- Set `RADWAY_COL` (`radway_main_name` or `radway_phase_name`)
- Optionally set `RADWAY_TARGET_TAX_NODES` for targeted augmentation

**Cell 11**: Stage 3 Membership build

- Start from Stage 2 membership
- Add Radway features as overlay
- Aggregate and export

**Cell 12**: Comparability table

- Merge Stage 1, 2, 3 indexes
- Export combined table

**Cell 13**: Optional merge with book metadata

- Merge with `book_wide` if available

#### 2. Remove/Disable Duplicate Definitions

**Delete or disable**:

- All duplicate function definitions (keep only Cell 4 version)
- All duplicate `CORE`/`COMPOSITES` definitions (keep only Stage 1-3 definitions in cells 7-11)
- All duplicate `available_subgroups` assignments
- Old data loading cells (keep only Cell 2)
- Old topic_spine building cells (keep only Cell 5)

**Wrap DEPRECATED section** (cell ~1539):

- Wrap entire deprecated block in `if False:` to prevent execution
- Or delete if not needed for reference

#### 3. Adapt Skeleton to Actual Column Names

**Key adaptations**:

- `book_topic_probs["prob"] `→ rename to `topic_prob` in Cell 2
- Tag parsing uses `primary_categories` and `secondary_categories` columns
- `secondary_dim` is a dict with keys like `activity`, `setting`, `sexual`
- Radway columns: `radway_main_name`, `radway_phase_name`
- Taxonomy: `taxonomy_main_name` → `tax_node_name` (28 nodes)

#### 4. Preserve Existing Stage Definitions

**Extract and preserve**:

- `CORE_STAGE1`, `CORE_STAGE2`, `CORE_STAGE3` definitions (cells 12-13)
- `REFINE_STAGE2`, `REFINE_STAGE3` definitions (cells 13-14)
- Tag lexicons (EXPLICIT_SEXUAL_DIM_VALUES, SOFT_AFFECTION_ACTIVITIES, etc.)

**Integrate into new structure**:

- Stage 1: Use `CORE_STAGE1` mapping to `tax_node_name`
- Stage 2: Use `CORE_STAGE2` + `REFINE_STAGE2` with tag refinement
- Stage 3: Use `CORE_STAGE3` + `REFINE_STAGE3` with Radway augmentation

### Files to Modify

- `notebooks/07_analysis/indexing_hypothesis_testing/indexing_hypothesis_testing.ipynb`
- Replace cells 1-15 with new clean structure (cells 1-13)
- Disable/delete DEPRECATED section (cell ~1539)
- Remove all duplicate function/definition cells
- Keep only one data loading path
- Keep only one topic_spine building path

### Validation

After restructuring:

1. Notebook runs top-to-bottom without errors
2. Each function/definition appears exactly once
3. `topic_spine` is built once and never rebuilt
4. All stages produce consistent outputs
5. DEPRECATED section cannot overwrite active code