# Correct Cell Execution Order

## Current Problem
Cell 4 (COMPOSITES) runs before Cell 6 (GATING + COMPOSITE SELECTION), but Cell 4 needs `CompositeSpec` which is defined in Cell 6.

## Correct Execution Order

### Cell 0: Markdown Header
- **Type**: Markdown
- **Purpose**: Documentation
- **Action**: No execution needed

### Cell 1: CONFIG
- **Type**: Code
- **Purpose**: Sets up paths, constants, and configuration
- **Dependencies**: None
- **Action**: Run first

### Cell 2: Data Loading
- **Type**: Code  
- **Purpose**: Loads topic_lookup, book_topic_probs, author_dom, topic_health
- **Dependencies**: Cell 1 (CONFIG)
- **Action**: Run second

### Cell 3: (Currently Cell 6) - GATING + COMPOSITE SELECTION
- **Type**: Code
- **Purpose**: Defines CompositeSpec class, helper functions, and builds gate table
- **Dependencies**: Cell 2 (needs topic_lookup, author_dom, topic_health)
- **Action**: Run third - **THIS MUST RUN BEFORE COMPOSITES CELL**
- **Contains**:
  - `CompositeSpec` class definition
  - Helper functions (`_split_csvish`, `combined_text`, `regex_gate`, etc.)
  - `build_topic_gate_table()` function
  - `gate` variable
  - `select_topics_for_composite()` function
  - Constants: `DEFAULT_MIN_TOPICS`, `SMALL_GROUP_POLICY`

### Cell 4: COMPOSITES
- **Type**: Code
- **Purpose**: Defines the COMPOSITES dictionary with all composite specifications
- **Dependencies**: Cell 3 (needs CompositeSpec class)
- **Action**: Run fourth

### Cell 5: (Currently Cell 9) - BUILD COMPOSITE TOPIC SETS + AUDIT
- **Type**: Code
- **Purpose**: Builds composite_topics dictionary and audit table
- **Dependencies**: Cell 4 (needs COMPOSITES dict), Cell 3 (needs gate, select_topics_for_composite)
- **Action**: Run fifth

### Remaining Cells
- Continue with the rest of the notebook in their current order

## How to Fix in Jupyter

### Option 1: Manual Reordering (Recommended)
1. In Jupyter, use the cell toolbar to move cells:
   - Right-click on Cell 6 (GATING + COMPOSITE SELECTION)
   - Select "Move Cell Up" until it's positioned right after Cell 2
   - It should become the new Cell 3

### Option 2: Cut and Paste
1. Select Cell 6 (GATING + COMPOSITE SELECTION) - select the entire cell
2. Cut it (Ctrl+X or Cmd+X)
3. Click after Cell 2 (Data Loading)
4. Paste it (Ctrl+V or Cmd+V)
5. It will become the new Cell 3

### Option 3: Run Cells in Correct Order
If you can't reorder, just run cells in this order:
1. Cell 0 (skip - markdown)
2. Cell 1 (CONFIG)
3. Cell 2 (Data Loading)
4. **Cell 6 (GATING + COMPOSITE SELECTION)** ← Run this before Cell 4!
5. Cell 4 (COMPOSITES)
6. **Cell 9 (BUILD COMPOSITE TOPIC SETS)** ← Run this after Cell 4
7. Continue with remaining cells

## Verification
After reordering, the execution order should be:
```
Cell 0: Markdown (skip)
Cell 1: CONFIG ✓
Cell 2: Data Loading ✓
Cell 3: GATING + COMPOSITE SELECTION ✓ (defines CompositeSpec)
Cell 4: COMPOSITES ✓ (uses CompositeSpec)
Cell 5: BUILD COMPOSITE TOPIC SETS + AUDIT ✓
... rest of cells
```

## Quick Check
Run this in a cell to verify order:
```python
# Check if CompositeSpec exists before COMPOSITES cell
print("CompositeSpec defined:", 'CompositeSpec' in globals())
print("gate defined:", 'gate' in globals())
print("COMPOSITES defined:", 'COMPOSITES' in globals())
```

