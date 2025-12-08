# Stage 3: Radway Narrative Functions

## Overview

**Goal**: Map sentences/topics to Radway's 13 narrative functions to analyze story structure and track narrative progression across books. Compare narrative patterns between bad/mid/good rated books.

**Status**: 📋 Planned (can run independently or after Stages 1-2)

## Radway's 13 Functions

1. Initial conflict & isolation
2. Heroine's isolation
3. Hero's initial coldness
4. Heroine's vulnerability
5. Hero's recognition of heroine's specialness
6. Hero's transformation
7. Heroine's recognition of hero's transformation
8. Mutual recognition
9. Commitment & restoration
10. HEA (Happily Ever After)
11. (Additional functions as per your Radway framework)

## Approach

### Primary Method: Zero-Shot Classification + Topics Over Time

1. **Zero-shot classification**: Map each sentence to Radway function
   - Input: Sentence text with context (chapter, position)
   - Labels: 13 Radway functions + "none of the above"
   - Use natural-language descriptions of each function

2. **Topics over time**: Track function prevalence across narrative phases
   - Use chapter/position as "time" dimension
   - Bin into beginning/middle/end (or more granular)
   - Compare patterns for bad/mid/good books

3. **Optional**: Aggregate at topic level
   - Which BERTopic topics map to which Radway functions?
   - Use for cross-validation with topic-based analysis

## Implementation Outline

1. **Prepare sentence data**: Ensure chapter/position metadata available
2. **Define Radway function labels**: Natural-language descriptions from Radway summaries
3. **Run zero-shot classification**: Assign function to each sentence
4. **Compute topics over time**: Use BERTopic's `topics_over_time` with chapter/position
5. **Narrative phase analysis**: Aggregate functions by beginning/middle/end
6. **Quality comparisons**: Compare narrative progression patterns across rating classes
7. **Visualization**: Plot function prevalence over narrative time for each quality group

## Expected Outputs

- Per-sentence Radway function labels
- Topic-to-function mapping (optional)
- Narrative progression plots (function × time × quality)
- Statistical comparisons of narrative structure

## Dependencies

```python
from bertopic import BERTopic
from transformers import pipeline
import pandas as pd
import plotly.graph_objects as go
```

## Key Research Questions

- Do good books show earlier emotional processing?
- Is explicit sex front-loaded in lower-rated books?
- Do good books follow Radway's structure more closely?
- How does narrative arc differ by quality?

## Next Steps

See detailed implementation plan when ready to proceed.

