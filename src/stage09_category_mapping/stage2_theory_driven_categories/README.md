# Stage 2: Theory-Driven Categories (Luxury/Emotion/Erotica)

## Overview

**Goal**: Map existing BERTopic topics to predefined theoretical categories (luxury lifestyle, emotional depth, erotic content) using zero-shot classification. This stage is only needed if Stage 1's natural clusters don't align with research questions.

**Status**: 📋 Planned (implement after Stage 1 evaluation)

## When to Run This Stage

- ✅ Stage 1 natural clusters don't clearly map to theory categories
- ✅ You need explicit theory-driven categories regardless of natural structure
- ❌ Skip if Stage 1 already produces interpretable theory-aligned meta-topics

## Approach

### Primary Method: Zero-Shot Classification

Map existing topics (from Stage 1 or original model) to theory categories without retraining.

**Categories**:
1. **Luxury & Lifestyle Promise**: Billionaire wealth, luxury goods, lifestyle markers
2. **Explicit Sexual Content**: High-explicitness erotic scenes
3. **Non-Explicit Sensual Content**: Romantic tension, light sensuality
4. **Psychological/Emotional Relationship Content**: Inner thoughts, emotional processing, relationship depth
5. **Non-Romantic Plot/Other**: Work, family, external conflict, etc.

### Alternative Methods (if zero-shot insufficient)

- **Semi-Supervised BERTopic**: Retrain with partial labels to bias topics toward categories
- **Guided Topics**: Use seed words to force specific category topics

## Implementation Outline

1. **Prepare topic representations**: Combine topic label, LLM description, top keywords, representative sentences
2. **Define category labels**: Create natural-language descriptions for each theory category
3. **Run zero-shot classification**: Use transformer-based zero-shot classifier (e.g., `transformers` pipeline)
4. **Map topics to categories**: Assign each topic to primary (and optionally secondary) category
5. **Aggregate per book**: Compute category proportions per book
6. **Statistical analysis**: Test category differences across rating classes (similar to Stage 1)

## Expected Outputs

- Topic-to-category mapping table
- Book-level category proportions
- Statistical test results (category × rating class)
- Visualization of category prevalence by quality

## Dependencies

```python
from transformers import pipeline
import pandas as pd
```

## Next Steps

See detailed implementation plan after Stage 1 completion and evaluation.

