# Category Mapping: Three-Stage Plan for Topics-to-Categories Mapping

## Overview

This directory contains the implementation plan and code for mapping BERTopic topics to interpretable categories using a three-stage approach. The goal is to transform ~300 fine-grained topics into meaningful groups that can be analyzed for their relationship to book quality (bad/mid/good ratings).

## Prerequisites

Before starting, ensure you have:

- **Trained BERTopic model**: Located at `models/retrained/paraphrase-MiniLM-L6-v2/model_1_with_categories/`
- **LLM labels/descriptions**: Topic labels and descriptions generated in previous stages
- **Sentence-level corpus**: With book metadata (ratings, chapters, etc.)
- **Data files**:
  - `data/processed/chapters.csv`: Sentence-level data with book/chapter metadata
  - `data/processed/goodreads.csv`: Book-level ratings and metadata

## Three-Stage Approach

### Stage 1: Natural Clusters (Hierarchical Topics)
**Goal**: Discover data-driven topic groupings without theoretical priors.

**Method**: Use BERTopic's hierarchical topics to build a tree structure over existing topics, then reduce to interpretable meta-topics (40-80 topics). Test which natural clusters are associated with book quality.

**Key Tools**: 
- Hierarchical topics
- Topic reduction
- Topics per class analysis
- ANOVA testing

**Output**: Meta-topics with statistical significance for quality differences.

**Status**: ⚠️ **Next to implement** - Detailed plan in `stage1_natural_clusters/README.md`

---

### Stage 2: Theory-Driven Categories (Luxury/Emotion/Erotica)
**Goal**: Map topics to predefined theoretical categories (luxury lifestyle, emotional depth, erotic content).

**Method**: Use zero-shot classification to map existing topics to theory categories. If results are unsatisfactory, optionally retrain with semi-supervised or guided BERTopic.

**Key Tools**:
- Zero-shot topic classification (primary)
- Semi-supervised BERTopic (optional, if zero-shot insufficient)
- Guided topics (optional, if strong priors needed)

**Output**: Topic-to-category mappings with confidence scores.

**Status**: 📋 Planned - See `stage2_theory_driven_categories/README.md`

---

### Stage 3: Radway Narrative Functions
**Goal**: Map sentences/topics to Radway's 13 narrative functions to analyze story structure.

**Method**: Zero-shot classification to Radway functions, combined with topics-over-time analysis to track narrative progression.

**Key Tools**:
- Zero-shot classification (Radway's 13 functions)
- Topics over time (using chapter/position as "time")
- Narrative phase analysis (beginning/middle/end)

**Output**: Per-sentence Radway function labels, narrative progression plots, quality comparisons.

**Status**: 📋 Planned - See `stage3_radway_functions/README.md`

---

## Decision Logic

### When to Skip Stage 2

If Stage 1 naturally produces interpretable clusters that align with your research questions (e.g., clear "luxury", "emotion", "erotica" meta-topics), you may skip Stage 2 and proceed directly to Stage 3.

### When to Use Each Stage

- **Stage 1**: Always run first to understand natural structure
- **Stage 2**: Run if Stage 1 doesn't yield theory-aligned clusters OR if you need explicit theory-driven categories
- **Stage 3**: Run for narrative structure analysis (independent of Stages 1-2)

## BERTopic Features Used by Stage

| Feature | Stage 1 | Stage 2 | Stage 3 |
|---------|---------|---------|---------|
| **Hierarchical Topics** | ✅ Core | 🔶 Helpful | 🔶 Optional |
| **Topics Over Time** | ❌ Not needed | ❌ Not needed | ✅ Core |
| **Semi-Supervised** | ❌ Not needed | 🔶 Optional | 🔶 Optional |
| **Guided Topics** | ❌ Not appropriate | 🔶 Optional | ❌ Less ideal |
| **Zero-Shot** | ❌ Not needed | ✅ Excellent | ✅ Excellent |

## Directory Structure

```
category_mapping/
├── README.md (this file)
├── stage1_natural_clusters/
│   ├── README.md (detailed implementation plan)
│   └── [code files to be created]
├── stage2_theory_driven_categories/
│   ├── README.md (short plan)
│   └── [code files to be created]
└── stage3_radway_functions/
    ├── README.md (short plan)
    └── [code files to be created]
```

## Next Steps

1. **Start with Stage 1**: Read `stage1_natural_clusters/README.md` for detailed implementation steps
2. **Evaluate results**: After Stage 1, decide if Stage 2 is needed
3. **Proceed to Stage 3**: For narrative structure analysis

## References

- BERTopic Documentation: https://maartengr.github.io/BERTopic/
- Radway, J. (1984). *Reading the Romance: Women, Patriarchy, and Popular Literature*
- Project-specific data contracts: `docs/DATA_CONTRACTS.md`

