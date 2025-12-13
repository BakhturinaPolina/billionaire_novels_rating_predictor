# Model Structure and EDA Guide

## Model Overview

**Model Path:** `models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_radway_mappings`

This is the final BERTopic model containing:
- **368 topics** (excluding outlier topic -1)
- **Taxonomy mappings** (Stage 2): 361 topics mapped to theory-driven categories
- **Radway narrative function mappings** (Stage 3): 361 topics mapped to Radway's 13 functions

## Model Attributes

### Core BERTopic Attributes
- `topic_representations_`: Dict mapping topic_id → list of (keyword, score) tuples
- `topic_labels_`: Dict mapping topic_id → label string
- `custom_labels_`: List of labels (index 0 = topic -1, index 1 = topic 0, etc.)
- `topic_sizes_`: Dict mapping topic_id → number of documents
- `representative_docs_`: Dict mapping topic_id → list of representative document strings

### Custom Attributes (Added in This Project)
- `topic_taxonomy_`: Dict mapping topic_id → taxonomy metadata (361 topics)
- `topic_radway_`: Dict mapping topic_id → Radway function metadata (361 topics)

## Available Fields in the Model

### Taxonomy Fields (Stage 2)
- `main_category_id`: Taxonomy ID (e.g., "4.4")
- `main_category_name`: Human-readable category name
- `main_category_group`: Category group (e.g., "Relationship Trajectory (Main Couple)")
- `secondary_category_id`: Optional secondary taxonomy ID
- `secondary_category_name`: Optional secondary category name
- `secondary_category_group`: Optional secondary category group
- `other_plausible_ids`: List of other plausible taxonomy IDs
- `confidence`: Classification confidence ("low", "medium", "high")
- `is_noise`: Boolean indicating if topic is noise
- `rationale`: Explanation for taxonomy classification

### Radway Fields (Stage 3)
- `radway_main_id`: Primary Radway function ID ("R1" through "R13" or "none")
- `radway_main_name`: Human-readable function name
- `radway_secondary_id`: Optional secondary Radway function ID
- `radway_phase`: Narrative phase ("I", "II", "III", or "NA")
- `radway_phase_name`: Human-readable phase name
- `radway_is_none`: Boolean (True if function is "none")
- `radway_confidence`: Classification confidence ("low", "medium", "high")
- `radway_rationale`: Explanation for Radway classification

## Summary Statistics

From the EDA analysis:

- **Total topics**: 368
- **Topics with labels**: 368 (100%)
- **Topics with taxonomy**: 361 (98.1%)
- **Topics with Radway**: 361 (98.1%)
- **Topics with Radway function**: 246 (66.8% of topics with Radway)
- **Topics with Radway "none"**: 122 (33.2% of topics with Radway)
- **Unique taxonomy categories**: 30
- **Unique taxonomy groups**: 9
- **Unique Radway functions**: 13 (all functions represented)
- **Unique Radway phases**: 4 (I, II, III, NA)

## Top Distributions

### Top Taxonomy Categories
1. Conflict, Distance & Breakup Threats (58 topics)
2. Bonding, Everyday Intimacy & Growth (43 topics)
3. Negative Emotions & Distress (38 topics)
4. Violence, Threats & Coercion (22 topics)
5. Domestic Spaces & Routines (21 topics)

### Top Radway Functions (excluding "none")
1. Hero and heroine are physically or emotionally separated (54 topics) - Phase I
2. Heroine reacts antagonistically to the hero (41 topics) - Phase I
3. Hero treats heroine tenderly (37 topics) - Phase II
4. Heroine interprets hero's behaviour as purely sexual interest (21 topics) - Phase I
5. Heroine responds sexually and emotionally (20 topics) - Phase III

### Phase Distribution
- **Phase I** (Initial Conflict & Isolation): Most common phase
- **Phase II** (Turning Point & Recognition): Moderate representation
- **Phase III** (Commitment & Restoration): Least common phase
- **NA** (None): 122 topics (33.2%)

## How to Load and Explore the Model

### Basic Loading

```python
from bertopic import BERTopic
from pathlib import Path

model_path = Path("models/retrained/paraphrase-MiniLM-L6-v2/stage09_category_mapping/model_1_with_radway_mappings")
model = BERTopic.load(str(model_path))

# Access topic representations
topic_0_keywords = model.topic_representations_[0]
print(f"Topic 0 keywords: {topic_0_keywords[:5]}")

# Access labels
topic_0_label = model.topic_labels_[0]
print(f"Topic 0 label: {topic_0_label}")

# Access taxonomy
if hasattr(model, "topic_taxonomy_") and 0 in model.topic_taxonomy_:
    taxonomy = model.topic_taxonomy_[0]
    print(f"Taxonomy: {taxonomy['main_category_name']} ({taxonomy['main_category_group']})")

# Access Radway
if hasattr(model, "topic_radway_") and 0 in model.topic_radway_:
    radway = model.topic_radway_[0]
    print(f"Radway: {radway['radway_main_name']} (Phase {radway['radway_phase']})")
```

### Using the EDA Script

The EDA script extracts all fields into a DataFrame for easy analysis:

```bash
python -m src.stage09_category_mapping.stage3_radway_functions.scripts.eda_radway_model \
    --output-dir results/stage09_category_mapping/stage3_radway_functions/eda
```

This generates:
- `full_model_data.csv`: Complete DataFrame with all fields
- `full_model_data.parquet`: Same data in Parquet format (faster loading)
- `summary_statistics.json`: Summary statistics
- `taxonomy_distribution.png`: Taxonomy distribution visualizations
- `radway_distribution.png`: Radway function distribution visualizations
- `cross_tabulations.png`: Cross-tabulations between taxonomy and Radway

### Loading the EDA DataFrame

```python
import pandas as pd

# Load the full dataset
df = pd.read_parquet("results/stage09_category_mapping/stage3_radway_functions/eda/full_model_data.parquet")

# Filter topics with Radway functions (excluding "none")
df_with_functions = df[df["radway_is_none"] == False]

# Filter by phase
df_phase_i = df[df["radway_phase"] == "I"]
df_phase_ii = df[df["radway_phase"] == "II"]
df_phase_iii = df[df["radway_phase"] == "III"]

# Filter by taxonomy group
df_relationship = df[df["taxonomy_main_group"] == "Relationship Trajectory (Main Couple)"]
```

## Suggested EDA Analyses

### 1. Narrative Progression Analysis
- Track how topics progress through Radway phases (I → II → III)
- Identify which taxonomy categories are associated with each phase
- Analyze the relationship between conflict (Phase I) and resolution (Phase III)

### 2. Taxonomy-Radway Relationships
- Which taxonomy categories map to which Radway functions?
- Are certain taxonomy groups more likely to be "none" in Radway?
- Cross-tabulation analysis between taxonomy groups and Radway phases

### 3. Function Distribution Analysis
- Which Radway functions are most/least common?
- Are there functions that rarely appear?
- Distribution of functions across different taxonomy groups

### 4. Confidence Analysis
- Compare taxonomy confidence vs Radway confidence
- Identify topics with low confidence in both systems
- Topics with high confidence mismatches

### 5. Topic Quality Analysis
- Topics with "none" in Radway: are they background/setting topics?
- Topics with both taxonomy and Radway: are they core narrative topics?
- Topics missing taxonomy or Radway: data quality issues?

### 6. Phase-Specific Analysis
- **Phase I topics**: Conflict, isolation, antagonism
- **Phase II topics**: Turning point, tenderness, reinterpretation
- **Phase III topics**: Commitment, restoration, HEA

### 7. Book-Level Aggregation
- Aggregate topics by book to track narrative patterns
- Compare narrative structure across rating groups (bad/mid/good)
- Identify books with unusual narrative patterns

## Example Analysis Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_parquet("results/stage09_category_mapping/stage3_radway_functions/eda/full_model_data.parquet")

# 1. Phase distribution
phase_counts = df["radway_phase_name"].value_counts()
print("Phase distribution:")
print(phase_counts)

# 2. Top Radway functions (excluding none)
radway_functions = df[df["radway_is_none"] == False]["radway_main_name"].value_counts()
print("\nTop Radway functions:")
print(radway_functions.head(10))

# 3. Taxonomy group vs Radway phase
crosstab = pd.crosstab(df["taxonomy_main_group"], df["radway_phase_name"])
print("\nTaxonomy Group vs Radway Phase:")
print(crosstab)

# 4. Confidence comparison
conf_comparison = pd.crosstab(df["taxonomy_confidence"], df["radway_confidence"])
print("\nTaxonomy vs Radway Confidence:")
print(conf_comparison)

# 5. Topics with both high taxonomy and Radway confidence
high_conf_topics = df[
    (df["taxonomy_confidence"] == "high") & 
    (df["radway_confidence"] == "high") &
    (df["radway_is_none"] == False)
]
print(f"\nTopics with high confidence in both: {len(high_conf_topics)}")
```

## Next Steps

1. **Book-Level Analysis**: Aggregate topics by book to analyze narrative patterns
2. **Rating Comparison**: Compare narrative structure between bad/mid/good rated books
3. **Temporal Analysis**: Track narrative progression through book chapters
4. **Function Co-occurrence**: Analyze which Radway functions appear together
5. **Taxonomy-Radway Validation**: Validate that taxonomy categories align with Radway functions as expected

## Files Generated by EDA

- `full_model_data.csv`: Complete dataset (CSV format)
- `full_model_data.parquet`: Complete dataset (Parquet format, recommended)
- `summary_statistics.json`: Summary statistics
- `taxonomy_distribution.png`: Taxonomy visualizations
- `radway_distribution.png`: Radway function visualizations
- `cross_tabulations.png`: Cross-tabulation heatmaps

