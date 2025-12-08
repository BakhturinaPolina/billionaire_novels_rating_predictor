# Folder Renaming Plan

## Current State Analysis

### Results Folders → Source Code Mapping

| Results Folder | Source Stage | Code References | Status |
|----------------|--------------|-----------------|--------|
| `results/pareto/` | `stage04_selection/` | `results/pareto` | ✅ Consistent |
| `results/best_model_topics_exploration/` | `stage06_best_model_topics_exploration/` | `results/stage06_exploration` | ❌ Inconsistent |
| `results/EDA_topics_cleaning/` | `stage07_eda_topics_cleaning/` | `results/stage06_eda` | ❌ Inconsistent |
| `results/labeling_openrouter/` | `stage08_LLM_generated_labeling/` | `results/stage06_labeling_openrouter` | ❌ Inconsistent |
| `results/category_mapping/` | `stage_09_category_mapping/` | `results/stage06_labeling/category_mapping` | ❌ Inconsistent |
| `results/OCTIS experiments/` | `stage03_modeling/` | Various | ❌ Inconsistent (space in name) |
| `results/figures/` | `stage10_correlation_analysis/` | `results/figures` | ✅ Consistent |

## Proposed Better Names

### Results Folders (Stage-based, descriptive)

1. **`results/stage04_pareto_analysis/`** (or keep `pareto/` - already clear)
   - Current: `results/pareto/`
   - Source: `stage04_selection/`
   - **Decision: Keep `pareto/`** (short, clear, widely used)

2. **`results/stage06_topic_exploration/`**
   - Current: `results/best_model_topics_exploration/`
   - Source: `stage06_best_model_topics_exploration/`
   - Contains: metrics JSON, topics JSON with all representations

3. **`results/stage07_topic_quality/`**
   - Current: `results/EDA_topics_cleaning/`
   - Source: `stage07_eda_topics_cleaning/`
   - Contains: topic quality CSV, noise candidates CSV

4. **`results/stage08_llm_labeling/`**
   - Current: `results/labeling_openrouter/`
   - Source: `stage08_LLM_generated_labeling/`
   - Contains: LLM-generated labels JSON files

5. **`results/stage09_category_mapping/`**
   - Current: `results/category_mapping/`
   - Source: `stage_09_category_mapping/`
   - Contains: hierarchical clustering, category mapping results

6. **`results/stage03_model_evaluation/`**
   - Current: `results/OCTIS experiments/` (has space - problematic)
   - Source: `stage03_modeling/`
   - Contains: model evaluation results CSV

7. **`results/stage10_figures/`** (or keep `figures/` - already clear)
   - Current: `results/figures/`
   - Source: `stage10_correlation_analysis/`
   - **Decision: Keep `figures/`** (short, clear, widely used)

### Source Folders (More descriptive, consistent)

1. **`stage04_selection/`** → **`stage04_pareto_selection/`**
   - More descriptive of what it does

2. **`stage06_best_model_topics_exploration/`** → **`stage06_topic_exploration/`**
   - Shorter, clearer, matches results folder

3. **`stage07_eda_topics_cleaning/`** → **`stage07_topic_quality/`**
   - More descriptive, matches results folder

4. **`stage08_LLM_generated_labeling/`** → **`stage08_llm_labeling/`**
   - Shorter, clearer, matches results folder

5. **`stage_09_category_mapping/`** → **`stage09_category_mapping/`**
   - Fix underscore to hyphen for consistency

## Implementation Plan

1. Rename results folders
2. Rename source folders  
3. Update all path references in code
4. Update config files if needed
5. Update README/documentation

