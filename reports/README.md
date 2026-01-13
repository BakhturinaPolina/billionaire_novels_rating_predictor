# Reports Directory

This directory contains all research reports, findings, and technical documentation organized by pipeline stage and analysis type.

## Structure Overview

```
reports/
├── 01_stage_reports/              # Technical reports organized by pipeline stage
│   ├── stage01_ingestion/         # Stage 01: Data ingestion (empty - reports to be added)
│   ├── stage02_preprocessing/     # Stage 02: Data preprocessing (empty - reports to be added)
│   ├── stage03_modeling/          # Stage 03: Initial model training (empty - reports to be added)
│   ├── stage04_selection/         # Stage 04: Model selection & Pareto analysis
│   ├── stage05_retraining/        # Stage 05: Model retraining (empty - reports to be added)
│   ├── stage06_topic_exploration/ # Stage 06: Topic exploration (empty - reports to be added)
│   ├── stage07_topic_quality/     # Stage 07: Topic quality analysis (empty - reports to be added)
│   ├── stage08_llm_labeling/       # Stage 08: LLM labeling experiments & comparisons
│   ├── stage09_category_mapping/   # Stage 09: Category mapping (3 sub-stages)
│   └── stage10_correlation_analysis/ # Stage 10: Data preparation & correlation analysis
│
└── 02_findings/                   # Research findings and results
    ├── hypothesis_testing/         # Hypothesis testing results
    └── methodology_llm_labeling_and_taxonomy/ # LLM labeling and taxonomy methodology documentation
```

## Directory Descriptions

### 01_stage_reports/

Technical reports and documentation organized by pipeline stage. Each stage folder contains reports documenting methodology, decisions, and technical details. Folder names match the corresponding stages in `src/`.

#### stage01_ingestion/
(Empty - reports to be added)

#### stage02_preprocessing/
(Empty - reports to be added)

#### stage03_modeling/
- **modeling_and_retraining_report.md**: Comprehensive report documenting the modeling and retraining pipeline (Stages 03 and 05), including character name exclusion methodology and implementation details

#### stage04_selection/
- **pareto_efficient_model_selection.md**: Pareto efficiency analysis for model selection
- **pareto_analysis_results.md**: Detailed Pareto analysis results
- **hyperparameter_correlation_analysis.md**: Analysis of hyperparameter correlations

#### stage05_retraining/
(Empty - reports to be added)

#### stage06_topic_exploration/
(Empty - reports to be added)

#### stage07_topic_quality/
(Empty - reports to be added)

#### stage08_llm_labeling/
- **model_comparison.md**: Comprehensive comparison of 6 LLM models for topic labeling (most recent evaluation)
- **model_evaluation_criteria.md**: Criteria checklist used for model evaluation
- **model_and_prompt_reasoning.md**: Detailed reasoning for model and prompt choices (comprehensive 25KB document)
- **prompts.md**: Complete prompt documentation including snippets design, formatting, and theoretical reasoning

#### stage09_category_mapping/
Category mapping is organized into three sub-stages:

- **01_stage1_natural_clusters/**: Natural cluster discovery reports
  - duplicate_labels.md
  - hierarchical_topics_exploration.md
  - initial_step.md
  - metadata_attachment.md
  - probabilities_decision.md

- **02_stage2_theory_driven_categories/**: Theory-driven taxonomy mapping
  - model_comparison.md

- **03_stage3_radway_functions/**: Radway narrative function mapping
  - (Technical docs remain in `src/` directory)

#### stage10_correlation_analysis/
- **data_preparation_guide.md**: Guide for data preparation pipeline
- **fix_nan_probabilities.md**: Documentation of NaN probability fix
- **measurement_pipeline_composite_indices.md**: Measurement pipeline and composite indices documentation

### 02_findings/

Research findings and methodology documentation organized by analysis type.

#### hypothesis_testing/
Contains hypothesis testing results and subdirectories:
- **hypothesis_testing_results.md**: Main results summary
- **readme.md**: Overview of hypothesis testing structure
- **arc_analysis/**: Narrative arc analysis results
- **mass_appeal/**: Popularity/popularity predictor results
- **perceived_quality/**: Rating quality predictor results
- **validation_methods/**: Validation technique results

#### methodology_llm_labeling_and_taxonomy/
Research methodology documentation for LLM-based topic labeling and taxonomy mapping:
- **research_methodology.md**: Comprehensive research methodology document (draft research article section)
- **model_comparison_summary.md**: Summary of model comparisons for labeling
- **structure.md**: Structure documentation
- **readme.md**: Overview and documentation

## Naming Convention

### Stage Reports
- **Folder names**: Match `src/` stage folder names (e.g., `stage04_selection`, `stage08_llm_labeling`)
- **File names**: Descriptive, without redundant numbering prefixes (e.g., `pareto_analysis_results.md` instead of `01_pareto_analysis_results.md`)

### Findings
- **Folder names**: Descriptive names without numbering prefixes (e.g., `hypothesis_testing`, `methodology_llm_labeling_and_taxonomy`)
- **File names**: Descriptive, without redundant numbering where appropriate

## File Locations

### Source Code Documentation
Technical implementation documentation (README.md files, code comments) remains in the `src/` directory structure. Only reports and findings are moved to `reports/`.

## Quick Reference

**Looking for...**
- **Model comparison reports?** → `01_stage_reports/stage08_llm_labeling/`
- **Hypothesis testing results?** → `02_findings/hypothesis_testing/`
- **Data preparation guides?** → `01_stage_reports/stage10_correlation_analysis/`
- **Category mapping reports?** → `01_stage_reports/stage09_category_mapping/`
- **Pareto analysis?** → `01_stage_reports/stage04_selection/`
- **Modeling and retraining methodology?** → `01_stage_reports/stage03_modeling/`
- **LLM labeling methodology?** → `02_findings/methodology_llm_labeling_and_taxonomy/`

## Migration History

This structure was reorganized to align with `src/` stage naming conventions. All stage report folders now use the `stage##_name` format to match the source code structure.
