# LLM Labeling and Taxonomy Mapping Methodology

This directory contains detailed documentation about the LLM-based topic labeling and taxonomy mapping methodology used in this research project. These reports serve as:

1. **Research article drafts** - Detailed methodology sections for the research article
2. **Reproducibility documentation** - Complete methodology and implementation details for other researchers
3. **Internal reference** - Comprehensive documentation of design decisions and computational strategies

## Contents

### Main Research Report
- `00_research_methodology_report.md` - **Comprehensive methodology report** covering:
  - Theoretical foundations of LLM-based labeling and zero-shot classification
  - Methodological approach for all three stages (LLM labeling, taxonomy mapping, Radway mapping)
  - Computational tools and strategies (OpenRouter, model selection, processing strategies)
  - Prompt concepts and design patterns
  - Quality assurance and validation
  - Limitations and future directions

### Model Comparison Reports
- `01_model_comparison_summary.md` - Comparison of different LLM models for topic labeling (Mistral-Nemo, Grok, DeepSeek, etc.)

### Data Structure Documentation
- `02_structure.md` - Structure and organization of data preparation outputs

## Related Documentation

For implementation details, see:
- `reports/01_stage_reports/08_llm_labeling/` - Detailed reports on LLM labeling implementation
- `reports/01_stage_reports/09_category_mapping/` - Detailed reports on taxonomy and Radway mapping

For code implementation, see:
- `src/stage08_llm_labeling/` - LLM labeling code and OpenRouter experiments
- `src/stage09_category_mapping/` - Taxonomy and Radway mapping code

## Usage

These reports are drafts for research article methodology sections and may contain:
- Detailed theoretical foundations
- Methodological justifications
- Computational implementation details
- Design rationale and trade-offs
- Quality assurance procedures

For final results and statistical analysis, see:
- `reports/02_findings/01_hypothesis_testing/` - Hypothesis testing results
- `reports/02_findings/02_exploratory_analysis/` - Exploratory data analysis

