# Figures Directory

This directory contains figures and visualizations for Stage 08 (LLM Labeling) and Stage 09 (Category Mapping) methodology and results.

## Expected Figures

### Methodology Figures

1. **`pipeline_diagram.png`**
   - **Description**: Flowchart showing the three-stage pipeline (Stage 08 → Stage 09 Stage 2 → Stage 09 Stage 3)
   - **Type**: Flowchart/diagram
   - **Use**: Research article methodology section, presentation overview

2. **`prompt_architecture.png`**
   - **Description**: System prompt components and user prompt template structure
   - **Type**: Diagram/architecture
   - **Use**: Research article methodology section

3. **`model_comparison_results.png`**
   - **Description**: Bar chart comparing success rates, average words per label, and keyword copying across tested models
   - **Type**: Bar chart
   - **Use**: Research article methodology section, presentation model selection slide

### Results Figures

4. **`taxonomy_group_distribution.png`**
   - **Description**: Bar chart or pie chart showing distribution of topics across 6 taxonomy groups
   - **Type**: Bar chart or pie chart
   - **Use**: Research article results section, presentation results slide

5. **`radway_phase_distribution.png`**
   - **Description**: Bar chart showing distribution of topics across Radway phases (Phase I, II, III, and "none")
   - **Type**: Bar chart
   - **Use**: Research article results section, presentation results slide

6. **`taxonomy_radway_heatmap.png`**
   - **Description**: Heatmap showing which taxonomy categories map to which Radway functions
   - **Type**: Heatmap
   - **Use**: Research article results section, presentation mapping patterns slide

7. **`label_quality_examples.png`**
   - **Description**: Table or visualization showing example labels for different topic types
   - **Type**: Table or visualization
   - **Use**: Research article results section, presentation examples slide

## Figure Specifications

### Recommended Dimensions
- **Research Article**: 600-800px width, maintain aspect ratio
- **Presentation**: 1200-1600px width for high-resolution displays

### Format
- **Primary Format**: PNG (for presentations and web)
- **Alternative**: PDF (for research articles, vector graphics)

### Color Scheme
- Use consistent color palette across all figures
- Ensure accessibility (colorblind-friendly)
- Consider grayscale versions for print

## Data Sources

Figures should be generated from:
- Model comparison results: `reports/01_stage_reports/stage08_llm_labeling/stage08_llm_model_comparison_report.md`
- Taxonomy distribution: `results/stage09_category_mapping/stage2_theory_driven_categories/`
- Radway distribution: `results/stage09_category_mapping/stage3_radway_functions/eda/`
- Label examples: `results/stage08_llm_labeling/labels_pos_openrouter_*.json`

## Usage in Documents

Figures are referenced in:
- `stage08_research_article_draft_summary.md` (Section 8: Figures and Tables)
- `stage08_presentation_summary.md` (Appendix: Figure Placeholders)

## Notes

- Figures can be generated using Python (matplotlib, seaborn, plotly) or R
- Consider creating interactive versions (HTML) for web presentations
- Maintain consistent styling across all figures
- Include figure captions and legends as appropriate

