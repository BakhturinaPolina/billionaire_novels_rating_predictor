# Stage 10 Correlation Analysis Reports

## Current Reports

**`stage10_data_preparation_pipeline_comprehensive_report.md`** - Comprehensive single report documenting the complete data preparation pipeline.

**`stage10_statistical_results_pilot_n92.md`** - Statistical results from pilot analysis (N=92): themes, reach, and perceived quality effects.

This report consolidates information from:
- `data_preparation_guide.md` (archived)
- `fix_nan_probabilities.md` (archived)
- `measurement_pipeline_composite_indices.md` (archived)
- Analysis of data_preparation scripts (01-04)

## Statistical Results Report

**`stage10_statistical_results_pilot_n92.md`** includes:

1. **Macro-axis definitions** - Weighted combinations of CORE predictors
2. **Macro-axis level effects** - Effects on reach and quality outcomes
3. **CORE predictor level effects** - Top predictors for reach and quality
4. **Arc effects** - Narrative pacing effects (begin/middle/end changes)
5. **Predictive performance** - Cross-validation results
6. **Figures** - Visualization links for all key results
7. **Interpretation** - Clear summary of findings

**Key Findings:**
- Status/dominance axis predicts reach (visibility) but not quality
- Payoff/safety axis predicts both reach and quality
- Explicit erotics negatively associated with reach
- Late-story crisis escalation associated with higher ratings
- Pilot sample: N=92 (effect-size focused, not p-value focused)

## Data Preparation Report Structure

The consolidated data preparation report includes:

1. **Executive Summary** - Overview of pipeline and key outputs
2. **Pipeline Overview** - Data flow and script execution order
3. **Script Documentation** - Detailed documentation for each script:
   - Script 03: Generate Topic Probabilities
   - Script 04: Generate Tertile Topic Probabilities
   - Script 01: Data Validation & Extraction
   - Script 02: Book Aggregation
4. **Derived Indices** - Documentation of all hypothesis-aligned indices
5. **Data Quality & Validation** - Normalization, ID alignment, NaN handling
6. **Directory Structure** - Complete output file organization
7. **Troubleshooting** - Common issues and solutions
8. **Execution Workflow** - Complete pipeline run commands
9. **Next Steps** - Guidance for downstream analysis
10. **Key Design Decisions** - Rationale for pipeline choices
11. **Dependencies** - Required packages and data files
12. **Reproducibility** - Version control and caching
13. **Appendix** - File format specifications

## Archived Reports

Old reports have been archived to:
`archive/stage10_reports_old/`

- `data_preparation_guide.md` - Original detailed guide (now integrated)
- `fix_nan_probabilities.md` - NaN handling documentation (now integrated)
- `measurement_pipeline_composite_indices.md` - Composite index methodology (referenced, not integrated - see separate documentation)

## Related Documentation

- **Statistical Analysis**: `src/stage10_correlation_analysis/docs/STATISTICAL_ANALYSIS_REPORT.md`
- **Visualization Examples**: `src/stage10_correlation_analysis/docs/VISUALIZATION_EXAMPLES.md`
- **Composite Indices**: See `measurement_pipeline_composite_indices.md` in archive for detailed composite construction methodology

