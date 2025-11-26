# Documentation Structure Guide

## Overview

This document outlines the recommended documentation structure for the project. The goal is to provide comprehensive documentation without overwhelming readers with too many details in a single file.

## Main Documentation Files

### Root Level

1. **`README.md`** ✅
   - **Purpose**: Project overview, quick start, basic usage
   - **Audience**: New users, contributors, general audience
   - **Content**: Installation, quick start, project structure, basic pipeline overview
   - **Length**: ~200-300 lines

2. **`SCIENTIFIC_README.md`** ✅
   - **Purpose**: Detailed research methodology, hypotheses, research questions
   - **Audience**: Researchers, reviewers, academic readers
   - **Content**: Research objectives, hypotheses (H1-H6), dataset description, methodology, statistical analysis plan
   - **Length**: ~400-600 lines

3. **`LICENSE`** ✅
   - Standard license file

## Documentation Directory (`docs/`)

### Core Documentation

1. **`docs/METHODOLOGY.md`** ✅
   - **Purpose**: Technical implementation details
   - **Audience**: Developers, technical researchers
   - **Content**: Algorithms, GPU acceleration, data flow, configuration management
   - **Length**: ~300-400 lines

2. **`docs/DATA_CONTRACTS.md`** ✅
   - **Purpose**: Input/output data specifications
   - **Audience**: Developers, data engineers
   - **Content**: File formats, schemas, validation rules, data contracts per stage
   - **Length**: ~300-400 lines

3. **`docs/INDICES.md`** ✅
   - **Purpose**: Derived index definitions and formulas
   - **Audience**: Researchers, analysts
   - **Content**: All 10+ index definitions, formulas, interpretations, hypotheses
   - **Length**: ~200-300 lines

4. **`docs/DOCUMENTATION_STRUCTURE.md`** ✅ (this file)
   - **Purpose**: Guide to documentation organization
   - **Audience**: Contributors, maintainers

### Suggested Additional Documentation

5. **`docs/RESULTS.md`** (Suggested)
   - **Purpose**: Summary of research findings
   - **Audience**: Researchers, readers
   - **Content**: Hypothesis test results, key findings, visualizations summary
   - **Length**: ~200-300 lines
   - **Status**: To be created after analysis completion

6. **`docs/REPRODUCIBILITY.md`** (Suggested)
   - **Purpose**: Reproducibility guide
   - **Audience**: Researchers wanting to reproduce results
   - **Content**: Environment setup, seed values, version information, step-by-step reproduction
   - **Length**: ~200-300 lines

7. **`docs/TROUBLESHOOTING.md`** (Suggested)
   - **Purpose**: Common issues and solutions
   - **Audience**: Users encountering problems
   - **Content**: GPU setup issues, memory errors, data validation problems, FAQ
   - **Length**: ~150-200 lines

8. **`docs/CONTRIBUTING.md`** (Suggested)
   - **Purpose**: Contribution guidelines
   - **Audience**: Contributors
   - **Content**: Code style, testing, pull request process, development setup
   - **Length**: ~150-200 lines

## Stage-Specific Documentation

### In `src/stage*/README.md`

Each stage has its own README with:
- Overview and status
- Key functionality
- Usage examples
- Input/output specifications
- Implementation notes

**Current Status:**
- ✅ `stage03_modeling/README.md` (exists)
- ✅ `stage05_retraining/README.md` (exists)
- ✅ `stage01_ingestion/README.md` (created)
- ✅ `stage02_preprocessing/README.md` (created)
- ✅ `stage04_selection/README.md` (created)
- ✅ `stage06_labeling/README.md` (created)
- ✅ `stage07_analysis/README.md` (created)

## Notebook Documentation

### In `notebooks/`

Notebooks should include:
- Clear markdown cells explaining purpose
- Step-by-step explanations
- Results interpretation
- References to relevant documentation

**Key Notebooks:**
- `04_selection/pareto_efficiency_analysis.ipynb`: Pareto analysis exploration
- `_goodreads_bill_nov_metadata_EDA/`: Goodreads data exploration
- `_romantic novels dataset EDA/`: Dataset exploration
- `03_modeling/`: Modeling experiments (if any)

## Configuration Documentation

### In `configs/`

Each config file should have:
- Comments explaining parameters
- Default values documented
- Range/constraint information

**Suggested**: Add `configs/README.md` explaining:
- Purpose of each config file
- Parameter descriptions
- How to modify for different experiments

## Code Documentation

### Inline Documentation

- **Docstrings**: All functions and classes should have docstrings
- **Type Hints**: Use type hints for clarity
- **Comments**: Explain complex logic, not obvious code

### Module-Level Documentation

Each Python module should have:
- Module-level docstring explaining purpose
- Usage examples in docstrings
- References to related documentation

## Documentation Best Practices

### File Length Guidelines

- **Main README**: 200-300 lines (overview)
- **Scientific README**: 400-600 lines (detailed methodology)
- **Technical docs**: 200-400 lines per file
- **Stage READMEs**: 100-200 lines each

### Cross-Referencing

Use clear cross-references:
- "See [METHODOLOGY.md](docs/METHODOLOGY.md) for details"
- "For data contracts, see [DATA_CONTRACTS.md](docs/DATA_CONTRACTS.md)"

### Audience-Specific Sections

Label sections by audience:
- **For Users**: Quick start, basic usage
- **For Researchers**: Methodology, hypotheses
- **For Developers**: Technical details, implementation

### Version Information

Document:
- Software versions in requirements.txt
- Model versions in metadata
- Data versions in file naming

## Documentation Maintenance

### When to Update

- After major code changes
- When adding new features
- When fixing bugs that affect usage
- After completing analysis stages

### Review Process

- Technical accuracy review
- Clarity and readability review
- Cross-reference verification
- Link checking

## Suggested Documentation Roadmap

### Phase 1: Core Documentation ✅
- [x] README.md
- [x] SCIENTIFIC_README.md
- [x] docs/METHODOLOGY.md
- [x] docs/DATA_CONTRACTS.md
- [x] docs/INDICES.md
- [x] Stage READMEs

### Phase 2: Results & Reproducibility (After Analysis)
- [ ] docs/RESULTS.md
- [ ] docs/REPRODUCIBILITY.md
- [ ] Update SCIENTIFIC_README.md with findings

### Phase 3: User Support
- [ ] docs/TROUBLESHOOTING.md
- [ ] docs/CONTRIBUTING.md
- [ ] configs/README.md

### Phase 4: Advanced Documentation
- [ ] docs/API_REFERENCE.md (if exposing APIs)
- [ ] docs/EXTENDING.md (for extending the pipeline)
- [ ] docs/PERFORMANCE.md (benchmarking, optimization tips)

## Documentation Templates

### For New Documentation Files

```markdown
# [Title]

## Overview
Brief description of what this document covers.

## [Main Section 1]
Content...

## [Main Section 2]
Content...

## References
- Link to related documentation
- External references if applicable

---
**See Also:**
- [Related Doc 1](link)
- [Related Doc 2](link)
```

## Summary

The documentation structure follows a **hierarchical, audience-specific approach**:

1. **Entry Point**: README.md (overview)
2. **Research Details**: SCIENTIFIC_README.md (methodology)
3. **Technical Details**: docs/ (implementation)
4. **Stage-Specific**: src/stage*/README.md (usage)
5. **Results**: docs/RESULTS.md (findings, when ready)

This structure allows readers to:
- Get started quickly (README.md)
- Understand research context (SCIENTIFIC_README.md)
- Dive into technical details (docs/)
- Use specific stages (stage READMEs)
- Reproduce results (REPRODUCIBILITY.md)

---

**Last Updated**: 2025-01-XX  
**Maintainer**: [Your Name/Team]

