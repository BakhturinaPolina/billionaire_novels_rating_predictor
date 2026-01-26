# Folder Organization Plan for openrouter_experiments

## Overview
This document outlines a reorganization plan to improve navigation and maintainability of the `openrouter_experiments` folder.

---

## 📁 Proposed Folder Structure

```
openrouter_experiments/
├── __init__.py                          # Keep in root
├── README.md                            # Keep in root (main entry point)
│
├── core/                                # Core implementation files
│   ├── generate_labels_openrouter.py   # Main labeling logic
│   └── main_openrouter.py              # CLI entry point
│
├── tools/                               # Utility scripts and tools
│   ├── compare_models_openrouter.py    # Model comparison tool
│   ├── validate_label_quality.py       # Quality validation script
│   └── inspect_random_topics.py         # Topic inspection tool
│
├── docs/                                # Documentation files
│   ├── prompts.md                       # Prompt documentation
│   ├── prompts_with_snippets.md        # Enhanced prompts documentation
│   ├── SNIPPETS_LOGIC.md               # Snippets feature reasoning
│   ├── MODEL_SELECTION_RECOMMENDATION.md  # Model selection analysis
│   └── COST_EVALUATION_REPORT.md       # Cost analysis
│
├── evaluation/                          # Testing and evaluation scripts
│   ├── test_hybrid_performance.py       # Performance tests
│   └── eval_snippet_centrality.py      # Snippet centrality evaluation
│
└── archive/                             # Historical/obsolete files
    ├── BERTOPIC_REVIEW.md              # Code review notes (resolved)
    ├── HYBRID_APPROACH_IMPLEMENTED.md   # Implementation notes (historical)
    └── PERFORMANCE_REGRESSION_ANALYSIS.md  # Historical analysis
```

---

## 📋 File Categorization

### ✅ **Keep in Root**
- `__init__.py` - Package initialization
- `README.md` - Main documentation entry point

### 📦 **Move to `core/`**
**Purpose**: Core implementation files that are actively used

- `generate_labels_openrouter.py` (1901 lines)
  - Main labeling implementation with OpenRouter API
  - Core functionality for the module
  
- `main_openrouter.py` (501 lines)
  - CLI entry point
  - Primary interface for users

### 🔧 **Move to `tools/`**
**Purpose**: Utility scripts for specific tasks

- `compare_models_openrouter.py` (430 lines)
  - Multi-model comparison tool
  - Used for model selection experiments
  
- `validate_label_quality.py` (497 lines)
  - Quality validation script
  - Checks for hallucinations and quality issues
  
- `inspect_random_topics.py` (360 lines)
  - Topic inspection utility
  - Debugging and manual review tool

### 📚 **Move to `docs/`**
**Purpose**: Documentation and analysis files

- `prompts.md` (258 lines)
  - Prompt documentation (without snippets)
  
- `prompts_with_snippets.md` (160 lines)
  - Enhanced prompts with snippets documentation
  
- `SNIPPETS_LOGIC.md` (209 lines)
  - Theoretical reasoning for snippets feature
  
- `MODEL_SELECTION_RECOMMENDATION.md` (202 lines)
  - Model comparison analysis (mistral-nemo vs Grok)
  - Decision documentation
  
- `COST_EVALUATION_REPORT.md` (167 lines)
  - Cost analysis and token usage documentation

### 🧪 **Move to `evaluation/`**
**Purpose**: Testing and evaluation scripts

- `test_hybrid_performance.py` (413 lines)
  - Performance testing for hybrid approach
  - Benchmarking scripts
  
- `eval_snippet_centrality.py` (142 lines)
  - Snippet centrality evaluation
  - Experimental analysis tool

### 📦 **Move to `archive/`**
**Purpose**: Historical files that document resolved issues or past experiments

- `BERTOPIC_REVIEW.md` (139 lines)
  - Code review notes about BERTopic usage
  - Issues have been resolved (see HYBRID_APPROACH_IMPLEMENTED.md)
  
- `HYBRID_APPROACH_IMPLEMENTED.md` (148 lines)
  - Historical implementation notes
  - Documents performance optimization that was completed
  - Useful reference but not actively needed
  
- `PERFORMANCE_REGRESSION_ANALYSIS.md` (118 lines)
  - Historical analysis of performance issues
  - Documents problems that were already fixed
  - Useful for understanding evolution but not current

---

## 🔄 Migration Steps

1. **Create new subdirectories**:
   ```bash
   mkdir -p core tools docs evaluation archive
   ```

2. **Move files to appropriate folders** (update imports as needed)

3. **Update imports in affected files**:
   - Update `main_openrouter.py` imports if `generate_labels_openrouter.py` moves
   - Update any cross-references in documentation files
   - Update `README.md` to reflect new structure

4. **Update `README.md`**:
   - Update file paths in documentation
   - Add section explaining folder structure

---

## 📝 Rationale

### Why `core/`?
- Separates implementation from utilities
- Makes it clear which files are the main entry points
- Easier to find primary functionality

### Why `tools/`?
- Groups utility scripts together
- Distinguishes one-off tools from core functionality
- Makes it easier to find specific utilities

### Why `docs/`?
- Centralizes all documentation
- Separates reference material from code
- Easier to maintain and update documentation

### Why `evaluation/`?
- Groups testing and evaluation scripts
- Separates experimental code from production code
- Makes it clear these are for analysis/testing

### Why `archive/`?
- Preserves historical context without cluttering active workspace
- Documents evolution of the codebase
- Can be referenced if needed but doesn't interfere with daily work

---

## ⚠️ Considerations

1. **Import Updates**: Moving files will require updating import statements in:
   - `main_openrouter.py` (imports from `generate_labels_openrouter.py`)
   - Any other files that import from these modules
   - Test files that import these modules

2. **Documentation Updates**: 
   - Update `README.md` with new file locations
   - Update any cross-references in markdown files

3. **Backward Compatibility**: 
   - Consider keeping symlinks or `__init__.py` exports for backward compatibility
   - Or update all references at once

4. **Git History**: 
   - Files maintain their git history when moved with `git mv`
   - Use `git mv` instead of regular `mv` to preserve history

---

## 🎯 Benefits

1. **Better Navigation**: Clear separation of concerns makes it easier to find files
2. **Reduced Clutter**: Archive folder removes obsolete files from active workspace
3. **Improved Maintainability**: Related files are grouped together
4. **Clearer Purpose**: Folder names indicate what each section contains
5. **Easier Onboarding**: New contributors can understand structure quickly

