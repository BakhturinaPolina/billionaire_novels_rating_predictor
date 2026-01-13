# Stage 01: Data Ingestion Methodology Report

## Overview

This report documents the data ingestion pipeline implemented in **Stage 01: Data Ingestion**. This stage serves as the foundational data loading and preparation step that consolidates raw text files, Goodreads metadata, and optional BookNLP outputs into a unified format for downstream processing.

### Purpose

Stage 01 handles the initial data collection and consolidation phase of the research pipeline. The primary objectives are:

1. **Load raw text files** from the billionaire romance novel corpus
2. **Integrate Goodreads metadata** (ratings, review counts, publication information) with book texts
3. **Process BookNLP outputs** (if available) for character and entity extraction
4. **Establish data contracts** that ensure consistent data formats across the pipeline

### Key Components

- **Raw text file loading** from multiple sources (TXT, EPUB formats)
- **Goodreads metadata integration** for popularity and quality metrics
- **BookNLP entity extraction** (optional) for character name identification
- **Data validation** and error handling for missing or corrupted files
- **Progress tracking** for large-scale dataset processing

## Research Rationale

### Why Data Ingestion as a Separate Stage?

Separating data ingestion from preprocessing serves several important research and engineering purposes:

1. **Reproducibility**: By establishing clear input/output contracts, the pipeline ensures that preprocessing and modeling stages can be rerun independently without re-ingesting raw data.

2. **Data Validation**: Early validation of data integrity (file existence, format consistency, encoding issues) prevents downstream errors and ensures data quality from the start.

3. **Metadata Integration**: Combining text data with external metadata (Goodreads ratings) at the ingestion stage allows for consistent book identification and quality grouping throughout the pipeline.

4. **Modularity**: Separating ingestion from preprocessing enables researchers to experiment with different preprocessing strategies without re-processing raw files.

5. **Performance**: Ingestion can be optimized separately (e.g., parallel file loading, caching) without affecting preprocessing logic.

### Corpus Selection Rationale

The dataset includes **105 standalone billionaire romance novels** by **35 different authors**, selected from curated lists such as "100 Best Billionaire Romance Books of All Time". This selection strategy ensures:

- **Quality representation**: Focus on well-regarded works in the genre
- **Author diversity**: Multiple authors prevent single-author bias
- **Length consistency**: Each novel contains at least 100,000 words, ensuring sufficient text for topic modeling
- **Standalone format**: Excluding series prevents narrative continuity issues that could confound topic analysis

The final corpus contains **680,822 sentences** organized hierarchically by:
- **Author** → **Book** → **Chapter** → **Sentence**

This hierarchical structure facilitates multi-level analyses, such as tracking topic evolution within chapters or comparing thematic progression across multiple books.

## Data Sources

### Input Data

#### 1. Raw Text Files
- **Location**: `data/raw/Billionaire_Full_Novels_TXT/`
- **Format**: Plain text files (`.txt`) or EPUB files
- **Content**: Full novel texts, one file per book
- **Encoding**: UTF-8 (with handling for encoding issues)
- **Structure**: Raw text without preprocessing

#### 2. Goodreads Metadata
- **Location**: `data/processed/goodreads.csv`
- **Format**: CSV with columns:
  - `ID`: Unique book identifier
  - `Author`: Author name
  - `Title`: Book title
  - `Score`: Average rating (1-5 scale)
  - `RatingsCount`: Number of user ratings
  - `ReviewsCount`: Number of user reviews
  - `Pages`: Book length (pages)
  - Additional metadata fields
- **Purpose**: Provides popularity metrics and quality indicators for grouping books into Top/Medium/Trash tiers
- **Statistics**: 
  - 97-98 books with complete metadata
  - Rating distribution: Mean 3.99, Std 0.21, Range 3.26-4.42
  - All books have ≥100 ratings (minimum: 146, mean: 65,849)

#### 3. BookNLP Outputs (Optional)
- **Location**: `data/interim/booknlp/`
- **Format**: BookNLP processing outputs (character entities, quotes, etc.)
- **Purpose**: Provides character name extraction for stopword generation
- **Usage**: Used in Stage 02 preprocessing to build custom stoplist (4,444 character names)

### Output Data

#### Primary Outputs

1. **Processed Text Data**
   - Format: Structured text data ready for Stage 02 preprocessing
   - Contains: Book texts with metadata attached
   - Structure: Maintains hierarchical organization (Author → Book → Chapter)

2. **Merged Metadata**
   - Format: Integrated book metadata combining text sources and Goodreads data
   - Contains: Book identifiers, author information, ratings, review counts
   - Purpose: Enables consistent book identification across pipeline stages

3. **BookNLP Entity Information** (if available)
   - Format: Character names, entity mentions, quote attributions
   - Purpose: Supports character name exclusion in preprocessing (Stage 02)

## Data Processing Pipeline

### File Loading Strategy

The ingestion pipeline implements the following loading strategy:

1. **File Discovery**: Recursively scan input directories for text files
2. **Format Detection**: Identify file formats (TXT, EPUB) and apply appropriate parsers
3. **Encoding Handling**: Detect and handle encoding issues (UTF-8, with fallback for mojibake)
4. **Progress Tracking**: Log progress for large-scale processing (105 novels)

### Metadata Integration

The Goodreads metadata integration process:

1. **Load Goodreads CSV**: Read metadata from `data/processed/goodreads.csv`
2. **Book Matching**: Match books between text files and Goodreads metadata using:
   - Author name matching (with fuzzy matching for format differences)
   - Title matching (case-insensitive, punctuation-normalized)
3. **Data Enrichment**: Attach ratings, review counts, and other metadata to book records
4. **Validation**: Verify that matched books have required metadata fields

**Matching Challenges**:
- Author format differences: Goodreads uses lowercase with spaces (e.g., `sarina bowen`), while text files may use underscore-separated format (e.g., `sarina_bowen`)
- Title format differences: Goodreads uses lowercase titles, while text files use title case
- Solution: Implement fuzzy matching with configurable threshold (default: 0.85 similarity)

### BookNLP Integration (Optional)

If BookNLP outputs are available:

1. **Load Entity Files**: Read character entity files from BookNLP output directory
2. **Extract Character Names**: Parse character name lists from entity files
3. **Format for Stoplist**: Prepare character names for stopword list generation (Stage 02)
4. **Validation**: Verify character name extraction quality

## Data Contracts

### Input Contracts

**Raw Text Files**:
- Must exist in specified directory
- Must be readable (valid encoding)
- Must contain at least minimal text content

**Goodreads Metadata**:
- Must contain required columns: `ID`, `Author`, `Title`, `Score`, `RatingsCount`
- Must have valid rating values (1-5 scale)
- Must have non-zero rating counts for filtering

**BookNLP Outputs** (optional):
- Must follow BookNLP output format if present
- Character entity files must be parseable

### Output Contracts

**Processed Text Data**:
- Must maintain book-level organization
- Must preserve text content integrity
- Must include book identifiers for downstream matching

**Merged Metadata**:
- Must include all books from text corpus
- Must have matched Goodreads data where available
- Must maintain consistent book identification scheme

## Error Handling

The ingestion pipeline implements robust error handling:

1. **Missing Files**: Log warnings for missing files, continue processing other files
2. **Encoding Issues**: Attempt to fix common mojibake artifacts, log failures
3. **Metadata Mismatches**: Log unmatched books, continue with available metadata
4. **Corrupted Files**: Skip corrupted files with error logging, continue processing

## Integration with Other Stages

### Downstream Dependencies

**Stage 02 (Preprocessing)**:
- Receives processed text data from Stage 01
- Uses book metadata for consistent identification
- Uses BookNLP character names for stoplist generation

**Stage 03 (Modeling)**:
- Uses preprocessed text from Stage 02 (which depends on Stage 01)
- Uses book metadata for grouping and evaluation

**Stage 10 (Correlation Analysis)**:
- Uses book metadata for rating-based grouping (Top/Medium/Trash)
- Requires consistent book identification from Stage 01

### Typical Workflow

1. **Stage 01 - Ingestion**: Load raw texts, Goodreads data, BookNLP outputs
2. **Stage 02 - Preprocessing**: Clean and tokenize text, build custom stoplist
3. **Stage 03 - Modeling**: Train BERTopic models on preprocessed text
4. **Stage 10 - Analysis**: Use metadata for correlation analysis

## Implementation Status

### Current Implementation

⚠️ **Placeholder Implementation** - Core logic pending

The current `main.py` implementation shows the planned structure and data contracts, but full implementation is pending. The pipeline structure is designed to support:

- Configuration-based path management
- Input/output validation
- Progress tracking
- Error handling

### Planned Enhancements

When implementing, consider:

1. **Parallel Processing**: Implement parallel file loading for large datasets
2. **Caching**: Cache processed data to avoid re-ingestion on pipeline reruns
3. **Format Support**: Expand support for additional file formats (EPUB, PDF)
4. **Validation**: Implement comprehensive data validation checks
5. **Progress Reporting**: Add detailed progress bars and logging

## Technical Requirements

### Software Dependencies

- **pandas**: For CSV reading and data manipulation
- **pathlib**: For file system operations
- **click**: For command-line interface
- **fuzzywuzzy** or **rapidfuzz**: For fuzzy string matching (author/title matching)

### Hardware Requirements

- **Storage**: Sufficient disk space for raw text files (105 novels, ~100K+ words each)
- **Memory**: Adequate RAM for loading and processing large text files

## Future Improvements

### Data Quality Enhancements

- **Automatic encoding detection**: Implement robust encoding detection for various file formats
- **Metadata validation**: Add comprehensive validation for Goodreads metadata quality
- **Duplicate detection**: Identify and handle duplicate books across sources

### Performance Optimizations

- **Streaming processing**: Implement streaming for very large files to reduce memory usage
- **Incremental updates**: Support incremental ingestion for new books without reprocessing entire corpus
- **Caching layer**: Add caching to avoid redundant file reads

### Feature Additions

- **Additional metadata sources**: Integrate metadata from other sources (Amazon, Library of Congress)
- **Text quality metrics**: Compute and store text quality metrics during ingestion
- **Version tracking**: Track data versions and changes over time

## References

- Bamman, D., Underwood, T., & Smith, N. A. (2013). A Bayesian Mixed Effects Model of Literary Character. *Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics*.
- Jockers, M. L. (2013). *Macroanalysis: Digital Methods and Literary History*. University of Illinois Press.

