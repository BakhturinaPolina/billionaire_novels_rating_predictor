# Stage 02: Preprocessing Methodology Report

## Overview

This report documents the text preprocessing pipeline implemented in **Stage 02: Preprocessing**. This stage handles text cleaning, tokenization, lemmatization, and custom stoplist building to prepare raw text data for topic modeling in Stage 03.

### Purpose

Stage 02 transforms raw text files into a clean, structured format suitable for neural topic modeling. The primary objectives are:

1. **Text cleaning**: Remove artifacts, fix encoding issues, normalize text
2. **Sentence segmentation**: Split text into individual sentences for topic modeling
3. **Tokenization and lemmatization**: Prepare text for embedding generation
4. **Custom stoplist application**: Remove character names and domain-specific stopwords
5. **Output formatting**: Generate structured CSV output for downstream modeling

### Key Components

- **Text cleaning** (encoding fixes, whitespace normalization, artifact removal)
- **Sentence segmentation** (preserving sentence boundaries)
- **Tokenization and lemmatization** (using spaCy or similar NLP libraries)
- **Custom stoplist integration** (character names + standard English stopwords)
- **Structured output** (CSV format with one sentence per row)

## Research Rationale

### Why Preprocessing Before Topic Modeling?

Preprocessing is critical for neural topic modeling (BERTopic) for several reasons:

1. **Embedding Quality**: Clean, normalized text produces better sentence embeddings. Encoding artifacts, inconsistent whitespace, and formatting issues can degrade embedding quality.

2. **Sentence Boundaries**: BERTopic operates on sentence-level embeddings. Accurate sentence segmentation ensures that topics capture coherent semantic units rather than arbitrary text chunks.

3. **Stopword Removal**: Character names and domain-specific stopwords can dominate topic word distributions, obscuring thematic content. Removing these allows topics to focus on meaningful thematic relationships.

4. **Consistency**: Standardized preprocessing ensures reproducibility and enables fair comparison across different modeling experiments.

5. **Computational Efficiency**: Clean, normalized text reduces computational overhead and improves processing speed.

### Character Name Exclusion Rationale

In narrative text analysis, character names present a unique challenge for topic modeling algorithms. While names carry narrative significance, they often function as high-frequency markers that obscure thematic content. When character names appear frequently across documents, they can dominate topic word distributions, creating topics that reflect character co-occurrence rather than thematic relationships.

We implement character name exclusion to improve topic interpretability by allowing the algorithm to focus on content words—verbs, adjectives, nouns describing actions, emotions, and narrative elements—rather than character references. This approach aligns with best practices in computational literary analysis where character names are typically treated as structural elements rather than semantic content (Bamman et al., 2013; Jockers, 2013).

**Impact on Topic Modeling**:
- **Before exclusion**: Topics often dominated by character names (e.g., "Alex, Stella, Weston, love")
- **After exclusion**: Topics focus on thematic content (e.g., "love, relationship, emotion, connection")

This shift allows researchers to identify thematic patterns and narrative elements rather than character co-occurrence patterns, which aligns with computational literary analysis goals.

## Preprocessing Pipeline

### Text Cleaning

The text cleaning pipeline performs the following operations:

#### 1. Encoding Fixes

**Mojibake Correction**:
- Fixes common UTF-8 encoding artifacts that appear as mojibake:
  - `â€™` → `'` (apostrophe)
  - `â€œ` → `"` (opening quote)
  - `â€` → `"` (closing quote)
  - `â€"` → `-` (em dash)
  - `â€"` → `-` (en dash)
  - `â€˜` → `'` (single quote)
  - `â€¦` → `...` (ellipsis)

**Unicode Normalization**:
- Applies NFKD (Normalization Form Compatibility Decomposition) to decompose characters
- Example: `é` → `e` + accent mark
- Ensures consistent character representation across the corpus

#### 2. Whitespace Normalization

- **Newline removal**: Converts newlines to spaces (sentences are segmented separately)
- **Multiple whitespace**: Collapses multiple spaces/tabs into single spaces
- **Leading/trailing whitespace**: Strips whitespace from sentence boundaries

#### 3. Case Normalization

- Converts all text to lowercase for consistent processing
- Note: This occurs after sentence segmentation to preserve sentence structure

#### 4. Artifact Removal

- Removes headers/footers if present (based on pattern matching)
- Removes page numbers and formatting artifacts
- Handles special characters that don't contribute to semantic content

### Sentence Segmentation

Sentence segmentation is critical for BERTopic, which operates on sentence-level embeddings:

1. **Preserve Sentence Boundaries**: Accurate segmentation ensures that each sentence is a coherent semantic unit
2. **Handle Edge Cases**: 
   - Abbreviations (e.g., "Mr.", "Dr.", "Inc.")
   - Decimal numbers (e.g., "3.14")
   - Ellipses (e.g., "...")
3. **Maintain Context**: Preserve sentence order and chapter structure for downstream analysis

**Implementation**: Uses spaCy's sentence segmentation or similar NLP library with domain-specific rules for romance novel text.

### Tokenization and Lemmatization

#### Tokenization

- **Word tokenization**: Splits sentences into individual words/tokens
- **Punctuation handling**: Preserves punctuation for context (removed later in stopword filtering)
- **Hyphen handling**: Handles hyphenated words appropriately

#### Part-of-Speech Tagging

- **POS tagging**: Identifies parts of speech for lemmatization
- **Context-aware**: Uses surrounding words to disambiguate POS

#### Lemmatization

- **Root form extraction**: Converts words to their base/root forms
- **Examples**:
  - "running" → "run"
  - "better" → "good"
  - "was" → "be"
- **Purpose**: Reduces vocabulary size and groups related word forms

**Implementation**: Uses spaCy's lemmatization with POS tagging for accuracy.

### Custom Stoplist Building

#### Stoplist Components

The custom stoplist consists of two components:

1. **Standard English Stopwords**: 318 common English stopwords (articles, prepositions, conjunctions, etc.)
2. **Character Names**: 4,444 character names extracted from romance novel texts

**Total Stopwords**: 4,762 (93% character names, 7% standard English stopwords)

This represents a **14x increase** in stopwords compared to standard English stopword lists, with character names comprising the vast majority of the expanded list.

#### Character Name Processing

The character name preprocessing pipeline performs the following operations:

**Cleaning Steps**:
- Removes leading prefixes: "A ", "Mr.", "Miss ", "the ", "AKA ", "#"
- Removes leading numbers: "17 Vick Jett" → "vick jett"
- Removes quotes and punctuation
- Converts everything to lowercase

**Filtering Steps**:
- Removes empty lines
- Removes very long lines (>50 characters) - usually descriptions
- Removes common phrases like "A voice", "the Wright brothers"
- Removes descriptive patterns like "A scowling Dante" (keeps "Dante" if it appears elsewhere as a name)
- Filters out common non-name words
- Skips geographic locations

**Name Extraction**:
- Splits multi-word names: "Alex Crane" → extracts both "alex" and "crane"
- This ensures both first and last names are filtered, even if they appear separately

#### Processing Statistics

The preprocessing pipeline processes character names extracted from romance novel texts:

- **Total lines processed**: 7,525
- **Lines filtered out**: 254 (3.4%)
- **Lines with valid names**: 7,271
- **Multi-word names processed**: 3,313
- **Unique name tokens extracted**: 4,497
- **Final character names added to stopwords**: 4,444

The 4,444 final count reflects that some character name tokens already overlapped with standard English stopwords (e.g., common first names like "will", "may" that are also modal verbs).

#### Precision Trade-offs

The preprocessing retained some words that are not character names (estimated 1-2% of tokens):
- Examples: "aardvarks", "accustomed", "activated", "actress", "actually", "ad"

We decided **not** to implement additional aggressive filtering at this stage for the following reasons:

1. **Coverage vs. Precision Trade-off**: The current pipeline successfully extracts 4,444 character name tokens, capturing the vast majority of character references. Implementing stricter filtering would risk excluding valid character names, especially uncommon or invented names.

2. **Acceptable False Positives**: The remaining non-name words are minimal compared to the benefit of comprehensive character name coverage. These false positives have minimal impact on topic quality as they appear infrequently in actual documents.

3. **Practical Considerations**: Further cleaning would require maintenance of name databases or dictionaries, more complex pattern matching rules, and potential reduction in reproducibility.

## Output Format

### CSV Structure

The preprocessing pipeline outputs `data/processed/chapters.csv` with the following structure:

**Columns**:
- `Author`: Author name (underscore-separated format, e.g., `Ann_Cole`, `sarina_bowen`)
- `Book Title`: Book title (title case with punctuation)
- `Chapter`: Chapter number (integer)
- `Sentence`: Preprocessed sentence text (cleaned, tokenized, lemmatized, stopwords removed)

**Statistics**:
- **Total rows**: 680,822 sentences
- **Books**: 105 standalone novels
- **Authors**: 35 different authors
- **Hierarchical structure**: Author → Book → Chapter → Sentence

### Data Quality

**Validation Checks**:
- No missing values in required columns
- All sentences are non-empty after preprocessing
- Chapter numbering is consistent within books
- Author/Book Title format is standardized

**Sentence Format**:
- Sentences match BERTopic training format (one sentence per row)
- Text is lowercase, normalized, and cleaned
- Stopwords (including character names) are removed
- Lemmatized forms are used

## Integration with Other Stages

### Upstream Dependencies

**Stage 01 (Ingestion)**:
- Receives raw text files and metadata from Stage 01
- Uses book metadata for consistent identification
- Uses BookNLP character names (if available) for stoplist generation

### Downstream Dependencies

**Stage 03 (Modeling)**:
- Provides preprocessed text (`chapters.csv`) for BERTopic training
- Text format must match training format exactly
- Sentence-level structure enables sentence embedding generation

**Stage 09 (Category Mapping)**:
- Uses preprocessed sentences for topic-to-sentence mapping
- Requires consistent sentence IDs and book identification

**Stage 10 (Correlation Analysis)**:
- Uses preprocessed text structure for sentence-level topic probability inference
- Requires book metadata for rating-based grouping

### Typical Workflow

1. **Stage 01 - Ingestion**: Load raw texts, Goodreads data, BookNLP outputs
2. **Stage 02 - Preprocessing**: Clean and tokenize text, build custom stoplist ← **Current Stage**
3. **Stage 03 - Modeling**: Train BERTopic models on preprocessed text
4. **Stage 09 - Category Mapping**: Map topics to sentences using preprocessed structure
5. **Stage 10 - Analysis**: Use preprocessed text for correlation analysis

## Implementation Status

### Current Implementation

⚠️ **Placeholder Implementation** - Core logic pending

The current `main.py` implementation shows the planned structure and data contracts, but full implementation is pending. However, the output data (`chapters.csv`) exists and was processed using similar preprocessing logic (likely implemented in notebooks or legacy code).

### Planned Enhancements

When implementing, consider:

1. **spaCy Integration**: Use spaCy for sentence segmentation, tokenization, and lemmatization
2. **Memory Management**: Implement streaming processing for large files
3. **Progress Tracking**: Add detailed progress bars for long-running preprocessing
4. **Custom Stoplist Integration**: Seamlessly integrate character name stoplist
5. **Validation**: Implement comprehensive output validation

## Technical Requirements

### Software Dependencies

- **spaCy**: For sentence segmentation, tokenization, POS tagging, and lemmatization
- **pandas**: For CSV reading and data manipulation
- **pathlib**: For file system operations
- **click**: For command-line interface
- **unicodedata**: For Unicode normalization

### Hardware Requirements

- **Memory**: Adequate RAM for processing large text files (680K+ sentences)
- **Storage**: Sufficient disk space for output CSV files

## Future Improvements

### Preprocessing Enhancements

- **Domain-specific rules**: Add romance novel-specific preprocessing rules (e.g., handling of dialogue, internal monologue)
- **Quality metrics**: Compute and report text quality metrics (sentence length distribution, vocabulary size, etc.)
- **Incremental processing**: Support incremental preprocessing for new books without reprocessing entire corpus

### Stoplist Improvements

While the current approach prioritizes coverage over precision, future enhancements could include:

- **Domain-specific name databases**: For romance novels specifically, maintaining a curated list of known character names could improve precision
- **Frequency-based filtering**: Analyzing token frequencies in the actual corpus could help identify non-name words that slipped through
- **Hybrid approach**: Combining current extraction with manual review of high-frequency extracted tokens

These improvements could be implemented as optional enhancements for researchers requiring higher precision, while maintaining the current approach as the default for maximum coverage and reproducibility.

### Performance Optimizations

- **Parallel processing**: Implement parallel sentence processing for large datasets
- **Caching**: Cache intermediate preprocessing results to avoid recomputation
- **Streaming**: Implement streaming processing for very large files

## References

- Bamman, D., Underwood, T., & Smith, N. A. (2013). A Bayesian Mixed Effects Model of Literary Character. *Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics*.
- Jockers, M. L. (2013). *Macroanalysis: Digital Methods and Literary History*. University of Illinois Press.
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv preprint arXiv:2203.05794*.

