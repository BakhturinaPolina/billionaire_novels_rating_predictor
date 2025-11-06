# Character Names Exclusion Analysis

## Research Rationale

In narrative text analysis, character names present a unique challenge for topic modeling algorithms. While names carry narrative significance, they often function as high-frequency markers that obscure thematic content. When character names appear frequently across documents, they can dominate topic word distributions, creating topics that reflect character co-occurrence rather than thematic relationships.

We implemented character name exclusion to improve topic interpretability by allowing the algorithm to focus on content words—verbs, adjectives, nouns describing actions, emotions, and narrative elements—rather than character references. This approach aligns with best practices in computational literary analysis where character names are typically treated as structural elements rather than semantic content (Bamman et al., 2013; Jockers, 2013).

## Character Names Preprocessing Analysis

### Processing Statistics

We processed a character names file extracted from romance novel texts containing 7,525 entries. The preprocessing pipeline achieved the following results:

- **Total lines processed**: 7,525
- **Lines filtered out**: 254 (3.4%)
- **Lines with valid names**: 7,271
- **Multi-word names processed**: 3,313
- **Unique name tokens extracted**: 4,497
- **Final character names added to stopwords**: 4,444

The 4,444 final count reflects that some character name tokens already overlapped with standard English stopwords (e.g., common first names like "will", "may" that are also modal verbs).

### Stopwords Summary

After processing:
- **Standard English stopwords**: 318
- **Character names added**: 4,444
- **Total stopwords**: 4,762

This represents a 14x increase in stopwords, with character names comprising approximately 93% of the expanded list.

### Filtering Examples and Rationale

The preprocessing pipeline successfully filtered out various non-name patterns:

**Correctly Filtered**:
- **Prefix patterns**: "After T.J." → filtered (temporal prefix)
- **Common word phrases**: "A voice" → filtered (descriptive phrase)
- **Descriptive patterns**: "A scowling Dante" → filtered (keeps "dante" if it appears elsewhere as a name)
- **Long descriptions**: "AUTHOR K.A. LINDE … TheWrightBoss HEIDI SWORE SHE'D" → filtered (>50 characters, descriptive text)

These examples demonstrate the pipeline's ability to distinguish between character name entries and descriptive text or metadata.

### Remaining Non-Name Words

The preprocessing retained some words that are not character names:
- Examples: "aardvarks", "accustomed", "activated", "actress", "actually", "ad"

These false positives represent approximately 1-2% of extracted tokens. We analyzed several examples and found that:
- Some entries in the source file were not pure character names (e.g., "Aardvarks" as a text artifact)
- Some tokens like "actress" appear in context like "Actress Lena Marcie" where "Lena Marcie" is the actual name
- Words like "accustomed" and "actually" likely appear in quoted dialogue or descriptive text within name entries

## Decision on Further Cleaning

### Current Approach: Acceptable Precision

We decided **not** to implement additional aggressive filtering at this stage for the following reasons:

1. **Coverage vs. Precision Trade-off**: The current pipeline successfully extracts 4,444 character name tokens, capturing the vast majority of character references. Implementing stricter filtering (e.g., using name databases, stricter pattern matching) would risk excluding valid character names, especially:
   - Uncommon or invented names
   - Names with non-standard spellings
   - Names that might also be common words in other contexts

2. **Acceptable False Positives**: The remaining non-name words (estimated 1-2% of tokens) are minimal compared to the benefit of comprehensive character name coverage. These false positives:
   - Are already filtered if they appear in standard stopwords
   - Have minimal impact on topic quality (they appear infrequently in actual documents)
   - Would require complex linguistic processing to distinguish from valid names

3. **Practical Considerations**: Further cleaning would require:
   - Maintenance of name databases or dictionaries
   - More complex pattern matching rules
   - Potential reduction in reproducibility (name databases may vary by research domain)

### Future Improvement Possibility

While we maintain the current approach, future enhancements could include:

- **Domain-specific name databases**: For romance novels specifically, maintaining a curated list of known character names could improve precision
- **Frequency-based filtering**: Analyzing token frequencies in the actual corpus could help identify non-name words that slipped through
- **Hybrid approach**: Combining current extraction with manual review of high-frequency extracted tokens

These improvements could be implemented as optional enhancements for researchers requiring higher precision, while maintaining the current approach as the default for maximum coverage and reproducibility.

## Impact on Topic Modeling

Preliminary analysis indicates that character name exclusion improves topic interpretability:

- **Before exclusion**: Topics often dominated by character names (e.g., "Alex, Stella, Weston, love")
- **After exclusion**: Topics focus on thematic content (e.g., "love, relationship, emotion, connection")

This shift allows researchers to identify thematic patterns and narrative elements rather than character co-occurrence patterns, which aligns with computational literary analysis goals.

