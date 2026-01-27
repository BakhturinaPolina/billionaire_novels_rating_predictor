# Stage 02: Preprocessing — Methodology Report

## Purpose

Transform raw novel texts into clean, sentence-level data suitable for neural topic modeling (BERTopic).

## Research Rationale

### Why Preprocessing Before Topic Modeling?

1. **Embedding Quality**: Clean, normalized text produces better sentence embeddings
2. **Sentence Boundaries**: BERTopic operates on sentence-level embeddings; accurate segmentation ensures coherent semantic units
3. **Stopword Removal**: Character names can dominate topic distributions, obscuring thematic content
4. **Reproducibility**: Standardized preprocessing enables fair comparison across experiments

### Why Exclude Character Names?

Character names present a unique challenge for topic modeling. When names appear frequently, they can dominate topic word distributions, creating topics that reflect character co-occurrence rather than thematic relationships.

**Impact**:
- Before exclusion: Topics dominated by names (e.g., "Alex, Stella, Weston, love")
- After exclusion: Topics focus on themes (e.g., "love, relationship, emotion, connection")

This aligns with computational literary analysis practices where character names are treated as structural elements rather than semantic content (Bamman et al., 2013; Jockers, 2013).

---

## Processing Pipeline

### 1. Text Cleaning

**Encoding Fixes** (mojibake correction):
| Artifact | Corrected |
|----------|-----------|
| `â€™` | `'` (apostrophe) |
| `â€œ` / `â€` | `"` (quotes) |
| `â€"` | `-` (dashes) |
| `â€¦` | `...` (ellipsis) |

**Normalization**:
- Unicode NFKD normalization
- Collapse multiple whitespace
- Convert to lowercase (after sentence segmentation)

### 2. Sentence Segmentation

- Preserves sentence boundaries for coherent embedding units
- Handles edge cases: abbreviations (Mr., Dr.), decimal numbers, ellipses
- Maintains chapter structure for downstream analysis

### 3. Tokenization & Lemmatization

- Word tokenization with punctuation handling
- POS tagging for accurate lemmatization
- Root form extraction: "running" → "run", "better" → "good"

### 4. Custom Stoplist

**Components**:
| Source | Count | Percentage |
|--------|-------|------------|
| Character names | 4,444 | 93% |
| Standard English | 318 | 7% |
| **Total** | **4,762** | 100% |

This is a **14× expansion** over standard English stopword lists.

---

## Character Name Extraction

### Processing Statistics

| Metric | Value |
|--------|-------|
| Lines processed | 7,525 |
| Lines filtered | 254 (3.4%) |
| Valid name lines | 7,271 |
| Multi-word names | 3,313 |
| Unique tokens | 4,497 |
| Final stopwords | 4,444 |

### Processing Steps

1. **Cleaning**: Remove prefixes ("Mr.", "Miss", "the"), numbers, quotes, punctuation
2. **Filtering**: Remove long lines (>50 chars), common phrases, descriptive patterns
3. **Extraction**: Split multi-word names ("Alex Crane" → "alex", "crane")

### Precision Trade-off

The pipeline retains some non-name words (estimated 1–2%). We prioritize **coverage over precision** to ensure comprehensive character name exclusion. False positives have minimal impact as they appear infrequently.

---

## Output

### Statistics

| Metric | Value |
|--------|-------|
| Total sentences | 680,822 |
| Books | 105 |
| Authors | 35 |
| Structure | Author → Book → Chapter → Sentence |

### Format

CSV with columns: `Author`, `Book Title`, `Chapter`, `Sentence`

- Text is lowercase, normalized, lemmatized
- Stopwords (including character names) removed
- One sentence per row (matches BERTopic training format)

---

## Integration with Pipeline

| Stage | Dependency |
|-------|------------|
| ← Stage 01 | Receives raw texts and metadata |
| → Stage 03 | Provides `chapters.csv` for BERTopic training |
| → Stage 09 | Sentence structure for topic mapping |
| → Stage 10 | Sentence-level topic probability inference |

---