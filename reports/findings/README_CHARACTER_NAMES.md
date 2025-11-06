# Character Names as Stopwords: Usage Guide

## A. Why Exclude Character Names?

When analyzing text from novels or narrative works, character names frequently appear throughout the documents. While these names are important for story understanding, they can create "noisy" topics in topic modeling. 

**The problem**: Character names (like "Alex", "Stella", "Weston") often dominate topic word lists because they appear frequently across many documents, but they don't tell us about the actual thematic content. A topic might show "Alex, love, heart" instead of showing "love, relationship, emotion" - which is more meaningful for understanding themes.

**The solution**: By adding character names to the stopwords list, we filter them out before topic modeling. This helps the algorithm focus on content words that reveal actual themes, relationships, emotions, and plot elements rather than just character references.

## B. What Does the Pipeline Do?

The code automatically processes your character names file and adds those names to a list of words to ignore (called "stopwords"). Here's what happens in simple steps:

1. **You provide a text file** with character names (one name per line)
2. **The code reads the file** and cleans each line:
   - Removes prefixes like "A ", "Mr.", "the "
   - Removes numbers and special characters
   - Splits full names (like "Alex Crane") into individual parts ("alex", "crane")
3. **The code filters out non-names**:
   - Skips descriptions and phrases (like "A voice" or "the Wright brothers")
   - Removes common words that aren't names
   - Skips very long lines (usually descriptions, not names)
4. **The cleaned names are added** to the stopwords list
5. **During document processing**, these names are automatically ignored when building topics

The process includes detailed logging so you can see exactly what was extracted and added.

## C. How to Use This Feature

### Step 1: Prepare Your Character Names File

Create a text file (`.txt`) with one character name per line. The file can include:
- First names: `Alex`
- Full names: `Alex Crane`
- Variations: `A Alex`, `Mr. Crane`, `Alex Cannon` (prefixes will be removed automatically)

**Example file (`my_characters.txt`)**:
```
Alex
Stella
Weston
Hunter Fitzpatrick
A Dante
Mr. Sebastian
```

**Note**: The code will automatically clean and process these, so slight variations are okay.

### Step 2: Run the Training Script with Character Names

Add the `--character_names_file` argument when running the training script:

```bash
python src/bertopic/train_bertopic_from_tables.py \
  --dataset_csv "your_dataset.csv" \
  --out_dir "./bertopic_output" \
  --character_names_file "my_characters.txt" \
  --text_column "Sentence"
```

### Step 3: Review the Logs

The code will print detailed information about the character names processing:

```
Loading character names from: my_characters.txt
  Total lines in file: 7,525
  Processing lines...
    Processed 1,000/7,525 lines, extracted 931 unique name tokens so far...
  ...
  Processing complete!
    Total lines processed: 7,525
    Lines filtered out: 254 (3.4%)
    Lines with valid names: 7,271
    Unique name tokens extracted: 4,497
  Added 4,444 character names to stopwords list
  Total stopwords now: 4,762 (318 standard + 4,444 character names)
```

This shows you:
- How many names were found
- How many were filtered out (and why)
- How many unique name tokens were extracted
- The final stopwords count

### Step 4: Check Your Results

After training completes, you can verify that character names were filtered by:
- **Checking topics**: Character names should not appear in topic word lists (or appear very rarely)
- **Reviewing logs**: The log file in `out_dir/logs/train.log` contains full details

### Using Your Own Dataset

If you're using a different dataset (not romance novels), you can still use this feature:

1. **Extract character names** from your dataset using any method:
   - Named Entity Recognition (NER) tools
   - Manual extraction
   - Character lists from sources
   - Frequency analysis of capitalized words

2. **Create your names file** following the same format (one name per line)

3. **Run the script** with your character names file

4. **The same preprocessing logic** will apply to your names file automatically

### Combining with Custom Stopwords

You can use both custom stopwords and character names:

```bash
python src/bertopic/train_bertopic_from_tables.py \
  --dataset_csv "your_dataset.csv" \
  --out_dir "./bertopic_output" \
  --custom_stopwords_file "my_stopwords.txt" \
  --character_names_file "my_characters.txt" \
  --text_column "Sentence"
```

The order of processing is:
1. Standard English stopwords (built-in)
2. Custom stopwords (from your file)
3. Character names (from your file)

All are combined into one final stopwords list.

## Understanding the Preprocessing

The character names preprocessing does the following automatically:

**Cleaning steps**:
- Removes leading prefixes: "A ", "Mr.", "Miss ", "the ", "AKA ", "#"
- Removes leading numbers: "17 Vick Jett" → "vick jett"
- Removes quotes and punctuation
- Converts everything to lowercase

**Filtering steps**:
- Removes empty lines
- Removes very long lines (>50 characters) - usually descriptions
- Removes common phrases like "A voice", "the Wright brothers"
- Removes descriptive patterns like "A scowling Dante" (keeps "Dante")
- Filters out common non-name words
- Skips geographic locations

**Name extraction**:
- Splits multi-word names: "Alex Crane" → extracts both "alex" and "crane"
- This ensures both first and last names are filtered, even if they appear separately

**Note on precision**: Some non-name words might still be included (like "aardvarks", "actress"). This is by design - we prioritize catching all character names over perfect precision. If you notice specific words that should be filtered, you can add them to your custom stopwords file instead.

