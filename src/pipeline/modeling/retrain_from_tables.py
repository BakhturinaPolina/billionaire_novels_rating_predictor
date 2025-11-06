from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Set TOKENIZERS_PARALLELISM early to suppress HuggingFace warnings
# This must be set BEFORE importing any tokenizer-related libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Note: LD_LIBRARY_PATH for cuML/CUDA libraries is set by .venv/bin/activate.d/setup_cuml_paths.sh
# This ensures libraries are found before Python imports CuPy/cuML

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

try:
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    HAS_GENSIM = True
except Exception:
    HAS_GENSIM = False

# Import shared utilities
from src.utils.training_utils import check_gpu_availability, setup_output_dirs, setup_logging

@dataclass
class Config:
    dataset_csv: Path
    out_dir: Path
    text_column: str = "Sentence"
    coherence_sample: int = 25000
    save_topics_info: bool = False
    cache_embeddings: bool = True
    seed: int = 42
    w_diversity: float = 0.5
    w_coherence: float = 0.5
    penalty_outliers: float = 0.25
    custom_stopwords_file: Optional[Path] = None
    character_names_file: Optional[Path] = None
    chunk_size: int = 50000  # Process dataset in chunks
    verbose: bool = True  # Enable verbose logging


def param_rows_from_tables() -> List[Dict]:
    """
    Extract unique parameter combinations from top-performing models.
    Excludes whaleloops-phrase-bert and removes coherence/diversity scores (recalculated during training).
    
    Note: 'Iteration' is NOT a hyperparameter - it's a unique identifier to differentiate between
    different hyperparameter combinations tested for the same embedding model (e.g., 100 iterations
    of all-MiniLM-L12-v2 with different parameter sets). It's kept for tracking/traceability only.
    """
    # Combined unique entries from Top 10 Combined Score + Top 10 Pareto-efficient tables
    rows = [
        # Table 1: Top 10 Combined Score
        dict(Embeddings_Model="all-MiniLM-L12-v2", Iteration=66,
             bertopic__min_topic_size=102, bertopic__top_n_words=30, hdbscan__min_cluster_size=281,
             hdbscan__min_samples=72, umap__min_dist=0.005022, umap__n_components=2, umap__n_neighbors=7,
             vectorizer__min_df=0.001504),
        dict(Embeddings_Model="paraphrase-mpnet-base-v2", Iteration=14,
             bertopic__min_topic_size=63, bertopic__top_n_words=22, hdbscan__min_cluster_size=500,
             hdbscan__min_samples=72, umap__min_dist=0.077818, umap__n_components=9, umap__n_neighbors=11,
             vectorizer__min_df=0.009372),
        dict(Embeddings_Model="all-MiniLM-L12-v2", Iteration=75,
             bertopic__min_topic_size=142, bertopic__top_n_words=10, hdbscan__min_cluster_size=473,
             hdbscan__min_samples=14, umap__min_dist=0.004634, umap__n_components=5, umap__n_neighbors=15,
             vectorizer__min_df=0.001947),
        dict(Embeddings_Model="paraphrase-mpnet-base-v2", Iteration=0,
             bertopic__min_topic_size=127, bertopic__top_n_words=31, hdbscan__min_cluster_size=494,
             hdbscan__min_samples=28, umap__min_dist=0.058341, umap__n_components=10, umap__n_neighbors=11,
             vectorizer__min_df=0.007313),
        dict(Embeddings_Model="paraphrase-MiniLM-L6-v2", Iteration=19,
             bertopic__min_topic_size=64, bertopic__top_n_words=27, hdbscan__min_cluster_size=143,
             hdbscan__min_samples=32, umap__min_dist=0.085702, umap__n_components=9, umap__n_neighbors=44,
             vectorizer__min_df=0.005932),
        dict(Embeddings_Model="paraphrase-mpnet-base-v2", Iteration=13,
             bertopic__min_topic_size=14, bertopic__top_n_words=18, hdbscan__min_cluster_size=497,
             hdbscan__min_samples=32, umap__min_dist=0.086975, umap__n_components=8, umap__n_neighbors=9,
             vectorizer__min_df=0.009857),
        dict(Embeddings_Model="multi-qa-mpnet-base-cos-v1", Iteration=23,
             bertopic__min_topic_size=28, bertopic__top_n_words=28, hdbscan__min_cluster_size=492,
             hdbscan__min_samples=12, umap__min_dist=0.095922, umap__n_components=9, umap__n_neighbors=19,
             vectorizer__min_df=0.008294),
        dict(Embeddings_Model="all-MiniLM-L12-v2", Iteration=67,
             bertopic__min_topic_size=99, bertopic__top_n_words=24, hdbscan__min_cluster_size=258,
             hdbscan__min_samples=37, umap__min_dist=0.004852, umap__n_components=7, umap__n_neighbors=42,
             vectorizer__min_df=0.001174),
        dict(Embeddings_Model="multi-qa-mpnet-base-cos-v1", Iteration=28,
             bertopic__min_topic_size=29, bertopic__top_n_words=14, hdbscan__min_cluster_size=427,
             hdbscan__min_samples=11, umap__min_dist=0.008103, umap__n_components=9, umap__n_neighbors=18,
             vectorizer__min_df=0.005862),
        dict(Embeddings_Model="multi-qa-mpnet-base-cos-v1", Iteration=11,
             bertopic__min_topic_size=105, bertopic__top_n_words=24, hdbscan__min_cluster_size=497,
             hdbscan__min_samples=13, umap__min_dist=0.022149, umap__n_components=8, umap__n_neighbors=14,
             vectorizer__min_df=0.009229),
        # Table 2: Additional Pareto-efficient entries (excluding duplicates and whaleloops)
        dict(Embeddings_Model="paraphrase-MiniLM-L6-v2", Iteration=28,
             bertopic__min_topic_size=88, bertopic__top_n_words=11, hdbscan__min_cluster_size=145,
             hdbscan__min_samples=91, umap__min_dist=0.097223, umap__n_components=10, umap__n_neighbors=45,
             vectorizer__min_df=0.008545),
        dict(Embeddings_Model="paraphrase-distilroberta-base-v1", Iteration=6,
             bertopic__min_topic_size=110, bertopic__top_n_words=11, hdbscan__min_cluster_size=276,
             hdbscan__min_samples=58, umap__min_dist=0.029098, umap__n_components=3, umap__n_neighbors=36,
             vectorizer__min_df=0.007149),
        dict(Embeddings_Model="paraphrase-mpnet-base-v2", Iteration=1,
             bertopic__min_topic_size=57, bertopic__top_n_words=37, hdbscan__min_cluster_size=132,
             hdbscan__min_samples=57, umap__min_dist=0.053015, umap__n_components=4, umap__n_neighbors=39,
             vectorizer__min_df=0.004806),
    ]
    # Deduplicate by (Embeddings_Model, Iteration) - keep separate entries for different iterations
    seen = set()
    deduped = []
    for r in rows:
        key = (r["Embeddings_Model"], int(r["Iteration"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped

def create_subset_csv(dataset_csv: Path, text_column: str, subset_size: int = 10000, 
                      seed: int = 42, logger: Optional[logging.Logger] = None) -> Path:
    """
    Create a subset CSV by randomly sampling rows from the main dataset.
    
    Args:
        dataset_csv: Path to the main CSV file
        text_column: Name of the text column
        subset_size: Number of rows to sample (default: 10000)
        seed: Random seed for reproducibility
        logger: Optional logger instance
        
    Returns:
        Path to the created subset CSV file
    """
    if logger is None:
        logger = logging.getLogger("bertopic_simple")
    
    logger.info(f"Creating subset CSV: sampling {subset_size:,} rows from {dataset_csv}")
    
    # Read the full dataset
    df = pd.read_csv(dataset_csv)
    total_rows = len(df)
    
    if total_rows < subset_size:
        logger.warning(f"Dataset has only {total_rows:,} rows, less than requested {subset_size:,}. Using all rows.")
        subset_size = total_rows
    
    # Randomly sample
    df_subset = df.sample(n=subset_size, random_state=seed, replace=False)
    
    # Create output filename
    dataset_path = Path(dataset_csv)
    subset_path = dataset_path.parent / f"{dataset_path.stem}_subset_{subset_size}.csv"
    
    # Save subset
    df_subset.to_csv(subset_path, index=False)
    logger.info(f"✓ Subset CSV created: {subset_path}")
    logger.info(f"  Original rows: {total_rows:,}, Subset rows: {len(df_subset):,}")
    
    return subset_path

def simple_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(tok for tok in text.split() if tok)

def preprocess_character_names(character_names_file: Path, logger: logging.Logger) -> Set[str]:
    """
    Extract character names from a text file and preprocess them for use as stopwords.
    
    Args:
        character_names_file: Path to text file containing character names (one per line)
        logger: Logger instance for detailed logging
        
    Returns:
        Set of lowercase character name tokens (first names, last names, etc.)
    """
    logger.info("=" * 60)
    logger.info(f"Loading character names from: {character_names_file}")
    
    if not character_names_file.exists():
        logger.warning(f"Character names file not found: {character_names_file}")
        return set()
    
    # Read all lines
    try:
        lines = character_names_file.read_text(encoding="utf-8").splitlines()
        total_lines = len(lines)
        logger.info(f"  Total lines in file: {total_lines:,}")
    except Exception as e:
        logger.error(f"Failed to read character names file: {e}")
        return set()
    
    # Common non-name words to filter out
    common_non_names = {
        "voice", "god", "karma", "the", "a", "an", "and", "or", "but", "to", "of", "in", 
        "for", "on", "at", "by", "from", "with", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "must", "can", "this", "that", "these", "those", "it", "its", "they",
        "them", "their", "there", "here", "where", "when", "why", "how", "what", "which", "who",
        "whom", "whose", "where", "after", "before", "during", "while", "until", "since",
        "about", "above", "below", "between", "among", "through", "across", "around", "along",
        "against", "toward", "without", "within", "under", "over", "up", "down", "out", "off",
        "away", "back", "forward", "ahead", "behind", "beside", "beyond", "near", "far",
        "old", "new", "young", "big", "small", "large", "little", "good", "bad", "great",
        "beautiful", "ugly", "pretty", "handsome", "nice", "sweet", "dear", "poor", "rich",
        "happy", "sad", "angry", "mad", "glad", "sorry", "afraid", "scared", "brave",
        "strong", "weak", "tall", "short", "long", "wide", "narrow", "thick", "thin",
        "heavy", "light", "hot", "cold", "warm", "cool", "dry", "wet", "clean", "dirty",
        "fast", "slow", "quick", "slowly", "quickly", "always", "never", "often", "sometimes",
        "usually", "rarely", "seldom", "already", "yet", "still", "just", "now", "then",
        "today", "tomorrow", "yesterday", "soon", "late", "early", "recently", "lately",
        "suddenly", "immediately", "finally", "eventually", "recently", "ago", "before",
        "long", "while", "during", "since", "until", "when", "where", "why", "how", "what",
        "which", "who", "whom", "whose", "wherever", "whenever", "however", "whatever",
        "whoever", "whomever", "whichever", "wherever", "anywhere", "everywhere", "nowhere",
        "somewhere", "anytime", "sometime", "sometimes", "always", "never", "forever",
        "again", "once", "twice", "thrice", "first", "second", "third", "last", "next",
        "previous", "earlier", "later", "soon", "recently", "lately", "ago", "before",
        "after", "during", "while", "until", "since", "when", "where", "why", "how",
        "very", "quite", "rather", "too", "so", "such", "more", "most", "less", "least",
        "much", "many", "few", "little", "some", "any", "all", "each", "every", "both",
        "either", "neither", "none", "no", "not", "nothing", "nobody", "nowhere", "never",
        "noone", "someone", "anyone", "everyone", "something", "anything", "everything",
        "somewhere", "anywhere", "everywhere", "somehow", "anyhow", "anyway", "someway"
    }
    
    # Geographic/location names to filter
    geographic_names = {
        "abu", "dhabi", "dubai", "paris", "london", "new", "york", "los", "angeles",
        "san", "francisco", "chicago", "houston", "philadelphia", "phoenix", "san",
        "antonio", "san", "diego", "dallas", "san", "jose", "austin", "jacksonville",
        "fort", "worth", "columbus", "charlotte", "san", "francisco", "indianapolis",
        "seattle", "denver", "washington", "boston", "el", "paso", "detroit", "nashville",
        "memphis", "portland", "oklahoma", "las", "vegas", "louisville", "baltimore",
        "milwaukee", "albuquerque", "tucson", "fresno", "sacramento", "kansas", "city",
        "mesa", "atlanta", "omaha", "raleigh", "miami", "long", "beach", "virginia",
        "oakland", "minneapolis", "tulsa", "tampa", "new", "orleans", "wichita", "arlington",
        "bakersfield", "cleveland", "tulsa", "aurora", "anaheim", "santa", "ana", "st",
        "louis", "corpus", "christi", "riverside", "lexington", "pittsburgh", "anchorage",
        "stockton", "cincinnati", "st", "paul", "toledo", "greensboro", "newark", "plano",
        "henderson", "lincoln", "buffalo", "jersey", "city", "chula", "vista", "fort",
        "wayne", "orlando", "st", "petersburg", "chandler", "laredo", "norfolk", "durham",
        "madison", "lubbock", "iornton", "garland", "glendale", "hialeah", "reno", "chesapeake",
        "gilbert", "baton", "rouge", "irvine", "irving", "spokane", "fremont", "richmond",
        "boise", "san", "bernardino"
    }
    
    # Prefixes to remove
    prefixes_to_remove = [
        r'^["\']+',  # Leading quotes
        r'^\d+\s+',  # Leading numbers
        r'^#\s*',  # Leading #
        r'^a\s+',  # "A "
        r'^an\s+',  # "An "
        r'^aka\s+',  # "AKA "
        r'^the\s+',  # "the "
        r'^\.\.\.\s*',  # "... "
        r'^after\s+',  # "After "
        r'^before\s+',  # "Before "
        r'^dear\s+',  # "Dear "
        r'^mr\.?\s+',  # "Mr. "
        r'^mrs\.?\s+',  # "Mrs. "
        r'^miss\s+',  # "Miss "
        r'^ms\.?\s+',  # "Ms. "
        r'^dr\.?\s+',  # "Dr. "
        r'^professor\s+',  # "Professor "
        r'^prof\.?\s+',  # "Prof. "
    ]
    
    name_tokens = set()
    filtered_count = 0
    filtered_examples = []
    processed_count = 0
    multi_word_count = 0
    
    logger.info("  Processing lines...")
    
    for line_num, line in enumerate(lines, 1):
        original_line = line.strip()
        if not original_line:
            filtered_count += 1
            continue
        
        # Clean the line
        cleaned = original_line.lower().strip()
        
        # Remove prefixes
        for prefix_pattern in prefixes_to_remove:
            cleaned = re.sub(prefix_pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove leading/trailing quotes and punctuation
        cleaned = re.sub(r'^["\']+|["\']+$', '', cleaned)
        cleaned = cleaned.strip()
        
        # Skip if empty after cleaning
        if not cleaned:
            filtered_count += 1
            if len(filtered_examples) < 10:
                filtered_examples.append(original_line[:50])
            continue
        
        # Skip very long lines (likely descriptions/phrases)
        if len(cleaned) > 50:
            filtered_count += 1
            if len(filtered_examples) < 10:
                filtered_examples.append(original_line[:50] + "...")
            continue
        
        # Skip common patterns that indicate descriptions
        description_patterns = [
            r'^the\s+\w+\s+\w+',  # "the X Y"
            r'^a\s+\w+\s+\w+',  # "a X Y"
            r'^an\s+\w+\s+\w+',  # "an X Y"
            r'.*\s+who\s+',  # "... who ..."
            r'.*\s+that\s+',  # "... that ..."
            r'.*\s+which\s+',  # "... which ..."
            r'.*ing\s+\w+',  # "...ing X" (adjectives)
        ]
        
        skip_line = False
        for pattern in description_patterns:
            if re.search(pattern, cleaned, re.IGNORECASE):
                skip_line = True
                break
        
        if skip_line:
            filtered_count += 1
            if len(filtered_examples) < 10:
                filtered_examples.append(original_line[:50])
            continue
        
        # Extract tokens from the cleaned line
        # Remove punctuation and split by whitespace
        tokens = re.sub(r'[^\w\s]', ' ', cleaned).split()
        
        valid_tokens = []
        for token in tokens:
            # Skip tokens that are too short
            if len(token) < 2:
                continue
            
            # Skip tokens that are common non-name words
            if token in common_non_names:
                continue
            
            # Skip geographic names
            if token in geographic_names:
                continue
            
            # Skip tokens that don't start with a letter
            if not token[0].isalpha():
                continue
            
            # Keep the token
            valid_tokens.append(token)
        
        # If we have valid tokens, add them
        if valid_tokens:
            processed_count += 1
            if len(valid_tokens) > 1:
                multi_word_count += 1
            
            for token in valid_tokens:
                name_tokens.add(token)
        else:
            filtered_count += 1
            if len(filtered_examples) < 10:
                filtered_examples.append(original_line[:50])
        
        # Log progress every 1000 lines
        if line_num % 1000 == 0:
            logger.info(f"    Processed {line_num:,}/{total_lines:,} lines, extracted {len(name_tokens):,} unique name tokens so far...")
    
    # Final logging
    logger.info(f"  Processing complete!")
    logger.info(f"    Total lines processed: {total_lines:,}")
    logger.info(f"    Lines filtered out: {filtered_count:,} ({filtered_count/total_lines*100:.1f}%)")
    if filtered_examples:
        examples_str = ', '.join([f'"{ex}"' for ex in filtered_examples[:5]])
        logger.info(f"    Filtered examples: {examples_str}")
    logger.info(f"    Lines with valid names: {processed_count:,}")
    logger.info(f"    Multi-word names processed: {multi_word_count:,}")
    logger.info(f"    Unique name tokens extracted: {len(name_tokens):,}")
    
    # Show some examples
    if name_tokens:
        example_names = sorted(list(name_tokens))[:20]
        logger.info(f"    Example names added: {', '.join(example_names)}")
        if len(name_tokens) > 20:
            logger.info(f"    ... and {len(name_tokens) - 20} more")
    
    logger.info("=" * 60)
    
    return name_tokens

def load_and_preprocess_docs(dataset_csv: Path, text_column: str, custom_stops: Optional[Path], 
                             chunk_size: int, logger: logging.Logger, character_names_file: Optional[Path] = None) -> Tuple[List[str], List[str], Set[str]]:
    """
    Load documents from CSV and perform preprocessing.
    
    Returns:
        - raw_docs: A list of original, unprocessed document strings.
        - cleaned_docs: A list of cleaned document strings (for vectorizer), same length as raw_docs.
        - stops: A set of all stopwords used during cleaning.
    """
    logger.info(f"Loading dataset from: {dataset_csv}")
    
    # First, check total size
    logger.info("Counting total rows in dataset...")
    total_rows = sum(1 for _ in open(dataset_csv, 'r', encoding='utf-8')) - 1  # Subtract header
    logger.info(f"Total rows in dataset: {total_rows:,}")
    
    # Read first row to check columns
    df_sample = pd.read_csv(dataset_csv, nrows=1)
    if text_column not in df_sample.columns:
        raise ValueError(f"Missing column '{text_column}' in dataset. Available columns: {list(df_sample.columns)}")
    logger.info(f"Using text column: '{text_column}'")
    
    # Load stopwords
    stops = set(ENGLISH_STOP_WORDS)
    initial_count = len(stops)
    logger.info(f"Initial stopwords (English): {initial_count:,}")
    
    # Load custom stopwords if provided
    if custom_stops and custom_stops.exists():
        logger.info(f"Loading custom stopwords from: {custom_stops}")
        custom_count_before = len(stops)
        for line in custom_stops.read_text(encoding="utf-8").splitlines():
            for t in re.sub(r"[^\w\s]", " ", line.lower()).split():
                if t:
                    stops.add(t)
        custom_added = len(stops) - custom_count_before
        logger.info(f"  Added {custom_added:,} custom stopwords")
    
    # Load character names if provided
    if character_names_file:
        character_names = preprocess_character_names(character_names_file, logger)
        char_count_before = len(stops)
        stops.update(character_names)
        char_added = len(stops) - char_count_before
        logger.info(f"Added {char_added:,} character names to stopwords list")
        logger.info(f"Total stopwords now: {len(stops):,} ({initial_count:,} standard + {len(stops) - initial_count:,} custom/character names)")
        if character_names:
            example_chars = sorted(list(character_names))[:10]
            logger.info(f"  Example character names: {', '.join(example_chars)}")
    else:
        logger.info(f"Total stopwords: {len(stops):,} (no character names file provided)")
    
    raw_docs = []
    cleaned_docs = []
    chunks_processed = 0
    
    logger.info(f"Processing dataset in chunks of {chunk_size:,} rows...")
    for chunk_df in tqdm(pd.read_csv(dataset_csv, chunksize=chunk_size), 
                         desc="Loading chunks", total=(total_rows // chunk_size + 1)):
        chunks_processed += 1
        for x in chunk_df[text_column].astype(str).tolist():
            raw_docs.append(x)
            
            c = simple_clean(x)
            toks = [t for t in c.split() if t not in stops]
            cleaned_docs.append(" ".join(toks))
        
        if chunks_processed % 10 == 0:
            logger.info(f"  Processed {chunks_processed} chunks, loaded {len(raw_docs):,} docs so far...")
    
    logger.info(f"Loaded {len(raw_docs):,} raw documents from {chunks_processed} chunks.")
    return raw_docs, cleaned_docs, stops


def make_umap_hdbscan(n_neighbors:int, n_components:int, min_dist:float,
                      hdb_min_cluster_size:int, hdb_min_samples:int,
                      seed:int, logger: logging.Logger,
                      umap_metric: str = 'cosine',
                      hdbscan_metric: str = 'euclidean',
                      hdbscan_cluster_selection_method: str = 'eom'):
    """
    Create UMAP and HDBSCAN models, preferring GPU if available and functional.
    Uses early GPU availability check to avoid runtime failures.
    """
    # Check GPU availability first (uses cached result if already tested)
    gpu_available = check_gpu_availability()
    
    # Try GPU first, but only if availability check passed
    if gpu_available:
        try:
            from cuml.manifold import UMAP as GPU_UMAP
            from cuml.cluster import HDBSCAN as GPU_HDBSCAN
            
            logger.info("✓ GPU (cuML) available - using GPU UMAP + HDBSCAN")
            logger.info(f"  UMAP params: n_neighbors={n_neighbors}, n_components={n_components}, min_dist={min_dist}, metric={umap_metric}")
            logger.info(f"  HDBSCAN params: min_cluster_size={hdb_min_cluster_size}, min_samples={hdb_min_samples}, metric={hdbscan_metric}, cluster_selection_method={hdbscan_cluster_selection_method}")
            logger.info("  Initializing GPU UMAP model...")
            umap_model = GPU_UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist,
                                  metric=umap_metric, random_state=seed, verbose=logger.level <= logging.DEBUG)
            logger.info("  Initializing GPU HDBSCAN model...")
            hdbscan_model = GPU_HDBSCAN(min_cluster_size=hdb_min_cluster_size, min_samples=hdb_min_samples,
                                        metric=hdbscan_metric, cluster_selection_method=hdbscan_cluster_selection_method,
                                        prediction_data=True, gen_min_span_tree=True)
            logger.info("  GPU models initialized successfully")
            used_gpu = True
            return umap_model, hdbscan_model, used_gpu
        except Exception as e:
            logger.warning(f"GPU initialization failed despite availability check: {e}")
            logger.info("Falling back to CPU...")
            # Fall through to CPU implementation
    else:
        logger.info("GPU (cuML) not available or not functional - using CPU UMAP + HDBSCAN")
    
    # CPU implementation
    from umap import UMAP as CPU_UMAP
    import hdbscan as CPU_HDBSCAN
    logger.info(f"  UMAP params: n_neighbors={n_neighbors}, n_components={n_components}, min_dist={min_dist}, metric={umap_metric}")
    logger.info(f"  HDBSCAN params: min_cluster_size={hdb_min_cluster_size}, min_samples={hdb_min_samples}, metric={hdbscan_metric}, cluster_selection_method={hdbscan_cluster_selection_method}")
    umap_model = CPU_UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist,
                          metric=umap_metric, random_state=seed)
    hdbscan_model = CPU_HDBSCAN.HDBSCAN(min_cluster_size=hdb_min_cluster_size, min_samples=hdb_min_samples,
                                        metric=hdbscan_metric, cluster_selection_method=hdbscan_cluster_selection_method,
                                        prediction_data=True, gen_min_span_tree=True)
    used_gpu = False
    return umap_model, hdbscan_model, used_gpu

def get_embeddings(model_name: str, docs: List[str], embeddings_dir: Path,
                   device: str, cache: bool, logger: logging.Logger) -> Optional[np.ndarray]:
    """Load or compute embeddings, caching to models/embeddings/ directory."""
    key = model_name.replace("/", "_")
    fpath = embeddings_dir / f"{key}.npy"
    if cache and fpath.exists():
        logger.info(f"  ✓ Loading cached embeddings from: {fpath}")
        logger.info(f"    File size: {fpath.stat().st_size / (1024**2):.2f} MB")
        start_load = time.time()
        embs = np.load(fpath)
        load_time = time.time() - start_load
        logger.info(f"    Loaded {len(docs):,} embeddings ({embs.shape[1]} dims) in {load_time:.2f} seconds")
        return embs
    try:
        logger.info(f"  → Encoding embeddings: {model_name}")
        logger.info(f"    Device: {device}")
        logger.info(f"    Documents: {len(docs):,}")
        logger.info(f"    Batch size: 64")
        logger.info(f"    This may take several minutes...")
        
        start_encode = time.time()
        st = SentenceTransformer(model_name, device=device)
        load_model_time = time.time() - start_encode
        logger.info(f"    Model loaded in {load_model_time:.2f} seconds")
        
        logger.info(f"    Starting encoding (progress bar will show below)...")
        encode_start = time.time()
        embs = st.encode(docs, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        encode_time = time.time() - encode_start
        
        logger.info(f"    ✓ Encoding completed in {encode_time:.2f} seconds ({encode_time/60:.1f} minutes)")
        logger.info(f"    Embedding shape: {embs.shape} ({embs.shape[0]:,} docs × {embs.shape[1]} dims)")
        logger.info(f"    Throughput: {len(docs) / encode_time:.1f} docs/second")
        
        if cache:
            save_start = time.time()
            np.save(fpath, embs)
            save_time = time.time() - save_start
            file_size_mb = fpath.stat().st_size / (1024**2)
            logger.info(f"    ✓ Saved embeddings to: {fpath}")
            logger.info(f"    File size: {file_size_mb:.2f} MB (saved in {save_time:.2f} seconds)")
        
        return embs
    except Exception as e:
        error_msg = str(e)
        if "not a valid model identifier" in error_msg:
            logger.warning(f"  ✗ Model '{model_name}' not found on HuggingFace. This model will be skipped, but training will continue with available models.")
        else:
            logger.error(f"  ✗ Failed to load/encode for '{model_name}': {error_msg}")
        logger.debug(f"    Full error details: {e}", exc_info=True)
        return None

def topic_diversity(topics: Dict[int, List[Tuple[str, float]]], top_k:int=10) -> float:
    bag = []
    for k, items in topics.items():
        if k == -1:
            continue
        bag.extend([w for (w, _) in items[:top_k]])
    return 0.0 if not bag else len(set(bag)) / float(len(bag))

def compute_c_npmi(docs: List[str], topics: Dict[int, List[Tuple[str, float]]],
                   top_k:int=10, sample:int=25000, logger: Optional[logging.Logger] = None) -> Optional[float]:
    """
    Compute c_nPMI coherence metric.
    
    Args:
        docs: List of document strings
        topics: Dictionary mapping topic IDs to lists of (word, score) tuples
        top_k: Number of top words per topic to use
        sample: Number of documents to sample for computation (0 = use all)
        logger: Optional logger for verbose output
    """
    if not HAS_GENSIM:
        return None
    
    if logger:
        n_docs_total = len(docs)
        if sample > 0 and sample < n_docs_total:
            logger.info(f"    Sampling {sample:,} documents from {n_docs_total:,} total for coherence computation...")
        else:
            logger.info(f"    Using all {n_docs_total:,} documents for coherence computation...")
    
    start_time = time.time()
    
    # Sample documents if needed
    if sample > 0 and sample < len(docs):
        rng = np.random.default_rng(42)
        sampled_docs = rng.choice(docs, size=sample, replace=False).tolist()
        toks = [d.split() for d in sampled_docs]
    else:
        toks = [d.split() for d in docs]
    
    if logger:
        sample_time = time.time() - start_time
        logger.info(f"    Tokenized {len(toks):,} documents in {sample_time:.2f} seconds")
    
    topic_words = [[w for (w, _) in topics[k][:top_k]] for k in sorted([t for t in topics if t != -1])]
    if not topic_words:
        if logger:
            logger.warning("    No topics found for coherence computation")
        return None
    
    if logger:
        logger.info(f"    Computing coherence for {len(topic_words)} topics...")
        logger.info(f"    Building dictionary and corpus...")
    
    dict_start = time.time()
    dictionary = Dictionary(toks)
    corpus = [dictionary.doc2bow(x) for x in toks]
    dict_time = time.time() - dict_start
    if logger:
        logger.info(f"    Dictionary built: {len(dictionary)} unique tokens, corpus size: {len(corpus):,}")
        logger.info(f"    Dictionary construction took {dict_time:.2f} seconds")
    
    if logger:
        logger.info(f"    Computing c_nPMI coherence metric (this may take a minute)...")
    
    coherence_start = time.time()
    cm = CoherenceModel(topics=topic_words, texts=toks, corpus=corpus, dictionary=dictionary, coherence="c_npmi")
    coherence_score = float(cm.get_coherence())
    coherence_time = time.time() - coherence_start
    
    if logger:
        logger.info(f"    ✓ Coherence computation completed in {coherence_time:.2f} seconds")
    
    return coherence_score

def train_one(docs: List[str], embs: np.ndarray, row: Dict, cfg: Config, 
              runs_dir: Path, logger: logging.Logger, device: str = "cpu") -> Dict:
    """
    Train a single BERTopic model and save outputs to csv/runs/ directory.
    
    Note: row["Iteration"] is used only for logging/identification, NOT as a hyperparameter.
    """
    model_name = row["Embeddings_Model"]
    iteration = row["Iteration"]  # Used for identification/tracking only, not a hyperparameter
    logger.info(f"=" * 60)
    logger.info(f"Training run: {model_name} (Iteration {iteration})")
    logger.info(f"  Input: {len(docs):,} documents, {embs.shape[1]} dimensions")
    
    logger.info("  Initializing CountVectorizer...")
    min_df_value = float(row["vectorizer__min_df"])
    min_df_count = max(1, int(min_df_value * len(docs))) if min_df_value < 1.0 else int(min_df_value)
    vec = CountVectorizer(stop_words="english", min_df=min_df_value)
    logger.info(f"    min_df={min_df_value} (absolute minimum: {min_df_count:,} documents)")
    
    logger.info("  Initializing UMAP and HDBSCAN models...")
    # Extract non-numerical parameters with defaults from OCTIS
    umap_metric = row.get("umap__metric", "cosine")
    hdbscan_metric = row.get("hdbscan__metric", "euclidean")
    hdbscan_cluster_selection_method = row.get("hdbscan__cluster_selection_method", "eom")
    
    umap_model, hdbscan_model, used_gpu = make_umap_hdbscan(
        n_neighbors=int(row["umap__n_neighbors"]),
        n_components=int(row["umap__n_components"]),
        min_dist=float(row["umap__min_dist"]),
        hdb_min_cluster_size=int(row["hdbscan__min_cluster_size"]),
        hdb_min_samples=int(row["hdbscan__min_samples"]),
        seed=cfg.seed,
        logger=logger,
        umap_metric=umap_metric,
        hdbscan_metric=hdbscan_metric,
        hdbscan_cluster_selection_method=hdbscan_cluster_selection_method,
    )
    
    logger.info("  Loading embedding model for KeyBERTInspired representation...")
    logger.info(f"    Model: {model_name}, Device: {device}")
    load_start = time.time()
    embedding_model = SentenceTransformer(model_name, device=device)
    load_time = time.time() - load_start
    logger.info(f"    ✓ Model loaded in {load_time:.2f} seconds")
    
    logger.info("  Initializing BERTopic model...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vec,
        representation_model={"Main": KeyBERTInspired(top_n_words=int(row["bertopic__top_n_words"]))},
        top_n_words=int(row["bertopic__top_n_words"]),
        min_topic_size=int(row["bertopic__min_topic_size"]),
        language="english",
        calculate_probabilities=False,
        verbose=cfg.verbose,
    )
    
    logger.info("  Starting fit_transform (UMAP + HDBSCAN)...")
    logger.info(f"    This may take several minutes for {len(docs):,} documents...")
    logger.info(f"    Progress: UMAP dimensionality reduction → HDBSCAN clustering")
    start_time = time.time()
    topics, _ = topic_model.fit_transform(docs, embs)
    elapsed = time.time() - start_time
    logger.info(f"  ✓ fit_transform completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"    Throughput: {len(docs) / elapsed:.1f} documents/second")
    
    logger.info("  Extracting topic information...")
    topics_info = topic_model.get_topic_info()
    topics_dict = topic_model.get_topics()

    logger.info("  Computing metrics...")
    n_out = int((np.array(topics) == -1).sum())
    outlier_ratio = n_out / max(1, len(topics))
    n_topics = int((topics_info["Topic"] != -1).sum())
    logger.info(f"    Topics found: {n_topics}")
    logger.info(f"    Outliers: {n_out:,} ({outlier_ratio*100:.1f}%)")
    
    logger.info("  Computing topic diversity...")
    diversity = topic_diversity(topics_dict, top_k=min(10, int(row["bertopic__top_n_words"])) )
    logger.info(f"    Topic diversity: {diversity:.3f}")
    
    if HAS_GENSIM:
        logger.info(f"  Computing c_nPMI coherence (sample={cfg.coherence_sample})...")
        # Coherence is calculated on the cleaned text (`docs`)
        c_npmi = compute_c_npmi(docs, topics_dict, top_k=10, sample=cfg.coherence_sample, logger=logger)
        logger.info(f"    c_nPMI: {c_npmi:.4f}" if c_npmi is not None else "    c_nPMI: None")
    else:
        c_npmi = None
        logger.info("  Skipping c_nPMI (gensim not available)")

    if cfg.save_topics_info:
        short = f"{row['Embeddings_Model'].replace('/','_')}_it{row['Iteration']}"
        run_dir = runs_dir / short
        run_dir.mkdir(parents=True, exist_ok=True)
        topics_info.to_csv(run_dir / "topics_info.csv", index=False)
        prev = []
        for _, r in topics_info[topics_info["Topic"] != -1].head(15).iterrows():
            t = int(r["Topic"])
            words = ", ".join([w for w, _ in topics_dict[t]][:10])
            prev.append({"Topic": t, "Count": int(r["Count"]), "TopWords": words})
        pd.DataFrame(prev).to_csv(run_dir / "preview.csv", index=False)

    return {
        "n_topics": n_topics,
        "outlier_ratio": outlier_ratio,
        "topic_diversity": diversity,
        "coherence_c_npmi": c_npmi,
        "using_gpu": used_gpu,
    }

def rank_and_score(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    def minmax(col):
        vals = pd.to_numeric(df[col], errors="coerce")
        lo, hi = vals.min(), vals.max()
        if pd.isna(lo) or pd.isna(hi) or math.isclose(lo, hi):
            return pd.Series([0.5] * len(vals), index=vals.index)
        return (vals - lo) / (hi - lo)
    df["norm_diversity"] = minmax("topic_diversity")
    df["norm_coherence"] = minmax("coherence_c_npmi") if df["coherence_c_npmi"].notna().any() else 0.0
    df["score"] = (
        cfg.w_diversity * df["norm_diversity"] +
        cfg.w_coherence * df["norm_coherence"] -
        cfg.penalty_outliers * df["outlier_ratio"].astype(float)
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--text_column", type=str, default="Sentence")
    ap.add_argument("--coherence_sample", type=int, default=25000)
    ap.add_argument("--save_topics_info", action="store_true")
    ap.add_argument("--no_cache_embeddings", action="store_true")
    ap.add_argument("--custom_stopwords_file", type=Path)
    ap.add_argument("--character_names_file", type=Path, help="Path to text file containing character names (one per line) to add to stopwords")
    ap.add_argument("--chunk_size", type=int, default=50000, help="Dataset chunk size for processing large files")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    ap.add_argument("--create_subset", action="store_true", help="Create a subset CSV with 10,000 randomly sampled rows for testing")
    args = ap.parse_args()

    cfg = Config(
        dataset_csv=args.dataset_csv,
        out_dir=args.out_dir,
        text_column=args.text_column,
        coherence_sample=max(0, args.coherence_sample),
        save_topics_info=args.save_topics_info,
        cache_embeddings=not args.no_cache_embeddings,
        custom_stopwords_file=args.custom_stopwords_file,
        character_names_file=args.character_names_file,
        chunk_size=args.chunk_size,
        verbose=args.verbose,
    )

    # Create organized output directory structure
    dirs = setup_output_dirs(cfg.out_dir)
    logger = setup_logging(dirs["logs"])
    
    logger.info("=" * 80)
    logger.info("BERTopic Training from Tables")
    logger.info("=" * 80)
    logger.info(f"Output directory structure created in: {cfg.out_dir}")
    logger.info(f"  - Logs: {dirs['logs']}")
    logger.info(f"  - Models/Embeddings: {dirs['embeddings']}")
    logger.info(f"  - CSV files: {dirs['csv']}")
    logger.info(f"  - JSON: {dirs['json']}")
    logger.info(f"  - TXT: {dirs['txt']}")
    logger.info(f"\nConfiguration:")
    logger.info(f"  - Dataset: {cfg.dataset_csv}")
    logger.info(f"  - Text column: {cfg.text_column}")
    logger.info(f"  - Chunk size: {cfg.chunk_size:,}")
    logger.info(f"  - Verbose: {cfg.verbose}")
    logger.info(f"  - Coherence sample: {cfg.coherence_sample:,}")
    logger.info(f"  - Cache embeddings: {cfg.cache_embeddings}")
    logger.info(f"  - Save topics info: {cfg.save_topics_info}")
    logger.info(f"  - Random seed: {cfg.seed}")
    
    # Log tokenizer parallelism setting
    tokenizer_parallelism = os.environ.get("TOKENIZERS_PARALLELISM", "not set")
    logger.info(f"  - TOKENIZERS_PARALLELISM: {tokenizer_parallelism} (warnings suppressed)")
    
    if not HAS_GENSIM:
        logger.warning("gensim not found → c_nPMI coherence will be skipped.")
    else:
        logger.info("✓ gensim available → c_nPMI coherence will be computed")

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # Check GPU availability for embeddings (separate from cuML GPU check)
    # Only use CUDA for embeddings if PyTorch CUDA is actually working
    device = "cpu"  # Default to CPU
    if torch.cuda.is_available():
        try:
            # Quick test to ensure CUDA actually works
            test_tensor = torch.zeros(1).cuda()
            device = "cuda"
            logger.info(f"Embedding device: {device} (PyTorch CUDA available)")
        except Exception as e:
            logger.warning(f"PyTorch CUDA available but not functional: {e}")
            logger.info("Using CPU for embeddings")
            device = "cpu"
    else:
        logger.info(f"Embedding device: {device} (PyTorch CUDA not available)")
    
    # Log GPU status for UMAP/HDBSCAN separately
    gpu_status = "available" if check_gpu_availability() else "not available or not functional"
    logger.info(f"UMAP/HDBSCAN GPU (cuML): {gpu_status}")

    # Create subset if requested
    if args.create_subset:
        subset_path = create_subset_csv(cfg.dataset_csv, cfg.text_column, subset_size=10000, seed=cfg.seed, logger=logger)
        logger.info(f"Subset created. Update --dataset_csv to use: {subset_path}")
        logger.info("Exiting. Run again with --dataset_csv pointing to the subset file to train on it.")
        return

    # Main execution
    # Correctly call load_and_preprocess_docs
    raw_docs, cleaned_docs, stops = load_and_preprocess_docs(cfg.dataset_csv, cfg.text_column, cfg.custom_stopwords_file, cfg.chunk_size, logger, cfg.character_names_file)

    grid = param_rows_from_tables()
    logger.info(f"Runs to train: {len(grid)}")

    embs_cache: Dict[str, Optional[np.ndarray]] = {}
    unique_models = sorted(set(r["Embeddings_Model"] for r in grid))
    logger.info("=" * 80)
    logger.info(f"Loading embeddings for {len(unique_models)} unique models")
    logger.info("=" * 80)
    logger.info(f"Models to process: {unique_models}")
    logger.info(f"Total documents: {len(raw_docs):,}")
    
    embeddings_start_time = time.time()
    for idx, name in enumerate(unique_models, 1):
        logger.info(f"\n[{idx}/{len(unique_models)}] Processing embedding model: {name}")
        embs_cache[name] = get_embeddings(
            name, raw_docs, dirs["embeddings"], device, cfg.cache_embeddings, logger
        )
    embeddings_total_time = time.time() - embeddings_start_time
    
    successful_models = [m for m, emb in embs_cache.items() if emb is not None]
    failed_models = [m for m, emb in embs_cache.items() if emb is None]
    
    logger.info("\n" + "=" * 80)
    logger.info("Embedding Loading Summary")
    logger.info("=" * 80)
    logger.info(f"Total time: {embeddings_total_time:.1f} seconds ({embeddings_total_time/60:.1f} minutes)")
    logger.info(f"Successfully loaded: {len(successful_models)} model(s)")
    logger.info(f"  {successful_models}")
    if failed_models:
        logger.warning(f"Failed to load: {len(failed_models)} model(s)")
        logger.warning(f"  {failed_models}")
        logger.warning("Runs using these models will be skipped")
    logger.info("=" * 80)

    # --------------------------------------------------------------------------------
    # Filter documents based on token count AFTER cleaning
    # --------------------------------------------------------------------------------
    threshold = 1
    valid_indices = [i for i, doc in enumerate(cleaned_docs) if len(doc.split()) >= threshold]
    original_count = len(raw_docs)
    filtered_count = original_count - len(valid_indices)

    logger.info(f"\nFiltering documents: keeping docs with >= {threshold} token(s) after cleaning.")
    logger.info(f"  - Original docs: {original_count:,}")
    logger.info(f"  - Docs to keep: {len(valid_indices):,}")
    logger.info(f"  - Filtered out: {filtered_count:,} ({filtered_count/original_count*100:.1f}%)")

    # Log examples of filtered documents
    if filtered_count > 0:
        filtered_examples = []
        count = 0
        for i, doc in enumerate(cleaned_docs):
            if i not in valid_indices:
                original_text_preview = raw_docs[i][:100].replace('\n', ' ')
                cleaned_text_preview = doc
                tok_len = len(doc.split())
                filtered_examples.append(f"Original: '{original_text_preview}' -> Cleaned: '{cleaned_text_preview}' ({tok_len} tokens)")
                count += 1
                if count >= 20:
                    break
        if filtered_examples:
            logger.info("  - Examples of filtered documents:")
            for example in filtered_examples:
                logger.info(f"    - {example}")

    final_cleaned_docs = [cleaned_docs[i] for i in valid_indices]
    
    # Filter all cached embeddings
    final_embs_cache = {}
    for name, embs in embs_cache.items():
        if embs is not None:
            final_embs_cache[name] = embs[valid_indices]
    # --------------------------------------------------------------------------------

    rows_out = []
    total_runs = len(grid)
    successful_runs = 0
    logger.info("=" * 80)
    logger.info(f"Starting training of {total_runs} runs")
    logger.info("=" * 80)
    
    for idx, row in enumerate(tqdm(grid, desc="Training runs"), 1):
        name = row["Embeddings_Model"]
        iteration = row["Iteration"]
        logger.info(f"\n[{idx}/{total_runs}] Processing: {name} (Iteration {iteration})")
        
        embs = final_embs_cache.get(name)
        if embs is None:
            logger.warning(f"  ✗ Skipping (no embeddings available)")
            continue
        try:
            # Pass the final_cleaned_docs to train_one
            result = train_one(final_cleaned_docs, embs, row, cfg, dirs["runs"], logger, device)
            successful_runs += 1
            logger.info(f"  ✓ Completed successfully")
            logger.info(f"     Topics: {result['n_topics']}, Diversity: {result['topic_diversity']:.3f}, "
                       f"Outlier ratio: {result['outlier_ratio']:.3f}")
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            logger.debug(f"Full error traceback:", exc_info=True)
            continue
        rec = {
            "Embeddings_Model": name,
            "Iteration": int(row["Iteration"]),
            **result,
            "bertopic__min_topic_size": int(row["bertopic__min_topic_size"]),
            "bertopic__top_n_words": int(row["bertopic__top_n_words"]),
            "hdbscan__min_cluster_size": int(row["hdbscan__min_cluster_size"]),
            "hdbscan__min_samples": int(row["hdbscan__min_samples"]),
            "umap__min_dist": float(row["umap__min_dist"]),
            "umap__n_components": int(row["umap__n_components"]),
            "umap__n_neighbors": int(row["umap__n_neighbors"]),
            "vectorizer__min_df": float(row["vectorizer__min_df"]),
        }
        rows_out.append(rec)

    logger.info("=" * 80)
    logger.info(f"Training completed: {successful_runs}/{total_runs} runs successful")
    logger.info("=" * 80)
    
    if not rows_out:
        logger.error("No successful runs. Exiting.")
        return

    logger.info("\nComputing rankings and scores...")
    df = pd.DataFrame(rows_out)
    ranked = rank_and_score(df, cfg)
    out_csv = dirs["csv"] / "summary.csv"
    ranked.to_csv(out_csv, index=False)
    logger.info(f"✓ Summary written: {out_csv}")
    
    cols = ["Embeddings_Model","Iteration","n_topics","outlier_ratio","topic_diversity","coherence_c_npmi","score"]
    show = ranked[cols].head(5)
    logger.info("\n" + "=" * 80)
    logger.info("Top-5 Leaderboard:")
    logger.info("=" * 80)
    logger.info("\n" + show.to_string(index=False))
    logger.info("\n" + "=" * 80)

if __name__ == "__main__":
    main()
