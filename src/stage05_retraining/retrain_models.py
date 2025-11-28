"""Core retraining logic for Pareto-efficient models."""

import os
import json
import pickle
import csv
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Helper functions for formatted output (matching stage03_modeling style)
def print_flush(*args, **kwargs):
    """Print with automatic flushing for real-time output."""
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


def print_section(title: str, level: int = 1):
    """Print a formatted section header."""
    print(f"[SECTION] Printing section header: {title} (level {level})", flush=True)
    char = "=" if level == 1 else "-"
    width = 80
    print(f"\n{char * width}", flush=True)
    print(f"{title.center(width)}", flush=True)
    print(f"{char * width}\n", flush=True)
    print(f"[SECTION] ✓ Section header printed", flush=True)


def print_step(step_num: int, description: str):
    """Print a numbered step."""
    print(f"[STEP] Printing step {step_num}: {description}", flush=True)
    print(f"\n[STEP {step_num}] {description}", flush=True)
    print("-" * 80, flush=True)
    print(f"[STEP] ✓ Step {step_num} header printed", flush=True)

# Enable fast Hugging Face downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Set persistent cache directories
cache_dir = os.path.join(os.getcwd(), "cache", "huggingface")
os.makedirs(cache_dir, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import GPU utilities
from src.common.gpu_models import print_gpu_status, check_rapids_availability
from src.stage03_modeling.memory_utils import (
    print_gpu_memory_usage, cleanup_gpu_memory, track_memory_peak,
    log_memory_usage
)

# Check RAPIDS availability
if not check_rapids_availability():
    raise ImportError(
        "RAPIDS (cuML) is required but not available. "
        "Install with: pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com"
    )

# Import RAPIDS models
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP

# Import BERTopic and utilities
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from bertopic.vectorizers import ClassTfidfTransformer
from octis.dataset.dataset import Dataset

# Import from stage03
from src.stage03_modeling.bertopic_octis_model import (
    BERTopicOctisModelWithEmbeddings,
    load_embedding_model,
    create_representation_models
)


def load_custom_stopwords(stoplist_path: Path) -> List[str]:
    """
    Load custom stopwords from a file.
    
    Args:
        stoplist_path: Path to the custom stoplist file
        
    Returns:
        List of stopwords
    """
    if not stoplist_path.exists():
        print(f"⚠️ Custom stoplist not found at: {stoplist_path}")
        return []
        
    print(f"📖 Loading custom stopwords from: {stoplist_path}")
    stopwords = set()
    try:
        with open(stoplist_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                word = line.strip()
                if word and not word.startswith('#'):
                    # Add original
                    stopwords.add(word)
                    # Add lowercase
                    stopwords.add(word.lower())
                    
        print(f"✅ Loaded {len(stopwords)} unique custom stopwords")
        return list(stopwords)
    except Exception as e:
        print(f"❌ Error loading custom stoplist: {e}")
        return []


def clean_text(text: str) -> str:
    """
    Clean text by fixing mojibake and normalizing unicode.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return str(text)
        
    # Fix common mojibake artifacts
    mojibake_map = {
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€“': '-',
        'â€”': '-',
        'â€˜': "'",
        'â€¦': '...',
        'â€': '"', # Generic catch-all for remaining quotes
    }
    
    for bad, good in mojibake_map.items():
        text = text.replace(bad, good)
    
    # Normalize unicode (decomposes characters, e.g., é -> e + accent)
    # NFKD compatibility decomposition
    text = unicodedata.normalize('NFKD', text)
    
    # Remove newlines and excessive whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip().lower()


def load_and_validate_csv(csv_path: Path) -> Tuple[pd.DataFrame, List[str], List[List[str]]]:
    """
    Load CSV file and validate structure.
    
    Returns:
        df: DataFrame with cleaned sentences
        dataset_as_list_of_strings: List of sentence strings
        dataset_as_list_of_lists: List of tokenized sentences
    """
    print_step(1, f"Loading CSV file: {csv_path}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"📁 File exists: {csv_path}")
    print(f"📊 File size: {csv_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Read CSV with latin1 encoding
    print("\n📖 Reading CSV with latin1 encoding...")
    all_rows = []
    
    with open(csv_path, 'r', encoding='latin1', errors='ignore') as file:
        reader = csv.reader(file, quotechar='"', delimiter=',', 
                           quoting=csv.QUOTE_ALL, skipinitialspace=True)
        headers = next(reader)  # Get headers
        print(f"📋 Headers: {headers}")
        
        for row in reader:
            if len(row) == 4:
                all_rows.append(row)
    
    print(f"✅ Total rows loaded: {len(all_rows)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_rows, columns=headers)
    print(f"📊 DataFrame shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Validate 'Sentence' column exists
    if 'Sentence' not in df.columns:
        raise ValueError(f"'Sentence' column not found. Available columns: {df.columns}")
    
    print("\n🧹 Cleaning sentences...")
    print("   - Fixing mojibake artifacts")
    print("   - Normalizing unicode")
    print("   - Removing newlines")
    print("   - Normalizing whitespace")
    print("   - Converting to lowercase")
    
    # Clean sentences
    df['Sentence'] = df['Sentence'].apply(clean_text)
    
    # Show sample
    print("\n📝 Sample cleaned sentences:")
    for i, sentence in enumerate(df['Sentence'].head(3)):
        print(f"   {i+1}. {sentence[:100]}...")
    
    # Convert to lists
    dataset_as_list_of_strings = df['Sentence'].tolist()
    print(f"\n✅ Total sentences: {len(dataset_as_list_of_strings)}")
    
    # Check for duplicates
    duplicate_count = df['Sentence'].duplicated().sum()
    print(f"🔍 Duplicate sentences: {duplicate_count} ({duplicate_count/len(df)*100:.2f}%)")
    
    # Tokenize (convert to list of lists)
    print("\n🔤 Tokenizing sentences...")
    dataset_as_list_of_lists = [sentence.split() for sentence in dataset_as_list_of_strings]
    
    # Filter empty documents
    non_empty = [doc for doc in dataset_as_list_of_lists if len(doc) > 0]
    print(f"✅ Non-empty documents: {len(non_empty)}")
    print(f"⚠️  Empty documents filtered: {len(dataset_as_list_of_lists) - len(non_empty)}")
    
    print(f"[STEP 1] ✓ CSV loading completed successfully")
    return df, dataset_as_list_of_strings, non_empty


def create_octis_dataset(csv_path: Path, octis_dataset_path: Path, 
                        df: pd.DataFrame) -> Path:
    """Create OCTIS dataset format (corpus.tsv) using cleaned DataFrame."""
    print_step(2, f"Creating OCTIS dataset format")
    
    # Create directory if needed
    octis_dataset_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 OCTIS dataset directory: {octis_dataset_path}")
    
    corpus_tsv_path = octis_dataset_path / 'corpus.tsv'
    print(f"📄 Output file: {corpus_tsv_path}")
    
    # Use cleaned DataFrame instead of re-reading raw CSV
    print("\n📖 Preparing OCTIS format from cleaned DataFrame...")
    print(f"   Using cleaned sentences (already processed for mojibake, unicode, etc.)")
    
    # Identify column names (handle different possible column name formats)
    sentence_col = 'Sentence'
    author_col = None
    book_title_col = None
    
    # Try to find author and book title columns
    possible_author_names = ['Author', 'author', 'Author_Name', 'author_name']
    possible_book_names = ['Book_Title', 'book_title', 'Book', 'book', 'Title', 'title']
    
    for col in df.columns:
        if col in possible_author_names:
            author_col = col
        if col in possible_book_names:
            book_title_col = col
    
    # If not found, try to infer from column order (assuming standard: Author, Book_Title, Chapter, Sentence)
    if author_col is None or book_title_col is None:
        cols = list(df.columns)
        if len(cols) >= 4:
            # Assume standard order: Author, Book_Title, Chapter, Sentence
            if author_col is None and cols[0].lower() not in ['sentence', 'chapter']:
                author_col = cols[0]
            if book_title_col is None and cols[1].lower() not in ['sentence', 'chapter']:
                book_title_col = cols[1]
    
    print(f"   Detected columns: Sentence='{sentence_col}', Author='{author_col}', Book_Title='{book_title_col}'")
    
    # Prepare TSV data using cleaned DataFrame
    tsv_data = []
    for idx, row in df.iterrows():
        sentence = row[sentence_col]  # Already cleaned by clean_text()
        partition = 'train'
        
        # Build label from author and book title if available
        if author_col and book_title_col and author_col in df.columns and book_title_col in df.columns:
            author = str(row[author_col]) if pd.notna(row[author_col]) else 'unknown'
            book_title = str(row[book_title_col]) if pd.notna(row[book_title_col]) else 'unknown'
            label = f"{author},{book_title}"
        else:
            # Fallback: use index or generic label
            label = f"doc_{idx}"
        
        tsv_data.append([sentence, partition, label])
    
    print(f"✅ Prepared {len(tsv_data)} rows for OCTIS format")
    print(f"   Format: sentence<TAB>partition<TAB>label")
    print(f"   All sentences are cleaned (mojibake fixed, unicode normalized, lowercase)")
    
    # Write TSV file
    print(f"\n💾 Writing corpus.tsv...")
    with open(corpus_tsv_path, mode='w', newline='', encoding='utf-8') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')
        for row in tsv_data:
            tsv_writer.writerow(row)
    
    print(f"✅ Created: {corpus_tsv_path}")
    print(f"📊 File size: {corpus_tsv_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Validate OCTIS can load it
    print("\n🔍 Validating OCTIS dataset...")
    try:
        octis_dataset = Dataset()
        octis_dataset.load_custom_dataset_from_folder(str(octis_dataset_path))
        print("✅ OCTIS dataset loaded successfully")
        print(f"   Corpus size: {len(octis_dataset.get_corpus())}")
    except Exception as e:
        print(f"❌ Failed to load OCTIS dataset: {e}")
        raise
    
    # Verify consistency: Compare sample sentences from training data and OCTIS corpus
    print("\n🔍 Verifying consistency between training text and OCTIS corpus...")
    print("   Comparing first 5 sentences from both sources:")
    octis_corpus = octis_dataset.get_corpus()
    
    # Get first 5 sentences from training data (already cleaned)
    training_samples = df[sentence_col].head(5).tolist()
    
    # Get first 5 sentences from OCTIS corpus (should also be cleaned)
    octis_samples = []
    for i in range(min(5, len(octis_corpus))):
        # OCTIS corpus is list of lists (tokenized), join back to string
        if isinstance(octis_corpus[i], list):
            octis_samples.append(' '.join(octis_corpus[i]))
        else:
            octis_samples.append(str(octis_corpus[i]))
    
    matches = 0
    for i, (train_sent, octis_sent) in enumerate(zip(training_samples, octis_samples)):
        # Normalize for comparison (strip whitespace)
        train_normalized = train_sent.strip().lower()
        octis_normalized = octis_sent.strip().lower()
        
        if train_normalized == octis_normalized:
            matches += 1
            print(f"   ✓ Sentence {i+1}: Match")
        else:
            print(f"   ✗ Sentence {i+1}: MISMATCH")
            print(f"      Training: {train_sent[:80]}...")
            print(f"      OCTIS:    {octis_sent[:80]}...")
    
    print(f"\n   Consistency check: {matches}/{len(training_samples)} sentences match")
    if matches == len(training_samples):
        print("   ✅ All sample sentences match - consistency verified!")
    else:
        print(f"   ⚠️  Warning: {len(training_samples) - matches} mismatches found")
        print("   This may indicate preprocessing inconsistency")
    
    print(f"[STEP 2] ✓ OCTIS dataset creation completed successfully")
    return corpus_tsv_path


def load_or_create_embeddings(embedding_model_name: str, 
                               dataset_as_list_of_strings: List[str],
                               cache_dir: Path,
                               use_cache: bool = True,
                               processing_batch_size: int = 10000) -> np.ndarray:
    """
    Load cached embeddings or create new ones with batch processing and incremental saving.
    
    This function processes embeddings in batches to avoid memory issues and allows
    resuming from interruptions. Each batch is saved immediately after processing.
    
    Args:
        embedding_model_name: Name of the embedding model
        dataset_as_list_of_strings: List of document strings to encode
        cache_dir: Directory for caching embeddings
        use_cache: Whether to use cached embeddings if available
        processing_batch_size: Number of documents to process per batch (default: 10000)
    
    Returns:
        numpy.ndarray: Complete embeddings array
    """
    print_step(3, f"Loading/Creating embeddings: {embedding_model_name}")
    print(f"[EMBEDDINGS] ========== Embedding Processing with Batch Saving ==========")
    print(f"[EMBEDDINGS] Model: {embedding_model_name}")
    print(f"[EMBEDDINGS] Total documents: {len(dataset_as_list_of_strings):,}")
    print(f"[EMBEDDINGS] Processing batch size: {processing_batch_size:,} documents")
    print(f"[EMBEDDINGS] Cache directory: {cache_dir}")
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # File paths
    embedding_file = cache_dir / f"{embedding_model_name}_embeddings.npy"
    metadata_file = cache_dir / f"{embedding_model_name}_embeddings_metadata.json"
    batch_dir = cache_dir / f"{embedding_model_name}_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[EMBEDDINGS] Final embeddings file: {embedding_file}")
    print(f"[EMBEDDINGS] Metadata file: {metadata_file}")
    print(f"[EMBEDDINGS] Batch directory: {batch_dir}")
    
    # Step 1: Check for final concatenated embeddings file
    print(f"\n[EMBEDDINGS] Step 1: Checking for final concatenated embeddings file...")
    if use_cache and embedding_file.exists():
        print(f"[EMBEDDINGS] ✓ Found final embeddings file: {embedding_file}")
        print(f"[EMBEDDINGS]   File size: {embedding_file.stat().st_size / (1024**2):.2f} MB")
        print(f"[EMBEDDINGS]   Loading final embeddings...")
        
        try:
            embeddings = np.load(embedding_file)
            print(f"[EMBEDDINGS] ✓ Loaded embeddings shape: {embeddings.shape}")
            print(f"[EMBEDDINGS]   Dataset size: {len(dataset_as_list_of_strings):,} documents")
            
            # Validate that cached embeddings match dataset size
            if embeddings.shape[0] != len(dataset_as_list_of_strings):
                print(f"[EMBEDDINGS] ⚠️  Cached embeddings size ({embeddings.shape[0]:,}) doesn't match dataset size ({len(dataset_as_list_of_strings):,})")
                print(f"[EMBEDDINGS]   Regenerating embeddings for this dataset...")
                # Continue to create new ones
            else:
                print(f"[EMBEDDINGS] ✅ Cached embeddings match dataset size, using cache")
                print(f"[EMBEDDINGS] ========== Embedding loading completed successfully ==========")
                print(f"[STEP 3] ✓ Embeddings loading completed successfully")
                return embeddings
        except Exception as e:
            print(f"[EMBEDDINGS] ❌ Error loading final embeddings file: {e}")
            print(f"[EMBEDDINGS]   Will attempt to resume from batch files or regenerate...")
    
    # Step 2: Check for existing batch files and metadata
    print(f"\n[EMBEDDINGS] Step 2: Checking for existing batch files and progress...")
    metadata = None
    completed_batches = []
    
    if metadata_file.exists():
        print(f"[EMBEDDINGS] ✓ Found metadata file: {metadata_file}")
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"[EMBEDDINGS]   Metadata loaded:")
            print(f"[EMBEDDINGS]     Total documents: {metadata.get('total_documents', 'unknown'):,}")
            print(f"[EMBEDDINGS]     Processing batch size: {metadata.get('processing_batch_size', 'unknown'):,}")
            print(f"[EMBEDDINGS]     Embedding dimensions: {metadata.get('embedding_dim', 'unknown')}")
            print(f"[EMBEDDINGS]     Total batches: {metadata.get('total_batches', 'unknown')}")
            print(f"[EMBEDDINGS]     Completed batches: {metadata.get('completed_batches', 0)}")
            
            # Validate metadata matches current dataset
            if metadata.get('total_documents') != len(dataset_as_list_of_strings):
                print(f"[EMBEDDINGS] ⚠️  Metadata document count ({metadata.get('total_documents', 0):,}) doesn't match current dataset ({len(dataset_as_list_of_strings):,})")
                print(f"[EMBEDDINGS]   Will regenerate embeddings...")
                metadata = None
            else:
                # Check which batch files exist
                total_batches = metadata.get('total_batches', 0)
                for batch_idx in range(total_batches):
                    batch_file = batch_dir / f"batch_{batch_idx:05d}.npy"
                    if batch_file.exists():
                        completed_batches.append(batch_idx)
                        batch_size_mb = batch_file.stat().st_size / (1024**2)
                        print(f"[EMBEDDINGS]   ✓ Batch {batch_idx}: {batch_file.name} ({batch_size_mb:.2f} MB)")
                    else:
                        print(f"[EMBEDDINGS]   ✗ Batch {batch_idx}: Missing")
                
                print(f"[EMBEDDINGS]   Found {len(completed_batches)}/{total_batches} completed batches")
        except Exception as e:
            print(f"[EMBEDDINGS] ⚠️  Error loading metadata: {e}")
            print(f"[EMBEDDINGS]   Will start fresh...")
            metadata = None
    
    # Step 3: Calculate batch information
    print(f"\n[EMBEDDINGS] Step 3: Calculating batch information...")
    total_documents = len(dataset_as_list_of_strings)
    total_batches = (total_documents + processing_batch_size - 1) // processing_batch_size
    print(f"[EMBEDDINGS]   Total documents: {total_documents:,}")
    print(f"[EMBEDDINGS]   Processing batch size: {processing_batch_size:,}")
    print(f"[EMBEDDINGS]   Total batches needed: {total_batches}")
    
    # Determine if we need to process embeddings
    need_processing = True
    if metadata and len(completed_batches) == total_batches:
        print(f"[EMBEDDINGS] ✓ All batches completed! Concatenating into final file...")
        need_processing = False
    elif len(completed_batches) > 0:
        print(f"[EMBEDDINGS] ⚠️  Found {len(completed_batches)} completed batches, will resume from batch {max(completed_batches) + 1}")
        need_processing = True
    else:
        print(f"[EMBEDDINGS]   No existing batches found, starting fresh...")
        need_processing = True
    
    # Step 4: Load embedding model if needed
    if need_processing:
        print(f"\n[EMBEDDINGS] Step 4: Loading embedding model...")
        print(f"[EMBEDDINGS]   Model name: {embedding_model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[EMBEDDINGS]   Device: {device}")
        
        # Set up persistent cache directory
        cache_dir_hf = os.path.join(os.getcwd(), "cache", "huggingface")
        os.makedirs(cache_dir_hf, exist_ok=True)
        print(f"[EMBEDDINGS]   HuggingFace cache: {cache_dir_hf}")
        
        print(f"[EMBEDDINGS]   Loading model (this may download if not cached)...")
        embedding_model = load_embedding_model(embedding_model_name, device=device)
        print(f"[EMBEDDINGS] ✓ Embedding model loaded")
        
        # Get embedding dimension from model (encode a dummy to check)
        print(f"[EMBEDDINGS]   Determining embedding dimensions...")
        dummy_embedding = embedding_model.encode(["test"], convert_to_numpy=True)
        embedding_dim = dummy_embedding.shape[1]
        print(f"[EMBEDDINGS]   Embedding dimensions: {embedding_dim}")
        
        # GPU-specific optimizations for encoding
        if device == "cuda":
            encoding_batch_size = 128
            print(f"[EMBEDDINGS]   GPU encoding batch size: {encoding_batch_size}")
        else:
            encoding_batch_size = 32
            print(f"[EMBEDDINGS]   CPU encoding batch size: {encoding_batch_size}")
        
        # Initialize or update metadata
        if not metadata:
            metadata = {
                'embedding_model': embedding_model_name,
                'total_documents': total_documents,
                'processing_batch_size': processing_batch_size,
                'encoding_batch_size': encoding_batch_size,
                'embedding_dim': int(embedding_dim),
                'total_batches': total_batches,
                'completed_batches': 0,
                'device': device,
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
            print(f"[EMBEDDINGS]   Created new metadata")
        else:
            metadata['last_updated'] = datetime.now().isoformat()
            print(f"[EMBEDDINGS]   Updated existing metadata")
        
        # Step 5: Process batches incrementally
        print(f"\n[EMBEDDINGS] Step 5: Processing batches incrementally...")
        print(f"[EMBEDDINGS]   Starting from batch: {len(completed_batches)}")
        print(f"[EMBEDDINGS]   Remaining batches: {total_batches - len(completed_batches)}")
        
        import time
        overall_start_time = time.time()
        
        for batch_idx in range(len(completed_batches), total_batches):
            batch_start_idx = batch_idx * processing_batch_size
            batch_end_idx = min((batch_idx + 1) * processing_batch_size, total_documents)
            batch_documents = dataset_as_list_of_strings[batch_start_idx:batch_end_idx]
            
            batch_file = batch_dir / f"batch_{batch_idx:05d}.npy"
            
            print(f"\n[EMBEDDINGS]   {'='*70}")
            print(f"[EMBEDDINGS]   Processing batch {batch_idx + 1}/{total_batches}")
            print(f"[EMBEDDINGS]   Documents: {batch_start_idx:,} to {batch_end_idx:,} ({len(batch_documents):,} documents)")
            print(f"[EMBEDDINGS]   Batch file: {batch_file.name}")
            
            # Check if batch already exists (shouldn't happen, but safety check)
            if batch_file.exists():
                print(f"[EMBEDDINGS]   ⚠️  Batch file already exists, skipping...")
                continue
            
            # Encode batch
            print(f"[EMBEDDINGS]   Encoding batch...")
            batch_start_time = time.time()
            
            try:
                batch_embeddings = embedding_model.encode(
                    batch_documents,
                    show_progress_bar=True,
                    batch_size=encoding_batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    device=device
                )
                batch_elapsed = time.time() - batch_start_time
                
                print(f"[EMBEDDINGS]   ✓ Batch encoded in {batch_elapsed:.1f} seconds")
                print(f"[EMBEDDINGS]   Batch embeddings shape: {batch_embeddings.shape}")
                print(f"[EMBEDDINGS]   Throughput: {len(batch_documents)/batch_elapsed:.0f} sentences/second")
                
                # Save batch immediately
                print(f"[EMBEDDINGS]   Saving batch to disk...")
                np.save(batch_file, batch_embeddings)
                batch_size_mb = batch_file.stat().st_size / (1024**2)
                print(f"[EMBEDDINGS]   ✓ Batch saved: {batch_file.name} ({batch_size_mb:.2f} MB)")
                
                # Update metadata
                metadata['completed_batches'] = batch_idx + 1
                metadata['last_updated'] = datetime.now().isoformat()
                
                # Save metadata after each batch
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"[EMBEDDINGS]   ✓ Metadata updated: {batch_idx + 1}/{total_batches} batches completed")
                
                # Clean up batch from memory
                del batch_embeddings
                if device == "cuda":
                    torch.cuda.empty_cache()
                
                # Progress summary
                elapsed_total = time.time() - overall_start_time
                remaining_batches = total_batches - (batch_idx + 1)
                if batch_elapsed > 0:
                    estimated_remaining = (elapsed_total / (batch_idx + 1)) * remaining_batches
                    print(f"[EMBEDDINGS]   Progress: {batch_idx + 1}/{total_batches} batches ({100*(batch_idx+1)/total_batches:.1f}%)")
                    print(f"[EMBEDDINGS]   Elapsed: {elapsed_total/60:.1f} minutes")
                    print(f"[EMBEDDINGS]   Estimated remaining: {estimated_remaining/60:.1f} minutes")
                
            except Exception as e:
                print(f"[EMBEDDINGS]   ❌ Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                print(f"[EMBEDDINGS]   Batch processing failed, but previous batches are saved")
                print(f"[EMBEDDINGS]   You can resume by running again - it will continue from batch {batch_idx + 1}")
                raise
        
        overall_elapsed = time.time() - overall_start_time
        print(f"\n[EMBEDDINGS] ✓ All batches processed successfully!")
        print(f"[EMBEDDINGS]   Total time: {overall_elapsed/60:.1f} minutes")
        print(f"[EMBEDDINGS]   Average throughput: {total_documents/overall_elapsed:.0f} sentences/second")
    
    # Step 6: Concatenate all batches into final file
    print(f"\n[EMBEDDINGS] Step 6: Concatenating batches into final embeddings file...")
    print(f"[EMBEDDINGS]   Loading all batch files...")
    
    batch_files = sorted([batch_dir / f"batch_{i:05d}.npy" for i in range(total_batches)])
    print(f"[EMBEDDINGS]   Found {len(batch_files)} batch files to concatenate")
    
    batch_arrays = []
    for idx, batch_file in enumerate(batch_files):
        if not batch_file.exists():
            raise FileNotFoundError(f"Batch file missing: {batch_file}")
        print(f"[EMBEDDINGS]   Loading batch {idx + 1}/{len(batch_files)}: {batch_file.name}")
        batch_array = np.load(batch_file)
        batch_arrays.append(batch_array)
        print(f"[EMBEDDINGS]     Shape: {batch_array.shape}, Size: {batch_file.stat().st_size / (1024**2):.2f} MB")
    
    print(f"[EMBEDDINGS]   Concatenating {len(batch_arrays)} batches...")
    embeddings = np.concatenate(batch_arrays, axis=0)
    print(f"[EMBEDDINGS]   ✓ Concatenated embeddings shape: {embeddings.shape}")
    
    # Clean up batch arrays from memory
    del batch_arrays
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Save final concatenated file
    print(f"[EMBEDDINGS]   Saving final concatenated embeddings file...")
    np.save(embedding_file, embeddings)
    final_size_mb = embedding_file.stat().st_size / (1024**2)
    print(f"[EMBEDDINGS]   ✓ Final embeddings saved: {embedding_file.name} ({final_size_mb:.2f} MB)")
    
    # Update metadata
    metadata['final_file_created'] = datetime.now().isoformat()
    metadata['final_file_size_mb'] = final_size_mb
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[EMBEDDINGS]   ✓ Metadata updated with final file information")
    
    print(f"\n[EMBEDDINGS] ========== Embedding processing completed successfully ==========")
    print(f"[STEP 3] ✓ Embeddings processing completed successfully")
    return embeddings


class RetrainableBERTopicModel(BERTopicOctisModelWithEmbeddings):
    """Extended wrapper that stores the trained BERTopic model for saving."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trained_topic_model = None  # Store the trained model here
    
    def train_model(self, dataset, hyperparameters: Dict[str, Any] = None, top_words: int = 10) -> Dict[str, Any]:
        """Train model and store the BERTopic instance for saving."""
        print(f"[TRAIN] ========== Starting train_model() ==========")
        print(f"[TRAIN] Model: {self.embedding_model_name}")
        
        # Set hyperparameters
        if hyperparameters:
            print(f"[TRAIN] Setting hyperparameters from provided dict...")
            print(f"[TRAIN] Hyperparameters provided: {len(hyperparameters)} parameters")
            self.set_hyperparameters(hyperparameters)
        else:
            print(f"[TRAIN] Using default hyperparameters")
        
        if self.verbose:
            print(f"[TRAIN] Hyperparameters:")
            import pprint
            pprint.pprint(self.hyperparameters)
        
        # Check GPU memory before training
        print(f"[TRAIN] Checking GPU memory before training...")
        mem_before = print_gpu_memory_usage("Pre-training Memory", verbose=True)
        
        # Create representation models
        print(f"[TRAIN] Creating representation models...")
        representation_model = create_representation_models()
        print(f"[TRAIN] ✓ Representation models created")
        
        # Create GPU models using RAPIDS
        print(f"[TRAIN] Creating GPU UMAP model (RAPIDS)...")
        umap_model = UMAP(**self.hyperparameters['umap'])
        print(f"[TRAIN] ✓ UMAP model created")
        
        print(f"[TRAIN] Creating GPU HDBSCAN model (RAPIDS)...")
        hdbscan_model = HDBSCAN(**self.hyperparameters['hdbscan'])
        print(f"[TRAIN] ✓ HDBSCAN model created")
        
        print(f"[TRAIN] Creating CountVectorizer...")
        
        # Load custom stopwords
        custom_stoplist_path = Path("data/processed/custom_stoplist.txt")
        custom_stopwords = load_custom_stopwords(custom_stoplist_path)
        
        # Combine with English stopwords
        all_stopwords = list(ENGLISH_STOP_WORDS.union(custom_stopwords))
        print(f"[TRAIN] Total stopwords: {len(all_stopwords)}")
        
        # Update vectorizer params
        vectorizer_params = self.hyperparameters.get('vectorizer', {}).copy()
        vectorizer_params['stop_words'] = all_stopwords
        
        # Enforce stricter token pattern to avoid empty strings and single characters
        # Default is r'(?u)\b\w\w+\b' which means word chars of length >= 2
        # User reported empty words, so ensure pattern handles it
        if 'token_pattern' not in vectorizer_params:
            vectorizer_params['token_pattern'] = r'(?u)\b[a-zA-Z]{2,}\b'
            print(f"[TRAIN] Set default token_pattern: {vectorizer_params['token_pattern']}")
        
        vectorizer_model = CountVectorizer(**vectorizer_params)
        print(f"[TRAIN] ✓ CountVectorizer created with {len(all_stopwords)} stopwords")
        
        print(f"[TRAIN] Creating ClassTfidfTransformer...")
        tfdf_model = ClassTfidfTransformer(**self.hyperparameters['tfdf_vectorizer'])
        print(f"[TRAIN] ✓ ClassTfidfTransformer created")
        
        # Create BERTopic model instance
        print(f"[TRAIN] Creating BERTopic model instance...")
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            **self.hyperparameters['bertopic']
        )
        print(f"[TRAIN] ✓ BERTopic model instance created")
        
        # Convert embeddings to numpy if needed
        print(f"[TRAIN] Preparing embeddings...")
        if hasattr(self.embeddings, 'get'):
            print(f"[TRAIN] Converting CuPy array to NumPy array...")
            embeddings_numpy = self.embeddings.get()
        elif not isinstance(self.embeddings, np.ndarray):
            print(f"[TRAIN] Converting {type(self.embeddings)} to NumPy array...")
            embeddings_numpy = np.asarray(self.embeddings)
        else:
            embeddings_numpy = self.embeddings
        
        # Validate embeddings shape
        if len(embeddings_numpy.shape) != 2:
            raise ValueError(f"Embeddings must be 2D array, got shape: {embeddings_numpy.shape}")
        if embeddings_numpy.shape[0] != len(self.dataset_as_list_of_strings):
            raise ValueError(
                f"Embeddings first dimension ({embeddings_numpy.shape[0]}) must match "
                f"number of documents ({len(self.dataset_as_list_of_strings)})"
            )
        
        print(f"[TRAIN] Final embeddings shape: {embeddings_numpy.shape}, dtype: {embeddings_numpy.dtype}")
        print(f"[TRAIN] Note: RAPIDS cuML will automatically use GPU for UMAP and HDBSCAN clustering")
        
        # Train model
        print(f"[TRAIN] Starting fit_transform...")
        print(f"[TRAIN] Input documents: {len(self.dataset_as_list_of_strings):,}")
        print(f"[TRAIN] This may take several minutes...")
        
        with track_memory_peak(f"BERTopic Training: {self.embedding_model_name}"):
            topics, probabilities = topic_model.fit_transform(
                self.dataset_as_list_of_strings,
                embeddings=embeddings_numpy
            )
        
        print(f"[TRAIN] ✓ fit_transform completed")
        print(f"[TRAIN] Topics found: {len(set(topics)) - (1 if -1 in topics else 0)}")
        print(f"[TRAIN] Outliers: {sum(1 for t in topics if t == -1)}")
        
        # Enhanced GPU memory cleanup
        print(f"[TRAIN] Performing GPU memory cleanup...")
        cleanup_result = cleanup_gpu_memory(verbose=self.verbose)
        
        # Store the trained model for saving
        self.trained_topic_model = topic_model
        
        # Create output dict (for compatibility with parent class)
        output_dict = {}
        output_dict['topics'] = []
        
        # Extract topic words
        num_topics = len(set(topics)) - (1 if -1 in topics else 0)
        for topic in range(num_topics):
            words = list(zip(*topic_model.get_topic(topic)))[0]
            output_dict['topics'].append(list(words))
        
        # Filter empty topic lists
        output_dict['topics'] = [words for words in output_dict['topics'] if len(words) > 0]
        
        if not output_dict['topics']:
            output_dict['topics'] = None
        else:
            # Fix for OCTIS save_model_output: pad all topic word lists to same length
            # This prevents "inhomogeneous shape" error when saving with np.savez_compressed
            # OCTIS expects topics to be numpy-compatible (same shape for all arrays)
            max_topic_length = max(len(words) for words in output_dict['topics'])
            print(f"[TRAIN] Padding topic word lists to length {max_topic_length} for numpy compatibility")
            
            # Pad each topic's word list to max_topic_length with empty strings
            padded_topics = []
            for words in output_dict['topics']:
                padded_words = words + [''] * (max_topic_length - len(words))
                padded_topics.append(padded_words)
            
            # Convert to numpy array (numpy will infer string dtype automatically)
            # All lists are now the same length, so numpy can create a homogeneous 2D array
            try:
                output_dict['topics'] = np.array(padded_topics)
                print(f"[TRAIN] ✓ Topics converted to numpy array (shape: {output_dict['topics'].shape}, dtype: {output_dict['topics'].dtype})")
            except Exception as e:
                print(f"[TRAIN] ⚠️ Warning: Could not convert topics to numpy array: {e}")
                print(f"[TRAIN] Keeping topics as list (may cause save issues)")
                # Keep as list if conversion fails (fallback)
        
        output_dict['topic-word-matrix'] = np.array([])  # Placeholder
        output_dict['topic-document-matrix'] = np.array(probabilities)
        
        return output_dict


def retrain_single_model(model_config: Dict[str, Any], 
                        dataset_path: Path,
                        octis_dataset_path: Path,
                        output_dir: Path) -> bool:
    """
    Retrain a single model with given configuration.
    
    Args:
        model_config: Model configuration from pareto_loader
        dataset_path: Path to dataset CSV
        octis_dataset_path: Path to OCTIS dataset directory
        output_dir: Base output directory
        
    Returns:
        True if successful, False otherwise
    """
    embedding_model_name = model_config['embedding_model']
    pareto_rank = model_config['pareto_rank']
    hyperparameters = model_config['hyperparameters']
    
    print_section(f"Retraining Model {pareto_rank}: {embedding_model_name}", level=1)
    
    try:
        print(f"[RETRAIN] Model Configuration:")
        print(f"   Embedding Model: {embedding_model_name}")
        print(f"   Pareto Rank: {pareto_rank}")
        print(f"   Coherence: {model_config['coherence']:.4f}")
        print(f"   Topic Diversity: {model_config['topic_diversity']:.4f}")
        print(f"   Combined Score: {model_config['combined_score']:.4f}")
        print(f"   Hyperparameters: {len(hyperparameters)} parameters")
        
        # Load dataset
        print_step(1, f"Loading dataset from: {dataset_path}")
        df, dataset_as_list_of_strings, dataset_as_list_of_lists = load_and_validate_csv(dataset_path)
        
        # Create OCTIS dataset format
        print_step(2, f"Creating OCTIS dataset format")
        create_octis_dataset(dataset_path, octis_dataset_path, df)
        
        # Load or create embeddings
        cache_dir = octis_dataset_path / "embeddings_cache"
        embeddings = load_or_create_embeddings(
            embedding_model_name,
            dataset_as_list_of_strings,
            cache_dir,
            use_cache=True
        )
        
        # Load embedding model
        print_step(4, f"Loading embedding model: {embedding_model_name}")
        print(f"\n📥 Loading embedding model (will be cached locally)...")
        embedding_model = load_embedding_model(embedding_model_name)
        print(f"✅ Embedding model loaded and cached")
        
        # Create model wrapper
        print_step(5, f"Creating BERTopicOctisModelWithEmbeddings wrapper")
        print(f"\n🏭 Instantiating RetrainableBERTopicModel...")
        model = RetrainableBERTopicModel(
            embedding_model=embedding_model,
            embedding_model_name=embedding_model_name,
            embeddings=embeddings,
            dataset_as_list_of_strings=dataset_as_list_of_strings,
            dataset_as_list_of_lists=dataset_as_list_of_lists,
            verbose=True
        )
        print(f"✅ Model wrapper created successfully")
        
        # Create OCTIS dataset for train_model call
        print_step(6, f"Loading OCTIS dataset")
        print(f"[RETRAIN] Creating Dataset instance...")
        octis_dataset = Dataset()
        print(f"[RETRAIN] Loading custom dataset from folder...")
        print(f"[RETRAIN] Dataset folder: {octis_dataset_path}")
        octis_dataset.load_custom_dataset_from_folder(str(octis_dataset_path))
        print(f"[RETRAIN] ✓ OCTIS dataset loaded successfully")
        print(f"   Corpus size: {len(octis_dataset.get_corpus())}")
        
        # Train model
        print_step(7, f"Training BERTopic model with hyperparameters")
        output_dict = model.train_model(
            dataset=octis_dataset,
            hyperparameters=hyperparameters,
            top_words=10
        )
        
        if output_dict['topics'] is None:
            print(f"[RETRAIN] ❌ Training failed - no topics generated")
            return False
        
        num_topics = len(output_dict['topics']) if output_dict['topics'] else 0
        print(f"[RETRAIN] ✓ Training completed successfully")
        print(f"[RETRAIN]   Topics generated: {num_topics}")
        
        # Create output directory
        print_step(8, f"Saving trained model")
        model_output_dir = output_dir / embedding_model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[RETRAIN] Output directory: {model_output_dir}")
        
        # Save model as pickle
        pickle_path = model_output_dir / f"model_{pareto_rank}.pkl"
        print(f"[RETRAIN] Saving model as pickle: {pickle_path}")
        with open(pickle_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"[RETRAIN] ✓ Saved pickle model")
        print(f"[RETRAIN]   File size: {pickle_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Save BERTopic native format (safetensors - recommended)
        # This format avoids GPU array issues and is much smaller than pickle
        if model.trained_topic_model is not None:
            bertopic_dir = model_output_dir / f"model_{pareto_rank}"
            print(f"[RETRAIN] Saving BERTopic native format (safetensors): {bertopic_dir}")
            # Use safetensors format - safe, small, and avoids GPU array issues
            # Convert embedding model name to sentence-transformers format if needed
            embedding_model_path = embedding_model_name
            if not embedding_model_path.startswith("sentence-transformers/"):
                embedding_model_path = f"sentence-transformers/{embedding_model_name}"
            
            model.trained_topic_model.save(
                str(bertopic_dir),
                serialization="safetensors",
                save_embedding_model=embedding_model_path,
                save_ctfidf=True
            )
            print(f"[RETRAIN] ✓ Saved BERTopic native format (safetensors)")
        else:
            print(f"[RETRAIN] ⚠️  No trained topic model to save in native format")
        
        # Save metadata
        metadata = {
            'embedding_model': embedding_model_name,
            'pareto_rank': pareto_rank,
            'hyperparameters': hyperparameters,
            'coherence': model_config['coherence'],
            'topic_diversity': model_config['topic_diversity'],
            'combined_score': model_config['combined_score'],
            'iteration': model_config['iteration'],
            'num_topics': num_topics,
            'training_timestamp': datetime.now().isoformat(),
        }
        
        metadata_path = model_output_dir / f"model_{pareto_rank}_metadata.json"
        print(f"[RETRAIN] Saving metadata: {metadata_path}")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"[RETRAIN] ✓ Saved metadata")
        
        print_section(f"✅ Successfully Retrained Model {pareto_rank}", level=1)
        return True
        
    except Exception as e:
        print(f"[RETRAIN] ❌ Error retraining model {pareto_rank}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup GPU memory
        print(f"[RETRAIN] Performing final GPU memory cleanup...")
        cleanup_gpu_memory(verbose=True)

