#!/usr/bin/env python3
"""
Test script for OCTIS+BERTopic pipeline validation.

This script tests the pipeline WITHOUT running optimization:
- Loads and validates data
- Creates OCTIS dataset format
- Generates embeddings
- Tests a single BERTopic training run
- Validates outputs

Usage:
    python -m src.stage03_modeling.test_octis_pipeline --subset  # Test with 10K subset
    python -m src.stage03_modeling.test_octis_pipeline --full    # Test with full dataset
"""

# Enable fast Hugging Face downloads (must be set before importing sentence_transformers)
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Set persistent cache directories for Hugging Face models
# This prevents re-downloading models on every run
cache_dir = os.path.join(os.getcwd(), "cache", "huggingface")
os.makedirs(cache_dir, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir

import argparse
import csv
import json
import os
import pickle
import re
import sys
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Set TOKENIZERS_PARALLELISM to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import GPU utilities (always uses RAPIDS)
from src.common.gpu_models import get_gpu_umap_hdbscan, print_gpu_status, check_rapids_availability

# Check RAPIDS availability (required)
print_gpu_status()
if not check_rapids_availability():
    print("\n❌ ERROR: RAPIDS (cuML) is required but not available!")
    print("   Install with: pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com")
    sys.exit(1)

# Import RAPIDS models (required)
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
print("✅ Using RAPIDS UMAP/HDBSCAN (GPU) - Required")

# Import libraries
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
    from bertopic.vectorizers import ClassTfidfTransformer
    from octis.dataset.dataset import Dataset
    from octis.models.model import AbstractModel
    import gensim.corpora as corpora

    # Import BERTopicOctisModelWithEmbeddings from stage03_modeling
    from src.stage03_modeling.bertopic_octis_model import (
        BERTopicOctisModelWithEmbeddings,
        load_embedding_model
    )
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("Please install: pip install bertopic octis sentence-transformers")
    sys.exit(1)

# Load config
try:
    from src.common.config import load_config, resolve_path
    paths_cfg = load_config(Path("configs/paths.yaml"))
    octis_cfg = load_config(Path("configs/octis.yaml"))
except Exception as e:
    print(f"⚠️  Could not load config: {e}")
    print("Using default paths")
    paths_cfg = {}
    octis_cfg = {}


# Helper function for real-time printing
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
    
    # Read CSV with latin1 encoding (as in legacy code)
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
    print("   - Removing newlines")
    print("   - Normalizing whitespace")
    print("   - Converting to lowercase")
    
    # Clean sentences (as in legacy code)
    df['Sentence'] = df['Sentence'].apply(
        lambda x: re.sub(r'\n+', ' ', str(x)) if isinstance(x, str) else str(x)
    )
    df['Sentence'] = df['Sentence'].apply(
        lambda x: re.sub(r'\s+', ' ', x).strip().lower()
    )
    
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
    
    return df, dataset_as_list_of_strings, non_empty


def create_octis_dataset(csv_path: Path, octis_dataset_path: Path, 
                        df: pd.DataFrame) -> Path:
    """
    Create OCTIS dataset format (corpus.tsv).
    
    Returns:
        Path to created corpus.tsv file
    """
    print_step(2, f"Creating OCTIS dataset format")
    
    # Create directory if needed
    octis_dataset_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 OCTIS dataset directory: {octis_dataset_path}")
    
    corpus_tsv_path = octis_dataset_path / 'corpus.tsv'
    print(f"📄 Output file: {corpus_tsv_path}")
    
    # Read original CSV to get raw sentences (before cleaning)
    print("\n📖 Reading original CSV for OCTIS format...")
    tsv_data = []
    
    with open(csv_path, mode='r', newline='', encoding='latin1') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # Skip header
        
        for row in csv_reader:
            if len(row) == 4:
                author, book_title, chapter, sentence = row
                partition = 'train'
                label = f"{author},{book_title}"
                tsv_data.append([sentence, partition, label])
    
    print(f"✅ Prepared {len(tsv_data)} rows for OCTIS format")
    print(f"   Format: sentence<TAB>partition<TAB>label")
    
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
    
    return corpus_tsv_path


def load_or_create_embeddings(embedding_model_name: str, 
                               dataset_as_list_of_strings: List[str],
                               cache_dir: Path,
                               use_cache: bool = True) -> np.ndarray:
    """
    Load cached embeddings or create new ones.
    
    Returns:
        Embeddings array
    """
    print_step(3, f"Loading/Creating embeddings: {embedding_model_name}")
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    embedding_file = cache_dir / f"{embedding_model_name}_embeddings.npy"
    
    # Check for cached embeddings
    if use_cache and embedding_file.exists():
        print(f"📁 Found cached embeddings: {embedding_file}")
        print(f"   Loading from cache...")
        embeddings = np.load(embedding_file)
        print(f"✅ Loaded embeddings shape: {embeddings.shape}")
        print(f"   Dataset size: {len(dataset_as_list_of_strings):,} documents")
        # Validate that cached embeddings match dataset size
        if embeddings.shape[0] != len(dataset_as_list_of_strings):
            print(f"⚠️  Cached embeddings size ({embeddings.shape[0]:,}) doesn't match dataset size ({len(dataset_as_list_of_strings):,})")
            print(f"   Regenerating embeddings for this dataset...")
            # Don't return cached embeddings, continue to create new ones
        else:
            print(f"✅ Cached embeddings match dataset size, using cache")
            return embeddings
    
    # Create embeddings
    print(f"🤖 Loading embedding model: {embedding_model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    # Set up persistent cache directory
    cache_dir = os.path.join(os.getcwd(), "cache", "huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"💾 Using persistent cache: {cache_dir}")
    
    # GPU-specific optimizations
    if device == "cuda":
        print(f"🚀 GPU optimizations enabled")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU Memory: {gpu_memory_gb:.2f} GB")
        
        # Optimize batch size based on GPU memory
        # RTX 2070 (8GB): can handle 512-1024 batch size
        # Larger batch = faster processing (up to GPU memory limit)
        if gpu_memory_gb >= 8:
            batch_size = 512  # Aggressive batch size for 8GB+ GPUs
        elif gpu_memory_gb >= 6:
            batch_size = 384
        else:
            batch_size = 256
        
        print(f"   Optimized batch size: {batch_size} (for {gpu_memory_gb:.1f}GB GPU)")
    else:
        print(f"⚠️  CPU mode - slower performance")
        batch_size = 32
    
    embedding_model = SentenceTransformer(embedding_model_name, device=device, cache_folder=cache_dir)
    
    # Enable FP16/mixed precision for 2x speedup on modern GPUs
    if device == "cuda" and hasattr(embedding_model, 'half'):
        try:
            embedding_model = embedding_model.half()  # Convert to FP16
            print(f"   ✅ FP16 (half precision) enabled - 2x faster")
        except Exception as e:
            print(f"   ⚠️  Could not enable FP16: {e}")
    
    print(f"\n🔄 Encoding {len(dataset_as_list_of_strings):,} sentences...")
    print(f"   Batch size: {batch_size} ({'GPU optimized' if device == 'cuda' else 'CPU'})")
    estimated_time = len(dataset_as_list_of_strings) / (batch_size * 10)  # Rough estimate: 10 batches/sec
    print(f"   Estimated time: ~{estimated_time/60:.1f} minutes")
    print("   Processing...")
    
    # Optimize encode parameters for maximum speed
    encode_kwargs = {
        'show_progress_bar': True,
        'batch_size': batch_size,
        'convert_to_numpy': True,
        'normalize_embeddings': False,  # Faster, normalize later if needed
        'device': device,  # Explicitly set device
    }
    
    # Verify GPU is being used
    if device == "cuda":
        print(f"   Verifying GPU usage...")
        initial_memory = torch.cuda.memory_allocated(0) / 1e9
        print(f"   GPU memory allocated: {initial_memory:.2f} GB")
        torch.cuda.empty_cache()  # Clear cache before encoding
    
    import time
    start_time = time.time()
    embeddings = embedding_model.encode(
        dataset_as_list_of_strings, 
        **encode_kwargs
    )
    elapsed_time = time.time() - start_time
    print(f"   ⚡ Encoding completed in {elapsed_time/60:.1f} minutes ({elapsed_time:.0f} seconds)")
    print(f"   Throughput: {len(dataset_as_list_of_strings)/elapsed_time:.0f} sentences/second")
    
    # Check GPU memory after encoding
    if device == "cuda":
        final_memory = torch.cuda.memory_allocated(0) / 1e9
        print(f"   GPU memory after encoding: {final_memory:.2f} GB")
        print(f"   GPU memory used: {final_memory - initial_memory:.2f} GB")
    
    print(f"✅ Created embeddings shape: {embeddings.shape}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    
    # Save to cache
    if use_cache:
        print(f"\n💾 Saving embeddings to cache: {embedding_file}")
        np.save(embedding_file, embeddings)
        print(f"✅ Cached embeddings saved")
    
    return embeddings


def create_bertopic_model(embedding_model_name: str, embeddings: np.ndarray,
                         dataset_as_list_of_strings: List[str],
                         dataset_as_list_of_lists: List[List[str]]):
    """
    Create BERTopicOctisModelWithEmbeddings instance using shared module.
    
    This function ensures the embedding model is downloaded and cached locally
    before creating the BERTopic model wrapper.
    """
    print_step(4, "Creating BERTopicOctisModelWithEmbeddings")
    
    # Load embedding model (ensures it's downloaded and cached locally)
    print("\n📥 Loading embedding model (will be cached locally)...")
    embedding_model = load_embedding_model(embedding_model_name)
    print("✅ Embedding model loaded and cached")
    
    # Create model instance using shared module
    print("\n🏭 Instantiating BERTopicOctisModelWithEmbeddings...")
    model = BERTopicOctisModelWithEmbeddings(
        embedding_model=embedding_model,
        embedding_model_name=embedding_model_name,
        embeddings=embeddings,
        dataset_as_list_of_strings=dataset_as_list_of_strings,
        dataset_as_list_of_lists=dataset_as_list_of_lists,
        verbose=True
    )
    
    print("✅ Model created successfully")
    return model


def test_single_training_run(model, octis_dataset, dataset_as_list_of_lists):
    """
    Test a single training run (not optimization).
    """
    print_step(5, "Testing single training run")
    
    print("\n🧪 Running single training run with default hyperparameters...")
    print("   (This validates the pipeline works without optimization)")
    
    output_dict = model.train_model(
        dataset=octis_dataset,
        hyperparameters={},  # Use defaults
        top_words=10
    )
    
    if output_dict['topics'] is None:
        print("\n❌ Training failed - no topics generated")
        return False
    
    print("\n✅ Training run successful!")
    print(f"   Topics generated: {len(output_dict['topics'])}")
    
    return True


def main():
    print("[MAIN] ========== Starting test_octis_pipeline main() ==========")
    print("[MAIN] Parsing command line arguments...")
    parser = argparse.ArgumentParser(description="Test OCTIS+BERTopic pipeline")
    parser.add_argument('--subset', action='store_true', 
                       help='Test with chapters_subset_10000.csv')
    parser.add_argument('--full', action='store_true',
                       help='Test with full chapters.csv')
    parser.add_argument('--embedding-model', default='paraphrase-MiniLM-L6-v2',
                       help='Embedding model to use (default: paraphrase-MiniLM-L6-v2 for sanity check)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable embedding cache')
    
    args = parser.parse_args()
    print("[MAIN] ✓ Arguments parsed")
    print(f"[MAIN] Arguments:")
    print(f"[MAIN]   subset: {args.subset}")
    print(f"[MAIN]   full: {args.full}")
    print(f"[MAIN]   embedding-model: {args.embedding_model}")
    print(f"[MAIN]   no-cache: {args.no_cache}")
    
    if not args.subset and not args.full:
        print("❌ Please specify --subset or --full")
        parser.print_help()
        sys.exit(1)
    
    print_section("OCTIS+BERTopic Pipeline Test", level=1)
    print(f"📋 Test mode: {'SUBSET (10K)' if args.subset else 'FULL DATASET'}")
    print(f"🤖 Embedding model: {args.embedding_model}")
    print(f"💾 Cache embeddings: {not args.no_cache}")
    
    try:
        # Determine paths
        if args.subset:
            csv_path = Path(paths_cfg.get("inputs", {}).get("chapters_subset_csv", 
                        "data/processed/chapters_subset_10000.csv"))
        else:
            csv_path = Path(paths_cfg.get("inputs", {}).get("chapters_csv",
                        "data/processed/chapters.csv"))
        
        octis_dataset_path = Path(paths_cfg.get("inputs", {}).get("octis_dataset",
                                  "data/interim/octis"))
        
        # Resolve paths (relative to project root)
        project_root = Path(__file__).parent.parent.parent
        if not csv_path.is_absolute():
            csv_path = project_root / csv_path
        if not octis_dataset_path.is_absolute():
            octis_dataset_path = project_root / octis_dataset_path
        
        # Step 1: Load and validate CSV
        df, dataset_as_list_of_strings, dataset_as_list_of_lists = load_and_validate_csv(csv_path)
        
        # Step 2: Create OCTIS dataset
        corpus_tsv_path = create_octis_dataset(csv_path, octis_dataset_path, df)
        
        # Load OCTIS dataset
        octis_dataset = Dataset()
        octis_dataset.load_custom_dataset_from_folder(str(octis_dataset_path))
        
        # Step 3: Load or create embeddings
        cache_dir = octis_dataset_path / "embeddings_cache"
        embeddings = load_or_create_embeddings(
            args.embedding_model,
            dataset_as_list_of_strings,
            cache_dir,
            use_cache=not args.no_cache
        )
        
        # Step 4: Create BERTopic model
        model = create_bertopic_model(
            args.embedding_model,
            embeddings,
            dataset_as_list_of_strings,
            dataset_as_list_of_lists
        )
        
        # Step 5: Test single training run
        success = test_single_training_run(model, octis_dataset, dataset_as_list_of_lists)
        
        if success:
            print_section("✅ ALL TESTS PASSED", level=1)
            print("The pipeline is working correctly!")
            print("\nNext steps:")
            print("  - Run with --full to test on complete dataset")
            print("  - Use bertopic_runner.py for full optimization")
        else:
            print_section("❌ TESTS FAILED", level=1)
            sys.exit(1)
            
    except Exception as e:
        print_section("❌ ERROR", level=1)
        print(f"Error: {e}")
        print("\nTraceback:")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

