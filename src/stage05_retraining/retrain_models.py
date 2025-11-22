"""Core retraining logic for Pareto-efficient models."""

import os
import json
import pickle
import csv
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch


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
    print("   - Removing newlines")
    print("   - Normalizing whitespace")
    print("   - Converting to lowercase")
    
    # Clean sentences
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
    
    print(f"[STEP 1] ✓ CSV loading completed successfully")
    return df, dataset_as_list_of_strings, non_empty


def create_octis_dataset(csv_path: Path, octis_dataset_path: Path, 
                        df: pd.DataFrame) -> Path:
    """Create OCTIS dataset format (corpus.tsv)."""
    print_step(2, f"Creating OCTIS dataset format")
    
    # Create directory if needed
    octis_dataset_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 OCTIS dataset directory: {octis_dataset_path}")
    
    corpus_tsv_path = octis_dataset_path / 'corpus.tsv'
    print(f"📄 Output file: {corpus_tsv_path}")
    
    # Read original CSV to get raw sentences
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
    
    print(f"[STEP 2] ✓ OCTIS dataset creation completed successfully")
    return corpus_tsv_path


def load_or_create_embeddings(embedding_model_name: str, 
                               dataset_as_list_of_strings: List[str],
                               cache_dir: Path,
                               use_cache: bool = True) -> np.ndarray:
    """Load cached embeddings or create new ones."""
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
            print(f"[STEP 3] ✓ Embeddings loading completed successfully")
            return embeddings
    
    # Create embeddings
    print(f"🤖 Loading embedding model: {embedding_model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    # Set up persistent cache directory
    cache_dir_hf = os.path.join(os.getcwd(), "cache", "huggingface")
    os.makedirs(cache_dir_hf, exist_ok=True)
    print(f"💾 Using persistent cache: {cache_dir_hf}")
    
    embedding_model = load_embedding_model(embedding_model_name, device=device)
    
    # GPU-specific optimizations
    if device == "cuda":
        batch_size = 128
        print(f"⚡ Using GPU batch size: {batch_size}")
    else:
        batch_size = 32
        print(f"🐌 Using CPU batch size: {batch_size}")
    
    print(f"\n📊 Encoding {len(dataset_as_list_of_strings):,} documents...")
    print(f"   This may take several minutes...")
    
    import time
    start_time = time.time()
    embeddings = embedding_model.encode(
        dataset_as_list_of_strings,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        device=device
    )
    elapsed_time = time.time() - start_time
    print(f"⚡ Encoding completed in {elapsed_time/60:.1f} minutes")
    print(f"   Throughput: {len(dataset_as_list_of_strings)/elapsed_time:.0f} sentences/second")
    print(f"✅ Encoding completed, shape: {embeddings.shape}")
    
    # Save to cache
    if use_cache:
        print(f"\n💾 Saving embeddings to cache...")
        np.save(embedding_file, embeddings)
        print(f"✅ Cached embeddings saved")
    
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
        vectorizer_model = CountVectorizer(**self.hyperparameters['vectorizer'])
        print(f"[TRAIN] ✓ CountVectorizer created")
        
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
        
        # Save BERTopic native format
        if model.trained_topic_model is not None:
            bertopic_dir = model_output_dir / f"model_{pareto_rank}"
            print(f"[RETRAIN] Saving BERTopic native format: {bertopic_dir}")
            model.trained_topic_model.save(str(bertopic_dir))
            print(f"[RETRAIN] ✓ Saved BERTopic native format")
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

