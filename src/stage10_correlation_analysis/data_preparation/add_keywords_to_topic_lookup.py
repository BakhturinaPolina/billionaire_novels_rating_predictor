#!/usr/bin/env python3
"""Add keywords to topic_lookup.parquet from BERTopic model.

This script extracts keywords from the BERTopic model's topic_representations_
and adds them to the existing topic_lookup.parquet file.
"""

from pathlib import Path
import sys
import pandas as pd

try:
    from bertopic import BERTopic
except ImportError:
    print("ERROR: BERTopic not available. Please install: pip install bertopic")
    sys.exit(1)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add utils to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import first_existing, safe_project_root

# Try to import defaults
try:
    from src.stage06_topic_exploration.explore_retrained_model import (
        DEFAULT_BASE_DIR,
        DEFAULT_EMBEDDING_MODEL,
    )
except ImportError:
    DEFAULT_BASE_DIR = PROJECT_ROOT / "models" / "retrained"
    DEFAULT_EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"


def find_model_path(base_dir: Path, embedding_model: str, project_root: Path) -> Path:
    """Find the BERTopic model path."""
    # Look for model with llm labels (final model)
    candidates = [
        project_root / base_dir / embedding_model / "bertopic_model_with_llm_labels",
        project_root / base_dir / embedding_model / "bertopic_model_with_llm_labels.pkl",
        project_root / base_dir / embedding_model / "bertopic_model",
        project_root / base_dir / embedding_model / "bertopic_model.pkl",
        base_dir / embedding_model / "bertopic_model_with_llm_labels",
        base_dir / embedding_model / "bertopic_model_with_llm_labels.pkl",
        base_dir / embedding_model / "bertopic_model",
        base_dir / embedding_model / "bertopic_model.pkl",
    ]
    
    model_path = first_existing(candidates)
    if model_path is None:
        # Try searching recursively
        print(f"\nSearching for model files in {base_dir}...")
        if base_dir.exists():
            for pkl_file in base_dir.rglob("*.pkl"):
                if "bertopic" in pkl_file.name.lower() or "model" in pkl_file.name.lower():
                    candidates.append(pkl_file)
            for model_dir in base_dir.rglob("*bertopic*"):
                if model_dir.is_dir():
                    candidates.append(model_dir)
        
        model_path = first_existing(candidates)
    
    if model_path is None:
        raise FileNotFoundError(
            f"Could not find BERTopic model. Tried:\n" + "\n".join([f"  - {c}" for c in candidates[:10]])
        )
    
    return model_path


def extract_keywords_from_model(model_path: Path) -> dict[int, str]:
    """Extract keywords from BERTopic model.
    
    Returns:
        Dictionary mapping topic_id to comma-separated keywords string
    """
    print(f"Loading BERTopic model from: {model_path}")
    
    # Load model
    if model_path.is_dir():
        model = BERTopic.load(str(model_path))
    elif model_path.suffix == ".pkl":
        import pickle
        with open(model_path, "rb") as f:
            loaded = pickle.load(f)
        if hasattr(loaded, "trained_topic_model"):
            model = loaded.trained_topic_model
        elif isinstance(loaded, BERTopic):
            model = loaded
        else:
            model = BERTopic.load(str(model_path))
    else:
        model = BERTopic.load(str(model_path))
    
    print(f"✓ Model loaded")
    
    # Extract keywords
    keywords_dict = {}
    
    if not hasattr(model, "topic_representations_"):
        print("⚠️  Model does not have topic_representations_ attribute")
        return keywords_dict
    
    topic_ids = [tid for tid in model.topic_representations_.keys() if tid != -1]
    print(f"Extracting keywords for {len(topic_ids)} topics...")
    
    for topic_id in sorted(topic_ids):
        if topic_id in model.topic_representations_:
            kws = model.topic_representations_[topic_id]
            # Format: list of (word, score) tuples
            keyword_list = [kw[0] for kw in kws[:10]]  # Top 10 keywords
            keywords_dict[topic_id] = ", ".join(keyword_list)
        else:
            keywords_dict[topic_id] = None
    
    print(f"✓ Extracted keywords for {len([k for k, v in keywords_dict.items() if v])} topics")
    return keywords_dict


def add_keywords_to_topic_lookup(
    topic_lookup_path: Path,
    keywords_dict: dict[int, str],
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Add keywords to topic_lookup DataFrame.
    
    Args:
        topic_lookup_path: Path to existing topic_lookup.parquet
        keywords_dict: Dictionary mapping topic_id to keywords string
        output_path: Optional output path (default: overwrite input)
        
    Returns:
        Updated DataFrame
    """
    print(f"\nLoading topic_lookup from: {topic_lookup_path}")
    df = pd.read_parquet(topic_lookup_path)
    
    print(f"  Current shape: {df.shape}")
    print(f"  Current columns: {list(df.columns)}")
    
    # Add keywords column
    df["keywords"] = df["topic_id"].map(keywords_dict)
    
    # Report coverage
    n_with_keywords = df["keywords"].notna().sum()
    print(f"\n✓ Added keywords column")
    print(f"  Topics with keywords: {n_with_keywords} / {len(df)} ({n_with_keywords/len(df)*100:.1f}%)")
    
    # Save
    if output_path is None:
        output_path = topic_lookup_path
    
    df.to_parquet(output_path, index=False)
    print(f"\n✓ Saved updated topic_lookup to: {output_path}")
    
    return df


def main():
    """Main entry point."""
    project_root = safe_project_root()
    
    # Paths
    base_dir = Path(DEFAULT_BASE_DIR) if DEFAULT_BASE_DIR else project_root / "models" / "retrained"
    topic_lookup_path = (
        project_root / "results" / "stage10_correlation_analysis" / 
        "data_preparation" / "taxonomy_radway_eda" / "topic_lookup.parquet"
    )
    
    if not topic_lookup_path.exists():
        print(f"ERROR: topic_lookup.parquet not found at: {topic_lookup_path}")
        sys.exit(1)
    
    # Find model
    embedding_model = DEFAULT_EMBEDDING_MODEL or "paraphrase-MiniLM-L6-v2"
    model_path = find_model_path(base_dir, embedding_model, project_root)
    
    # Extract keywords
    keywords_dict = extract_keywords_from_model(model_path)
    
    if not keywords_dict:
        print("ERROR: No keywords extracted from model")
        sys.exit(1)
    
    # Add to topic_lookup
    add_keywords_to_topic_lookup(topic_lookup_path, keywords_dict)
    
    print("\n" + "=" * 80)
    print("✓ Keywords successfully added to topic_lookup.parquet")
    print("=" * 80)


if __name__ == "__main__":
    main()

