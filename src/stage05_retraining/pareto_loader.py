"""Load and parse Pareto-efficient models from CSV."""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


def load_top_models(pareto_csv_path: Path, top_n: int = 4) -> List[Dict[str, Any]]:
    """
    Load top N Pareto-efficient models from CSV.
    
    Args:
        pareto_csv_path: Path to pareto.csv file
        top_n: Number of top models to select (default: 4)
        
    Returns:
        List of model configs, each containing:
        - embedding_model: Name of embedding model
        - pareto_rank: Pareto rank
        - hyperparameters: Dict of hyperparameters in format expected by set_hyperparameters()
        - coherence: Coherence score
        - topic_diversity: Topic diversity score
        - combined_score: Combined score
    """
    print(f"[PARETO_LOADER] ========== Loading Pareto CSV ==========")
    print(f"[PARETO_LOADER] CSV path: {pareto_csv_path}")
    print(f"[PARETO_LOADER] Top N models: {top_n}")
    
    if not pareto_csv_path.exists():
        raise FileNotFoundError(f"Pareto CSV not found: {pareto_csv_path}")
    
    print(f"📁 File exists: {pareto_csv_path}")
    print(f"📊 File size: {pareto_csv_path.stat().st_size / 1024:.2f} KB")
    
    # Read CSV
    print(f"\n📖 Reading CSV file...")
    df = pd.read_csv(pareto_csv_path)
    print(f"✅ Loaded {len(df)} rows from CSV")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Validate required columns
    print(f"\n🔍 Validating required columns...")
    required_cols = [
        'Embeddings_Model', 'pareto_rank',
        'bertopic__min_topic_size', 'bertopic__top_n_words',
        'hdbscan__min_cluster_size', 'hdbscan__min_samples',
        'umap__min_dist', 'umap__n_components', 'umap__n_neighbors',
        'vectorizer__min_df'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    print(f"✅ All required columns present")
    
    # Select top N by pareto_rank (handle duplicates by keeping all)
    print(f"\n🔍 Selecting top {top_n} models by pareto_rank...")
    df_sorted = df.sort_values('pareto_rank').head(top_n)
    print(f"✅ Selected {len(df_sorted)} models")
    
    # Extract model configs
    print(f"\n📝 Extracting model configurations...")
    model_configs = []
    for idx, row in df_sorted.iterrows():
        # Extract hyperparameters (already in correct format with double underscores)
        hyperparameters = {
            'bertopic__min_topic_size': int(row['bertopic__min_topic_size']),
            'bertopic__top_n_words': int(row['bertopic__top_n_words']),
            'hdbscan__min_cluster_size': int(row['hdbscan__min_cluster_size']),
            'hdbscan__min_samples': int(row['hdbscan__min_samples']),
            'umap__min_dist': float(row['umap__min_dist']),
            'umap__n_components': int(row['umap__n_components']),
            'umap__n_neighbors': int(row['umap__n_neighbors']),
            'vectorizer__min_df': float(row['vectorizer__min_df']),
        }
        
        config = {
            'embedding_model': row['Embeddings_Model'],
            'pareto_rank': int(row['pareto_rank']),
            'hyperparameters': hyperparameters,
            'coherence': float(row.get('Coherence', 0.0)),
            'topic_diversity': float(row.get('Topic_Diversity', 0.0)),
            'combined_score': float(row.get('Combined_Score', 0.0)),
            'iteration': int(row.get('Iteration', 0)),
        }
        
        model_configs.append(config)
        print(f"   [{config['pareto_rank']}] {config['embedding_model']}")
        print(f"      Coherence: {config['coherence']:.4f}, Diversity: {config['topic_diversity']:.4f}")
        print(f"      Combined Score: {config['combined_score']:.4f}")
    
    print(f"\n[PARETO_LOADER] ✓ Successfully loaded {len(model_configs)} model configurations")
    print(f"[PARETO_LOADER] ========== Pareto CSV loading completed ==========")
    return model_configs

