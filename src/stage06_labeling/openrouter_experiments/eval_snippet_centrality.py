"""Evaluation script to visualize snippet centrality vs rank.

This script helps sanity-check that centrality reranking is working correctly
by plotting cosine similarity to centroid vs snippet rank.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

from .generate_labels_openrouter import (
    extract_representative_docs_per_topic,
)


def compute_centrality_scores(docs, model_name="paraphrase-MiniLM-L6-v2"):
    """Compute cosine similarity of each doc embedding to the centroid.
    
    Args:
        docs: List of document strings
        model_name: SentenceTransformer model name
        
    Returns:
        Array of cosine similarity scores (one per doc)
    """
    if not docs:
        return np.array([])
    
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, normalize_embeddings=True)
    
    # Compute centroid
    centroid = embeddings.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0.0:
        return np.zeros(len(docs))
    
    # Normalize centroid
    centroid = centroid / norm
    
    # Compute cosine similarities (embeddings are already normalized)
    sims = embeddings @ centroid  # cosine similarity (normalized)
    
    return sims


def plot_centrality(docs, topic_id, top_k=40, embedding_model="paraphrase-MiniLM-L6-v2"):
    """Plot centrality scores vs snippet rank.
    
    Args:
        docs: List of document strings
        topic_id: Topic ID for title
        top_k: Maximum number of docs to plot
        embedding_model: SentenceTransformer model name for computing embeddings
    """
    docs = docs[:top_k]
    scores = compute_centrality_scores(docs, model_name=embedding_model)
    
    if scores.size == 0:
        print(f"No docs for topic {topic_id}")
        return
    
    # Sort indices by decreasing centrality
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    
    plt.figure()
    plt.plot(range(1, len(sorted_scores) + 1), sorted_scores, marker="o")
    plt.xlabel("Snippet rank (1 = most central)")
    plt.ylabel("Cosine similarity to centroid")
    plt.title(f"Centrality vs Snippet Rank for Topic {topic_id}")
    plt.grid(True)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize snippet centrality scores vs rank for a topic"
    )
    parser.add_argument(
        "--bertopic-model-path",
        type=str,
        required=True,
        help="Path to saved BERTopic model",
    )
    parser.add_argument(
        "--topic-id",
        type=int,
        required=True,
        help="Topic ID to inspect",
    )
    parser.add_argument(
        "--max-docs-per-topic",
        type=int,
        default=40,
        help="Maximum number of docs to analyze per topic (default: 40)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="paraphrase-MiniLM-L6-v2",
        help="SentenceTransformer model for computing embeddings (default: paraphrase-MiniLM-L6-v2)",
    )
    
    args = parser.parse_args()
    
    # Load BERTopic model
    print(f"Loading BERTopic model from: {args.bertopic_model_path}")
    topic_model: BERTopic = BERTopic.load(args.bertopic_model_path)
    print("✓ Model loaded")
    
    # Extract representative docs
    print(f"Extracting representative docs for topic {args.topic_id}...")
    topic_to_docs = extract_representative_docs_per_topic(
        topic_model,
        max_docs_per_topic=args.max_docs_per_topic,
    )
    
    docs = topic_to_docs.get(args.topic_id, [])
    print(f"Topic {args.topic_id}: {len(docs)} representative docs loaded.")
    
    if not docs:
        print(f"Warning: No representative docs found for topic {args.topic_id}")
        return
    
    # Plot centrality
    print(f"Computing centrality scores using {args.embedding_model}...")
    plot_centrality(
        docs,
        args.topic_id,
        top_k=args.max_docs_per_topic,
        embedding_model=args.embedding_model,
    )


if __name__ == "__main__":
    main()

