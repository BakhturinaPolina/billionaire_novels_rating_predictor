"""Test script to verify hybrid approach performance and label quality.

This script:
1. Tests performance improvement (should be 10x faster: 15-30s vs 3-4 minutes)
2. Validates label quality with 5 snippets (should still be good)
"""

import json
import os
import time
from pathlib import Path
from typing import Any

from src.stage06_exploration.explore_retrained_model import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDING_MODEL,
)
from src.stage06_labeling.generate_labels import (
    extract_pos_topics,
    load_bertopic_model,
)
from src.stage06_labeling.openrouter_experiments.generate_labels_openrouter import (
    DEFAULT_OPENROUTER_API_KEY,
    DEFAULT_OPENROUTER_MODEL,
    extract_representative_docs_per_topic,
    generate_all_labels,
    load_openrouter_client,
)
from src.stage06_labeling.openrouter_experiments.validate_label_quality import (
    validate_topic,
)


def test_snippet_extraction_performance(
    topic_model,
    num_topics: int = 5,
) -> dict[str, float]:
    """Test performance of snippet extraction (should be instant)."""
    print(f"\n{'='*80}")
    print("TEST 1: Snippet Extraction Performance")
    print(f"{'='*80}")
    
    # Get topic IDs
    if hasattr(topic_model, "topics_"):
        topic_ids = set(topic_model.topics_)
        topic_ids.discard(-1)
    elif hasattr(topic_model, "topic_representations_"):
        topic_ids = set(topic_model.topic_representations_.keys())
        topic_ids.discard(-1)
    else:
        print("✗ Cannot determine topic IDs")
        return {}
    
    # Limit to first N topics for testing
    topic_ids = sorted(topic_ids)[:num_topics]
    print(f"Testing with {len(topic_ids)} topics: {topic_ids}")
    
    # Measure extraction time
    start = time.perf_counter()
    topic_to_snippets = extract_representative_docs_per_topic(
        topic_model=topic_model,
        max_docs_per_topic=5,
    )
    elapsed = time.perf_counter() - start
    
    # Count snippets
    total_snippets = sum(len(snippets) for snippets in topic_to_snippets.values())
    avg_snippets = total_snippets / len(topic_ids) if topic_ids else 0
    
    print(f"\n✓ Snippet extraction completed")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Topics processed: {len(topic_ids)}")
    print(f"  Total snippets: {total_snippets}")
    print(f"  Avg snippets per topic: {avg_snippets:.1f}")
    print(f"  Time per topic: {elapsed/len(topic_ids):.3f}s")
    
    # Show snippet counts per topic
    print(f"\n  Snippet counts per topic:")
    for topic_id in topic_ids:
        count = len(topic_to_snippets.get(topic_id, []))
        print(f"    Topic {topic_id}: {count} snippets")
    
    return {
        "total_time": elapsed,
        "topics_processed": len(topic_ids),
        "time_per_topic": elapsed / len(topic_ids) if topic_ids else 0,
        "total_snippets": total_snippets,
        "avg_snippets_per_topic": avg_snippets,
    }


def test_label_generation_performance(
    pos_topics: dict[int, list[str]],
    client,
    model_name: str,
    topic_to_snippets: dict[int, list[str]],
    num_topics: int = 5,
) -> dict[str, float]:
    """Test performance of label generation (should be fast with 5 snippets)."""
    print(f"\n{'='*80}")
    print("TEST 2: Label Generation Performance")
    print(f"{'='*80}")
    
    # Limit to first N topics for testing
    topic_items = list(pos_topics.items())[:num_topics]
    limited_topics = dict(topic_items)
    topic_ids = list(limited_topics.keys())
    
    print(f"Testing with {len(topic_ids)} topics: {topic_ids}")
    
    # Measure generation time
    start = time.perf_counter()
    labels = generate_all_labels(
        pos_topics=limited_topics,
        client=client,
        model_name=model_name,
        max_new_tokens=100,
        batch_size=50,
        temperature=0.15,
        use_improved_prompts=False,
        topic_model=None,  # Not needed for generation
        topic_to_snippets=topic_to_snippets,
        max_snippets=5,
        max_chars_per_snippet=400,
    )
    elapsed = time.perf_counter() - start
    
    print(f"\n✓ Label generation completed")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Topics processed: {len(topic_ids)}")
    print(f"  Labels generated: {len(labels)}")
    print(f"  Time per topic: {elapsed/len(topic_ids):.2f}s")
    
    # Show sample labels
    print(f"\n  Sample labels:")
    for topic_id in topic_ids[:3]:
        label_data = labels.get(topic_id, {})
        label = label_data.get("label", "N/A")
        print(f"    Topic {topic_id}: {label}")
    
    return {
        "total_time": elapsed,
        "topics_processed": len(topic_ids),
        "time_per_topic": elapsed / len(topic_ids) if topic_ids else 0,
        "labels_generated": len(labels),
    }


def test_label_quality(
    labels_json_path: Path,
    bertopic_model_path: Path,
    output_csv: Path | None = None,
) -> dict[str, Any]:
    """Test label quality using validation script."""
    print(f"\n{'='*80}")
    print("TEST 3: Label Quality Validation")
    print(f"{'='*80}")
    
    if not labels_json_path.exists():
        print(f"✗ Labels JSON not found: {labels_json_path}")
        return {}
    
    print(f"Validating labels from: {labels_json_path}")
    
    # Load labels
    with open(labels_json_path, "r") as f:
        labels_data = json.load(f)
    
    # Load BERTopic model
    print("Loading BERTopic model...")
    # Try to infer parameters from path structure
    import re
    model_path = Path(bertopic_model_path)
    if model_path.is_file() or model_path.suffix == ".pkl":
        model_name = model_path.stem if model_path.suffix == ".pkl" else model_path.name
        base_dir_path = model_path.parent
        
        if len(base_dir_path.parts) >= 2 and base_dir_path.parts[-2] == "retrained":
            inferred_embedding_model = base_dir_path.parts[-1]
            inferred_base_dir = base_dir_path.parent
        else:
            inferred_embedding_model = DEFAULT_EMBEDDING_MODEL
            inferred_base_dir = DEFAULT_BASE_DIR
        
        rank_match = re.search(r"model_(\d+)", model_name)
        inferred_rank = int(rank_match.group(1)) if rank_match else 1
        
        suffix_match = re.search(r"model_\d+(.+)", model_name)
        inferred_suffix = suffix_match.group(1) if suffix_match else "_with_noise_labels"
        
        _, topic_model = load_bertopic_model(
            base_dir=inferred_base_dir,
            embedding_model=inferred_embedding_model,
            pareto_rank=inferred_rank,
            model_suffix=inferred_suffix,
        )
    else:
        from bertopic import BERTopic
        topic_model = BERTopic.load(str(model_path))
    
    # Get snippets for validation
    topic_to_snippets = extract_representative_docs_per_topic(
        topic_model=topic_model,
        max_docs_per_topic=5,
    )
    
    # Run validation
    print("Running validation...")
    validation_results = {}
    for topic_id_str, entry in labels_data.items():
        topic_id = int(topic_id_str)
        label = entry.get("label", "")
        scene_summary = entry.get("scene_summary", "")
        keywords = entry.get("keywords", [])
        snippets = topic_to_snippets.get(topic_id, []) if topic_to_snippets else None
        
        result = validate_topic(topic_id, label, scene_summary, keywords, snippets)
        validation_results[topic_id] = result.get("issues", "").split("; ") if result.get("issues") else []
    
    # Print summary
    total_topics = len(labels_data)
    issues_found = sum(len(issues) for issues in validation_results.values())
    topics_with_issues = sum(1 for issues in validation_results.values() if issues)
    
    print(f"\n✓ Validation completed")
    print(f"  Total topics: {total_topics}")
    print(f"  Topics with issues: {topics_with_issues} ({topics_with_issues/total_topics*100:.1f}%)")
    print(f"  Total issues found: {issues_found}")
    
    if topics_with_issues > 0:
        print(f"\n  Topics with issues:")
        for topic_id, issues in validation_results.items():
            if issues:
                print(f"    Topic {topic_id}: {len(issues)} issue(s)")
                for issue in issues[:2]:  # Show first 2 issues
                    print(f"      - {issue}")
    
    # Save validation report if requested
    if output_csv:
        import pandas as pd
        report_data = []
        for topic_id_str, entry in labels_data.items():
            topic_id = int(topic_id_str)
            issues = validation_results.get(topic_id, [])
            report_data.append({
                "topic_id": topic_id,
                "label": entry.get("label", ""),
                "summary": entry.get("scene_summary", ""),
                "num_issues": len(issues),
                "issues": "; ".join(issues) if issues else "",
            })
        df = pd.DataFrame(report_data)
        df.to_csv(output_csv, index=False)
        print(f"\n  Validation report saved to: {output_csv}")
    
    return {
        "total_topics": total_topics,
        "topics_with_issues": topics_with_issues,
        "total_issues": issues_found,
        "validation_results": validation_results,
    }


def main():
    """Run all performance and quality tests."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test hybrid approach performance and label quality"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base directory for BERTopic models",
    )
    parser.add_argument(
        "--num-topics",
        type=int,
        default=5,
        help="Number of topics to test (default: 5)",
    )
    parser.add_argument(
        "--test-quality",
        action="store_true",
        help="Also test label quality (requires existing labels JSON)",
    )
    parser.add_argument(
        "--labels-json",
        type=Path,
        help="Path to existing labels JSON for quality testing",
    )
    parser.add_argument(
        "--bertopic-model-path",
        type=Path,
        help="Path to BERTopic model for quality testing",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        help="Output CSV for validation report",
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Hybrid Approach Performance & Quality Test")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Embedding model: {args.embedding_model}")
    print(f"  Base dir: {args.base_dir}")
    print(f"  Test topics: {args.num_topics}")
    print()
    
    # Load BERTopic model
    print("Loading BERTopic model...")
    _, topic_model = load_bertopic_model(
        base_dir=args.base_dir,
        embedding_model=args.embedding_model,
        pareto_rank=1,
    )
    print("✓ BERTopic model loaded")
    
    # Extract POS topics
    print("Extracting POS topics...")
    pos_topics = extract_pos_topics(
        topic_model=topic_model,
        top_k=15,
        limit=args.num_topics,
    )
    print(f"✓ Extracted {len(pos_topics)} topics")
    
    # Test 1: Snippet extraction performance
    snippet_perf = test_snippet_extraction_performance(
        topic_model=topic_model,
        num_topics=args.num_topics,
    )
    
    # Get snippets for label generation
    topic_to_snippets = extract_representative_docs_per_topic(
        topic_model=topic_model,
        max_docs_per_topic=5,
    )
    
    # Test 2: Label generation performance (only if API key is available)
    api_key = os.getenv("OPENROUTER_API_KEY", DEFAULT_OPENROUTER_API_KEY)
    if api_key:
        print("\nInitializing OpenRouter client...")
        client, model_name = load_openrouter_client(
            api_key=api_key,
            model_name=DEFAULT_OPENROUTER_MODEL,
        )
        print(f"✓ OpenRouter client initialized (model: {model_name})")
        
        label_perf = test_label_generation_performance(
            pos_topics=pos_topics,
            client=client,
            model_name=model_name,
            topic_to_snippets=topic_to_snippets,
            num_topics=args.num_topics,
        )
    else:
        print("\n⚠ OPENROUTER_API_KEY not set, skipping label generation test")
        label_perf = {}
    
    # Test 3: Label quality (if requested and files exist)
    if args.test_quality and args.labels_json and args.bertopic_model_path:
        quality_results = test_label_quality(
            labels_json_path=args.labels_json,
            bertopic_model_path=args.bertopic_model_path,
            output_csv=args.validation_output,
        )
    else:
        print("\n⚠ Skipping quality test (use --test-quality with --labels-json and --bertopic-model-path)")
        quality_results = {}
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"\nPerformance Results:")
    if snippet_perf:
        print(f"  Snippet extraction: {snippet_perf.get('total_time', 0):.3f}s "
              f"({snippet_perf.get('time_per_topic', 0):.3f}s per topic)")
        print(f"    Expected: < 0.1s total (instant)")
    if label_perf:
        print(f"  Label generation: {label_perf.get('total_time', 0):.2f}s "
              f"({label_perf.get('time_per_topic', 0):.2f}s per topic)")
        print(f"    Expected: ~1-2s per topic (with 5 snippets)")
        print(f"    For {args.num_topics} topics: ~{label_perf.get('time_per_topic', 0) * args.num_topics:.1f}s")
        print(f"    For 15 topics: ~{label_perf.get('time_per_topic', 0) * 15:.1f}s")
    
    if quality_results:
        print(f"\nQuality Results:")
        print(f"  Topics validated: {quality_results.get('total_topics', 0)}")
        print(f"  Topics with issues: {quality_results.get('topics_with_issues', 0)}")
        print(f"  Total issues: {quality_results.get('total_issues', 0)}")
    
    print(f"\n{'='*80}")
    print("✓ All tests completed")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

