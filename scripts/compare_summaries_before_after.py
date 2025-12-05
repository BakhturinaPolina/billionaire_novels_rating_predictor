"""Compare scene summaries before and after prompt/token changes.

This script helps visualize the improvement in scene summaries by comparing
truncated summaries (from max_tokens=16) with improved summaries (from max_tokens=60).

Usage:
    python scripts/compare_summaries_before_after.py \
      --before-json results/stage06_labeling_openrouter/labels_pos_openrouter_mistralai_mistral-nemo_romance_aware_paraphrase-MiniLM-L6-v2.json \
      --after-json results/stage06_labeling_openrouter/labels_pos_openrouter_mistralai_mistral-nemo_romance_aware_paraphrase-MiniLM-L6-v2.json \
      --limit-topics 15
"""

import argparse
import json
from pathlib import Path


def count_tokens(text: str) -> int:
    """Simple token count (word count approximation)."""
    return len(text.split())


def analyze_summary(summary: str) -> dict:
    """Analyze a scene summary for quality indicators."""
    s = summary.strip()
    tokens = count_tokens(s)
    
    has_punctuation = s and s[-1] in ".?!"
    ends_on_function_word = False
    if tokens > 0:
        last_word = s.split()[-1].strip(".,!?\"'").lower()
        function_words = {"a", "an", "the", "to", "for", "of", "in", "at", "on", "with", 
                         "about", "into", "as", "from", "by", "and", "or", "but"}
        ends_on_function_word = last_word in function_words
    
    return {
        "tokens": tokens,
        "has_punctuation": has_punctuation,
        "ends_on_function_word": ends_on_function_word,
        "is_complete": has_punctuation and not ends_on_function_word and 5 <= tokens <= 40,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare scene summaries before and after prompt/token improvements"
    )
    parser.add_argument(
        "--before-json",
        type=str,
        required=True,
        help="Path to labels JSON with old summaries (max_tokens=16)",
    )
    parser.add_argument(
        "--after-json",
        type=str,
        default=None,
        help="Path to labels JSON with new summaries (max_tokens=60). If not provided, only shows before analysis.",
    )
    parser.add_argument(
        "--limit-topics",
        type=int,
        default=None,
        help="Limit comparison to first N topics",
    )
    args = parser.parse_args()

    # Load before JSON
    before_path = Path(args.before_json)
    if not before_path.is_file():
        raise FileNotFoundError(f"Before JSON not found: {before_path}")
    
    with before_path.open("r", encoding="utf-8") as f:
        before_data = json.load(f)

    # Load after JSON if provided
    after_data = None
    if args.after_json:
        after_path = Path(args.after_json)
        if not after_path.is_file():
            print(f"Warning: After JSON not found: {after_path}")
            print("Showing before analysis only...")
        else:
            with after_path.open("r", encoding="utf-8") as f:
                after_data = json.load(f)

    # Process topics
    topic_items = sorted(before_data.items(), key=lambda x: int(x[0]))
    if args.limit_topics:
        topic_items = topic_items[:args.limit_topics]

    print("=" * 100)
    print("SCENE SUMMARY COMPARISON")
    print("=" * 100)
    print(f"Before JSON: {before_path.name}")
    if after_data:
        print(f"After JSON: {Path(args.after_json).name}")
    else:
        print("After JSON: (not provided - showing before analysis only)")
    print(f"Topics analyzed: {len(topic_items)}")
    print()

    # Statistics
    before_stats = {
        "total": 0,
        "complete": 0,
        "truncated": 0,
        "no_punctuation": 0,
        "ends_function_word": 0,
        "avg_tokens": 0,
    }
    
    after_stats = {
        "total": 0,
        "complete": 0,
        "truncated": 0,
        "no_punctuation": 0,
        "ends_function_word": 0,
        "avg_tokens": 0,
    }

    # Detailed comparison
    comparisons = []
    
    for topic_id_str, before_entry in topic_items:
        topic_id = int(topic_id_str)
        before_summary = before_entry.get("scene_summary", "") or ""
        before_analysis = analyze_summary(before_summary)
        
        before_stats["total"] += 1
        before_stats["avg_tokens"] += before_analysis["tokens"]
        if before_analysis["is_complete"]:
            before_stats["complete"] += 1
        else:
            before_stats["truncated"] += 1
        if not before_analysis["has_punctuation"]:
            before_stats["no_punctuation"] += 1
        if before_analysis["ends_on_function_word"]:
            before_stats["ends_function_word"] += 1

        after_summary = ""
        after_analysis = None
        if after_data and topic_id_str in after_data:
            after_entry = after_data[topic_id_str]
            after_summary = after_entry.get("scene_summary", "") or ""
            after_analysis = analyze_summary(after_summary)
            
            after_stats["total"] += 1
            after_stats["avg_tokens"] += after_analysis["tokens"]
            if after_analysis["is_complete"]:
                after_stats["complete"] += 1
            else:
                after_stats["truncated"] += 1
            if not after_analysis["has_punctuation"]:
                after_stats["no_punctuation"] += 1
            if after_analysis["ends_on_function_word"]:
                after_stats["ends_function_word"] += 1

        comparisons.append({
            "topic_id": topic_id,
            "label": before_entry.get("label", ""),
            "before": before_summary,
            "before_analysis": before_analysis,
            "after": after_summary,
            "after_analysis": after_analysis,
        })

    # Calculate averages
    if before_stats["total"] > 0:
        before_stats["avg_tokens"] = before_stats["avg_tokens"] / before_stats["total"]
    if after_stats["total"] > 0:
        after_stats["avg_tokens"] = after_stats["avg_tokens"] / after_stats["total"]

    # Print statistics
    print("STATISTICS")
    print("-" * 100)
    print(f"{'Metric':<30} {'Before (max_tokens=16)':<25} {'After (max_tokens=60)':<25}")
    print("-" * 100)
    print(f"{'Total topics':<30} {before_stats['total']:<25} {after_stats['total']:<25}")
    print(f"{'Complete summaries':<30} {before_stats['complete']:<25} {after_stats['complete']:<25}")
    print(f"{'Truncated summaries':<30} {before_stats['truncated']:<25} {after_stats['truncated']:<25}")
    print(f"{'Missing punctuation':<30} {before_stats['no_punctuation']:<25} {after_stats['no_punctuation']:<25}")
    print(f"{'Ends on function word':<30} {before_stats['ends_function_word']:<25} {after_stats['ends_function_word']:<25}")
    before_avg = f"{before_stats['avg_tokens']:.1f}"
    after_avg = f"{after_stats['avg_tokens']:.1f}" if after_stats['total'] > 0 else "N/A"
    print(f"{'Avg tokens per summary':<30} {before_avg:<25} {after_avg:<25}")
    print()

    # Print detailed comparisons
    print("DETAILED COMPARISON (First 15 topics)")
    print("=" * 100)
    
    for comp in comparisons[:15]:
        print(f"\nTopic {comp['topic_id']}: {comp['label']}")
        print("-" * 100)
        
        before_status = "✓ COMPLETE" if comp['before_analysis']['is_complete'] else "✗ TRUNCATED"
        print(f"BEFORE ({before_status}):")
        print(f"  Tokens: {comp['before_analysis']['tokens']}")
        print(f"  Punctuation: {'✓' if comp['before_analysis']['has_punctuation'] else '✗'}")
        print(f"  Ends on function word: {'✗' if comp['before_analysis']['ends_on_function_word'] else '✓'}")
        print(f"  Summary: {comp['before']!r}")
        
        if comp['after_analysis']:
            after_status = "✓ COMPLETE" if comp['after_analysis']['is_complete'] else "✗ TRUNCATED"
            print(f"\nAFTER ({after_status}):")
            print(f"  Tokens: {comp['after_analysis']['tokens']}")
            print(f"  Punctuation: {'✓' if comp['after_analysis']['has_punctuation'] else '✗'}")
            print(f"  Ends on function word: {'✗' if comp['after_analysis']['ends_on_function_word'] else '✓'}")
            print(f"  Summary: {comp['after']!r}")
        else:
            print("\nAFTER: (not available)")
        print()

    if len(comparisons) > 15:
        print(f"... ({len(comparisons) - 15} more topics not shown)")


if __name__ == "__main__":
    main()

