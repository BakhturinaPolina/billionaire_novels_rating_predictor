"""Post-processing script to flag likely truncated or suspicious scene summaries.

This script analyzes labels JSON files produced by the OpenRouter labeling pipeline
and identifies scene summaries that may be truncated, incomplete, or low-quality.

Usage:
    python scripts/check_truncated_summaries.py \
      --labels-json results/stage06_labeling_openrouter/labels_pos_openrouter_mistralai_mistral-nemo_romance_aware_paraphrase-MiniLM-L6-v2.json \
      --max-examples 40
"""

import argparse
import json
from pathlib import Path


SUSPECT_FINAL_TOKENS = {
    "a", "an", "the",
    "to", "for", "of", "in", "at", "on", "with",
    "about", "into", "as", "from", "by",
    "and", "or", "but",
    "that", "this", "these", "those",
}


def is_suspicious_summary(summary: str) -> tuple[bool, list[str]]:
    """Heuristic checks for truncated or low-quality scene summaries.
    
    Args:
        summary: Scene summary string to check
        
    Returns:
        Tuple of (is_suspicious, list_of_reasons)
    """
    reasons: list[str] = []
    s = summary.strip()

    if not s:
        reasons.append("empty")
        return True, reasons

    # Basic tokenization
    tokens = s.split()
    num_tokens = len(tokens)

    # Very short or very long summaries can be suspicious
    if num_tokens < 5:
        reasons.append(f"too_short({num_tokens})")
    if num_tokens > 40:
        reasons.append(f"too_long({num_tokens})")

    # Last character / punctuation
    last_char = s[-1]
    if last_char not in ".?!":
        reasons.append(f"no_terminal_punctuation('{last_char}')")

    # Final token heuristic (common truncated function words)
    last_token = tokens[-1].strip(".,!?\"'").lower()
    if last_token in SUSPECT_FINAL_TOKENS:
        reasons.append(f"suspicious_final_token('{last_token}')")

    # Fragment-like endings
    if s.endswith((" to", " for", " with", " about", " as", " from", " by")):
        reasons.append("ends_with_function_phrase")

    return (len(reasons) > 0), reasons


def main():
    parser = argparse.ArgumentParser(
        description="Flag suspicious or truncated scene summaries in labels JSON files"
    )
    parser.add_argument(
        "--labels-json",
        type=str,
        required=True,
        help="Path to labels JSON file produced by the OpenRouter labeling pipeline.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=50,
        help="Maximum number of suspicious examples to print.",
    )
    parser.add_argument(
        "--limit-topics",
        type=int,
        default=None,
        help="Limit processing to first N topics (useful for testing).",
    )
    args = parser.parse_args()

    labels_path = Path(args.labels_json)
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels JSON not found: {labels_path}")

    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    suspicious = []
    total = 0

    # Sort topic IDs to process in order
    topic_items = sorted(data.items(), key=lambda x: int(x[0]))
    
    # Apply limit if specified
    if args.limit_topics:
        topic_items = topic_items[:args.limit_topics]

    for topic_id_str, entry in topic_items:
        total += 1
        scene_summary = entry.get("scene_summary", "") or ""
        is_bad, reasons = is_suspicious_summary(scene_summary)
        if is_bad:
            suspicious.append((int(topic_id_str), scene_summary, reasons))

    suspicious.sort(key=lambda x: x[0])

    print(f"Total topics: {total}")
    print(f"Suspicious summaries: {len(suspicious)}")
    print()

    for topic_id, summary, reasons in suspicious[: args.max_examples]:
        print("=" * 80)
        print(f"Topic {topic_id}")
        print(f"Reasons: {', '.join(reasons)}")
        print(f"Scene summary: {summary!r}")

    if len(suspicious) > args.max_examples:
        print(f"\n... ({len(suspicious) - args.max_examples} more suspicious summaries not shown)")


if __name__ == "__main__":
    main()

