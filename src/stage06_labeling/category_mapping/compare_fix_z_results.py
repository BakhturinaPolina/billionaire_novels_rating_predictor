#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare before/after results of fix Z script.

This script analyzes the impact of the fix Z process by comparing
original category mappings with fixed category mappings.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from src.stage06_labeling.category_mapping.map_topics_to_categories import CATS


def load_json(fp: Path) -> Dict:
    """Load JSON file."""
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_categories(cat_data: Dict) -> Dict[str, float]:
    """Extract categories dict from category data (handles both formats)."""
    if isinstance(cat_data, dict) and "categories" in cat_data:
        return cat_data["categories"]
    elif isinstance(cat_data, dict) and all(isinstance(v, (int, float)) for v in cat_data.values()):
        return cat_data
    else:
        return {}


def get_primary_category(cats: Dict[str, float]) -> str:
    """Get primary category (highest weight)."""
    if not cats:
        return "Z_noise_oog"  # Default for empty
    return max(cats.items(), key=lambda x: x[1])[0]


def identify_changed_topics(
    original: Dict,
    fixed: Dict,
) -> List[Tuple[str, Dict[str, float], Dict[str, float]]]:
    """
    Identify topics that changed between original and fixed mappings.
    
    Returns:
        List of (topic_id, original_cats, fixed_cats) tuples
    """
    changed = []
    
    for topic_id in original.keys():
        orig_cats = extract_categories(original[topic_id])
        fixed_cats = extract_categories(fixed.get(topic_id, {}))
        
        # Check if primary category changed
        orig_primary = get_primary_category(orig_cats)
        fixed_primary = get_primary_category(fixed_cats)
        
        if orig_primary != fixed_primary:
            changed.append((topic_id, orig_cats, fixed_cats))
    
    return changed


def generate_comparison_report(
    original: Dict,
    fixed: Dict,
    fix_results: Dict = None,
    labels: Dict = None,
) -> str:
    """
    Generate a markdown comparison report.
    
    Args:
        original: Original category mappings
        fixed: Fixed category mappings
        fix_results: Optional fix Z results from LLM
        labels: Optional topic labels for context
        
    Returns:
        Markdown report string
    """
    lines = []
    lines.append("# Fix Z Results Comparison Report\n")
    lines.append("This report compares category mappings before and after running the fix Z script.\n")
    
    # Statistics
    orig_z_count = 0
    fixed_z_count = 0
    changed_count = 0
    
    changed_topics = identify_changed_topics(original, fixed)
    changed_count = len(changed_topics)
    
    # Count Z topics
    for topic_id, cat_data in original.items():
        cats = extract_categories(cat_data)
        if get_primary_category(cats) == "Z_noise_oog":
            orig_z_count += 1
    
    for topic_id, cat_data in fixed.items():
        cats = extract_categories(cat_data)
        if get_primary_category(cats) == "Z_noise_oog":
            fixed_z_count += 1
    
    lines.append("## Summary Statistics\n")
    lines.append(f"- **Total topics**: {len(original)}")
    lines.append(f"- **Topics originally classified as Z_noise_oog**: {orig_z_count}")
    lines.append(f"- **Topics still classified as Z_noise_oog after fix**: {fixed_z_count}")
    lines.append(f"- **Topics reassigned**: {changed_count}")
    lines.append(f"- **Reduction in Z topics**: {orig_z_count - fixed_z_count}\n")
    
    # Category reassignment statistics
    if changed_topics:
        lines.append("## Category Reassignments\n")
        
        reassignment_counts = Counter()
        for topic_id, orig_cats, fixed_cats in changed_topics:
            orig_primary = get_primary_category(orig_cats)
            fixed_primary = get_primary_category(fixed_cats)
            reassignment_counts[(orig_primary, fixed_primary)] += 1
        
        lines.append("| From Category | To Category | Count |")
        lines.append("|---------------|-------------|-------|")
        for (from_cat, to_cat), count in reassignment_counts.most_common():
            lines.append(f"| {from_cat} | {to_cat} | {count} |")
        lines.append("")
        
        # Category distribution changes
        lines.append("## Category Distribution Changes\n")
        
        orig_dist = Counter()
        fixed_dist = Counter()
        
        for topic_id, cat_data in original.items():
            cats = extract_categories(cat_data)
            orig_dist[get_primary_category(cats)] += 1
        
        for topic_id, cat_data in fixed.items():
            cats = extract_categories(cat_data)
            fixed_dist[get_primary_category(cats)] += 1
        
        lines.append("| Category | Original Count | Fixed Count | Change |")
        lines.append("|----------|----------------|-------------|--------|")
        
        all_cats = set(orig_dist.keys()) | set(fixed_dist.keys())
        for cat in sorted(all_cats):
            orig_count = orig_dist.get(cat, 0)
            fixed_count = fixed_dist.get(cat, 0)
            change = fixed_count - orig_count
            change_str = f"+{change}" if change > 0 else str(change)
            lines.append(f"| {cat} | {orig_count} | {fixed_count} | {change_str} |")
        lines.append("")
    
    # Detailed changes
    if changed_topics:
        lines.append("## Detailed Changes\n")
        lines.append("### Topics Reassigned from Z_noise_oog\n")
        lines.append("| Topic ID | Label | Keywords | Original | Fixed | Rationale |")
        lines.append("|----------|-------|----------|----------|-------|-----------|")
        
        for topic_id, orig_cats, fixed_cats in changed_topics:
            orig_primary = get_primary_category(orig_cats)
            if orig_primary != "Z_noise_oog":
                continue
            
            fixed_primary = get_primary_category(fixed_cats)
            
            # Get label and keywords if available
            label = ""
            keywords_str = ""
            if labels and topic_id in labels:
                label = labels[topic_id].get("label", "")
                keywords = labels[topic_id].get("keywords", [])
                keywords_str = ", ".join(keywords[:5]) + ("..." if len(keywords) > 5 else "")
            
            # Get rationale if available
            rationale = ""
            if fix_results and topic_id in fix_results:
                rationale = fix_results[topic_id].get("rationale", "")[:100]
            
            lines.append(f"| {topic_id} | {label} | {keywords_str} | {orig_primary} | {fixed_primary} | {rationale} |")
        lines.append("")
    
    # LLM confidence metrics (if available)
    if fix_results:
        lines.append("## LLM Processing Statistics\n")
        
        total_processed = len(fix_results)
        fixed_count = sum(1 for r in fix_results.values() if not r.get("is_noise", True))
        kept_noise_count = sum(1 for r in fix_results.values() if r.get("is_noise", True))
        
        total_time = sum(r.get("api_time", 0) for r in fix_results.values())
        total_tokens = sum(r.get("api_tokens", 0) for r in fix_results.values())
        
        lines.append(f"- **Topics processed**: {total_processed}")
        lines.append(f"- **Topics fixed**: {fixed_count}")
        lines.append(f"- **Topics kept as noise**: {kept_noise_count}")
        lines.append(f"- **Total API time**: {total_time:.2f}s")
        lines.append(f"- **Average time per topic**: {total_time/total_processed:.2f}s" if total_processed > 0 else "-")
        lines.append(f"- **Total tokens used**: {total_tokens}")
        lines.append(f"- **Average tokens per topic**: {total_tokens//total_processed if total_processed > 0 else 0}")
        lines.append("")
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare before/after results of fix Z script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--original",
        type=Path,
        required=True,
        help="Path to original category probabilities JSON",
    )
    
    parser.add_argument(
        "--fixed",
        type=Path,
        required=True,
        help="Path to fixed category probabilities JSON",
    )
    
    parser.add_argument(
        "--fix-results",
        type=Path,
        default=None,
        help="Optional: Path to fix Z results JSON (for LLM statistics)",
    )
    
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional: Path to labels JSON (for topic context)",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output comparison report (markdown)",
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading original mappings from {args.original}")
    original = load_json(args.original)
    
    print(f"Loading fixed mappings from {args.fixed}")
    fixed = load_json(args.fixed)
    
    fix_results = None
    if args.fix_results:
        print(f"Loading fix results from {args.fix_results}")
        fix_results = load_json(args.fix_results)
    
    labels = None
    if args.labels:
        print(f"Loading labels from {args.labels}")
        labels = load_json(args.labels)
    
    # Generate report
    print("Generating comparison report...")
    report = generate_comparison_report(original, fixed, fix_results, labels)
    
    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"✓ Comparison report saved to {args.output}")


if __name__ == "__main__":
    main()

