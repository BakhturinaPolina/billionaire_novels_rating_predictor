#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate category assignments against theoretical framework.

This script identifies:
1. Topics misclassified as Z_noise_oog that should be romance categories
2. Topics with incorrect category assignments
3. Edge cases and potential improvements
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from src.stage06_labeling.category_mapping.map_topics_to_categories import CATS, TRIG


def load_data(summary_csv: Path, probs_json: Path, labels_json: Path = None) -> Tuple[Dict, Dict, Dict]:
    """Load category mapping data."""
    # Load summary CSV
    summary = {}
    with open(summary_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            summary[row['topic_id']] = row
    
    # Load probabilities JSON
    with open(probs_json, 'r', encoding='utf-8') as f:
        probs = json.load(f)
    
    # Load labels JSON if provided
    labels = {}
    if labels_json and labels_json.exists():
        with open(labels_json, 'r', encoding='utf-8') as f:
            labels = json.load(f)
    
    return summary, probs, labels


def find_z_misclassifications(summary: Dict, labels: Dict) -> List[Dict]:
    """Find topics classified as Z that should be romance categories."""
    issues = []
    
    # Romance-relevant keywords that should NOT be Z
    romance_keywords = {
        'intercourse': ['C_explicit', 'B_mutual_intimacy'],
        'sex': ['C_explicit', 'B_mutual_intimacy'],
        'intimacy': ['B_mutual_intimacy'],
        'kiss': ['B_mutual_intimacy'],
        'touch': ['B_mutual_intimacy'],
        'mouth play': ['B_mutual_intimacy', 'C_explicit'],
        'moan': ['B_mutual_intimacy', 'C_explicit'],
        'outfit': ['O_appearance_aesthetics'],
        'clothing': ['O_appearance_aesthetics'],
        'fashion': ['O_appearance_aesthetics'],
        'lingerie': ['O_appearance_aesthetics', 'D_luxury_wealth_status'],
        'nighttime': ['B_mutual_intimacy', 'H_domestic_nesting'],
        'night': ['B_mutual_intimacy', 'H_domestic_nesting'],
    }
    
    for topic_id, row in summary.items():
        if row['primary_category'] != 'Z_noise_oog':
            continue
        
        label = row['label'].lower()
        keywords = row['keywords'].lower()
        search_text = f"{label} {keywords}"
        
        # Check for romance-relevant terms
        for keyword, expected_cats in romance_keywords.items():
            if keyword in search_text:
                issues.append({
                    'topic_id': topic_id,
                    'label': row['label'],
                    'keywords': row['keywords'],
                    'current_category': 'Z_noise_oog',
                    'issue': f"Contains '{keyword}' but classified as Z",
                    'expected_categories': expected_cats,
                    'severity': 'high' if keyword in ['intercourse', 'sex', 'intimacy'] else 'medium'
                })
                break
    
    return issues


def validate_explicit_vs_intimacy(summary: Dict) -> List[Dict]:
    """Check if explicit topics are correctly classified vs mutual intimacy."""
    issues = []
    
    explicit_keywords = ['clit', 'pussy', 'erection', 'cock', 'penetrat', 'orgasm', 'nipple']
    intimacy_keywords = ['tender', 'kiss', 'caress', 'whisper', 'foreplay']
    
    for topic_id, row in summary.items():
        label = row['label'].lower()
        keywords = row['keywords'].lower()
        search_text = f"{label} {keywords}"
        primary = row['primary_category']
        
        # Check if explicit keywords present but classified as B
        has_explicit = any(kw in search_text for kw in explicit_keywords)
        has_intimacy = any(kw in search_text for kw in intimacy_keywords)
        
        if has_explicit and primary == 'B_mutual_intimacy' and not has_intimacy:
            # Should be C_explicit or split B/C
            issues.append({
                'topic_id': topic_id,
                'label': row['label'],
                'keywords': row['keywords'],
                'current_category': primary,
                'issue': 'Contains explicit keywords but classified as B_mutual_intimacy only',
                'expected_categories': ['C_explicit', 'B_mutual_intimacy'],
                'severity': 'medium'
            })
    
    return issues


def check_category_coverage(summary: Dict) -> Dict:
    """Check distribution of categories."""
    category_counts = {}
    for row in summary.values():
        cat = row['primary_category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    return category_counts


def check_regex_patterns(summary: Dict, labels: Dict) -> List[Dict]:
    """Check if regex patterns are missing obvious matches."""
    issues = []
    
    # Test specific problematic cases
    test_cases = [
        {
            'topic_id': '32',
            'label': 'Hesitant Intercourse',
            'keywords': 'offense, say, answer, head, worries, intercourse, plans, word, question, hesitation',
            'expected': 'C_explicit'
        },
        {
            'topic_id': '39',
            'label': 'Moist Mouth Play',
            'keywords': 'lip, teeth, nose, tongue, mouth, bridge, cheek, moan, lower, saliva',
            'expected': 'B_mutual_intimacy'
        },
        {
            'topic_id': '31',
            'label': "Tomorrow's Outfit",
            'keywords': 'clothes, outfit, wardrobe, clothing, outfits, tomorrow, selling, different, fashion, compatible',
            'expected': 'O_appearance_aesthetics'
        }
    ]
    
    for case in test_cases:
        topic_id = case['topic_id']
        if topic_id in summary:
            row = summary[topic_id]
            if row['primary_category'] != case['expected']:
                # Check if regex should match
                label = case['label'].lower()
                keywords = case['keywords'].lower()
                search_text = f"{label} {keywords}"
                
                # Test C_explicit patterns
                c_patterns = TRIG.get('C_explicit', [])
                c_matches = [p for p in c_patterns if re.search(p, search_text, re.I)]
                
                # Test B_mutual_intimacy patterns
                b_patterns = TRIG.get('B_mutual_intimacy', [])
                b_matches = [p for p in b_patterns if re.search(p, search_text, re.I)]
                
                # Test O_appearance_aesthetics patterns
                o_patterns = TRIG.get('O_appearance_aesthetics', [])
                o_matches = [p for p in o_patterns if re.search(p, search_text, re.I)]
                
                issues.append({
                    'topic_id': topic_id,
                    'label': case['label'],
                    'current_category': row['primary_category'],
                    'expected_category': case['expected'],
                    'issue': f"Should be {case['expected']} but is {row['primary_category']}",
                    'c_patterns_matched': c_matches,
                    'b_patterns_matched': b_matches,
                    'o_patterns_matched': o_matches,
                    'severity': 'high'
                })
    
    return issues


def generate_validation_report(
    z_issues: List[Dict],
    explicit_issues: List[Dict],
    regex_issues: List[Dict],
    category_dist: Dict
) -> str:
    """Generate markdown validation report."""
    lines = []
    lines.append("# Category Mapping Validation Report\n")
    lines.append("## Summary\n")
    lines.append(f"- **Total topics analyzed**: {sum(category_dist.values())}")
    lines.append(f"- **Z misclassifications found**: {len(z_issues)}")
    lines.append(f"- **Explicit/Intimacy issues**: {len(explicit_issues)}")
    lines.append(f"- **Regex pattern issues**: {len(regex_issues)}")
    lines.append("")
    
    # Category distribution
    lines.append("## Category Distribution\n")
    lines.append("| Category | Count | Percentage |")
    lines.append("|----------|-------|------------|")
    total = sum(category_dist.values())
    for cat in sorted(category_dist.keys()):
        count = category_dist[cat]
        pct = (count / total * 100) if total > 0 else 0
        lines.append(f"| {cat} | {count} | {pct:.1f}% |")
    lines.append("")
    
    # Z misclassifications
    if z_issues:
        lines.append("## ⚠️ Z Misclassifications (High Priority)\n")
        lines.append("Topics classified as Z_noise_oog that contain romance-relevant keywords:\n")
        for issue in sorted(z_issues, key=lambda x: x['severity'], reverse=True):
            lines.append(f"### Topic {issue['topic_id']}: {issue['label']}")
            lines.append(f"- **Current**: {issue['current_category']}")
            lines.append(f"- **Issue**: {issue['issue']}")
            lines.append(f"- **Expected**: {', '.join(issue['expected_categories'])}")
            lines.append(f"- **Keywords**: {issue['keywords'][:100]}")
            lines.append(f"- **Severity**: {issue['severity']}")
            lines.append("")
    
    # Explicit/Intimacy issues
    if explicit_issues:
        lines.append("## Explicit vs Intimacy Classification Issues\n")
        for issue in explicit_issues:
            lines.append(f"### Topic {issue['topic_id']}: {issue['label']}")
            lines.append(f"- **Current**: {issue['current_category']}")
            lines.append(f"- **Issue**: {issue['issue']}")
            lines.append(f"- **Expected**: {', '.join(issue['expected_categories'])}")
            lines.append("")
    
    # Regex pattern issues
    if regex_issues:
        lines.append("## Regex Pattern Matching Issues\n")
        for issue in regex_issues:
            lines.append(f"### Topic {issue['topic_id']}: {issue['label']}")
            lines.append(f"- **Current**: {issue['current_category']}")
            lines.append(f"- **Expected**: {issue['expected_category']}")
            lines.append(f"- **Issue**: {issue['issue']}")
            if issue.get('c_patterns_matched'):
                lines.append(f"- **C_explicit patterns matched**: {len(issue['c_patterns_matched'])}")
            if issue.get('b_patterns_matched'):
                lines.append(f"- **B_mutual_intimacy patterns matched**: {len(issue['b_patterns_matched'])}")
            if issue.get('o_patterns_matched'):
                lines.append(f"- **O_appearance_aesthetics patterns matched**: {len(issue['o_patterns_matched'])}")
            lines.append("")
    
    # Recommendations
    lines.append("## Recommendations\n")
    if z_issues:
        lines.append("1. **Run fix Z script** to reclassify topics with romance-relevant keywords")
        lines.append("2. **Review regex patterns** for 'intercourse', 'mouth play', 'outfit'")
        lines.append("3. **Update patterns** if needed to catch these cases")
    else:
        lines.append("✓ No Z misclassifications found")
    
    if explicit_issues:
        lines.append("4. **Review explicit vs intimacy boundaries** - some topics may need split assignments")
    
    lines.append("")
    lines.append("## Next Steps\n")
    lines.append("1. Review identified issues")
    lines.append("2. Run `fix_z_topics.py` if Z misclassifications found")
    lines.append("3. Update regex patterns in `map_topics_to_categories.py` if needed")
    lines.append("4. Re-run category mapping after fixes")
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate category assignments against theoretical framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/stage06_labeling/category_mapping/topic_to_category_summary.csv"),
        help="Path to category summary CSV"
    )
    
    parser.add_argument(
        "--probs",
        type=Path,
        default=Path("results/stage06_labeling/category_mapping/topic_to_category_probs.json"),
        help="Path to category probabilities JSON"
    )
    
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional: Path to labels JSON for context"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage06_labeling/category_mapping/validation_report.md"),
        help="Path to output validation report"
    )
    
    args = parser.parse_args()
    
    print("Loading data...")
    summary, probs, labels = load_data(args.summary, args.probs, args.labels)
    
    print("Finding Z misclassifications...")
    z_issues = find_z_misclassifications(summary, labels)
    
    print("Validating explicit vs intimacy classifications...")
    explicit_issues = validate_explicit_vs_intimacy(summary)
    
    print("Checking regex patterns...")
    regex_issues = check_regex_patterns(summary, labels)
    
    print("Analyzing category distribution...")
    category_dist = check_category_coverage(summary)
    
    print("Generating validation report...")
    report = generate_validation_report(z_issues, explicit_issues, regex_issues, category_dist)
    
    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ Validation report saved to {args.output}")
    print(f"\nSummary:")
    print(f"  - Z misclassifications: {len(z_issues)}")
    print(f"  - Explicit/Intimacy issues: {len(explicit_issues)}")
    print(f"  - Regex pattern issues: {len(regex_issues)}")


if __name__ == "__main__":
    main()

