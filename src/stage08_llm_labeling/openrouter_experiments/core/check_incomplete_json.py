#!/usr/bin/env python3
"""Check for incomplete JSON files from interrupted label generation."""

import json
import sys
from pathlib import Path

def check_json_file(json_path: Path) -> tuple[bool, str, int]:
    """
    Check if JSON file is complete and valid.
    
    Returns:
        (is_valid, error_message, topic_count)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if file ends properly
        content_stripped = content.rstrip()
        if not content_stripped.endswith('}'):
            return False, "Missing closing brace '}'", 0
        
        # Try to parse JSON
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                topic_count = len(data)
                return True, "Valid JSON", topic_count
            else:
                return False, f"JSON is not a dict (got {type(data).__name__})", 0
        except json.JSONDecodeError as e:
            # Try to count topics before the error
            lines = content.split('\n')
            topic_count = sum(1 for line in lines if '"' in line and ':' in line)
            return False, f"JSON decode error: {e}", topic_count
            
    except FileNotFoundError:
        return False, "File not found", 0
    except Exception as e:
        return False, f"Error reading file: {e}", 0


def main():
    """Check all JSON files in results/stage08_llm_labeling."""
    results_dir = Path("results/stage08_llm_labeling")
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    json_files = list(results_dir.rglob("*.json"))
    
    if not json_files:
        print("No JSON files found")
        return
    
    print(f"Checking {len(json_files)} JSON files...\n")
    
    incomplete_files = []
    valid_files = []
    
    for json_file in sorted(json_files):
        is_valid, error_msg, topic_count = check_json_file(json_file)
        
        if is_valid:
            print(f"✓ {json_file.name}: Valid ({topic_count} topics)")
            valid_files.append((json_file, topic_count))
        else:
            print(f"✗ {json_file.name}: {error_msg} ({topic_count} topics found)")
            incomplete_files.append((json_file, error_msg, topic_count))
    
    print("\n" + "=" * 80)
    print(f"Summary: {len(valid_files)} valid, {len(incomplete_files)} incomplete")
    
    if incomplete_files:
        print("\nIncomplete files:")
        for json_file, error_msg, topic_count in incomplete_files:
            print(f"  - {json_file}")
            print(f"    Error: {error_msg}")
            print(f"    Topics found: {topic_count}")


if __name__ == "__main__":
    main()
