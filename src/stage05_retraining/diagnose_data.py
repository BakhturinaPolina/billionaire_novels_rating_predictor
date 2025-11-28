"""Data diagnosis tool for preprocessing validation."""

import pandas as pd
import re
from pathlib import Path
from typing import Optional


def diagnose_data(
    dataset_path: Path,
    stoplist_path: Optional[Path] = None,
    sample_size: int = 50
) -> bool:
    """
    Diagnose data preprocessing issues in the dataset.
    
    Args:
        dataset_path: Path to the dataset CSV file
        stoplist_path: Optional path to custom stoplist file
        sample_size: Number of rows to sample for tokenization checks (default: 50)
        
    Returns:
        True if diagnosis completed successfully, False otherwise
    """
    print("=" * 80)
    print("Data Preprocessing Diagnosis")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    
    if not dataset_path.exists():
        print(f"❌ Error: {dataset_path} not found.")
        return False
    
    # Default stoplist path if not provided
    if stoplist_path is None:
        stoplist_path = Path("data/processed/custom_stoplist.txt")
    
    print(f"Stoplist: {stoplist_path}")
    print()

    # 1. Check encodings
    encodings_to_check = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    successful_encoding = None
    
    print(f"\n[DIAGNOSE] Step 1: Checking encodings for {dataset_path}...")
    for enc in encodings_to_check:
        try:
            with open(dataset_path, 'r', encoding=enc) as f:
                content = f.read(10000) # Read first 10kb
            print(f"  - {enc}: Success")
            if successful_encoding is None:
                successful_encoding = enc
        except UnicodeDecodeError:
            print(f"  - {enc}: Failed")
            
    if successful_encoding:
        print(f"\nUsing encoding: {successful_encoding}")
        
        # 2. Search for artifacts
        artifacts = ["reeseâ", "â€", "â€", "â€œ"]
        print(f"\nSearching for artifacts: {artifacts}")
        
        found_artifacts = {a: 0 for a in artifacts}
        line_count = 0
        
        try:
            with open(dataset_path, 'r', encoding=successful_encoding, errors='replace') as f:
                for i, line in enumerate(f):
                    line_count += 1
                    for artifact in artifacts:
                        if artifact in line:
                            found_artifacts[artifact] += 1
                            if found_artifacts[artifact] <= 3:
                                print(f"    Found '{artifact}' in line {i+1}: {line.strip()[:100]}...")
        except Exception as e:
            print(f"  ❌ Error reading file line by line: {e}")
            
        print("\n[DIAGNOSE] Artifact counts:")
        for k, v in found_artifacts.items():
            print(f"  - '{k}': {v} occurrences")
            
        # 3. Tokenization check on a sample
        print(f"\n[DIAGNOSE] Step 3: Checking tokenization on sample ({sample_size} rows)...")
        try:
            df = pd.read_csv(dataset_path, encoding=successful_encoding, nrows=sample_size)
            if 'Sentence' in df.columns:
                sentences = df['Sentence'].fillna('').astype(str).tolist()
                
                # Simple whitespace split vs regex
                for i, sent in enumerate(sentences[:5]):
                    # Simulate what might be happening
                    # The user mentioned empty words: ""
                    
                    # Regex split that might leave empty strings if not handled
                    words_simple = sent.split()
                    words_regex = re.split(r'\W+', sent)
                    
                    empty_words = [w for w in words_regex if w == '']
                    if empty_words:
                        print(f"  Line {i}: Found {len(empty_words)} empty tokens using re.split(r'\\W+')")
                        
                    # Check for single chars
                    single_chars = [w for w in words_simple if len(w) == 1]
                    if single_chars:
                        print(f"  Line {i}: Found single chars: {single_chars}")
                        
            else:
                print("  'Sentence' column not found in CSV")
        except Exception as e:
            print(f"Error analyzing dataframe: {e}")

    # 4. Check custom stoplist
    print(f"\n[DIAGNOSE] Step 4: Checking custom stoplist: {stoplist_path}")
    if stoplist_path.exists():
        try:
            with open(stoplist_path, 'r', encoding='utf-8') as f:
                stopwords = [line.strip() for line in f if line.strip()]
            print(f"  ✓ Loaded {len(stopwords)} stopwords")
            print(f"  Sample: {stopwords[:5]}")
            
            # Check for duplicates
            if len(stopwords) != len(set(stopwords)):
                print(f"  ⚠️  Warning: {len(stopwords) - len(set(stopwords))} duplicates found")
                
            # Check case sensitivity needs
            upper_case = [w for w in stopwords if any(c.isupper() for c in w)]
            if upper_case:
                print(f"  ⚠️  Warning: {len(upper_case)} stopwords contain uppercase letters (should be normalized?)")
                print(f"  Sample uppercase: {upper_case[:5]}")
        except Exception as e:
            print(f"  ❌ Error reading stoplist: {e}")
    else:
        print(f"  ⚠️  Stoplist not found at {stoplist_path}")
    
    print("\n" + "=" * 80)
    print("[DIAGNOSE] ✓ Diagnosis completed")
    print("=" * 80)
    return True

if __name__ == "__main__":
    import sys
    from src.common.config import load_config, resolve_path
    
    # Load config to get dataset path
    config_path = Path("configs/paths.yaml")
    if config_path.exists():
        config = load_config(config_path)
        dataset_path_str = config.get("inputs", {}).get("chapters_csv", "data/processed/chapters.csv")
        dataset_path = resolve_path(Path(dataset_path_str))
    else:
        # Fallback to default
        dataset_path = Path("data/processed/chapters.csv")
    
    print(f"[DIAGNOSE] Using dataset: {dataset_path}")
    success = diagnose_data(dataset_path)
    sys.exit(0 if success else 1)

