"""CLI entry point for mapping topic labels to theory-aligned categories."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from src.common.config import load_config, resolve_path
from src.common.logging import setup_logging
from src.stage06_labeling.category_mapping.map_topics_to_categories import (
    CATS,
    aggregate_book_props,
    compute_indices,
    infer_categories,
    load_book_topic_probs,
    load_labels,
    save_csv,
    save_json,
)

DEFAULT_OUTPUT_DIR = Path("results/stage06_labeling/category_mapping")


class Tee:
    """Write to both file and stdout/stderr with immediate flushing."""
    def __init__(self, file_path: Path, stream):
        # Open file in line-buffered mode for immediate flushing
        self.file = open(file_path, 'w', encoding='utf-8', buffering=1)  # Line buffering
        self.stream = stream
    
    def write(self, data):
        self.file.write(data)
        self.stream.write(data)
        # Force immediate flush for both
        self.file.flush()
        self.stream.flush()
    
    def flush(self):
        self.file.flush()
        self.stream.flush()
    
    def close(self):
        self.flush()
        self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Map topic labels to theory-aligned categories (A-P, Q, R, S)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to labels JSON file (e.g., labels_pos_openrouter_romance_aware_paraphrase-MiniLM-L6-v2.json)",
    )
    
    parser.add_argument(
        "--book-topic-probs",
        type=Path,
        default=None,
        help="Optional: Path to book-topic probability CSV (columns: book_id, topic_id, prob)",
    )
    
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for category mapping files",
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to paths configuration file (optional)",
    )
    
    parser.add_argument(
        "--fix-z",
        action="store_true",
        help="Run LLM-based fix for topics misclassified as Z_noise_oog (requires OPENROUTER_API_KEY)",
    )
    
    return parser.parse_args()


def main():
    """Main entry point for category mapping CLI."""
    args = parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"stage06_category_mapping_{timestamp}.log"
    
    # Redirect stdout/stderr to log file and console
    with Tee(log_file, sys.stdout) as tee_stdout, Tee(log_file, sys.stderr) as tee_stderr:
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr
        
        try:
            print(f"[CATEGORY_MAPPING] Starting category mapping stage")
            print(f"[CATEGORY_MAPPING] Labels file: {args.labels}")
            print(f"[CATEGORY_MAPPING] Output directory: {args.outdir}")
            
            # Resolve paths using config if provided
            if args.config:
                config = load_config(args.config)
                args.labels = resolve_path(args.labels, config)
                args.outdir = resolve_path(args.outdir, config)
                if args.book_topic_probs:
                    args.book_topic_probs = resolve_path(args.book_topic_probs, config)
            
            # Validate inputs
            if not args.labels.exists():
                raise FileNotFoundError(f"Labels file not found: {args.labels}")
            
            if args.book_topic_probs and not args.book_topic_probs.exists():
                raise FileNotFoundError(f"Book-topic-probs file not found: {args.book_topic_probs}")
            
            # Load labels
            print(f"[CATEGORY_MAPPING] Loading labels from {args.labels}")
            topics = load_labels(args.labels)
            print(f"[CATEGORY_MAPPING] Loaded {len(topics)} topics")
            
            # Map topics to categories
            print(f"[CATEGORY_MAPPING] Mapping topics to categories...")
            topic_to_cat = {}
            rows = []
            
            for tid, rec in topics.items():
                label = rec.get("label", "")
                keywords = rec.get("keywords", [])
                # Ensure model uses both label and keywords for inference
                cats = infer_categories(label, keywords=keywords)
                
                # Store with keywords for JSON output
                topic_to_cat[str(tid)] = {
                    "categories": cats,
                    "label": label,
                    "keywords": keywords
                }
                
                # Flatten for CSV (include keywords as comma-separated string)
                keywords_str = ", ".join(keywords) if keywords else ""
                
                # Find primary category (highest weight)
                primary_cat = None
                if cats:
                    primary_cat = max(cats.items(), key=lambda x: x[1])[0]
                
                if not cats:
                    rows.append({
                        "topic_id": tid, 
                        "label": label, 
                        "keywords": keywords_str,
                        "category": "", 
                        "weight": "0.000000",
                        "is_primary": "False"
                    })
                else:
                    for c, w in cats.items():
                        rows.append({
                            "topic_id": tid, 
                            "label": label, 
                            "keywords": keywords_str,
                            "category": c, 
                            "weight": f"{w:.6f}",
                            "is_primary": "True" if c == primary_cat else "False"
                        })
            
            # Create output directory
            args.outdir.mkdir(parents=True, exist_ok=True)
            
            # Save outputs
            print(f"[CATEGORY_MAPPING] Saving topic-to-category mappings...")
            save_json(topic_to_cat, args.outdir / "topic_to_category_probs.json")
            save_csv(rows, args.outdir / "topic_to_category_final.csv",
                    fieldnames=["topic_id", "label", "keywords", "category", "weight", "is_primary"])
            
            # Also create a summary CSV with one row per topic (primary category only)
            summary_rows = []
            for tid, rec in topics.items():
                label = rec.get("label", "")
                keywords = rec.get("keywords", [])
                keywords_str = ", ".join(keywords) if keywords else ""
                
                # Extract categories from topic_to_cat structure
                cat_data = topic_to_cat.get(str(tid), {})
                if isinstance(cat_data, dict) and "categories" in cat_data:
                    cats = cat_data["categories"]
                else:
                    # Fallback: cat_data might be the categories dict directly
                    cats = cat_data if isinstance(cat_data, dict) and all(isinstance(v, (int, float)) for v in cat_data.values()) else {}
                
                if cats:
                    # Get primary category
                    primary_cat = max(cats.items(), key=lambda x: x[1])[0]
                    primary_weight = cats[primary_cat]
                    # Get all categories as comma-separated string
                    all_cats = ", ".join([f"{c}({w:.3f})" for c, w in sorted(cats.items(), key=lambda x: x[1], reverse=True)])
                else:
                    primary_cat = ""
                    primary_weight = 0.0
                    all_cats = ""
                
                summary_rows.append({
                    "topic_id": tid,
                    "label": label,
                    "keywords": keywords_str,
                    "primary_category": primary_cat,
                    "primary_weight": f"{primary_weight:.6f}",
                    "all_categories": all_cats
                })
            
            save_csv(summary_rows, args.outdir / "topic_to_category_summary.csv",
                    fieldnames=["topic_id", "label", "keywords", "primary_category", "primary_weight", "all_categories"])
            
            print(f"[CATEGORY_MAPPING] ✓ Saved topic_to_category_probs.json")
            print(f"[CATEGORY_MAPPING] ✓ Saved topic_to_category_final.csv (long format)")
            print(f"[CATEGORY_MAPPING] ✓ Saved topic_to_category_summary.csv (one row per topic)")
            
            # Optional: aggregate to book-level and compute indices
            if args.book_topic_probs:
                print(f"[CATEGORY_MAPPING] Loading book-topic probabilities from {args.book_topic_probs}")
                book_rows = load_book_topic_probs(args.book_topic_probs)
                print(f"[CATEGORY_MAPPING] Loaded {len(book_rows)} book-topic probability rows")
                
                print(f"[CATEGORY_MAPPING] Aggregating to book-level category proportions...")
                # Extract just the categories dict for aggregation (backward compatible)
                topic_to_cat_simple = {
                    tid: data.get("categories", data) if isinstance(data, dict) and "categories" in data else data
                    for tid, data in topic_to_cat.items()
                }
                agg = aggregate_book_props(book_rows, topic_to_cat_simple)
                print(f"[CATEGORY_MAPPING] Aggregated {len(agg)} books")
                
                # Save book category props
                b_rows = []
                for book, vec in agg.items():
                    for c in CATS:
                        b_rows.append({"book_id": book, "category": c, "value": f"{vec.get(c,0.0):.6f}"})
                save_csv(b_rows, args.outdir / "book_category_props.csv",
                         fieldnames=["book_id", "category", "value"])
                print(f"[CATEGORY_MAPPING] ✓ Saved book_category_props.csv")
                
                # Compute and save indices
                print(f"[CATEGORY_MAPPING] Computing derived indices...")
                i_rows = []
                for book, vec in agg.items():
                    idx = compute_indices(vec)
                    i_rows.append({"book_id": book, **{k: f"{v:.6f}" for k, v in idx.items()}})
                
                fns = ["book_id", "Love_over_Sex", "HEA_Index", "Explicitness_Ratio", "Luxury_Saturation",
                       "Corporate_Frame_Share", "Family_Fertility_Index", "Comms_Density",
                       "Dark_vs_Tender", "Miscommunication_Balance", "Protective_minus_Jealousy"]
                save_csv(i_rows, args.outdir / "indices_book.csv", fieldnames=fns)
                print(f"[CATEGORY_MAPPING] ✓ Saved indices_book.csv")
            
            # Optional: Fix Z topics using LLM
            if args.fix_z:
                print(f"\n[CATEGORY_MAPPING] Running LLM-based fix for Z_noise_oog topics...")
                try:
                    from src.stage06_labeling.category_mapping.fix_z_topics import (
                        fix_topic_with_llm,
                        identify_z_topics,
                        convert_llm_categories_to_weights,
                    )
                    from openai import OpenAI
                    import os
                    
                    api_key = os.environ.get("OPENROUTER_API_KEY", "")
                    if not api_key:
                        print("[CATEGORY_MAPPING] WARNING: OPENROUTER_API_KEY not set. Skipping fix Z step.")
                    else:
                        # Identify Z topics
                        z_topics = identify_z_topics(topic_to_cat)
                        print(f"[CATEGORY_MAPPING] Found {len(z_topics)} topics classified as Z_noise_oog only")
                        
                        if z_topics:
                            # Initialize OpenRouter client
                            client = OpenAI(
                                api_key=api_key,
                                base_url="https://openrouter.ai/api/v1",
                                timeout=60,
                            )
                            
                            # Process Z topics
                            fixed_count = 0
                            for topic_id in z_topics[:10]:  # Limit to 10 for now
                                topic_id_str = str(topic_id)
                                if topic_id_str not in topics:
                                    continue
                                
                                label = topics[topic_id_str].get("label", "")
                                keywords = topics[topic_id_str].get("keywords", [])
                                
                                try:
                                    result = fix_topic_with_llm(
                                        topic_id=topic_id,
                                        label=label,
                                        keywords=keywords,
                                        client=client,
                                        model_name="mistralai/mistral-nemo",
                                        temperature=0.3,
                                    )
                                    
                                    if not result.get("is_noise", True):
                                        # Update category mappings
                                        new_cats = convert_llm_categories_to_weights(
                                            result.get("primary_categories", []),
                                            result.get("secondary_categories", []),
                                        )
                                        
                                        if isinstance(topic_to_cat[topic_id_str], dict) and "categories" in topic_to_cat[topic_id_str]:
                                            topic_to_cat[topic_id_str]["categories"] = new_cats
                                        else:
                                            topic_to_cat[topic_id_str] = new_cats
                                        
                                        fixed_count += 1
                                        print(f"[CATEGORY_MAPPING] Fixed topic {topic_id}: {result.get('primary_categories', [])}")
                                    
                                    import time
                                    time.sleep(0.5)  # Rate limiting
                                    
                                except Exception as e:
                                    print(f"[CATEGORY_MAPPING] ERROR fixing topic {topic_id}: {e}")
                            
                            if fixed_count > 0:
                                # Save updated mappings
                                save_json(topic_to_cat, args.outdir / "topic_to_category_probs.json")
                                print(f"[CATEGORY_MAPPING] ✓ Updated category mappings ({fixed_count} topics fixed)")
                        else:
                            print(f"[CATEGORY_MAPPING] No Z topics to fix")
                            
                except ImportError as e:
                    print(f"[CATEGORY_MAPPING] WARNING: Could not import fix_z_topics: {e}")
                    print("[CATEGORY_MAPPING] Skipping fix Z step")
                except Exception as e:
                    print(f"[CATEGORY_MAPPING] ERROR in fix Z step: {e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"[CATEGORY_MAPPING] ✓ Category mapping complete!")
            print(f"[CATEGORY_MAPPING] Output directory: {args.outdir}")
            
        except Exception as e:
            print(f"[CATEGORY_MAPPING] ERROR: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        finally:
            # Restore stdout/stderr
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()

