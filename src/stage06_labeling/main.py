"""CLI entry point for generating topic labels from POS representation using Mistral-7B-Instruct."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

from src.common.config import load_config, resolve_path
from src.common.logging import setup_logging
from src.stage06_exploration.explore_retrained_model import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDING_MODEL,
)
from src.stage06_labeling.generate_labels import (
    compare_topics_sources,
    extract_pos_topics,
    extract_pos_topics_from_json,
    generate_all_labels,
    generate_labels_streaming,
    integrate_labels_to_bertopic,
    load_bertopic_model,
    load_labeling_model,
    save_labels,
)

DEFAULT_OUTPUT_DIR = Path("results/stage06_labeling")
DEFAULT_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_NUM_KEYWORDS = 15  # Increased from 8 to 15 for richer context and better labels
DEFAULT_MAX_TOKENS = 40  # Increased to 40 for more descriptive labels with clauses/parentheses
DEFAULT_BATCH_SIZE = 50


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
        description="Generate topic labels from POS representation using Mistral-7B-Instruct.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name (e.g., 'paraphrase-MiniLM-L6-v2')",
    )
    
    parser.add_argument(
        "--pareto-rank",
        type=int,
        default=1,
        help="Pareto rank of the model to load",
    )
    
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base directory containing retrained models",
    )
    
    parser.add_argument(
        "--use-native",
        action="store_true",
        help="Load native safetensors instead of pickle wrapper",
    )
    
    parser.add_argument(
        "--num-keywords",
        type=int,
        default=DEFAULT_NUM_KEYWORDS,
        help=f"Number of top keywords per topic to use for label generation (default: {DEFAULT_NUM_KEYWORDS}, recommended: 15-20 for better labels)",
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum number of tokens to generate per label (default: {DEFAULT_MAX_TOKENS})",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for labels JSON file",
    )
    
    parser.add_argument(
        "--no-integrate",
        action="store_true",
        help="Skip integrating labels back into BERTopic model (NOT RECOMMENDED - labels should be saved to model)",
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Mistral model name from Hugging Face (default: mistralai/Mistral-7B-Instruct-v0.2)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use for label generation (default: auto-detect, prefers GPU when available)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of topics to process before logging progress",
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paths.yaml"),
        help="Path to paths configuration file",
    )
    
    parser.add_argument(
        "--topics-json",
        type=Path,
        default=None,
        help="Path to topics JSON file (optional, for comparison/inspection with BERTopic topics)",
    )
    
    parser.add_argument(
        "--limit-topics",
        type=int,
        default=None,
        help="Limit number of topics to process (useful for testing)",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for topic labeling."""
    args = parse_args()
    
    # Load configuration first to get logs directory
    try:
        paths_cfg = load_config(args.config)
        outputs = paths_cfg.get("outputs", {})
        logs_dir = resolve_path(Path(outputs.get("logs", "logs")))
    except Exception as e:
        # Fallback to default logs directory if config loading fails
        logs_dir = Path("logs")
        print(f"[LABELING_CMD] Warning: Could not load config: {e}")
        print(f"[LABELING_CMD] Using default logs directory: {logs_dir}")
    
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"stage06_labeling_{timestamp}.log"
    logger = setup_logging(logs_dir, log_file=log_file)
    logger.info("=" * 80)
    logger.info("Stage 06: POS Topic Labeling with Mistral-7B-Instruct")
    logger.info("=" * 80)
    
    # Set up Tee to capture all print output to log file
    log_path = logs_dir / log_file
    stdout_tee = Tee(log_path, sys.stdout)
    stderr_tee = Tee(log_path, sys.stderr)
    
    try:
        # Set unbuffered mode for immediate output
        os.environ['PYTHONUNBUFFERED'] = '1'
        
        # Redirect stdout and stderr to Tee
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = stdout_tee
        sys.stderr = stderr_tee
        
        # Force immediate flush
        sys.stdout.flush()
        sys.stderr.flush()
        
        logger.info(f"[LABELING_CMD] Log file: {log_path}")
        logger.info(f"[LABELING_CMD] ========== Starting labeling command ==========")
        print("[LABELING_CMD] ========== Starting labeling command ==========")
        print("=" * 80)
        print("Stage 06: POS Topic Labeling with Mistral-7B-Instruct")
        print("=" * 80)
        print(f"[LABELING_CMD] Arguments:")
        print(f"[LABELING_CMD]   Embedding model: {args.embedding_model}")
        print(f"[LABELING_CMD]   Pareto rank: {args.pareto_rank}")
        print(f"[LABELING_CMD]   Base dir: {args.base_dir}")
        print(f"[LABELING_CMD]   Use native: {args.use_native}")
        print(f"[LABELING_CMD]   Num keywords: {args.num_keywords}")
        print(f"[LABELING_CMD]   Max tokens: {args.max_tokens}")
        print(f"[LABELING_CMD]   Output dir: {args.output_dir}")
        print(f"[LABELING_CMD]   Model name: {args.model_name}")
        print(f"[LABELING_CMD]   Batch size: {args.batch_size}")
        print(f"[LABELING_CMD]   No integrate: {args.no_integrate}")
        print(f"[LABELING_CMD]   Topics JSON: {args.topics_json} (for inspection/comparison)")
        print(f"[LABELING_CMD]   Limit topics: {args.limit_topics}")
        print()
        
        # Step 1: Always load BERTopic model (primary source for topics and integration)
        print("[LABELING_CMD] Step 1: Loading BERTopic model...")
        sys.stdout.flush()
        wrapper, topic_model = load_bertopic_model(
            base_dir=args.base_dir,
            embedding_model=args.embedding_model,
            pareto_rank=args.pareto_rank,
            use_native=args.use_native,
        )
        print(f"[LABELING_CMD] ✓ Loaded BERTopic model (use_native={args.use_native})")
        sys.stdout.flush()
        print()
        
        # Step 2: Extract POS topics from BERTopic model (primary source)
        print("[LABELING_CMD] Step 2: Extracting POS representation topics from BERTopic model...")
        sys.stdout.flush()
        try:
            pos_topics_dict = extract_pos_topics(
                topic_model=topic_model,
                top_k=args.num_keywords,
                limit=args.limit_topics,
            )
            limit_msg = f" (limited to {args.limit_topics})" if args.limit_topics else ""
            print(f"[LABELING_CMD] ✓ Extracted {len(pos_topics_dict)} topics from BERTopic model{limit_msg}")
            sys.stdout.flush()
        except ValueError as e:
            print(f"[LABELING_CMD] ✗ Error: {e}")
            print("\n[LABELING_CMD] Hint: Run explore_retrained_model.py with --save-topics first")
            print("[LABELING_CMD]       to generate POS representation.")
            sys.stdout.flush()
            logger.error(f"[LABELING_CMD] Failed to extract POS topics: {e}")
            return
        print()
        
        # Step 2b: Optionally load from JSON for comparison/inspection
        json_topics_iter = None
        if args.topics_json:
            topics_json_path = args.topics_json
            if topics_json_path.exists():
                print(f"[LABELING_CMD] Step 2b: Loading topics from JSON for comparison/inspection: {topics_json_path}")
                sys.stdout.flush()
                try:
                    json_topics_iter = extract_pos_topics_from_json(
                        json_path=topics_json_path,
                        top_k=args.num_keywords,
                    )
                    # Convert iterator to dict for comparison
                    json_topics_dict = dict(json_topics_iter)
                    json_topics_iter = iter(json_topics_dict.items())  # Recreate iterator for later use
                    
                    # Compare sources
                    comparison = compare_topics_sources(pos_topics_dict, json_topics_dict)
                    print(f"[LABELING_CMD] Comparison results:")
                    print(f"[LABELING_CMD]   BERTopic topics: {comparison['bertopic_topics_count']}")
                    print(f"[LABELING_CMD]   JSON topics: {comparison['json_topics_count']}")
                    print(f"[LABELING_CMD]   Common topics: {comparison['common_topics']}")
                    print(f"[LABELING_CMD]   Keyword matches: {comparison['keyword_matches']}")
                    print(f"[LABELING_CMD]   Keyword differences: {comparison['keyword_differences']}")
                    if comparison['only_in_bertopic'] > 0:
                        print(f"[LABELING_CMD]   Only in BERTopic: {comparison['only_in_bertopic']}")
                    if comparison['only_in_json'] > 0:
                        print(f"[LABELING_CMD]   Only in JSON: {comparison['only_in_json']}")
                    sys.stdout.flush()
                    logger.info(f"[LABELING_CMD] Topics comparison: {comparison}")
                except Exception as e:
                    print(f"[LABELING_CMD] ✗ Warning: Could not load/compare JSON: {e}")
                    print("[LABELING_CMD] Continuing with BERTopic topics only...")
                    sys.stdout.flush()
                    logger.warning(f"[LABELING_CMD] Failed to load JSON for comparison: {e}")
            else:
                print(f"[LABELING_CMD] Step 2b: JSON file not found: {topics_json_path}")
                print("[LABELING_CMD] Skipping JSON comparison...")
                sys.stdout.flush()
        print()
        
        # Step 3: Load Mistral model
        print("[LABELING_CMD] Step 3: Loading Mistral-7B-Instruct labeling model...")
        sys.stdout.flush()
        
        # Check and report GPU availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[LABELING_CMD] GPU detected: {gpu_name} ({gpu_memory:.2f} GB VRAM)")
            print("[LABELING_CMD] Using GPU with 4-bit quantization for efficient inference")
        else:
            print("[LABELING_CMD] ⚠️  WARNING: No GPU detected. Using CPU (will be very slow)")
        sys.stdout.flush()
        
        tokenizer, model = load_labeling_model(
            model_name=args.model_name,
            device=args.device,
            use_quantization=True,  # Use 4-bit quantization by default for memory efficiency
        )
        print(f"[LABELING_CMD] ✓ Loaded {args.model_name}")
        sys.stdout.flush()
        print()
        
        # Step 4: Generate labels from topics (use streaming if JSON available for memory efficiency)
        # Create model-specific filename
        model_name_safe = args.embedding_model.replace("/", "_").replace("\\", "_")
        labels_filename = f"labels_pos_{model_name_safe}"
        labels_path = args.output_dir / labels_filename
        
        # Use streaming mode if JSON file is provided (more memory-efficient)
        use_streaming = args.topics_json and args.topics_json.exists()
        
        if use_streaming:
            print(f"[LABELING_CMD] Step 4: Generating labels using STREAMING mode (memory-efficient, batch_size={args.batch_size})...")
            print("[LABELING_CMD]   Labels will be written incrementally to disk")
            sys.stdout.flush()
            # Recreate iterator from JSON for streaming
            pos_topics_iter = extract_pos_topics_from_json(
                json_path=args.topics_json,
                top_k=args.num_keywords,
            )
            topic_labels = generate_labels_streaming(
                pos_topics_iter=pos_topics_iter,
                tokenizer=tokenizer,
                model=model,
                output_path=labels_path,
                max_new_tokens=args.max_tokens,
                device=args.device,
                batch_size=args.batch_size,
            )
            print(f"[LABELING_CMD] ✓ Generated {len(topic_labels)} labels (streaming mode)")
            print(f"[LABELING_CMD] ✓ Labels already saved to {labels_path.with_suffix('.json')}")
            sys.stdout.flush()
            print()
        else:
            print(f"[LABELING_CMD] Step 4: Generating labels for all topics (batch_size={args.batch_size})...")
            sys.stdout.flush()
            topic_labels = generate_all_labels(
                pos_topics=pos_topics_dict,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=args.max_tokens,
                device=args.device,
                batch_size=args.batch_size,
            )
            print(f"[LABELING_CMD] ✓ Generated {len(topic_labels)} labels")
            sys.stdout.flush()
            print()
            
            # Step 5: Save labels to JSON
            print("[LABELING_CMD] Step 5: Saving labels to JSON...")
            sys.stdout.flush()
            save_labels(
                topic_labels=topic_labels,
                output_path=labels_path,
            )
            print(f"[LABELING_CMD] ✓ Saved labels to {labels_path.with_suffix('.json')}")
            sys.stdout.flush()
            print()
        
        # Step 6: Always integrate labels into BERTopic model (unless --no-integrate is set)
        if not args.no_integrate:
            print("[LABELING_CMD] Step 6: Integrating labels into BERTopic model...")
            sys.stdout.flush()
            try:
                integrate_labels_to_bertopic(
                    topic_model=topic_model,
                    topic_labels=topic_labels,
                )
                print("[LABELING_CMD] ✓ Labels integrated into BERTopic model")
                print("[LABELING_CMD]   (Labels will appear in BERTopic visualizations)")
                sys.stdout.flush()
            except Exception as e:
                print(f"[LABELING_CMD] ✗ Error: Could not integrate labels: {e}")
                print("[LABELING_CMD]   Labels are saved to JSON file but NOT in model")
                sys.stdout.flush()
                logger.error(f"[LABELING_CMD] Failed to integrate labels: {e}")
                raise  # Re-raise since integration is important
        else:
            print("[LABELING_CMD] Step 6: Skipping BERTopic integration (--no-integrate flag set)")
            print("[LABELING_CMD] ⚠️  WARNING: Labels are NOT saved to model!")
            sys.stdout.flush()
            logger.warning("[LABELING_CMD] Labels not integrated into model (--no-integrate set)")
        print()
        
        # Summary
        print("=" * 80)
        print("[LABELING_CMD] Labeling Summary")
        print("=" * 80)
        print(f"[LABELING_CMD] Topics processed: {len(topic_labels)}")
        print(f"[LABELING_CMD] Labels saved to: {labels_path.with_suffix('.json')}")
        if not args.no_integrate:
            print("[LABELING_CMD] Labels integrated into BERTopic: Yes")
        else:
            print("[LABELING_CMD] Labels integrated into BERTopic: No")
        print(f"[LABELING_CMD] Log file: {log_path}")
        print("[LABELING_CMD] ========== labeling command completed ==========")
        sys.stdout.flush()
        logger.info(f"[LABELING_CMD] ========== labeling command completed ==========")
        logger.info(f"[LABELING_CMD] Log file saved to: {log_path}")
    
    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        stdout_tee.close()
        stderr_tee.close()


if __name__ == "__main__":
    main()

