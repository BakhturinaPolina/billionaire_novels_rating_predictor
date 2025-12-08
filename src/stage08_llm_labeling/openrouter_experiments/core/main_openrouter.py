"""CLI entry point for generating topic labels from POS representation using OpenRouter API.
Supports mistralai/mistral-nemo and google/gemini-2.5-flash (with reasoning)."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from src.common.config import load_config, resolve_path
from src.common.logging import setup_logging
from src.stage06_topic_exploration.explore_retrained_model import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDING_MODEL,
)
from src.stage08_llm_labeling.generate_labels import (
    compare_topics_sources,
    extract_pos_topics,
    extract_pos_topics_from_json,
    integrate_labels_to_bertopic,
    load_bertopic_model,
)
from src.stage08_llm_labeling.openrouter_experiments.core.generate_labels_openrouter import (
    DEFAULT_OPENROUTER_API_KEY,
    DEFAULT_OPENROUTER_MODEL,
    extract_representative_docs_per_topic,
    generate_all_labels,
    generate_labels_streaming,
    load_openrouter_client,
    save_labels_openrouter,
    test_openrouter_authentication,
)

DEFAULT_OUTPUT_DIR = Path("results/stage08_llm_labeling")
DEFAULT_NUM_KEYWORDS = 15
DEFAULT_MAX_TOKENS = 100  # Need tokens for label (2–6 words, ~10 tokens) + scene summary (12–25 words, ~50 tokens) + formatting (~10 tokens) = ~70 tokens, using 100 for safety
DEFAULT_BATCH_SIZE = 50
DEFAULT_TEMPERATURE = 0.35  # Balanced for natural phrasing variation without excessive randomness


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
        description="Generate topic labels from POS representation using OpenRouter API. Supports mistralai/mistral-nemo and google/gemini-2.5-flash (with reasoning).",
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
        "--model-suffix",
        type=str,
        default="_with_noise_labels",
        help="Optional suffix to append to model filename/directory (default: '_with_noise_labels')",
    )
    
    parser.add_argument(
        "--model-stage",
        type=str,
        default="stage07_topic_quality",
        help="Stage subfolder to load model from (default: 'stage07_topic_quality' to use model from stage07)",
    )
    
    parser.add_argument(
        "--num-keywords",
        type=int,
        default=DEFAULT_NUM_KEYWORDS,
        help=f"Number of top keywords per topic to use for label generation (default: {DEFAULT_NUM_KEYWORDS})",
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
        "--api-key",
        type=str,
        default=DEFAULT_OPENROUTER_API_KEY,
        help="OpenRouter API key (default: uses hardcoded key)",
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        # default=DEFAULT_OPENROUTER_MODEL,  # Commented: mistralai/mistral-nemo (primary model)
        # For reasoning experiments, use: google/gemini-2.5-flash
        default="google/gemini-2.5-flash",  # Changed for reasoning experiments
        help=f"OpenRouter model name (default: google/gemini-2.5-flash for reasoning experiments, was {DEFAULT_OPENROUTER_MODEL})",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature for generation (default: {DEFAULT_TEMPERATURE})",
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
    
    parser.add_argument(
        "--use-improved-prompts",
        action="store_true",
        help="Use romance-aware prompt with JSON output (includes label, scene_summary, categories, is_noise, rationale)",
    )
    
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["none", "low", "medium", "high"],
        default="none",
        help=(
            "Optional reasoning effort for supported models, passed via "
            "OpenRouter extra_body['reasoning']['effort']. "
            "Use 'none' to disable. Supported by models like google/gemini-2.5-flash."
        ),
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for topic labeling with OpenRouter API."""
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
    log_file = f"stage08_llm_labeling_{timestamp}.log"
    logger = setup_logging(logs_dir, log_file=log_file)
    logger.info("=" * 80)
    # logger.info("Stage 06: POS Topic Labeling with OpenRouter API (mistralai/mistral-nemo)")
    logger.info("Stage 06: POS Topic Labeling with OpenRouter API")
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
        # print("Stage 06: POS Topic Labeling with OpenRouter API (mistralai/mistral-nemo)")
        print("Stage 06: POS Topic Labeling with OpenRouter API")
        print("=" * 80)
        print(f"[LABELING_CMD] Arguments:")
        print(f"[LABELING_CMD]   Embedding model: {args.embedding_model}")
        print(f"[LABELING_CMD]   Pareto rank: {args.pareto_rank}")
        print(f"[LABELING_CMD]   Base dir: {args.base_dir}")
        print(f"[LABELING_CMD]   Use native: {args.use_native}")
        print(f"[LABELING_CMD]   Model suffix: {args.model_suffix or '(none)'}")
        print(f"[LABELING_CMD]   Model stage: {args.model_stage or '(none)'}")
        print(f"[LABELING_CMD]   Num keywords: {args.num_keywords}")
        print(f"[LABELING_CMD]   Max tokens: {args.max_tokens}")
        print(f"[LABELING_CMD]   Output dir: {args.output_dir}")
        print(f"[LABELING_CMD]   Model name: {args.model_name}")
        print(f"[LABELING_CMD]   Temperature: {args.temperature}")
        print(f"[LABELING_CMD]   Batch size: {args.batch_size}")
        print(f"[LABELING_CMD]   No integrate: {args.no_integrate}")
        print(f"[LABELING_CMD]   Topics JSON: {args.topics_json} (for inspection/comparison)")
        print(f"[LABELING_CMD]   Limit topics: {args.limit_topics}")
        print(f"[LABELING_CMD]   Use improved prompts: {args.use_improved_prompts}")
        print(f"[LABELING_CMD]   Reasoning effort: {args.reasoning_effort}")
        print()
        
        # Step 1: Always load BERTopic model (primary source for topics and integration)
        print("[LABELING_CMD] Step 1: Loading BERTopic model...")
        sys.stdout.flush()
        wrapper, topic_model = load_bertopic_model(
            base_dir=args.base_dir,
            embedding_model=args.embedding_model,
            pareto_rank=args.pareto_rank,
            use_native=args.use_native,
            model_suffix=args.model_suffix,
            stage_subfolder=args.model_stage,
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
        
        # Step 3: Initialize OpenRouter API client
        print("[LABELING_CMD] Step 3: Initializing OpenRouter API client...")
        print(f"[LABELING_CMD]   Model: {args.model_name}")
        print(f"[LABELING_CMD]   Base URL: https://openrouter.ai/api/v1")
        # Log API key status (masked for security)
        api_key_display = f"{args.api_key[:10]}...{args.api_key[-4:]}" if len(args.api_key) > 14 else "***"
        print(f"[LABELING_CMD]   API key: {api_key_display} ({'provided' if args.api_key else 'MISSING - using default'})")
        sys.stdout.flush()
        if not args.api_key or args.api_key == "":
            print("[LABELING_CMD] ⚠️  WARNING: No API key provided! Using default from environment or code.")
            print("[LABELING_CMD]   Set OPENROUTER_API_KEY environment variable or use --api-key flag")
            sys.stdout.flush()
        client, model_name = load_openrouter_client(
            api_key=args.api_key,
            model_name=args.model_name,
        )
        print(f"[LABELING_CMD] ✓ Initialized OpenRouter client for {model_name}")
        sys.stdout.flush()
        
        # Test API authentication before proceeding
        print("[LABELING_CMD] Testing API authentication...")
        sys.stdout.flush()
        auth_success = test_openrouter_authentication(client, model_name)
        if not auth_success:
            print("[LABELING_CMD] ✗ Authentication test FAILED")
            print("[LABELING_CMD]   Cannot proceed with label generation")
            print("[LABELING_CMD]   Please fix API key/account issues and try again")
            sys.stdout.flush()
            logger.error("[LABELING_CMD] Authentication test failed - aborting")
            return
        print(f"[LABELING_CMD] ✓ Authentication test passed")
        print(f"[LABELING_CMD]   Ready to generate labels via API")
        sys.stdout.flush()
        print()
        
        # Step 3b: Extract representative documents for snippets
        # Use fast get_representative_docs() method (pre-computed, typically 3-5 docs per topic)
        print("[LABELING_CMD] Step 3b: Extracting representative documents for snippets...")
        sys.stdout.flush()
        topic_to_snippets = extract_representative_docs_per_topic(
            topic_model,
            max_docs_per_topic=5,  # BERTopic typically provides 3-5 docs, cap at 5
        )
        snippets_count = len([tid for tid, docs in topic_to_snippets.items() if docs])
        avg_snippets = sum(len(docs) for docs in topic_to_snippets.values()) / max(snippets_count, 1)
        print(f"[LABELING_CMD] ✓ Extracted representative docs for {snippets_count} topics")
        print(f"[LABELING_CMD]   Average snippets per topic: {avg_snippets:.1f}")
        print(f"[LABELING_CMD]   Snippets will be included in prompts for better label precision")
        sys.stdout.flush()
        print()
        
        # Step 4: Generate labels from topics (use streaming if JSON available for memory efficiency)
        # Create model-specific filename with romance-aware suffix, model name, and reasoning effort
        model_name_safe = args.embedding_model.replace("/", "_").replace("\\", "_")
        model_name_file = model_name.replace("/", "_").replace(":", "_")
        reasoning_suffix = f"_reasoning_{args.reasoning_effort}" if args.reasoning_effort != "none" else ""
        limit_suffix = f"_limit{args.limit_topics}" if args.limit_topics else ""
        labels_filename = f"labels_pos_openrouter_{model_name_file}_romance_aware_{model_name_safe}{reasoning_suffix}{limit_suffix}"
        labels_path = args.output_dir / labels_filename
        # Debug: Log the full filename to verify it's not being truncated
        logger.info(f"[LABELING_CMD] Generated filename: {labels_filename} (length: {len(labels_filename)})")
        logger.info(f"[LABELING_CMD] Full path: {labels_path} (path length: {len(str(labels_path))})")
        
        # Use streaming mode if JSON file is provided (more memory-efficient)
        use_streaming = args.topics_json and args.topics_json.exists()
        
        if use_streaming:
            print(f"[LABELING_CMD] Step 4: Generating labels using STREAMING mode (memory-efficient, batch_size={args.batch_size})...")
            print("[LABELING_CMD]   Labels will be written incrementally to disk")
            print(f"[LABELING_CMD]   Using topics from JSON: {args.topics_json}")
            print(f"[LABELING_CMD]   Temperature: {args.temperature}")
            print(f"[LABELING_CMD]   Max tokens per label: {args.max_tokens}")
            sys.stdout.flush()
            # Recreate iterator from JSON for streaming
            pos_topics_iter = extract_pos_topics_from_json(
                json_path=args.topics_json,
                top_k=args.num_keywords,
            )
            topic_labels = generate_labels_streaming(
                pos_topics_iter=pos_topics_iter,
                client=client,
                model_name=model_name,
                output_path=labels_path,
                max_new_tokens=args.max_tokens,
                batch_size=args.batch_size,
                temperature=args.temperature,
                limit=args.limit_topics,
                use_improved_prompts=args.use_improved_prompts,
                topic_model=topic_model,
                topic_to_snippets=topic_to_snippets,
                reasoning_effort=args.reasoning_effort,
            )
            print(f"[LABELING_CMD] ✓ Generated {len(topic_labels)} labels (streaming mode)")
            json_display_path = str(labels_path.parent) + "/" + labels_path.name + ".json"
            print(f"[LABELING_CMD] ✓ Labels already saved to {json_display_path}")
            sys.stdout.flush()
            print()
        else:
            print(f"[LABELING_CMD] Step 4: Generating labels for all topics (batch_size={args.batch_size})...")
            print(f"[LABELING_CMD]   Total topics to process: {len(pos_topics_dict)}")
            print(f"[LABELING_CMD]   Temperature: {args.temperature}")
            print(f"[LABELING_CMD]   Max tokens per label: {args.max_tokens}")
            print(f"[LABELING_CMD]   Rate limit delay: 4.0s between API calls")
            sys.stdout.flush()
            topic_labels = generate_all_labels(
                pos_topics=pos_topics_dict,
                client=client,
                model_name=model_name,
                max_new_tokens=args.max_tokens,
                batch_size=args.batch_size,
                temperature=args.temperature,
                use_improved_prompts=args.use_improved_prompts,
                topic_model=topic_model,
                topic_to_snippets=topic_to_snippets,
                reasoning_effort=args.reasoning_effort,
            )
            print(f"[LABELING_CMD] ✓ Generated {len(topic_labels)} labels")
            sys.stdout.flush()
            print()
            
            # Step 5: Save labels to JSON
            print("[LABELING_CMD] Step 5: Saving labels to JSON...")
            sys.stdout.flush()
            save_labels_openrouter(
                topic_data=topic_labels,
                output_path=labels_path,
            )
            # Use manual string construction to avoid truncation in display
            json_display_path = str(labels_path.parent) + "/" + labels_path.name + ".json"
            print(f"[LABELING_CMD] ✓ Saved labels to {json_display_path}")
            sys.stdout.flush()
            print()
        
        # Step 6: Always integrate labels into BERTopic model (unless --no-integrate is set)
        if not args.no_integrate:
            print("[LABELING_CMD] Step 6: Integrating labels into BERTopic model...")
            sys.stdout.flush()
            try:
                # Extract just the labels for BERTopic integration
                labels_only: dict[int, str] = {
                    topic_id: data["label"] 
                    for topic_id, data in topic_labels.items()
                }
                integrate_labels_to_bertopic(
                    topic_model=topic_model,
                    topic_labels=labels_only,
                )
                print("[LABELING_CMD] ✓ Labels integrated into BERTopic model")
                print("[LABELING_CMD]   (Labels will appear in BERTopic visualizations)")
                sys.stdout.flush()
                
                # Save model with LLM labels to stage08 subfolder (both formats)
                print("[LABELING_CMD] Step 7: Saving model with LLM labels to stage08_llm_labeling subfolder...")
                sys.stdout.flush()
                try:
                    import pickle
                    import shutil
                    from src.stage06_topic_exploration.explore_retrained_model import backup_existing_file, stage_timer
                    
                    # Create stage subfolder path
                    stage_subfolder = Path(args.base_dir) / args.embedding_model / "stage08_llm_labeling"
                    stage_subfolder.mkdir(parents=True, exist_ok=True)
                    
                    # 1. Save as native BERTopic model (directory format)
                    native_model_dir = stage_subfolder / f"model_{args.pareto_rank}_with_llm_labels"
                    if native_model_dir.exists() and native_model_dir.is_dir():
                        shutil.rmtree(native_model_dir)
                    
                    with stage_timer(f"Saving native BERTopic model with LLM labels to {native_model_dir}"):
                        topic_model.save(str(native_model_dir))
                        logger.info("Saved native BERTopic model with LLM labels to %s", native_model_dir)
                    
                    # 2. Save as wrapper pickle (file format) - only if wrapper was loaded
                    if wrapper is not None:
                        wrapper_pickle_path = stage_subfolder / f"model_{args.pareto_rank}_with_llm_labels.pkl"
                        backup_existing_file(wrapper_pickle_path)
                        
                        with stage_timer(f"Saving wrapper with LLM labels to {wrapper_pickle_path.name}"):
                            with open(wrapper_pickle_path, "wb") as f:
                                pickle.dump(wrapper, f)
                            logger.info("Saved wrapper with LLM labels to %s", wrapper_pickle_path)
                    else:
                        logger.info("Wrapper not available (loaded native model), skipping wrapper save")
                    
                    print(f"[LABELING_CMD] ✓ Saved model to {stage_subfolder}")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"[LABELING_CMD] ⚠️  Warning: Could not save model: {e}")
                    print("[LABELING_CMD]   Labels are integrated but model was not saved")
                    sys.stdout.flush()
                    logger.warning(f"[LABELING_CMD] Failed to save model: {e}")
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
        # Extract labels count (topic_labels now contains dict with label and keywords)
        topics_count = len(topic_labels)
        print(f"[LABELING_CMD] Topics processed: {topics_count}")
        json_display_path = str(labels_path.parent) + "/" + labels_path.name + ".json"
        print(f"[LABELING_CMD] Labels saved to: {json_display_path}")
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

