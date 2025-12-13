#!/usr/bin/env python
"""Check which Hugging Face models are available via HF Inference API.

This script probes a batch of candidate HF models to determine which ones are
currently available and callable via Hugging Face's serverless Inference API.
This is useful for identifying working models for topic labeling pipelines
without playing "model-name roulette".

Usage:
    export HF_TOKEN="hf_xxx"
    python check_hf_inference_models.py [--models-file MODELS.txt] [--output-dir OUTPUT]
    
    Or from project root:
    python -m src.stage08_llm_labeling.openrouter_experiments.tools.check_hf_inference_models
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports
_project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from huggingface_hub import InferenceClient, HfApi
except ImportError:
    print("ERROR: huggingface_hub not installed. Install with: pip install huggingface_hub")
    sys.exit(1)

from src.common.config import load_config, resolve_path
from src.common.logging import setup_logging


# Default candidate models for literary/topic labeling tasks
DEFAULT_CANDIDATE_MODELS = [
    # Meta Llama models (instruction-tuned)
    "meta-llama/Llama-3-2-13B-Instruct",
    "meta-llama/Llama-3-8B-Instruct",
    "meta-llama/Llama-3-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    
    # Mistral models
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    
    # Mistral Nemo variants (good for literary tasks)
    "mistralai/Mistral-Nemo-Instruct-2407",
    "nbeerbower/mistral-nemo-gutenberg-12B-v2",
    
    # Qwen models (strong instruction following)
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    
    # Phi models (Microsoft)
    "microsoft/Phi-3-medium-4k-instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    
    # Gemma models (Google)
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    
    # Other instruction-tuned models
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "teknium/OpenHermes-2.5-Mistral-7B",
    "HuggingFaceH4/zephyr-7b-beta",
    
    # Code models (sometimes good at structured output)
    "bigcode/starcoder",
    "bigcode/starcoder2-15b",
    
    # Smaller/test models
    "gpt2",  # Basic test
    "distilgpt2",
]


def check_model(
    model_id: str,
    client: InferenceClient,
    logger: Any,
    timeout: int = 60,
    test_prompt: str = "Hello",
) -> Dict[str, Any]:
    """
    Check if a model is available via HF Inference API.
    
    Args:
        model_id: Hugging Face model identifier
        client: InferenceClient instance
        logger: Logger instance
        timeout: Request timeout in seconds
        test_prompt: Test prompt to send
        
    Returns:
        Dictionary with status information:
        - status: "OK", "NO_OUTPUT", "ERROR", or "TIMEOUT"
        - error_type: Exception type name (if error)
        - error_message: Error message (if error)
        - response_preview: First 100 chars of response (if successful)
        - response_length: Length of response (if successful)
    """
    start_time = time.time()
    logger.info(f"  Testing model: {model_id}")
    logger.debug(f"    Test prompt: '{test_prompt}'")
    
    try:
        # Attempt a minimal generation to test availability
        logger.debug(f"    Sending request to HF Inference API...")
        response = client.text_generation(
            model=model_id,
            inputs=test_prompt,
            parameters={
                "max_new_tokens": 5,  # Minimal tokens for quick test
                "temperature": 0.1,  # Low temperature for deterministic output
            },
            timeout=timeout,
        )
        
        elapsed = time.time() - start_time
        logger.debug(f"    Response received in {elapsed:.2f}s")
        
        # Extract response text
        if isinstance(response, str):
            response_text = response
        elif isinstance(response, dict):
            response_text = response.get("generated_text", "")
            if not response_text and "text" in response:
                response_text = response["text"]
        else:
            response_text = str(response)
        
        if response_text:
            preview = response_text[:100].replace("\n", " ")
            logger.info(f"  ✓ Model {model_id} is AVAILABLE")
            logger.debug(f"    Response preview: {preview}...")
            logger.debug(f"    Response length: {len(response_text)} chars")
            
            return {
                "status": "OK",
                "error_type": None,
                "error_message": None,
                "response_preview": preview,
                "response_length": len(response_text),
                "response_time_seconds": elapsed,
            }
        else:
            logger.warning(f"  ⚠ Model {model_id} responded but produced NO OUTPUT")
            return {
                "status": "NO_OUTPUT",
                "error_type": None,
                "error_message": "API responded but no text generated",
                "response_preview": None,
                "response_length": 0,
                "response_time_seconds": elapsed,
            }
            
    except TimeoutError as e:
        elapsed = time.time() - start_time
        logger.error(f"  ✗ Model {model_id} TIMEOUT after {elapsed:.2f}s")
        logger.debug(f"    Error: {type(e).__name__}: {str(e)}")
        return {
            "status": "TIMEOUT",
            "error_type": "TimeoutError",
            "error_message": str(e),
            "response_preview": None,
            "response_length": None,
            "response_time_seconds": elapsed,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        error_type = type(e).__name__
        error_msg = str(e)
        logger.error(f"  ✗ Model {model_id} ERROR: {error_type}")
        logger.debug(f"    Error message: {error_msg}")
        
        # Categorize common errors
        if "not found" in error_msg.lower() or "404" in error_msg:
            status = "NOT_FOUND"
        elif "unauthorized" in error_msg.lower() or "403" in error_msg:
            status = "UNAUTHORIZED"
        elif "rate limit" in error_msg.lower() or "429" in error_msg:
            status = "RATE_LIMITED"
        else:
            status = "ERROR"
        
        return {
            "status": status,
            "error_type": error_type,
            "error_message": error_msg,
            "response_preview": None,
            "response_length": None,
            "response_time_seconds": elapsed,
        }


def load_models_from_file(file_path: Path) -> List[str]:
    """Load model IDs from a text file (one per line)."""
    models = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):  # Skip empty lines and comments
                models.append(line)
    return models


def batch_check(
    models: List[str],
    hf_token: str,
    logger: Any,
    delay_between_requests: float = 1.0,
    timeout: int = 60,
) -> Dict[str, Dict[str, Any]]:
    """
    Check multiple models in batch.
    
    Args:
        models: List of model IDs to check
        hf_token: Hugging Face API token
        logger: Logger instance
        delay_between_requests: Seconds to wait between requests (rate limiting)
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary mapping model_id -> check result
    """
    results = {}
    client = InferenceClient(token=hf_token, timeout=timeout)
    
    total = len(models)
    logger.info(f"Checking {total} models via HF Inference API...")
    logger.info(f"Rate limiting: {delay_between_requests}s delay between requests")
    logger.info("=" * 80)
    
    for idx, model_id in enumerate(models, 1):
        logger.info(f"[{idx}/{total}] Checking {model_id}")
        
        result = check_model(model_id, client, logger, timeout=timeout)
        results[model_id] = result
        
        # Rate limiting: wait between requests (except for last one)
        if idx < total:
            logger.debug(f"Waiting {delay_between_requests}s before next request...")
            time.sleep(delay_between_requests)
        
        logger.info("")  # Blank line for readability
    
    return results


def save_results(
    results: Dict[str, Dict[str, Any]],
    output_dir: Path,
    logger: Any,
) -> None:
    """Save results to JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON (full details)
    json_path = output_dir / f"hf_inference_check_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved detailed results to: {json_path}")
    
    # Save CSV (summary)
    csv_path = output_dir / f"hf_inference_check_{timestamp}.csv"
    import csv as csv_module
    
    with open(csv_path, "w", newline="") as f:
        writer = csv_module.writer(f)
        writer.writerow([
            "model_id",
            "status",
            "error_type",
            "error_message",
            "response_preview",
            "response_length",
            "response_time_seconds",
        ])
        
        for model_id, result in results.items():
            writer.writerow([
                model_id,
                result["status"],
                result.get("error_type", ""),
                result.get("error_message", ""),
                result.get("response_preview", ""),
                result.get("response_length", ""),
                result.get("response_time_seconds", ""),
            ])
    
    logger.info(f"Saved summary CSV to: {csv_path}")
    
    # Print summary statistics
    status_counts = {}
    for result in results.values():
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total models checked: {len(results)}")
    for status, count in sorted(status_counts.items()):
        logger.info(f"  {status}: {count}")
    
    # List available models
    available = [m for m, r in results.items() if r["status"] == "OK"]
    if available:
        logger.info("")
        logger.info("AVAILABLE MODELS (status=OK):")
        for model_id in available:
            logger.info(f"  ✓ {model_id}")
    else:
        logger.warning("No models are currently available (status=OK)")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Check which Hugging Face models are available via Inference API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--models-file",
        type=Path,
        default=None,
        help="Path to text file with model IDs (one per line). If not provided, uses default list.",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save results. Defaults to results/stage08_llm_labeling/",
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between requests (rate limiting). Default: 1.0",
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds. Default: 60",
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paths.yaml"),
        help="Path to configuration file. Default: configs/paths.yaml",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level. Default: INFO",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Get HF token from environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set.")
        print("Please set it with: export HF_TOKEN='hf_xxx'")
        sys.exit(1)
    
    # Load configuration for paths
    try:
        paths_cfg = load_config(args.config)
        outputs = paths_cfg.get("outputs", {})
        logs_dir = resolve_path(Path(outputs.get("logs", "logs")))
        if args.output_dir is None:
            results_dir = resolve_path(Path(outputs.get("results", "results")))
            args.output_dir = results_dir / "stage08_llm_labeling"
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        logs_dir = Path("logs")
        if args.output_dir is None:
            args.output_dir = Path("results/stage08_llm_labeling")
    
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_level_map = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"hf_inference_check_{timestamp}.log"
    logger = setup_logging(logs_dir, log_level=log_level_map[args.log_level], log_file=log_file)
    
    logger.info("=" * 80)
    logger.info("HF Inference API Model Availability Checker")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"HF Token: {'*' * 20}...{hf_token[-4:] if len(hf_token) > 4 else '****'}")
    logger.info("")
    
    # Load model list
    if args.models_file and args.models_file.exists():
        logger.info(f"Loading models from file: {args.models_file}")
        models = load_models_from_file(args.models_file)
        logger.info(f"Loaded {len(models)} models from file")
    else:
        if args.models_file:
            logger.warning(f"Models file not found: {args.models_file}, using default list")
        else:
            logger.info("Using default candidate models list")
        models = DEFAULT_CANDIDATE_MODELS.copy()
        logger.info(f"Default list contains {len(models)} models")
    
    if not models:
        logger.error("No models to check!")
        sys.exit(1)
    
    logger.info("")
    
    # Run batch check
    try:
        results = batch_check(
            models=models,
            hf_token=hf_token,
            logger=logger,
            delay_between_requests=args.delay,
            timeout=args.timeout,
        )
        
        # Save results
        logger.info("")
        save_results(results, args.output_dir, logger)
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("Check complete!")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {type(e).__name__}: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
