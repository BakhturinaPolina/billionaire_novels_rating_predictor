#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fix topics misclassified as Z_noise_oog using LLM prompt.

This script identifies topics that were classified as Z_noise_oog (noise)
and uses an LLM to reclassify them. Topics that are actually romance-relevant
will be reassigned to appropriate categories.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.stage06_labeling.prompts.prompts import FIX_Z_PROMPT

# OpenRouter API configuration
DEFAULT_OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
DEFAULT_OPENROUTER_MODEL = "mistralai/mistral-nemo"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((Exception,)),
    reraise=True,
)
def fix_topic_with_llm(
    topic_id: int,
    label: str,
    keywords: list[str],
    client: OpenAI,
    model_name: str,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """
    Fix a single topic using LLM prompt.
    
    Args:
        topic_id: Topic ID
        label: Topic label
        keywords: List of keywords
        client: OpenRouter OpenAI client
        model_name: Model name to use
        temperature: Sampling temperature
        
    Returns:
        Dictionary with fixed categories and metadata
    """
    # Build topic JSON for prompt
    topic_json = {
        "topic_id": topic_id,
        "label": label,
        "keywords": keywords,
        "current_category": "Z_noise_oog"
    }
    
    # Format prompt with topic JSON
    prompt = FIX_Z_PROMPT.replace("<TOPIC_JSON_HERE>", json.dumps(topic_json, indent=2))
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Call OpenRouter API
    try:
        api_start = time.perf_counter()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=500,
            temperature=temperature,
        )
        api_elapsed = time.perf_counter() - api_start
        
        if not response.choices:
            raise ValueError("Empty API response")
        
        # Parse JSON response
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON from response (might have markdown code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        result["api_time"] = api_elapsed
        result["api_tokens"] = getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') else 0
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON response for topic {topic_id}: {e}")
        print(f"Response content: {content[:500]}")
        raise
    except Exception as e:
        print(f"ERROR: API call failed for topic {topic_id}: {e}")
        raise


def load_labels(fp: Path) -> Dict:
    """Load topic labels JSON file."""
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def load_category_probs(fp: Path) -> Dict:
    """Load category probabilities JSON file."""
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict, fp: Path):
    """Save object as JSON file."""
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def identify_z_topics(category_probs: Dict) -> list[int]:
    """
    Identify topics that have only Z_noise_oog as their category.
    
    Args:
        category_probs: Dictionary mapping topic_id to category weights
        
    Returns:
        List of topic IDs that are only Z_noise_oog
    """
    z_topics = []
    
    for topic_id_str, cat_data in category_probs.items():
        # Handle both old format (direct dict) and new format (with "categories" key)
        if isinstance(cat_data, dict) and "categories" in cat_data:
            cats = cat_data["categories"]
        else:
            cats = cat_data
        
        # Check if only Z_noise_oog
        if cats and len(cats) == 1 and "Z_noise_oog" in cats:
            z_topics.append(int(topic_id_str))
    
    return sorted(z_topics)


def convert_llm_categories_to_weights(
    primary_categories: list[str],
    secondary_categories: list[str] = None,
) -> Dict[str, float]:
    """
    Convert LLM category lists to weight dictionary.
    
    Args:
        primary_categories: List of primary category codes
        secondary_categories: Optional list of secondary category codes
        
    Returns:
        Dictionary mapping category codes to weights (sums to 1.0)
    """
    if not primary_categories:
        return {"Z_noise_oog": 1.0}
    
    all_cats = primary_categories + (secondary_categories or [])
    
    if len(all_cats) == 1:
        return {all_cats[0]: 1.0}
    elif len(all_cats) == 2:
        return {all_cats[0]: 0.67, all_cats[1]: 0.33}
    else:
        # 3 categories: 0.5, 0.3, 0.2
        weights = {}
        weights[all_cats[0]] = 0.5
        weights[all_cats[1]] = 0.3
        if len(all_cats) > 2:
            weights[all_cats[2]] = 0.2
        return weights


def main():
    """Main entry point for fixing Z topics."""
    parser = argparse.ArgumentParser(
        description="Fix topics misclassified as Z_noise_oog using LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to labels JSON file",
    )
    
    parser.add_argument(
        "--category-probs",
        type=Path,
        required=True,
        help="Path to category probabilities JSON file",
    )
    
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for fixed category mappings",
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_OPENROUTER_API_KEY,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_OPENROUTER_MODEL,
        help="OpenRouter model name",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for LLM",
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of topics to process (for testing)",
    )
    
    args = parser.parse_args()
    
    # Validate API key
    if not args.api_key:
        print("ERROR: OpenRouter API key required. Set OPENROUTER_API_KEY env var or use --api-key")
        sys.exit(1)
    
    # Load data
    print(f"Loading labels from {args.labels}")
    labels = load_labels(args.labels)
    print(f"Loaded {len(labels)} topics")
    
    print(f"Loading category probabilities from {args.category_probs}")
    category_probs = load_category_probs(args.category_probs)
    print(f"Loaded category mappings for {len(category_probs)} topics")
    
    # Identify Z topics
    z_topics = identify_z_topics(category_probs)
    print(f"Found {len(z_topics)} topics classified as Z_noise_oog only")
    
    if args.limit:
        z_topics = z_topics[:args.limit]
        print(f"Limited to {len(z_topics)} topics for processing")
    
    if not z_topics:
        print("No Z topics to fix. Exiting.")
        return
    
    # Initialize OpenRouter client
    print(f"Initializing OpenRouter client (model: {args.model})")
    client = OpenAI(
        api_key=args.api_key,
        base_url=DEFAULT_OPENROUTER_BASE_URL,
        timeout=60,
    )
    
    # Process Z topics
    print(f"\nProcessing {len(z_topics)} topics...")
    fixed_count = 0
    kept_noise_count = 0
    errors = []
    
    updated_category_probs = category_probs.copy()
    fix_results = {}
    
    for idx, topic_id in enumerate(z_topics, 1):
        topic_id_str = str(topic_id)
        
        if topic_id_str not in labels:
            print(f"  [{idx}/{len(z_topics)}] Topic {topic_id}: Not found in labels, skipping")
            continue
        
        label = labels[topic_id_str].get("label", "")
        keywords = labels[topic_id_str].get("keywords", [])
        
        print(f"  [{idx}/{len(z_topics)}] Topic {topic_id}: '{label}'")
        
        try:
            result = fix_topic_with_llm(
                topic_id=topic_id,
                label=label,
                keywords=keywords,
                client=client,
                model_name=args.model,
                temperature=args.temperature,
            )
            
            fix_results[topic_id_str] = result
            
            if result.get("is_noise", True):
                print(f"    → Kept as noise (Z_noise_oog)")
                kept_noise_count += 1
            else:
                # Convert LLM categories to weights
                new_cats = convert_llm_categories_to_weights(
                    result.get("primary_categories", []),
                    result.get("secondary_categories", []),
                )
                
                # Update category mappings
                if isinstance(updated_category_probs[topic_id_str], dict) and "categories" in updated_category_probs[topic_id_str]:
                    updated_category_probs[topic_id_str]["categories"] = new_cats
                else:
                    updated_category_probs[topic_id_str] = new_cats
                
                cat_str = ", ".join(result.get("primary_categories", []))
                print(f"    → Fixed: {cat_str}")
                fixed_count += 1
            
            # Small delay to avoid rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"    → ERROR: {e}")
            errors.append((topic_id, str(e)))
    
    # Save results
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # Save updated category probabilities
    output_file = args.outdir / "topic_to_category_probs_fixed.json"
    save_json(updated_category_probs, output_file)
    print(f"\n✓ Saved fixed category mappings to {output_file}")
    
    # Save fix results
    results_file = args.outdir / "fix_z_results.json"
    save_json(fix_results, results_file)
    print(f"✓ Saved fix results to {results_file}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total Z topics processed: {len(z_topics)}")
    print(f"  Fixed (reassigned): {fixed_count}")
    print(f"  Kept as noise: {kept_noise_count}")
    print(f"  Errors: {len(errors)}")
    print(f"{'='*60}")
    
    if errors:
        print(f"\nErrors encountered:")
        for topic_id, error in errors:
            print(f"  Topic {topic_id}: {error}")


if __name__ == "__main__":
    main()

