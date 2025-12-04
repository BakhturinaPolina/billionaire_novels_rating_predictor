#!/usr/bin/env python3
"""Calculate the cost of running topic labeling for all topics across models."""

# Pricing from OpenRouter (per million tokens)
# Source: https://openrouter.ai/mistralai/mistral-nemo
MISTRAL_NEMO_PRICING = {
    "input_per_m": 0.02,  # $0.02 per million input tokens
    "output_per_m": 0.04,  # $0.04 per million output tokens
}

# Grok pricing (need to check, but typically similar or slightly higher)
# Using approximate pricing - adjust if needed
GROK_PRICING = {
    "input_per_m": 0.02,  # Approximate, check actual pricing
    "output_per_m": 0.04,  # Approximate, check actual pricing
}

# Model context limits
MISTRAL_NEMO_CONTEXT = 128_000  # 128k tokens
GROK_CONTEXT = 128_000  # Approximate, check actual

# Topic count from logs
TOTAL_TOPICS = 368

# Models to run
MODELS = [
    "mistralai/mistral-nemo",
    "x-ai/grok-4.1-fast",
]

# Estimate token usage per request
# System prompt: ~2000 tokens (with examples)
# User prompt: ~250 tokens (keywords + snippets + formatting)
# Output: ~15 tokens (2-6 word label)
TOKENS_PER_REQUEST = {
    "system": 2000,
    "user": 250,
    "output": 15,
}

def calculate_cost(num_topics: int, models: list[str]) -> dict:
    """Calculate total cost for labeling all topics."""
    results = {}
    
    for model in models:
        # Determine pricing based on model
        if "mistral-nemo" in model:
            pricing = MISTRAL_NEMO_PRICING
            context_limit = MISTRAL_NEMO_CONTEXT
        elif "grok" in model:
            pricing = GROK_PRICING
            context_limit = GROK_CONTEXT
        else:
            # Default to mistral pricing
            pricing = MISTRAL_NEMO_PRICING
            context_limit = MISTRAL_NEMO_CONTEXT
        
        # Calculate tokens per request
        input_tokens_per_request = TOKENS_PER_REQUEST["system"] + TOKENS_PER_REQUEST["user"]
        output_tokens_per_request = TOKENS_PER_REQUEST["output"]
        
        # Total tokens for all topics
        total_input_tokens = num_topics * input_tokens_per_request
        total_output_tokens = num_topics * output_tokens_per_request
        
        # Convert to millions
        total_input_m = total_input_tokens / 1_000_000
        total_output_m = total_output_tokens / 1_000_000
        
        # Calculate costs
        input_cost = total_input_m * pricing["input_per_m"]
        output_cost = total_output_m * pricing["output_per_m"]
        total_cost = input_cost + output_cost
        
        # Check if context limit is exceeded (shouldn't be for our use case)
        exceeds_context = input_tokens_per_request > context_limit
        
        results[model] = {
            "num_topics": num_topics,
            "input_tokens_per_request": input_tokens_per_request,
            "output_tokens_per_request": output_tokens_per_request,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_input_m": total_input_m,
            "total_output_m": total_output_m,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "exceeds_context": exceeds_context,
            "context_limit": context_limit,
        }
    
    # Calculate combined total
    combined_total = sum(r["total_cost"] for r in results.values())
    
    return {
        "models": results,
        "combined_total": combined_total,
    }

def main():
    """Print cost breakdown."""
    print("=" * 80)
    print("Topic Labeling Cost Calculation")
    print("=" * 80)
    print(f"Total topics: {TOTAL_TOPICS}")
    print(f"Models: {', '.join(MODELS)}")
    print()
    
    results = calculate_cost(TOTAL_TOPICS, MODELS)
    
    for model, data in results["models"].items():
        print(f"\n{model}:")
        print(f"  Topics: {data['num_topics']}")
        print(f"  Tokens per request:")
        print(f"    Input: {data['input_tokens_per_request']:,} tokens")
        print(f"    Output: {data['output_tokens_per_request']:,} tokens")
        print(f"  Total tokens:")
        print(f"    Input: {data['total_input_tokens']:,} tokens ({data['total_input_m']:.4f}M)")
        print(f"    Output: {data['total_output_tokens']:,} tokens ({data['total_output_m']:.4f}M)")
        print(f"  Costs:")
        print(f"    Input: ${data['input_cost']:.4f}")
        print(f"    Output: ${data['output_cost']:.4f}")
        print(f"    Total: ${data['total_cost']:.4f}")
        if data['exceeds_context']:
            print(f"  ⚠️  WARNING: Input exceeds context limit of {data['context_limit']:,} tokens!")
        else:
            print(f"  ✓ Within context limit ({data['context_limit']:,} tokens)")
    
    print("\n" + "=" * 80)
    print(f"COMBINED TOTAL COST: ${results['combined_total']:.4f}")
    print("=" * 80)
    
    # Also show per-topic cost
    cost_per_topic = results['combined_total'] / TOTAL_TOPICS
    print(f"\nCost per topic (across both models): ${cost_per_topic:.6f}")
    print(f"Cost per topic per model: ${cost_per_topic / len(MODELS):.6f}")

if __name__ == "__main__":
    main()

