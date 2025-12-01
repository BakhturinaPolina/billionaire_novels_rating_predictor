#!/bin/bash
# Full pipeline script for OpenRouter labeling → Category mapping → BERTopic integration
# Run this script after setting OPENROUTER_API_KEY environment variable

set -e  # Exit on error

echo "=========================================="
echo "Full Pipeline: OpenRouter → Categories → BERTopic"
echo "=========================================="
echo ""

# Check for API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY environment variable is not set"
    echo "Please set it with: export OPENROUTER_API_KEY=your_key_here"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Configuration
EMBEDDING_MODEL="paraphrase-MiniLM-L6-v2"
TOPICS_JSON="results/stage06_exploration/topics_all_representations_paraphrase-MiniLM-L6-v2.json"
OUTPUT_DIR="results/stage06_labeling_openrouter"
CATEGORY_OUTDIR="results/stage06_labeling/category_mapping"
MODEL_PATH="models/retrained/paraphrase-MiniLM-L6-v2/model_1.pkl"
MODEL_OUTPUT="models/retrained/paraphrase-MiniLM-L6-v2/model_1_with_categories.pkl"

echo "Step 1: Generating labels with OpenRouter API (using POS keywords from JSON)..."
echo "------------------------------------------------------------"
python -m src.stage06_labeling.openrouter_experiments.main_openrouter \
    --embedding-model "$EMBEDDING_MODEL" \
    --topics-json "$TOPICS_JSON" \
    --num-keywords 15 \
    --use-improved-prompts \
    --output-dir "$OUTPUT_DIR"

LABELS_JSON="$OUTPUT_DIR/labels_pos_openrouter_romance_aware_${EMBEDDING_MODEL//\//_}.json"

if [ ! -f "$LABELS_JSON" ]; then
    echo "ERROR: Labels file not found: $LABELS_JSON"
    exit 1
fi

echo ""
echo "✓ Step 1 complete: Labels saved to $LABELS_JSON"
echo ""

echo "Step 2: Mapping labels to categories..."
echo "------------------------------------------------------------"
python -m src.stage06_labeling.category_mapping.main_category_mapping \
    --labels "$LABELS_JSON" \
    --outdir "$CATEGORY_OUTDIR"

CATEGORY_PROBS="$CATEGORY_OUTDIR/topic_to_category_probs.json"

if [ ! -f "$CATEGORY_PROBS" ]; then
    echo "ERROR: Category probabilities file not found: $CATEGORY_PROBS"
    exit 1
fi

echo ""
echo "✓ Step 2 complete: Category mappings saved to $CATEGORY_PROBS"
echo ""

echo "Step 3: Integrating categories into BERTopic model..."
echo "------------------------------------------------------------"
python -m src.stage06_labeling.category_mapping.integrate_categories_to_bertopic \
    --category-probs "$CATEGORY_PROBS" \
    --labels "$LABELS_JSON" \
    --embedding-model "$EMBEDDING_MODEL" \
    --output-dir "$(dirname $MODEL_PATH)/model_1_with_categories"

if [ ! -f "$MODEL_OUTPUT" ]; then
    echo "ERROR: Updated model file not found: $MODEL_OUTPUT"
    exit 1
fi

echo ""
echo "✓ Step 3 complete: Updated model saved to $MODEL_OUTPUT"
echo ""

echo "Step 4: Validating final results..."
echo "------------------------------------------------------------"
python -m src.stage06_labeling.category_mapping.validate_assignments \
    --summary "$CATEGORY_OUTDIR/topic_to_category_summary.csv" \
    --probs "$CATEGORY_PROBS" \
    --labels "$LABELS_JSON" \
    --output "$CATEGORY_OUTDIR/validation_report_final.md"

echo ""
echo "=========================================="
echo "✓ Full pipeline complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - Labels: $LABELS_JSON"
echo "  - Categories: $CATEGORY_PROBS"
echo "  - Updated model: $MODEL_OUTPUT"
echo "  - Validation: $CATEGORY_OUTDIR/validation_report_final.md"
echo ""

