# Model Comparison Summary

## Overview
Comparison of 4 models for topic labeling on 30 topics from romance novel analysis.

## Results Summary

### ✅ **SUCCESSFUL MODELS**

#### 1. **mistralai/mistral-nemo** (BEST - 100% SUCCESS!)
- **Success Rate**: 100.0% (30/30 multi-word labels)
- **Average words per label**: 2.30
- **Keyword copying**: 0% (0 topics)
- **Quality**: Excellent - produces descriptive, meaningful labels with perfect success rate
- **Examples**:
  - "Negotiated Deal"
  - "Tongue and Breast Foreplay"
  - "Clitoral Tongue Strokes"
  - "Dinner Invitation"
  - "Unexpected Connection"
  - "Adrenaline-fueled Heart Racing"
  - "Car Makeout"
  - "Board Game Foreplay"

#### 2. **mistralai/mistral-7b-instruct:free** (EXCELLENT)
- **Success Rate**: 93.3% (28/30 multi-word labels)
- **Average words per label**: 2.80
- **Keyword copying**: 6.7% (2 topics)
- **Quality**: Excellent - produces descriptive, meaningful labels
- **Examples**:
  - "Work and Career Tasks"
  - "Tongue and Breast Foreplay"
  - "Clitoral Stimulation and Pussy"
  - "Dinner Invitation"
  - "Tender Forehead Kisses"

#### 3. **venice/uncensored:free** (GOOD)
- **Success Rate**: 76.7% (23/30 multi-word labels)
- **Average words per label**: 2.73
- **Keyword copying**: 23.3% (7 topics)
- **Quality**: Good - produces descriptive labels but has more failures
- **Examples**:
  - "Negotiating Favor Exchange"
  - "Mouth and Hip Movement"
  - "Clitoral Stimulation With Tongue"
  - "Unusual Relationship Behavior"
  - "Long Lost Love"

### ❌ **FAILED MODELS**

#### 4. **x-ai/grok-4.1-fast** (COMPLETE FAILURE)
- **Success Rate**: 0% (0/30 multi-word labels)
- **Average words per label**: 1.00
- **Keyword copying**: 100% (30 topics)
- **Issue**: Model only returns single-word labels (just copies first keyword)
- **Examples of failures**:
  - "means" (should be "Work and Career Tasks")
  - "hips" (should be "Tongue and Breast Foreplay")
  - "clit" (should be "Clitoral Stimulation")

#### 5. **deepseek/deepseek-chat-v3-0324** (COMPLETE FAILURE)
- **Success Rate**: 0% (0/30 multi-word labels)
- **Average words per label**: 1.00
- **Keyword copying**: 100% (30 topics)
- **Issue**: Model only returns single-word labels (just copies first keyword)
- **Examples of failures**:
  - "means" (should be "Work and Career Tasks")
  - "hips" (should be "Tongue and Breast Foreplay")
  - "clit" (should be "Clitoral Stimulation")

## Detailed Comparison

### Topic 0: "means, idea, promise, work, help..."
- ✅ **mistralai/mistral-7b-instruct:free**: "Work and Career Tasks"
- ✅ **venice/uncensored:free**: "Negotiating Favor Exchange"
- ❌ **x-ai/grok-4.1-fast**: "means" (keyword copy)
- ❌ **deepseek/deepseek-chat-v3-0324**: "means" (keyword copy)

### Topic 1: "hips, tongue, breasts, mouth, body..."
- ✅ **mistralai/mistral-7b-instruct:free**: "Tongue and Breast Foreplay"
- ✅ **venice/uncensored:free**: "Mouth and Hip Movement"
- ❌ **x-ai/grok-4.1-fast**: "hips" (keyword copy)
- ❌ **deepseek/deepseek-chat-v3-0324**: "hips" (keyword copy)

### Topic 10: "door, doors, knock, hallway, stairs..."
- ✅ **mistralai/mistral-7b-instruct:free**: "Doorways and Hallways"
- ❌ **venice/uncensored:free**: "door" (keyword copy)
- ❌ **x-ai/grok-4.1-fast**: "door" (keyword copy)
- ❌ **deepseek/deepseek-chat-v3-0324**: "door" (keyword copy)

## Recommendations

1. **Use mistralai/mistral-nemo** for production labeling (BEST CHOICE)
   - Perfect success rate (100.0%)
   - Excellent label quality
   - No keyword copying
   - Clean, descriptive labels

2. **Use mistralai/mistral-7b-instruct:free** as backup
   - Very high success rate (93.3%)
   - Excellent label quality
   - Free tier available

3. **Use venice/uncensored:free** as second backup
   - Good quality but lower success rate (76.7%)
   - Free tier available

3. **Do NOT use**:
   - ❌ x-ai/grok-4.1-fast (complete failure)
   - ❌ deepseek/deepseek-chat-v3-0324 (complete failure)

## Model Performance Summary

| Rank | Model | Success Rate | Avg Words | Keyword Copy | Status |
|------|-------|--------------|-----------|--------------|--------|
| 🥇 1 | mistralai/mistral-nemo | 100.0% | 2.30 | 0% | ✅ BEST |
| 🥈 2 | mistralai/mistral-7b-instruct:free | 93.3% | 2.80 | 6.7% | ✅ EXCELLENT |
| 🥉 3 | venice/uncensored:free | 76.7% | 2.73 | 23.3% | ✅ GOOD |
| 4 | x-ai/grok-4.1-fast | 0.0% | 1.00 | 100% | ❌ FAILED |
| 5 | deepseek/deepseek-chat-v3-0324 | 0.0% | 1.00 | 100% | ❌ FAILED |

## Next Steps

1. Delete failed model output files:
   - `labels_pos_openrouter_x-ai_grok-4.json`
   - `labels_pos_openrouter_deepseek_deepseek-chat-v3-0324_paraphrase-MiniLM-L6-v2.json`

2. Run full labeling with **mistralai/mistral-7b-instruct:free** for all topics

3. Generate comparison CSV/JSON if needed for manual review

