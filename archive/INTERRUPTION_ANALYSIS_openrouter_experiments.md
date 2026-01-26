# Interruption Analysis for Label Generation in Core Module

## Summary

Analysis of interruption handling in `generate_labels_openrouter.py` and `main_openrouter.py` for label generation processes.

## Current Interruption Handling

### 1. Streaming JSON Writer (`generate_labels_streaming`)

**Location**: `generate_labels_openrouter.py:1434-1609`

**Current Behavior**:
- Uses Python context manager (`with open(...)`) for file handling
- Writes JSON incrementally with `f.flush()` after each topic
- Closes JSON with `f.write("\n}\n")` at the end of the loop
- **Issue**: If interrupted (Ctrl+C, SIGTERM, etc.), the closing brace `}` may not be written, leaving invalid JSON

**Code Structure**:
```python
with open(json_path, "w", encoding="utf-8") as f:
    f.write("{\n")
    first_item = True
    
    for topic_id, keywords in pos_topics_iter:
        # ... process topic ...
        f.write(f'  "{topic_id}": {entry_json}')
        f.flush()  # Immediate write
        
    f.write("\n}\n")  # This may not execute on interruption
    f.flush()
```

### 2. Error Handling

**Current Exception Handling**:
- `generate_label_from_keywords_openrouter` has `@retry` decorator with 3 attempts
- Catches `Exception` broadly but doesn't handle `KeyboardInterrupt` specifically
- Errors are logged but don't prevent file corruption on interruption

**Missing Signal Handlers**:
- No `signal.signal()` handlers for SIGINT/SIGTERM
- No `try/except KeyboardInterrupt` in streaming loop
- No cleanup logic to close JSON properly on interruption

## Potential Issues

### 1. Incomplete JSON Files

**Risk**: If process is interrupted during streaming:
- JSON file may be missing closing brace `}`
- File may end with trailing comma (invalid JSON)
- Partial topic entry may be incomplete

**Detection**: Check if JSON file ends with `}\n`:
```bash
tail -c 2 file.json  # Should be "}\n"
```

### 2. Log File Analysis

**Recent Log File**: `logs/stage08_llm_labeling_20251211_000807.log`
- Only 124 lines (very short)
- Ends with progress bars, no completion message
- No "Successfully generated" log entry
- **Likely interrupted run**

### 3. No Recovery Mechanism

**Current State**: 
- No code to detect/resume from incomplete JSON
- No validation of JSON completeness before loading
- No repair utility for corrupted JSON files

## Recommendations

### 1. Add Signal Handlers

```python
import signal
import sys

def signal_handler(sig, frame):
    """Handle interruption gracefully"""
    LOGGER.warning("Interruption detected, closing JSON file...")
    # Close JSON properly
    if 'json_file' in globals():
        json_file.write("\n}\n")
        json_file.flush()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
```

### 2. Add Try/Except for KeyboardInterrupt

```python
try:
    for topic_id, keywords in pos_topics_iter:
        # ... process topic ...
except KeyboardInterrupt:
    LOGGER.warning("Interrupted by user, closing JSON file...")
    f.write("\n}\n")
    f.flush()
    raise
```

### 3. Add JSON Validation

```python
def validate_json_file(json_path: Path) -> bool:
    """Check if JSON file is complete and valid"""
    try:
        with open(json_path, 'r') as f:
            content = f.read()
            if not content.rstrip().endswith('}'):
                return False
            json.loads(content)
            return True
    except (json.JSONDecodeError, FileNotFoundError):
        return False
```

### 4. Add Recovery/Resume Logic

```python
def load_partial_labels(json_path: Path) -> dict[int, dict[str, Any]]:
    """Load labels from potentially incomplete JSON file"""
    try:
        with open(json_path, 'r') as f:
            content = f.read().rstrip()
            # Try to repair if missing closing brace
            if not content.endswith('}'):
                content = content.rstrip(',\n') + '\n}'
            return json.loads(content)
    except json.JSONDecodeError as e:
        LOGGER.error("Cannot recover from corrupted JSON: %s", e)
        return {}
```

## Log Analysis Findings

### Completed Runs
- `stage08_llm_labeling_20251210_231007.log`: Completed successfully (361 topics)
- Contains "Successfully generated and saved" message
- JSON file is valid

### Potentially Interrupted Runs
- `stage08_llm_labeling_20251211_000807.log`: Only 124 lines, no completion message
- Ends abruptly with progress bars
- **Note**: This log appears to be from embedding model loading (progress bars from sentence-transformers), not from the actual labeling process
- No corresponding incomplete JSON file found

### JSON File Validation Results

**All JSON files validated** (15 files checked):
- All files are valid JSON with proper closing braces
- No incomplete or corrupted files detected
- Files range from 5 topics (test runs) to 361 topics (full runs)

**Validation Script**: `check_incomplete_json.py` can be used to verify JSON files:
```bash
python3 src/stage08_llm_labeling/openrouter_experiments/core/check_incomplete_json.py
```

## Code Locations

1. **Streaming Function**: `generate_labels_openrouter.py:1434-1609`
2. **Main Entry Point**: `main_openrouter.py:221-599`
3. **Error Handling**: `generate_labels_openrouter.py:1415-1431`
4. **File Writing**: `generate_labels_openrouter.py:1561-1576`

## Next Steps

1. ✅ **Completed**: Checked for incomplete JSON files - all are valid
2. **Recommended**: Add signal handlers to `generate_labels_streaming()` for graceful shutdown
3. **Recommended**: Add KeyboardInterrupt handling in main loop to ensure JSON closure
4. ✅ **Completed**: Created utility to validate JSON files (`check_incomplete_json.py`)
5. **Future Enhancement**: Add resume capability to continue from last processed topic

## Tools Created

1. **`check_incomplete_json.py`**: Validates all JSON files in results directory
   - Checks for proper closing braces
   - Validates JSON syntax
   - Reports topic counts
   - Identifies incomplete files

## Current Status

✅ **No incomplete JSON files found** - All 15 JSON files in `results/stage08_llm_labeling/` are valid.

⚠️ **Interruption handling not implemented** - While no files are currently corrupted, the code lacks explicit interruption handling that could prevent future issues.
