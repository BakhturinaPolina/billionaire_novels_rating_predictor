"""I/O utilities for data loading and saving."""

from pathlib import Path
from typing import Any, Optional
import pandas as pd
import pickle
import json


def load_data(
    file_path: Path,
    file_type: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Load data from file (CSV, JSON, pickle, etc.).
    
    Args:
        file_path: Path to data file
        file_type: File type (auto-detected from extension if None)
        **kwargs: Additional arguments passed to loader
        
    Returns:
        Loaded data
    """
    file_path = Path(file_path)
    
    if file_type is None:
        file_type = file_path.suffix.lower()
    
    if file_type == ".csv":
        return pd.read_csv(file_path, **kwargs)
    elif file_type in [".json", ".jsonl"]:
        with open(file_path, "r") as f:
            if file_type == ".jsonl":
                return [json.loads(line) for line in f]
            return json.load(f)
    elif file_type == ".pkl" or file_type == ".pickle":
        with open(file_path, "rb") as f:
            return pickle.load(f)
    elif file_type == ".parquet":
        return pd.read_parquet(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def save_data(
    data: Any,
    file_path: Path,
    file_type: Optional[str] = None,
    **kwargs
) -> None:
    """
    Save data to file.
    
    Args:
        data: Data to save
        file_path: Output file path
        file_type: File type (auto-detected from extension if None)
        **kwargs: Additional arguments passed to saver
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if file_type is None:
        file_type = file_path.suffix.lower()
    
    if file_type == ".csv":
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False, **kwargs)
        else:
            raise ValueError("Data must be DataFrame for CSV output")
    elif file_type == ".json":
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, **kwargs)
    elif file_type == ".pkl" or file_type == ".pickle":
        with open(file_path, "wb") as f:
            pickle.dump(data, f, **kwargs)
    elif file_type == ".parquet":
        if isinstance(data, pd.DataFrame):
            data.to_parquet(file_path, index=False, **kwargs)
        else:
            raise ValueError("Data must be DataFrame for Parquet output")
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

