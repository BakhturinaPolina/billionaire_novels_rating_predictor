"""Configuration loading utilities."""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_path(config: Dict[str, Any], *keys: str, default: Optional[str] = None) -> Path:
    """
    Get nested path from config dictionary.
    
    Args:
        config: Configuration dictionary
        *keys: Nested keys to traverse
        default: Default value if key not found
        
    Returns:
        Path object
    """
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            if default is not None:
                return Path(default)
            raise KeyError(f"Key path {' -> '.join(keys)} not found in config")
    
    return Path(value)


def resolve_path(path: Path, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve relative path to absolute.
    
    Args:
        path: Path to resolve (may be relative)
        base_dir: Base directory for relative paths (defaults to project root)
        
    Returns:
        Absolute Path
    """
    if path.is_absolute():
        return path
    
    if base_dir is None:
        # Assume project root is parent of src/
        base_dir = Path(__file__).parent.parent.parent
    
    return (base_dir / path).resolve()

