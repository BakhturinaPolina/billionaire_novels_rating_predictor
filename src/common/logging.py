"""Logging utilities."""

import logging
from pathlib import Path
from typing import Optional


def setup_logging(
    logs_dir: Path,
    log_level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging to both file and console.
    
    Args:
        logs_dir: Directory for log files
        log_level: Logging level
        log_file: Log file name (defaults to 'pipeline.log')
        
    Returns:
        Configured logger
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    if log_file is None:
        log_file = "pipeline.log"
    
    log_path = logs_dir / log_file
    
    # Create logger
    logger = logging.getLogger("pipeline")
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        "%(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

