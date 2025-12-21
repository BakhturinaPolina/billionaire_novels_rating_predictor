"""Shared utilities for statistical analysis scripts.

This module provides common functions for:
- Project root detection
- ID normalization
- Path discovery
- Overlap diagnostics
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd


# ID normalization regex
_ID_FLOAT_FIX_RE = re.compile(r"^([0-9]+)\.0+$")


def safe_project_root(project_root_var: Optional[Path] = None) -> Path:
    """Ensure project_root is defined, exists, and is absolute.
    
    Attempts to infer sensible location if not provided or not valid.
    
    Args:
        project_root_var: Optional project root path
        
    Returns:
        Absolute Path to project root
        
    Raises:
        RuntimeError: If no valid project root can be determined
    """
    _pr = project_root_var
    try:
        if _pr is None:
            _pr = Path(os.environ.get("PROJECT_ROOT", ""))
        if _pr in (None, "", Path("")):
            _pr = Path.cwd()
        else:
            _pr = Path(_pr)
    except Exception:
        _pr = Path.cwd()
    
    _pr = _pr.expanduser().resolve()
    
    # If not valid, try scanning for src/results
    SEARCH_MARKERS = ["src", "results"]
    
    def looks_like_root(p: Path) -> bool:
        return all((p / pth).exists() for pth in SEARCH_MARKERS)
    
    # If path is ".", doesn't exist, or doesn't have src/results, look up tree
    if not _pr.exists() or _pr == Path(".") or not looks_like_root(_pr):
        # Check up from cwd
        for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
            if looks_like_root(p):
                _pr = p
                break
        else:
            # Fallback to hardcoded known path
            KNOWN = Path("/home/polina/Documents/goodreads_romance_research_cursor/billionaire_novels_rating_predictor")
            if looks_like_root(KNOWN):
                _pr = KNOWN.resolve()
            else:
                raise RuntimeError("Could not determine a valid project_root!")
    return _pr


def normalize_id(s: pd.Series) -> pd.Series:
    """Normalize identifiers for safe joins.
    
    This function handles the output format from generate_topic_probabilities_goodreads.py,
    which may produce IDs with .0 suffixes (e.g., '60416566.0') when parquet files store
    numeric IDs as floats. This normalizes them to match the format in goodreads.csv.
    
    Handles:
    - Float strings like '60416566.0' -> '60416566' (matches script output format)
    - Whitespace trimming
    - Empty/null values
    
    Args:
        s: Series of IDs to normalize
        
    Returns:
        Normalized Series with string dtype
    """
    s = s.astype("string").str.strip()
    # Remove .0 suffix from float strings (e.g., '60416566.0' -> '60416566')
    s = s.str.replace(_ID_FLOAT_FIX_RE, r"\1", regex=True)
    s = s.mask(s.str.lower().isin(["", "nan", "none"]))
    return s


def assert_overlap(
    a: pd.Series,
    b: pd.Series,
    name_a: str,
    name_b: str,
    min_share: float = 0.5,
    logger: Optional[object] = None,
) -> tuple[set, float, float, float]:
    """Check overlap between two ID series and report coverage metrics.
    
    Args:
        a: First ID series
        b: Second ID series
        name_a: Name for first series (for logging)
        name_b: Name for second series (for logging)
        min_share: Minimum acceptable overlap share (default: 0.5)
        logger: Optional logger object (if None, uses print)
        
    Returns:
        Tuple of (overlap set, share, coverage_a, coverage_b)
        
    Raises:
        AssertionError: If zero overlap detected
    """
    log = logger.info if logger and hasattr(logger, "info") else print
    
    A = set(a.dropna().unique().tolist())
    B = set(b.dropna().unique().tolist())
    overlap = A & B
    
    # Multiple metrics for clarity
    share = len(overlap) / max(1, min(len(A), len(B)))  # Overlap relative to smaller set
    coverage_a = len(overlap) / max(1, len(A))  # How much of A is covered by B
    coverage_b = len(overlap) / max(1, len(B))  # How much of B is covered by A
    
    log(f"{name_a}: {len(A):,} unique")
    log(f"{name_b}: {len(B):,} unique")
    log(f"OVERLAP: {len(overlap):,}")
    log(f"  Overlap/min: {share:.2%} (overlap relative to smaller set)")
    log(f"  Coverage of {name_a} by {name_b}: {coverage_a:.2%}")
    log(f"  Coverage of {name_b} by {name_a}: {coverage_b:.2%}")
    
    # Show sample of actual overlapping IDs
    if overlap:
        sample_overlap = sorted(list(overlap))[:5]
        log(f"  Sample overlapping IDs: {sample_overlap}")
    
    if len(overlap) == 0:
        # Show sample values to help debug
        sample_a = sorted(list(A))[:5] if A else []
        sample_b = sorted(list(B))[:5] if B else []
        raise AssertionError(
            f"❌ Zero overlap between {name_a} and {name_b}.\n"
            f"   Sample {name_a} IDs: {sample_a}\n"
            f"   Sample {name_b} IDs: {sample_b}\n"
            f"   Fix: Check that book_id values match between datasets (may need to check ID column names or merge upstream)."
        )
    if share < min_share:
        log(f"⚠️ Low overlap (<{min_share:.0%}). This may still be a problem—inspect mismatched IDs.")
    return overlap, share, coverage_a, coverage_b


def first_existing(paths: list[Path]) -> Optional[Path]:
    """Return first existing path from list, or None if none exist.
    
    Args:
        paths: List of paths to check
        
    Returns:
        First existing path, or None
    """
    for p in paths:
        if p.exists():
            return p
    return None


def glob_first(base: Path, patterns: list[str]) -> Optional[Path]:
    """Return first matching path from glob patterns.
    
    Args:
        base: Base directory to search
        patterns: List of glob patterns
        
    Returns:
        First matching path, or None
    """
    for pat in patterns:
        matches = sorted(base.glob(pat))
        if matches:
            return matches[0]
    return None


def load_table(path: Path) -> pd.DataFrame:
    """Load CSV or Parquet file.
    
    Args:
        path: Path to file
        
    Returns:
        Loaded DataFrame
        
    Raises:
        ValueError: If file type is unsupported
    """
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")

