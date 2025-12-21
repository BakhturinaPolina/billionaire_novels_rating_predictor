#!/usr/bin/env python3
"""Data validation and extraction for final BERTopic model (taxonomy + Radway).

This script is the entry point for Stage 10 analysis. It:
- loads the final BERTopic model with taxonomy & Radway mappings
- merges Stage 08 label metadata (label, scene summary, category tags)
- exports a topic-level lookup table used downstream to aggregate book-level proportions
- runs QA checks (missing mappings, keyword quality, confidence distribution)
- performs ID alignment diagnostics between datasets

Outputs are written to: results/stage10_correlation_analysis/data_preparation/
  - taxonomy_radway_eda/ - Topic lookup and metadata
  - diagnostics/ - ID alignment and missing books reports
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from bertopic import BERTopic
except ImportError:
    BERTopic = None  # type: ignore

# Import shared utilities
from utils import (
    assert_overlap,
    first_existing,
    load_table,
    normalize_id,
    safe_project_root,
)

# Try to import project defaults
try:
    from src.stage06_topic_exploration.explore_retrained_model import (
        DEFAULT_BASE_DIR,
        DEFAULT_EMBEDDING_MODEL,
    )
except ImportError:
    DEFAULT_BASE_DIR = None
    DEFAULT_EMBEDDING_MODEL = None


def setup_logging(output_dir: Path, log_file: str = "01_data_validation_extraction.log") -> logging.Logger:
    """Set up logging to both file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("data_validation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(output_dir / log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_labels_metadata(labels_path: Path, project_root: Optional[Path] = None) -> dict[int, dict[str, Any]]:
    """Load full metadata from Stage 08 labels JSON.
    
    Accepts both:
      - rich JSON per topic (dict with label, scene_summary, etc.)
      - simple mapping {topic_id: "label"}
    
    Args:
        labels_path: Path to labels JSON file
        project_root: Optional project root for path reconstruction
        
    Returns:
        Dictionary mapping topic_id to metadata dict
    """
    labels_path = Path(labels_path)
    
    # If path doesn't exist, try to find project root and reconstruct path
    if not labels_path.exists():
        if project_root is None:
            project_root = safe_project_root()
        
        # Extract the path components after project root
        path_str = str(labels_path)
        if "results" in path_str and "stage08_llm_labeling" in path_str:
            filename = labels_path.name
            labels_path = project_root / "results" / "stage08_llm_labeling" / filename
    
    if not labels_path.exists():
        # Fall back: try to find any labels_*.json in the same folder
        labels_dir = labels_path.parent
        
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory does not exist: {labels_dir}")
        
        # Try non-recursive glob first
        candidates = sorted(labels_dir.glob("labels_*.json"))
        
        # If no files found, try recursive search
        if not candidates:
            candidates = sorted(labels_dir.rglob("labels_*.json"))
        
        if not candidates:
            available_files = [f.name for f in labels_dir.iterdir() if f.is_file()]
            raise FileNotFoundError(
                f"No labels file found at {labels_path}\n"
                f"Directory exists: {labels_dir}\n"
                f"Available files (first 10): {available_files[:10]}"
            )
        
        # Use the newest file
        labels_path = candidates[-1]
        print(f"⚠️ LABELS_PATH not found; using newest candidate: {labels_path.name}")

    print(f"Loading labels metadata from: {labels_path}")
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata: dict[int, dict[str, Any]] = {}
    for topic_id_str, topic_data in data.items():
        topic_id = int(topic_id_str)
        if isinstance(topic_data, dict):
            metadata[topic_id] = topic_data.copy()
        else:
            metadata[topic_id] = {"label": str(topic_data)}

    print(f"✓ Loaded Stage 08 label metadata for {len(metadata)} topics")
    return metadata


def load_model(model_path: Path) -> Any:
    """Load BERTopic model from path.
    
    Args:
        model_path: Path to BERTopic model directory
        
    Returns:
        Loaded BERTopic model
        
    Raises:
        ImportError: If BERTopic is not available
        FileNotFoundError: If model path doesn't exist
    """
    if BERTopic is None:
        raise ImportError("BERTopic is not available in this environment.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    return BERTopic.load(str(model_path))


def extract_all_fields(
    model: Any,
    labels_metadata: Optional[dict[int, dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Extract topic-level table with all metadata fields.
    
    Args:
        model: BERTopic model instance
        labels_metadata: Optional Stage 08 labels metadata
        
    Returns:
        DataFrame with one row per topic
    """
    rows = []

    # topic ids (exclude -1 outlier)
    topic_ids = [tid for tid in getattr(model, "topic_representations_", {}).keys() if tid != -1]

    for topic_id in sorted(topic_ids):
        row: dict[str, Any] = {"topic_id": topic_id}

        # Keywords
        if hasattr(model, "topic_representations_") and topic_id in model.topic_representations_:
            kws = model.topic_representations_[topic_id]
            row["keywords"] = ", ".join([kw[0] for kw in kws[:10]])
            row["num_keywords"] = len(kws)
            row["all_keywords"] = [kw[0] for kw in kws]

        # Stage 08 label metadata (preferred)
        if labels_metadata and topic_id in labels_metadata:
            meta = labels_metadata[topic_id]
            row["label"] = meta.get("label")
            row["scene_summary"] = meta.get("scene_summary")
            row["primary_categories"] = ", ".join(meta.get("primary_categories", [])) or None
            row["secondary_categories"] = ", ".join(meta.get("secondary_categories", [])) or None
            row["label_is_noise"] = bool(meta.get("is_noise", False))
            row["label_rationale"] = meta.get("rationale")
        else:
            row["label"] = getattr(model, "topic_labels_", {}).get(topic_id)
            row["scene_summary"] = None
            row["primary_categories"] = None
            row["secondary_categories"] = None
            row["label_is_noise"] = None
            row["label_rationale"] = None

        # Taxonomy (Stage 2)
        tax = getattr(model, "topic_taxonomy_", {}).get(topic_id, {})
        row.update({
            "taxonomy_main_id": tax.get("main_category_id"),
            "taxonomy_main_name": tax.get("main_category_name"),
            "taxonomy_main_group": tax.get("main_category_group"),
            "taxonomy_secondary_id": tax.get("secondary_category_id"),
            "taxonomy_secondary_name": tax.get("secondary_category_name"),
            "taxonomy_secondary_group": tax.get("secondary_category_group"),
            "taxonomy_confidence": tax.get("confidence"),
            "taxonomy_is_noise": bool(tax.get("is_noise", False)),
        })

        # Radway (Stage 3)
        rad = getattr(model, "topic_radway_", {}).get(topic_id, {})
        row.update({
            "radway_main_id": rad.get("radway_main_id"),
            "radway_main_name": rad.get("radway_main_name"),
            "radway_secondary_id": rad.get("radway_secondary_id"),
            "radway_phase": rad.get("radway_phase") if rad.get("radway_phase") is not None else "NA",
            "radway_phase_name": rad.get("radway_phase_name"),
            "radway_is_none": bool(rad.get("radway_is_none", False)),
            "radway_confidence": rad.get("radway_confidence"),
            "radway_rationale": rad.get("radway_rationale"),
        })

        rows.append(row)

    return pd.DataFrame(rows)


def run_qa_checks(df_topics: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    """Run QA checks on topic data.
    
    Args:
        df_topics: Topic DataFrame
        output_dir: Output directory for reports
        logger: Logger instance
        
    Returns:
        DataFrame of topics needing review
    """
    # Basic schema sanity
    expected_cols = {
        "topic_id", "label", "taxonomy_main_id", "taxonomy_main_name", "taxonomy_main_group",
        "radway_main_id", "radway_main_name", "radway_phase_name", "radway_is_none"
    }
    missing = expected_cols - set(df_topics.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    # Duplicates
    dup = df_topics["topic_id"].duplicated().sum()
    logger.info(f"Duplicate topic_id rows: {dup}")

    # Missing mappings
    n_total = len(df_topics)
    n_tax_missing = df_topics["taxonomy_main_id"].isna().sum()
    n_rad_missing = df_topics["radway_main_id"].isna().sum()
    logger.info(f"Taxonomy missing: {n_tax_missing}/{n_total}")
    logger.info(f"Radway missing:   {n_rad_missing}/{n_total}")

    # Radway none vs function
    rad_none = (df_topics["radway_is_none"] == True).sum()
    rad_fn = (df_topics["radway_is_none"] == False).sum()
    logger.info(f"Radway function topics: {rad_fn}")
    logger.info(f"Radway none topics:     {rad_none}")

    # Keyword-quality check: many empty tokens often indicate artifacts
    def empty_ratio(all_keywords_cell) -> float:
        if isinstance(all_keywords_cell, str):
            try:
                lst = ast.literal_eval(all_keywords_cell)
            except Exception:
                return np.nan
        else:
            lst = all_keywords_cell
        if not lst:
            return np.nan
        return sum(1 for w in lst if not w) / len(lst)

    df_topics["_empty_kw_ratio"] = df_topics["all_keywords"].apply(empty_ratio)
    bad_kw = df_topics[df_topics["_empty_kw_ratio"] > 0.5].copy()

    logger.info(f"Topics with >50% empty keywords: {len(bad_kw)}")

    # Save a 'needs review' list
    needs_review = df_topics[
        df_topics["taxonomy_main_id"].isna()
        | df_topics["radway_main_id"].isna()
        | (df_topics["_empty_kw_ratio"] > 0.5)
    ][["topic_id", "label", "keywords", "taxonomy_main_name", "taxonomy_main_group",
       "radway_main_id", "radway_phase_name", "taxonomy_confidence", "radway_confidence",
       "_empty_kw_ratio"]]

    needs_review_path = output_dir / "topics_needs_review.csv"
    needs_review.to_csv(needs_review_path, index=False)
    logger.info(f"✓ Wrote: {needs_review_path}  (n={len(needs_review)})")

    return needs_review


def create_book_id_mapping(
    goodreads_path: Path,
    chapters_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Create a mapping from chapters book_id to Goodreads ID.
    
    Returns a DataFrame with columns: chapters_book_id, goodreads_id
    """
    if chapters_path is None and data_dir is not None:
        chapters_candidates = [
            data_dir / "chapters.csv",
            data_dir.parent / "raw" / "chapters.csv",
            data_dir.parent / "chapters.csv",
        ]
        chapters_path = first_existing(chapters_candidates)
    
    if chapters_path is None or not chapters_path.exists():
        return None
    
    try:
        chapters_df = pd.read_csv(chapters_path)
        goodreads_df = pd.read_csv(goodreads_path)
        
        # Check what ID columns exist
        chapters_id_col = None
        for col in ["book_id", "ID", "id"]:
            if col in chapters_df.columns:
                chapters_id_col = col
                break
        
        goodreads_id_col = None
        for col in ["ID", "goodreads_book_id", "goodreads_id"]:
            if col in goodreads_df.columns:
                goodreads_id_col = col
                break
        
        if chapters_id_col is None or goodreads_id_col is None:
            return None
        
        # Try to match by Author + Title
        if "Author" in chapters_df.columns and "Book Title" in chapters_df.columns:
            if "Author" in goodreads_df.columns and "Title" in goodreads_df.columns:
                mapping = chapters_df[[chapters_id_col, "Author", "Book Title"]].merge(
                    goodreads_df[[goodreads_id_col, "Author", "Title"]],
                    left_on=["Author", "Book Title"],
                    right_on=["Author", "Title"],
                    how="inner"
                )
                mapping = mapping[[chapters_id_col, goodreads_id_col]].rename(columns={
                    chapters_id_col: "chapters_book_id",
                    goodreads_id_col: "goodreads_id"
                })
                mapping["chapters_book_id"] = normalize_id(mapping["chapters_book_id"])
                mapping["goodreads_id"] = normalize_id(mapping["goodreads_id"])
                return mapping
        
        return None
    except Exception:
        return None


def check_id_alignment(
    meta: pd.DataFrame,
    project_root: Path,
    output_dir: Path,
    excluded_book_ids: list[str],
    logger: logging.Logger,
) -> None:
    """Check ID alignment between metadata and generated outputs."""
    results_dir = project_root / "results"
    data_dir = project_root / "data" / "processed"
    
    goodreads_path = first_existing([
        data_dir / "goodreads.csv",
        data_dir / "books_meta.csv",
        results_dir / "books_meta.csv",
    ])
    
    if goodreads_path is None:
        logger.warning("Could not find goodreads.csv for ID alignment check")
        return
    
    # Candidate outputs to check (organized structure first)
    candidate_outputs = {
        "book_topic_probs": [
            results_dir / "stage10_correlation_analysis" / "data_preparation" / "topic_probabilities" / "book_topic_probs.parquet",
            results_dir / "stage10_correlation_analysis" / "data_preparation" / "topic_probabilities" / "book_topic_probs.csv",
            results_dir / "stage10_correlation_analysis" / "data_preparation" / "book_topic_probs.parquet",
            results_dir / "stage10_correlation_analysis" / "data_preparation" / "book_topic_probs.csv",
            results_dir / "stage10_correlation_analysis" / "book_topic_probs.parquet",
            results_dir / "stage10_correlation_analysis" / "book_topic_probs.csv",
        ],
        "chapter_topic_probs": [
            results_dir / "stage10_correlation_analysis" / "data_preparation" / "topic_probabilities" / "chapter_topic_probs.parquet",
            results_dir / "stage10_correlation_analysis" / "data_preparation" / "topic_probabilities" / "chapter_topic_probs.csv",
            results_dir / "stage10_correlation_analysis" / "data_preparation" / "chapter_topic_probs.parquet",
            results_dir / "stage10_correlation_analysis" / "data_preparation" / "chapter_topic_probs.csv",
            results_dir / "stage10_correlation_analysis" / "chapter_topic_probs.parquet",
            results_dir / "stage10_correlation_analysis" / "chapter_topic_probs.csv",
        ],
        "book_category_props": [
            results_dir / "stage09_category_mapping" / "stage2_theory_driven_categories" / "book_category_proportions.parquet",
        ],
    }
    
    loaded = {}
    for name, cands in candidate_outputs.items():
        p = first_existing(cands)
        loaded[name] = p
        if p:
            logger.info(f"{name}: {p}")
    
    # Create book_id mapping if needed
    book_id_mapping = create_book_id_mapping(goodreads_path, data_dir=data_dir)
    
    # Load and compare overlaps
    reports = []
    meta_ids = meta["book_id"]
    
    for name, p in loaded.items():
        if p is None:
            continue
        
        df = load_table(p)
        
        # Determine id column
        id_col = None
        for c in ["book_id", "goodreads_book_id", "goodreads_id", "work_id", "bookId", "ID", "id"]:
            if c in df.columns:
                id_col = c
                break
        if id_col is None:
            logger.warning(f"{name}: no id column found. Columns: {df.columns.tolist()}")
            continue
        
        df_ids = normalize_id(df[id_col])
        
        # Try mapping if needed
        if book_id_mapping is not None:
            sample_df_ids = set(df_ids.dropna().head(10).tolist())
            sample_meta_ids = set(meta_ids.dropna().head(10).tolist())
            
            if not (sample_df_ids & sample_meta_ids):
                logger.info(f"  🔄 No overlap after normalization, attempting book_id mapping...")
                df_mapped = pd.DataFrame({"chapters_book_id": df_ids})
                df_mapped = df_mapped.merge(
                    book_id_mapping,
                    left_on="chapters_book_id",
                    right_on="chapters_book_id",
                    how="left"
                )
                mapped_count = df_mapped["goodreads_id"].notna().sum()
                if mapped_count > 0:
                    logger.info(f"  ✓ Mapped {mapped_count}/{len(df_mapped)} ({mapped_count/len(df_mapped)*100:.1f}%) IDs")
                    df_ids = df_mapped["goodreads_id"]
        
        logger.info(f"\n--- {name} vs goodreads/meta ---")
        overlap, share, coverage_outputs, coverage_meta = assert_overlap(
            df_ids, meta_ids, name, "goodreads/meta", min_share=0.5, logger=logger
        )
        reports.append({
            "table": name,
            "path": str(p),
            "id_col": id_col,
            "unique_ids": int(df_ids.nunique()),
            "meta_unique_ids": int(meta_ids.nunique()),
            "overlap_ids": int(len(overlap)),
            "overlap_share": float(share),
            "coverage_of_meta": float(coverage_meta),
            "coverage_of_outputs": float(coverage_outputs),
        })
    
    if reports:
        report_df = pd.DataFrame(reports).sort_values("overlap_share", ascending=False)
        
        # Save report to diagnostics subdirectory
        diagnostics_dir = results_dir / "stage10_correlation_analysis" / "data_preparation" / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        out_path = diagnostics_dir / "id_alignment_report.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(out_path, index=False)
        logger.info(f"✓ Wrote: {out_path}")
        
        # Validation summary
        min_overlap_share = report_df["overlap_share"].min()
        min_coverage_meta = report_df["coverage_of_meta"].min()
        
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)
        
        if min_coverage_meta >= 0.95:
            logger.info(f"✅ SUCCESS: All datasets cover ≥95% of metadata (coverage: {min_coverage_meta:.1%})")
        elif min_coverage_meta >= 0.90:
            logger.info(f"⚠️  WARNING: Metadata coverage is <95% (coverage: {min_coverage_meta:.1%})")
            logger.info(f"   {int((1-min_coverage_meta) * report_df['meta_unique_ids'].iloc[0])} books from metadata are missing in outputs.")
        else:
            logger.warning(f"❌ ERROR: Low metadata coverage detected (coverage: {min_coverage_meta:.1%})")
        
        if min_overlap_share >= 0.95:
            logger.info(f"✅ Internal consistency: All output datasets are internally consistent (overlap: {min_overlap_share:.1%})")
        else:
            logger.warning(f"⚠️  Internal consistency issue: Some datasets have mismatched IDs (overlap: {min_overlap_share:.1%})")


def trace_missing_books(
    meta: pd.DataFrame,
    project_root: Path,
    excluded_book_ids: list[str],
    logger: logging.Logger,
) -> None:
    """Trace missing books through pipeline."""
    results_dir = project_root / "results"
    data_dir = project_root / "data" / "processed"
    
    meta_ids_normalized = meta["book_id"]
    meta_set = set(meta_ids_normalized.dropna().unique())
    
    # Get output ids (check organized structure first, then fallback to old locations)
    book_topic_path = results_dir / "stage10_correlation_analysis" / "data_preparation" / "topic_probabilities" / "book_topic_probs.parquet"
    if not book_topic_path.exists():
        book_topic_path = results_dir / "stage10_correlation_analysis" / "data_preparation" / "book_topic_probs.parquet"
    if not book_topic_path.exists():
        book_topic_path = results_dir / "stage10_correlation_analysis" / "book_topic_probs.parquet"
    if book_topic_path.exists():
        book_topic = pd.read_parquet(book_topic_path)
        out_ids = normalize_id(book_topic["book_id"])
        out_set = set(out_ids.dropna().unique())
        
        missing_from_outputs = sorted(meta_set - out_set)
        logger.info(f"📊 Missing from outputs: {len(missing_from_outputs)} books")
        logger.info(f"   Missing IDs: {missing_from_outputs}")
        
        if missing_from_outputs:
            missing_meta_rows = meta[meta["book_id"].isin(missing_from_outputs)].copy()
            
            # Save missing books report to diagnostics subdirectory
            diagnostics_dir = results_dir / "stage10_correlation_analysis" / "data_preparation" / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            missing_report_path = diagnostics_dir / "missing_books_in_outputs.csv"
            missing_meta_rows.to_csv(missing_report_path, index=False)
            logger.info(f"✓ Saved missing books report: {missing_report_path}")
            
            # Check sentence_df coverage
            sent_path = data_dir / "sentence_df_with_topics.parquet"
            if sent_path.exists():
                sent = pd.read_parquet(sent_path, columns=["book_id"])
                sent["book_id_norm"] = normalize_id(sent["book_id"])
                sent_set = set(sent["book_id_norm"].dropna().unique())
                
                missing_in_sentence_df = sorted(set(missing_from_outputs) - sent_set)
                logger.info(f"📄 Missing in sentence_df: {len(missing_in_sentence_df)}")
                if missing_in_sentence_df:
                    logger.info(f"   IDs: {missing_in_sentence_df}")
                    logger.info("   → These books were filtered out before sentence_df creation")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Data validation and extraction for final BERTopic model"
    )
    
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to BERTopic model (auto-detected if not provided)",
    )
    
    parser.add_argument(
        "--labels-path",
        type=Path,
        help="Path to Stage 08 labels JSON (auto-detected if not provided)",
    )
    
    parser.add_argument(
        "--goodreads-path",
        type=Path,
        help="Path to goodreads.csv metadata (auto-detected if not provided)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for taxonomy_radway_eda (default: results/stage10_correlation_analysis/data_preparation/taxonomy_radway_eda)",
    )
    
    parser.add_argument(
        "--excluded-book-ids",
        type=Path,
        help="CSV file with excluded book IDs (default: excluded_book_ids.csv in script directory)",
    )
    
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Project root path (auto-detected if not provided)",
    )
    
    parser.add_argument(
        "--skip-id-alignment",
        action="store_true",
        help="Skip ID alignment diagnostics",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Determine project root
    project_root = args.project_root or safe_project_root()
    
    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use organized structure: data_preparation/taxonomy_radway_eda/
        base_dir = project_root / "results" / "stage10_correlation_analysis" / "data_preparation"
        output_dir = base_dir / "taxonomy_radway_eda"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(output_dir)
    logger.info("="*80)
    logger.info("Data Validation and Extraction")
    logger.info("="*80)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load excluded book IDs
    if args.excluded_book_ids:
        excluded_df = pd.read_csv(args.excluded_book_ids)
        excluded_book_ids = excluded_df["book_id"].astype(str).tolist()
    else:
        # Try default location
        default_excluded = SCRIPT_DIR.parent / "excluded_book_ids.csv"
        if default_excluded.exists():
            excluded_df = pd.read_csv(default_excluded)
            excluded_book_ids = excluded_df["book_id"].astype(str).tolist()
        else:
            excluded_book_ids = ['19561986', '19619918', '25781538', '52061964', '53491034']
    logger.info(f"Excluded book IDs: {excluded_book_ids}")
    
    # Determine model path
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        # Try to use defaults
        if DEFAULT_BASE_DIR and DEFAULT_EMBEDDING_MODEL:
            base_dir = DEFAULT_BASE_DIR
            embedding_model = DEFAULT_EMBEDDING_MODEL
        else:
            base_dir = project_root / "models"
            embedding_model = "paraphrase-MiniLM-L6-v2"
        
        stage_subfolder = "stage09_category_mapping"
        model_suffix = "_with_radway_mappings"
        model_path = base_dir / embedding_model / stage_subfolder / f"model_1{model_suffix}"
        
        # Also check retrained directory
        retrained_path = project_root / "models" / "retrained" / embedding_model / stage_subfolder / f"model_1{model_suffix}"
        if retrained_path.exists():
            model_path = retrained_path
    
    # Determine labels path
    if args.labels_path:
        labels_path = Path(args.labels_path)
    else:
        labels_filename = (
            "labels_pos_openrouter_mistralai_Mistral-Nemo-Instruct-2407_"
            "romance_aware_paraphrase-MiniLM-L6-v2.json"
        )
        labels_path = project_root / "results" / "stage08_llm_labeling" / labels_filename
    
    # Load labels metadata
    logger.info("\n" + "="*80)
    logger.info("Loading Stage 08 labels metadata")
    logger.info("="*80)
    labels_metadata = load_labels_metadata(labels_path, project_root)
    
    # Load model
    logger.info("\n" + "="*80)
    logger.info("Loading BERTopic model")
    logger.info("="*80)
    model = None
    tried_paths = [model_path]
    
    # Check alternate retrained location
    retrained_alt = project_root / "models" / "retrained" / "paraphrase-MiniLM-L6-v2" / "stage09_category_mapping" / "model_1_with_radway_mappings"
    if retrained_alt.exists():
        tried_paths.insert(0, retrained_alt)
    
    for candidate_path in tried_paths:
        if candidate_path.exists() and BERTopic is not None:
            logger.info(f"🔍 Trying to load BERTopic model from: {candidate_path}")
            try:
                model = load_model(candidate_path)
                logger.info(f"✓ Model loaded from: {candidate_path}")
                break
            except Exception as e:
                logger.warning(f"Failed to load model from {candidate_path}: {e}")
    
    if model is None:
        # Try to load exported CSV
        full_csv = output_dir / "full_model_data.csv"
        if full_csv.exists():
            logger.info(f"⚠️ Model not available; loading exported CSV: {full_csv}")
            df_topics = pd.read_csv(full_csv)
            logger.info(f"✓ Loaded exported table: {full_csv} ({df_topics.shape[0]} rows)")
        else:
            raise FileNotFoundError(
                f"Model not found at any checked location and no exported CSV found.\n"
                f"Checked paths: {tried_paths}\n"
                f"Expected CSV: {full_csv}"
            )
    else:
        # Extract topic-level table
        logger.info("\n" + "="*80)
        logger.info("Extracting topic-level table")
        logger.info("="*80)
        df_topics = extract_all_fields(model, labels_metadata=labels_metadata)
        logger.info(f"Extracted {len(df_topics)} topics")
    
    # Run QA checks
    logger.info("\n" + "="*80)
    logger.info("Running QA checks")
    logger.info("="*80)
    run_qa_checks(df_topics, output_dir, logger)
    
    # Export topic lookup and summary tables
    logger.info("\n" + "="*80)
    logger.info("Exporting outputs")
    logger.info("="*80)
    
    # Summary statistics
    summary = {
        "total_topics": int(len(df_topics)),
        "topics_with_labels": int(df_topics["label"].notna().sum()),
        "topics_with_taxonomy": int(df_topics["taxonomy_main_id"].notna().sum()),
        "topics_with_radway": int(df_topics["radway_main_id"].notna().sum()),
        "topics_with_radway_function": int((df_topics["radway_is_none"] == False).sum()),
        "topics_with_radway_none": int((df_topics["radway_is_none"] == True).sum()),
        "unique_taxonomy_categories": int(df_topics["taxonomy_main_name"].nunique(dropna=True)),
        "unique_taxonomy_groups": int(df_topics["taxonomy_main_group"].nunique(dropna=True)),
        "unique_radway_functions": int(df_topics.loc[df_topics["radway_is_none"] == False, "radway_main_name"].nunique(dropna=True)),
        "unique_radway_phases": int(df_topics["radway_phase_name"].nunique(dropna=True)),
    }
    summary_path = output_dir / "summary_statistics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Wrote: {summary_path}")
    
    # Topic lookup (for merging with book_topic_probs.parquet later)
    topic_lookup = df_topics[[
        "topic_id",
        "taxonomy_main_id", "taxonomy_main_name", "taxonomy_main_group",
        "taxonomy_secondary_id", "taxonomy_secondary_name", "taxonomy_secondary_group",
        "taxonomy_confidence", "taxonomy_is_noise",
        "radway_main_id", "radway_main_name", "radway_phase", "radway_phase_name", "radway_is_none", "radway_confidence",
        "label", "scene_summary", "primary_categories", "secondary_categories", "label_is_noise",
    ]].copy()
    
    topic_lookup_path = output_dir / "topic_lookup.parquet"
    topic_lookup.to_parquet(topic_lookup_path, index=False)
    logger.info(f"✓ Wrote: {topic_lookup_path}")
    
    # Also export full table
    full_csv = output_dir / "full_model_data.csv"
    full_parquet = output_dir / "full_model_data.parquet"
    df_topics.to_csv(full_csv, index=False)
    df_topics.to_parquet(full_parquet, index=False)
    logger.info(f"✓ Wrote: {full_csv}")
    logger.info(f"✓ Wrote: {full_parquet}")
    
    # Cross-tabs
    ct_group_phase = pd.crosstab(df_topics["taxonomy_main_group"], df_topics["radway_phase_name"])
    ct_group_phase.to_csv(output_dir / "crosstab_taxonomy_group_x_radway_phase.csv")
    
    ct_top = pd.crosstab(df_topics["taxonomy_main_name"], df_topics["radway_main_id"])
    ct_top.to_csv(output_dir / "crosstab_taxonomy_main_x_radway_id.csv")
    logger.info("✓ Wrote crosstab CSVs")
    
    # ID alignment diagnostics
    if not args.skip_id_alignment:
        logger.info("\n" + "="*80)
        logger.info("ID Alignment Diagnostics")
        logger.info("="*80)
        
        # Load Goodreads metadata
        if args.goodreads_path:
            goodreads_path = Path(args.goodreads_path)
        else:
            data_dir = project_root / "data" / "processed"
            results_dir = project_root / "results"
            goodreads_path = first_existing([
                data_dir / "goodreads.csv",
                data_dir / "books_meta.csv",
                results_dir / "books_meta.csv",
            ])
        
        if goodreads_path and goodreads_path.exists():
            meta = pd.read_csv(goodreads_path)
            
            # Choose Goodreads id column
            goodreads_id_col = None
            for c in ["goodreads_book_id", "book_id", "goodreads_id", "work_id", "ID", "id"]:
                if c in meta.columns:
                    goodreads_id_col = c
                    break
            if goodreads_id_col is None:
                logger.warning(f"No Goodreads id column found in {goodreads_path}")
            else:
                meta["book_id"] = normalize_id(meta[goodreads_id_col])
                
                # Apply cohort exclusion
                before = meta["book_id"].nunique()
                meta = meta[~meta["book_id"].astype("string").isin(excluded_book_ids)].copy()
                after = meta["book_id"].nunique()
                logger.info(f"Cohort meta unique book_id: {before} → {after} (excluded {before-after})")
                
                # Check ID alignment
                check_id_alignment(meta, project_root, output_dir, excluded_book_ids, logger)
                
                # Trace missing books
                trace_missing_books(meta, project_root, excluded_book_ids, logger)
        else:
            logger.warning("Could not find goodreads.csv for ID alignment check")
    
    logger.info("\n" + "="*80)
    logger.info("✓ Data validation and extraction complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()

