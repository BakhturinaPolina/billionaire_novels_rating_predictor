"""Exploratory Data Analysis (EDA) for BERTopic model with Taxonomy and Radway mappings.

This script provides comprehensive EDA for the final model combining:
- Taxonomy mappings (Stage 2: Theory-Driven Categories)
- Radway narrative function mappings (Stage 3: Narrative Functions)
- Topic representations and labels
- Statistical summaries and visualizations
- Cross-tabulations between taxonomy categories and Radway functions

Located in Stage 10 (Correlation Analysis) as it analyzes relationships between
both Stage 2 and Stage 3 classification systems.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bertopic import BERTopic

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.stage06_topic_exploration.explore_retrained_model import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    load_native_bertopic_model,
)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def load_model_with_radway(
    base_dir: Path = DEFAULT_BASE_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    model_suffix: str = "_with_radway_mappings",
    stage_subfolder: str = "stage09_category_mapping",
) -> BERTopic:
    """Load the final model with Radway mappings."""
    model_path = base_dir / embedding_model / stage_subfolder / f"model_1{model_suffix}"
    print(f"Loading model from: {model_path}")
    return BERTopic.load(str(model_path))


def extract_all_fields(model: BERTopic) -> pd.DataFrame:
    """Extract all available fields from the model into a DataFrame."""
    rows = []

    # Get topic IDs (exclude outlier -1)
    topic_ids = [
        tid
        for tid in model.topic_representations_.keys()
        if tid != -1
    ]

    for topic_id in sorted(topic_ids):
        row = {"topic_id": topic_id}

        # Topic representations (keywords)
        if hasattr(model, "topic_representations_") and topic_id in model.topic_representations_:
            keywords = model.topic_representations_[topic_id]
            row["keywords"] = ", ".join([kw[0] for kw in keywords[:10]])  # Top 10 keywords
            row["num_keywords"] = len(keywords)

        # Custom labels (use topic_labels_ which is a dict)
        if hasattr(model, "topic_labels_") and model.topic_labels_:
            row["label"] = model.topic_labels_.get(topic_id, None)
        elif hasattr(model, "custom_labels_") and model.custom_labels_:
            if isinstance(model.custom_labels_, list):
                # List format: index corresponds to topic_id (with -1 at index 0)
                label_idx = topic_id + 1 if topic_id >= 0 else 0
                row["label"] = model.custom_labels_[label_idx] if label_idx < len(model.custom_labels_) else None
            elif isinstance(model.custom_labels_, dict):
                row["label"] = model.custom_labels_.get(topic_id, None)
            else:
                row["label"] = None
        else:
            row["label"] = None

        # Taxonomy mappings (Stage 2)
        if hasattr(model, "topic_taxonomy_") and model.topic_taxonomy_:
            taxonomy = model.topic_taxonomy_.get(topic_id, {})
            row.update({
                "taxonomy_main_id": taxonomy.get("main_category_id"),
                "taxonomy_main_name": taxonomy.get("main_category_name"),
                "taxonomy_main_group": taxonomy.get("main_category_group"),
                "taxonomy_secondary_id": taxonomy.get("secondary_category_id"),
                "taxonomy_secondary_name": taxonomy.get("secondary_category_name"),
                "taxonomy_secondary_group": taxonomy.get("secondary_category_group"),
                "taxonomy_confidence": taxonomy.get("confidence"),
                "taxonomy_is_noise": taxonomy.get("is_noise", False),
            })
        else:
            row.update({
                "taxonomy_main_id": None,
                "taxonomy_main_name": None,
                "taxonomy_main_group": None,
                "taxonomy_secondary_id": None,
                "taxonomy_secondary_name": None,
                "taxonomy_secondary_group": None,
                "taxonomy_confidence": None,
                "taxonomy_is_noise": None,
            })

        # Radway mappings (Stage 3)
        if hasattr(model, "topic_radway_") and model.topic_radway_:
            radway = model.topic_radway_.get(topic_id, {})
            radway_phase = radway.get("radway_phase")
            # Ensure "NA" is preserved as string, not None
            if radway_phase is None:
                radway_phase = "NA"
            row.update({
                "radway_main_id": radway.get("radway_main_id"),
                "radway_main_name": radway.get("radway_main_name"),
                "radway_secondary_id": radway.get("radway_secondary_id"),
                "radway_phase": radway_phase,
                "radway_phase_name": radway.get("radway_phase_name"),
                "radway_is_none": radway.get("radway_is_none", False),
                "radway_confidence": radway.get("radway_confidence"),
                "radway_rationale": radway.get("radway_rationale"),
            })
        else:
            row.update({
                "radway_main_id": None,
                "radway_main_name": None,
                "radway_secondary_id": None,
                "radway_phase": "NA",  # Use "NA" string instead of None for consistency
                "radway_phase_name": None,
                "radway_is_none": None,
                "radway_confidence": None,
                "radway_rationale": None,
            })

        rows.append(row)

    return pd.DataFrame(rows)


def analyze_taxonomy_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze distribution of taxonomy categories."""
    print("\n" + "=" * 80)
    print("TAXONOMY DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Main category distribution
    print("\n--- Main Category Distribution ---")
    main_cat_counts = df["taxonomy_main_name"].value_counts()
    print(main_cat_counts.head(20))

    # Main category group distribution
    print("\n--- Main Category Group Distribution ---")
    main_group_counts = df["taxonomy_main_group"].value_counts()
    print(main_group_counts)

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top 15 main categories
    top_main = df["taxonomy_main_name"].value_counts().head(15)
    axes[0, 0].barh(range(len(top_main)), top_main.values)
    axes[0, 0].set_yticks(range(len(top_main)))
    axes[0, 0].set_yticklabels(top_main.index, fontsize=8)
    axes[0, 0].set_xlabel("Count")
    axes[0, 0].set_title("Top 15 Main Categories")
    axes[0, 0].invert_yaxis()

    # Main category groups
    main_group_counts.plot(kind="bar", ax=axes[0, 1], color="steelblue")
    axes[0, 1].set_xlabel("Category Group")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Distribution by Category Group")
    axes[0, 1].tick_params(axis="x", rotation=45, labelsize=8)

    # Confidence distribution
    conf_counts = df["taxonomy_confidence"].value_counts()
    conf_counts.plot(kind="pie", ax=axes[1, 0], autopct="%1.1f%%")
    axes[1, 0].set_title("Taxonomy Confidence Distribution")
    axes[1, 0].set_ylabel("")

    # Noise vs non-noise
    noise_counts = df["taxonomy_is_noise"].value_counts()
    noise_counts.plot(kind="bar", ax=axes[1, 1], color=["coral", "lightblue"])
    axes[1, 1].set_xlabel("Is Noise")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Noise vs Non-Noise Topics")
    axes[1, 1].set_xticklabels(["Non-Noise", "Noise"], rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / "taxonomy_distribution.png", dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved taxonomy distribution plot to {output_dir / 'taxonomy_distribution.png'}")
    plt.close()


def analyze_radway_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze distribution of Radway narrative functions."""
    print("\n" + "=" * 80)
    print("RADWAY NARRATIVE FUNCTION DISTRIBUTION")
    print("=" * 80)

    # Main Radway function distribution
    print("\n--- Main Radway Function Distribution ---")
    radway_main_counts = df["radway_main_name"].value_counts()
    print(radway_main_counts)

    # Phase distribution
    print("\n--- Phase Distribution ---")
    phase_counts = df["radway_phase_name"].value_counts()
    print(phase_counts)

    # Confidence distribution
    print("\n--- Radway Confidence Distribution ---")
    radway_conf_counts = df["radway_confidence"].value_counts()
    print(radway_conf_counts)

    # None vs function distribution
    print("\n--- None vs Function Distribution ---")
    none_counts = df["radway_is_none"].value_counts()
    print(none_counts)

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Main Radway functions (excluding "none")
    radway_without_none = df[df["radway_is_none"] == False]["radway_main_name"].value_counts()
    axes[0, 0].barh(range(len(radway_without_none)), radway_without_none.values)
    axes[0, 0].set_yticks(range(len(radway_without_none)))
    axes[0, 0].set_yticklabels(radway_without_none.index, fontsize=8)
    axes[0, 0].set_xlabel("Count")
    axes[0, 0].set_title("Radway Functions Distribution (excluding 'none')")
    axes[0, 0].invert_yaxis()

    # Phase distribution
    phase_counts.plot(kind="bar", ax=axes[0, 1], color="steelblue")
    axes[0, 1].set_xlabel("Phase")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Distribution by Narrative Phase")
    axes[0, 1].tick_params(axis="x", rotation=45, labelsize=8)

    # Confidence distribution
    radway_conf_counts.plot(kind="pie", ax=axes[1, 0], autopct="%1.1f%%")
    axes[1, 0].set_title("Radway Confidence Distribution")
    axes[1, 0].set_ylabel("")

    # None vs function
    none_counts.plot(kind="bar", ax=axes[1, 1], color=["coral", "lightgreen"])
    axes[1, 1].set_xlabel("Is None")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("None vs Narrative Function")
    axes[1, 1].set_xticklabels(["Function", "None"], rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / "radway_distribution.png", dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved Radway distribution plot to {output_dir / 'radway_distribution.png'}")
    plt.close()


def analyze_cross_tabulations(df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze cross-tabulations between taxonomy and Radway."""
    print("\n" + "=" * 80)
    print("CROSS-TABULATION ANALYSIS")
    print("=" * 80)

    # Taxonomy group vs Radway phase
    print("\n--- Taxonomy Group vs Radway Phase ---")
    crosstab_group_phase = pd.crosstab(
        df["taxonomy_main_group"],
        df["radway_phase_name"],
        margins=True,
    )
    print(crosstab_group_phase)

    # Taxonomy main category vs Radway function (top categories only)
    print("\n--- Top Taxonomy Categories vs Radway Functions (top 10) ---")
    top_taxonomy = df["taxonomy_main_name"].value_counts().head(10).index
    df_top_tax = df[df["taxonomy_main_name"].isin(top_taxonomy)]
    crosstab_cat_func = pd.crosstab(
        df_top_tax["taxonomy_main_name"],
        df_top_tax["radway_main_name"],
        margins=True,
    )
    print(crosstab_cat_func)

    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Taxonomy group vs Radway phase heatmap
    crosstab_group_phase_plot = pd.crosstab(
        df["taxonomy_main_group"],
        df["radway_phase_name"],
    )
    sns.heatmap(
        crosstab_group_phase_plot,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        ax=axes[0],
        cbar_kws={"label": "Count"},
    )
    axes[0].set_title("Taxonomy Group vs Radway Phase")
    axes[0].set_xlabel("Radway Phase")
    axes[0].set_ylabel("Taxonomy Group")
    axes[0].tick_params(axis="x", rotation=45, labelsize=8)
    axes[0].tick_params(axis="y", rotation=0, labelsize=8)

    # Top taxonomy categories vs Radway functions (top 10 functions)
    top_radway = df[df["radway_is_none"] == False]["radway_main_name"].value_counts().head(10).index
    df_filtered = df[
        (df["taxonomy_main_name"].isin(top_taxonomy))
        & (df["radway_main_name"].isin(top_radway))
    ]
    if not df_filtered.empty:
        crosstab_cat_func_plot = pd.crosstab(
            df_filtered["taxonomy_main_name"],
            df_filtered["radway_main_name"],
        )
        sns.heatmap(
            crosstab_cat_func_plot,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            ax=axes[1],
            cbar_kws={"label": "Count"},
        )
        axes[1].set_title("Top Taxonomy Categories vs Top Radway Functions")
        axes[1].set_xlabel("Radway Function")
        axes[1].set_ylabel("Taxonomy Category")
        axes[1].tick_params(axis="x", rotation=45, labelsize=7)
        axes[1].tick_params(axis="y", rotation=0, labelsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / "cross_tabulations.png", dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved cross-tabulation plot to {output_dir / 'cross_tabulations.png'}")
    plt.close()


def generate_summary_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate comprehensive summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    summary = {
        "total_topics": len(df),
        "topics_with_labels": df["label"].notna().sum(),
        "topics_with_taxonomy": df["taxonomy_main_id"].notna().sum(),
        "topics_with_radway": df["radway_main_id"].notna().sum(),
        "topics_with_radway_function": (df["radway_is_none"] == False).sum(),
        "topics_with_radway_none": (df["radway_is_none"] == True).sum(),
        "unique_taxonomy_categories": df["taxonomy_main_name"].nunique(),
        "unique_taxonomy_groups": df["taxonomy_main_group"].nunique(),
        "unique_radway_functions": df[df["radway_is_none"] == False]["radway_main_name"].nunique(),
        "unique_radway_phases": df["radway_phase_name"].nunique(),
    }

    print("\n--- Overall Statistics ---")
    for key, value in summary.items():
        print(f"{key}: {value}")

    # Save summary to JSON (convert numpy types to native Python types)
    summary_serializable = {k: int(v) if isinstance(v, (np.integer, int)) else v for k, v in summary.items()}
    with open(output_dir / "summary_statistics.json", "w") as f:
        json.dump(summary_serializable, f, indent=2)
    print(f"\n✓ Saved summary statistics to {output_dir / 'summary_statistics.json'}")


def export_dataframe(df: pd.DataFrame, output_dir: Path) -> None:
    """Export the full DataFrame to CSV and Parquet."""
    csv_path = output_dir / "full_model_data.csv"
    parquet_path = output_dir / "full_model_data.parquet"

    # Ensure radway_phase="NA" is preserved as string (not coerced to NaN)
    if "radway_phase" in df.columns:
        df = df.copy()
        df["radway_phase"] = df["radway_phase"].fillna("NA")
        # Convert any remaining NaN to "NA" string explicitly
        df["radway_phase"] = df["radway_phase"].astype(str).replace("nan", "NA")

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    print(f"\n✓ Exported full data to:")
    print(f"  - CSV: {csv_path}")
    print(f"  - Parquet: {parquet_path}")


def main():
    """Main EDA function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="EDA for BERTopic model with Taxonomy (Stage 2) and Radway (Stage 3) mappings"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage09_category_mapping/stage3_radway_functions/eda"),
        help="Output directory for EDA results",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base directory for models",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    model = load_model_with_radway(
        base_dir=args.base_dir,
        embedding_model=args.embedding_model,
    )

    # Extract all fields
    print("\n" + "=" * 80)
    print("EXTRACTING DATA")
    print("=" * 80)
    df = extract_all_fields(model)
    print(f"✓ Extracted data for {len(df)} topics")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")

    # Run analyses
    generate_summary_statistics(df, args.output_dir)
    analyze_taxonomy_distribution(df, args.output_dir)
    analyze_radway_distribution(df, args.output_dir)
    analyze_cross_tabulations(df, args.output_dir)
    export_dataframe(df, args.output_dir)

    print("\n" + "=" * 80)
    print("EDA COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

