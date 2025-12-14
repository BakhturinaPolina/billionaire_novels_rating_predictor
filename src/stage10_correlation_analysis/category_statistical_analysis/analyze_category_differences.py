"""Run statistical analysis and create visualizations for taxonomy category differences.

This script uses stats_helpers and visualization_helpers to:
1. Run Kruskal-Wallis tests for each category
2. Identify significant differences
3. Create visualizations for top categories
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from src.stage10_correlation_analysis.category_statistical_analysis.stats_helpers import (
    kruskal_by_rating,
    pairwise_comparisons,
)
from src.stage10_correlation_analysis.category_statistical_analysis.visualization_helpers import (
    plot_category_prevalence,
    plot_volcano,
    plot_effect_size_bars,
    plot_pairwise_comparisons,
    plot_pvalue_heatmap,
)


def load_category_names(taxonomy_json_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load category names and groups from taxonomy mappings JSON.
    
    Parameters
    ----------
    taxonomy_json_path:
        Path to taxonomy_mappings_*.json file
        
    Returns
    -------
    Dictionary mapping category_id -> {name, group}
    """
    with open(taxonomy_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    category_info = {}
    for topic_data in data.values():
        cat_id = topic_data.get("main_category_id")
        if cat_id and cat_id not in category_info:
            category_info[cat_id] = {
                "category_name": topic_data.get("main_category_name", cat_id),
                "category_group": topic_data.get("main_category_group", "Unknown"),
            }
    
    return category_info

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze taxonomy category differences across rating classes"
    )
    parser.add_argument(
        "--book-cat",
        type=Path,
        default=Path(
            "results/stage09_category_mapping/stage2_theory_driven_categories/book_category_proportions.parquet"
        ),
        help="Path to book_category_proportions.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/stage10_correlation_analysis/category_statistical_analysis"
        ),
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top categories to visualize (default: 10)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for p-values (default: 0.05)",
    )
    parser.add_argument(
        "--taxonomy-json",
        type=Path,
        default=Path(
            "results/stage09_category_mapping/stage2_theory_driven_categories/taxonomy_mappings_openrouter_mistralai_Mistral-Nemo-Instruct-2407_paraphrase-MiniLM-L6-v2.json"
        ),
        help="Path to taxonomy_mappings_*.json for category names",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading book category proportions from {args.book_cat}")
    book_cat = pd.read_parquet(args.book_cat)
    print(f"Loaded {len(book_cat):,} rows")
    print(f"  Books: {book_cat['book_id'].nunique()}")
    print(f"  Categories: {book_cat['main_category_id'].nunique()}")
    print(f"  Rating classes: {sorted(book_cat['rating_class'].unique())}")

    # Load category names
    print(f"\nLoading category names from {args.taxonomy_json}")
    category_info = load_category_names(args.taxonomy_json)
    print(f"Loaded {len(category_info)} category names")

    # Run statistical tests
    print("\n" + "=" * 80)
    print("Running Kruskal-Wallis tests...")
    print("=" * 80)
    kw_results = kruskal_by_rating(book_cat)
    print(f"Tests completed for {len(kw_results)} categories")

    # Add category names and groups
    kw_results["category_name"] = kw_results["category_id"].map(
        lambda x: category_info.get(x, {}).get("category_name", x)
    )
    kw_results["category_group"] = kw_results["category_id"].map(
        lambda x: category_info.get(x, {}).get("category_group", "Unknown")
    )

    # Sort by p-value
    kw_results = kw_results.sort_values("p_value")
    kw_results["significant"] = kw_results["p_value"] < args.alpha

    # Print summary
    n_sig = kw_results["significant"].sum()
    print(f"\nSignificant differences (p < {args.alpha}): {n_sig}/{len(kw_results)}")

    # Show top categories with names
    print("\n" + "=" * 80)
    print("Top categories by significance:")
    print("=" * 80)
    # Reorder columns for better readability
    display_cols = [
        "category_id",
        "category_name",
        "category_group",
        "p_value",
        "H_statistic",
        "significant",
        "groups",
        "n_books_per_group",
    ]
    display_df = kw_results[display_cols].head(args.top_n)
    print(display_df.to_string(index=False))
    
    # Print significant categories with full details
    if n_sig > 0:
        print("\n" + "=" * 80)
        print("Significant Categories (p < 0.05):")
        print("=" * 80)
        sig_df = kw_results[kw_results["significant"]][display_cols]
        for _, row in sig_df.iterrows():
            print(f"\n{row['category_id']}: {row['category_name']}")
            print(f"  Group: {row['category_group']}")
            print(f"  p-value: {row['p_value']:.6f}")
            print(f"  H-statistic: {row['H_statistic']:.4f}")
            print(f"  Groups tested: {', '.join(row['groups'])}")
            print(f"  Sample sizes: {row['n_books_per_group']}")

    # Save statistical results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_output = args.output_dir / "kruskal_wallis_results.csv"
    kw_results.to_csv(stats_output, index=False)
    print(f"\nSaved statistical results to: {stats_output}")

    # Create visualizations
    print("\n" + "=" * 80)
    print("Creating visualizations...")
    print("=" * 80)

    figs_dir = args.output_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overview plots
    print("\n1. Creating overview plots...")
    
    # Volcano plot (save as SVG for interactive use)
    try:
        print("   Creating volcano plot...")
        fig, ax = plot_volcano(kw_results, alpha=args.alpha)
        # Save as SVG for interactive plots (better for web/publications)
        fig.savefig(figs_dir / "volcano_plot.svg", format="svg", bbox_inches="tight", pad_inches=0.2)
        # Also save as PNG for compatibility
        fig.savefig(figs_dir / "volcano_plot.png", dpi=150, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
    except Exception as e:
        print(f"   Warning: Failed to create volcano plot: {e}")
    
    # Effect size bar chart
    try:
        print("   Creating effect size bar chart...")
        fig, ax = plot_effect_size_bars(
            kw_results, 
            top_n=min(20, len(kw_results)), 
            alpha=args.alpha,
            exclude_noise=True  # Exclude noise/technical/paratext categories
        )
        fig.savefig(figs_dir / "effect_size_bars.png", dpi=150, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
    except Exception as e:
        print(f"   Warning: Failed to create effect size chart: {e}")
    
    # P-value heatmap
    try:
        print("   Creating p-value heatmap...")
        fig, ax = plot_pvalue_heatmap(kw_results)
        fig.savefig(figs_dir / "pvalue_heatmap.png", dpi=150, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
    except Exception as e:
        print(f"   Warning: Failed to create p-value heatmap: {e}")

    # 2. Individual category plots (enhanced with violin plots)
    print(f"\n2. Creating individual category plots (top {args.top_n})...")
    top_categories_df = kw_results.head(args.top_n)
    top_categories = top_categories_df["category_id"].tolist()

    for i, (_, row) in enumerate(top_categories_df.iterrows(), 1):
        cat_id = row["category_id"]
        cat_name = row["category_name"]
        try:
            print(f"   [{i}/{len(top_categories)}] Plotting {cat_id}: {cat_name}...")
            # Use violin plot for better distribution visualization
            p_val = row["p_value"]
            sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < args.alpha else ""
            # Create full title with significance marker
            full_title = f"{cat_id}: {cat_name} {sig_marker}\nPrevalence by rating class (p={p_val:.4f})"
            fig, ax = plot_category_prevalence(book_cat, cat_id, plot_type="violin", category_name=cat_name)
            # Update title to include p-value and significance marker
            ax.set_title(full_title, fontsize=12, fontweight="bold", pad=10)
            fig.savefig(figs_dir / f"category_{cat_id}_prevalence.png", dpi=150, bbox_inches="tight", pad_inches=0.2)
            plt.close(fig)
        except Exception as e:
            print(f"     Warning: Failed to plot category {cat_id} ({cat_name}): {e}")

    # 3. Post-hoc pairwise comparisons for significant categories
    print(f"\n3. Creating pairwise comparison plots for significant categories...")
    sig_categories = kw_results[kw_results["significant"]]
    
    if len(sig_categories) > 0:
        for i, (_, row) in enumerate(sig_categories.iterrows(), 1):
            cat_id = row["category_id"]
            cat_name = row["category_name"]
            try:
                print(f"   [{i}/{len(sig_categories)}] Pairwise comparisons for {cat_id}: {cat_name}...")
                pairwise_res = pairwise_comparisons(book_cat, cat_id, alpha=args.alpha)
                if not pairwise_res.empty:
                    fig, ax = plot_pairwise_comparisons(pairwise_res, cat_id, cat_name)
                    fig.savefig(figs_dir / f"category_{cat_id}_pairwise.png", dpi=150, bbox_inches="tight", pad_inches=0.2)
                    plt.close(fig)
            except Exception as e:
                print(f"     Warning: Failed to create pairwise plot for {cat_id}: {e}")
    else:
        print("   No significant categories found for pairwise comparisons.")

    print(f"\nSaved all visualizations to: {figs_dir}")

    # Summary
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)
    print(f"Statistical results: {stats_output}")
    print(f"Visualizations: {figs_dir}")
    print(f"Total categories analyzed: {len(kw_results)}")
    print(f"Significant categories (p < {args.alpha}): {n_sig}")
    
    if n_sig > 0:
        print("\nSignificant categories:")
        for _, row in kw_results[kw_results["significant"]].iterrows():
            print(f"  - {row['category_id']}: {row['category_name']} (p={row['p_value']:.4f})")

