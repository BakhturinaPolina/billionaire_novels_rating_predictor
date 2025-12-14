"""Stage 09: Visualization helpers for taxonomy category analysis.

Enhanced visualizations for category prevalence across rating classes:
- Box plots, violin plots, strip plots
- Volcano plots (p-value vs effect size)
- Effect size bar charts
- Post-hoc pairwise comparisons
- P-value heatmaps
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, Tuple, List


def plot_category_prevalence(
    book_cat: pd.DataFrame,
    category_id: str,
    rating_order=("bad", "mid", "good"),
    plot_type: str = "violin",
):
    """
    Enhanced plot for one taxonomy category across rating classes.
    Supports box plots, violin plots, or both.

    Parameters
    ----------
    book_cat:
        DataFrame with columns:
        - 'main_category_id'
        - 'rating_class'
        - 'prop'
    category_id:
        Taxonomy category ID to plot (e.g., "4.4", "2.3").
    rating_order:
        Tuple of rating class labels in desired order.
    plot_type:
        'box', 'violin', or 'both'

    Returns
    -------
    fig, ax:
        Matplotlib figure and axes objects.
    """
    sub = book_cat[book_cat["main_category_id"] == category_id].copy()
    if sub.empty:
        raise ValueError(f"No rows for category_id={category_id}")

    fig, ax = plt.subplots(figsize=(7, 5))

    if plot_type in ["violin", "both"]:
        sns.violinplot(
            data=sub,
            x="rating_class",
            y="prop",
            order=rating_order,
            ax=ax,
            inner="box",
            alpha=0.7,
        )
    if plot_type in ["box", "both"]:
        sns.boxplot(
            data=sub,
            x="rating_class",
            y="prop",
            order=rating_order,
            ax=ax,
            width=0.3,
        )
    
    # Add individual points
    sns.stripplot(
        data=sub,
        x="rating_class",
        y="prop",
        order=rating_order,
        ax=ax,
        alpha=0.5,
        jitter=0.2,
        dodge=False,
        size=3,
        color="black",
    )

    ax.set_title(f"Category {category_id}: prevalence by rating class", fontsize=12, fontweight="bold")
    ax.set_ylabel("Proportion of sentences per book", fontsize=10)
    ax.set_xlabel("Rating class", fontsize=10)
    plt.tight_layout()
    return fig, ax


def plot_volcano(
    kw_results: pd.DataFrame,
    alpha: float = 0.05,
    effect_threshold: float = 0.01,
    figsize: Tuple[int, int] = (10, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Volcano plot: -log10(p-value) vs effect size (eta-squared).
    
    Parameters
    ----------
    kw_results:
        DataFrame with columns: category_id, p_value, eta_squared, category_name (optional)
    alpha:
        Significance threshold
    effect_threshold:
        Effect size threshold for highlighting
    figsize:
        Figure size
    
    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate -log10(p-value)
    kw_results = kw_results.copy()
    kw_results["neg_log10_p"] = -np.log10(kw_results["p_value"] + 1e-10)
    kw_results["significant"] = kw_results["p_value"] < alpha
    kw_results["large_effect"] = kw_results["eta_squared"] >= effect_threshold
    
    # Color points
    colors = []
    for _, row in kw_results.iterrows():
        if row["significant"] and row["large_effect"]:
            colors.append("#d62728")  # Red: significant + large effect
        elif row["significant"]:
            colors.append("#ff7f0e")  # Orange: significant only
        elif row["large_effect"]:
            colors.append("#2ca02c")  # Green: large effect only
        else:
            colors.append("#7f7f7f")  # Gray: neither
    
    # Scatter plot
    scatter = ax.scatter(
        kw_results["eta_squared"],
        kw_results["neg_log10_p"],
        c=colors,
        s=100,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
    )
    
    # Add labels for significant categories
    for _, row in kw_results[kw_results["significant"]].iterrows():
        label = row.get("category_name", row["category_id"])
        ax.annotate(
            label,
            (row["eta_squared"], row["neg_log10_p"]),
            fontsize=8,
            alpha=0.7,
            xytext=(5, 5),
            textcoords="offset points",
        )
    
    # Add threshold lines
    ax.axhline(-np.log10(alpha), color="red", linestyle="--", alpha=0.5, label=f"p = {alpha}")
    ax.axvline(effect_threshold, color="blue", linestyle="--", alpha=0.5, label=f"η² = {effect_threshold}")
    
    # Labels and title
    ax.set_xlabel("Effect Size (η²)", fontsize=11, fontweight="bold")
    ax.set_ylabel("-log₁₀(p-value)", fontsize=11, fontweight="bold")
    ax.set_title("Volcano Plot: Statistical Significance vs Effect Size", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color="#d62728", label="Significant + Large Effect"),
        mpatches.Patch(color="#ff7f0e", label="Significant Only"),
        mpatches.Patch(color="#2ca02c", label="Large Effect Only"),
        mpatches.Patch(color="#7f7f7f", label="Neither"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    
    plt.tight_layout()
    return fig, ax


def plot_effect_size_bars(
    kw_results: pd.DataFrame,
    top_n: int = 15,
    alpha: float = 0.05,
    figsize: Tuple[int, int] = (10, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Bar chart of effect sizes, ranked by eta-squared.
    
    Parameters
    ----------
    kw_results:
        DataFrame with columns: category_id, eta_squared, p_value, category_name (optional)
    top_n:
        Number of top categories to show
    alpha:
        Significance threshold
    figsize:
        Figure size
    
    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by effect size
    plot_data = kw_results.copy()
    plot_data = plot_data.sort_values("eta_squared", ascending=True).tail(top_n)
    plot_data["significant"] = plot_data["p_value"] < alpha
    
    # Colors
    colors = ["#d62728" if sig else "#7f7f7f" for sig in plot_data["significant"]]
    
    # Get labels
    labels = [
        plot_data.loc[idx].get("category_name", plot_data.loc[idx]["category_id"])
        for idx in plot_data.index
    ]
    
    # Bar plot
    bars = ax.barh(range(len(plot_data)), plot_data["eta_squared"], color=colors, alpha=0.7)
    
    # Add p-value annotations
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        p_str = f"p={row['p_value']:.3f}" if row['p_value'] >= 0.001 else "p<0.001"
        ax.text(
            row["eta_squared"] + 0.001,
            i,
            p_str,
            va="center",
            fontsize=8,
            alpha=0.7,
        )
    
    # Labels
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Effect Size (η²)", fontsize=11, fontweight="bold")
    ax.set_title(f"Top {top_n} Categories by Effect Size", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color="#d62728", label=f"Significant (p < {alpha})"),
        mpatches.Patch(color="#7f7f7f", label="Not Significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    
    plt.tight_layout()
    return fig, ax


def plot_pairwise_comparisons(
    pairwise_results: pd.DataFrame,
    category_id: str,
    category_name: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize pairwise comparison results for a category.
    
    Parameters
    ----------
    pairwise_results:
        DataFrame from pairwise_comparisons() function
    category_id:
        Category ID for title
    category_name:
        Optional category name for title
    figsize:
        Figure size
    
    Returns
    -------
    fig, ax
    """
    if pairwise_results.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No pairwise comparisons available", 
                ha="center", va="center", transform=ax.transAxes)
        return fig, ax
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create comparison labels
    pairwise_results = pairwise_results.copy()
    pairwise_results["comparison"] = (
        pairwise_results["group1"] + " vs " + pairwise_results["group2"]
    )
    
    # Sort by p-value
    pairwise_results = pairwise_results.sort_values("p_value_corrected")
    
    # Colors
    colors = [
        "#d62728" if sig else "#7f7f7f"
        for sig in pairwise_results["significant"]
    ]
    
    # Bar plot
    y_pos = np.arange(len(pairwise_results))
    bars = ax.barh(y_pos, -np.log10(pairwise_results["p_value_corrected"] + 1e-10), 
                   color=colors, alpha=0.7)
    
    # Add significance threshold
    ax.axvline(-np.log10(0.05), color="red", linestyle="--", alpha=0.5, label="p = 0.05")
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pairwise_results["comparison"], fontsize=10)
    ax.set_xlabel("-log₁₀(corrected p-value)", fontsize=11, fontweight="bold")
    
    title = f"Pairwise Comparisons: {category_id}"
    if category_name:
        title += f" ({category_name})"
    ax.set_title(title, fontsize=12, fontweight="bold")
    
    # Add median difference annotations
    for i, (_, row) in enumerate(pairwise_results.iterrows()):
        direction = "↑" if row["median_diff"] > 0 else "↓"
        ax.text(
            -np.log10(row["p_value_corrected"] + 1e-10) + 0.1,
            i,
            f"{direction} {row['median_diff']:.4f}",
            va="center",
            fontsize=8,
        )
    
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig, ax


def plot_pvalue_heatmap(
    kw_results: pd.DataFrame,
    group_by: str = "category_group",
    figsize: Tuple[int, int] = (12, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Heatmap of p-values, optionally grouped by category group.
    
    Parameters
    ----------
    kw_results:
        DataFrame with columns: category_id, p_value, category_group (optional)
    group_by:
        Column to group by (e.g., 'category_group') or None
    figsize:
        Figure size
    
    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    plot_data = kw_results.copy()
    
    if group_by and group_by in plot_data.columns:
        # Sort by group, then by p-value
        plot_data = plot_data.sort_values([group_by, "p_value"])
        labels = [
            f"{row['category_id']}\n({row.get('category_name', '')[:30]})"
            for _, row in plot_data.iterrows()
        ]
    else:
        plot_data = plot_data.sort_values("p_value")
        labels = [
            f"{row['category_id']}\n({row.get('category_name', '')[:30]})"
            for _, row in plot_data.iterrows()
        ]
    
    # Create heatmap data (single column of p-values)
    heatmap_data = plot_data[["p_value"]].T
    
    # Create heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",  # Red-Yellow-Green reversed (red = significant)
        vmin=0,
        vmax=0.1,
        cbar_kws={"label": "p-value"},
        yticklabels=["p-value"],
        xticklabels=labels,
        ax=ax,
    )
    
    ax.set_title("P-value Overview by Category", fontsize=13, fontweight="bold")
    ax.set_xlabel("Category", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig, ax

