"""Stage 10: Visualization helpers for taxonomy category analysis.

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

# Set default style parameters for better readability
plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
})


def plot_category_prevalence(
    book_cat: pd.DataFrame,
    category_id: str,
    rating_order=("bad", "mid", "good"),
    plot_type: str = "violin",
    category_name: Optional[str] = None,
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
    category_name:
        Optional category name for title

    Returns
    -------
    fig, ax:
        Matplotlib figure and axes objects.
    """
    sub = book_cat[book_cat["main_category_id"] == category_id].copy()
    if sub.empty:
        raise ValueError(f"No rows for category_id={category_id}")

    # Adjust figure size based on whether we have a long category name
    title_height = 0.15 if category_name and len(category_name) > 40 else 0.12
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(top=1 - title_height, bottom=0.12, left=0.12, right=0.95)

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

    # Create title with proper wrapping
    if category_name:
        # Truncate very long names and add ellipsis
        display_name = category_name if len(category_name) <= 60 else category_name[:57] + "..."
        title = f"{category_id}: {display_name}\nPrevalence by rating class"
    else:
        title = f"Category {category_id}: prevalence by rating class"
    
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Proportion of sentences per book", fontsize=10, labelpad=8)
    ax.set_xlabel("Rating class", fontsize=10, labelpad=8)
    
    # Improve tick labels
    ax.tick_params(axis='both', which='major', labelsize=9, pad=5)
    
    plt.tight_layout(rect=[0, 0, 1, 1 - title_height])
    return fig, ax


def plot_volcano(
    kw_results: pd.DataFrame,
    alpha: float = 0.05,
    effect_threshold: float = 0.01,
    figsize: Tuple[int, int] = (12, 8),
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
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.1, top=0.92)
    
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
        s=120,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.8,
        zorder=3,
    )
    
    # Add labels for significant categories with improved positioning
    sig_data = kw_results[kw_results["significant"]].copy()
    if len(sig_data) > 0:
        # Get axis limits for better positioning
        x_range = kw_results["eta_squared"].max() - kw_results["eta_squared"].min()
        y_range = kw_results["neg_log10_p"].max() - kw_results["neg_log10_p"].min()
        
        for _, row in sig_data.iterrows():
            label = row.get("category_name", row["category_id"])
            # Truncate long labels
            if len(label) > 40:
                label = label[:37] + "..."
            
            # Adjust offset based on position to avoid overlaps
            x_offset = x_range * 0.02
            y_offset = y_range * 0.02
            
            ax.annotate(
                label,
                (row["eta_squared"], row["neg_log10_p"]),
                fontsize=9,
                alpha=0.8,
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray", linewidth=0.5),
                zorder=4,
            )
    
    # Add threshold lines
    ax.axhline(-np.log10(alpha), color="red", linestyle="--", alpha=0.6, linewidth=1.5, label=f"p = {alpha}", zorder=1)
    ax.axvline(effect_threshold, color="blue", linestyle="--", alpha=0.6, linewidth=1.5, label=f"η² = {effect_threshold}", zorder=1)
    
    # Labels and title
    ax.set_xlabel("Effect Size (η²)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("-log₁₀(p-value)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("Volcano Plot: Statistical Significance vs Effect Size", fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, zorder=0)
    
    # Legend with better positioning
    legend_elements = [
        mpatches.Patch(color="#d62728", label="Significant + Large Effect"),
        mpatches.Patch(color="#ff7f0e", label="Significant Only"),
        mpatches.Patch(color="#2ca02c", label="Large Effect Only"),
        mpatches.Patch(color="#7f7f7f", label="Neither"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10, framealpha=0.9, edgecolor="gray")
    
    # Improve tick labels
    ax.tick_params(axis='both', which='major', labelsize=10, pad=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, ax


def plot_effect_size_bars(
    kw_results: pd.DataFrame,
    top_n: int = 15,
    alpha: float = 0.05,
    figsize: Optional[Tuple[int, int]] = None,
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
        Figure size (auto-calculated if None)
    
    Returns
    -------
    fig, ax
    """
    # Sort by effect size
    plot_data = kw_results.copy()
    plot_data = plot_data.sort_values("eta_squared", ascending=True).tail(top_n)
    plot_data["significant"] = plot_data["p_value"] < alpha
    
    # Dynamic figure sizing based on number of categories
    if figsize is None:
        height = max(6, len(plot_data) * 0.5)
        width = 12
        figsize = (width, height)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate left margin based on longest label
    labels = [
        plot_data.loc[idx].get("category_name", plot_data.loc[idx]["category_id"])
        for idx in plot_data.index
    ]
    max_label_len = max(len(label) for label in labels)
    left_margin = max(0.25, min(0.4, 0.15 + max_label_len * 0.01))
    
    fig.subplots_adjust(left=left_margin, right=0.95, bottom=0.1, top=0.92)
    
    # Colors
    colors = ["#d62728" if sig else "#7f7f7f" for sig in plot_data["significant"]]
    
    # Truncate long labels for display
    display_labels = []
    for label in labels:
        if len(label) > 50:
            display_labels.append(label[:47] + "...")
        else:
            display_labels.append(label)
    
    # Bar plot
    bars = ax.barh(range(len(plot_data)), plot_data["eta_squared"], color=colors, alpha=0.7, height=0.7)
    
    # Add p-value annotations with better positioning
    max_eta = plot_data["eta_squared"].max()
    annotation_offset = max_eta * 0.02
    
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        p_str = f"p={row['p_value']:.3f}" if row['p_value'] >= 0.001 else "p<0.001"
        x_pos = row["eta_squared"] + annotation_offset
        
        # Check if annotation would go beyond plot
        x_max = ax.get_xlim()[1]
        if x_pos > x_max * 0.95:
            # Place inside bar instead
            x_pos = row["eta_squared"] * 0.5
            text_color = "white" if row["significant"] else "black"
        else:
            text_color = "black"
        
        ax.text(
            x_pos,
            i,
            p_str,
            va="center",
            fontsize=9,
            alpha=0.9,
            color=text_color,
            fontweight="bold" if text_color == "white" else "normal",
        )
    
    # Labels
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(display_labels, fontsize=10)
    ax.set_xlabel("Effect Size (η²)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(f"Top {top_n} Categories by Effect Size", fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, axis="x", alpha=0.3, zorder=0)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color="#d62728", label=f"Significant (p < {alpha})"),
        mpatches.Patch(color="#7f7f7f", label="Not Significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10, framealpha=0.9, edgecolor="gray")
    
    # Improve tick labels
    ax.tick_params(axis='both', which='major', labelsize=10, pad=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, ax


def plot_pairwise_comparisons(
    pairwise_results: pd.DataFrame,
    category_id: str,
    category_name: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
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
        Figure size (auto-calculated if None)
    
    Returns
    -------
    fig, ax
    """
    if pairwise_results.empty:
        if figsize is None:
            figsize = (8, 5)
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No pairwise comparisons available", 
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        return fig, ax
    
    # Dynamic figure sizing
    if figsize is None:
        height = max(5, len(pairwise_results) * 1.2)
        width = 10
        figsize = (width, height)
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.2, right=0.95, bottom=0.12, top=0.88)
    
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
                   color=colors, alpha=0.7, height=0.6)
    
    # Add significance threshold
    ax.axvline(-np.log10(0.05), color="red", linestyle="--", alpha=0.6, linewidth=1.5, label="p = 0.05", zorder=1)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pairwise_results["comparison"], fontsize=11)
    ax.set_xlabel("-log₁₀(corrected p-value)", fontsize=12, fontweight="bold", labelpad=10)
    
    # Create title with proper wrapping
    title = f"Pairwise Comparisons: {category_id}"
    if category_name:
        # Truncate very long names
        display_name = category_name if len(category_name) <= 50 else category_name[:47] + "..."
        title += f"\n({display_name})"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    
    # Add median difference annotations with better positioning
    max_x = -np.log10(pairwise_results["p_value_corrected"] + 1e-10).max()
    annotation_offset = max_x * 0.05
    
    for i, (_, row) in enumerate(pairwise_results.iterrows()):
        direction = "↑" if row["median_diff"] > 0 else "↓"
        x_pos = -np.log10(row["p_value_corrected"] + 1e-10) + annotation_offset
        
        # Check if annotation would go beyond plot
        x_max = ax.get_xlim()[1]
        if x_pos > x_max * 0.95:
            # Place inside bar instead
            x_pos = -np.log10(row["p_value_corrected"] + 1e-10) * 0.5
            text_color = "white"
            fontweight = "bold"
        else:
            text_color = "black"
            fontweight = "normal"
        
        ax.text(
            x_pos,
            i,
            f"{direction} {row['median_diff']:.4f}",
            va="center",
            fontsize=10,
            color=text_color,
            fontweight=fontweight,
            alpha=0.9,
        )
    
    ax.grid(True, axis="x", alpha=0.3, zorder=0)
    ax.legend(fontsize=10, framealpha=0.9, edgecolor="gray")
    
    # Improve tick labels
    ax.tick_params(axis='both', which='major', labelsize=10, pad=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig, ax


def plot_pvalue_heatmap(
    kw_results: pd.DataFrame,
    group_by: str = "category_group",
    figsize: Optional[Tuple[int, int]] = None,
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
        Figure size (auto-calculated if None)
    
    Returns
    -------
    fig, ax
    """
    # Prepare data
    plot_data = kw_results.copy()
    
    # Dynamic figure sizing
    n_categories = len(plot_data)
    if figsize is None:
        width = max(14, n_categories * 0.6)
        height = 6
        figsize = (width, height)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Adjust margins based on number of categories
    bottom_margin = max(0.15, min(0.3, 0.1 + n_categories * 0.002))
    fig.subplots_adjust(left=0.1, right=0.92, bottom=bottom_margin, top=0.92)
    
    if group_by and group_by in plot_data.columns:
        # Sort by group, then by p-value
        plot_data = plot_data.sort_values([group_by, "p_value"])
        labels = []
        for _, row in plot_data.iterrows():
            cat_name = row.get('category_name', '')
            # Truncate and format label
            if len(cat_name) > 25:
                cat_name = cat_name[:22] + "..."
            labels.append(f"{row['category_id']}\n({cat_name})")
    else:
        plot_data = plot_data.sort_values("p_value")
        labels = []
        for _, row in plot_data.iterrows():
            cat_name = row.get('category_name', '')
            if len(cat_name) > 25:
                cat_name = cat_name[:22] + "..."
            labels.append(f"{row['category_id']}\n({cat_name})")
    
    # Create heatmap data (single column of p-values)
    heatmap_data = plot_data[["p_value"]].T
    
    # Create heatmap with improved formatting
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",  # Red-Yellow-Green reversed (red = significant)
        vmin=0,
        vmax=0.1,
        cbar_kws={"label": "p-value", "shrink": 0.8, "pad": 0.02},
        yticklabels=["p-value"],
        xticklabels=labels,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"fontsize": 9},
    )
    
    ax.set_title("P-value Overview by Category", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Category", fontsize=12, labelpad=10)
    ax.set_ylabel("", fontsize=10)
    
    # Improve x-axis label rotation and positioning
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, ax

