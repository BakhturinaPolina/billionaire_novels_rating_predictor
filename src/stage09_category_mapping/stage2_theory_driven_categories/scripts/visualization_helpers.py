"""Stage 09: Visualization helpers for taxonomy category analysis.

Box plots and strip plots for category prevalence across rating classes.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_category_prevalence(
    book_cat: pd.DataFrame,
    category_id: str,
    rating_order=("low", "mid", "high"),
):
    """
    Box + jitter plot for one taxonomy category across rating classes.

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

    Returns
    -------
    fig, ax:
        Matplotlib figure and axes objects.
    """
    sub = book_cat[book_cat["main_category_id"] == category_id].copy()
    if sub.empty:
        raise ValueError(f"No rows for category_id={category_id}")

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.boxplot(
        data=sub,
        x="rating_class",
        y="prop",
        order=rating_order,
        ax=ax,
    )
    sns.stripplot(
        data=sub,
        x="rating_class",
        y="prop",
        order=rating_order,
        ax=ax,
        alpha=0.4,
        jitter=0.2,
        dodge=False,
    )

    ax.set_title(f"Category {category_id}: prevalence by rating class")
    ax.set_ylabel("Proportion of sentences per book")
    ax.set_xlabel("Rating class")
    plt.tight_layout()
    return fig, ax

