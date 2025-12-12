"""Stage 09: Statistical helpers for taxonomy category analysis.

Kruskal-Wallis tests for category prevalence differences across rating classes.
"""

from typing import List

import pandas as pd
from scipy.stats import kruskal


def kruskal_by_rating(book_cat: pd.DataFrame) -> pd.DataFrame:
    """
    For each taxonomy category, run a Kruskal–Wallis test
    over rating_class (e.g. low/mid/high) on book-level proportions.

    Parameters
    ----------
    book_cat:
        DataFrame with columns:
        - 'main_category_id'
        - 'rating_class'
        - 'prop'

    Returns
    -------
    DataFrame with columns:
        - category_id
        - groups (list of rating classes tested)
        - n_books_per_group (list of sample sizes)
        - H_statistic
        - p_value
    """
    results: List[dict] = []

    cats = sorted(book_cat["main_category_id"].dropna().unique())
    for cat in cats:
        sub = book_cat[book_cat["main_category_id"] == cat]

        groups = []
        labels = []
        ns = []

        for rating in sorted(sub["rating_class"].unique()):
            vals = sub.loc[sub["rating_class"] == rating, "prop"].dropna()
            if len(vals) >= 5:  # avoid tiny groups
                groups.append(vals.to_numpy())
                labels.append(rating)
                ns.append(len(vals))

        if len(groups) >= 2:
            stat, p = kruskal(*groups)
            results.append(
                {
                    "category_id": cat,
                    "groups": labels,
                    "n_books_per_group": ns,
                    "H_statistic": stat,
                    "p_value": p,
                }
            )

    return pd.DataFrame(results)

