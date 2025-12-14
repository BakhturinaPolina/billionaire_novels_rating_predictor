"""Stage 09: Statistical helpers for taxonomy category analysis.

Kruskal-Wallis tests for category prevalence differences across rating classes.
Includes effect size calculations and post-hoc pairwise comparisons.
"""

from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu


def kruskal_eta_squared(H: float, n: int, k: int) -> float:
    """
    Calculate eta-squared effect size for Kruskal-Wallis test.
    
    Parameters
    ----------
    H: H-statistic from Kruskal-Wallis
    n: Total sample size
    k: Number of groups
    
    Returns
    -------
    Eta-squared value (0-1, where 0.01=small, 0.06=medium, 0.14=large)
    """
    return (H - k + 1) / (n - k)


def kruskal_by_rating(book_cat: pd.DataFrame) -> pd.DataFrame:
    """
    For each taxonomy category, run a Kruskal–Wallis test
    over rating_class (e.g. low/mid/high) on book-level proportions.
    Includes effect size calculation.

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
        - eta_squared (effect size)
        - total_n (total sample size)
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
            total_n = sum(ns)
            k = len(groups)
            eta_sq = kruskal_eta_squared(stat, total_n, k)
            
            results.append(
                {
                    "category_id": cat,
                    "groups": labels,
                    "n_books_per_group": ns,
                    "H_statistic": stat,
                    "p_value": p,
                    "eta_squared": eta_sq,
                    "total_n": total_n,
                }
            )

    return pd.DataFrame(results)


def pairwise_comparisons(
    book_cat: pd.DataFrame, 
    category_id: str,
    alpha: float = 0.05,
    correction: str = "bonferroni"
) -> pd.DataFrame:
    """
    Perform pairwise Mann-Whitney U tests for a category across rating classes.
    
    Parameters
    ----------
    book_cat:
        DataFrame with columns: 'main_category_id', 'rating_class', 'prop'
    category_id:
        Category to test
    alpha:
        Significance level (before correction)
    correction:
        Multiple comparison correction: 'bonferroni' or 'none'
    
    Returns
    -------
    DataFrame with columns:
        - group1, group2: Pair of groups compared
        - U_statistic: Mann-Whitney U statistic
        - p_value: Uncorrected p-value
        - p_value_corrected: Corrected p-value
        - significant: Whether corrected p < alpha
        - median_diff: Difference in medians (group2 - group1)
    """
    sub = book_cat[book_cat["main_category_id"] == category_id].copy()
    if sub.empty:
        return pd.DataFrame()
    
    # Get unique rating classes
    rating_classes = sorted(sub["rating_class"].unique())
    if len(rating_classes) < 2:
        return pd.DataFrame()
    
    # Calculate all pairwise comparisons
    pairs = []
    for i, group1 in enumerate(rating_classes):
        for group2 in rating_classes[i+1:]:
            vals1 = sub[sub["rating_class"] == group1]["prop"].dropna()
            vals2 = sub[sub["rating_class"] == group2]["prop"].dropna()
            
            if len(vals1) >= 3 and len(vals2) >= 3:
                # Two-sided test
                u_stat, p_val = mannwhitneyu(vals1, vals2, alternative='two-sided')
                median_diff = vals2.median() - vals1.median()
                
                pairs.append({
                    "group1": group1,
                    "group2": group2,
                    "U_statistic": u_stat,
                    "p_value": p_val,
                    "median_diff": median_diff,
                })
    
    if not pairs:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(pairs)
    
    # Apply correction
    n_comparisons = len(results_df)
    if correction == "bonferroni" and n_comparisons > 1:
        results_df["p_value_corrected"] = results_df["p_value"] * n_comparisons
        results_df["p_value_corrected"] = results_df["p_value_corrected"].clip(upper=1.0)
    else:
        results_df["p_value_corrected"] = results_df["p_value"]
    
    results_df["significant"] = results_df["p_value_corrected"] < alpha
    
    return results_df

