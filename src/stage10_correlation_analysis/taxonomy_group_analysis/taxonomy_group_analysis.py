"""
Taxonomy Group Analysis: Main Groups & Subgroups Across Tiers

Scope: Group-level analysis aggregating 368 topics into taxonomy main groups and subgroups.
Uses dual normalization (absolute vs conditional shares) to handle OTHER bucket variation.
Compares distributions across Top/Middle/Trash tiers.

Key Pipeline Truths:
1. OTHER bucket (~0.32 mean mass) differs by tier - must normalize conditionally
2. Author-signature topics (41 high-dominant) excluded from main comparisons
3. Gate 3 filter: prevalence >= 0.10 AND NOT author-dominant

Data sources:
- Topic lookup: results/stage10_correlation_analysis/data_preparation/taxonomy_radway_eda/topic_lookup.parquet
- Author dominance: results/stage10_correlation_analysis/topic_analysis_all_368/tables/topic_author_dominance.parquet
- Book topic probs: results/stage10_correlation_analysis/data_preparation/topic_probabilities/book_topic_probs.parquet
- Topic health: results/stage10_correlation_analysis/topic_analysis_all_368/tables/topic_health_table.parquet
- Book features: results/stage10_correlation_analysis/data_preparation/book_features/book_taxonomy_main_props_wide.parquet

Outputs: results/stage10_correlation_analysis/topic_analysis_all_368/ (figures, tables)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests

import warnings
warnings.filterwarnings('ignore', message='.*IProgress not found.*')
warnings.filterwarnings('ignore', category=UserWarning, module='tqdm.auto')

# Set plotting defaults
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size."""
    x = np.asarray(x)
    y = np.asarray(y)
    gt = np.sum(x[:, None] > y[None, :])
    lt = np.sum(x[:, None] < y[None, :])
    return (gt - lt) / (len(x) * len(y))


def compute_epsilon_squared(kruskal_stat: float, n: int, k: int) -> float:
    """Epsilon-squared effect size for Kruskal-Wallis."""
    return (kruskal_stat - (k - 1)) / (n - k)


def unify_book_id_dtype(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure book_id is Int64 dtype."""
    if 'book_id' in df.columns:
        df['book_id'] = pd.to_numeric(df['book_id'], errors='coerce').astype('Int64')
    return df


def compute_book_group_shares(
    book_topic_probs: pd.DataFrame,
    topic_mapping: pd.DataFrame,
    exclude_author_dominant: bool = True,
    exclude_noise: bool = True,
    apply_gate3: bool = True
) -> pd.DataFrame:
    """
    Compute book-level group shares with dual normalization.

    Returns:
        DataFrame with columns:
        - book_id, rating_tier, taxonomy_main_group
        - abs_share: sum(topic_probs in group) / 1.0
        - cond_share: sum(topic_probs in group) / MODELED_MASS
        - modeled_mass: 1 - OTHER - noise_topics
        - other_share: book-level OTHER mass
        - n_topics_in_group: number of topics contributing
    """
    # Merge taxonomy info
    df = book_topic_probs.merge(
        topic_mapping[['topic_id', 'taxonomy_main_group', 'taxonomy_main_name',
                      'author_dominance_flag', 'label_is_noise', 'taxonomy_is_noise',
                      'gate3_pass']],
        on='topic_id',
        how='left'
    )

    # Apply filters
    if exclude_author_dominant:
        df = df[df['author_dominance_flag'] != 'high']

    if exclude_noise:
        df = df[~df['label_is_noise'].fillna(False).infer_objects(copy=False)]
        df = df[~df['taxonomy_is_noise'].fillna(False).infer_objects(copy=False)]

    if apply_gate3:
        df = df[df['gate3_pass'].fillna(False).infer_objects(copy=False)]

    # Compute OTHER share per book (from original data, before filtering)
    # OTHER is topic_id == -1
    other_by_book = book_topic_probs[book_topic_probs['topic_id'] == -1].groupby('book_id')['prob'].sum().reset_index()
    other_by_book.columns = ['book_id', 'other_share']

    # Compute noise mass per book (from filtered df, if any noise topics remain after filtering)
    noise_mask = df['label_is_noise'].fillna(False).infer_objects(copy=False) | df['taxonomy_is_noise'].fillna(False).infer_objects(copy=False)
    noise_by_book = df[noise_mask].groupby('book_id')['prob'].sum().reset_index()
    noise_by_book.columns = ['book_id', 'noise_share']

    # Compute modeled mass per book (sum of all filtered topics)
    modeled_by_book = df.groupby('book_id')['prob'].sum().reset_index()
    modeled_by_book.columns = ['book_id', 'modeled_mass']

    # Merge other and noise info
    modeled_by_book = modeled_by_book.merge(other_by_book, on='book_id', how='left')
    modeled_by_book = modeled_by_book.merge(noise_by_book, on='book_id', how='left')
    modeled_by_book['other_share'] = modeled_by_book['other_share'].fillna(0)
    modeled_by_book['noise_share'] = modeled_by_book['noise_share'].fillna(0)

    # Compute absolute shares per (book, main_group)
    abs_shares = df.groupby(['book_id', 'taxonomy_main_group']).agg({
        'prob': 'sum',
        'topic_id': 'nunique'
    }).reset_index()
    abs_shares.columns = ['book_id', 'taxonomy_main_group', 'abs_share', 'n_topics_in_group']

    # Merge rating_tier (handle case where it might not be in df)
    if 'rating_tier' in df.columns:
        abs_shares = abs_shares.merge(
            df[['book_id', 'rating_tier']].drop_duplicates(),
            on='book_id',
            how='left'
        )
    elif 'rating_tier' in book_topic_probs.columns:
        # If rating_tier not in df, get it from original book_topic_probs
        abs_shares = abs_shares.merge(
            book_topic_probs[['book_id', 'rating_tier']].drop_duplicates(),
            on='book_id',
            how='left'
        )
    else:
        # Create a dummy rating_tier column if not available
        abs_shares['rating_tier'] = None

    # Merge modeled_mass and compute conditional shares
    abs_shares = abs_shares.merge(modeled_by_book, on='book_id', how='left')
    abs_shares['cond_share'] = abs_shares['abs_share'] / abs_shares['modeled_mass'].replace(0, np.nan)

    # Reorder columns
    result = abs_shares[[
        'book_id', 'rating_tier', 'taxonomy_main_group',
        'abs_share', 'cond_share', 'modeled_mass', 'other_share', 'n_topics_in_group'
    ]].copy()

    return result


def plot_main_group_distribution(main_group: str, df: pd.DataFrame, save_dir: Path):
    """Create violin + box plot for a main group across tiers."""
    group_data = df[df['taxonomy_main_group'] == main_group].copy()

    if len(group_data) == 0:
        return None

    fig = go.Figure()

    tiers = ['top', 'middle', 'trash']
    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)']

    for tier, color in zip(tiers, colors):
        tier_data = group_data[group_data['rating_tier'] == tier]['cond_share']
        if len(tier_data) > 0:
            fig.add_trace(go.Violin(
                y=tier_data,
                name=tier.capitalize(),
                box_visible=True,
                meanline_visible=True,
                fillcolor=f"rgba({color.replace('rgb(', '').replace(')', '')}, 0.6)",
                line_color=color,
                opacity=0.7
            ))

    fig.update_layout(
        title=f"Main Group Distribution: {main_group}",
        yaxis_title="Conditional Share",
        xaxis_title="Rating Tier",
        violinmode="group",
        height=500,
        showlegend=True
    )

    safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in main_group)[:50]
    html_path = save_dir / f"main_group_{safe_name}_violin.html"
    fig.write_html(str(html_path))

    return fig


def plot_subgroup_panel(main_group: str, df: pd.DataFrame, save_dir: Path):
    """Create small multiples panel of subgroup distributions."""
    group_data = df[df['taxonomy_main_group'] == main_group].copy()

    if len(group_data) == 0:
        return None

    subgroups = sorted(group_data['taxonomy_main_name'].dropna().unique())
    n_subgroups = len(subgroups)

    if n_subgroups == 0:
        return None

    # Create subplots
    n_cols = min(3, n_subgroups)
    n_rows = int(np.ceil(n_subgroups / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subgroups,
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    tiers = ['top', 'middle', 'trash']
    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)']

    for idx, subgroup in enumerate(subgroups):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1

        subgroup_data = group_data[group_data['taxonomy_main_name'] == subgroup]

        for tier, color in zip(tiers, colors):
            tier_data = subgroup_data[subgroup_data['rating_tier'] == tier]['within_group_share']
            if len(tier_data) > 0:
                fig.add_trace(
                    go.Violin(
                        y=tier_data,
                        name=tier,
                        box_visible=True,
                        meanline_visible=True,
                        showlegend=(idx == 0),
                        fillcolor=f"rgba({color.replace('rgb(', '').replace(')', '')}, 0.6)",
                        line_color=color,
                        opacity=0.7
                    ),
                    row=row,
                    col=col
                )

    fig.update_layout(
        title=f"Subgroup Distributions: {main_group}",
        height=300 * n_rows,
        showlegend=True
    )

    safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in main_group)[:50]
    html_path = save_dir / f"subgroup_panel_{safe_name}.html"
    fig.write_html(str(html_path))

    return fig


def setup_paths(project_root: Path) -> Dict[str, Path]:
    """Setup all data and output paths."""
    data_prep_dir = project_root / "results" / "stage10_correlation_analysis" / "data_preparation"
    book_features_dir = data_prep_dir / "book_features"
    topic_probs_dir = data_prep_dir / "topic_probabilities"
    taxonomy_radway_dir = data_prep_dir / "taxonomy_radway_eda"
    
    topic_analysis_dir = project_root / "results" / "stage10_correlation_analysis" / "topic_analysis_all_368"
    topic_tables_dir = topic_analysis_dir / "tables"
    
    # Data file paths
    paths = {
        'topic_lookup': taxonomy_radway_dir / "topic_lookup.parquet",
        'author_dominance': topic_tables_dir / "topic_author_dominance.parquet",
        'book_topic_probs': topic_probs_dir / "book_topic_probs.parquet",
        'topic_health': topic_tables_dir / "topic_health_table.parquet",
        'book_wide': book_features_dir / "book_taxonomy_main_props_wide.parquet",
        'goodreads': project_root / "data" / "processed" / "goodreads.csv",
        'output_dir': topic_analysis_dir,
        'fig_dir': topic_analysis_dir / "figures",
        'table_dir': topic_tables_dir,
        'main_group_fig_dir': topic_analysis_dir / "figures" / "main_group_distributions",
        'subgroup_fig_dir': topic_analysis_dir / "figures" / "subgroup_distributions",
    }
    
    # Create output directories
    for d in [paths['output_dir'], paths['fig_dir'], paths['table_dir'], 
              paths['main_group_fig_dir'], paths['subgroup_fig_dir']]:
        d.mkdir(parents=True, exist_ok=True)
    
    return paths


def load_data(paths: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    """Load all required data files."""
    print("=" * 80)
    print("Loading data:")
    print("=" * 80)
    
    data = {}
    
    # Topic lookup (taxonomy mappings)
    data['topic_lookup'] = pd.read_parquet(paths['topic_lookup'])
    print(f"\n✓ Loaded topic_lookup: {data['topic_lookup'].shape}")
    print(f"  Columns: {list(data['topic_lookup'].columns)}")
    
    # Author dominance
    data['author_dominance'] = pd.read_parquet(paths['author_dominance'])
    print(f"\n✓ Loaded author_dominance: {data['author_dominance'].shape}")
    print(f"  High author-dominant topics: {(data['author_dominance']['author_dominance_flag'] == 'high').sum()}")
    
    # Book topic probabilities
    data['book_topic_probs'] = pd.read_parquet(paths['book_topic_probs'])
    print(f"\n✓ Loaded book_topic_probs: {data['book_topic_probs'].shape}")
    
    # Topic health (for prevalence)
    data['topic_health'] = pd.read_parquet(paths['topic_health'])
    print(f"\n✓ Loaded topic_health: {data['topic_health'].shape}")
    
    # Book features (for rating_tier)
    data['book_wide'] = pd.read_parquet(paths['book_wide'])
    print(f"\n✓ Loaded book_wide: {data['book_wide'].shape}")
    
    # Load goodreads for rating_class if needed
    if not paths['goodreads'].exists():
        print(f"\n⚠️  goodreads.csv not found at {paths['goodreads']}")
        data['goodreads'] = None
    else:
        goodreads = pd.read_csv(paths['goodreads'])
        if 'ID' in goodreads.columns:
            goodreads = goodreads.rename(columns={'ID': 'book_id'})
        goodreads['book_id'] = pd.to_numeric(goodreads['book_id'], errors='coerce')
        data['goodreads'] = goodreads
        print(f"\n✓ Loaded goodreads: {goodreads.shape}")
    
    # Unify book_id dtype
    data['book_topic_probs'] = unify_book_id_dtype(data['book_topic_probs'])
    data['book_wide'] = unify_book_id_dtype(data['book_wide'])
    
    if data['goodreads'] is not None:
        data['goodreads'] = unify_book_id_dtype(data['goodreads'])
    
    # Create rating_tier from rating_class in book_wide
    if 'rating_tier' not in data['book_wide'].columns:
        if 'rating_class' in data['book_wide'].columns:
            rating_map = {"good": "top", "bad": "trash", "mid": "middle"}
            data['book_wide']['rating_tier'] = data['book_wide']['rating_class'].map(rating_map)
            print(f"\n✓ Created rating_tier from rating_class in book_wide")
        elif data['goodreads'] is not None and 'Score' in data['goodreads'].columns:
            # Create rating_class from Score quantiles, then map to rating_tier
            book_ratings = data['goodreads']['Score'].dropna()
            if len(book_ratings) > 0:
                low_q, high_q = book_ratings.quantile([0.33, 0.66])
                def assign_rating_class(score):
                    if pd.isna(score):
                        return None
                    if score < low_q:
                        return "bad"
                    elif score <= high_q:
                        return "mid"
                    else:
                        return "good"
                data['goodreads']['rating_class'] = data['goodreads']['Score'].apply(assign_rating_class)
                rating_map = {"good": "top", "bad": "trash", "mid": "middle"}
                data['goodreads']['rating_tier'] = data['goodreads']['rating_class'].map(rating_map)
                data['goodreads'] = unify_book_id_dtype(data['goodreads'])
                data['book_wide'] = data['book_wide'].merge(
                    data['goodreads'][['book_id', 'rating_tier']].drop_duplicates(),
                    on='book_id',
                    how='left'
                )
                print(f"\n✓ Created rating_tier from goodreads Score")
    
    # Merge rating_tier onto book_topic_probs
    if 'rating_tier' in data['book_wide'].columns:
        data['book_topic_probs'] = data['book_topic_probs'].merge(
            data['book_wide'][['book_id', 'rating_tier']].drop_duplicates(),
            on='book_id',
            how='left'
        )
        print(f"\n✓ Merged rating_tier onto book_topic_probs")
        print(f"  Books per tier: {data['book_topic_probs'].groupby('rating_tier')['book_id'].nunique()}")
        print(f"  Rows with rating_tier: {data['book_topic_probs']['rating_tier'].notna().sum()} / {len(data['book_topic_probs'])}")
    else:
        print("\n⚠️  rating_tier not found - will need to handle this in downstream cells")
    
    return data


def build_topic_taxonomy_mapping(data: Dict[str, pd.DataFrame], paths: Dict[str, Path]) -> pd.DataFrame:
    """Build topic_taxonomy_mapping.csv (spine export)."""
    print("=" * 80)
    print("Building topic_taxonomy_mapping.csv (spine):")
    print("=" * 80)
    
    # Start with topic_lookup
    mapping = data['topic_lookup'].copy()
    
    # Merge author dominance info
    mapping = mapping.merge(
        data['author_dominance'][['topic_id', 'author_dominance_flag', 'top_author_share', 'is_author_driven']],
        on='topic_id',
        how='left'
    )
    
    # Merge topic health (for prevalence)
    mapping = mapping.merge(
        data['topic_health'][['topic_id', 'prevalence']],
        on='topic_id',
        how='left'
    )
    
    # Compute Gate 3 pass: prevalence >= 0.10 AND NOT author-dominant
    mapping['gate3_pass'] = (
        (mapping['prevalence'] >= 0.10) &
        (~mapping['is_author_driven'].fillna(False).infer_objects(copy=False))
    )
    
    # Add mapping_notes column (empty for now, can be filled manually later)
    mapping['mapping_notes'] = ''
    
    # Select and order columns for export
    core_cols = ['topic_id', 'label', 'keywords', 'scene_summary', 'label_is_noise']
    taxonomy_cols = ['taxonomy_main_group', 'taxonomy_main_name', 'taxonomy_confidence', 'taxonomy_is_noise']
    taxonomy_secondary_cols = ['taxonomy_secondary_group', 'taxonomy_secondary_name']
    radway_cols = ['radway_phase_name', 'radway_phase', 'radway_main_name', 'radway_confidence', 'radway_is_none']
    quality_cols = ['author_dominance_flag', 'top_author_share', 'prevalence', 'gate3_pass', 'mapping_notes']
    
    # Build final mapping table with available columns
    export_cols = []
    for col_list in [core_cols, taxonomy_cols, taxonomy_secondary_cols, radway_cols, quality_cols]:
        for col in col_list:
            if col in mapping.columns:
                export_cols.append(col)
    
    mapping_export = mapping[export_cols].copy()
    
    # Save
    mapping_path = paths['table_dir'] / "topic_taxonomy_mapping.csv"
    mapping_export.to_csv(mapping_path, index=False)
    print(f"\n✓ Saved topic_taxonomy_mapping.csv: {mapping_path}")
    print(f"  Shape: {mapping_export.shape}")
    print(f"  Topics with taxonomy_main_group: {mapping_export['taxonomy_main_group'].notna().sum()}")
    print(f"  Topics passing Gate 3: {mapping_export['gate3_pass'].sum()}")
    print(f"  High author-dominant topics: {(mapping_export['author_dominance_flag'] == 'high').sum()}")
    
    # Display sample
    print(f"\nSample mapping (first 3 rows):")
    print(mapping_export.head(3).to_string())
    
    return mapping_export


def compute_main_group_shares(data: Dict[str, pd.DataFrame], mapping_export: pd.DataFrame, paths: Dict[str, Path]) -> pd.DataFrame:
    """Compute book_main_group_shares."""
    print("=" * 80)
    print("Computing book_main_group_shares:")
    print("=" * 80)
    
    book_main_group_shares = compute_book_group_shares(
        data['book_topic_probs'],
        mapping_export,
        exclude_author_dominant=True,
        exclude_noise=True,
        apply_gate3=True
    )
    
    print(f"\n✓ Computed book_main_group_shares: {book_main_group_shares.shape}")
    print(f"  Unique books: {book_main_group_shares['book_id'].nunique()}")
    print(f"  Unique main groups: {book_main_group_shares['taxonomy_main_group'].nunique()}")
    print(f"\nMain groups:")
    print(book_main_group_shares['taxonomy_main_group'].value_counts().sort_index())
    
    # Summary statistics
    print(f"\nSummary statistics:")
    print(book_main_group_shares.groupby('taxonomy_main_group').agg({
        'abs_share': ['mean', 'std'],
        'cond_share': ['mean', 'std'],
        'modeled_mass': 'mean',
        'other_share': 'mean'
    }).round(4))
    
    # Save
    main_group_shares_path = paths['table_dir'] / "book_main_group_shares.parquet"
    book_main_group_shares.to_parquet(main_group_shares_path, index=False)
    book_main_group_shares.to_csv(paths['table_dir'] / "book_main_group_shares.csv", index=False)
    print(f"\n✓ Saved to: {main_group_shares_path}")
    
    return book_main_group_shares


def compute_subgroup_shares(data: Dict[str, pd.DataFrame], mapping_export: pd.DataFrame, 
                            book_main_group_shares: pd.DataFrame, paths: Dict[str, Path]) -> pd.DataFrame:
    """Compute book_subgroup_shares (within-group normalization)."""
    print("=" * 80)
    print("Computing book_subgroup_shares:")
    print("=" * 80)
    
    # Merge taxonomy info
    df_sub = data['book_topic_probs'].merge(
        mapping_export[['topic_id', 'taxonomy_main_group', 'taxonomy_main_name',
                       'author_dominance_flag', 'label_is_noise', 'taxonomy_is_noise',
                       'gate3_pass', 'taxonomy_confidence']],
        on='topic_id',
        how='left'
    )
    
    # Apply filters (same as main groups)
    df_sub = df_sub[df_sub['author_dominance_flag'] != 'high']
    df_sub = df_sub[~df_sub['label_is_noise'].fillna(False).infer_objects(copy=False)]
    df_sub = df_sub[~df_sub['taxonomy_is_noise'].fillna(False).infer_objects(copy=False)]
    df_sub = df_sub[df_sub['gate3_pass'].fillna(False).infer_objects(copy=False)]
    
    # Compute subgroup sums per (book, main_group, subgroup)
    # Note: taxonomy_confidence is categorical ('high', 'medium', 'low'), so we use 'first' instead of mean
    subgroup_sums = df_sub.groupby(['book_id', 'taxonomy_main_group', 'taxonomy_main_name']).agg({
        'prob': 'sum',
        'taxonomy_confidence': 'first'  # Use 'first' for categorical data (representative value)
    }).reset_index()
    subgroup_sums.columns = ['book_id', 'taxonomy_main_group', 'taxonomy_main_name',
                             'subgroup_sum', 'mapping_confidence']
    
    # Compute group sums per (book, main_group)
    group_sums = df_sub.groupby(['book_id', 'taxonomy_main_group'])['prob'].sum().reset_index()
    group_sums.columns = ['book_id', 'taxonomy_main_group', 'group_sum']
    
    # Merge to compute within-group shares
    book_subgroup_shares = subgroup_sums.merge(group_sums, on=['book_id', 'taxonomy_main_group'], how='left')
    book_subgroup_shares['within_group_share'] = (
        book_subgroup_shares['subgroup_sum'] / book_subgroup_shares['group_sum'].replace(0, np.nan)
    )
    
    # Merge rating_tier
    if 'rating_tier' in df_sub.columns:
        book_subgroup_shares = book_subgroup_shares.merge(
            df_sub[['book_id', 'rating_tier']].drop_duplicates(),
            on='book_id',
            how='left'
        )
    elif 'rating_tier' in data['book_topic_probs'].columns:
        book_subgroup_shares = book_subgroup_shares.merge(
            data['book_topic_probs'][['book_id', 'rating_tier']].drop_duplicates(),
            on='book_id',
            how='left'
        )
    else:
        book_subgroup_shares['rating_tier'] = None
    
    # Compute absolute share (for reference)
    book_subgroup_shares = book_subgroup_shares.merge(
        book_main_group_shares[['book_id', 'taxonomy_main_group', 'modeled_mass']],
        on=['book_id', 'taxonomy_main_group'],
        how='left'
    )
    book_subgroup_shares['abs_share_subgroup'] = (
        book_subgroup_shares['subgroup_sum'] / book_subgroup_shares['modeled_mass'].replace(0, np.nan)
    )
    
    # Reorder columns
    book_subgroup_shares = book_subgroup_shares[[
        'book_id', 'rating_tier', 'taxonomy_main_group', 'taxonomy_main_name',
        'abs_share_subgroup', 'within_group_share', 'group_sum',
        'mapping_confidence'
    ]].copy()
    
    print(f"\n✓ Computed book_subgroup_shares: {book_subgroup_shares.shape}")
    print(f"  Unique books: {book_subgroup_shares['book_id'].nunique()}")
    print(f"  Unique subgroups: {book_subgroup_shares['taxonomy_main_name'].nunique()}")
    
    # Show subgroups per main group
    print(f"\nSubgroups per main group:")
    for main_group in sorted(book_subgroup_shares['taxonomy_main_group'].dropna().unique()):
        n_subgroups = book_subgroup_shares[book_subgroup_shares['taxonomy_main_group'] == main_group]['taxonomy_main_name'].nunique()
        print(f"  {main_group}: {n_subgroups} subgroups")
    
    # Save
    subgroup_shares_path = paths['table_dir'] / "book_subgroup_shares.parquet"
    book_subgroup_shares.to_parquet(subgroup_shares_path, index=False)
    book_subgroup_shares.to_csv(paths['table_dir'] / "book_subgroup_shares.csv", index=False)
    print(f"\n✓ Saved to: {subgroup_shares_path}")
    
    return book_subgroup_shares


def analyze_main_groups(book_main_group_shares: pd.DataFrame, paths: Dict[str, Path]) -> pd.DataFrame:
    """Main-Group Tier Comparisons: plots + Kruskal-Wallis + effect sizes."""
    print("=" * 80)
    print("Main-Group Tier Comparisons:")
    print("=" * 80)
    
    # Get unique main groups
    main_groups = sorted(book_main_group_shares['taxonomy_main_group'].dropna().unique())
    print(f"\nAnalyzing {len(main_groups)} main groups:")
    
    results_main = []
    
    for main_group in main_groups:
        group_data = book_main_group_shares[
            book_main_group_shares['taxonomy_main_group'] == main_group
        ].copy()
        
        if len(group_data) == 0:
            continue
        
        # Extract data by tier
        top_data = group_data[group_data['rating_tier'] == 'top']['cond_share'].values
        middle_data = group_data[group_data['rating_tier'] == 'middle']['cond_share'].values
        trash_data = group_data[group_data['rating_tier'] == 'trash']['cond_share'].values
        
        # Kruskal-Wallis test
        kruskal_stat = np.nan
        kruskal_p = np.nan
        epsilon_sq = np.nan
        
        all_tiers = []
        for tier_data in [top_data, middle_data, trash_data]:
            if len(tier_data) > 0:
                all_tiers.append(tier_data)
        
        if len(all_tiers) >= 2:
            try:
                kruskal_stat, kruskal_p = kruskal(*all_tiers)
                n_total = len(top_data) + len(middle_data) + len(trash_data)
                epsilon_sq = compute_epsilon_squared(kruskal_stat, n_total, 3)
            except:
                pass
        
        # Pairwise comparisons with Holm correction
        pairwise_results = {}
        for pair_name, (data1, data2) in [
            ('top_vs_trash', (top_data, trash_data)),
            ('top_vs_middle', (top_data, middle_data)),
            ('middle_vs_trash', (middle_data, trash_data))
        ]:
            if len(data1) > 0 and len(data2) > 0:
                try:
                    _, p_val = mannwhitneyu(data1, data2, alternative='two-sided')
                    delta = cliffs_delta(data1, data2)
                    pairwise_results[pair_name] = {'p': p_val, 'delta': delta}
                except:
                    pairwise_results[pair_name] = {'p': np.nan, 'delta': np.nan}
            else:
                pairwise_results[pair_name] = {'p': np.nan, 'delta': np.nan}
        
        # Summary statistics
        result = {
            'taxonomy_main_group': main_group,
            'n_books': len(group_data),
            'top_median': np.median(top_data) if len(top_data) > 0 else np.nan,
            'top_mean': np.mean(top_data) if len(top_data) > 0 else np.nan,
            'top_q1': np.percentile(top_data, 25) if len(top_data) > 0 else np.nan,
            'top_q3': np.percentile(top_data, 75) if len(top_data) > 0 else np.nan,
            'middle_median': np.median(middle_data) if len(middle_data) > 0 else np.nan,
            'middle_mean': np.mean(middle_data) if len(middle_data) > 0 else np.nan,
            'middle_q1': np.percentile(middle_data, 25) if len(middle_data) > 0 else np.nan,
            'middle_q3': np.percentile(middle_data, 75) if len(middle_data) > 0 else np.nan,
            'trash_median': np.median(trash_data) if len(trash_data) > 0 else np.nan,
            'trash_mean': np.mean(trash_data) if len(trash_data) > 0 else np.nan,
            'trash_q1': np.percentile(trash_data, 25) if len(trash_data) > 0 else np.nan,
            'trash_q3': np.percentile(trash_data, 75) if len(trash_data) > 0 else np.nan,
            'kruskal_stat': kruskal_stat,
            'kruskal_p': kruskal_p,
            'epsilon_squared': epsilon_sq,
            'top_vs_trash_delta': pairwise_results['top_vs_trash']['delta'],
            'top_vs_trash_p': pairwise_results['top_vs_trash']['p'],
            'top_vs_middle_delta': pairwise_results['top_vs_middle']['delta'],
            'top_vs_middle_p': pairwise_results['top_vs_middle']['p'],
            'middle_vs_trash_delta': pairwise_results['middle_vs_trash']['delta'],
            'middle_vs_trash_p': pairwise_results['middle_vs_trash']['p'],
        }
        
        results_main.append(result)
        
        # Create plot
        plot_main_group_distribution(main_group, book_main_group_shares, paths['main_group_fig_dir'])
        
        print(f"  ✓ {main_group}: n={len(group_data)}, KW p={kruskal_p:.4f}, ε²={epsilon_sq:.4f}")
    
    # Apply Holm correction to pairwise p-values
    all_pairwise_p = []
    pairwise_indices = []
    for i, r in enumerate(results_main):
        for col in ['top_vs_trash_p', 'top_vs_middle_p', 'middle_vs_trash_p']:
            if not np.isnan(r[col]):
                all_pairwise_p.append(r[col])
                pairwise_indices.append((i, col))
    
    if len(all_pairwise_p) > 0:
        _, p_adj_holm, _, _ = multipletests(all_pairwise_p, method='holm', alpha=0.05)
        
        # Map back
        for idx, (result_idx, col) in enumerate(pairwise_indices):
            results_main[result_idx][f"{col}_adj"] = p_adj_holm[idx]
        # Fill NaN for missing values
        for r in results_main:
            for col in ['top_vs_trash_p', 'top_vs_middle_p', 'middle_vs_trash_p']:
                if f"{col}_adj" not in r:
                    r[f"{col}_adj"] = np.nan
    
    # Create results DataFrame
    main_group_results = pd.DataFrame(results_main)
    
    # Save results
    main_group_results_path = paths['table_dir'] / "main_group_comparisons.parquet"
    main_group_results.to_parquet(main_group_results_path, index=False)
    main_group_results.to_csv(paths['table_dir'] / "main_group_comparisons.csv", index=False)
    
    print(f"\n✓ Saved main group comparison results to: {main_group_results_path}")
    print(f"\nMain group results summary:")
    print(main_group_results[['taxonomy_main_group', 'kruskal_p', 'epsilon_squared',
                             'top_median', 'middle_median', 'trash_median']].to_string(index=False))
    
    return main_group_results


def create_main_group_bar_chart(main_group_results: pd.DataFrame, paths: Dict[str, Path]):
    """Create bar chart of mean conditional shares per tier (macro figure)."""
    print("=" * 80)
    print("Creating macro bar chart of main group shares:")
    print("=" * 80)
    
    # Prepare data for bar chart
    bar_data = []
    for _, row in main_group_results.iterrows():
        main_group = row['taxonomy_main_group']
        for tier in ['top', 'middle', 'trash']:
            bar_data.append({
                'main_group': main_group,
                'tier': tier.capitalize(),
                'mean_share': row[f'{tier}_mean'],
                'median_share': row[f'{tier}_median']
            })
    
    bar_df = pd.DataFrame(bar_data)
    
    # Create bar chart
    fig = px.bar(
        bar_df,
        x='main_group',
        y='mean_share',
        color='tier',
        barmode='group',
        title='Mean Conditional Shares by Main Group and Tier',
        labels={'mean_share': 'Mean Conditional Share', 'main_group': 'Main Group'},
        color_discrete_map={'Top': 'rgb(31, 119, 180)', 'Middle': 'rgb(255, 127, 14)', 'Trash': 'rgb(44, 160, 44)'}
    )
    
    fig.update_layout(height=600, xaxis_tickangle=-45)
    fig.write_html(str(paths['main_group_fig_dir'] / "main_groups_bar_chart.html"))
    
    print(f"✓ Saved bar chart to: {paths['main_group_fig_dir'] / 'main_groups_bar_chart.html'}")


def analyze_subgroups(book_subgroup_shares: pd.DataFrame, main_groups: List[str], paths: Dict[str, Path]) -> pd.DataFrame:
    """Subgroup Distributions per Main Group."""
    print("=" * 80)
    print("Subgroup-within-Group Comparisons:")
    print("=" * 80)
    
    results_subgroup = []
    
    for main_group in main_groups:
        group_data = book_subgroup_shares[
            book_subgroup_shares['taxonomy_main_group'] == main_group
        ].copy()
        
        if len(group_data) == 0:
            continue
        
        subgroups = sorted(group_data['taxonomy_main_name'].dropna().unique())
        print(f"\n{main_group}: {len(subgroups)} subgroups")
        
        # Create panel plot
        plot_subgroup_panel(main_group, book_subgroup_shares, paths['subgroup_fig_dir'])
        
        # Analyze each subgroup
        for subgroup in subgroups:
            sub_data = group_data[group_data['taxonomy_main_name'] == subgroup]
            
            top_data = sub_data[sub_data['rating_tier'] == 'top']['within_group_share'].values
            middle_data = sub_data[sub_data['rating_tier'] == 'middle']['within_group_share'].values
            trash_data = sub_data[sub_data['rating_tier'] == 'trash']['within_group_share'].values
            
            # Kruskal-Wallis
            kruskal_stat = np.nan
            kruskal_p = np.nan
            epsilon_sq = np.nan
            
            all_tiers = []
            for tier_data in [top_data, middle_data, trash_data]:
                if len(tier_data) > 0:
                    all_tiers.append(tier_data)
            
            if len(all_tiers) >= 2:
                try:
                    kruskal_stat, kruskal_p = kruskal(*all_tiers)
                    n_total = len(top_data) + len(middle_data) + len(trash_data)
                    epsilon_sq = compute_epsilon_squared(kruskal_stat, n_total, 3)
                except:
                    pass
            
            # Pairwise comparisons
            top_trash_delta = np.nan
            top_trash_p = np.nan
            if len(top_data) > 0 and len(trash_data) > 0:
                try:
                    _, top_trash_p = mannwhitneyu(top_data, trash_data, alternative='two-sided')
                    top_trash_delta = cliffs_delta(top_data, trash_data)
                except:
                    pass
            
            result = {
                'taxonomy_main_group': main_group,
                'taxonomy_main_name': subgroup,
                'n_books': len(sub_data),
                'top_median': np.median(top_data) if len(top_data) > 0 else np.nan,
                'top_mean': np.mean(top_data) if len(top_data) > 0 else np.nan,
                'middle_median': np.median(middle_data) if len(middle_data) > 0 else np.nan,
                'middle_mean': np.mean(middle_data) if len(middle_data) > 0 else np.nan,
                'trash_median': np.median(trash_data) if len(trash_data) > 0 else np.nan,
                'trash_mean': np.mean(trash_data) if len(trash_data) > 0 else np.nan,
                'kruskal_stat': kruskal_stat,
                'kruskal_p': kruskal_p,
                'epsilon_squared': epsilon_sq,
                'top_vs_trash_delta': top_trash_delta,
                'top_vs_trash_p': top_trash_p,
            }
            
            results_subgroup.append(result)
            
            if not np.isnan(kruskal_p) and kruskal_p < 0.05:
                print(f"  ✓ {subgroup}: KW p={kruskal_p:.4f}, δ={top_trash_delta:.3f}")
    
    # Apply Holm correction
    subgroup_p_values = [r['kruskal_p'] for r in results_subgroup if not np.isnan(r['kruskal_p'])]
    if len(subgroup_p_values) > 0:
        _, p_adj_holm, _, _ = multipletests(subgroup_p_values, method='holm', alpha=0.05)
        
        idx = 0
        for r in results_subgroup:
            if not np.isnan(r['kruskal_p']):
                r['kruskal_p_adj'] = p_adj_holm[idx]
                idx += 1
            else:
                r['kruskal_p_adj'] = np.nan
    
    subgroup_results = pd.DataFrame(results_subgroup)
    
    # Save results
    subgroup_results_path = paths['table_dir'] / "subgroup_comparisons.parquet"
    subgroup_results.to_parquet(subgroup_results_path, index=False)
    subgroup_results.to_csv(paths['table_dir'] / "subgroup_comparisons.csv", index=False)
    
    print(f"\n✓ Saved subgroup comparison results to: {subgroup_results_path}")
    print(f"\nSignificant subgroups (p < 0.05):")
    sig_subgroups = subgroup_results[subgroup_results['kruskal_p'] < 0.05].sort_values('kruskal_p')
    if len(sig_subgroups) > 0:
        print(sig_subgroups[['taxonomy_main_group', 'taxonomy_main_name', 'kruskal_p',
                             'top_vs_trash_delta', 'top_median', 'trash_median']].to_string(index=False))
    else:
        print("  (None found)")
    
    return subgroup_results


def analyze_topic_drivers(subgroup_results: pd.DataFrame, mapping_export: pd.DataFrame, paths: Dict[str, Path]):
    """Within-Subgroup Topic Driver Ranking for significant subgroups."""
    print("=" * 80)
    print("Topic Driver Ranking for Significant Subgroups:")
    print("=" * 80)
    
    # Load topic-level statistics for Cliff's delta
    topic_stats_path = paths['table_dir'] / "topic_leaderboard_all.parquet"
    if topic_stats_path.exists():
        topic_stats = pd.read_parquet(topic_stats_path)
        print(f"✓ Loaded topic statistics: {topic_stats.shape}")
    else:
        print(f"⚠️  topic_leaderboard_all.parquet not found. Skipping topic driver analysis.")
        return
    
    # Merge topic stats with mapping to get taxonomy info
    taxonomy_cols = []
    for col in ['taxonomy_main_group', 'taxonomy_main_name']:
        if col in mapping_export.columns:
            taxonomy_cols.append(col)
        else:
            print(f"⚠️  Expected taxonomy column '{col}' not found in mapping_export.")
    
    export_cols = ['topic_id', 'author_dominance_flag', 'label', 'keywords', 'scene_summary']
    export_cols = [col for col in export_cols if col in mapping_export.columns]
    merge_cols = ['topic_id'] + taxonomy_cols + export_cols[1:]
    
    topic_drivers_all = topic_stats.merge(
        mapping_export[merge_cols],
        on='topic_id',
        how='left'
    )
    
    # Filter to Gate 3 passing topics
    if 'gate3_pass' in topic_drivers_all.columns:
        topic_drivers_all = topic_drivers_all[
            (topic_drivers_all['author_dominance_flag'] != 'high') &
            (topic_drivers_all['gate3_pass'].fillna(False))
        ]
    else:
        topic_drivers_all = topic_drivers_all[
            (topic_drivers_all['author_dominance_flag'] != 'high')
        ]
    
    # Get significant subgroups
    sig_subgroups = subgroup_results[
        (subgroup_results['kruskal_p'] < 0.05) &
        (subgroup_results['kruskal_p'].notna())
    ].copy()
    
    if not all(col in sig_subgroups.columns for col in ['taxonomy_main_group', 'taxonomy_main_name']):
        print("⚠️  'taxonomy_main_group' and/or 'taxonomy_main_name' missing in subgroup_results. Skipping topic driver analysis.")
        return
    
    print(f"\nAnalyzing {len(sig_subgroups)} significant subgroups:")
    
    topic_driver_results = []
    
    for _, sub_row in sig_subgroups.iterrows():
        main_group = sub_row.get('taxonomy_main_group', None)
        subgroup = sub_row.get('taxonomy_main_name', None)
        
        if main_group is None or subgroup is None:
            continue
        
        has_main_group = 'taxonomy_main_group' in topic_drivers_all.columns
        has_main_name = 'taxonomy_main_name' in topic_drivers_all.columns
        
        if not has_main_group or not has_main_name:
            print(f"⚠️  taxonomy columns not found in topic_drivers_all; skipping subgroup ({main_group}, {subgroup})")
            continue
        
        subgroup_topics = topic_drivers_all[
            (topic_drivers_all['taxonomy_main_group'] == main_group) &
            (topic_drivers_all['taxonomy_main_name'] == subgroup)
        ].copy()
        
        if len(subgroup_topics) == 0:
            continue
        
        # Rank by absolute Cliff's delta (Top vs Trash)
        if 'cliffs_top_trash' in subgroup_topics.columns:
            subgroup_topics['abs_cliffs_top_trash'] = subgroup_topics['cliffs_top_trash'].abs()
            subgroup_topics = subgroup_topics.sort_values('abs_cliffs_top_trash', ascending=False)
            
            # Keep top 15 drivers
            top_drivers = subgroup_topics.head(15)
            
            print(f"\n{subgroup} ({main_group}):")
            print(f"  Total topics: {len(subgroup_topics)}")
            print(f"  Top 5 drivers:")
            for idx, (_, topic_row) in enumerate(top_drivers.head(5).iterrows(), 1):
                delta_val = topic_row.get('cliffs_top_trash', np.nan)
                mass_val = topic_row.get('mass', np.nan)
                print(f"    {idx}. {topic_row.get('label', 'N/A')}: δ={delta_val:.3f}, "
                      f"mass={mass_val:.4f}, "
                      f"author_dom={topic_row.get('author_dominance_flag', 'N/A')}")
            
            # Store results
            for _, topic_row in top_drivers.iterrows():
                topic_driver_results.append({
                    'taxonomy_main_group': main_group,
                    'taxonomy_main_name': subgroup,
                    'topic_id': topic_row.get('topic_id', np.nan),
                    'label': topic_row.get('label', ''),
                    'cliffs_top_trash': topic_row.get('cliffs_top_trash', np.nan),
                    'abs_cliffs_top_trash': topic_row.get('abs_cliffs_top_trash', np.nan),
                    'top_median': topic_row.get('top_median', np.nan),
                    'trash_median': topic_row.get('trash_median', np.nan),
                    'prevalence': topic_row.get('prevalence', np.nan),
                    'mass': topic_row.get('mass', np.nan),
                    'author_dominance_flag': topic_row.get('author_dominance_flag', ''),
                    'keywords': topic_row.get('keywords', ''),
                    'scene_summary': topic_row.get('scene_summary', ''),
                })
    
    # Create DataFrame
    topic_drivers_df = pd.DataFrame(topic_driver_results)
    
    # Save
    if len(topic_drivers_df) > 0:
        drivers_path = paths['table_dir'] / "subgroup_topic_drivers.parquet"
        topic_drivers_df.to_parquet(drivers_path, index=False)
        topic_drivers_df.to_csv(paths['table_dir'] / "subgroup_topic_drivers.csv", index=False)
        print(f"\n✓ Saved topic drivers to: {drivers_path}")
    else:
        print("\n⚠️  No topic drivers found")


def main():
    """Main execution function."""
    print(f"✓ PROJECT_ROOT: {PROJECT_ROOT}")
    
    # Setup paths
    paths = setup_paths(PROJECT_ROOT)
    
    # Load data
    data = load_data(paths)
    
    # Build topic taxonomy mapping
    mapping_export = build_topic_taxonomy_mapping(data, paths)
    
    # Compute main group shares
    book_main_group_shares = compute_main_group_shares(data, mapping_export, paths)
    
    # Compute subgroup shares
    book_subgroup_shares = compute_subgroup_shares(data, mapping_export, book_main_group_shares, paths)
    
    # Analyze main groups
    main_group_results = analyze_main_groups(book_main_group_shares, paths)
    
    # Create bar chart
    create_main_group_bar_chart(main_group_results, paths)
    
    # Analyze subgroups
    main_groups = sorted(book_main_group_shares['taxonomy_main_group'].dropna().unique())
    subgroup_results = analyze_subgroups(book_subgroup_shares, main_groups, paths)
    
    # Analyze topic drivers
    analyze_topic_drivers(subgroup_results, mapping_export, paths)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

