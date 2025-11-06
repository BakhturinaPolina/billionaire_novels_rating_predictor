#!/usr/bin/env python3
import pandas as pd, numpy as np, re
from pathlib import Path
import sys
from src.common.config import load_config, resolve_path

# Load config
try:
    config_path = Path("configs/scoring.yaml")
    if not config_path.exists():
        config_path = Path(__file__).parent.parent.parent / "configs" / "scoring.yaml"
    cfg = load_config(config_path)
    paths_cfg = load_config(Path("configs/paths.yaml"))
    
    TOPICS_CSV = resolve_path(Path(cfg["inputs"]["topics_csv"]))
    META_CSV = resolve_path(Path(cfg["inputs"]["goodreads_csv"]))
    OUT_PARQUET = resolve_path(Path(cfg["output"]["prepared_books_parquet"]))
except Exception as e:
    # Fallback to defaults if config loading fails
    print(f"Warning: Could not load config, using defaults: {e}")
    TOPICS_CSV = "results/topics/by_book.csv"
    META_CSV = "data/processed/goodreads.csv"
    OUT_PARQUET = "data/processed/prepared_books.parquet"

def extract_year(x):
    if pd.isna(x): return np.nan
    m = re.search(r'(\d{4})', str(x))
    return int(m.group(1)) if m else np.nan

def main():
    topics = pd.read_csv(TOPICS_CSV)
    meta = pd.read_csv(META_CSV)
    topics.columns = [c.replace('\\n',' ').strip() for c in topics.columns]
    meta.columns = [c.replace('\\n',' ').strip() for c in meta.columns]
    topics['Title_key'] = topics['Book_Title'].str.strip().str.lower()
    meta['Title_key'] = meta['Title'].str.strip().str.lower()
    df = topics.merge(meta[['Title_key','Title','Author','RatingsCount','Score','Popularity_ReadingNow','Popularity_Wishlisted','Pages','PublishedDate']], on='Title_key', how='left')
    non_topic = {'Book_Title','Title_key','Title','Author','RatingsCount','Score','Popularity_ReadingNow','Popularity_Wishlisted','Pages','PublishedDate'}
    topic_cols = [c for c in df.columns if c not in non_topic]
    mat = df[topic_cols].fillna(0)
    row_sum = mat.sum(axis=1).replace(0, np.nan)
    for c in topic_cols:
        df[c] = mat[c].div(row_sum)
    for col in ['RatingsCount','Score','Popularity_ReadingNow','Popularity_Wishlisted']:
        df[f'z_{col}'] = (df[col]-df[col].mean())/df[col].std(ddof=0)
    signals = [f'z_{c}' for c in ['RatingsCount','Score','Popularity_ReadingNow','Popularity_Wishlisted']]
    df['popularity_index'] = df[signals].mean(axis=1)
    qs = df['popularity_index'].quantile([0.3333, 0.6667])
    low, high = qs.iloc[0], qs.iloc[1]
    def grp(v):
        if pd.isna(v): return np.nan
        if v<=low: return 'Trash'
        elif v<=high: return 'Medium'
        return 'Top'
    df['Group'] = df['popularity_index'].apply(grp)
    def extract_year(x):
        if pd.isna(x): return np.nan
        m = re.search(r'(\d{4})', str(x))
        return int(m.group(1)) if m else np.nan
    df['Year'] = df['PublishedDate'].apply(extract_year)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {OUT_PARQUET}")

if __name__ == "__main__":
    main()
