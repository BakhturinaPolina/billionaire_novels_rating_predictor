#!/usr/bin/env python3
import pandas as pd, numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from src.common.config import load_config, resolve_path

# Load config
try:
    config_path = Path("configs/scoring.yaml")
    if not config_path.exists():
        config_path = Path(__file__).parent.parent.parent / "configs" / "scoring.yaml"
    cfg = load_config(config_path)
    
    AP_PARQUET = resolve_path(Path(cfg["inputs"]["ap_composites_parquet"]))
    OUT_CSV = resolve_path(Path(cfg["output"]["statistics_csv"]))
except Exception as e:
    # Fallback to defaults if config loading fails
    print(f"Warning: Could not load config, using defaults: {e}")
    AP_PARQUET = "results/topics/ap_composites.parquet"
    OUT_CSV = "results/analysis/AP_composites_deltas_with_BHFDR.csv"

def main():
    comp = pd.read_parquet(AP_PARQUET)
    comp_names = [c for c in comp.columns if '_' in c and c[1]== '_']
    rows=[]
    for c in comp_names:
        for g in ['Top','Medium','Trash']:
            gvals = comp.loc[comp['Group']==g, c].dropna()
            rvals = comp.loc[comp['Group']!=g, c].dropna()
            if len(gvals)>=5 and len(rvals)>=5:
                try: stat, p = mannwhitneyu(gvals, rvals, alternative='two-sided')
                except ValueError: stat, p = ttest_ind(gvals, rvals, equal_var=False)
            else: p = np.nan
            delta = (gvals.mean() - comp[c].mean())*100.0
            rows.append({'Group':g,'Composite':c,'Delta_pp':delta,'p_raw':p})
    df = pd.DataFrame(rows)
    mask = df['p_raw'].notna()
    rej, p_adj, _, _ = multipletests(df.loc[mask,'p_raw'], method='fdr_bh')
    df.loc[mask,'p_adj'] = p_adj
    df['sig'] = ''
    df.loc[mask & (df['p_adj']<0.05),'sig'] = '*'
    df.loc[mask & (df['p_adj']<0.01),'sig'] = '**'
    df.loc[mask & (df['p_adj']<0.001),'sig'] = '***'
    df['Delta_pp'] = df['Delta_pp'].round(2)
    df['p_raw'] = df['p_raw'].round(4)
    df['p_adj'] = df['p_adj'].round(4)
    df = df.sort_values(['Group','Delta_pp'], ascending=[True, False])
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

if __name__ == "__main__":
    main()
