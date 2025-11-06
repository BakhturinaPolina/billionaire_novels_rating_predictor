#!/usr/bin/env python3
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

AP_PARQUET = "ap_composites.parquet"

def partial_dep_two_lines(pipe, Xc, x_name, mod_name, out_png, title):
    x_vals = np.linspace(Xc[x_name].quantile(0.02), Xc[x_name].quantile(0.98), 60)
    mod_low = Xc[mod_name].quantile(0.25)
    mod_high = Xc[mod_name].quantile(0.75)
    base = Xc.mean().copy()
    for ccol in Xc.columns:
        if ccol.startswith('Author_'): base[ccol]=0.0
    frames=[]
    for level, mval in [('Low',mod_low),('High',mod_high)]:
        grid = pd.DataFrame([base.values]*len(x_vals), columns=Xc.columns)
        grid[x_name] = x_vals; grid[mod_name] = mval
        grid['AC'] = grid['A_Reassurance_Commitment']*grid['C_Explicit_Eroticism']
        grid['DA'] = grid['D_Power_Wealth_Luxury']*grid['A_Reassurance_Commitment']
        probs = pipe.predict_proba(grid)
        top_idx = list(pipe.named_steps['logisticregression'].classes_).index('Top')
        frames.append(pd.DataFrame({'x':x_vals,'p_top':probs[:,top_idx],'Moderator':level}))
    plot_df = pd.concat(frames, ignore_index=True)
    plt.figure(figsize=(8,5))
    for level in ['Low','High']:
        sub = plot_df[plot_df['Moderator']==level]
        plt.plot(sub['x'], sub['p_top'], label=f"{mod_name} {level}")
    plt.xlabel(x_name.replace('_',' ')); plt.ylabel("Predicted P(Top)"); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=200, bbox_inches='tight'); plt.close()

def main():
    comp = pd.read_parquet(AP_PARQUET)
    comps = [c for c in comp.columns if '_' in c and c[1]=='_']
    X = comp[comps].copy()
    X['AC'] = X['A_Reassurance_Commitment'] * X['C_Explicit_Eroticism']
    X['DA'] = X['D_Power_Wealth_Luxury'] * X['A_Reassurance_Commitment']
    X['Pages'] = comp['Pages'].fillna(comp['Pages'].median())
    yr = comp['Year']
    X['Year'] = yr.fillna(yr.median()) if not yr.isna().all() else 0.0
    auth = pd.get_dummies(comp['Author'], prefix='Author', drop_first=True)
    Xc = pd.concat([X, auth], axis=1)
    y = comp['Group'].astype('category')
    pipe = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(penalty='l2', solver='saga', multi_class='multinomial', max_iter=6000, random_state=0))
    pipe.fit(Xc, y)
    partial_dep_two_lines(pipe, Xc, 'A_Reassurance_Commitment','C_Explicit_Eroticism', "PD_AxC_Top.png", "A×C simple slopes for P(Top)")
    partial_dep_two_lines(pipe, Xc, 'A_Reassurance_Commitment','D_Power_Wealth_Luxury', "PD_DxA_Top.png", "D×A simple slopes for P(Top)")
    print("Wrote PD_AxC_Top.png and PD_DxA_Top.png")

if __name__ == "__main__":
    main()
