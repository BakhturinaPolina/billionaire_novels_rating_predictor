#!/usr/bin/env python3
import pandas as pd, numpy as np, textwrap
from pathlib import Path
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from PIL import Image

SIG_CSV = "AP_composites_deltas_with_BHFDR.csv"
OUT_DIR = Path("charts_with_stars")
PDF_OUT = "Consolidated_Table_Plots.pdf"

def table_to_png(df, title, path, topn=12):
    import matplotlib.pyplot as plt
    chunks=[]
    for g in ['Top','Medium','Trash']:
        sub = df[df['Group']==g].head(topn).copy()
        chunks.append(sub.assign(Group=g))
    cat = pd.concat(chunks, ignore_index=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    tbl = ax.table(cellText=cat.values, colLabels=cat.columns, loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    ax.set_title(title); fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches='tight'); plt.close(fig)

def plot_deltas_with_stars(group, df, out_path, N=10):
    sub = df[df['Group']==group].sort_values('Delta_pp', ascending=False).head(N).copy()
    labels = [c.replace('_',' ') for c in sub['Composite']]
    plt.figure(figsize=(10,6))
    plt.barh([textwrap.fill(l, 28) for l in labels], sub['Delta_pp'])
    plt.gca().invert_yaxis()
    for i, (v, star) in enumerate(zip(sub['Delta_pp'], sub['sig'])):
        plt.text(v + (0.02 if v>=0 else -0.02), i, star, va='center')
    plt.xlabel("Delta vs Overall (percentage points)")
    plt.title(f"Most Distinctive (+) with BH–FDR: {group}")
    plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close()

def place_image(c, path, x, y, mw, mh):
    img = Image.open(path); w,h = img.size
    s = min(mw/w, mh/h); nw, nh = w*s, h*s
    c.drawImage(path, x, y-nh, width=nw, height=nh, preserveAspectRatio=True, mask='auto')
    return nh

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sig = pd.read_csv(SIG_CSV)
    table_png = OUT_DIR / "Delta_Table_With_Stars.png"
    table_to_png(sig, "Δpp with BH–FDR Stars (Top rows per group)", str(table_png), topn=12)
    top_png = OUT_DIR / "Top_deltas_with_stars.png"
    med_png = OUT_DIR / "Medium_deltas_with_stars.png"
    trash_png = OUT_DIR / "Trash_deltas_with_stars.png"
    plot_deltas_with_stars('Top', sig, str(top_png), 10)
    plot_deltas_with_stars('Medium', sig, str(med_png), 10)
    plot_deltas_with_stars('Trash', sig, str(trash_png), 10)
    axc_png = Path("PD_AxC_Top.png")
    dxa_png = Path("PD_DxA_Top.png")
    c = canvas.Canvas(PDF_OUT, pagesize=A4)
    w,h = A4; m=1.2*cm; y=h-m
    c.setFont("Helvetica-Bold", 14); c.drawString(m, y, "Composites Δpp with BH–FDR + Interaction Plots")
    y -= 0.8*cm; c.setFont("Helvetica", 9)
    c.drawString(m, y, "Δpp tables with BH–FDR stars, star-annotated delta charts, and simple-slope (A×C, D×A) plots.")
    y -= 0.6*cm
    if table_png.exists(): y -= place_image(c, str(table_png), m, y, w-2*m, 12*cm) + 0.4*cm
    c.showPage()
    y = h-m; c.setFont("Helvetica-Bold", 12); c.drawString(m, y, "Star-Annotated Δpp Charts (Top & Trash)")
    y -= 0.7*cm; col_w = (w - 2*m - 0.8*cm)/2
    if top_png.exists(): place_image(c, str(top_png), m, y, col_w, 9*cm)
    if trash_png.exists(): place_image(c, str(trash_png), m+col_w+0.8*cm, y, col_w, 9*cm)
    c.showPage()
    y = h-m; c.setFont("Helvetica-Bold", 12); c.drawString(m, y, "Medium Δpp + Interactions (A×C, D×A)")
    y -= 0.7*cm
    if med_png.exists(): y -= place_image(c, str(med_png), m, y, w-2*m, 8*cm) + 0.3*cm
    if axc_png.exists(): y -= place_image(c, str(axc_png), m, y, w-2*m, 8*cm) + 0.3*cm
    if dxa_png.exists(): y -= place_image(c, str(dxa_png), m, y, w-2*m, 8*cm) + 0.3*cm
    c.showPage(); c.save()
    print(f"Wrote {PDF_OUT}")

if __name__ == "__main__":
    main()
