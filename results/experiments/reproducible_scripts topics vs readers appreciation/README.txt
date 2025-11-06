
Reproducible Pipeline (run from a folder containing the input CSVs):
1) python 00_prepare_data_and_groups.py
2) python 01_build_AP_composites.py
3) python 02_group_deltas_with_BHFDR.py
4) python 03_interaction_plots_PD.py
5) python 04_star_charts_and_pdf.py

Artifacts produced:
- prepared_books.parquet, ap_composites.parquet
- AP_composites_deltas_with_BHFDR.csv
- PD_AxC_Top.png, PD_DxA_Top.png
- charts_with_stars/*.png
- Consolidated_Table_Plots.pdf
