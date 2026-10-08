# Output files

All paths are relative to `[data] out_dir`; `{img}` is the image id parsed from the file name. Label images and
masks are calibrated ImageJ TIFFs (µm, with z spacing), on the binned grid (1.26 × 1.26 × 3.16 µm by default).

## 00_inventory/

| file | content |
|---|---|
| `inventory.csv` | one row per stack: image, group, cell_line, team, path, shape, axes, dtype, stored calibration, n_z, per channel p50/p99/p99.9/max and the mean z-profile |
| `mip/{img}_MIP.tif` | maximum-intensity projection per channel (raw resolution) |
| `contact_{group}_fluo.png`, `contact_{group}_TL.png` | contact sheets (red = actin, blue/cyan = DAPI; grey = TL) |

## 01_preprocess/

| file | content |
|---|---|
| `{img}_dapi.tif`, `{img}_actin.tif` | uint16 Z Y X, relative intensity × 10 000 (1.0 = stack p99.9 after background subtraction) |
| `{img}_{ch}_bg4.tif` | opening background, counts per raw pixel, 4× downsampled |
| `{img}_{ch}_bgmed4.tif` | local-mean background, counts per raw pixel, 4× downsampled |
| `{img}_tl.tif` | float32 Y X, binned mean of transmitted light |
| `{img}_pre.json` | parameters, normalisation reference, background levels, per-slice p99 profile |
| `qc/{img}_pre.png` | DAPI / actin MIP, brightest actin slice, TL |

## 02_nuclei/

| file | content |
|---|---|
| `{img}_nuclei.tif` | uint16 3D labels |
| `{img}_nuclei_features.csv` | every candidate: label, vol_um3, max_area2d_um2, n_slices, mean_in, mean_ring, contrast, keep, new_label |
| `{img}_nuc.json` | parameters; counts after stitching / z-splitting / filtering; per-slice 2D counts |
| `qc/{img}_nuc.png` | DAPI MIP, projected nucleus outlines, one slice |

## 03_cells/

| file | content |
|---|---|
| `{img}_cells.tif` | uint16 3D labels (cell *i* ↔ nucleus *i*) |
| `{img}_fg.tif` | uint8 actin foreground (255) |
| `{img}_clusters.tif` | uint16 connected components of all cells |
| `{img}_cells.json` | parameters, Li threshold (`T2_excess_counts`), n_cells, n_clusters, foreground fraction |
| `qc/{img}_cells.png`, `qc_crops/{img}_crops.png` | outlines on the actin MIP / one slice; zoomed quadrants (`--crops`) |

## 04_measure/ – `{img}_nuclei.csv`, `{img}_cells.csv`

| column | unit | meaning |
|---|---|---|
| image, group, cell_line, team | | identifiers |
| label | | label in the 3D label image |
| z_um, y_um, x_um | µm | centroid |
| volume_um3 | µm³ | voxel count × voxel volume |
| n_slices | | z-extent in slices |
| proj_area_um2 | µm² | area of the xy projection |
| major_um, minor_um | µm | axes of the moment-equivalent ellipse of the projection |
| AR_xy | | major / minor |
| orientation_deg | ° | major-axis angle, 0–180, from +x towards image-up |
| solidity | | projection area / convex-hull area |
| AR_3d | | inertia-ellipsoid AR (NaN if < 3 slices) |
| touches_xy_border, touches_z_end | bool | object reaches the xy edge / first or last slice |
| mean_intensity | rel. | mean normalised DAPI (nuclei) or actin (cells) |
| cluster_id, cluster_n_cells | | *cells only*: connected cluster and its size |
| is_sheet | bool | *cells only*: projected area > `sheet_area_um2` |
| nucleus_volume_um3 | µm³ | *cells only*: volume of the cell's own nucleus |

## 05_fibres/

`{img}_fibre_mask.tif`, `{img}_fibre_angle.tif` (°), `{img}_coherence.tif` (Y X), `qc/{img}_fibres.png`, and
`{img}_nuclei_align.csv`, `{img}_cells_align.csv` = the stage-4 tables plus:

| column | meaning |
|---|---|
| dist_to_fibre_um | distance from the centroid to the nearest fibre pixel |
| on_fibre | dist ≤ `on_fibre_um` |
| fibre_angle_deg, fibre_coherence | at that fibre pixel |
| align_deg | object–fibre angle folded to 0–90° |

Images outside `fibre_groups` get a copy of the stage-4 table without these columns.

## 06_stats/

| file | content |
|---|---|
| `all_nuclei.csv`, `all_cells.csv` | all objects of all images + `included` |
| `per_image_summary.csv` | per image: `{nuc,cell}_n`, `_AR_median`, `_AR_mean`, `_frac_AR_gt2`, `_align_n`, `_align_median_deg`, `_frac_aligned`, `_align_elong_median_deg`, `_align_elong_n`, `_ok` (≥ min objects), `cell_sheet_frac_area` |
| `anova_tests.csv` | ANOVA table (term, df, df_resid, F, p), diagnostics, within-line contrasts (t, p, p_holm, AR_ratio, 95% CI, n) |
| `group_tests.csv` | Kruskal–Wallis, Mann–Whitney (p_holm), alignment Wilcoxon tests |
| `fig_AR_by_group.*`, `fig_alignment.*` | quick-look plots |
| `figures/Fig1–4.{pdf,png}`, `figures/source_data_*.csv` | publication figures and their data |
| `Table1_AR_summary.*`, `Table2_alignment_summary.*` | CSV, Markdown and (with python-docx) Word tables |
