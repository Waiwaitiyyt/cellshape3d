# Case study: UWA BMEG5001 2026 scaffold experiment

## Data

- HEK and HeLa cells in three constructs: **G** (GelMA hydrogel), **GS** (GelMA-infused PCL fibre scaffold) and
  **S** (PCL fibre scaffold); stained with DAPI and rhodamine-phalloidin.
- 70 confocal z-stacks (one construct per team per condition, 13 teams), Leica HC PL APO 10x/0.40 dry, zoom 1.8,
  0.63 µm/px, z-step 3.16 µm, 3 channels (DAPI, F-actin, transmitted light).
- Folder layout: `2026 Confocal data/<GelMa (G)|GelMa+Scaffold (GS)|Scaffold (S)>/<HEK|HeLa>/2026-BMEG5001_2026-<group>-<line>-G<team>_001.tif`.
- Crosslinking time (odd teams 3 min, even teams 1 min) is confounded with team; per-group medians by crosslink time
  showed no consistent difference.

## Reproducing the analysis

```bash
pip install -e ".[stardist,docx]"
cellshape3d list -c configs/bmeg5001_2026.toml        # should list 70 stacks
cellshape3d all  -c configs/bmeg5001_2026.toml        # writes to ./results
```

`configs/bmeg5001_2026.toml` assumes the repository sits in the same folder as `2026 Confocal data`; edit `raw_dir`
otherwise. The nuclei stage needs TensorFlow (StarDist); the original run used a cloud workspace for that stage only.

The package is a refactor of the original step scripts (`analysis/3D_pipeline/scripts/step0…step6`). It was checked
against the original outputs:

- one full stack (S-HEK-G7_001) re-run through preprocess → cells → measure → fibres, using the original nuclei:
  backgrounds, TL projection, fibre mask and coherence were bit-identical; aspect ratio, orientation, area and border
  flags were identical for all 1,703 nuclei and 1,703 cells; normalised intensities differed by ≤ 1 grey level in
  0.1–0.3 % of voxels and one voxel of one cell changed (floating-point differences between library versions);
- stats, figures and tables re-run on the original stage-5 tables for all 70 images: ANOVA, contrasts,
  non-parametric tests, per-image summaries, Table 1/2 and all figure source data were identical.

## Main results

Statistical unit = image; n (images) per group G/GS/S: HEK 11/11/11, HeLa 12/12/13; 57,209 nuclei and 57,209 cells measured before inclusion criteria.

Two-way ANOVA on log(per-image median AR), scaffold × cell line + team (type II):

| Metric | Scaffold | Cell line | Scaffold × cell line | Team (block) |
|---|---|---|---|---|
| Nucleus AR | F(2,52) = 23.37, P = 5.7 × 10⁻⁸ | F(1,52) = 0.29, P = 0.59 | F(2,52) = 2.32, P = 0.11 | F(12,52) = 1.59, P = 0.12 |
| Cell AR | F(2,52) = 53.58, P = 2.3 × 10⁻¹³ | F(1,52) = 0.85, P = 0.36 | F(2,52) = 3.05, P = 0.056 | F(12,52) = 1.47, P = 0.17 |

Within-line scaffold contrasts (AR ratio, 95% CI, Holm-adjusted P):

| Metric | Line | GS vs G | S vs G | S vs GS |
|---|---|---|---|---|
| Nucleus | HEK | 1.054 (1.018–1.091), P = 0.0073 | 1.057 (1.021–1.094), P = 0.0071 | 1.003 (0.969–1.039), P = 0.86 |
| Nucleus | HeLa | 1.055 (1.020–1.091), P = 0.0044 | 1.105 (1.070–1.142), P = 2.9 × 10⁻⁷ | 1.048 (1.014–1.082), P = 0.0056 |
| Cell | HEK | 1.112 (1.049–1.179), P = 0.0013 | 1.188 (1.120–1.260), P = 9.9 × 10⁻⁷ | 1.069 (1.007–1.134), P = 0.028 |
| Cell | HeLa | 1.074 (1.015–1.136), P = 0.014 | 1.266 (1.198–1.337), P = 4.7 × 10⁻¹¹ | 1.179 (1.115–1.245), P = 4.1 × 10⁻⁷ |

Diagnostics: residuals approximately normal for cell AR (Shapiro–Wilk P = 0.43; Levene P = 0.95); mild
non-normality for nucleus AR (P = 0.019, one GS-HeLa outlier image), where the Kruskal–Wallis / Mann–Whitney
analysis gives the same conclusions.

Interpretation:

1. Scaffold has a strong effect on nucleus and cell AR. Cell line has no main effect, and the team block explains
   little variance.
2. GS and S are both more elongated than G in both cell lines (+5–27 % AR). S vs GS differs clearly in HeLa but not in
   HEK nuclei (P = 0.86); for HEK cells it is borderline (ANOVA P = 0.028, Mann–Whitney P = 0.057).
3. Elongated cells align with the local fibre axis: alignment index A = ⟨cos 2Δθ⟩ = 0.43 ± 0.04 (GS) and
   0.63 ± 0.03 (S), both P < 10⁻⁶ against 0. Alignment increases with elongation (per-image Spearman ρ of AR vs |Δθ|:
   GS −0.29 ± 0.04, S −0.39 ± 0.02).
