# Methods and design decisions

This document explains what each stage does, which parameters it uses (all in the config, defaults shown), and why
the method was chosen, including the alternatives that were tried on the BMEG5001 data and rejected.

## Acquisition facts that shape the analysis

From the microscope settings of the original data set (Leica, HyD detectors):

- HC PL APO 10x/0.40 **dry**, zoom 1.8 → 0.63 µm/px; z-step 3.16 µm; 405 + 543 nm excitation simultaneously.
- C1 = HyD 430–520 nm (DAPI), C2 = HyD 550–700 nm (rhodamine-phalloidin, F-actin), C3 = transmitted light.
- HyD photon counting: integer counts with variance ≈ mean (Poisson; checked on GelMA haze patches: mean 0.43–0.86,
  variance 0.51–0.76). Signals are very low (often 0–50 counts).
- The optical section of a 0.4 NA dry objective is ~6–10 µm and there is axial focal shift (refractive-index
  mismatch), so z is blurred and stretched → **shape is measured in xy** (projection of each 3D object); the 3D AR is
  only reported.
- The z-spacing is not stored in the exported TIFF metadata; every output written by the pipeline carries it
  (1.26 × 1.26 × 3.16 µm after 2×2 binning).

## Stage 1 – preprocessing (`preprocess.py`)

Per channel: 2×2 xy mean binning → Gaussian denoise (σ = 0.5, 1, 1 voxels in z, y, x) → per-slice background by grey
opening (40 µm) on a 4× downsampled slice, smoothed and upsampled → subtraction → division by the stack's 99.9th
percentile (saved as uint16, value = relative intensity × 10 000).

- Why percentile normalisation: in empty background the photon counts are mostly 0, so a MAD noise estimate collapses
  to ~0. Thresholds are chosen adaptively per stack in later stages instead.
- Two backgrounds are saved (both 4× downsampled, in counts per raw pixel):
  - `*_bg4.tif`, the opening background (lower envelope), used by the cells stage;
  - `*_bgmed4.tif`, a local mean (median filter over 50 µm, grey dilation 20 µm, smoothing), used by the classical
    nuclei detector. The opening underestimates the mean of autofluorescent haze (GelMA: mean ~0.5 counts, envelope
    ~0.02), which made a Poisson test far too permissive. The dilation compensates for dark PCL-fibre shadows that
    pull the median down at pore edges.
- Transmitted light: binned mean projection over z (`*_tl.tif`).

## Stage 2 – nuclei (`nuclei.py`)

**StarDist (default).** The pretrained `2D_versatile_fluo` model runs on every z-slice of the normalised DAPI stack
(xy Gaussian σ = 1, percentile normalisation 1–99.8, prob 0.5, NMS 0.4; 2D objects < 12 px dropped).

1. *Stitching*: 2D labels of consecutive slices are linked greedily by IoU (≥ 0.25).
2. *z-splitting*: a nucleus has a single intensity maximum along z. A chain is cut at a slice whose mean intensity is
   < 0.75 × the smaller of the peaks above and below (two nuclei at different depths stitched together); the minimum
   slice is dropped.
3. *Filtering* false positives: largest 2D area > 450 µm² (haze / fibre sheets), volume < 100 µm³ (specks), mean
   intensity < 1.5 × the mean of a 3-px ring around the object (not brighter than its surroundings).

Why: classical thresholds (Li, Otsu, triangle, Poisson significance) could not separate nuclei from GelMA haze and
PCL-fibre autofluorescence, which look like textured bands / sheets of ~0.4–2 counts in the DAPI channel. A global
threshold either captured the haze (GS-HeLa-G5: > 900 false seeds) or was unstable. StarDist is shape-based
(star-convex objects) and ignores flat bands.

**Classical (fallback, `method = "classical"`).** Seeds = local maxima of DAPI smoothed at nucleus scale whose value
exceeds `B + max(6·sd_Poisson(B), 1·B)` with B the local-mean background; watershed of −DAPI within the significant
mask; each nucleus trimmed to voxels ≥ 30 % of its own peak above background; < 80 µm³ removed. It needs no TensorFlow
and is used in the tests, but it over-segments hazy hydrogel stacks.

## Stage 3 – cells (`cells.py`, `noise.py`)

1. Foreground: actin counts smoothed with σ = (0.7, 1, 1) must exceed `B + max(6·sd, 1·B)`, with B the opening
   background and sd the Poisson s.d. of the smoothed background
   (`sd = sqrt(max(B, 0.1) · Σw² / bin²)`, `Σw² = 1 / ((2√π)³ σz σy σx)`).
   The 50 µm median background is not used here because inside clusters / spheroids it rises to cell level and whole
   cells were rejected (cells shrank to their nuclei).
2. Among significant voxels a per-stack Li threshold on log(excess over B) removes dim fibre / hydrogel
   autofluorescence. (A per-cell trim at 25 % of the cell's p95 was tried and also shrank cells to their nuclei.)
3. Foreground ∪ nuclei, holes filled slice by slice, components < 50 µm³ removed.
4. Seeded 3D watershed on the smoothed actin (cortical actin forms bright ridges between touching cells),
   markers = nuclei, mask = foreground. Foreground not reachable from a nucleus is discarded, so autofluorescent patches
   without a nucleus never become cells. Each cell keeps only the part connected to its own nucleus.
   Cell label *i* is nucleus label *i*.
5. Clusters = connected components of all cells.

## Stage 4 – measurement (`measure.py`)

For every label: xy projection (any over z) → `regionprops` moment ellipse → major/minor axis, `AR_xy`, orientation
(converted to degrees from +x towards image-up; verified on synthetic ellipses at 0/30/90/150°), solidity; voxel
volume; centroid in µm; 3D AR from the eigenvalues of the coordinate covariance (needs ≥ 3 slices);
`touches_xy_border` and `touches_z_end` flags. Objects with projected area < 20 µm² are not reported.
Cells also get `cluster_id`, `cluster_n_cells`, `is_sheet` (projected area > 2500 µm²) and their nucleus volume.

## Stage 5 – fibres (`fibres.py`)

TL mean → flat-field (divide by a Gaussian, σ = 60 px) → fibres are dark → Otsu (dark class), components < 300 px
removed, opening with a disk of radius 2. Local orientation from the structure tensor (gradient σ = 2 px, integration
σ = 6 px ≈ 7.6 µm): fibre axis ⟂ dominant gradient; coherence = (λ1 − λ2)/(λ1 + λ2).
For each object centroid: nearest fibre pixel (Euclidean distance transform) → distance, fibre angle and coherence
there; `on_fibre` = distance ≤ 10 µm; `align_deg` = |object axis − fibre axis| folded to 0–90°.

## Stage 6 – statistics (`stats.py`)

See the README for the summary. Details:

- The block factor (team) is included because almost every team made all conditions, so images of one team are not
  independent. The intraclass correlation in the original data was small (≈ 0.1).
- Type II sums of squares because the design is unbalanced.
- Contrasts use the pooled residual variance of the full model; block terms cancel in within-block comparisons. With
  treatment coding the contrast vector is the difference of two design rows; effects are back-transformed with exp()
  to AR ratios.
- Holm adjustment is applied separately within each cell line (3 contrasts for 3 groups).
- Alignment index used in Fig. 3: A = mean cos 2Δθ per image (1 = parallel, 0 = random).

## Stage 7 – figures and tables (`report.py`)

Journal style (183 mm max width, Arial 5–7 pt, 0.5 pt lines, Okabe–Ito colours + marker shapes, vector PDF with
embedded TrueType fonts, 600 dpi RGB PNG). Every plotted value is written to `source_data_*.csv`.
Table superscript letters are a compact letter display (insert–absorb) of the Holm-adjusted ANOVA contrasts:
within a cell line, groups that share a letter do not differ at α = 0.05.
