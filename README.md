# cellshape3d

[![CI](https://github.com/OWNER/cellshape3d/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/cellshape3d/actions/workflows/ci.yml)

Cell and nucleus **shape** from 3D confocal z-stack TIFFs: segments nuclei (DAPI) and cells (F-actin) in 3D, measures
aspect ratio, orientation, size and volume of every object, measures the local fibre direction from transmitted
light, and compares groups with image-level statistics. It produces publication figures and tables.

It was built for the BMEG5001 (UWA, 2026) scaffold study: HEK and HeLa cells in GelMA (G), GelMA-infused PCL
scaffolds (GS) and PCL scaffolds (S), imaged at 10x/0.40. Groups, cell lines, channels, voxel size and file naming
are all set in a config file, so other designs work without code changes.

```
raw z-stack (Z C Y X)
  └─ 1 preprocess ── bin 2x2, denoise, background, normalise
       ├─ 2 nuclei ── StarDist2D per slice → z-stitch → z-split → filter      (DAPI)
       │    └─ 3 cells ── Poisson-significant actin → seeded 3D watershed      (F-actin)
       │         └─ 4 measure ── xy AR, orientation, area, volume, 3D AR, flags
       └─ TL mean ─────── 5 fibres ── structure tensor → fibre axis, coherence, object–fibre angle
                                └─ 6 stats ── per-image medians → two-way ANOVA (+ block), Kruskal–Wallis, Wilcoxon
                                     └─ figures / tables
```

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/OWNER/cellshape3d.git
cd cellshape3d
pip install -e .                 # core pipeline (classical nuclei detector)
pip install -e ".[stardist]"     # + StarDist / TensorFlow (recommended nucleus detector)
pip install -e ".[docx]"         # + Word export of the tables
pip install -e ".[dev]"          # + pytest, ruff
```

With [uv](https://docs.astral.sh/uv/): `uv venv --python 3.12 && uv pip install -e ".[dev]"`.
On OneDrive / SharePoint folders, create the virtual environment outside the synced folder or set
`UV_LINK_MODE=copy` (OneDrive rejects hard links).

## Quick start

1. Copy `configs/example.toml` and set the data folder, file-name pattern, voxel size, channel order and design.
2. Check that every stack is found and parsed:
   ```bash
   cellshape3d list -c my.toml
   ```
3. Run everything, or one stage at a time (each stage is resumable and skips images that are already done):
   ```bash
   cellshape3d all -c my.toml
   cellshape3d preprocess -c my.toml --only S-HEK-G7_001   # one image
   cellshape3d cells -c my.toml --force --crops            # recompute, with zoomed QC crops
   ```
4. **Look at the QC images** (`*/qc/*.png`) before you use any number.

| stage | command | output folder | what it does |
|---|---|---|---|
| 0 | `inventory` | `00_inventory/` | shape, axes, metadata and intensity statistics per stack; MIPs; contact sheets |
| 1 | `preprocess` | `01_preprocess/` | channel split, 2×2 binning, Gaussian denoise, background (40 µm opening and 50 µm local median), normalised DAPI / actin stacks, TL mean |
| 2 | `nuclei` | `02_nuclei/` | 3D nuclei: StarDist2D per slice → IoU z-stitching → split at z-intensity minima → filters (2D area ≤ 450 µm², volume ≥ 100 µm³, contrast ≥ 1.5) |
| 3 | `cells` | `03_cells/` | actin foreground (Poisson significance + Li threshold on log excess) → seeded 3D watershed from nuclei → clusters |
| 4 | `measure` | `04_measure/` | per nucleus / cell: xy AR, orientation, area, volume, 3D AR, solidity, border flags, cluster size, sheet flag |
| 5 | `fibres` | `05_fibres/` | fibre mask, orientation and coherence from TL; distance to fibre; object–fibre angle |
| 6 | `stats` | `06_stats/` | inclusion criteria, per-image summary, two-way ANOVA + contrasts, non-parametric tests, quick-look plots |
| 7 | `figures`, `tables` | `06_stats/figures/`, `06_stats/Table*` | publication figures (PDF + 600 dpi PNG, with source data), summary tables (CSV, Markdown, Word) |

`python -m cellshape3d …` works the same as `cellshape3d …`.

## Input data

- Multi-channel z-stacks readable by `tifffile` (ImageJ hyperstacks, OME-TIFF, Leica exports); any axis order, as long
  as it has Z, C, Y and X. Set the channel indices in `[acquisition]`.
- **The z-step is often not stored in exported TIFFs**: set `dz_um` explicitly. All outputs are written as
  calibrated ImageJ TIFFs (µm, with z spacing).
- The image id and design factors come from the file name via a regex with named groups
  (`group`, `cell_line`, optional `team` = blocking factor), e.g. for `2026-BMEG5001_2026-GS-HeLa-G3_001.tif`:
  ```toml
  name_pattern = '(?P<group>GS|G|S)-(?P<cell_line>HEK|HeLa)-G(?P<team>\d+)_\d+'
  ```
- Fibre analysis needs a transmitted-light channel (`ch_tl`) and runs only for the `fibre_groups`.

## Measurements

| column | meaning |
|---|---|
| `AR_xy` | **primary shape metric**: major / minor axis of the moment-equivalent ellipse of the object's xy projection |
| `orientation_deg` | major-axis angle, 0–180°, from +x (image right) towards image-up; same convention as `fibre_angle_deg` |
| `proj_area_um2`, `volume_um3` | projected area and voxel volume |
| `AR_3d` | inertia-ellipsoid AR in physical units; reported only (z is blurred at low NA, see [limitations](#known-limitations)) |
| `align_deg` | angle between object axis and local fibre axis, folded to 0–90° (0 = parallel, 45 = random) |
| `fibre_coherence` | structure-tensor coherence at the nearest fibre pixel (~1 = single straight fibre, low at crossings) |

Full column lists for every output are in [docs/outputs.md](docs/outputs.md).

## Statistics

- **Statistical unit = image** (one construct). Objects within an image are not independent; each image is summarised by
  the median of its objects. Images with fewer than 10 included objects are left out of AR tests.
- Inclusion: nuclei not touching the xy border with projected area 30–600 µm²; cells not touching the border,
  ≥ 50 µm² and not a "sheet" (> 2500 µm², continuous actin that cannot be split into cells).
- Primary test: two-way ANOVA (type II) on log(per-image median AR), group × cell line + block (team); within-line
  group contrasts from the same model, Holm-adjusted, reported as AR ratios with 95% CI. Diagnostics: Shapiro–Wilk
  (residuals) and Levene.
- Sensitivity: Kruskal–Wallis per cell line + Holm-adjusted Mann–Whitney U.
- Alignment: elongated objects (AR > 1.5) ≤ 10 µm from a fibre with coherence ≥ 0.6; one-sample Wilcoxon of
  per-image median angles against 45°.

All thresholds are in the config (`[stats]`, `[measure]`, …). Methods and the reasons behind each choice, including
approaches that were tried and rejected, are in [docs/methods.md](docs/methods.md).

## Known limitations

1. **Cell segmentation is the weakest link on fibre scaffolds.** Cells on fibres form continuous actin sheets; at
   10x/0.4 single-cell borders are not visible, so the watershed splits sheets into nucleus territories whose
   outlines are partly arbitrary. Large sheets are excluded, but check `03_cells/qc_crops` (`--crops`). Nucleus AR is
   the more defensible primary metric; cell AR is supportive.
2. With simultaneous 405/543 nm excitation the DAPI channel can carry whole-cell signal and hydrogel
   autofluorescence; some "nuclei" in hydrogel-only samples can be nearly cell-sized.
3. GelMA haze and PCL fibres autofluoresce in the DAPI channel. StarDist plus the filters remove most of it; the
   classical detector (`method = "classical"`, no TensorFlow) does not, and should be used only on clean samples.
4. Shape is measured in xy (projection of each 3D object). The axial resolution of a 0.4 NA dry objective
   (~6–10 µm optical section plus focal shift) makes z-based shape unreliable.
5. TL fibre orientation is a projection through all scaffold layers; at 0/90° crossings it is ambiguous
   (low coherence, excluded).

## Development

```bash
pytest                 # unit tests + end-to-end run on synthetic stacks (~30 s)
ruff check src tests
```

The tests build synthetic 3-channel stacks with known nucleus positions, cell elongation and fibre direction, and
check each stage against them (orientation convention, AR, z-stitching / splitting, false-positive filters, fibre angle
and coherence, ANOVA contrast recovery) before running the whole pipeline through figures and tables.
GitHub Actions runs lint + tests on Ubuntu and Windows for Python 3.10–3.12 (`.github/workflows/ci.yml`).

Repository layout:

```
src/cellshape3d/   config.py io.py preprocess.py noise.py nuclei.py cells.py measure.py fibres.py
                   stats.py report.py qc.py morph.py pipeline.py cli.py
configs/           bmeg5001_2026.toml (the original study), example.toml (template)
docs/              methods.md, outputs.md, case_study_bmeg5001.md
tests/             synthetic-data unit and end-to-end tests
```

## Case study

[docs/case_study_bmeg5001.md](docs/case_study_bmeg5001.md) describes the original data set (70 stacks, 13 teams,
57,209 nuclei and as many cells measured), how to reproduce the analysis with `configs/bmeg5001_2026.toml`, and the main results.
