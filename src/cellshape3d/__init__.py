"""cellshape3d: cell and nucleus geometry from 3D confocal z-stacks.

Stages (each has its own output folder under the configured ``out_dir``):

0. inventory   - stack shapes, metadata, intensity statistics, MIPs, contact sheets
1. preprocess  - channel split, xy binning, denoising, background estimation, normalisation
2. nuclei      - 3D nuclei from the DAPI channel (StarDist2D per slice + z-stitching, or classical)
3. cells       - seeded 3D watershed of the F-actin channel from the nuclei
4. measure     - per-object morphology (xy aspect ratio, orientation, area, volume, 3D AR, flags)
5. fibres      - fibre orientation / coherence from transmitted light, object-to-fibre alignment
6. stats       - inclusion, per-image summaries, two-way ANOVA + non-parametric tests
7. figures / tables - publication figures and summary tables
"""
__version__ = "0.1.0"

from .config import Config, load_config  # noqa: F401
