"""Stage 3: 3D cell segmentation from F-actin, seeded by the nuclei of stage 2.

1. Foreground = actin significantly above the lower-envelope background B (opening background of stage 1):
   smoothed counts > B + max(k * Poisson sd(B), f * B). The 50-um median background is NOT used here: inside cell
   clusters / spheroids it rises to cell level and whole cells were rejected (cells shrank to their nuclei).
   Second stage: among significant voxels, a per-stack Li threshold on log(excess over B) removes dim PCL-fibre /
   GelMA autofluorescence (a per-cell trim at 25 % of the cell's p95 was tried and also shrank cells to nuclei).
   Union with the nuclei (a nucleus is always inside its cell), holes filled slice by slice, specks removed.
2. Seeded watershed on the smoothed actin image (cortical actin forms bright ridges between touching cells),
   markers = 3D nuclei, mask = foreground. Foreground not reachable from any nucleus is discarded, so autofluorescent
   fibre / haze patches without a nucleus never become cells.
3. Only the part of each cell connected to its own nucleus is kept.
4. clusters = connected components of all cells (isolated cell vs. cluster / spheroid).
Cell label i always corresponds to nucleus label i.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, segmentation

from .config import Config
from .morph import remove_small_objects
from .noise import significance_threshold


def fill_holes_2d(m: np.ndarray) -> np.ndarray:
    return np.stack([ndi.binary_fill_holes(s) for s in m])


def keep_nucleus_component(cells: np.ndarray, nuc: np.ndarray) -> np.ndarray:
    """For every cell keep only the connected part(s) that contain its own nucleus."""
    out = np.zeros_like(cells)
    for i, sl in enumerate(ndi.find_objects(cells), start=1):
        if sl is None:
            continue
        keep = cells[sl] == i
        lab, nl = ndi.label(keep)
        if nl > 1:
            ids = np.unique(lab[(nuc[sl] == i) & keep])
            keep = np.isin(lab, ids[ids > 0])
        out[sl][keep] = i
    return out


def segment_cells(x: np.ndarray, B: np.ndarray, nuc: np.ndarray, cfg: Config) -> dict:
    """x: binned actin counts, B: background (same grid), nuc: 3D nucleus labels.

    Returns dict(cells uint16, fg bool, clusters uint16, smoothed float, info dict).
    """
    p, a = cfg.cells, cfg.acquisition
    sp = a.spacing
    vox = sp[0] * sp[1] * sp[2]
    nuc = nuc.astype(np.int32)
    xs = ndi.gaussian_filter(x, p.smooth_zyx)
    cand = xs > significance_threshold(B, p.smooth_zyx, p.k_poisson, p.f_contrast, p.b_floor, a.bin)
    ex = (xs - B)[cand]
    T2 = float(np.exp(filters.threshold_li(np.log(np.maximum(ex, 1e-3))))) if ex.size > 100 else 0.0
    fg = (cand & ((xs - B) > T2)) | (nuc > 0)
    fg = remove_small_objects(fill_holes_2d(fg), max(int(p.min_fg_um3 / vox), 1))
    cells = segmentation.watershed(xs, markers=nuc, mask=fg)
    cells = keep_nucleus_component(cells, nuc) if nuc.max() else np.zeros_like(cells)
    cells = cells.astype(np.uint16)
    clusters, ncl = ndi.label(cells > 0)
    info = dict(T2_excess_counts=T2, n_cells=int(len(np.unique(cells)) - 1), n_nuclei=int(nuc.max()),
                n_clusters=int(ncl), fg_frac=float(fg.mean()), cell_frac=float((cells > 0).mean()))
    return dict(cells=cells, fg=fg, clusters=clusters.astype(np.uint16), smoothed=xs, info=info)
