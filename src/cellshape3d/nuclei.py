"""Stage 2: 3D nuclei from the DAPI channel.

Default ("stardist"): StarDist2D ('2D_versatile_fluo', pretrained) on every z-slice, 2D masks stitched across z into
3D nuclei (greedy IoU linking), chains split at intensity minima along z, and false positives filtered:
largest 2D area > 450 um2 (haze / fibre sheets), volume < 100 um3 (specks), mean intensity < 1.5x its surrounding
ring (not brighter than the surroundings).

Why StarDist: classical thresholds (Li / Otsu / triangle / Poisson significance) could not separate nuclei from
(i) GelMA autofluorescence haze and (ii) PCL-fibre autofluorescence, which appear in the DAPI channel as textured
bands / sheets of ~0.4-2 counts. StarDist is shape-based (star-convex nuclei) and ignores flat bands and haze.

Fallback ("classical", no TensorFlow): local maxima of the smoothed DAPI counts that are significantly above the local
mean background (Poisson test, see :mod:`cellshape3d.noise`), watershed within the significant mask, each nucleus
trimmed to >= 30 % of its own peak. Works on clean stacks; over-segments hazy hydrogel stacks.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .config import Config
from .noise import significance_threshold


def relabel(lab: np.ndarray) -> np.ndarray:
    """Consecutive labels 1..n (order preserved)."""
    ids = np.unique(lab)
    ids = ids[ids > 0]
    lut = np.zeros(int(lab.max()) + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[lab]


def stitch(lab2d: np.ndarray, iou_thr: float) -> np.ndarray:
    """Link independent 2D labels of consecutive slices into 3D objects (greedy by best IoU >= ``iou_thr``)."""
    out = np.zeros_like(lab2d, dtype=np.int32)
    nxt = 1
    for z in range(lab2d.shape[0]):
        cur = lab2d[z]
        ids = np.unique(cur)
        ids = ids[ids > 0]
        mapping = {}
        if z > 0 and ids.size:
            prev = out[z - 1]
            pairs = np.stack([cur.ravel(), prev.ravel()], 1)
            pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
            if len(pairs):
                u, cnt = np.unique(pairs, axis=0, return_counts=True)
                a_cur = dict(zip(*np.unique(cur[cur > 0], return_counts=True)))
                a_prev = dict(zip(*np.unique(prev[prev > 0], return_counts=True)))
                iou = cnt / np.array([a_cur[c] + a_prev[p] - n for (c, p), n in zip(u, cnt)])
                used = set()
                for k in np.argsort(-iou):
                    c, p = u[k]
                    if iou[k] < iou_thr:
                        break
                    if c in mapping or p in used:
                        continue
                    mapping[c] = p
                    used.add(p)
        for c in ids:
            if c not in mapping:
                mapping[c] = nxt
                nxt += 1
        lut = np.zeros(int(cur.max()) + 1, np.int32)
        for c, v in mapping.items():
            lut[c] = v
        out[z] = lut[cur]
    return out


def split_z(lab: np.ndarray, x: np.ndarray, dip: float) -> np.ndarray:
    """Split z-chains at intensity minima.

    A nucleus has one intensity maximum along z. Chains created by stitching nuclei stacked at different depths show
    several maxima: a chain is cut at a slice whose mean intensity < ``dip`` * min(peak above, peak below); the
    minimum slice itself is dropped.
    """
    out = lab.copy()
    nxt = int(lab.max()) + 1
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None or sl[0].stop - sl[0].start < 3:
            continue
        m = lab[sl] == i
        xs = x[sl]
        prof = np.array([xs[k][m[k]].mean() if m[k].any() else 0 for k in range(m.shape[0])])
        cuts = []
        for k in range(1, len(prof) - 1):
            left, right = prof[:k].max(), prof[k + 1:].max()
            if prof[k] < dip * min(left, right) and prof[k] <= prof[k - 1] and prof[k] <= prof[k + 1]:
                cuts.append(k)
        if not cuts:
            continue
        seg = np.searchsorted(np.array(cuts), np.arange(len(prof)), side="right")   # segment index per slice
        cutset = set(cuts)
        sub = out[sl]
        for k in range(len(prof)):
            if k in cutset:
                sub[k][m[k]] = 0
                continue
            if seg[k] > 0:
                sub[k][m[k]] = nxt + seg[k] - 1
        nxt += int(seg.max())
    return out


def nucleus_features(lab: np.ndarray, x: np.ndarray, spacing) -> pd.DataFrame:
    """Per-label volume, largest 2D area, z-extent, mean intensity inside and in a 3-px 2D ring, and contrast."""
    n = int(lab.max())
    if n == 0:
        return pd.DataFrame(columns=["label", "vol_um3", "max_area2d_um2", "n_slices", "mean_in", "mean_ring", "contrast"])
    dz, py, px = spacing
    idx = np.arange(1, n + 1)
    vol = ndi.sum(np.ones_like(lab), lab, idx) * py * px * dz
    mean_in = ndi.mean(x, lab, idx)
    dil = np.stack([ndi.grey_dilation(l, size=(7, 7)) for l in lab])    # ring: 3-px dilation minus all nuclei
    ring = np.where(lab == 0, dil, 0)
    mean_ring = ndi.mean(x, ring, idx)
    area2d = np.zeros(n + 1)
    for z in range(lab.shape[0]):
        area2d = np.maximum(area2d, np.bincount(lab[z].ravel(), minlength=n + 1))
    nz = np.array([s[0].stop - s[0].start if s is not None else 0 for s in ndi.find_objects(lab, max_label=n)])
    return pd.DataFrame(dict(label=idx, vol_um3=vol, max_area2d_um2=area2d[1:] * py * px, n_slices=nz,
                             mean_in=mean_in, mean_ring=mean_ring,
                             contrast=np.asarray(mean_in) / np.maximum(mean_ring, 1e-6)))


def filter_nuclei(lab: np.ndarray, x: np.ndarray, cfg: Config) -> tuple[np.ndarray, pd.DataFrame]:
    """Remove false positives (too large in 2D, too small in 3D, not brighter than the ring); relabel 1..n."""
    p = cfg.nuclei
    f = nucleus_features(lab, x, cfg.acquisition.spacing)
    if f.empty:
        return lab, f.assign(keep=pd.Series(dtype=bool), new_label=pd.Series(dtype=int))
    f["keep"] = (f.max_area2d_um2 <= p.max_area2d_um2) & (f.vol_um3 >= p.min_vol_um3) & (f.contrast >= p.min_contrast)
    lut = np.zeros(int(lab.max()) + 1, np.int32)
    k = f.label[f.keep].values
    lut[k] = np.arange(1, len(k) + 1)
    f["new_label"] = lut[f.label.values]
    return lut[lab], f


def load_stardist_predictor(cfg: Config) -> Callable[[np.ndarray], np.ndarray]:
    """Return ``predict(image2d_normalised) -> labels2d`` backed by a pretrained StarDist2D model."""
    try:
        from stardist.models import StarDist2D
    except ImportError as e:  # pragma: no cover
        raise ImportError("nuclei.method = 'stardist' needs `pip install cellshape3d[stardist]` "
                          "(or set [nuclei] method = 'classical')") from e
    model = StarDist2D.from_pretrained(cfg.nuclei.model)
    p = cfg.nuclei
    return lambda im: model.predict_instances(im, prob_thresh=p.prob_thresh, nms_thresh=p.nms_thresh)[0]


def segment_nuclei_stardist(x: np.ndarray, cfg: Config, predict: Callable[[np.ndarray], np.ndarray]):
    """x: preprocessed DAPI (Z Y X, any linear scale). Returns (labels uint16, features DataFrame, info dict)."""
    p = cfg.nuclei
    x = ndi.gaussian_filter(x.astype(np.float32), (0, p.pre_smooth_xy, p.pre_smooth_xy))
    lo, hi = np.percentile(x[:, ::2, ::2], p.norm_pct)
    xn = (x - lo) / max(hi - lo, 1e-6)
    lab2d = np.zeros(x.shape, np.int32)
    per = []
    for z in range(x.shape[0]):
        l = np.asarray(predict(xn[z])).astype(np.int32)
        areas = np.bincount(l.ravel())
        small = np.where(areas < p.min_area_px)[0]
        l[np.isin(l, small[small > 0])] = 0
        lab2d[z] = l
        per.append(int(len(np.unique(l)) - 1))
    lab3d = relabel(stitch(lab2d, p.stitch_iou))
    n_stitched = int(lab3d.max())
    lab3d = relabel(split_z(lab3d, x, p.z_split_dip))
    n_split = int(lab3d.max())
    lab3d, feat = filter_nuclei(lab3d, x, cfg)
    info = dict(n_after_stitch=n_stitched, n_after_zsplit=n_split, n_nuclei_3d=int(lab3d.max()),
                per_slice_2d=per, norm_lo=float(lo), norm_hi=float(hi))
    return lab3d.astype(np.uint16), feat, info


def _ellipsoid(rz: int, rxy: int) -> np.ndarray:
    z, y, x = np.ogrid[-rz:rz + 1, -rxy:rxy + 1, -rxy:rxy + 1]
    return (z / max(rz, 1e-6)) ** 2 + (y / rxy) ** 2 + (x / rxy) ** 2 <= 1


def segment_nuclei_classical(x: np.ndarray, B: np.ndarray, cfg: Config):
    """x: binned DAPI counts, B: local mean background (same grid). Returns (labels uint16, features, info)."""
    from skimage import feature, segmentation
    p, a = cfg.nuclei, cfg.acquisition
    sp = a.spacing
    vox = sp[0] * sp[1] * sp[2]
    gs = ndi.gaussian_filter(x, p.cl_smooth_zyx)
    g1 = ndi.gaussian_filter(x, p.cl_seed_smooth_zyx)
    T_mask = significance_threshold(B, p.cl_smooth_zyx, p.cl_k_poisson, p.cl_f_contrast, p.cl_b_floor, a.bin)
    T_seed = significance_threshold(B, p.cl_seed_smooth_zyx, p.cl_k_poisson, p.cl_f_contrast, p.cl_b_floor, a.bin)
    mask = gs > T_mask
    rz = max(1, int(round(p.cl_seed_radius_um[0] / sp[0])))
    rxy = max(1, int(round(p.cl_seed_radius_um[1] / sp[1])))
    pk = feature.peak_local_max(g1, footprint=_ellipsoid(rz, rxy), labels=mask.astype(np.int32), exclude_border=False)
    pk = pk[g1[tuple(pk.T)] > T_seed[tuple(pk.T)]]
    markers = np.zeros(x.shape, np.int32)
    markers[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    nuc = segmentation.watershed(-gs, markers, mask=mask)
    if len(pk):
        peak = np.zeros(len(pk) + 1, np.float32)
        peak[1:] = ndi.maximum(gs, nuc, np.arange(1, len(pk) + 1))
        nuc[(gs - B) < p.cl_frac_of_peak * (peak[nuc] - B)] = 0
    vol = np.bincount(nuc.ravel(), minlength=len(pk) + 1) * vox
    keep = np.where(vol[1:] >= p.cl_min_nucleus_um3)[0] + 1
    lut = np.zeros(len(pk) + 1, np.int32)
    lut[keep] = np.arange(1, len(keep) + 1)
    nuc = lut[nuc]
    feat = nucleus_features(nuc, x, sp)
    feat["keep"] = True
    feat["new_label"] = feat["label"]
    info = dict(n_seeds=int(len(pk)), n_nuclei_3d=int(nuc.max()), mask_frac=float(mask.mean()))
    return nuc.astype(np.uint16), feat, info
