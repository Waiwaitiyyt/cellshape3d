"""Stage 1: preprocessing and background estimation.

Per channel (DAPI, actin): 2x2 xy binning (mean) -> light 3D Gaussian denoise -> per-slice large-scale background
(grey opening on a 4x downsampled slice = lower envelope) -> subtraction -> normalisation by the stack's 99.9th
percentile. Photon counts of the HyD detectors are very low (often 0-50), so a MAD noise estimate collapses to ~0 in
empty background; per-stack percentile normalisation is used instead and thresholds are chosen adaptively per stack
in the later stages.

A second background, the local *mean* (median filter over ~50 um), is also computed. The opening is a lower envelope
and underestimates the mean of autofluorescence haze (GelMA: mean ~0.5 counts, envelope ~0.02), which makes a
Poisson significance test far too permissive; the classical nuclei detector therefore uses the median background.
The actin stage uses the opening background, because inside cell clusters the 50 um median rises to cell level.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .config import Config


def bin_xy(x: np.ndarray, b: int) -> np.ndarray:
    """Mean-bin the last two axes of a Z Y X stack by ``b`` (edges that do not fill a bin are cropped)."""
    z, y, w = x.shape
    return x[:, :y // b * b, :w // b * b].reshape(z, y // b, b, w // b, b).mean(axis=(2, 4), dtype=np.float32)


def opening_background(slice2d: np.ndarray, px_um: float, ds: int, open_um: float) -> np.ndarray:
    """Large-scale lower-envelope background of one slice, returned ``ds``-times downsampled."""
    small = slice2d[::ds, ::ds]
    k = int(round(open_um / (px_um * ds))) | 1
    bg = ndi.grey_opening(small, size=(k, k))
    return ndi.gaussian_filter(bg, k / 2)


def upsample_bg(bg_small: np.ndarray, shape) -> np.ndarray:
    """Z x y x x (downsampled) -> Z x Y x X by linear interpolation."""
    return ndi.zoom(bg_small, (1, shape[1] / bg_small.shape[1], shape[2] / bg_small.shape[2]), order=1)


def median_background(x: np.ndarray, px_um: float, ds: int, win_um: float, edge_dilate_um: float) -> np.ndarray:
    """Local mean background (counts) of a binned Z Y X stack, ``ds``-times downsampled in xy.

    Block-mean downsample -> per-slice median filter (window ``win_um``) -> grey dilation (``edge_dilate_um``) ->
    Gaussian smooth. Dark PCL fibre shadows pull the median down at pore edges, so haze next to a fibre looked
    brighter than background; the small dilation lifts the background back to the haze level there.
    """
    z, h, w = x.shape
    small = x[:, :h // ds * ds, :w // ds * ds].reshape(z, h // ds, ds, w // ds, ds).mean((2, 4))
    k = int(round(win_um / (px_um * ds))) | 1
    out = np.stack([ndi.median_filter(sl, size=k, mode="reflect") for sl in small])
    kd = int(round(edge_dilate_um / (px_um * ds))) | 1
    out = np.stack([ndi.grey_dilation(sl, size=(kd, kd)) for sl in out])
    return ndi.gaussian_filter(out, (0.5, k / 3, k / 3)).astype(np.float32)


def preprocess_channel(raw_ch: np.ndarray, cfg: Config) -> dict:
    """Process one raw channel (Z Y X counts). Returns rel (float, 1.0 = p99.9), bg_small, bgmed_small, meta."""
    p, a = cfg.preprocess, cfg.acquisition
    xb = bin_xy(raw_ch.astype(np.float32), a.bin)
    x = ndi.gaussian_filter(xb, p.gauss_sigma_zyx)
    bg_small = np.stack([opening_background(sl, a.px_bin_um, p.bg_downsample, p.bg_open_um) for sl in x])
    bg = upsample_bg(bg_small, x.shape)
    x = np.clip(x - bg, 0, None)
    ref = float(np.percentile(x[:, ::2, ::2], p.norm_percentile)) or 1.0
    rel = x / ref
    meta = dict(norm_ref_counts=ref, bg_mean_counts=float(bg.mean()),
                bgstd_counts=float(x[x < np.percentile(x, 60)].std()),
                zprofile_p99=[round(float(np.percentile(sl, 99)), 2) for sl in rel])
    bgmed = median_background(xb, a.px_bin_um, p.bg_downsample, p.bgmed_win_um, p.bgmed_edge_dilate_um)
    return dict(rel=rel, bg_small=bg_small.astype(np.float32), bgmed_small=bgmed, meta=meta)


def to_uint16(rel: np.ndarray, scale: int) -> np.ndarray:
    return np.clip(rel * scale, 0, 65535).astype(np.uint16)


def tl_mean(raw_tl: np.ndarray, b: int) -> np.ndarray:
    """Binned mean projection of the transmitted-light channel (for fibre orientation)."""
    return bin_xy(raw_tl.astype(np.float32), b).mean(0)


def load_counts(raw: np.ndarray, ch: int, bg_small: np.ndarray, b: int) -> tuple[np.ndarray, np.ndarray]:
    """Binned raw counts (mean per raw pixel, NOT background-subtracted) and the upsampled background B, both ZYX."""
    x = bin_xy(raw[:, ch].astype(np.float32), b)
    B = ndi.zoom(bg_small.astype(np.float32), (1, x.shape[1] / bg_small.shape[1], x.shape[2] / bg_small.shape[2]), order=1)
    return x, np.clip(B, 0, None)
