"""Stage 5: fibre orientation from transmitted light and object-to-fibre alignment.

TL mean projection (stage 1, binned) -> flat-field (divide by a large Gaussian) -> fibres are dark.
Fibre mask: Otsu on the flat-fielded image (dark class), small objects removed, opened.
Local fibre orientation: structure tensor (gradient sigma 2 px, integration sigma 6 px ~ 7.6 um); fibre axis =
direction of least intensity change (perpendicular to the dominant gradient).
Coherence = (l1 - l2) / (l1 + l2): ~1 for a single straight fibre, low at 0/90-degree crossings (ambiguous).
For each object centroid: fibre angle and coherence at the nearest fibre pixel, distance to that fibre, and
alignment = |object axis - fibre axis| folded to 0-90 degrees (0 = parallel, 45 = random expectation).
Note: TL is a projection of all scaffold layers, so at crossings orthogonal layers overlap; use the coherence to
restrict to unambiguous positions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters
from skimage.feature import structure_tensor

from .config import FibresCfg
from .morph import binary_opening, disk, remove_small_objects


def fibre_fields(tl: np.ndarray, p: FibresCfg):
    """Returns (flat-fielded TL, fibre mask, fibre angle deg [0, 180) from +x towards image-up, coherence)."""
    tl = tl.astype(np.float32)
    flat = tl / np.maximum(ndi.gaussian_filter(tl, p.flat_sigma_px), 1e-6)
    t = filters.threshold_otsu(flat)
    mask = remove_small_objects(flat < t, p.min_fibre_area_px)
    mask = binary_opening(mask, disk(2))
    Arr, Arc, Acc = structure_tensor(ndi.gaussian_filter(flat, p.grad_sigma_px), sigma=p.integ_sigma_px, order="rc")
    gx2, gy2, gxy = Acc, Arr, -Arc            # y axis points up -> flip the sign of the mixed term
    theta_grad = 0.5 * np.degrees(np.arctan2(2 * gxy, gx2 - gy2))
    fib = (theta_grad + 90.0) % 180.0
    coh = np.sqrt((gx2 - gy2) ** 2 + 4 * gxy ** 2) / np.maximum(gx2 + gy2, 1e-12)
    return flat, mask, fib.astype(np.float32), coh.astype(np.float32)


def axial_diff(a_deg, b_deg) -> np.ndarray:
    """Unsigned difference of two axial angles, folded to [0, 90]."""
    d = np.abs(np.asarray(a_deg) - np.asarray(b_deg)) % 180.0
    return np.minimum(d, 180.0 - d)


def signed_rel_angle(cell_deg, fibre_deg) -> np.ndarray:
    """Object minus fibre axis angle wrapped to (-90, 90]; positive = object rotated counter-clockwise."""
    d = (np.asarray(cell_deg) - np.asarray(fibre_deg) + 90.0) % 180.0 - 90.0
    return np.where(d == -90.0, 90.0, d)


def align_to_fibres(df: pd.DataFrame, mask, fib, coh, px_um: float, on_fibre_um: float) -> pd.DataFrame:
    """Add dist_to_fibre_um, on_fibre, fibre_angle_deg, fibre_coherence, align_deg to a measure table."""
    if df.empty:
        return df.assign(dist_to_fibre_um=[], on_fibre=[], fibre_angle_deg=[], fibre_coherence=[], align_deg=[])
    if not mask.any():
        return df.assign(dist_to_fibre_um=np.nan, on_fibre=False, fibre_angle_deg=np.nan, fibre_coherence=np.nan,
                         align_deg=np.nan)
    dist, (iy, ix) = ndi.distance_transform_edt(~mask, return_indices=True)
    yy = np.clip((df.y_um / px_um).round().astype(int), 0, mask.shape[0] - 1)
    xx = np.clip((df.x_um / px_um).round().astype(int), 0, mask.shape[1] - 1)
    ny, nx = iy[yy, xx], ix[yy, xx]
    df = df.copy()
    df["dist_to_fibre_um"] = dist[yy, xx] * px_um
    df["on_fibre"] = df.dist_to_fibre_um <= on_fibre_um
    df["fibre_angle_deg"] = fib[ny, nx]
    df["fibre_coherence"] = coh[ny, nx]
    df["align_deg"] = axial_diff(df.orientation_deg, df.fibre_angle_deg)
    return df
