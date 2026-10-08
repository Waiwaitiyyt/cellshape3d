"""Stage 4: per-object morphology of 3D label images.

Main shape metric = 2D aspect ratio of the object's xy projection (max over z of the 3D mask):
AR = major / minor axis of the moment-equivalent ellipse (skimage regionprops).
The 3D AR (inertia ellipsoid in physical units) is reported but should not be used for conclusions with a
low-NA dry objective: z-step 3.16 um and optical section ~6-10 um (plus focal shift from RI mismatch) blur and
stretch z.
Orientation: angle of the major axis in degrees, 0-180, from +x (image right) towards -y (image up); the same
convention as the fibre orientation of stage 5.
Flags: touches_xy_border (exclude), touches_z_end (first / last slice, object may be truncated).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops


def inertia_ar3d(coords_um: np.ndarray) -> float:
    """sqrt(largest / smallest eigenvalue) of the coordinate covariance; NaN if < 3 z-slices."""
    if len(coords_um) < 5 or len(np.unique(coords_um[:, 0])) < 3:
        return np.nan
    c = coords_um - coords_um.mean(0)
    ev = np.sort(np.linalg.eigvalsh(np.cov(c.T)))
    return float(np.sqrt(ev[2] / max(ev[0], 1e-9)))


def skimage_to_image_angle(orientation_rad: float) -> float:
    """skimage orientation (angle between the row axis and the major axis) -> degrees from +x towards image-up."""
    return (np.degrees(orientation_rad) - 90.0) % 180.0


def measure_objects(lab: np.ndarray, spacing, intensity: np.ndarray | None = None,
                    min_proj_area_um2: float = 20.0) -> pd.DataFrame:
    """One row per label with projected-area >= ``min_proj_area_um2``. spacing = (z, y, x) um; y and x equal."""
    dz, py, px = spacing
    pxa = py * px
    rows = []
    nz, H, W = lab.shape
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        proj = m.any(0)
        area = proj.sum() * pxa
        if area < min_proj_area_um2:
            continue
        rp = regionprops(proj.astype(np.uint8))[0]
        maj, mnr = rp.axis_major_length * py, rp.axis_minor_length * py
        zz, yy, xx = np.nonzero(m)
        coords = np.c_[(zz + sl[0].start) * dz, (yy + sl[1].start) * py, (xx + sl[2].start) * px]
        rows.append(dict(label=i, z_um=coords[:, 0].mean(), y_um=coords[:, 1].mean(), x_um=coords[:, 2].mean(),
                         volume_um3=len(zz) * dz * pxa, n_slices=sl[0].stop - sl[0].start,
                         proj_area_um2=area, major_um=maj, minor_um=mnr, AR_xy=maj / max(mnr, 1e-6),
                         orientation_deg=skimage_to_image_angle(rp.orientation), solidity=rp.solidity,
                         AR_3d=inertia_ar3d(coords),
                         touches_xy_border=bool(sl[1].start == 0 or sl[2].start == 0 or sl[1].stop == H or sl[2].stop == W),
                         touches_z_end=bool(sl[0].start == 0 or sl[0].stop == nz),
                         mean_intensity=float(intensity[sl][m].mean()) if intensity is not None else np.nan))
    cols = ["label", "z_um", "y_um", "x_um", "volume_um3", "n_slices", "proj_area_um2", "major_um", "minor_um", "AR_xy",
            "orientation_deg", "solidity", "AR_3d", "touches_xy_border", "touches_z_end", "mean_intensity"]
    return pd.DataFrame(rows, columns=cols)


def add_cell_context(dc: pd.DataFrame, cells: np.ndarray, clusters: np.ndarray, nuc: np.ndarray, spacing,
                     sheet_area_um2: float) -> pd.DataFrame:
    """Cluster membership, sheet flag (projected area > ``sheet_area_um2``) and own-nucleus volume for each cell."""
    dc = dc.copy()
    if dc.empty:
        for c in ("cluster_id", "cluster_n_cells", "is_sheet", "nucleus_volume_um3"):
            dc[c] = pd.Series(dtype=float)
        return dc
    n = int(cells.max())
    cid = np.zeros(n + 1, np.int64)
    cid[1:] = np.asarray(ndi.maximum(clusters, cells, np.arange(1, n + 1))).astype(np.int64)
    dc["cluster_id"] = cid[dc.label.values]
    dc["cluster_n_cells"] = dc.groupby("cluster_id").label.transform("count")
    dc["is_sheet"] = dc.proj_area_um2 > sheet_area_um2
    nv = np.bincount(nuc.ravel(), minlength=n + 1)
    dc["nucleus_volume_um3"] = nv[dc.label.values] * spacing[0] * spacing[1] * spacing[2]
    return dc
