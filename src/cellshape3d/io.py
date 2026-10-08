"""Reading raw stacks, discovering images, writing calibrated ImageJ TIFFs."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile

from .config import Config


def list_stacks(cfg: Config) -> list[dict]:
    """All raw stacks under ``data.raw_dir`` whose stem matches ``data.name_pattern``.

    Returns dicts with keys path, image, group, cell_line, team (int or None), sorted by path.
    Files that do not match the pattern are skipped.
    """
    pat = re.compile(cfg.data.name_pattern)
    out = []
    for f in sorted(Path(cfg.data.raw_dir).glob(cfg.data.glob)):
        m = pat.search(f.stem)
        if not m:
            continue
        gd = m.groupdict()
        team = gd.get("team")
        out.append(dict(path=f, image=gd.get("image") or m.group(0), group=gd["group"], cell_line=gd["cell_line"],
                        team=int(team) if team is not None and team.isdigit() else team))
    return out


def read_stack(path) -> np.ndarray:
    """Return the first series of a TIFF as Z C Y X (a single-channel or single-slice file gets a length-1 axis)."""
    with tifffile.TiffFile(path) as t:
        ser = t.series[0]
        a = ser.asarray()
        ax = ser.axes
    for k in "ZC":
        if k not in ax:
            a = a[np.newaxis]
            ax = k + ax
    a = np.moveaxis(a, (ax.index("Z"), ax.index("C")), (0, 1))
    return a.reshape(a.shape[:2] + a.shape[-2:])


def save_ij(path, arr, spacing, axes="ZYX", **kw):
    """Save an ImageJ-compatible TIFF carrying um calibration (z spacing included); spacing = (z, y, x) um."""
    md = {"axes": axes, "unit": "um", "spacing": spacing[0]}
    tifffile.imwrite(path, arr, imagej=True, resolution=(1 / spacing[2], 1 / spacing[1]),
                     metadata=md, compression="zlib", **kw)


def read_tif(path) -> np.ndarray:
    return tifffile.imread(path)
