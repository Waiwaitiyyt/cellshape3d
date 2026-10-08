"""Small binary-morphology helpers with fixed semantics across scikit-image versions
(``remove_small_objects(min_size=...)`` and ``binary_opening`` are deprecated in scikit-image >= 0.26)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop face-connected components with fewer than ``min_size`` voxels (scikit-image < 0.26 semantics)."""
    lab, n = ndi.label(mask)
    if n == 0:
        return mask.astype(bool)
    size = np.bincount(lab.ravel())
    keep = size >= min_size
    keep[0] = False
    return keep[lab]


def binary_opening(mask: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Erosion (image border treated as foreground) followed by dilation, as ``skimage.morphology.binary_opening``."""
    return ndi.binary_dilation(ndi.binary_erosion(mask, footprint, border_value=1), footprint)


def disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r
