"""Photon-counting significance helpers.

The HyD detector output is integer photon counts with variance == mean (checked on GelMA haze patches:
mean 0.43-0.86, var 0.51-0.76), so local noise follows Poisson statistics. For an image smoothed with a normalised
Gaussian (sigma in binned voxels), the s.d. of the smoothed background (counts per raw pixel) is
sqrt(B * sum(w^2) / BIN^2) with sum(w^2) = 1 / ((2 sqrt(pi))^3 sz sy sx).
"""
from __future__ import annotations

import numpy as np


def sum_w2(sigma_zyx) -> float:
    """Sum of squared weights of a normalised 3D Gaussian kernel (continuous approximation)."""
    return 1.0 / ((2 * np.sqrt(np.pi)) ** 3 * np.prod(sigma_zyx))


def significance_threshold(B: np.ndarray, sigma_zyx, k: float, f: float, b_floor: float, bin_: int) -> np.ndarray:
    """Smoothed intensity required at each voxel: B + max(k * Poisson sd(B), f * B).

    i.e. statistically significant (k sd) AND at least (1 + f) times the local background.
    """
    sd = np.sqrt(np.maximum(B, b_floor) * sum_w2(sigma_zyx) / bin_ ** 2)
    return B + np.maximum(k * sd, f * B)
