import numpy as np
import pytest

from cellshape3d.config import Config
from cellshape3d.noise import significance_threshold, sum_w2
from cellshape3d.preprocess import bin_xy, load_counts, median_background, preprocess_channel, to_uint16


def test_bin_xy_mean_and_crop():
    x = np.arange(2 * 5 * 6, dtype=np.float32).reshape(2, 5, 6)
    b = bin_xy(x, 2)
    assert b.shape == (2, 2, 3)
    assert b[0, 0, 0] == pytest.approx(np.mean([0, 1, 6, 7]))


def test_sum_w2_matches_discrete_kernel():
    from scipy import ndimage as ndi
    sig = (1.0, 2.0, 2.0)
    d = np.zeros((21, 31, 31))
    d[10, 15, 15] = 1
    k = ndi.gaussian_filter(d, sig)
    assert sum_w2(sig) == pytest.approx((k ** 2).sum(), rel=0.02)


def test_significance_threshold_monotonic():
    B = np.array([0.0, 0.5, 5.0, 50.0])
    T = significance_threshold(B, (0.7, 1, 1), k=6, f=1, b_floor=0.1, bin_=2)
    assert np.all(T > B) and np.all(np.diff(T) > 0)
    assert T[-1] == pytest.approx(100.0)          # bright background: the contrast term (2 x B) dominates


def test_preprocess_removes_offset_and_normalises(rng):
    cfg = Config()
    raw = rng.poisson(20, (6, 128, 128)).astype(np.uint16)       # flat background ...
    raw[2:4, 50:60, 50:60] += 200                                  # ... plus one bright object
    r = preprocess_channel(raw, cfg)
    rel = r["rel"]
    assert rel.shape == (6, 64, 64)
    assert np.percentile(rel[:, ::2, ::2], 99.9) == pytest.approx(1.0, rel=0.05)
    assert rel[2:4, 25:30, 25:30].mean() > 10 * np.median(rel)
    assert r["bg_small"].shape == (6, 16, 16)
    assert 15 < r["bgmed_small"].mean() < 25          # local mean background ~ Poisson mean
    assert to_uint16(rel, 10000).dtype == np.uint16


def test_median_background_handles_odd_shapes():
    out = median_background(np.ones((2, 30, 37), np.float32), 1.26, 4, 50, 20)
    assert out.shape == (2, 7, 9)
    np.testing.assert_allclose(out, 1.0, rtol=1e-5)


def test_load_counts_upsamples_background():
    raw = np.ones((3, 2, 64, 64), np.uint16)
    x, B = load_counts(raw, 1, np.full((3, 8, 8), 0.5, np.float32), 2)
    assert x.shape == B.shape == (3, 32, 32)
    np.testing.assert_allclose(B, 0.5)
