import numpy as np
from scipy import ndimage as ndi

from cellshape3d.config import Config
from cellshape3d.nuclei import (
    filter_nuclei,
    nucleus_features,
    relabel,
    segment_nuclei_classical,
    segment_nuclei_stardist,
    split_z,
    stitch,
)
from cellshape3d.preprocess import bin_xy
from tests.conftest import ellipsoid_mask, make_stack


def disc(shape, cy, cx, r):
    y, x = np.ogrid[:shape[0], :shape[1]]
    return (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2


def test_stitch_links_overlapping_and_separates_distinct():
    lab = np.zeros((3, 40, 40), np.int32)
    lab[0][disc((40, 40), 10, 10, 4)] = 1
    lab[1][disc((40, 40), 11, 10, 4)] = 7          # same object, shifted 1 px, different 2D id
    lab[1][disc((40, 40), 30, 30, 4)] = 3          # new object
    lab[2][disc((40, 40), 30, 31, 4)] = 1
    out = relabel(stitch(lab, 0.25))
    assert out.max() == 2
    assert out[0, 10, 10] == out[1, 11, 10]
    assert out[1, 30, 30] == out[2, 30, 31] != out[0, 10, 10]


def test_stitch_respects_iou_threshold():
    lab = np.zeros((2, 40, 40), np.int32)
    lab[0][disc((40, 40), 10, 10, 4)] = 1
    lab[1][disc((40, 40), 10, 16, 4)] = 1          # barely overlapping
    assert relabel(stitch(lab, 0.25)).max() == 2


def test_split_z_cuts_two_peaks():
    lab = np.zeros((7, 20, 20), np.int32)
    lab[:, 5:15, 5:15] = 1
    prof = np.array([5, 10, 5, 1, 6, 12, 6], float)          # two maxima, dip at z=3
    x = np.broadcast_to(prof[:, None, None], lab.shape).astype(float)
    out = relabel(split_z(lab, x, 0.75))
    assert out.max() == 2
    assert out[3].max() == 0                                   # minimum slice dropped
    assert out[0, 10, 10] != out[6, 10, 10]


def test_split_z_keeps_single_peak():
    lab = np.ones((5, 4, 4), np.int32)
    x = np.broadcast_to(np.array([1, 3, 5, 3, 1.0])[:, None, None], lab.shape)
    np.testing.assert_array_equal(split_z(lab, x, 0.75), lab)


def test_features_and_filter():
    cfg = Config()
    shape = (6, 80, 80)
    lab = np.zeros(shape, np.int32)
    x = np.full(shape, 1.0)
    good = ellipsoid_mask(shape, (3, 20, 20), (2, 5, 5))
    lab[good] = 1
    x[good] = 10                                   # bright, nucleus-sized
    sheet = np.zeros(shape, bool)
    sheet[2:4, 40:78, 2:78] = True                 # haze sheet: far too large in 2D
    lab[sheet] = 2
    x[sheet] = 10
    dim = ellipsoid_mask(shape, (3, 20, 60), (2, 5, 5))
    lab[dim] = 3                                   # not brighter than its surroundings
    f = nucleus_features(lab, x, cfg.acquisition.spacing)
    assert f.set_index("label").loc[1, "contrast"] > 5
    out, f = filter_nuclei(lab, x, cfg)
    assert f.keep.tolist() == [True, False, False]
    assert out.max() == 1 and out[3, 20, 20] == 1


def test_stardist_path_with_fake_predictor(rng):
    """The StarDist wrapper (normalise -> per-slice predict -> stitch -> split -> filter) with a stand-in model."""
    cfg = Config()
    a, centres = make_stack(rng, n_cells=4)
    x = bin_xy(a[:, 0].astype(np.float32), 2)

    def predict(im):
        return ndi.label(im > 0.5)[0]

    lab, feat, info = segment_nuclei_stardist(x, cfg, predict)
    assert info["n_nuclei_3d"] == 4 and lab.max() == 4
    assert lab.dtype == np.uint16
    for z, y, xx in centres:
        assert lab[z, y // 2, xx // 2] > 0


def test_classical_finds_synthetic_nuclei(rng):
    cfg = Config()
    a, centres = make_stack(rng, n_cells=4)
    x = bin_xy(a[:, 0].astype(np.float32), 2)
    B = np.full_like(x, 0.3)
    lab, feat, info = segment_nuclei_classical(x, B, cfg)
    assert lab.max() == 4
    assert len(feat) == 4
    for z, y, xx in centres:
        assert lab[z, y // 2, xx // 2] > 0
