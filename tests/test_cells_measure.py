import numpy as np
import pytest

from cellshape3d.cells import keep_nucleus_component, segment_cells
from cellshape3d.config import Config
from cellshape3d.measure import add_cell_context, inertia_ar3d, measure_objects
from cellshape3d.preprocess import bin_xy
from tests.conftest import ellipsoid_mask, make_stack

SP = (3.16, 1.26, 1.26)


@pytest.mark.parametrize("angle", [0, 30, 90, 150])
def test_orientation_and_ar_of_synthetic_ellipse(angle):
    lab = ellipsoid_mask((5, 101, 101), (2, 50, 50), (2, 8, 24), angle).astype(np.int32)
    df = measure_objects(lab, SP)
    r = df.iloc[0]
    assert r.AR_xy == pytest.approx(3.0, rel=0.05)
    d = abs(r.orientation_deg - angle) % 180
    assert min(d, 180 - d) < 2.0
    assert r.major_um == pytest.approx(2 * 24 * 1.26, rel=0.05)     # full major axis = 2 * radius
    assert not r.touches_xy_border and r.touches_z_end


def test_measure_flags_and_area_filter():
    lab = np.zeros((4, 50, 50), np.int32)
    lab[1:3, 0:10, 10:20] = 1          # touches the border
    lab[1:3, 20:22, 20:22] = 2         # too small (4 px * 1.59 um2 < 20 um2)
    df = measure_objects(lab, SP, min_proj_area_um2=20)
    assert df.label.tolist() == [1]
    assert df.touches_xy_border.iloc[0] and not df.touches_z_end.iloc[0]
    assert df.volume_um3.iloc[0] == pytest.approx(200 * 3.16 * 1.26 ** 2)


def test_inertia_ar3d_needs_three_slices():
    c = np.array([[0, 0, 0], [0, 1, 0], [3.16, 0, 1], [3.16, 1, 1], [0, 2, 2]], float)
    assert np.isnan(inertia_ar3d(c))


def test_keep_nucleus_component_drops_detached_parts():
    cells = np.zeros((1, 10, 20), np.int32)
    nuc = np.zeros_like(cells)
    cells[0, 2:8, 2:8] = 1
    cells[0, 2:8, 12:18] = 1           # detached fragment without the nucleus
    nuc[0, 4:6, 4:6] = 1
    out = keep_nucleus_component(cells, nuc)
    assert out[0, 5, 5] == 1 and out[0, 5, 15] == 0


def test_segment_cells_one_cell_per_nucleus(rng):
    cfg = Config()
    a, centres = make_stack(rng, n_cells=4, elong=2.0, angle_deg=30)
    x = bin_xy(a[:, 1].astype(np.float32), 2)
    nuc = np.zeros(x.shape, np.int32)
    for i, (z, y, xx) in enumerate(centres, start=1):
        nuc[ellipsoid_mask(x.shape, (z, y // 2, xx // 2), (1.5, 2.5, 2.5))] = i
    r = segment_cells(x, np.full_like(x, 0.3), nuc, cfg)
    cells = r["cells"]
    assert r["info"]["n_cells"] == 4
    for i in range(1, 5):
        assert np.all(cells[nuc == i] == i)                 # nucleus lies inside its own cell
        assert (cells == i).sum() > 3 * (nuc == i).sum()    # and the cell is larger than the nucleus
    df = measure_objects(cells, cfg.acquisition.spacing)
    assert df.AR_xy.median() > 1.5
    d = np.abs(df.orientation_deg - 30) % 180
    assert np.all(np.minimum(d, 180 - d) < 10)
    ctx = add_cell_context(df, cells, r["clusters"], nuc, cfg.acquisition.spacing, 2500)
    assert ctx.cluster_n_cells.eq(1).all() and not ctx.is_sheet.any()
    assert (ctx.nucleus_volume_um3 > 0).all()
