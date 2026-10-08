"""Synthetic 3-channel confocal stacks (DAPI, actin, transmitted light) with known geometry."""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from cellshape3d.config import config_from_dict


def ellipsoid_mask(shape, centre, radii, angle_deg=0.0):
    """Boolean Z Y X ellipsoid; ``angle_deg`` rotates the x radius towards image-up (y decreasing)."""
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dz, dy, dx = z - centre[0], y - centre[1], x - centre[2]
    t = np.radians(angle_deg)
    u = dx * np.cos(t) - dy * np.sin(t)        # along the major axis (image-up positive)
    v = dx * np.sin(t) + dy * np.cos(t)
    return (dz / radii[0]) ** 2 + (u / radii[2]) ** 2 + (v / radii[1]) ** 2 <= 1


def make_stack(rng, nz=8, ny=128, nx=128, n_cells=4, elong=1.0, angle_deg=0.0, fibres=False):
    """Raw-resolution stack (Z C Y X uint16). Cells sit on a grid; elongation/angle set the actin shape."""
    dapi = rng.poisson(0.3, (nz, ny, nx)).astype(np.float32)
    actin = rng.poisson(0.3, (nz, ny, nx)).astype(np.float32)
    k = int(np.ceil(np.sqrt(n_cells)))
    cy = np.linspace(ny / (2 * k), ny - ny / (2 * k), k)
    cx = np.linspace(nx / (2 * k), nx - nx / (2 * k), k)
    centres = [(nz // 2, int(y), int(x)) for y in cy for x in cx][:n_cells]
    for c in centres:
        dapi[ellipsoid_mask(dapi.shape, c, (1.6, 5, 5 * min(elong, 1.6)), angle_deg)] += 40
        actin[ellipsoid_mask(actin.shape, c, (2.2, 9, 9 * elong), angle_deg)] += 25
    if fibres:
        tl = np.full((nz, ny, nx), 200.0, np.float32)
        t = np.radians(angle_deg)
        yy, xx = np.mgrid[:ny, :nx]
        d = (xx * np.sin(t) + yy * np.cos(t)) % 32       # stripes parallel to the cell axis, period 32 px
        tl[:, (d < 10)] = 80.0
    else:
        tl = np.full((nz, ny, nx), 200.0, np.float32) + rng.normal(0, 2, (nz, ny, nx))
    a = np.stack([dapi, actin, tl], 1)
    return np.clip(a, 0, 65535).astype(np.uint16), centres


def write_stack(path, a):
    tifffile.imwrite(path, a, imagej=True, metadata={"axes": "ZCYX"})


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_cfg(tmp_path):
    """Config for tiny synthetic stacks: small pixel counts, classical nuclei, low object minimum."""
    return config_from_dict({
        "data": {"raw_dir": str(tmp_path / "raw"), "out_dir": str(tmp_path / "out"), "glob": "**/*.tif"},
        "nuclei": {"method": "classical", "min_vol_um3": 20.0, "cl_min_nucleus_um3": 20.0},
        "fibres": {"flat_sigma_px": 20, "min_fibre_area_px": 30},
        "stats": {"min_objects_per_image": 1},
    })


@pytest.fixture
def dataset(tmp_path, small_cfg, rng):
    """12 stacks: 3 groups x 2 cell lines x 2 teams; GS/S cells are elongated along fibres."""
    raw = tmp_path / "raw"
    raw.mkdir()
    elong = {"G": 1.0, "GS": 1.6, "S": 2.2}
    for team in (1, 2):
        for g in ("G", "GS", "S"):
            for line in ("HEK", "HeLa"):
                a, _ = make_stack(rng, elong=elong[g] + 0.1 * team, angle_deg=30.0, fibres=g != "G")
                write_stack(raw / f"2026-BMEG5001_2026-{g}-{line}-G{team}_001.tif", a)
    return small_cfg
