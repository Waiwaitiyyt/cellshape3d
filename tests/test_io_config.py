import numpy as np
import pytest
import tifffile

from cellshape3d.config import Config, config_from_dict, load_config
from cellshape3d.io import list_stacks, read_stack, save_ij


def test_defaults_spacing():
    c = Config()
    assert c.acquisition.px_bin_um == pytest.approx(1.26)
    assert c.acquisition.spacing == pytest.approx((3.16, 1.26, 1.26))


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown key"):
        config_from_dict({"cells": {"k_poison": 3}})


def test_load_toml_resolves_paths_and_tuples(tmp_path):
    p = tmp_path / "cfg" / "c.toml"
    p.parent.mkdir()
    p.write_text('[data]\nraw_dir = "../raw"\n[acquisition]\nch_tl = -1\n[cells]\nsmooth_zyx = [1, 2, 2]\n')
    c = load_config(p)
    assert c.data.raw_dir == (tmp_path / "raw").resolve()
    assert c.acquisition.ch_tl is None
    assert c.cells.smooth_zyx == (1, 2, 2)


def test_repo_configs_load():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "configs"
    for f in root.glob("*.toml"):
        load_config(f)


def test_list_stacks_parses_names(tmp_path):
    cfg = config_from_dict({"data": {"raw_dir": str(tmp_path), "glob": "**/*.tif"}})
    (tmp_path / "GelMa (G)" / "HEK").mkdir(parents=True)
    for n in ["2026-BMEG5001_2026-G-HEK-G10_001", "2026-BMEG5001_2026-GS-HeLa-G3_001", "notes"]:
        tifffile.imwrite(tmp_path / "GelMa (G)" / "HEK" / f"{n}.tif", np.zeros((2, 2), np.uint8))
    s = {x["image"]: x for x in list_stacks(cfg)}
    assert set(s) == {"G-HEK-G10_001", "GS-HeLa-G3_001"}
    assert (s["GS-HeLa-G3_001"]["group"], s["GS-HeLa-G3_001"]["cell_line"], s["GS-HeLa-G3_001"]["team"]) == ("GS", "HeLa", 3)


def test_read_stack_reorders_to_zcyx(tmp_path):
    a = np.arange(3 * 4 * 5 * 6, dtype=np.uint16).reshape(3, 4, 5, 6)     # Z C Y X
    tifffile.imwrite(tmp_path / "a.tif", a, imagej=True, metadata={"axes": "ZCYX"})
    np.testing.assert_array_equal(read_stack(tmp_path / "a.tif"), a)
    tifffile.imwrite(tmp_path / "b.tif", a[:, 0], imagej=True, metadata={"axes": "ZYX"})
    assert read_stack(tmp_path / "b.tif").shape == (3, 1, 5, 6)


def test_save_ij_carries_calibration(tmp_path):
    save_ij(tmp_path / "x.tif", np.zeros((2, 4, 4), np.uint16), (3.16, 1.26, 1.26))
    with tifffile.TiffFile(tmp_path / "x.tif") as t:
        assert t.imagej_metadata["spacing"] == pytest.approx(3.16)
        assert t.imagej_metadata["unit"] == "um"
        xr = t.pages[0].tags["XResolution"].value
        assert xr[1] / xr[0] == pytest.approx(1.26, rel=1e-4)
