"""End-to-end: synthetic stacks -> every stage -> statistics, figures and tables."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cellshape3d.cli import main
from cellshape3d.pipeline import run_stage


@pytest.mark.slow
def test_full_pipeline(dataset):
    cfg = dataset
    out = Path(cfg.data.out_dir)
    for st in ["inventory", "preprocess", "nuclei", "cells", "measure", "fibres", "stats", "figures", "tables"]:
        assert run_stage(st, cfg) == [], st

    inv = pd.read_csv(out / "00_inventory" / "inventory.csv")
    assert len(inv) == 12 and (inv.n_z == 8).all()

    for img in ["G-HEK-G1_001", "S-HeLa-G2_001"]:
        nuc = json.loads((out / "02_nuclei" / f"{img}_nuc.json").read_text())
        assert nuc["n_nuclei_3d"] == 4
        cells = pd.read_csv(out / "04_measure" / f"{img}_cells.csv")
        assert len(cells) == 4

    s_cells = pd.read_csv(out / "05_fibres" / "S-HEK-G1_001_cells_align.csv")
    assert s_cells.on_fibre.all()
    assert s_cells.align_deg.median() < 15            # synthetic cells are parallel to the stripes

    summ = pd.read_csv(out / "06_stats" / "per_image_summary.csv")
    med = summ.groupby("group").cell_AR_median.median()
    assert med["G"] < med["GS"] < med["S"]

    anova = pd.read_csv(out / "06_stats" / "anova_tests.csv")
    assert {"scaffold", "cell line", "scaffold x cell line", "team (block)"} <= set(anova.term)

    for f in ["Fig1_aspect_ratio", "Fig2_alignment", "Fig3_relative_angle", "Fig4_dtheta_vs_AR"]:
        assert (out / "06_stats" / "figures" / f"{f}.pdf").exists()
    t1 = (out / "06_stats" / "Table1_AR_summary.md").read_text(encoding="utf-8")
    assert "Cell AR" in t1
    assert (out / "06_stats" / "Table2_alignment_summary.csv").exists()

    # resumable: a second run skips everything
    assert run_stage("cells", cfg) == []


def test_cli_list_and_preprocess_only(dataset, tmp_path, capsys):
    cfg = dataset
    cfgfile = tmp_path / "c.toml"
    cfgfile.write_text(
        f'[data]\nraw_dir = "{Path(cfg.data.raw_dir).as_posix()}"\nout_dir = "{Path(cfg.data.out_dir).as_posix()}"\n'
        f'glob = "**/*.tif"\n')
    assert main(["list", "-c", str(cfgfile)]) == 0
    assert "S-HeLa-G2_001" in capsys.readouterr().out
    assert main(["preprocess", "-c", str(cfgfile), "--only", "G-HEK-G1_001"]) == 0
    pre = sorted(p.name for p in (Path(cfg.data.out_dir) / "01_preprocess").glob("*_pre.json"))
    assert pre == ["G-HEK-G1_001_pre.json"]
    rel = __import__("tifffile").imread(Path(cfg.data.out_dir) / "01_preprocess" / "G-HEK-G1_001_dapi.tif")
    assert rel.dtype == np.uint16 and rel.shape == (8, 64, 64)
