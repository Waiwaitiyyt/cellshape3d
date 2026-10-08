"""Stage runners: file layout, resumability and per-image processing.

Each per-image stage skips images whose final output already exists (use ``force=True`` to recompute) and keeps
going when one image fails, reporting the failures at the end.

Output layout (``data.out_dir``):
  00_inventory/  inventory.csv, mip/{img}_MIP.tif, contact_{group}_{fluo,TL}.png
  01_preprocess/ {img}_{dapi,actin}.tif, {img}_{dapi,actin}_bg4.tif, {img}_{dapi,actin}_bgmed4.tif, {img}_tl.tif,
                 {img}_pre.json, qc/
  02_nuclei/     {img}_nuclei.tif, {img}_nuclei_features.csv, {img}_nuc.json, qc/
  03_cells/      {img}_cells.tif, {img}_fg.tif, {img}_clusters.tif, {img}_cells.json, qc/, qc_crops/
  04_measure/    {img}_nuclei.csv, {img}_cells.csv
  05_fibres/     {img}_fibre_{mask,angle}.tif, {img}_coherence.tif, {img}_{nuclei,cells}_align.csv, qc/
  06_stats/      all_nuclei.csv, all_cells.csv, per_image_summary.csv, group_tests.csv, anova_tests.csv, fig_*.png,
                 figures/, Table1_AR_summary.*, Table2_alignment_summary.*
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import tifffile

from . import qc
from .cells import segment_cells
from .config import Config
from .fibres import align_to_fibres, fibre_fields
from .io import list_stacks, read_stack, read_tif, save_ij
from .measure import add_cell_context, measure_objects
from .nuclei import load_stardist_predictor, segment_nuclei_classical, segment_nuclei_stardist
from .preprocess import load_counts, preprocess_channel, tl_mean, to_uint16

STAGES = ["inventory", "preprocess", "nuclei", "cells", "measure", "fibres", "stats", "figures", "tables"]


def run_batch(items: list[dict], is_done: Callable[[dict], bool], process: Callable[[dict], None],
              only: Iterable[str] | None = None, force: bool = False, log=print) -> list[str]:
    """Run ``process`` on every item not yet done; returns the image ids that failed."""
    only = set(only) if only else None
    todo = [it for it in items if (only is None or it["image"] in only) and (force or not is_done(it))]
    failed = []
    for k, it in enumerate(todo, start=1):
        t1 = time.time()
        try:
            process(it)
            status = "ok"
        except Exception as e:  # keep going; report at the end
            traceback.print_exc()
            failed.append(it["image"])
            status = f"ERROR {e}"
        log(f"[{k}/{len(todo)}] {it['image']}: {status} ({time.time() - t1:.1f}s)")
    log(f"done: {len(todo) - len(failed)} processed, {len(failed)} failed, {len(items) - len(todo)} skipped")
    return failed


def _json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def _meta(s):
    return {k: s[k] for k in ("group", "cell_line", "team")}


# ---------------------------------------------------------------------------------------------------- stage 0
def stage_inventory(cfg: Config, items, only=None, force=False):
    d = cfg.stage_dir("00_inventory")
    (d / "mip").mkdir(exist_ok=True)
    (d / "rows").mkdir(exist_ok=True)

    def process(s):
        r = {k: (str(v) if isinstance(v, Path) else v) for k, v in s.items()}
        with tifffile.TiffFile(s["path"]) as t:
            ser = t.series[0]
            r.update(shape=str(ser.shape), axes=ser.axes, dtype=str(ser.dtype))
            ij = t.imagej_metadata or {}
            r.update(ij_spacing=ij.get("spacing"), ij_unit=ij.get("unit"))
            xr = t.pages[0].tags.get("XResolution")
            r["xres_px_per_unit"] = (xr.value[0] / xr.value[1]) if xr else None
        a = read_stack(s["path"])
        r["n_z"] = a.shape[0]
        tifffile.imwrite(d / "mip" / f"{s['image']}_MIP.tif", a.max(axis=0), imagej=True, metadata={"axes": "CYX"})
        for c in range(a.shape[1]):
            x = a[:, c]
            p50, p99, p999 = np.percentile(x[:, ::4, ::4], [50, 99, 99.9])
            r.update({f"C{c + 1}_p50": p50, f"C{c + 1}_p99": p99, f"C{c + 1}_p999": p999, f"C{c + 1}_max": int(x.max())})
            r[f"C{c + 1}_zprofile"] = json.dumps([round(float(v), 2) for v in x.reshape(x.shape[0], -1).mean(1)])
        _json(d / "rows" / f"{s['image']}.json", r)

    failed = run_batch(items, lambda s: (d / "rows" / f"{s['image']}.json").exists(), process, only, force)
    rows = [json.load(open(f)) for s in items if (f := d / "rows" / f"{s['image']}.json").exists()]
    if rows:
        pd.DataFrame(rows).to_csv(d / "inventory.csv", index=False)
    a = cfg.acquisition
    for grp in cfg.design.groups:
        mips = {s["image"]: read_tif(d / "mip" / f"{s['image']}_MIP.tif") for s in items
                if s["group"] == grp and (d / "mip" / f"{s['image']}_MIP.tif").exists()}
        if not mips:
            continue
        qc.contact_sheet(mips, d / f"contact_{grp}_fluo.png", lambda m: qc.mip_rgb(m, a.ch_dapi, a.ch_actin))
        if a.ch_tl is not None:
            qc.contact_sheet(mips, d / f"contact_{grp}_TL.png", lambda m: qc._norm(m[a.ch_tl].astype(float)))
    return failed


# ---------------------------------------------------------------------------------------------------- stage 1
def stage_preprocess(cfg: Config, items, only=None, force=False):
    D, Q = cfg.stage_dir("01_preprocess"), cfg.stage_dir("01_preprocess/qc")
    a, p = cfg.acquisition, cfg.preprocess
    sp_bg = (a.dz_um, a.px_bin_um * p.bg_downsample, a.px_bin_um * p.bg_downsample)

    def process(s):
        img = s["image"]
        raw = read_stack(s["path"])
        meta = dict(image=img, n_z=raw.shape[0], bin=a.bin, spacing_zyx_um=a.spacing, **vars(p))
        rel = {}
        for name, ch in [("dapi", a.ch_dapi), ("actin", a.ch_actin)]:
            r = preprocess_channel(raw[:, ch], cfg)
            save_ij(D / f"{img}_{name}_bg4.tif", r["bg_small"], sp_bg)
            save_ij(D / f"{img}_{name}_bgmed4.tif", r["bgmed_small"], sp_bg)
            save_ij(D / f"{img}_{name}.tif", to_uint16(r["rel"], p.scale), a.spacing)
            meta.update({f"{name}_{k}": v for k, v in r["meta"].items()})
            rel[name] = r["rel"]
        tl = None
        if a.ch_tl is not None and a.ch_tl < raw.shape[1]:
            tl = tl_mean(raw[:, a.ch_tl], a.bin)
            save_ij(D / f"{img}_tl.tif", tl.astype(np.float32), a.spacing, axes="YX")
        _json(D / f"{img}_pre.json", meta)
        qc.preprocess_qc(Q / f"{img}_pre.png", img, rel["dapi"], rel["actin"], tl, meta["actin_zprofile_p99"])

    return run_batch(items, lambda s: (D / f"{s['image']}_pre.json").exists(), process, only, force)


# ---------------------------------------------------------------------------------------------------- stage 2
def stage_nuclei(cfg: Config, items, only=None, force=False, predict=None):
    D, Q, P = cfg.stage_dir("02_nuclei"), cfg.stage_dir("02_nuclei/qc"), Path(cfg.data.out_dir) / "01_preprocess"
    a, method = cfg.acquisition, cfg.nuclei.method
    if method not in ("stardist", "classical"):
        raise ValueError(f"unknown nuclei.method {method!r}")
    items = [s for s in items if (P / f"{s['image']}_pre.json").exists()]
    state = {"predict": predict}

    def process(s):
        img = s["image"]
        x = read_tif(P / f"{img}_dapi.tif").astype(np.float32)
        if method == "stardist":
            if state["predict"] is None:
                state["predict"] = load_stardist_predictor(cfg)
            lab, feat, info = segment_nuclei_stardist(x, cfg, state["predict"])
        else:
            counts, B = load_counts(read_stack(s["path"]), a.ch_dapi, read_tif(P / f"{img}_dapi_bgmed4.tif"), a.bin)
            lab, feat, info = segment_nuclei_classical(counts, B, cfg)
        feat.to_csv(D / f"{img}_nuclei_features.csv", index=False)
        save_ij(D / f"{img}_nuclei.tif", lab, a.spacing)
        _json(D / f"{img}_nuc.json", dict(vars(cfg.nuclei), image=img, **info))
        qc.nuclei_qc(Q / f"{img}_nuc.png", img, x, lab)

    return run_batch(items, lambda s: (D / f"{s['image']}_nuc.json").exists(), process, only, force)


# ---------------------------------------------------------------------------------------------------- stage 3
def stage_cells(cfg: Config, items, only=None, force=False, crops=False):
    D, Q = cfg.stage_dir("03_cells"), cfg.stage_dir("03_cells/qc")
    P, N = Path(cfg.data.out_dir) / "01_preprocess", Path(cfg.data.out_dir) / "02_nuclei"
    a = cfg.acquisition
    items = [s for s in items if (N / f"{s['image']}_nuclei.tif").exists()]

    def process(s):
        img = s["image"]
        x, B = load_counts(read_stack(s["path"]), a.ch_actin, read_tif(P / f"{img}_actin_bg4.tif"), a.bin)
        nuc = read_tif(N / f"{img}_nuclei.tif")
        r = segment_cells(x, B, nuc, cfg)
        save_ij(D / f"{img}_cells.tif", r["cells"], a.spacing)
        save_ij(D / f"{img}_fg.tif", r["fg"].astype(np.uint8) * 255, a.spacing)
        save_ij(D / f"{img}_clusters.tif", r["clusters"], a.spacing)
        _json(D / f"{img}_cells.json", dict(vars(cfg.cells), image=img, **r["info"]))
        qc.cells_qc(Q / f"{img}_cells.png", img, r["smoothed"], r["cells"], nuc)
        if crops:
            qc.cell_crops_qc(cfg.stage_dir("03_cells/qc_crops") / f"{img}_crops.png", img,
                             read_tif(P / f"{img}_actin.tif").astype(np.float32), r["cells"], nuc)

    return run_batch(items, lambda s: (D / f"{s['image']}_cells.json").exists(), process, only, force)


# ---------------------------------------------------------------------------------------------------- stage 4
def stage_measure(cfg: Config, items, only=None, force=False):
    D = cfg.stage_dir("04_measure")
    out = Path(cfg.data.out_dir)
    a, m, scale = cfg.acquisition, cfg.measure, cfg.preprocess.scale
    items = [s for s in items if (out / "03_cells" / f"{s['image']}_cells.json").exists()]

    def process(s):
        img = s["image"]
        nuc = read_tif(out / "02_nuclei" / f"{img}_nuclei.tif")
        dapi = read_tif(out / "01_preprocess" / f"{img}_dapi.tif").astype(np.float32) / scale
        dn = measure_objects(nuc, a.spacing, dapi, m.min_proj_area_um2)
        dn = dn.assign(**_meta(s))
        dn.insert(0, "image", img)
        dn.to_csv(D / f"{img}_nuclei.csv", index=False)
        cells = read_tif(out / "03_cells" / f"{img}_cells.tif")
        clusters = read_tif(out / "03_cells" / f"{img}_clusters.tif")
        actin = read_tif(out / "01_preprocess" / f"{img}_actin.tif").astype(np.float32) / scale
        dc = measure_objects(cells, a.spacing, actin, m.min_proj_area_um2)
        dc = add_cell_context(dc, cells, clusters, nuc, a.spacing, m.sheet_area_um2).assign(**_meta(s))
        dc.insert(0, "image", img)
        dc.to_csv(D / f"{img}_cells.csv", index=False)

    return run_batch(items, lambda s: (D / f"{s['image']}_cells.csv").exists(), process, only, force)


# ---------------------------------------------------------------------------------------------------- stage 5
def stage_fibres(cfg: Config, items, only=None, force=False):
    D, Q = cfg.stage_dir("05_fibres"), cfg.stage_dir("05_fibres/qc")
    out = Path(cfg.data.out_dir)
    a, p = cfg.acquisition, cfg.fibres
    items = [s for s in items if (out / "04_measure" / f"{s['image']}_cells.csv").exists()]

    def process(s):
        img = s["image"]
        tl_path = out / "01_preprocess" / f"{img}_tl.tif"
        if s["group"] not in cfg.design.fibre_groups or not tl_path.exists():
            for k in ("nuclei", "cells"):       # no fibres: pass the measurements through unchanged
                pd.read_csv(out / "04_measure" / f"{img}_{k}.csv").to_csv(D / f"{img}_{k}_align.csv", index=False)
            return
        flat, mask, fib, coh = fibre_fields(read_tif(tl_path), p)
        save_ij(D / f"{img}_fibre_mask.tif", mask.astype(np.uint8) * 255, a.spacing, axes="YX")
        save_ij(D / f"{img}_fibre_angle.tif", fib, a.spacing, axes="YX")
        save_ij(D / f"{img}_coherence.tif", coh, a.spacing, axes="YX")
        for k in ("nuclei", "cells"):
            df = pd.read_csv(out / "04_measure" / f"{img}_{k}.csv")
            align_to_fibres(df, mask, fib, coh, a.px_bin_um, p.on_fibre_um).to_csv(D / f"{img}_{k}_align.csv", index=False)
        qc.fibres_qc(Q / f"{img}_fibres.png", img, flat, mask, fib, coh)

    return run_batch(items, lambda s: (D / f"{s['image']}_cells_align.csv").exists(), process, only, force)


# ---------------------------------------------------------------------------------------------------- stage 6
def load_object_tables(cfg: Config):
    F = Path(cfg.data.out_dir) / "05_fibres"
    tabs = []
    for kind in ("nuclei", "cells"):
        files = sorted(F.glob(f"*_{kind}_align.csv"))
        if not files:
            raise FileNotFoundError(f"no *_{kind}_align.csv in {F}; run the fibres stage first")
        tabs.append(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
    return tabs


def stage_stats(cfg: Config, log=print):
    from .report import quicklook_figures
    from .stats import run_stats
    D = cfg.stage_dir("06_stats")
    nuc, cel = load_object_tables(cfg)
    nuc, cel, summ, tests, anova = run_stats(nuc, cel, cfg)
    nuc.to_csv(D / "all_nuclei.csv", index=False)
    cel.to_csv(D / "all_cells.csv", index=False)
    summ.to_csv(D / "per_image_summary.csv", index=False)
    tests.to_csv(D / "group_tests.csv", index=False)
    anova.to_csv(D / "anova_tests.csv", index=False)
    if len(anova):
        log(anova.reindex(columns=["metric", "test", "term", "cell_line", "F", "p", "p_holm", "AR_ratio"]).to_string())
    quicklook_figures(nuc, cel, summ, cfg, D)
    return summ, tests, anova


def run_stage(name: str, cfg: Config, only=None, force=False, **kw):
    if name in ("stats", "figures", "tables"):
        if name == "stats":
            stage_stats(cfg)
        elif name == "figures":
            from .report import make_figures
            make_figures(cfg)
        else:
            from .report import make_tables
            make_tables(cfg)
        return []
    items = list_stacks(cfg)
    if not items:
        raise FileNotFoundError(f"no stacks matching {cfg.data.name_pattern!r} under {cfg.data.raw_dir}/{cfg.data.glob}")
    fn = {"inventory": stage_inventory, "preprocess": stage_preprocess, "nuclei": stage_nuclei, "cells": stage_cells,
          "measure": stage_measure, "fibres": stage_fibres}[name]
    return fn(cfg, items, only=only, force=force, **kw)
