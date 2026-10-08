"""Pipeline configuration: one TOML file -> nested dataclasses.

Every section and key is optional; missing keys fall back to the defaults below (the values used for the
BMEG5001 2026 data set: Leica HC PL APO 10x/0.40 dry, zoom 1.8, 0.63 um/px, z-step 3.16 um).
Relative paths in ``[data]`` are resolved against the directory that contains the config file.
"""
from __future__ import annotations

import dataclasses as dc
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@dc.dataclass
class DataCfg:
    raw_dir: Path = Path("data/raw")
    out_dir: Path = Path("results")
    glob: str = "**/*.tif"
    # Regex searched in each file stem. Named groups ``group`` and ``cell_line`` are required, ``team`` (block /
    # replicate id) is optional; ``image`` overrides the image id (default: the whole match).
    name_pattern: str = r"(?P<group>GS|G|S)-(?P<cell_line>HEK|HeLa)-G(?P<team>\d+)_\d+"


@dc.dataclass
class AcquisitionCfg:
    px_xy_um: float = 0.63      # raw pixel size
    dz_um: float = 3.16         # z-step (not stored in the Leica-exported TIFF metadata)
    bin: int = 2                # xy binning used for segmentation
    ch_dapi: int = 0            # 0-based channel indices in the file
    ch_actin: int = 1
    ch_tl: int | None = 2       # transmitted light; None if not acquired (disables fibre analysis)

    @property
    def px_bin_um(self) -> float:
        return self.px_xy_um * self.bin

    @property
    def spacing(self) -> tuple[float, float, float]:
        """(z, y, x) voxel size in um after binning."""
        return (self.dz_um, self.px_bin_um, self.px_bin_um)


@dc.dataclass
class DesignCfg:
    groups: list[str] = dc.field(default_factory=lambda: ["G", "GS", "S"])     # first = reference level
    fibre_groups: list[str] = dc.field(default_factory=lambda: ["GS", "S"])    # groups that contain fibres
    cell_lines: list[str] = dc.field(default_factory=lambda: ["HEK", "HeLa"])
    block: str | None = "team"                                                 # blocking factor or None


@dc.dataclass
class PreprocessCfg:
    gauss_sigma_zyx: tuple[float, float, float] = (0.5, 1.0, 1.0)
    bg_downsample: int = 4
    bg_open_um: float = 40.0          # opening size of the lower-envelope background
    norm_percentile: float = 99.9
    scale: int = 10000                # saved uint16 value = (I / p99.9) * scale
    bgmed_win_um: float = 50.0        # local-median background (classical nuclei detector)
    bgmed_edge_dilate_um: float = 20.0


@dc.dataclass
class NucleiCfg:
    method: str = "stardist"          # "stardist" or "classical"
    model: str = "2D_versatile_fluo"
    pre_smooth_xy: float = 1.0
    norm_pct: tuple[float, float] = (1.0, 99.8)
    prob_thresh: float = 0.5
    nms_thresh: float = 0.4
    stitch_iou: float = 0.25
    min_area_px: int = 12
    z_split_dip: float = 0.75
    max_area2d_um2: float = 450.0
    min_vol_um3: float = 100.0
    min_contrast: float = 1.5
    # classical detector (no TensorFlow; weaker in autofluorescent haze, see docs/methods.md)
    cl_smooth_zyx: tuple[float, float, float] = (0.7, 1.5, 1.5)
    cl_seed_smooth_zyx: tuple[float, float, float] = (1.0, 2.5, 2.5)
    cl_seed_radius_um: tuple[float, float] = (6.3, 5.0)
    cl_k_poisson: float = 6.0
    cl_f_contrast: float = 1.0
    cl_b_floor: float = 0.1
    cl_frac_of_peak: float = 0.3
    cl_min_nucleus_um3: float = 80.0


@dc.dataclass
class CellsCfg:
    smooth_zyx: tuple[float, float, float] = (0.7, 1.0, 1.0)
    k_poisson: float = 6.0
    f_contrast: float = 1.0
    b_floor: float = 0.1
    min_fg_um3: float = 50.0


@dc.dataclass
class MeasureCfg:
    min_proj_area_um2: float = 20.0
    sheet_area_um2: float = 2500.0


@dc.dataclass
class FibresCfg:
    flat_sigma_px: float = 60
    grad_sigma_px: float = 2.0
    integ_sigma_px: float = 6.0
    min_fibre_area_px: int = 300
    on_fibre_um: float = 10.0


@dc.dataclass
class StatsCfg:
    nuc_area_um2: tuple[float, float] = (30.0, 600.0)
    cell_min_area_um2: float = 50.0
    min_objects_per_image: int = 10
    align_coherence_min: float = 0.6
    aligned_deg: float = 20.0
    elong_ar: float = 1.5
    alpha: float = 0.05


@dc.dataclass
class ReportCfg:
    panel_cell_lines: list[str] | None = None   # panel order in Fig. 1 (default: design.cell_lines)
    bin_deg: int = 10
    seed: int = 0


@dc.dataclass
class Config:
    data: DataCfg = dc.field(default_factory=DataCfg)
    acquisition: AcquisitionCfg = dc.field(default_factory=AcquisitionCfg)
    design: DesignCfg = dc.field(default_factory=DesignCfg)
    preprocess: PreprocessCfg = dc.field(default_factory=PreprocessCfg)
    nuclei: NucleiCfg = dc.field(default_factory=NucleiCfg)
    cells: CellsCfg = dc.field(default_factory=CellsCfg)
    measure: MeasureCfg = dc.field(default_factory=MeasureCfg)
    fibres: FibresCfg = dc.field(default_factory=FibresCfg)
    stats: StatsCfg = dc.field(default_factory=StatsCfg)
    report: ReportCfg = dc.field(default_factory=ReportCfg)

    def stage_dir(self, name: str) -> Path:
        d = Path(self.data.out_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        return d


def _build(cls, values: dict[str, Any]):
    fields = {f.name: f for f in dc.fields(cls)}
    unknown = set(values) - set(fields)
    if unknown:
        raise ValueError(f"unknown key(s) in [{cls.__name__}]: {sorted(unknown)}")
    kw = {}
    for k, v in values.items():
        default = getattr(cls(), k)
        if dc.is_dataclass(default):
            kw[k] = _build(type(default), v)
        elif isinstance(default, tuple):
            kw[k] = tuple(v)
        elif isinstance(default, Path):
            kw[k] = Path(v)
        else:
            kw[k] = v
    return cls(**kw)


def config_from_dict(d: dict[str, Any], base_dir: Path | None = None) -> Config:
    if "acquisition" in d and d["acquisition"].get("ch_tl", 0) == -1:
        d["acquisition"]["ch_tl"] = None          # TOML has no null
    cfg = _build(Config, d)
    if base_dir is not None:
        for k in ("raw_dir", "out_dir"):
            p = getattr(cfg.data, k)
            if not p.is_absolute():
                setattr(cfg.data, k, (base_dir / p).resolve())
    return cfg


def load_config(path: str | Path | None = None) -> Config:
    """Load a TOML config; ``None`` returns the defaults (paths relative to the working directory)."""
    if path is None:
        return Config()
    path = Path(path)
    with open(path, "rb") as f:
        return config_from_dict(tomllib.load(f), base_dir=path.resolve().parent)
