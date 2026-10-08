"""Command-line interface: ``cellshape3d <stage> [-c config.toml] [--only img1,img2] [--force]``."""
from __future__ import annotations

import argparse
import sys

from .config import load_config
from .pipeline import STAGES, run_stage

IMAGE_STAGES = STAGES[:6]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cellshape3d", description=__doc__)
    ap.add_argument("stage", choices=STAGES + ["all", "list"],
                    help="stage to run; 'all' runs preprocess..tables; 'list' prints the discovered stacks")
    ap.add_argument("-c", "--config", help="TOML config file (default: built-in defaults)")
    ap.add_argument("--only", help="comma-separated image ids to process (per-image stages)")
    ap.add_argument("--force", action="store_true", help="recompute images whose outputs already exist")
    ap.add_argument("--crops", action="store_true", help="cells stage: also write zoomed QC crops")
    ap.add_argument("--nuclei-method", choices=["stardist", "classical"], help="override [nuclei] method")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.nuclei_method:
        cfg.nuclei.method = args.nuclei_method
    only = args.only.split(",") if args.only else None

    if args.stage == "list":
        from .io import list_stacks
        for s in list_stacks(cfg):
            print(f"{s['image']}\t{s['group']}\t{s['cell_line']}\t{s['team']}\t{s['path']}")
        return 0

    stages = STAGES[1:] if args.stage == "all" else [args.stage]
    failed = []
    for st in stages:
        print(f"== {st} ==", flush=True)
        kw = {"crops": True} if (st == "cells" and args.crops) else {}
        failed += run_stage(st, cfg, only=only if st in IMAGE_STAGES else None, force=args.force, **kw) or []
    if failed:
        print(f"FAILED images: {sorted(set(failed))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
