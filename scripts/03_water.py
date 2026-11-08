#!/usr/bin/env python
"""Slab-water and subducted-carbon delivery beneath each deposit and null point (mantle frame).

Reads the point list of a features table, samples the cumulative water grids and writes ONE
table data/derived/water_<tag>.
"""
import argparse
import sys

import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import load_config, read_table, write_table  # noqa: E402
from rifts.models import load_models  # noqa: E402
from rifts.water import WaterGrids, water_metrics  # noqa: E402


def bad_steps_for(cfg, name):
    """Steps flagged by scripts/10_grid_audit.py (column `flag`), plus any listed in config."""
    steps = set(next((g.get("bad_steps", []) for g in cfg["water"]["grid_sets"] if g["name"] == name), []))
    audit = cfg["paths"]["derived"] / "grid_audit.csv"
    if audit.exists():
        a = pd.read_csv(audit)
        steps |= set(a.loc[(a["grid_set"] == name) & a["flag"], "age_ma"].astype(int))
    return sorted(steps)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--grid-dir", help="override the directory of the first grid set")
    ap.add_argument("--sets", nargs="*", help="grid sets to sample (default: all in config)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    feat = read_table(cfg["paths"]["derived"] / f"features_{args.tag}")
    pts = (feat[["pt_id", "age_ma", "lon0", "lat0", "pid"]].drop_duplicates("pt_id")
           .rename(columns={"lon0": "lon", "lat0": "lat"}))
    mantle, _ = load_models(cfg)
    sets = cfg["water"].get("grid_sets") or [dict(name="h2o", dir=cfg["paths"]["water_grid_dir"],
                                                  folder_template=cfg["paths"]["water_folder_template"],
                                                  variables=cfg["water"]["variables"],
                                                  components=cfg["water"]["components"])]
    if args.grid_dir:
        sets[0]["dir"] = args.grid_dir
    if args.sets:
        sets = [g for g in sets if g["name"] in args.sets]
    w = pts[["pt_id"]].copy()
    for gs in sets:
        grids = WaterGrids(gs["dir"], cfg["paths"]["water_grid_glob"], gs["variables"], gs.get("components"),
                           gs.get("folder_template", "{component}"), bad_steps_for(cfg, gs["name"]))
        ages = grids.ages
        # the carbon organic/carbonate sets cover only part of the record; sample where grids exist
        sel = pts["age_ma"].round().between(ages.min(), ages.max())
        print(f"{gs['name']}: {len(ages)} grid ages {ages.min()}-{ages.max()} Ma; {sel.sum()} points")
        if grids.bad_steps:
            print(f"  correcting spurious steps at {grids.bad_steps} Ma")
        r = water_metrics(pts[sel], mantle, grids, gs.get("windows_myr", cfg["water"]["windows_myr"]),
                          cfg["water"]["along_track"], stock=gs.get("stock", False))
        r = r.rename(columns={c: f"{gs['name']}_{c}" for c in r.columns if c != "pt_id"})
        w = w.merge(r, on="pt_id", how="left")
    target = cfg["paths"]["derived"] / f"water_{args.tag}"
    if args.sets:
        # re-running some sets: keep the other sets' columns from the existing table
        try:
            old = read_table(target)
            names = tuple(f"{g['name']}_" for g in sets)
            keep = [c for c in old.columns if c == "pt_id" or not c.startswith(names)]
            w = old[keep].merge(w, on="pt_id", how="outer")
        except FileNotFoundError:
            pass
    out = write_table(w, target)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
