#!/usr/bin/env python
"""Attach present-day host-basin (Evenick 2021) and nearest-rift (GloRiDa) context to the
deposits and to the null points of a features table.

Writes ONE table data/derived/context_<tag>. GloRiDa is used under a separate licence; raw GloRiDa files are not redistributed.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import load_config, read_table, write_table  # noqa: E402
from rifts.context import evenick_join, glorida_join, rift_stage  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--out-tag", default=None, help="output tag (default: --tag)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    f = read_table(cfg["paths"]["derived"] / f"features_{args.tag}")
    pts = (f[["pt_id", "lon0", "lat0", "age_ma"]].drop_duplicates("pt_id")
           .rename(columns={"lon0": "lon", "lat0": "lat"}).reset_index(drop=True))
    out = pts[["pt_id"]]
    basins = Path(cfg["paths"]["basins_file"])
    if basins.exists():
        out = out.merge(evenick_join(pts, basins), on="pt_id", how="left")
        print(f"basins joined: {out['basin_name'].notna().sum()} of {len(out)} points inside a basin")
    c = cfg["context"]
    rifts = (Path(cfg["paths"]["rift_db_file"]) if c["rift_source"] == "structures"
             else cfg["paths"]["derived"] / "rift_outlines.gpkg")
    if rifts.exists():
        r = glorida_join(pts, rifts, c["rift_max_km"], c["rift_max_lead_myr"], c["rift_stage_tol_myr"],
                         source=c["rift_source"])
        for pre in ("rift", "arift"):
            r[f"{pre}_stage"] = rift_stage(pts["age_ma"], r[f"{pre}_onset_ma"], r[f"{pre}_end_ma"],
                                           c["rift_stage_tol_myr"])
            r[f"{pre}_onset_minus_age_myr"] = r[f"{pre}_onset_ma"] - pts["age_ma"].to_numpy()
        out = out.merge(r, on="pt_id", how="left")
        print(f"rifts joined: {r['rift_id'].notna().sum()} of {len(r)} points within {c['rift_max_km']} km "
              f"of a dated rift; {r['arift_id'].notna().sum()} of an age-compatible one")
    path = write_table(pd.DataFrame(out), cfg["paths"]["derived"] / f"context_{args.out_tag or args.tag}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
