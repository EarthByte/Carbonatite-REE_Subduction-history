#!/usr/bin/env python
"""Sample plate kinematics for deposits and their age-matched null points.

For every point and every lag in config (formation age + lag) the script records, in the
mantle model: reconstructed position, absolute plate speed and azimuth, distance to the
nearest divergent / continental-divergent / convergent / subduction / ridge boundary with
the rate, obliquity and boundary motion there, and boundary length and mean rate within
the configured radii; and, in the palaeomagnetic model, palaeolatitude and plate speed.

Output: ONE table, data/derived/features_<tag>.parquet (or .csv.gz).

Start with a pilot:   python scripts/02_sample_kinematics.py --pilot
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import load_config, read_table, write_table  # noqa: E402
from rifts.kinematics import sample_points  # noqa: E402
from rifts.models import load_models  # noqa: E402
from rifts.nulls import age_matched_nulls, continental_candidates  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--classes", nargs="*", help="deposit classes to include (default: all)")
    ap.add_argument("--pilot", action="store_true",
                    help="30 deposits per class, 300 nulls per class, lags 0 and 20 only")
    ap.add_argument("--nulls-per-class", type=int, default=None)
    ap.add_argument("--age-draws", type=int, default=0,
                    help="extra Monte Carlo realisations of each deposit age within its range")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--deposits", default="deposits",
                    help="deposit table in data/derived (default: the shared compilation; the "
                         "kimberlite paper uses deposits_kimb23 from scripts/72)")
    ap.add_argument("--null-weights", choices=("locality", "district"), default="locality",
                    help="draw reference ages from the localities of a class (default) or with each "
                         "district weighted equally, matching a district-weighted comparison")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg["nulls"]["seed"])
    dep = read_table(cfg["paths"]["derived"] / args.deposits)
    if args.classes:
        dep = dep[dep["dclass"].isin(args.classes)]
    lags = cfg["time"]["lags_myr"]
    n_null = args.nulls_per_class or max(500, cfg["nulls"]["n_points"] // max(dep["dclass"].nunique(), 1))
    if args.pilot:
        dep = dep.sample(frac=1, random_state=1).groupby("dclass").head(30)
        lags, n_null = [0, 20], 300
    tag = args.tag or ("pilot" if args.pilot else "full")

    pts = dep.assign(pt_id=dep["dep_id"], null_for="", draw=0)
    if args.age_draws:
        reps = []
        for k in range(1, args.age_draws + 1):
            lo, hi = dep["age_min_ma"].to_numpy(), dep["age_max_ma"].to_numpy()
            a = rng.uniform(np.minimum(lo, hi), np.maximum(lo, hi))
            reps.append(dep.assign(age_ma=a, pt_id=dep["dep_id"] + f"#{k}", null_for="", draw=k))
        pts = pd.concat([pts] + reps, ignore_index=True)

    mantle, paleomag = load_models(cfg)
    print("null candidates")
    cand = continental_candidates(mantle, 4 * n_null, seed=cfg["nulls"]["seed"])
    nulls = []
    for c, g in dep.groupby("dclass"):
        terr = g["pid"].unique() if cfg["nulls"]["mode"] == "class_terranes" else None
        w = g["w_cluster"].to_numpy(float) if args.null_weights == "district" else None
        nn = age_matched_nulls(cand, g["age_ma"].to_numpy(), n_null, seed=cfg["nulls"]["seed"] + sum(map(ord, c)),
                               terrane_pids=terr, weights=w)
        nn["pt_id"] = [f"NULL_{c}_{i:05d}" for i in range(len(nn))]
        nn["dclass"], nn["null_for"], nn["draw"] = "NULL", c, 0
        nulls.append(nn)
    nulls = pd.concat(nulls, ignore_index=True)
    nulls["pid_pm"], _, _ = paleomag.assign_plate_ids(nulls["lon"], nulls["lat"])
    allpts = pd.concat([pts, nulls], ignore_index=True)
    print(f"{len(pts)} deposit points, {len(nulls)} null points, {len(lags)} lags")

    t0 = time.time()
    feat = sample_points(allpts[["pt_id", "lon", "lat", "pid", "pid_pm", "age_ma", "poly_begin_ma"]],
                         mantle, paleomag, lags, cfg)
    allpts["cluster_id"] = allpts["cluster_id"].fillna(allpts["pt_id"])
    allpts["w_cluster"] = allpts["w_cluster"].fillna(1.0)
    meta = allpts[["pt_id", "dclass", "null_for", "draw", "age_ma", "age_basis", "subtype", "metal_mt",
                   "source", "lon", "lat", "pid", "cluster_id", "w_cluster"]].rename(columns={"lon": "lon0", "lat": "lat0"})
    feat = feat.merge(meta, on="pt_id", how="left")
    out = write_table(feat, cfg["paths"]["derived"] / f"features_{tag}")
    print(f"wrote {out} ({len(feat)} rows) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
