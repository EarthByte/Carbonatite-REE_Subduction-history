#!/usr/bin/env python
"""Craton-edge distance for every deposit and null point, at the age of the point.

Lithospheric architecture is the standing alternative explanation for every association in
this study: sediment-hosted base metals sit at craton edges (Hoggard et al. 2020; Huston et
al. 2023; Shirmard et al. 2025), and Glerum et al. (2026) argue that what looks like a rift
control may be the craton edge the rift followed. Testing that needs the architecture as a
covariate rather than as a picture, and it needs it at the time each deposit formed, not
today.

The measure is the signed distance to the nearest edge of the craton outlines of Shirmard et
al. (2025), delineated from REVEAL full-waveform tomography and cut by the static polygons of
the plate model (scripts/44_shirmard_cratons.py), reconstructed to the point's own time with the same rotation model as everything else:
negative inside a craton, positive outside. Present-day REVEAL shear-wave velocity at 150 km
and its horizontal gradient are sampled at the point's present-day position beside it, as the
link to the published results.

Output: ONE table, data/derived/craton_<tag>.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import load_config, read_table  # noqa: E402
from rifts.cratons import (craton_distance, craton_snapshot, gradient_grid,  # noqa: E402
                           load_cratons, sample_grid)
from rifts.models import load_models  # noqa: E402

BIN_MYR = 5.0     # craton polygons move less than 10 km over half a bin


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--min-cover", type=float, default=0.5,
                    help="withhold the reconstructed craton distance where fewer than this "
                         "fraction of the present-day craton polygons exist at that time")
    args = ap.parse_args()
    cfg = load_config(args.config)
    der = cfg["paths"]["derived"]
    ext = cfg["paths"]["external"]

    f = read_table(der / f"features_{args.tag}")
    p = (f[(f["lag_myr"] == 0) & (f["draw"] == 0)]
         .drop_duplicates("pt_id")[["pt_id", "dclass", "null_for", "lon0", "lat0",
                                    "mlon", "mlat", "time_ma", "in_model"]]
         .reset_index(drop=True))
    print(f"{len(p)} points, {int(p['mlon'].notna().sum())} of them reconstructable")

    mantle, _ = load_models(cfg)
    cratons = load_cratons(cfg["paths"]["craton_file"])
    print(f"{len(cratons)} craton features")

    now = craton_snapshot(cratons, mantle.rotation_model, 0.0)
    p["craton_present_km"] = craton_distance(now, p["lon0"], p["lat0"])
    n0 = len(now.polygons)

    p["craton_dist_km"] = np.nan
    p["craton_cover"] = np.nan
    p["bin"] = (np.rint(p["time_ma"] / BIN_MYR) * BIN_MYR).astype(float)
    counts = []
    for t, sub in p.groupby("bin"):
        ok = sub["mlon"].notna()
        if not ok.any():
            continue
        snap = craton_snapshot(cratons, mantle.rotation_model, t)
        if snap is None:
            counts.append((t, 0))
            continue
        counts.append((t, len(snap.polygons)))
        idx = sub.index[ok]
        p.loc[idx, "craton_cover"] = len(snap.polygons) / n0
        p.loc[idx, "craton_dist_km"] = craton_distance(snap, p.loc[idx, "mlon"],
                                                       p.loc[idx, "mlat"])
    # A craton polygon exists in the reconstruction only over its own valid time, which records
    # when the block was assembled in the plate model rather than when its keel formed. Before
    # about 1 Ga much of the set is absent, and the nearest *surviving* polygon can be an ocean
    # away, so the distance stops meaning what it means later. Where coverage falls below the
    # threshold the measure is withheld rather than reported as a large distance, and
    # ``craton_present_km``, which is the published measure, carries those points instead.
    thin = p["craton_cover"] < args.min_cover
    p.loc[thin, ["craton_dist_km"]] = np.nan
    p["craton_abs_km"] = p["craton_dist_km"].abs()
    print(f"{n0} craton polygons at the present day; coverage through time:")
    cc = pd.DataFrame(counts, columns=["time_ma", "n_poly"]).assign(cover=lambda x: x.n_poly / n0)
    step = cc.assign(era=pd.cut(cc["time_ma"], [-1, 250, 541, 1000, 1300, 1600, 1800]))
    print(step.groupby("era", observed=True)["cover"].agg(["min", "median", "max"]).round(2).to_string())
    print(f"{int(thin.sum())} points fall below the {args.min_cover:.0%} coverage threshold and "
          f"carry no reconstructed craton distance")

    path = cfg["paths"].get("vsh_file", ext / "reveal_vsh_150km.nc")
    if Path(path).exists():
        import xarray as xr
        da = xr.open_dataset(path)["vsh"]
        p["vsh150"] = sample_grid(da, p["lon0"], p["lat0"])
        g = da.copy(data=gradient_grid(da))
        p["vsh_grad150"] = sample_grid(g, p["lon0"], p["lat0"])
    else:
        print(f"{path} not found: REVEAL velocity columns skipped")

    keep = ["pt_id", "craton_dist_km", "craton_abs_km", "craton_present_km", "craton_cover"]
    keep += [c for c in ("vsh150", "vsh_grad150") if c in p]
    out = der / f"craton_{args.tag}.csv"
    tmp = out.with_suffix(out.suffix + ".partial")
    p[keep].to_csv(tmp, index=False)
    tmp.replace(out)
    print(f"wrote {out}")

    p["group"] = p["dclass"].where(p["dclass"] != "NULL", p["null_for"])
    dep = p[p["dclass"] != "NULL"]
    t = dep.groupby("group").agg(n=("pt_id", "size"),
                                 within_200_now=("craton_present_km", lambda s: (s.abs() <= 200).mean()),
                                 within_120_now=("craton_present_km", lambda s: (s.abs() <= 120).mean()),
                                 within_200_then=("craton_abs_km", lambda s: (s <= 200).mean()),
                                 inside=("craton_dist_km", lambda s: (s < 0).mean()),
                                 med_then=("craton_abs_km", "median"))
    nul = p[p["dclass"] == "NULL"].groupby("group").agg(
        null_within_200_then=("craton_abs_km", lambda s: (s <= 200).mean()),
        null_med_then=("craton_abs_km", "median"))
    print("\nshare of each class within reach of a craton edge, at the present day and at the "
          "age of the deposit, against its own nulls")
    print(t.join(nul).round(3).to_string())
    era = pd.cut(dep["time_ma"], [-1, 250, 541, 1000, 1300, 1600, 1800])
    print("\nthe same measure by age, as a check on the reconstruction: a rising distance and a "
          "falling share inside a craton with age is the coverage effect, not geology")
    print(dep.groupby(era, observed=True).agg(n=("pt_id", "size"),
                                              med_then=("craton_abs_km", "median"),
                                              inside=("craton_dist_km", lambda s: (s < 0).mean()),
                                              med_now=("craton_present_km", "median"))
          .round(1).to_string())
    print("\nHoggard et al. (2020) put about 85 % of sediment-hosted base metals within 200 km of "
          "the present-day lithospheric step and Shirmard et al. (2025) about 85 % within 120 km "
          "of a REVEAL craton bound; the first two columns are this compilation's version of "
          "those statements on the clastic-dominated Zn-Pb (CD) and sediment-hosted Cu (SSC) rows.")
    if "vsh150" in p:
        v = p.groupby(p["dclass"] != "NULL")[["vsh150", "vsh_grad150"]].median()
        print("\nmedian REVEAL velocity and gradient, nulls then deposits")
        print(v.round(4).to_string())


if __name__ == "__main__":
    main()
