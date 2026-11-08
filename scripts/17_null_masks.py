#!/usr/bin/env python
"""Flag every point for the two conditions that would make a null an unfair reference.

A null point stands for a place where a deposit of the class could have formed and been
found. It cannot do that under an ice sheet, and it cannot do that in a region nobody has
explored, because a region nobody has explored holds no deposits of any kind and a
difference measured against it is a statement about exploration history.

Exploration is measured as the distance to the nearest deposit of another family in the
same compilation (the Zn-Pb classes count as one family, the mantle-sourced classes as
another), so the footprint of a class is never defined by deposits found while looking for
that class. The same distance is computed for the deposits themselves, because a radius
that excludes the deposits it is meant to keep is not a footprint but a filter.

Output: ONE table, data/derived/null_mask_<tag>.csv, read by 04_screen.py --null-set.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import load_config, read_table  # noqa: E402
from rifts.nulls import family, ice_mask, nearest_km  # noqa: E402

RADII = (250, 500, 1000)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    cfg = load_config(args.config)
    der = cfg["paths"]["derived"]

    f = read_table(der / f"features_{args.tag}")
    p = (f.drop_duplicates("pt_id")[["pt_id", "dclass", "null_for", "draw", "lon0", "lat0", "pid"]]
         .query("draw == 0").reset_index(drop=True))
    # reference points of scripts/57 carry "<class>|<kind of rift>"; the class sets the family
    p["group"] = p["dclass"].where(p["dclass"] != "NULL", p["null_for"].astype(str).str.split("|").str[0])
    # "reference", not "null": the word NULL round-trips through a CSV as a missing value
    p["kind"] = np.where(p["dclass"] == "NULL", "reference", "deposit")
    print(f"{len(p)} points ({(p['kind'] == 'reference').sum()} nulls) from features_{args.tag}")

    ice_paths = [cfg["paths"][k] for k in ("ice_file", "ice_shelf_file") if k in cfg["paths"]]
    p["ice"] = ice_mask(p["lon0"], p["lat0"], ice_paths)

    dep = read_table(der / "deposits")
    dep = dep[np.isfinite(dep["lon"]) & np.isfinite(dep["lat"])]
    dep["family"] = [family(c) for c in dep["dclass"]]
    p["other_family_km"] = np.nan
    for g, sub in p.groupby("group"):
        ref = dep[dep["family"] != family(g)]
        p.loc[sub.index, "other_family_km"] = nearest_km(sub["lon0"], sub["lat0"],
                                                         ref["lon"].to_numpy(), ref["lat"].to_numpy())

    host = {c: {int(v) for v in g} for c, g in dep.groupby("dclass")["pid"]}
    p["terrane_ok"] = [int(pid) in host.get(g, set()) for pid, g in zip(p["pid"], p["group"])]
    p.loc[p["kind"] == "deposit", "terrane_ok"] = True

    out = der / f"null_mask_{args.tag}.csv"
    tmp = out.with_suffix(out.suffix + ".partial")
    p.drop(columns=["draw"]).to_csv(tmp, index=False)
    tmp.replace(out)
    print(f"wrote {out}")

    rows = []
    for g, sub in p.groupby("group"):
        d, n = sub[sub["kind"] == "deposit"], sub[sub["kind"] == "reference"]
        r = dict(group=g, n_dep=len(d), n_null=len(n),
                 dep_ice=float(d["ice"].mean()) if len(d) else np.nan,
                 null_ice=float(n["ice"].mean()) if len(n) else np.nan,
                 null_terrane=float(n["terrane_ok"].mean()) if len(n) else np.nan)
        for km in RADII:
            r[f"dep_in_{km}"] = float((d["other_family_km"] <= km).mean()) if len(d) else np.nan
            r[f"null_in_{km}"] = float((n["other_family_km"] <= km).mean()) if len(n) else np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    print("\nshare of each side inside the explored footprint, at three radii, and under ice")
    print(t.round(3).to_string(index=False))
    print("\nA radius is usable for a class when nearly all of its deposits sit inside it: the "
          "mask is meant to remove nulls in unexplored ground, not deposits in remote ground. "
          "Nulls remaining after ice and footprint, and after the terrane restriction, are what "
          "the masked screens are measured on.")
    for km in RADII:
        left = p[(p["kind"] == "reference") & ~p["ice"] & (p["other_family_km"] <= km)]
        print(f"  {km:>5} km: {len(left):6d} nulls left of {(p['kind'] == 'reference').sum()}, "
              f"fewest for one class {left.groupby('group').size().min() if len(left) else 0}")
    auto = (p[p["kind"] == "deposit"].groupby("group")["other_family_km"]
            .quantile(0.95).clip(lower=250.0).round(0))
    print("\nthe per-class radius that keeps 95 % of the class's own deposits (--explored-km auto), "
          "floored at 250 km")
    print(auto.to_string())
    terr = p[(p["kind"] == "reference") & p["terrane_ok"]]
    print(f"  terranes: {len(terr):6d} nulls left, fewest for one class "
          f"{terr.groupby('group').size().min() if len(terr) else 0}")


if __name__ == "__main__":
    main()
