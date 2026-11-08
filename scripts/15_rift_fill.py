#!/usr/bin/env python
"""Sediment fill, crustal thinning and the syn-rift share of every rift, and of every point.

Reads the Bird and Mooney GST-1 sediment-thickness grid and the crustal thickness of the Earth
Crustal Model 1 (ECM1; Mooney et al. 2023), gridded by scripts/51_ecm1_crust.py. Two outputs
come from them.

Per point, for deposits and nulls alike: the fill above it, the crust beneath it and McKenzie's
stretching factor. These replace the basin-polygon averages of `01b_geological_context.py` for
the half of the points that fall outside a mapped basin.

Per rift: a rift-normal transect through every bin of the rift's spine, a wedge-plus-drape fit
to the sediment thickness along it, and the length-weighted syn-rift and post-rift thickness that
follow, beside McKenzie's independent prediction of the same split from the rift's stretching
factor and age.

Writes data/derived/rift_fill_<tag>.csv and data/derived/point_fill_<tag>.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.basins import Grid, mckenzie_prediction, rift_fill, stretching_factor  # noqa: E402
from rifts.common import load_config, read_table  # noqa: E402
from rifts.obliquity import spine  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--transect-factor", type=float, default=3.0,
                    help="transect length as a multiple of the local rift width")
    ap.add_argument("--transect-points", type=int, default=81)
    ap.add_argument("--max-bins", type=int, default=9)
    args = ap.parse_args()
    cfg = load_config(args.config)
    der = cfg["paths"]["derived"]
    ext = Path(cfg["paths"].get("external", "data/external"))
    sed = Grid(ext / "Bird_Mooney_GST14_WGS84.nc")
    crust = Grid(cfg["paths"]["crust_file"])                 # ECM1 (scripts/51), km
    print(f"grids: sediment {sed.z.shape}, crust {crust.z.shape}")

    # ---------------------------------------------------------------- per point
    f = read_table(der / f"features_{args.tag}")
    pts = f[["pt_id", "lon0", "lat0"]].drop_duplicates("pt_id")
    p = pd.DataFrame({"pt_id": pts["pt_id"].to_numpy(),
                      "site_sed_m": sed.sample(pts["lon0"], pts["lat0"]),
                      "site_crust_km": crust.sample(pts["lon0"], pts["lat0"])})
    p["site_beta"] = stretching_factor(p["site_crust_km"])
    out = der / f"point_fill_{args.tag}.csv"
    tmp = out.with_suffix(out.suffix + ".partial")
    p.to_csv(tmp, index=False)
    tmp.replace(out)
    print(f"wrote {out}  ({p['site_sed_m'].notna().sum()} of {len(p)} points sampled)")

    # ---------------------------------------------------------------- per rift
    if not (der / "rift_outlines.gpkg").exists():
        # the per-point measures above are all that a workflow without the rift outlines needs
        # (the carbonatite paper); the per-rift decomposition needs the outlines of scripts/06
        print("rift_outlines.gpkg not found: per-point fill written, per-rift decomposition skipped")
        return
    import geopandas as gpd
    o = gpd.read_file(der / "rift_outlines.gpkg", layer="outlines")
    rows = []
    for _, r in o.iterrows():
        sp = spine(r.geometry, pad_km=50.0, max_bins=args.max_bins)
        if sp is None:
            continue
        m = rift_fill(sp, sed, crust, args.transect_factor, args.transect_points)
        if m:
            rows.append(dict(outline_id=r["outline_id"], **m))
    rf = pd.DataFrame(rows)
    meta = o[["outline_id", "onset_ma", "end_ma", "is_active"]].copy()
    meta["end_ma"] = meta["end_ma"].where(meta["end_ma"].notna(),
                                          np.where(meta["is_active"].astype(bool), 0.0, np.nan))
    rf = rf.merge(meta.drop(columns=["is_active"]), on="outline_id", how="left")
    pred = mckenzie_prediction(rf["beta"], rf["onset_ma"], rf["end_ma"])
    rf = pd.concat([rf, pred], axis=1)
    rf["syn_share_excess"] = rf["syn_share"] - rf["pred_syn_share"]
    out2 = der / f"rift_fill_{args.tag}.csv"
    tmp = out2.with_suffix(out2.suffix + ".partial")
    rf.to_csv(tmp, index=False)
    tmp.replace(out2)
    print(f"wrote {out2}  ({len(rf)} rifts)")

    good = rf[rf["fit_resid_rel"] < 0.5]
    print(f"\n{len(good)} rifts fitted with a residual below half the thickness range")
    print(f"  fill      median {good['fill_total_m'].median():.0f} m "
          f"(wedge {good['fill_wedge_m'].median():.0f}, drape {good['fill_drape_m'].median():.0f})")
    print(f"  syn share median {good['syn_share'].median():.2f} observed against "
          f"{good['pred_syn_share'].median():.2f} predicted")
    print(f"  beta      median {good['beta'].median():.2f}")
    b = good.dropna(subset=["onset_ma"]).copy()
    b["era"] = pd.cut(b["onset_ma"], [-1, 100, 250, 540, 3000],
                      labels=["0-100 Ma", "100-250 Ma", "250-540 Ma", ">540 Ma"])
    print("\nby rift onset age:")
    print(b.groupby("era", observed=True)[["syn_share", "pred_syn_share", "beta", "fill_total_m"]]
          .median().round(2).to_string())


if __name__ == "__main__":
    main()
