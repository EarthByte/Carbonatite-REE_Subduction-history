#!/usr/bin/env python
"""Subduction-fertilised mantle domains of Spandler et al. (2026) tested against reference points.

Spandler, Merdith & Griffin (2026, Sci. Adv. 12, eaeb2942) drew polygons 900 km inboard of
convergent margins that were active for at least 100 Myr at a fixed position, and found 67% of
304 carbonatites and 72% of 108 magmatic REE deposits inside a polygon older than them, against
33% of random points on present-day continental crust. This step

1. reproduces their counts from their published polygons and tables (kind ``reproduction``);
2. repeats the test on this study's compilation, district-weighted, against the age-matched
   reference points of each class, with points beneath present-day ice removed: the share of
   deposits and of reference points inside a polygon formed before them (kind ``share``);
3. repeats it at equal distance to the nearest craton outline (Shirmard et al. 2025; at the
   age of the point and at the present day), at equal exploration density (distance to the
   nearest deposit of another family) and at equal shear-wave velocity gradient at 150 km
   (kind ``share at equal``): within quintiles of the conditioning variable, reference shares
   are averaged with the deposits' weights;
4. splits their own REE ore deposits by host (carbonatite, alkaline-silicate, volcanic) and asks,
   for each host and for their carbonatites, how often a deposit lies inside a polygon formed
   before it, and inside one formed at most 400, 800 or 1,200 Myr before it, against the share
   expected at the deposit ages: the share of this study's reference positions on continental
   crust that a polygon of that age covers, averaged over the deposits, over all positions and
   within the deposit's quintile of present-day craton-edge distance (kind ``spandler by host``);
5. asks whether a polygon marks more subducted volatiles: among the reference points, the
   probability that a point inside a polygon has more slab water or carbon than a point
   outside (kind ``polygon vs volatiles``), for every delivery measure of
   data/derived/water_<tag> and, where it exists, primer_history_<tag> (scripts/82), pooled over all
   reference points and within deciles of their ages (group "... within age deciles"), because a
   polygon older than the point is more often available to a young point.

Intervals come from 1,000 resamples of districts (deposits) and of reference points.
Output: ONE table, data/derived/spandler_reassessment_<tag>.csv, through a .partial name.

    python scripts/81_spandler_reassessment.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import ROOT, load_config, read_table  # noqa: E402

SPANDLER = ROOT / "Papers" / "Spandler_carbonatites_subduction_lithospheric_mantle_fertilisation"
POLY_FILES = ["Li_Polygons_on_Cao_0-900_km.gpml", "mesopro_polygons_0-900_km.gpml",
              "neopro_polygons_0-900_km.gpml", "phanero_polygons_0-900_km.gpml"]
CLASSES = ["CARB", "AIREE", "KIMB", "PCU"]
CONDITION = {"craton_dist_km": "distance to the nearest craton outline at the age of the point",
             "craton_present_km": "distance to the nearest craton outline at the present day",
             "other_family_km": "distance to the nearest deposit of another family (exploration density)",
             "vsh_grad150": "horizontal gradient of shear-wave velocity at 150 km"}


def load_polygons(folder):
    """The fertilised-domain polygons of Spandler et al. (2026) from the four POLY_FILES, as (polygon, age)
    pairs; the age is the begin time of the feature, 2000 Ma where it is the distant past."""
    import pygplates
    out = []
    for f in POLY_FILES:
        for ft in pygplates.FeatureCollection(str(folder / "data" / "mantle_fertilisation_polygons" / f)):
            a = ft.get_valid_time()[0]
            a = 2000.0 if np.isinf(a) else float(a)
            for g in ft.get_all_geometries():
                out.append((pygplates.PolygonOnSphere(g), a))
    return out


def polygon_ages(lat, lon, polys):
    import pygplates
    res = []
    for la, lo in zip(lat, lon):
        pt = pygplates.PointOnSphere(float(la), float(lo))
        res.append([a for g, a in polys if g.is_point_in_polygon(pt)])
    return res


def spandler_tables(folder):
    """Tables S1 (carbonatites) and S2 (REE ore deposits) of Spandler et al. (2026) with numeric age,
    latitude and longitude; rows missing any of the three are dropped."""
    x = pd.read_excel(folder / "sciadv.aeb2942_tables_s1_and_s2" / "aeb2942_table_s1.xlsx", sheet_name=None, header=None)
    c = x["carbonatites"].iloc[3:, :5]
    c.columns = ["country", "name", "age", "lat", "lon"]
    r = x["REE ore deposits"].iloc[2:, :6]
    r.columns = ["name", "country", "type", "age", "lat", "lon"]
    out = []
    for d in (c, r):
        for k in ("age", "lat", "lon"):
            d[k] = pd.to_numeric(d[k], errors="coerce")
        out.append(d.dropna(subset=["lat", "lon", "age"]).reset_index(drop=True))
    return out


def wshare(v, w):
    return float(np.average(v, weights=w)) if len(v) and np.sum(w) > 0 else np.nan


def boot_share(dep, ref, col, n=1000, seed=1):
    """District-resampled deposit share, point-resampled reference share, and their difference."""
    rng = np.random.default_rng(seed)
    dg = dep.groupby("cluster_id")
    keys = np.array(list(dg.groups))
    dsum = dg.apply(lambda g: (g[col] * g["w"]).sum()).reindex(keys).to_numpy()
    dw = dg["w"].sum().reindex(keys).to_numpy()
    rv = ref[col].to_numpy(float)
    d, r = [], []
    for _ in range(n):
        k = rng.integers(0, len(keys), len(keys))
        d.append(dsum[k].sum() / dw[k].sum())
        r.append(rv[rng.integers(0, len(rv), len(rv))].mean())
    d, r = np.array(d), np.array(r)
    return np.percentile(d, [2.5, 97.5]), np.percentile(r, [2.5, 97.5]), np.percentile(d - r, [2.5, 97.5])


def stratified(dep, ref, col, cond, nbins=5):
    """Reference share averaged over quintiles of ``cond`` with the deposits' weights."""
    both = pd.concat([dep[cond], ref[cond]]).dropna()
    if len(both) < 20:
        return np.nan, np.nan, 0.0
    edges = np.unique(np.quantile(both, np.linspace(0, 1, nbins + 1)))
    b = lambda s: np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)  # noqa: E731
    d = dep.dropna(subset=[cond]).assign(bin=lambda f: b(f[cond]))
    r = ref.dropna(subset=[cond]).assign(bin=lambda f: b(f[cond]))
    num_d = num_r = wt = 0.0
    for k, g in d.groupby("bin"):
        rr = r[r["bin"] == k]
        if len(rr) < 5:
            continue
        num_d += (g[col] * g["w"]).sum()
        num_r += rr[col].mean() * g["w"].sum()
        wt += g["w"].sum()
    matched = wt / d["w"].sum() if d["w"].sum() else 0.0
    return (num_d / wt if wt else np.nan), (num_r / wt if wt else np.nan), matched


def auc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], np.sort(b[np.isfinite(b)])
    if not len(a) or not len(b):
        return np.nan
    lo, hi = np.searchsorted(b, a, "left"), np.searchsorted(b, a, "right")
    return float(np.mean((lo + 0.5 * (hi - lo)) / len(b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--spandler-dir", default=str(SPANDLER))
    ap.add_argument("--resamples", type=int, default=1000)
    a = ap.parse_args()
    cfg = load_config(a.config)
    der = cfg["paths"]["derived"]
    sp = Path(a.spandler_dir)
    polys = load_polygons(sp / "Extended Dataset 2")
    rows = []
    tables = spandler_tables(sp)

    # 1. reproduction
    for name, t in zip(("carbonatites", "REE ore deposits"), tables):
        pa = polygon_ages(t["lat"], t["lon"], polys)
        prior = np.array([any(x > age for x in p) for p, age in zip(pa, t["age"])])
        rows.append(dict(kind="reproduction", group=name, n=len(t), share_dep=prior.mean(),
                         note="inside a polygon older than the occurrence"))

    # 2-3. this study's classes against age-matched reference points
    pts = pd.read_csv(der / f"null_mask_{a.tag}.csv")
    f = read_table(der / f"features_{a.tag}")
    f = f[(f["lag_myr"] == 0) & (f["draw"] == 0)][["pt_id", "age_ma", "in_model", "cluster_id", "w_cluster"]]
    pts = pts.merge(f, on="pt_id", how="inner").merge(pd.read_csv(der / f"craton_{a.tag}.csv"), on="pt_id", how="left")
    pts = pts[pts["group"].isin(CLASSES) & ~pts["ice"].astype(bool) & pts["in_model"].astype(bool)].copy()
    pa = polygon_ages(pts["lat0"], pts["lon0"], polys)
    pts["in_any"] = [len(p) > 0 for p in pa]
    pts["primed"] = [any(x > t for x in p) for p, t in zip(pa, pts["age_ma"])]
    pts["primed_le1800"] = [any((x > t) and (x <= 1800) for x in p) for p, t in zip(pa, pts["age_ma"])]
    pts["primed_within800"] = [any((x > t) and (x - t <= 800) for x in p) for p, t in zip(pa, pts["age_ma"])]
    pts["oldest_polygon_ma"] = [max(p) if p else np.nan for p in pa]
    pts["w"] = np.where(pts["kind"] == "deposit", pts["w_cluster"], 1.0)
    for cls in CLASSES:
        dep = pts[(pts["group"] == cls) & (pts["kind"] == "deposit")]
        ref = pts[(pts["group"] == cls) & (pts["kind"] == "reference")]
        for col in ("primed", "primed_le1800", "primed_within800"):
            dci, rci, xci = boot_share(dep, ref, col, a.resamples)
            rows.append(dict(kind="share", group=cls, measure=col, n=dep["cluster_id"].nunique(), n_ref=len(ref),
                             share_dep=wshare(dep[col], dep["w"]), dep_lo=dci[0], dep_hi=dci[1],
                             share_ref=ref[col].mean(), ref_lo=rci[0], ref_hi=rci[1], diff_lo=xci[0], diff_hi=xci[1]))
            for cond, label in CONDITION.items():
                if cond not in pts:
                    continue
                sd, sr, m = stratified(dep, ref, col, cond)
                rows.append(dict(kind="share at equal", group=cls, measure=col, condition=cond, note=label,
                                 share_dep=sd, share_ref=sr, matched=m))

    # 4. their deposits split by host, against the share expected at their ages on continental crust,
    #    over all positions and within quintiles of present-day craton-edge distance
    rng = np.random.default_rng(81)
    refs = pts[pts["kind"] == "reference"].reset_index(drop=True)
    ref_pa = [pa[k] for k in np.flatnonzero((pts["kind"] == "reference").to_numpy())]
    keep = np.flatnonzero(~refs.duplicated(["lat0", "lon0"]).to_numpy())
    ref_ages = [np.array(ref_pa[k], float) for k in keep]
    ref_cd = refs["craton_present_km"].to_numpy(float)[keep]
    edges = np.unique(np.nanquantile(ref_cd, np.linspace(0, 1, 6)))
    cbin = lambda v: np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)  # noqa: E731
    ref_bin = cbin(ref_cd)

    def covered(ages, t, window):
        return bool(((ages > t) & ((ages - t <= window) if window else True)).any())

    def expected(t, window, b=None):
        idx = range(len(ref_ages)) if b is None else np.flatnonzero(ref_bin == b)
        return float(np.mean([covered(ref_ages[k], t, window) for k in idx])) if len(idx) else np.nan

    from rifts.cratons import craton_distance, craton_snapshot, load_cratons
    from rifts.models import load_models
    mantle, _ = load_models(cfg)
    now = craton_snapshot(load_cratons(cfg["paths"]["craton_file"]), mantle.rotation_model, 0.0)
    carb, ree = tables
    ree = ree.assign(host=ree["type"].astype(str).str.strip().str.lower().map(
        lambda v: "carbonatite" if v.startswith("carbonatite") else ("alkaline-silicate" if v.startswith("alkaline")
                                                                    else "volcanic")))
    sets_ = [("carbonatites (all)", carb)] + [(f"REE ore, {h}", g) for h, g in ree.groupby("host")]
    for name, t in sets_:
        pa_t = [np.array(x, float) for x in polygon_ages(t["lat"], t["lon"], polys)]
        ages = t["age"].to_numpy(float)
        cd = craton_distance(now, t["lon"].to_numpy(float), t["lat"].to_numpy(float))
        bins = cbin(cd)
        for label, window in (("primed", None), ("primed_within400", 400.0), ("primed_within800", 800.0),
                              ("primed_within1200", 1200.0)):
            obs = np.array([covered(x, a, window) for x, a in zip(pa_t, ages)])
            for cond, exp in (("", np.array([expected(a, window) for a in ages])),
                              ("craton_present_km", np.array([expected(a, window, b) for a, b in zip(ages, bins)]))):
                ok = np.isfinite(exp)
                o, e = obs[ok], exp[ok]
                b = [(o[k].mean() - e[k].mean()) for k in (rng.integers(0, len(o), len(o)) for _ in range(a.resamples))]
                rows.append(dict(kind="spandler by host", group=name, measure=label, condition=cond or None, n=int(ok.sum()),
                                 share_dep=o.mean(), share_ref=e.mean(), diff_lo=np.percentile(b, 2.5),
                                 diff_hi=np.percentile(b, 97.5),
                                 note=f"median age {np.median(ages):.0f} Ma; median craton-edge distance "
                                      f"{np.nanmedian(cd):.0f} km; expected = reference positions covered by a "
                                      "polygon of the deposit's age" + (", within the deposit's quintile of present-day "
                                                                        "craton-edge distance" if cond else "")))

    # 5. does a polygon mark more subducted volatiles?
    vol = []
    for name in (f"water_{a.tag}", f"primer_history_{a.tag}"):
        try:
            vol.append(read_table(der / name))
        except FileNotFoundError:
            pass
    if vol:
        v = vol[0]
        for extra in vol[1:]:
            v = v.merge(extra, on="pt_id", how="outer")
        ref = pts[pts["kind"] == "reference"].merge(v, on="pt_id", how="left")
        meas = [c for c in v.columns if c != "pt_id" and pd.api.types.is_numeric_dtype(v[c])]
        # "primed" requires a polygon older than the point, so young reference points (the Cenozoic-heavy
        # porphyry set) are primed more often than old ones, and the pooled comparison mixes the age
        # distributions. The comparison is therefore repeated within deciles of the reference ages,
        # pooling the within-decile ranks (a stratified AUC, each decile weighted by its inside points),
        # and the pooled version is kept for comparison (review 2026-10-18).
        dec = pd.qcut(ref["age_ma"].rank(method="first"), 10, labels=False)
        for flag in ("primed", "primed_within800"):
            for c in meas:
                inside, outside = ref.loc[ref[flag], c], ref.loc[~ref[flag], c]
                rows.append(dict(kind="polygon vs volatiles", group="all reference points", measure=c, condition=flag,
                                 n=int(inside.notna().sum()), n_ref=int(outside.notna().sum()),
                                 auc=auc(inside, outside), median_in=inside.median(), median_out=outside.median()))
                num, wsum, n_in, n_out = 0.0, 0.0, 0, 0
                for k in range(10):
                    m = dec == k
                    i_k, o_k = ref.loc[m & ref[flag], c].dropna(), ref.loc[m & ~ref[flag], c].dropna()
                    if len(i_k) and len(o_k):
                        num += auc(i_k, o_k) * len(i_k)
                        wsum += len(i_k)
                        n_in += len(i_k)
                        n_out += len(o_k)
                rows.append(dict(kind="polygon vs volatiles", group="all reference points, within age deciles", measure=c,
                                 condition=flag, n=n_in, n_ref=n_out, auc=num / wsum if wsum else np.nan,
                                 median_in=inside.median(), median_out=outside.median()))
    out = der / f"spandler_reassessment_{a.tag}.csv"
    tmp = out.with_name(out.name + ".partial")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(out)
    print(f"wrote {out}")
    print(pd.DataFrame(rows).query("kind != 'polygon vs volatiles'").round(3).to_string())
    pv = pd.DataFrame(rows).query("kind == 'polygon vs volatiles'")
    key = ["out_total_path_full", "sub_total_path_full", "c_mantle_path_full", "out_sed_path_full"]
    print(pv[pv["measure"].isin(key)][["group", "measure", "condition", "n", "n_ref", "auc"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
