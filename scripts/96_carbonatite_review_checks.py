#!/usr/bin/env python
"""Checks of the carbonatite and alkaline-silicate REE results asked for by the review of 9 November 2026.

Four parts, each written as rows of one long table:

``episodes`` The subduction-episode measures of scripts/82 under other definitions. For alkaline-silicate
    REE districts against carbonatite districts reweighted to their ages (scripts/84 ``direct``), and for the
    post-collisional and post-subduction districts alone: the time since the most recent episode on an
    overriding plate within 700, 900 or 1,100 km of a trench lasting at least 100 Myr, and within 900 km
    lasting at least 50 or 150 Myr; the share of districts with such an episode; and, among districts with
    a 900 km, 100 Myr episode, how many lay within 900 km of an active trench at the first reconstructed step
    after formation, and for how long they stayed there (``over_900km_after_myr``). A lag of one 5 Myr step
    can mark an episode that ended just before formation or one still under way at it; the second measure
    tells them apart.

``domains`` The share of districts inside a subduction-fertilised domain of Spandler et al. (2026) against
    the share expected when reference positions are matched on present-day craton-edge distance, the
    shear-wave-speed gradient at 150 km and exploration density together (joint tertiles, 27 cells), rather
    than on one variable at a time as in scripts/81, with 95% intervals from resampling districts and
    reference points. Domains formed at any time before the site, and any domain formed within 800 Myr before
    it (the definition of scripts/81, read from the polygons).

``skill`` Paired differences in held-out skill between the random-forest models of scripts/84, repetition
    by repetition on identical folds: without the subducted-volatile (source) measures, without the
    lithospheric measures and with all measures, each against the model without exposure; and exposure
    alone against the model without exposure. Mean, smallest and largest difference over the repetitions,
    and the number of repetitions in which it is positive.

``isotopes`` The rank correlations of Supplementary Table 5 (isotope measure against subduction measure,
    all samples), oriented so that higher means more enriched, with analytic p-values and Benjamini–Hochberg
    q-values over the family of correlations in that table; and the correlation of the εNd departure with the
    sediment-bound slab water using only districts that received some (no ties at zero delivery).

Inputs: data/derived/carbonatite_features_<tag> (scripts/83, with the scripts/82 columns),
carbonatite_systems_<tag>.csv (scripts/84), carbonatite_isotope_districts_<tag>.csv (scripts/86),
config/aireee_settings.csv. Output: ONE table, data/derived/carbonatite_review_checks_<tag>.csv, through a
.partial name.

    python scripts/96_carbonatite_review_checks.py --tag carb26
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import ROOT, load_config, read_table  # noqa: E402

ISO_ORIENT = {"d_eNd": -1, "eNd_i": -1, "d_sr": 1, "sr_i": 1, "pb206_204": 1, "la": 1}   # as scripts/86
ISO_Q = ["out_total_path_full", "out_sed_path_full", "sub_total_path_full", "c_mantle_path_full",
         "out_total_path_400myr", "over_900km_myr", "craton_dist_km", "vsh_grad150"]          # Supplementary Table 5
EPISODES = [("over_700km_since_myr", 700, 100), ("over_900km_since_myr", 900, 100), ("over_1100km_since_myr", 1100, 100),
            ("over_900km_since_L50myr", 900, 50), ("over_900km_since_L150myr", 900, 150)]


def load84():
    """scripts/84 as a module, for its comparison functions (direct, district_flags, aireee_setting)."""
    spec = importlib.util.spec_from_file_location("s84", Path(__file__).with_name("84_carbonatite_systems.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def analysed(p):
    """The records compared in scripts/84: not beneath present-day ice, on a block the model holds."""
    return p[~p["ice"].fillna(False).astype(bool) & p["in_model"].fillna(True).astype(bool)]


def district_median(d, col):
    """One value per district: the median over its records (records of a district share a history)."""
    return d.groupby("cluster_id")[col].median()


def part_episodes(p, s84, boot, rng):
    """Alkaline-silicate REE districts (all, and post-collisional) against age-matched carbonatites on the
    episode measures under other radii and durations, and the persistence of the setting after formation."""
    rows = []
    carb = p[(p["group"] == "CARB") & (p["kind"] == "deposit")]
    alk = p[(p["group"] == "AIREE") & (p["kind"] == "deposit")]
    setting = s84.aireee_setting(alk, ROOT)
    subsets = [("AIREE", alk), ("AIREE post-collisional", alk[(setting == "post-collisional").to_numpy()])]
    for label, a in subsets:
        for col, km, myr in EPISODES:
            if col not in p:
                print(f"  {col} not in the features table: run scripts/82 --parts topo, then scripts/83")
                continue
            flag = col.replace("since", "episode") if "_L" in col else col.replace("since_myr", "episode")
            r = s84.direct(a, carb, col, boot, rng, age_match=True)
            share_a = district_median(a, flag).mean() if flag in a else np.nan
            share_c = district_median(carb, flag).mean() if flag in carb else np.nan
            rows.append(dict(part="episodes", set=f"{label} vs CARB, age-matched", measure=col, radius_km=km,
                             min_episode_myr=myr, n=r["n_districts"], n_ref=r["n_ref"], value=r["auc"],
                             lo=r.get("lo"), hi=r.get("hi"), p=r.get("p"), median_dep=r["median_dep"],
                             median_ref=r["median_ref"], share_dep=share_a, share_ref=share_c))
        # persistence after formation, among districts with a 900 km, 100 Myr episode
        for cls, d in ((label, a), ("CARB", carb)):
            if "over_900km_after_myr" not in d:
                continue
            since = district_median(d, "over_900km_since_myr")
            aft = district_median(d, "over_900km_after_myr")
            ep = since.dropna().index
            if not len(ep):
                continue
            rows.append(dict(part="persistence", set=cls if cls != "CARB" else f"CARB (with {label})", measure="over_900km_after_myr",
                             radius_km=900, min_episode_myr=100, n=len(ep),
                             share_dep=float((since[ep] <= 5.0).mean()),           # lag of one 5 Myr step
                             value=float((aft[ep] > 0).mean()),                     # still within 900 km after formation
                             median_dep=float(aft[ep][aft[ep] > 0].median()) if (aft[ep] > 0).any() else np.nan,
                             note="share: lag of one step; value: within 900 km at the first step after formation; "
                                  "median: Myr it stayed there, among those"))
    return rows


def joint_expected(dep, ref, col, conds, nbins=3):
    """Reference share averaged over joint cells of ``conds`` (tertiles of each) with the deposits' weights;
    cells with fewer than five reference points are left out. Returns the deposit share in the matched cells,
    the expected share and the share of deposit weight matched."""
    both = pd.concat([dep[conds], ref[conds]]).dropna()
    edges = {c: np.unique(np.quantile(both[c], np.linspace(0, 1, nbins + 1))) for c in conds}

    def cell(f):
        f = f.dropna(subset=conds)
        k = np.zeros(len(f), int)
        for c in conds:
            e = edges[c]
            k = k * nbins + np.clip(np.searchsorted(e, f[c].to_numpy(), side="right") - 1, 0, len(e) - 2)
        return f.assign(_cell=k)
    d, r = cell(dep), cell(ref)
    rs = r.groupby("_cell").agg(share=(col, "mean"), n=(col, "size"))
    rs = rs[rs["n"] >= 5]
    m = d[d["_cell"].isin(rs.index)]
    w = m["w"].to_numpy(float)
    if not len(m) or w.sum() == 0:
        return np.nan, np.nan, 0.0
    obs = float(np.average(m[col].astype(float), weights=w))
    exp = float(np.average(rs.loc[m["_cell"], "share"].to_numpy(), weights=w))
    return obs, exp, float(w.sum() / d["w"].sum()) if len(d) else 0.0


def any_domain_within(p, window=800.0):
    """Whether any domain of Spandler et al. (2026) at the present-day position of each row formed within
    ``window`` Myr before its age, the definition of scripts/81 and Supplementary Table 4. The feature column
    fml_age_gap_myr of scripts/83 holds the age of the OLDEST domain, which gives a smaller share (a site
    inside an old and a young domain counts as old), so the polygons are read again here."""
    spec = importlib.util.spec_from_file_location("sp81", Path(__file__).with_name("81_spandler_reassessment.py"))
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    polys = sp.load_polygons(sp.SPANDLER / "Extended Dataset 2")
    key = pd.DataFrame({"lat": p["lat0"].round(4).to_numpy(), "lon": p["lon0"].round(4).to_numpy()})
    uniq = key.drop_duplicates().reset_index(drop=True)
    ages = sp.polygon_ages(uniq["lat"], uniq["lon"], polys)
    look = {(la, lo): a for la, lo, a in zip(uniq["lat"], uniq["lon"], ages)}
    t = p["age_ma"].to_numpy(float)
    return np.array([any((x > ti) and (x - ti <= window) for x in look[(la, lo)])
                     for la, lo, ti in zip(key["lat"], key["lon"], t)], float)


def part_domains(p, boot, rng):
    """Observed share inside a fertilised domain against the share expected under joint matching."""
    rows = []
    p = p[p["group"].isin(["CARB", "AIREE", "KIMB", "PCU"])]
    p = p.assign(primed=p["fml_primed"].astype(float),
                 primed_within800=any_domain_within(p, 800.0),
                 w=np.where(p["kind"] == "deposit", p["w_cluster"], 1.0))
    conds = ["craton_present_km", "vsh_grad150", "other_family_km"]
    for cls in ("CARB", "AIREE", "KIMB", "PCU"):
        dep = p[(p["group"] == cls) & (p["kind"] == "deposit")]
        ref = p[(p["group"] == cls) & (p["kind"] == "reference")]
        for col in ("primed", "primed_within800"):
            obs, exp, matched = joint_expected(dep, ref, col, conds)
            g = list(dep.groupby("cluster_id").indices.values())
            diffs = []
            for _ in range(boot):
                dd = dep.iloc[np.concatenate([g[k] for k in rng.integers(0, len(g), len(g))])]
                rr = ref.iloc[rng.integers(0, len(ref), len(ref))]
                o, e, _ = joint_expected(dd, rr, col, conds)
                diffs.append(o - e)
            lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
            rows.append(dict(part="domains", set=cls, measure=col, n=dep["cluster_id"].nunique(), n_ref=len(ref),
                             share_dep=float(np.average(dep[col], weights=dep["w"])), share_ref=float(ref[col].mean()),
                             value=obs - exp, lo=lo, hi=hi, median_dep=obs, median_ref=exp, matched=matched,
                             note="value: observed minus expected share in the jointly matched cells (craton-edge distance, "
                                  "shear-wave gradient at 150 km, exploration density; tertiles of each)"))
    return rows


def part_skill(sysdf):
    """Paired differences in held-out skill between models, repetition by repetition on identical folds."""
    rows = []
    sk = sysdf[sysdf["part"] == "skill"].dropna(subset=["skills"])
    pairs = [("without exposure and source", "without exposure", "volatile (source) measures removed"),
             ("without exposure and lithosphere", "without exposure", "lithospheric measures removed"),
             ("without exposure and trigger", "without exposure", "trigger measures removed"),
             ("all features", "without exposure", "exposure measures added"),
             ("exposure alone", "without exposure", "exposure alone against the geological measures")]
    for s_, g in sk.groupby("set"):
        reps = {r["model"]: np.array([float(v) for v in str(r["skills"]).split(";")]) for _, r in g.iterrows()}
        for a, b, lab in pairs:
            if a in reps and b in reps and len(reps[a]) == len(reps[b]):
                d = reps[a] - reps[b]
                rows.append(dict(part="skill", set=s_, measure=f"{a} minus {b}", n=len(d), value=float(d.mean()),
                                 lo=float(d.min()), hi=float(d.max()), share_dep=float((d > 0).mean()),
                                 median_dep=float(reps[a].mean()), median_ref=float(reps[b].mean()), note=lab))
    return rows


def bh(p):
    """Benjamini–Hochberg q-values."""
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    o = np.argsort(p[ok])
    m = ok.sum()
    adj = np.minimum.accumulate((p[ok][o] * m / np.arange(1, m + 1))[::-1])[::-1]
    qq = np.empty(m)
    qq[o] = np.minimum(adj, 1.0)
    q[ok] = qq
    return q


def part_isotopes(iso):
    """Analytic p and BH q for the correlations of Supplementary Table 5, and the zero-free εNd correlation."""
    from scipy.stats import spearmanr
    rows = []
    d = iso[iso["variant"] == "all samples"]
    for cls, g in d.groupby("cls"):
        for m, sgn in ISO_ORIENT.items():
            for q in ISO_Q:
                x = g[[m, q]].dropna()
                if len(x) < 5 or x[q].nunique() < 2 or x[m].nunique() < 2:
                    continue
                rho, pv = spearmanr(x[m] * sgn, x[q])
                rows.append(dict(part="isotopes", set=cls, measure=f"{m} vs {q}", n=len(x), value=float(rho), p=float(pv)))
    t = pd.DataFrame(rows)
    if len(t):
        t["q"] = np.nan
        for cls, g in t.groupby("set"):
            t.loc[g.index, "q"] = bh(g["p"])
    rows = t.to_dict("records")
    c = d[(d["cls"] == "CARB")][["d_eNd", "out_sed_path_full"]].dropna()
    nz = c[c["out_sed_path_full"] > 0]
    if len(nz) >= 5:
        rho, pv = spearmanr(nz["d_eNd"] * ISO_ORIENT["d_eNd"], nz["out_sed_path_full"])
        rows.append(dict(part="isotopes", set="CARB", measure="d_eNd vs out_sed_path_full, delivery above zero",
                         n=len(nz), value=float(rho), p=float(pv), share_dep=float(len(nz) / len(c)),
                         note="share: districts with delivery above zero among those with εNd"))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--tag", default="carb26")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--parts", nargs="*", default=["episodes", "domains", "skill", "isotopes"],
                    choices=["episodes", "domains", "skill", "isotopes"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    der = Path(cfg["paths"]["derived"])
    s84 = load84()
    p = read_table(der / f"carbonatite_features_{a.tag}")
    p = analysed(s84.district_flags(p))
    rng = np.random.default_rng(96)
    rows = []
    if "episodes" in a.parts:
        rows += part_episodes(p, s84, a.boot, rng)
        print("  episodes: done", flush=True)
    if "domains" in a.parts:
        rows += part_domains(p, min(a.boot, 500), rng)
        print("  domains: done", flush=True)
    if "skill" in a.parts:
        rows += part_skill(pd.read_csv(der / f"carbonatite_systems_{a.tag}.csv", low_memory=False))
        print("  skill: done", flush=True)
    if "isotopes" in a.parts:
        rows += part_isotopes(pd.read_csv(der / f"carbonatite_isotope_districts_{a.tag}.csv"))
        print("  isotopes: done", flush=True)
    out = der / f"carbonatite_review_checks_{a.tag}.csv"
    t = pd.DataFrame(rows)
    if set(a.parts) != {"episodes", "domains", "skill", "isotopes"} and out.exists():
        old = pd.read_csv(out)
        t = pd.concat([old[~old["part"].isin(t["part"].unique() if len(t) else [])], t], ignore_index=True)
    tmp = out.with_name(out.name + ".partial")
    t.to_csv(tmp, index=False)
    tmp.replace(out)
    print(f"wrote {out}")
    show = t[t["part"].isin(["episodes", "persistence", "domains", "skill"])]
    print(show[["part", "set", "measure", "n", "value", "lo", "hi", "share_dep", "median_dep", "median_ref"]].round(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
