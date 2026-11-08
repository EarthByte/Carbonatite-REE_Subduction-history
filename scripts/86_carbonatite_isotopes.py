#!/usr/bin/env python
"""Sr-Nd-Pb isotopes of carbonatites and alkaline-silicate rocks against the subduction history of
their lithosphere.

The recycling hypotheses (Hou et al. 2015; Spandler et al. 2026; Chen et al. 2026) hold that the
rare earths and carbon of carbonatites come from subducted sediment, carbonate and altered crust
stored in the lithosphere. Their geochemical prediction is that a carbonatite whose source received
more subducted material, or received it more recently, carries a more enriched signature: lower
initial epsilon-Nd and higher initial 87Sr/86Sr than the depleted mantle of its age. This step
tests that prediction against the slab input delivered beneath each district along its plate path
(scripts/82 and 83), the time on an overriding plate, and the fertilised domains of Spandler et al.

Samples: the GEOROC precompiled rock-type files (DIGIS Team 2026, doi 10.25625/2JETOA, CC BY-SA 4.0)
for carbonatite and phoscorite (carbonatite districts, CARB) and syenite (alkaline-silicate REE
districts, AIREE), whole rock and minerals. A sample joins the district of the nearest deposit of
the class within ``MATCH_KM`` (present-day positions) and takes that deposit's emplacement age.

Initial ratios: GEOROC initial values where reported; otherwise from the measured ratio and the
parent/daughter concentrations of the same sample at the district age (87Rb/86Sr = 2.8936 Rb/Sr,
lambda 87Rb 1.393e-11 /yr; 147Sm/144Nd = 0.60485 Sm/Nd, lambda 147Sm 6.54e-12 /yr; CHUR
143Nd/144Nd 0.512630 and 147Sm/144Nd 0.1960, Bouvier et al. 2008). Carbonatite Sr with no Rb is
taken as initial where Sr exceeds 1,000 ppm (Rb/Sr < 0.001 in carbonatite, so the correction is
below the analytical error for any age considered); otherwise the sample is dropped. Measures:

  eNd_i        initial epsilon-Nd
  d_eNd        eNd_i minus the depleted mantle at that age (DePaolo 1981: 0.25t^2 - 3t + 8.5, t in Ga);
               negative = enriched relative to the convecting mantle of the time
  sr_i         initial 87Sr/86Sr
  pb206_204    206Pb/204Pb: the GEOROC initial value where reported, otherwise the measured ratio
               (no age correction from U and Th is made)
  la           La (ppm), the REE enrichment that Chen et al. (2026) relate to deposit size
  d_sr         sr_i minus the depleted mantle at that age (0.70250 today, 87Rb/86Sr 0.046)

District values are medians of their samples. Tests, each with 1,000 resamples of districts:

  dose         Spearman correlation between each isotopic measure and each subduction measure across
               districts, and the same with the age of the district partialled out (ranks)
  domain       probability that a district inside a fertilised domain (formed before it, and within
               800 Myr before it) is more enriched than one outside
  recycling    the same split at 300 Ma (Banerjee et al. 2021 found recycled carbon only in younger
               carbonatites), and ore-bearing against barren districts
  classes      CARB against AIREE on every isotopic measure

Every test is repeated on samples rich enough in Sr and Nd that assimilated crust moves their isotopes
little (variant "high Sr and Nd"); correlations carry intervals from resampling districts and from
resampling spatial blocks (12 k-means blocks), the range when each block is left out, and partial
correlations with the age, and with the age, craton-edge distance and Vs at 150 km, partialled out.

Output: ONE table, data/derived/carbonatite_isotopes_<tag>.csv (tests), and ONE table of district
values, data/derived/carbonatite_isotope_districts_<tag>.csv, through .partial names.

    python scripts/86_carbonatite_isotopes.py --tag full
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import EARTH_RADIUS_KM, load_config, lonlat_to_xyz, read_table  # noqa: E402

GEOROC = Path("data") / "raw" / "georoc_2026-09"
FILES = {"CARB": ["2026-09-2JETOA_CARBONATITE.csv", "2026-09-2JETOA_PHOSCORITE.csv"],
         "AIREE": ["2026-09-2JETOA_SYENITE.csv"]}
MATCH_KM = 25.0
L_RB, L_SM = 1.393e-11, 6.54e-12
CHUR_ND, CHUR_SM = 0.512630, 0.1960
DM_SR0, DM_RBSR = 0.70250, 0.046
DOSE = ["out_total_path_full", "out_sed_path_full", "out_crust_path_full", "out_mantle_path_full",
        "sub_total_path_full", "c_mantle_path_full", "c300_path_full", "out_total_path_400myr",
        "out_total_path_800myr", "out_total_path_lag_myr", "over_900km_myr", "over_900km_since_myr",
        "over_900km_stable_myr", "fml_primed", "fml_age_gap_myr", "craton_dist_km", "vsh_grad150"]
MEASURES = ["d_eNd", "eNd_i", "d_sr", "sr_i", "pb206_204", "la"]
#: the direction of enrichment for each measure: +1 when a higher value is more enriched
ENRICHED = {"d_eNd": -1, "eNd_i": -1, "d_sr": 1, "sr_i": 1, "pb206_204": 1, "la": 1}


def read_georoc(root, files):
    """The GEOROC precompiled files ``files`` as one table per sample: position (centre of the reported
    range), material, La, Rb, Sr, Sm and Nd contents, measured and initial Sr and Nd ratios, epsilon-Nd
    and Pb-206/204 (initial where reported, otherwise measured); samples without a position are dropped."""
    out = []
    for f in files:
        p = root / GEOROC / f
        if not p.exists():
            print(f"  {p} not found")
            continue
        d = pd.read_csv(p, encoding="latin-1", low_memory=False)
        num = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d else np.nan  # noqa: E731
        out.append(pd.DataFrame({
            "file": f, "sample": d["SAMPLE NAME"], "location": d["LOCATION"], "citation": d["CITATIONS"],
            "material": d["MATERIAL"].astype(str).str.split().str[0],
            "lat": (num("LATITUDE MIN") + num("LATITUDE MAX")) / 2, "lon": (num("LONGITUDE MIN") + num("LONGITUDE MAX")) / 2,
            "la": num("LA(PPM)"), "rb": num("RB(PPM)"), "sr": num("SR(PPM)"), "sm": num("SM(PPM)"), "nd": num("ND(PPM)"),
            "sr_m": num("SR87_SR86"), "sr_ini": num("SR87_SR86_INI"), "nd_m": num("ND143_ND144"),
            "nd_ini": num("ND143_ND144_INI"), "eps_nd": num("EPSILON_ND"), "pb206_204": num("PB206_PB204_INI")
            .fillna(num("PB206_PB204"))}))
    t = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return t[np.isfinite(t["lat"]) & np.isfinite(t["lon"])].reset_index(drop=True) if len(t) else t


def match(samples, deposits):
    """District id and age of the nearest deposit within MATCH_KM of each sample."""
    from scipy.spatial import cKDTree
    tree = cKDTree(lonlat_to_xyz(deposits["lon0"].to_numpy(float), deposits["lat0"].to_numpy(float)))
    d, k = tree.query(lonlat_to_xyz(samples["lon"].to_numpy(float), samples["lat"].to_numpy(float)))
    km = 2 * np.arcsin(np.clip(d / 2, 0, 1)) * EARTH_RADIUS_KM
    ok = km <= MATCH_KM
    s = samples[ok].copy()
    dep = deposits.iloc[k[ok]].reset_index(drop=True)
    s["pt_id"], s["cluster_id"], s["age_ma"], s["match_km"] = (dep["pt_id"].to_numpy(), dep["cluster_id"].to_numpy(),
                                                               dep["age_ma"].to_numpy(float), km[ok])
    return s.reset_index(drop=True)


def initial_ratios(s, cls):
    """Initial 87Sr/86Sr and epsilon-Nd at the district age (reported initial values first, otherwise
    age-corrected from the parent/daughter contents of the sample; carbonatite Sr without Rb taken as
    initial above 1,000 ppm Sr), their departures from the depleted mantle of that age (d_sr, d_eNd),
    and implausible values set missing."""
    t = s["age_ma"].to_numpy(float) * 1e6
    g_rb, g_sm = np.expm1(L_RB * t), np.expm1(L_SM * t)
    rbsr = 2.8936 * s["rb"] / s["sr"]
    sr_calc = s["sr_m"] - rbsr * g_rb
    if cls == "CARB":
        # carbonatite: negligible Rb where Sr is high
        sr_calc = sr_calc.where(rbsr.notna(), s["sr_m"].where(s["sr"] > 1000))
    s["sr_i"] = s["sr_ini"].fillna(sr_calc)
    smnd = 0.60485 * s["sm"] / s["nd"]
    nd_i = s["nd_ini"].fillna(s["nd_m"] - smnd * g_sm)
    chur_t = CHUR_ND - CHUR_SM * g_sm
    s["eNd_i"] = ((nd_i / chur_t - 1) * 1e4)
    # a reported epsilon-Nd is taken only where no ratio is given (its age convention is not stated)
    s["eNd_i"] = s["eNd_i"].fillna(s["eps_nd"].where(s["nd_m"].isna()))
    tg = s["age_ma"].to_numpy(float) / 1e3
    s["d_eNd"] = s["eNd_i"] - (0.25 * tg ** 2 - 3 * tg + 8.5)
    s["d_sr"] = s["sr_i"] - (DM_SR0 - DM_RBSR * g_rb)
    # implausible values (mixed-up columns, typos) are dropped rather than trimmed
    s.loc[(s["sr_i"] < 0.700) | (s["sr_i"] > 0.730), ["sr_i", "d_sr"]] = np.nan
    s.loc[s["eNd_i"].abs() > 40, ["eNd_i", "d_eNd"]] = np.nan
    return s


def districts(s, feats):
    """District medians of the isotopic measures and the number of samples, age and position of each
    district, joined to the features of its matched deposit (``feats``)."""
    g = s.groupby("cluster_id")
    d = g[MEASURES].median()
    d["n_samples"] = g.size()
    d["age_ma"] = g["age_ma"].median()
    d["pt_id"] = g["pt_id"].first()
    d["lon0"], d["lat0"] = g["lon"].median(), g["lat"].median()
    d = d.reset_index()
    return d.merge(feats, on="pt_id", how="left", suffixes=("", "_f"))


def spearman(x, y):
    from scipy.stats import rankdata
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8 or np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1]), int(ok.sum())


def partial_spearman(x, y, Z):
    """Rank correlation of x and y with the columns of Z partialled out (residuals of ranks on the
    ranks of Z by least squares)."""
    from scipy.stats import rankdata
    Z = np.atleast_2d(np.asarray(Z, float).T).T if np.ndim(Z) == 1 else np.asarray(Z, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(axis=1)
    if ok.sum() < 10:
        return np.nan
    R = np.column_stack([np.ones(ok.sum())] + [rankdata(z) for z in Z[ok].T])
    res = lambda r: r - R @ np.linalg.lstsq(R, r, rcond=None)[0]  # noqa: E731
    a, b = res(rankdata(x[ok])), res(rankdata(y[ok]))
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan


def blocks(lon, lat, n=12, seed=86):
    """Spatial blocks (k-means on position) so that resampling keeps neighbouring districts together."""
    from sklearn.cluster import KMeans
    lo, la = np.radians(lon), np.radians(lat)
    xyz = np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    return KMeans(min(n, len(lon)), n_init=4, random_state=seed).fit_predict(xyz)


def auc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], np.sort(b[np.isfinite(b)])
    if not len(a) or not len(b):
        return np.nan
    lo, hi = np.searchsorted(b, a, "left"), np.searchsorted(b, a, "right")
    return float(np.mean((lo + 0.5 * (hi - lo)) / len(b)))


def run_tests(dist, cls, nboot, rng, variant):
    """Tests of one class and variant: for every isotopic measure (signed so that higher is more enriched)
    and every subduction or lithosphere quantity of DOSE, the Spearman correlation with intervals from
    resampling districts and spatial blocks, the leave-one-block-out range and partial correlations
    (age; age, craton-edge distance and Vs at 150 km); and the probability that districts inside each
    split (fertilised domain, younger than 300 Ma, ore-bearing) are more enriched than the rest, with an
    interval from resampling districts."""
    rows = []
    n = len(dist)
    blk = blocks(dist["lon0"].to_numpy(float), dist["lat0"].to_numpy(float))
    groups = [np.flatnonzero(blk == k) for k in np.unique(blk)]
    lith = dist[["age_ma", "craton_dist_km", "vsh150"]].to_numpy(float)
    for m in MEASURES:
        y = dist[m].to_numpy(float) * ENRICHED[m]           # higher = more enriched
        if np.isfinite(y).sum() < 8:
            continue
        for q in DOSE:
            if q not in dist:
                continue
            x = dist[q].to_numpy(float)
            r, k = spearman(x, y)
            rp = partial_spearman(x, y, dist["age_ma"].to_numpy(float))
            rl = partial_spearman(x, y, lith)
            b, bb = [], []
            for _ in range(nboot):
                i = rng.integers(0, n, n)
                b.append(spearman(x[i], y[i])[0])
                j = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
                bb.append(spearman(x[j], y[j])[0])
            b, bb = np.array(b, float), np.array(bb, float)
            b, bb = b[np.isfinite(b)], bb[np.isfinite(bb)]
            loo = [spearman(x[blk != g], y[blk != g])[0] for g in np.unique(blk)]
            rows.append(dict(part="dose", cls=cls, variant=variant, measure=m, quantity=q, n_districts=k, rho=r,
                             rho_age_partialled=rp, rho_age_lithosphere_partialled=rl,
                             lo=np.percentile(b, 2.5) if len(b) else np.nan, hi=np.percentile(b, 97.5) if len(b) else np.nan,
                             block_lo=np.percentile(bb, 2.5) if len(bb) else np.nan,
                             block_hi=np.percentile(bb, 97.5) if len(bb) else np.nan,
                             leave_block_out_min=np.nanmin(loo) if np.isfinite(loo).any() else np.nan,
                             leave_block_out_max=np.nanmax(loo) if np.isfinite(loo).any() else np.nan))
        for name, mask in (("inside a fertilised domain formed before it", dist["fml_primed"] == 1),
                           ("inside a domain formed within 800 Myr before it", dist["fml_age_gap_myr"] <= 800),
                           ("younger than 300 Ma", dist["age_ma"] < 300),
                           ("ore-bearing (REE or Nb ore within 10 km)", dist["ore"] == 1)):
            a, c = y[mask.to_numpy()], y[~mask.to_numpy()]
            b = []
            for _ in range(nboot):
                i = rng.integers(0, n, n)
                mm = mask.to_numpy()[i]
                b.append(auc(y[i][mm], y[i][~mm]))
            rows.append(dict(part="split", cls=cls, variant=variant, measure=m, quantity=name, n_districts=int(np.isfinite(a).sum()),
                             n_ref=int(np.isfinite(c).sum()), auc=auc(a, c), lo=np.nanpercentile(b, 2.5),
                             hi=np.nanpercentile(b, 97.5), median_dep=np.nanmedian(a * ENRICHED[m]) if np.isfinite(a).any() else np.nan,
                             median_ref=np.nanmedian(c * ENRICHED[m]) if np.isfinite(c).any() else np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    cfg = load_config(a.config)
    der = cfg["paths"]["derived"]
    root = Path(__file__).resolve().parents[1]
    feats = read_table(der / f"carbonatite_features_{a.tag}")
    rng = np.random.default_rng(86)
    rows, allsamp, alldist = [], [], []
    for cls, files in FILES.items():
        dep = feats[(feats["group"] == cls) & (feats["kind"] == "deposit")].reset_index(drop=True)
        s = read_georoc(root, files)
        if not len(s):
            continue
        # samples within MATCH_KM of a deposit of the class take its district and age
        s = initial_ratios(match(s, dep), cls)
        s = s[s[["sr_i", "eNd_i", "pb206_204", "la"]].notna().any(axis=1)]
        cols = ["pt_id"] + [c for c in DOSE + ["vsh150", "ore"] if c in feats and c != "pt_id"]
        cols = list(dict.fromkeys(cols))
        # variant "robust": samples whose Sr and Nd are high enough that crustal assimilation moves
        # their isotopes little (carbonatite Sr > 1,000 ppm, Nd > 100 ppm; syenite Sr > 500, Nd > 50)
        hi_sr, hi_nd = (1000, 100) if cls == "CARB" else (500, 50)
        robust = s.assign(sr_i=s["sr_i"].where(s["sr"] > hi_sr), d_sr=s["d_sr"].where(s["sr"] > hi_sr),
                          eNd_i=s["eNd_i"].where(s["nd"] > hi_nd), d_eNd=s["d_eNd"].where(s["nd"] > hi_nd))
        for variant, ss in (("all samples", s), ("high Sr and Nd", robust)):
            d = districts(ss, feats[cols].drop_duplicates("pt_id"))
            d["cls"], d["variant"] = cls, variant
            print(f"  {cls}, {variant}: {len(ss)} samples in {d['cluster_id'].nunique()} districts "
                  f"(eNd {d['eNd_i'].notna().sum()}, Sr {d['sr_i'].notna().sum()}, Pb {d['pb206_204'].notna().sum()})",
                  flush=True)
            rows += run_tests(d, cls, a.boot, rng, variant)
            alldist.append(d)
        allsamp.append(s.assign(cls=cls))
    dist = pd.concat(alldist, ignore_index=True)
    # the two classes against each other, all samples
    base = dist[dist["variant"] == "all samples"]
    carb, alk = base[base["cls"] == "CARB"], base[base["cls"] == "AIREE"]
    for m in MEASURES:
        x, y = carb[m].to_numpy(float), alk[m].to_numpy(float)
        rows.append(dict(part="classes", cls="AIREE vs CARB", measure=m, n_districts=int(np.isfinite(y).sum()),
                         n_ref=int(np.isfinite(x).sum()), auc=auc(y, x), median_dep=np.nanmedian(y), median_ref=np.nanmedian(x)))
    for name, t in (("carbonatite_isotopes", pd.DataFrame(rows)), ("carbonatite_isotope_districts", dist)):
        out = der / f"{name}_{a.tag}.csv"
        tmp = out.with_name(out.name + ".partial")
        t.to_csv(tmp, index=False)
        tmp.replace(out)
        print(f"wrote {out}")
    t = pd.DataFrame(rows)
    show = t[(t["part"] == "dose") & t["measure"].isin(["d_eNd", "d_sr"]) &
             t["quantity"].isin(["out_total_path_full", "out_sed_path_full", "sub_total_path_full", "c_mantle_path_full",
                                 "out_total_path_800myr", "over_900km_since_myr", "fml_age_gap_myr", "craton_dist_km"])]
    print(show[["cls", "variant", "measure", "quantity", "n_districts", "rho", "lo", "hi", "block_lo", "block_hi",
                "rho_age_partialled", "rho_age_lithosphere_partialled", "leave_block_out_min", "leave_block_out_max"]]
          .round(2).to_string())
    print(t[t["part"].isin(["split", "classes"]) & t["measure"].isin(["d_eNd", "d_sr"])]
          [["part", "cls", "variant", "measure", "quantity", "n_districts", "n_ref", "auc", "lo", "hi", "median_dep", "median_ref"]]
          .round(3).to_string())


if __name__ == "__main__":
    main()
