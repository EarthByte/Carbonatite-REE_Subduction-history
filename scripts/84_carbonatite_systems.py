#!/usr/bin/env python
"""Carbonatite and alkaline rare-earth mineral systems: comparisons and ranked feature analysis.

Reads data/derived/carbonatite_features_<tag> (scripts/83) and its stage table. Two parts.

``compare``. For every feature and every comparison set, the probability that the value at a
district exceeds that at a reference point (ties half; each district counts once; reference points
re-weighted to the district-weighted ages in deciles; points beneath present-day ice and outside
the plate model removed), with a 95% interval from resampling districts and reference points, and
a two-sided p-value from the same resamples, Holm-adjusted within each comparison set and
mineral-system stage, and the same probability at equal craton-edge distance, at equal exploration density and at both
(joint tertile cells, scripts/71). Comparison sets:

  CARB, AIREE, KIMB, PCU            each class against its reference points
  CARB ore, CARB barren             carbonatites with and without rare-earth or niobium ore (scripts/83)
  CARB ore vs barren                the two directly, both sides weighted by district (with the
                                    time in the humid tropics after formation: lateritic upgrading)
  AIREE vs CARB                     alkaline-igneous REE deposits against carbonatites directly:
                                    do the two occupy different lithospheric and tectonic niches?
  AIREE <era> vs CARB <era>, age-matched  the same within the Phanerozoic and within the Proterozoic
  AIREE <setting> vs CARB, age-matched  rift/anorogenic and post-collisional districts (config/aireee_settings.csv)
  AIREE vs CARB, age-matched        the same with carbonatite districts reweighted to the ages of the
                                    alkaline-igneous districts (deciles)
  CARB, AIREE in fertilised domains both sides restricted to the domains of Spandler et al. (2026)
                                    formed before them (does the time since fertilisation differ?)
  CARB, AIREE inside / outside cratons  both sides split by the craton outline at the age of the point
  ... mineable                      the same restricted, on both sides, to sites under at most 1 km
                                    of present-day sediment, where a system could be found and mined

``rank``. Random forests (300 trees, at least three points per leaf) separate the districts of a
class from its reference points, deposits weighted by district and both sides reweighted to equal
total weight. Folds are 30 spatial blocks (k-means on position) with the points of one district
kept together, five folds, ten repetitions, scored on the blocks held out. Three models per class:
all features; without the exposure features (sediment cover, mineability, exploration density)
and the upgrading features (time in the humid tropics after formation);
the exposure features alone; and, as an ablation, the model without exposure with each further
stage (source, lithosphere, trigger) removed in turn, whose loss of skill measures what that stage
adds that the others do not carry. Held-out skill (the probability that a held-out district scores
above a held-out reference point), permutation importance, and held-out SHAP values summed within
groups of features correlated at |rho| >= 0.8 and within mineral-system stages.

Outputs (each ONE file, written through a .partial name):
  data/derived/carbonatite_systems_<tag>.csv        comparisons, skill and importance (long table)
  data/derived/carbonatite_shap_<tag>.parquet       held-out SHAP value of every feature per point

    python scripts/84_carbonatite_systems.py --boot 200 --repeats 2     # pilot
    python scripts/84_carbonatite_systems.py
"""
import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import ROOT, load_config, read_table, write_table  # noqa: E402
from rifts.stats import impute_median_fold  # noqa: E402

warnings.filterwarnings("ignore")
_spec = importlib.util.spec_from_file_location("fc", Path(__file__).with_name("71_kimberlite_fate_contrast.py"))
fc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fc)
wauc, strat_auc, cells_of, age_weights = fc.wauc, fc.strat_auc, fc.cells_of, fc.age_weights

MAX_MISSING = 0.4
CUTOFF = 0.8
CONDITIONS = {"equal craton-edge distance": ["craton_dist_km"],
              "equal exploration density": ["other_family_km"],
              "equal craton-edge distance and exploration density": ["craton_dist_km", "other_family_km"]}


def aireee_setting(d, root):
    """Tectonic setting of each alkaline-silicate REE record from config/aireee_settings.csv (this
    study's reading of the literature in the classes of Beard et al. 2023: 'rift' for rift and
    anorogenic within-plate complexes, 'post-collisional' for late- to post-orogenic and post-subduction
    complexes, 'uncertain' where the literature does not settle it), matched to the nearest listed
    complex within 25 km; records without a match are 'uncertain'."""
    cfg = pd.read_csv(root / "config" / "aireee_settings.csv")
    lon, lat = np.radians(d["lon0"].to_numpy(float)), np.radians(d["lat0"].to_numpy(float))   # present-day position
    clon, clat = np.radians(cfg["lon"].to_numpy(float)), np.radians(cfg["lat"].to_numpy(float))
    cosd = np.sin(lat)[:, None] * np.sin(clat)[None, :] + np.cos(lat)[:, None] * np.cos(clat)[None, :] * np.cos(lon[:, None] - clon[None, :])
    km = 6371.0 * np.arccos(np.clip(cosd, -1, 1))
    j = km.argmin(1)
    out = cfg["setting"].to_numpy()[j].astype(object)
    out[km.min(1) > 25.0] = "uncertain"
    return pd.Series(out, index=d.index)


def district_flags(p):
    """Ore status and era of every carbonatite and alkaline-silicate district, given to each of its records,
    so that no district falls on both sides of a split (review 2026-11-09). A district is ore-bearing when any
    of its records has a qualifying deposit within 10 km, REE-ore-bearing when any has REE ore, and
    niobium-ore-only when it has niobium ore and no REE ore; its era is that of the median age of its
    records, and it lies inside or outside a craton by the median of their distances to the craton edge.
    Reference points keep their own values and no ore."""
    p = p.copy()
    dep = (p["kind"] == "deposit").to_numpy()
    key = p["cluster_id"].fillna(p["pt_id"]).to_numpy()[dep]
    d = p.loc[dep]
    for c in ("ore", "ore_ree", "ore_nb_only"):
        if c in p:
            p.loc[dep, c] = d[c].astype(bool).groupby(key).transform("any").to_numpy()
    if "ore_ree" in p and "ore_nb_only" in p:
        p.loc[dep, "ore_nb_only"] = p.loc[dep, "ore_nb_only"].astype(bool) & ~p.loc[dep, "ore_ree"].astype(bool)
    p["district_age_ma"] = p["age_ma"].astype(float)
    p.loc[dep, "district_age_ma"] = d["age_ma"].astype(float).groupby(key).transform("median").to_numpy()
    # inside or outside a craton outline at the age of the site: the median over the district's records
    p["district_craton_km"] = p["craton_dist_km"].astype(float)
    p.loc[dep, "district_craton_km"] = d["craton_dist_km"].astype(float).groupby(key).transform("median").to_numpy()
    for c in ("ore", "ore_ree", "ore_nb_only"):
        if c in p:
            p[c] = p[c].fillna(False).astype(bool)
    return p


def sets(p, mineable=False):
    """(label, deposit rows, reference rows) for every comparison."""
    ok = ~p["ice"].fillna(False).astype(bool) & p["in_model"].fillna(True).astype(bool)
    if mineable:
        ok &= p["mineable"] == 1
    p = p[ok]
    out = []
    for cls in ("CARB", "AIREE", "KIMB", "PCU"):
        d, r = p[(p["group"] == cls) & (p["kind"] == "deposit")], p[(p["group"] == cls) & (p["kind"] == "reference")]
        out.append((cls, d, r))
        if cls == "CARB":
            out.append(("CARB ore", d[d["ore"]], r))
            out.append(("CARB barren", d[~d["ore"].astype(bool)], r))
            # the ore class by commodity (review 2026-11-08): REE ore, and niobium ore without REE
            if "ore_ree" in d:
                out.append(("CARB REE ore", d[d["ore_ree"].astype(bool)], r))
                out.append(("CARB Nb ore only", d[d["ore_nb_only"].astype(bool)], r))
        if cls in ("CARB", "AIREE"):
            # inside the fertilised domains only, both sides: is the time since fertilisation (and the
            # arrival of slab water) different from that of reference points in the same domains?
            dm, rm = d["fml_primed"] == 1, r["fml_primed"] == 1
            out.append((f"{cls} in fertilised domains", d[dm], r[rm]))
            # on and off cratons (craton outline at the age of the point)
            for name, sd, sr in (("inside cratons", d["district_craton_km"] < 0, r["craton_dist_km"] < 0),
                                 ("outside cratons", d["district_craton_km"] >= 0, r["craton_dist_km"] >= 0)):
                out.append((f"{cls} {name}", d[sd], r[sr]))
            # by era, each against the reference points of the same era: whether the present-day
            # lithospheric measures separate the young districts as they do the old (review 2026-11-08)
            for era, lo, hi in (("Phanerozoic", 0.0, 541.0), ("Proterozoic", 541.0, 2500.0)):
                # a district takes the era of its median age (district_flags), a reference point its own age
                sd = (d["district_age_ma"] >= lo) & (d["district_age_ma"] < hi)
                sr = (r["age_ma"] >= lo) & (r["age_ma"] < hi)
                if d.loc[sd, "cluster_id"].nunique() >= 5:
                    out.append((f"{cls} {era}", d[sd], r[sr]))
    return out


def boot_p(b):
    """Two-sided p-value that the probability is 0.5, from the resampled estimates."""
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    if not len(b):
        return np.nan
    return float(min(1.0, 2 * min(np.mean(b <= 0.5), np.mean(b >= 0.5)) + 1.0 / len(b)))


def holm(rows):
    """Holm-adjusted p-values within each comparison set and mineral-system stage (a family)."""
    t = pd.DataFrame(rows)
    if "p" not in t:
        return rows
    t["p_holm"] = np.nan
    for _, g in t[t["p"].notna()].groupby(["set", "stage"]):
        order = g["p"].sort_values().index
        m = len(order)
        adj = np.maximum.accumulate([min(1.0, (m - i) * t.loc[k, "p"]) for i, k in enumerate(order)])
        t.loc[order, "p_holm"] = adj
    return t.to_dict("records")


def spatial_blocks(lon, lat, n_blocks=30, seed=84):
    """Block label of every point: k-means on the unit sphere over all points given (deposits and
    reference points together), so that a block bootstrap resamples regions rather than points."""
    from sklearn.cluster import KMeans
    lon, lat = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    xyz = np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    n = min(n_blocks, len(xyz))
    return KMeans(n, n_init=4, random_state=seed).fit_predict(xyz)


def block_interval(x, wx, bx, y, wy, by, boot, rng):
    """95% interval of the weighted AUC when entire spatial blocks are resampled with replacement on
    both sides (a block bootstrap; blocks from spatial_blocks)."""
    labels = np.unique(np.r_[bx, by])
    gx = {b: np.flatnonzero(bx == b) for b in labels}
    gy = {b: np.flatnonzero(by == b) for b in labels}
    vals = []
    for _ in range(boot):
        pick = rng.choice(labels, len(labels), replace=True)
        ix = np.concatenate([gx[b] for b in pick]) if any(len(gx[b]) for b in pick) else np.array([], int)
        iy = np.concatenate([gy[b] for b in pick]) if any(len(gy[b]) for b in pick) else np.array([], int)
        vals.append(wauc(x[ix], wx[ix], y[iy], wy[iy]) if len(ix) and len(iy) else np.nan)
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    return (np.nanpercentile(vals, [2.5, 97.5]) if len(vals) >= 20 else (np.nan, np.nan))


def compare_one(d, r, col, boot, rng):
    """Probability that the value at a district of ``d`` exceeds that at a reference point of ``r`` for
    feature ``col`` (districts weighted, reference points reweighted to the district-weighted ages),
    with 95% interval and two-sided p from ``boot`` resamples of districts and of reference points, and
    the same probability at each of CONDITIONS (joint tertile cells)."""
    x, y = d[col].to_numpy(float), r[col].to_numpy(float)
    wx = d["w_cluster"].to_numpy(float)
    wy = age_weights(d["age_ma"].to_numpy(float), wx, r["age_ma"].to_numpy(float))
    okr = np.isfinite(y) & (wy > 0)
    row = dict(n_districts=int(d.loc[np.isfinite(x), "cluster_id"].nunique()), n_ref=int(np.isfinite(y).sum()),
               # effective number of reference points after the age reweighting (Kish): (sum w)^2 / sum w^2
               ess_ref=float(wy[okr].sum() ** 2 / np.sum(wy[okr] ** 2)) if okr.any() else np.nan,
               median_dep=float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan,
               median_ref=float(np.nanmedian(y)) if np.isfinite(y).any() else np.nan,
               auc=wauc(x, wx, y, wy))
    # the same probability with the reference ages matched in twenty bins instead of ten
    row["auc at 20 age bins"] = wauc(x, wx, y, age_weights(d["age_ma"].to_numpy(float), wx, r["age_ma"].to_numpy(float), n=20))
    groups = list(pd.Series(range(len(d))).groupby(d["cluster_id"].to_numpy()).indices.values())
    if boot and np.isfinite(row["auc"]):
        b = []
        for _ in range(boot):
            ix = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
            iy = rng.integers(0, len(y), len(y))
            b.append(wauc(x[ix], wx[ix], y[iy], wy[iy]))
        row["lo"], row["hi"] = np.nanpercentile(b, [2.5, 97.5])
        row["p"] = boot_p(b)
        # interval from resampling spatial blocks of both sides together (review 2026-11-08)
        if "_block" in d and "_block" in r:
            row["lo_block"], row["hi_block"] = block_interval(x, wx, d["_block"].to_numpy(), y, wy, r["_block"].to_numpy(),
                                                              min(boot, 500), rng)
    for name, conds in CONDITIONS.items():
        if not all(c in d for c in conds):
            continue
        cd, cr = cells_of([d[c].to_numpy(float) for c in conds], [r[c].to_numpy(float) for c in conds])
        row[f"auc at {name}"] = strat_auc(x, wx, cd, y, wy, cr)
    return row


def matched_weights(age1, w1, age2, w2, n=10):
    """District weights of the second set rescaled so that its age distribution matches the
    district-weighted ages of the first, in deciles of the first set's ages."""
    q = np.unique(np.nanquantile(age1, np.linspace(0, 1, n + 1)[1:-1]))
    b1, b2 = np.searchsorted(q, age1), np.searchsorted(q, age2)
    s1 = np.bincount(b1, w1, len(q) + 1) / w1.sum()
    s2 = np.bincount(b2, w2, len(q) + 1) / w2.sum()
    return w2 * np.where(s2[b2] > 0, s1[b2] / np.maximum(s2[b2], 1e-12), 0.0)


def direct(d1, d2, col, boot, rng, age_match=False):
    """The same probability between two sets of districts, ``d1`` against ``d2``, both weighted by district
    and resampled by district; with ``age_match`` the second set is reweighted to the ages of the first
    (matched_weights)."""
    x, y = d1[col].to_numpy(float), d2[col].to_numpy(float)
    wx, wy = d1["w_cluster"].to_numpy(float), d2["w_cluster"].to_numpy(float)
    if age_match:
        wy = matched_weights(d1["age_ma"].to_numpy(float), wx, d2["age_ma"].to_numpy(float), wy)
    # districts counted are those with a value, so that a measure defined for part of a class (the
    # time since the last subduction episode) reports the number it rests on (review 2026-10-18)
    row = dict(n_districts=int(d1.loc[np.isfinite(x), "cluster_id"].nunique()),
               n_ref=int(d2.loc[np.isfinite(y), "cluster_id"].nunique()),
               median_dep=float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan,
               median_ref=float(np.nanmedian(y)) if np.isfinite(y).any() else np.nan, auc=wauc(x, wx, y, wy))
    g1 = list(pd.Series(range(len(d1))).groupby(d1["cluster_id"].to_numpy()).indices.values())
    g2 = list(pd.Series(range(len(d2))).groupby(d2["cluster_id"].to_numpy()).indices.values())
    if boot and np.isfinite(row["auc"]):
        b = []
        for _ in range(boot):
            i1 = np.concatenate([g1[k] for k in rng.integers(0, len(g1), len(g1))])
            i2 = np.concatenate([g2[k] for k in rng.integers(0, len(g2), len(g2))])
            b.append(wauc(x[i1], wx[i1], y[i2], wy[i2]))
        row["lo"], row["hi"] = np.nanpercentile(b, [2.5, 97.5])
        row["p"] = boot_p(b)
        if "_block" in d1 and "_block" in d2:
            row["lo_block"], row["hi_block"] = block_interval(x, wx, d1["_block"].to_numpy(), y, wy, d2["_block"].to_numpy(),
                                                              min(boot, 500), rng)
        if "since" in col and "episode" not in col:
            # the end of an episode is read from topologies resolved every 5 Myr: the probability when every
            # interval on both sides is perturbed by up to ±10 Myr (uniform), median and 5th–95th percentiles
            # over 200 draws (review 2026-11-08)
            jit = [wauc(x + rng.uniform(-10, 10, len(x)), wx, y + rng.uniform(-10, 10, len(y)), wy) for _ in range(200)]
            row["auc termination ±10 Myr"], row["lo termination ±10 Myr"], row["hi termination ±10 Myr"] = \
                np.nanpercentile(jit, [50, 5, 95])
    return row


def run_compare(p, feats, stage, boot, seed=84):
    """Every comparison of sets(): each class against its reference points, carbonatites with and without
    ore against theirs and against each other, alkaline-silicate REE districts against carbonatites
    (directly, age-matched, by era and by tectonic setting), for all sites and for mineable sites only.
    Returns the rows with Holm-adjusted p-values within each set and stage."""
    rng = np.random.default_rng(seed)
    p = p.copy()
    p["_block"] = spatial_blocks(p["lon0"], p["lat0"], 30, seed)
    rows = []
    for mineable in (False, True):
        tag = " mineable" if mineable else ""
        ss = sets(p, mineable)
        for label, d, r in ss:
            for col in feats:
                if col == "mineable" and mineable:
                    continue
                row = compare_one(d, r, col, boot, rng)
                rows.append(dict(part="compare", set=label + tag, feature=col, stage=stage.get(col, ""), **row))
        ore = next(d for lab, d, _ in ss if lab == "CARB ore")
        bar = next(d for lab, d, _ in ss if lab == "CARB barren")
        carb = next(d for lab, d, _ in ss if lab == "CARB")
        alk = next(d for lab, d, _ in ss if lab == "AIREE")
        ree = next((d for lab, d, _ in ss if lab == "CARB REE ore"), None)
        for col in feats:
            rows.append(dict(part="compare", set="CARB ore vs barren" + tag, feature=col, stage=stage.get(col, ""),
                             **direct(ore, bar, col, boot, rng)))
            if ree is not None and ree["cluster_id"].nunique() >= 5:
                rows.append(dict(part="compare", set="CARB REE ore vs barren" + tag, feature=col, stage=stage.get(col, ""),
                                 **direct(ree, bar, col, boot, rng)))
            rows.append(dict(part="compare", set="AIREE vs CARB" + tag, feature=col, stage=stage.get(col, ""),
                             **direct(alk, carb, col, boot, rng)))
            rows.append(dict(part="compare", set="AIREE vs CARB, age-matched" + tag, feature=col,
                             stage=stage.get(col, ""), **direct(alk, carb, col, boot, rng, age_match=True)))
        # the alkaline-silicate class split at the Proterozoic-Phanerozoic boundary, each part against the
        # carbonatites of its own era reweighted to its ages: whether the short lag after subduction belongs
        # to the Phanerozoic, largely post-collisional districts or extends to the Proterozoic ones
        for era, lo, hi in (("Phanerozoic", 0.0, 541.0), ("Proterozoic", 541.0, 2500.0)):
            alk_e = alk[(alk["district_age_ma"] >= lo) & (alk["district_age_ma"] < hi)]
            carb_e = carb[(carb["district_age_ma"] >= lo) & (carb["district_age_ma"] < hi)]
            if alk_e["cluster_id"].nunique() < 5 or carb_e["cluster_id"].nunique() < 5:
                continue
            for col in feats:
                rows.append(dict(part="compare", set=f"AIREE {era} vs CARB {era}, age-matched" + tag, feature=col,
                                 stage=stage.get(col, ""), **direct(alk_e, carb_e, col, boot, rng, age_match=True)))
        # the same by tectonic setting (config/aireee_settings.csv, after Beard et al. 2023): rift and anorogenic
        # within-plate complexes against post-collisional and post-subduction ones, each against all carbonatites
        # reweighted to its ages; the setting classification is a reading of the literature, held in the config
        # file with its basis per complex so that it can be checked and revised
        setting = aireee_setting(alk, ROOT)
        for name in ("rift", "post-collisional"):
            alk_s = alk[(setting == name).to_numpy()]
            if alk_s["cluster_id"].nunique() < 5:
                continue
            for col in feats:
                rows.append(dict(part="compare", set=f"AIREE {name} vs CARB, age-matched" + tag, feature=col,
                                 stage=stage.get(col, ""), **direct(alk_s, carb, col, boot, rng, age_match=True)))
        if not mineable:
            counts = alk.drop_duplicates("cluster_id").assign(setting=setting[~alk["cluster_id"].duplicated()].to_numpy())
            print("  alkaline-silicate districts by setting: " + ", ".join(f"{k} {v}" for k, v in counts["setting"].value_counts().items()), flush=True)
        print(f"  compared{tag}", flush=True)
    return holm(rows)


# ------------------------------------------------------------------------------ ranking
def clusters(Xdf):
    """Group the features by absolute Spearman correlation (average linkage, cut at CUTOFF). Returns {name:
    members}, each group named after its members or after its first member and the number of others."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    rho = Xdf.corr(method="spearman").abs().fillna(0).to_numpy().copy()
    np.fill_diagonal(rho, 1.0)
    Z = linkage(squareform(1 - rho, checks=False), method="average")
    ids = fcluster(Z, 1 - CUTOFF, criterion="distance")
    groups = {}
    for name, i in zip(Xdf.columns, ids):
        groups.setdefault(i, []).append(name)
    return {" + ".join(v) if len(v) <= 3 else f"{v[0]} + {len(v) - 1} correlated": v for v in groups.values()}


def folds_for(lon, lat, district, n_blocks, seed):
    """Five cross-validation folds from ``n_blocks`` spatial blocks (k-means on the unit sphere), with all
    points of a district in the block of its first point; blocks are assigned to folds at random with
    ``seed``. Returns (train, test) index pairs."""
    from sklearn.cluster import KMeans
    lo, la = np.radians(lon), np.radians(lat)
    xyz = np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    b = KMeans(n_blocks, n_init=4, random_state=seed).fit_predict(xyz)
    # a district's points take the block of its first point
    s = pd.Series(b)
    first = s.groupby(pd.Series(district)).transform("first")
    b = np.where(pd.notna(district), first, b)
    blocks = np.unique(b)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(blocks, rng.permutation(len(blocks)) % 5))
    f = np.array([fold_of[v] for v in b])
    return [(np.nonzero(f != k)[0], np.nonzero(f == k)[0]) for k in range(5)]


def run_rank(p, feats, stage, repeats, n_blocks, seed=84):
    """Random forests that separate the districts of each class (and of ore-bearing carbonatites) from
    their reference points, with spatially blocked folds repeated ``repeats`` times. Models: all
    features, without exposure and upgrading, exposure alone, and the model without exposure with each
    of source, lithosphere and trigger removed in turn (skill only). Missing values are filled in each fold
    with the medians of its training points. For the first two models also permutation importance and held-out
    SHAP values, summed within correlation groups and within stages. Returns the rows and the table of
    held-out SHAP values per point."""
    import shap
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score
    rows, shap_rows = [], []
    # exposure and post-formation upgrading say whether a system is found and enriched, not where it
    # forms; both are kept out of the models of formation
    exposure = [c for c in feats if stage.get(c) == "exposure"]
    after = [c for c in feats if stage.get(c) in ("exposure", "upgrading")]
    feats = [c for c in feats if stage.get(c) != "sensitivity"]   # compared only, never ranked
    ok = ~p["ice"].fillna(False).astype(bool) & p["in_model"].fillna(True).astype(bool)
    for label, sel in (("CARB", p["group"] == "CARB"), ("CARB ore", (p["group"] == "CARB") & ((p["kind"] == "reference") | p["ore"])),
                       ("AIREE", p["group"] == "AIREE"), ("KIMB", p["group"] == "KIMB"), ("PCU", p["group"] == "PCU")):
        s = p[sel & ok].reset_index(drop=True)
        y = (s["kind"] == "deposit").to_numpy(int)
        if y.sum() < 15:
            continue
        w = np.where(y == 1, s["w_cluster"].to_numpy(float), 1.0)
        w = np.where(y == 1, w / w[y == 1].sum(), w / w[y == 0].sum())
        use = [c for c in feats if s[c].notna().mean() >= 1 - MAX_MISSING and s[c].nunique() > 1]
        X = s[use].astype(float)
        # gaps stay in X (correlation groups and SHAP directions from measured values); each fold fills
        # them from its training points only
        groups = clusters(X)
        district = np.where(y == 1, s["cluster_id"].astype(str), None)
        reps = [folds_for(s["lon0"].to_numpy(float), s["lat0"].to_numpy(float), district, n_blocks, seed + k)
                for k in range(repeats)]
        noexp = [c for c in use if c not in after]
        models = [("all features", use), ("without exposure", noexp), ("exposure alone", [c for c in use if c in exposure])]
        # ablation: each stage removed in turn from the model without exposure; skill only
        for st_name in ("source", "lithosphere", "trigger"):
            models.append((f"without exposure and {st_name}", [c for c in noexp if stage.get(c) != st_name]))
        for model, cols in models:
            if not cols:
                continue
            detail = model in ("all features", "without exposure")
            Xm = X[cols].to_numpy()
            skill, perm, sv_acc, n_acc = [], np.zeros(len(cols)), np.zeros((len(s), len(cols))), np.zeros(len(s))
            for k, folds in enumerate(reps):
                score = np.full(len(s), np.nan)
                for tr, te in folds:
                    if y[tr].min() == y[tr].max() or y[te].sum() == 0 or y[te].sum() == len(te):
                        continue
                    m = RandomForestClassifier(300, min_samples_leaf=3, n_jobs=-1, random_state=seed + k)
                    Xtr, Xte = impute_median_fold(Xm, tr, te)
                    m.fit(Xtr, y[tr], sample_weight=w[tr])
                    score[te] = m.predict_proba(Xte)[:, 1]
                    if not detail:
                        continue
                    pi = permutation_importance(m, Xte, y[te], sample_weight=w[te], scoring="roc_auc",
                                                n_repeats=5, random_state=seed)
                    perm += pi.importances_mean / (5 * repeats)
                    if detail:
                        v = shap.TreeExplainer(m).shap_values(Xte)
                        v = v[1] if isinstance(v, list) else (v[..., 1] if v.ndim == 3 else v)
                        sv_acc[te] += v
                        n_acc[te] += 1
                good = np.isfinite(score)
                skill.append(roc_auc_score(y[good], score[good], sample_weight=w[good]))
                print(f"    {label}, {model}: repetition {k + 1} of {len(reps)}, skill {skill[-1]:.2f}", flush=True)
            rows.append(dict(part="skill", set=label, model=model, n_districts=int(s.loc[y == 1, "cluster_id"].nunique()),
                             n_ref=int((y == 0).sum()), auc=float(np.mean(skill)), lo=float(np.min(skill)),
                             hi=float(np.max(skill)), n_features=len(cols),
                             skills=";".join(f"{v:.4f}" for v in skill)))
            if not detail:
                continue
            for c, v in zip(cols, perm):
                rows.append(dict(part="permutation importance", set=label, model=model, feature=c,
                                 stage=stage.get(c, ""), importance=float(v)))
            if not detail:
                continue
            with np.errstate(invalid="ignore"):
                sv = sv_acc / n_acc[:, None]
            mean_abs = np.nanmean(np.abs(sv), axis=0)
            tot = np.nansum(mean_abs)
            for c, v in zip(cols, mean_abs):
                # direction: rank correlation of the feature with its SHAP value
                rho = pd.Series(X[c]).corr(pd.Series(sv[:, cols.index(c)]), method="spearman")
                rows.append(dict(part="shap", set=label, model=model, feature=c, stage=stage.get(c, ""),
                                 importance=float(v / tot), direction=float(rho)))
            for gname, members in groups.items():
                idx = [cols.index(c) for c in members if c in cols]
                if idx:
                    rows.append(dict(part="shap group", set=label, model=model, feature=gname,
                                     stage="/".join(sorted({stage.get(c, '') for c in members})),
                                     importance=float(np.nanmean(np.abs(np.nansum(sv[:, idx], axis=1))) / tot)))
            for st in sorted({stage.get(c, "") for c in cols}):
                idx = [i for i, c in enumerate(cols) if stage.get(c, "") == st]
                rows.append(dict(part="shap stage", set=label, model=model, feature=st, stage=st,
                                 importance=float(np.nanmean(np.abs(np.nansum(sv[:, idx], axis=1))) / tot)))
            t = pd.DataFrame(sv, columns=[f"shap:{c}" for c in cols])
            t.insert(0, "pt_id", s["pt_id"].to_numpy())
            t.insert(1, "set", label)
            t.insert(2, "model", model)
            t.insert(3, "y", y)
            shap_rows.append(t)
            print(f"  {label}, {model}: held-out skill {np.mean(skill):.2f}", flush=True)
    return rows, (pd.concat(shap_rows, ignore_index=True) if shap_rows else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--parts", nargs="*", default=["compare", "rank"], choices=["compare", "rank"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=30)
    a = ap.parse_args()
    cfg = load_config(a.config)
    der = cfg["paths"]["derived"]
    p = read_table(der / f"carbonatite_features_{a.tag}")
    # the mineral-system stage of every feature present (scripts/83)
    sp = der / f"carbonatite_feature_stages_{a.tag}.csv"
    if not sp.exists():                  # written before the stage table carried the tag
        print(f"  {sp.name} not found: using carbonatite_feature_stages.csv")
        sp = der / "carbonatite_feature_stages.csv"
    st = pd.read_csv(sp)
    st = st[st["present"]]
    stage = dict(zip(st["feature"], st["stage"]))
    feats = [c for c in st["feature"] if c in p]
    for c in feats:
        p[c] = pd.to_numeric(p[c], errors="coerce") if p[c].dtype != bool else p[c].astype(float)
    p = district_flags(p)
    rows = []
    if "compare" in a.parts:
        rows += run_compare(p, feats, stage, a.boot)
    sh = None
    if "rank" in a.parts:
        r, sh = run_rank(p, feats, stage, a.repeats, a.blocks)
        rows += r
    out = der / f"carbonatite_systems_{a.tag}.csv"
    # a run of one part keeps the rows of the other part from the existing table
    if set(a.parts) != {"compare", "rank"} and out.exists():
        old = pd.read_csv(out)
        rows = old[~old["part"].isin(["compare"] if "compare" in a.parts else ["skill", "permutation importance", "shap",
                                                                               "shap group", "shap stage"])].to_dict("records") + rows
    tmp = out.with_name(out.name + ".partial")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(out)
    print(f"wrote {out}")
    if sh is not None:
        print(f"wrote {write_table(sh, der / f'carbonatite_shap_{a.tag}')}")


if __name__ == "__main__":
    main()
