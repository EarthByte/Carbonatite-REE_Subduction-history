#!/usr/bin/env python
"""Kimberlites and rift fate: the paired contrast, eruption-time geography and its limits.

scripts/53 compares kimberlite districts with age-matched continental reference points in their
distance to the nearest rift of one fate that stopped rifting a given interval before eruption.
This step tests the quantity the kimberlite paper rests on, the contrast between margins and
failed rifts, and explains how it behaves when positions are reconstructed to eruption. Every
estimate is the district-weighted probability that the value at a kimberlite exceeds that at a
reference point (ties half; 0.5 is no difference); the contrast is the margin estimate minus the
failed-rift estimate on the same resamples, negative when kimberlites lie relatively closer to
margins than to failed rifts, each judged against its own reference distribution.

  contrast     margins, failed rifts and their difference, in present-day positions and in
               positions at eruption, under: resampling by district; resampling by shared rift
               (districts whose nearest margin or nearest failed rift in the window is the same
               rift move together, as connected groups); localities with measured ages only;
               within tertiles of craton-edge and trench distance together; without the age
               balance. Holm over the cessation windows within each frame. Throughout, reference
               points are reweighted to the district-weighted kimberlite ages (deciles), because
               they were drawn to the locality-weighted ages.
  combined     1,000 draws, each perturbing eruption and rift ages and then resampling by
               shared rift, so that one interval carries both limitations. Measured eruption
               ages are drawn within their stated range (+/- 2 Myr assumed where none is
               given); inferred ages are perturbed by one shared offset per inferred value
               (+/- 10 Myr assumed; 5 and 20 Myr as checks), because localities given one
               inferred age share one assumption. Rift end ages are drawn within the young and
               old end ages of the rift database where these differ, and +/- 5 Myr assumed
               where they do not; onsets likewise. Reference ages are held at their drawn
               values and the age balance of every draw is reported.
  physical     the kilometre difference between the nearest margin and the nearest failed rift
               of the window at each point (each capped at the search range), compared between
               kimberlites and reference points: below 0.5, kimberlites lie physically closer to
               margins relative to failed rifts than reference points do.
  cap          the search range of 1,000, 1,500, 2,000, 3,000 and 5,000 km and without limit:
               the share of each side with a qualifying rift within range, the estimate among
               points with one on both sides, and the censored estimate.
  background   both sides restricted to cratonic lithosphere at eruption (inside a craton
               outline; within 300 km of one; shear-wave velocity at 150 km at or above the
               10th percentile of the kimberlites; inside a craton outline and within 1,000 km of
               a kimberlite district), reference points reweighted to the kimberlite ages within
               each, resampled by district and by shared rift, and under the combined draws,
               with the districts and points retained and the remaining imbalance.
  frames       the 2x2 decomposition (kimberlites and reference points each in either frame),
               and per point the nearest qualifying rift in both frames, classified as the same
               rift on the point's plate, the same rift with a changed distance, a different
               rift, or a rift qualifying in one frame only; the district contributions to the
               change; the nearest rifts of the kimberlite districts with their fate,
               classification fields and end ages.
  cells        matched cells: districts and weight retained, residual standardised differences
               of craton-edge and trench distance within cells.
  diamond      diamondiferous against barren districts with districts holding both kinds
               resampled once for both sides.
  sliding      the contrast in 10 Myr windows every 5 Myr.
  fate         the rift database records multiphase rifts as one rift per phase, so an early phase
               of a rift system that later broke a continent apart is a separate rift whose
               activity ended before breakup, and falls among the failed rifts. Every contrast is
               repeated with fate assigned to rift systems: a failed-rift phase within 100 km
               (200 km as a check) of a margin outline whose rifting ended no more than 50 Myr
               after it joins the margins ("rift systems"), or is left out of both
               ("interior failed rifts").

Output: ONE long table, data/derived/kimberlite_fate_contrast_<tag>.csv (``kind`` says which
part a row belongs to), written through a .partial name.

    python scripts/71_kimberlite_fate_contrast.py --draws 50 --boot 200     # pilot
    python scripts/71_kimberlite_fate_contrast.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import load_config, read_table  # noqa: E402
from rifts.context import rift_table, RIFT_TYPES, read_outlines  # noqa: E402
from rifts.frames import deposit_table, deposit_rows, load_screen_frame, reference_rows, within_age  # noqa: E402
from rifts.multiplicity import boot_p, holm  # noqa: E402
from rifts.riftlag import rift_distances  # noqa: E402

MAX_KM = 3000.0
WINDOWS = ((0.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 70.0), (70.0, 120.0))
MAIN = ("20_40", "40_70")
SLIDE = tuple((float(a), float(a + 10)) for a in range(0, 115, 5))
CAPS = (1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 25000.0)
FATES = ("margin", "failed rift")
SYSTEM_KM, SYSTEM_GAP = (100.0, 200.0), 50.0


def system_phases(rifts, fate, km, gap):
    """Failed-rift phases that lie within ``km`` of a margin outline whose rifting ended no
    more than ``gap`` Myr after theirs: earlier phases of a rift system that broke up."""
    from scipy.spatial import cKDTree
    from rifts.common import EARTH_RADIUS_KM, lonlat_to_xyz
    from rifts.riftlag import _rift_vertices
    xy, owner = _rift_vertices(rifts)
    X = lonlat_to_xyz(xy[:, 0], xy[:, 1])
    is_m = np.isin(owner, np.nonzero(fate == "margin")[0])
    tree, om = cKDTree(X[is_m]), owner[is_m]
    end = rifts["end_ma"].to_numpy(float)
    chord = 2 * np.sin(km / EARTH_RADIUS_KM / 2)
    out = np.zeros(len(rifts), bool)
    for j in np.nonzero(fate == "failed rift")[0]:
        hits = tree.query_ball_point(X[owner == j], chord)
        ms = np.unique(om[np.concatenate([np.asarray(h, int) for h in hits])]) if len(hits) else []
        ms = [m for m in ms if 0 <= end[j] - end[m] <= gap]
        out[j] = len(ms) > 0
    return out
CELL_BY = ("craton_dist_km", "sz_dist_km")
MEASURED_SD, INFERRED_SD, END_SD = 2.0, 10.0, 5.0


def wname(lo, hi):
    return f"{int(lo)}_{int(hi)}"


# --------------------------------------------------------------------------- estimators
def wauc(x, wx, y, wy):
    """P(x > y), ties half, weights on both sides."""
    ok, oy = np.isfinite(x) & (wx > 0), np.isfinite(y) & (wy > 0)
    x, wx, y, wy = x[ok], wx[ok], y[oy], wy[oy]
    if len(x) < 3 or len(y) < 10:
        return np.nan
    o = np.argsort(y)
    ys, cw = y[o], np.r_[0.0, np.cumsum(wy[o])]
    lo, hi = np.searchsorted(ys, x, "left"), np.searchsorted(ys, x, "right")
    frac = (cw[lo] + 0.5 * (cw[hi] - cw[lo])) / cw[-1]
    return float(np.sum(wx * frac) / wx.sum())


def cells_of(dvals, rvals, n_bins=3):
    """Joint tertile cells over both sides together (labels for deposits, references)."""
    cd, cr = np.zeros(len(dvals[0]), int), np.zeros(len(rvals[0]), int)
    for a, b in zip(dvals, rvals):
        edges = np.nanquantile(np.r_[a, b], np.linspace(0, 1, n_bins + 1)[1:-1])
        cd = cd * n_bins + np.searchsorted(edges, a)
        cr = cr * n_bins + np.searchsorted(edges, b)
    return cd, cr


def strat_auc(x, wx, cx, y, wy, cy, min_dep=3, min_ref=5):
    """AUC within cells: the weighted AUC of every cell (labels ``cx`` and ``cy``) holding at least
    ``min_dep`` kimberlite rows and ``min_ref`` reference rows, averaged with the kimberlite weight of
    each cell; NaN if no cell qualifies."""
    num = den = 0.0
    for c in np.unique(cx):
        mx, my = cx == c, cy == c
        if mx.sum() < min_dep or my.sum() < min_ref:
            continue
        a = wauc(x[mx], wx[mx], y[my], wy[my])
        if np.isfinite(a):
            num += a * wx[mx].sum()
            den += wx[mx].sum()
    return num / den if den else np.nan


def age_weights(dage, dw, rage, n=10):
    """Reference weights giving the reference points the district-weighted kimberlite age
    distribution, in deciles of the kimberlite ages."""
    q = np.unique(np.nanquantile(dage, np.linspace(0, 1, n + 1)[1:-1]))
    bd, br = np.searchsorted(q, dage), np.searchsorted(q, rage)
    sd = np.bincount(bd, dw, len(q) + 1) / dw.sum()
    sr = np.bincount(br, minlength=len(q) + 1) / len(rage)
    return np.where(sr[br] > 0, sd[br] / np.maximum(sr[br], 1e-12), 0.0)


def smd(a, wa, b, wb):
    ma, mb = np.average(a, weights=wa), np.average(b, weights=wb)
    va, vb = np.average((a - ma) ** 2, weights=wa), np.average((b - mb) ** 2, weights=wb)
    return float((ma - mb) / np.sqrt(0.5 * (va + vb))) if va + vb > 0 else np.nan


# ------------------------------------------------------------------------ rift distances
def nearest(D, ages, onset, end, sel, lo, hi, max_km, active=False):
    """Distance (censored at max_km) and index (-1) of the nearest rift of ``sel`` whose
    rifting ended lo-hi Myr before each point, or that was active at its age."""
    if active:
        m = (onset[None, :] >= ages[:, None]) & (end[None, :] <= ages[:, None])
    else:
        lag = end[None, :] - ages[:, None]
        m = (lag > lo) & (lag <= hi)
    m &= sel[None, :] & np.isfinite(D) & (D <= max_km)
    dm = np.where(m, D, np.inf)
    j = np.argmin(dm, axis=1)
    d = dm[np.arange(len(D)), j]
    has = np.isfinite(d)
    return np.where(has, d, max_km), np.where(has, j, -1), has


class Side:
    """One class: kimberlite (or control) rows and reference rows with their distances."""

    def __init__(self, d, r, Dd, Dr):
        """Hold the rows and their distance matrices to the rifts (``Dd`` kimberlites, ``Dr`` reference
        points), the district weights, reference weights that match the district-weighted kimberlite
        ages, and the row groups of every district for resampling."""
        self.d, self.r = d.reset_index(drop=True), r.reset_index(drop=True)
        self.Dd, self.Dr = Dd, Dr
        self.wd = self.d["w_cluster"].to_numpy(float)
        self.age_d = self.d["age_ma"].to_numpy(float)
        self.age_r = self.r["age_ma"].to_numpy(float)
        # reference points reweighted to the district-weighted kimberlite ages (deciles)
        self.wr = age_weights(self.age_d, self.wd, self.age_r)
        self.cluster = self.d["cluster_id"].to_numpy()
        self.groups = list(pd.Series(range(len(self.d))).groupby(self.cluster).indices.values())


def window_values(s, rifts, sel, lo, hi, max_km=MAX_KM, ages_d=None, ages_r=None, end=None, onset=None,
                  active=False):
    """nearest() for the kimberlite rows and the reference rows of ``s``: the distance to the nearest rift
    of ``sel`` whose rifting ended lo-hi Myr before each point (or that was active at its age, with
    ``active``), its index, and whether one qualifies. ``onset``, ``end``, ``ages_d`` and ``ages_r``
    replace the rift and point ages when given."""
    on = rifts["onset_ma"].to_numpy(float) if onset is None else onset
    en = rifts["end_ma"].to_numpy(float) if end is None else end
    ad = s.age_d if ages_d is None else ages_d
    ar = s.age_r if ages_r is None else ages_r
    xd = nearest(s.Dd, ad, on, en, sel, lo, hi, max_km, active)
    xr = nearest(s.Dr, ar, on, en, sel, lo, hi, max_km, active)
    return xd, xr


def rift_groups(cluster, ids_list):
    """Connected groups of districts linked by a shared nearest rift in any of ``ids_list``."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    codes, uniq = pd.factorize(cluster)
    n = len(uniq)
    rows, cols = [], []
    for ids in ids_list:
        df = pd.DataFrame({"c": codes, "r": ids})
        df = df[df["r"] >= 0].drop_duplicates()
        for _, g in df.groupby("r"):
            c = g["c"].to_numpy()
            rows += [c[0]] * (len(c) - 1)
            cols += list(c[1:])
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    unit = lab[codes]
    return list(pd.Series(range(len(cluster))).groupby(unit).indices.values()), int(lab.max() + 1)


def draw_idx(groups, n_ref, rng):
    pick = rng.integers(0, len(groups), len(groups))
    return np.concatenate([groups[k] for k in pick]), rng.integers(0, n_ref, n_ref)


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--max-age", type=float, default=400.0)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--draws", type=int, default=1000)
    a = ap.parse_args()
    cfg = load_config(a.config)
    der = cfg["paths"]["derived"]
    root = Path(__file__).resolve().parents[1]

    frame, _, _ = load_screen_frame(cfg, a.tag, verbose=False)
    frame = within_age(frame, a.max_age)
    dep = deposit_table(der, a.tag)
    frame = frame.merge(dep[["dep_id", "subtype", "age_min_ma", "age_max_ma"]].rename(columns={"dep_id": "pt_id"}),
                        on="pt_id", how="left", suffixes=("", "_dep"))
    if "subtype_dep" in frame:
        frame["subtype"] = frame["subtype"].fillna(frame["subtype_dep"])

    rifts = read_outlines(der / "rift_outlines.gpkg")
    rifts = rifts[rifts["GPGIM_TYPE"].isin(RIFT_TYPES)]
    rifts = rifts[np.isfinite(rifts["onset_ma"]) & np.isfinite(rifts["end_ma"])]
    st = pd.read_csv(rift_table(der, "rift_settings", a.tag)).drop_duplicates("outline_id")
    rifts = rifts.merge(st[["outline_id", "fate", "feature_type", "rift_subtype"]], left_on="RIFTID",
                        right_on="outline_id", how="left").reset_index(drop=True)
    # end-age bounds of the rift database (private extract; derived attributes only)
    ends = root / "private" / "rift_fate_end_ages.csv"
    if ends.exists():
        e = pd.read_csv(ends)[["outline_id", "end_young_ma", "end_old_ma", "onset_young_ma", "onset_old_ma",
                               "confidence", "end_strat_young", "end_strat_old"]]
        rifts = rifts.merge(e, on="outline_id", how="left")
    else:
        for c in ("end_young_ma", "end_old_ma", "onset_young_ma", "onset_old_ma"):
            rifts[c] = np.nan
        rifts["confidence"] = rifts["end_strat_young"] = rifts["end_strat_old"] = ""
    onset0, end0 = rifts["onset_ma"].to_numpy(float), rifts["end_ma"].to_numpy(float)
    e_lo = np.fmin(rifts["end_young_ma"], rifts["end_old_ma"]).to_numpy(float)
    e_hi = np.fmax(rifts["end_young_ma"], rifts["end_old_ma"]).to_numpy(float)
    e_has = np.isfinite(e_lo) & np.isfinite(e_hi) & (e_hi > e_lo)
    o_lo = np.fmin(rifts["onset_young_ma"], rifts["onset_old_ma"]).to_numpy(float)
    o_hi = np.fmax(rifts["onset_young_ma"], rifts["onset_old_ma"]).to_numpy(float)
    o_has = np.isfinite(o_lo) & np.isfinite(o_hi) & (o_hi > o_lo)
    active_rift = rifts["fate"].eq("active").to_numpy()
    fsel = {f: rifts["fate"].eq(f).to_numpy() for f in FATES}
    fate_arr = rifts["fate"].to_numpy()
    DEFS = {"database phases": fsel}
    for km in SYSTEM_KM:
        sysp = system_phases(rifts, fate_arr, km, SYSTEM_GAP)
        DEFS[f"rift systems ({int(km)} km)"] = {"margin": fsel["margin"] | sysp, "failed rift": fsel["failed rift"] & ~sysp}
        DEFS[f"interior failed rifts ({int(km)} km)"] = {"margin": fsel["margin"], "failed rift": fsel["failed rift"] & ~sysp}
        print(f"  {sysp.sum()} failed-rift phases lie within {int(km)} km of a margin that ended up to "
              f"{int(SYSTEM_GAP)} Myr after them")
        if km == SYSTEM_KM[0]:
            sys_flag = sysp
    print(f"{len(rifts)} rifts; end-age range from the database for {e_hi.size and e_has.mean():.0%}")

    from rifts.models import load_models
    mantle, _ = load_models(cfg)
    rows = []

    sides = {}
    for cls in ("KIMB", "PCU"):
        d, r = deposit_rows(frame, cls), reference_rows(frame, cls)
        both = pd.concat([d, r], ignore_index=True)
        pts = pd.DataFrame({"pt_id": both["pt_id"].to_numpy(), "lon": both["lon0"].to_numpy(float),
                            "lat": both["lat0"].to_numpy(float), "mlon": both["mlon"].to_numpy(float),
                            "mlat": both["mlat"].to_numpy(float), "age_ma": both["age_ma"].to_numpy(float)})
        Dp = rift_distances(pts, rifts)
        Dr = rift_distances(pts, rifts, reconstructed=True, model=mantle)
        nd = len(d)
        sides[cls] = {"present": Side(d, r, Dp[:nd], Dp[nd:]), "reconstructed": Side(d, r, Dr[:nd], Dr[nd:])}
        print(f"{cls}: {len(d)} localities in {d['cluster_id'].nunique()} districts, {len(r)} reference points")

    def est(s, xd, xr, idx=None, wr=None, cells=None):
        """Weighted AUC of the kimberlite distances ``xd`` against the reference distances ``xr``: for one
        bootstrap draw ``idx`` when given, with reference weights ``wr`` (default s.wr), and within
        cells (strat_auc) when ``cells`` is given."""
        x, y, wx = xd, xr, s.wd
        wy = s.wr if wr is None else wr
        if idx is not None:
            x, wx, y, wy = x[idx[0]], wx[idx[0]], y[idx[1]], wy[idx[1]]
        if cells is None:
            return wauc(x, wx, y, wy)
        cx, cy = cells
        if idx is not None:
            cx, cy = cx[idx[0]], cy[idx[1]]
        return strat_auc(x, wx, cx, y, wy, cy)

    def cells_for(s):
        dv = [s.d[c].to_numpy(float) for c in CELL_BY]
        rv = [s.r[c].to_numpy(float) for c in CELL_BY]
        return cells_of(dv, rv)

    def summarise(b):
        b = np.asarray(b, float)
        return dict(lo=np.nanpercentile(b, 2.5), hi=np.nanpercentile(b, 97.5), p=boot_p(b))

    def summarise_diff(b):
        b = np.asarray(b, float)
        return dict(lo=np.nanpercentile(b, 2.5), hi=np.nanpercentile(b, 97.5), p=boot_p(b, 0.0))

    # ------------------------------------------------- contrast under each treatment
    for fdef, fs in DEFS.items():
     for cls in ("KIMB", "PCU"):
        if fdef != "database phases" and cls == "PCU":
            continue
        for fr in ("present", "reconstructed"):
            s = sides[cls][fr]
            cells = cells_for(s)
            measured = (s.d["age_basis"].astype(str) != "inferred").to_numpy()
            for lo, hi in WINDOWS:
                w = wname(lo, hi)
                (xm, im, hm), (ym, jm, gm) = window_values(s, rifts, fs["margin"], lo, hi)
                (xf, i_f, hf), (yf, jf, gf) = window_values(s, rifts, fs["failed rift"], lo, hi)
                grp_rift, n_units = rift_groups(s.cluster, [im, i_f])
                treat = {
                    "district": (s.groups, None, None, None),
                    "shared rift": (grp_rift, None, None, None),
                    "measured ages": ([g[measured[g]] for g in s.groups if measured[g].any()], None, None, measured),
                    "craton-edge and trench tertiles": (s.groups, None, cells, None),
                    "without age balance": (s.groups, np.ones(len(s.r)), None, None),
                }
                for tname, (groups, wr, cl, keep) in treat.items():
                    if keep is not None:
                        wd_save = s.wd
                        s.wd = np.where(keep, s.wd, 0.0)
                    obs_m = est(s, xm, ym, wr=wr, cells=cl)
                    obs_f = est(s, xf, yf, wr=wr, cells=cl)
                    obs_q = est(s, xm - xf, ym - yf, wr=wr, cells=cl)
                    rng = np.random.default_rng(71)
                    bm, bf, bq = [], [], []
                    for _ in range(a.boot if cls == "KIMB" or tname == "district" else 0):
                        idx = draw_idx(groups, len(s.r), rng)
                        bm.append(est(s, xm, ym, idx, wr, cl))
                        bf.append(est(s, xf, yf, idx, wr, cl))
                        bq.append(est(s, xm - xf, ym - yf, idx, wr, cl))
                    if keep is not None:
                        s.wd = wd_save
                    bm, bf, bq = (np.asarray(v, float) for v in (bm, bf, bq))
                    if not len(bm):
                        continue
                    n_d = int(pd.Series(s.cluster[np.concatenate(groups)]).nunique())
                    base = dict(kind="contrast", dclass=cls, frame=fr, window=w, treatment=tname, fate_def=fdef,
                                n_districts=n_d, n_units=len(groups))
                    rows.append(dict(base, quantity="margins", auc=obs_m, **summarise(bm)))
                    rows.append(dict(base, quantity="failed rifts", auc=obs_f, **summarise(bf)))
                    rows.append(dict(base, quantity="margins minus failed rifts", auc=obs_m - obs_f,
                                     **summarise_diff(bm - bf)))
                    rows.append(dict(base, quantity="margin km minus failed-rift km", auc=obs_q, **summarise(bq)))
                if cls == "KIMB" and fr == "reconstructed" and fdef == "database phases":
                    # leave-one-unit-out (review 2026-11-08): the margin estimate with each group of districts
                    # that share nearest rifts left out in turn, and the effective number of units (the inverse
                    # Herfindahl index of the unit weights), the units being the breakup events as sampled
                    wu = np.asarray([s.wd[g].sum() for g in grp_rift])
                    ess_units = float(wu.sum() ** 2 / (wu ** 2).sum())
                    for k, g in enumerate(grp_rift):
                        keep_idx = np.concatenate([gg for j, gg in enumerate(grp_rift) if j != k])
                        idx = (keep_idx, np.arange(len(s.r)))
                        rows.append(dict(kind="leave one unit out", dclass=cls, frame=fr, window=w, fate_def=fdef,
                                         treatment="shared rift", unit=k, unit_share=float(wu[k] / wu.sum()),
                                         n_districts_left_out=int(pd.Series(s.cluster[g]).nunique()),
                                         quantity="margins", auc=est(s, xm, ym, idx), auc_full=est(s, xm, ym)))
                        rows[-1]["auc_failed"] = est(s, xf, yf, idx)
                    rows.append(dict(kind="leave one unit out", dclass=cls, frame=fr, window=w, fate_def=fdef,
                                     treatment="shared rift", quantity="effective number of units", auc=ess_units,
                                     n_units=n_units, n_districts=int(s.d["cluster_id"].nunique())))
                if cls == "KIMB" and fr == "present" and fdef == "database phases":
                    print(f"  {w}: shared-rift units {n_units} of {s.d['cluster_id'].nunique()} districts")
        print(f"  contrast {cls} {fdef} done", flush=True)

    t = pd.DataFrame(rows)
    # Holm over the five cessation windows within each class, frame and treatment
    t["p_holm"] = np.nan
    m = (t["kind"] == "contrast") & (t["quantity"] == "margins minus failed rifts")
    for _, g in t[m].groupby(["dclass", "frame", "treatment", "fate_def"]):
        t.loc[g.index, "p_holm"] = holm(g["p"].to_numpy())
    for q in ("margins", "failed rifts"):
        m = (t["kind"] == "contrast") & (t["quantity"] == q)
        for _, g in t[m].groupby(["dclass", "frame", "treatment", "fate_def"]):
            t.loc[g.index, "p_holm"] = holm(g["p"].to_numpy())
    rows = t.to_dict("records")

    # ------------------------------------- combined: age uncertainty and shared rifts together
    combos = [("database phases", INFERRED_SD), ("database phases", 5.0), ("database phases", 20.0)] + \
             [(k, INFERRED_SD) for k in DEFS if k != "database phases"]
    for fdef, inf_sd in combos:
        fs = DEFS[fdef]
        for fr in ("present", "reconstructed"):
            s = sides["KIMB"][fr]
            amin = s.d["age_min_ma"].to_numpy(float)
            amax = s.d["age_max_ma"].to_numpy(float)
            inferred = (s.d["age_basis"].astype(str) == "inferred").to_numpy()
            has_rng = np.isfinite(amin) & np.isfinite(amax) & (amax > amin) & ~inferred
            inf_code, inf_vals = pd.factorize(np.where(inferred, s.age_d, np.nan))
            rng = np.random.default_rng(171)
            res = {(w, q): [] for w in [wname(*x) for x in WINDOWS] for q in ("m", "f", "d", "q")}
            bal = []
            ndraw = a.draws if (inf_sd == INFERRED_SD and fdef in ("database phases", "rift systems (100 km)",
                                                                  "interior failed rifts (100 km)")) else max(a.draws // 4, 50)
            for k in range(ndraw):
                ag = np.where(has_rng, rng.uniform(np.where(has_rng, amin, 0), np.where(has_rng, amax, 1)),
                              s.age_d + rng.uniform(-MEASURED_SD, MEASURED_SD, len(s.age_d)))
                if len(inf_vals):
                    # one shared offset per inferred age value; a set with measured ages only
                    # (the Gernon et al. 2023 compilation) has none to draw
                    off = rng.uniform(-inf_sd, inf_sd, len(inf_vals))
                    ag = np.where(inferred & (inf_code >= 0), s.age_d + off[np.maximum(inf_code, 0)], ag)
                en = np.where(e_has, rng.uniform(np.where(e_has, e_lo, 0), np.where(e_has, e_hi, 1)),
                              end0 + rng.uniform(-END_SD, END_SD, len(end0)))
                en = np.where(active_rift, 0.0, np.maximum(en, 0.0))
                on = np.where(o_has, rng.uniform(np.where(o_has, o_lo, 0), np.where(o_has, o_hi, 1)),
                              onset0 + rng.uniform(-END_SD, END_SD, len(onset0)))
                on = np.maximum(on, en)
                wr_k = age_weights(ag, s.wd, s.age_r)
                bal.append(smd(ag, s.wd, s.age_r, np.maximum(wr_k, 1e-12)))
                for lo, hi in WINDOWS:
                    w = wname(lo, hi)
                    (xm, im, _), (ym, _, _) = window_values(s, rifts, fs["margin"], lo, hi, ages_d=ag, end=en, onset=on)
                    (xf, i_f, _), (yf, _, _) = window_values(s, rifts, fs["failed rift"], lo, hi, ages_d=ag, end=en,
                                                              onset=on)
                    groups, _ = rift_groups(s.cluster, [im, i_f])
                    idx = draw_idx(groups, len(s.r), rng)
                    am, af = est(s, xm, ym, idx, wr_k), est(s, xf, yf, idx, wr_k)
                    res[(w, "m")].append(am)
                    res[(w, "f")].append(af)
                    res[(w, "d")].append(am - af)
                    res[(w, "q")].append(est(s, xm - xf, ym - yf, idx, wr_k))
            for lo, hi in WINDOWS:
                w = wname(lo, hi)
                base = dict(kind="combined", dclass="KIMB", frame=fr, window=w, fate_def=fdef,
                            treatment=f"age uncertainty and shared rifts (inferred +/-{int(inf_sd)} Myr)",
                            n_draws=ndraw, age_smd_median=float(np.nanmedian(bal)),
                            age_smd_max=float(np.nanmax(np.abs(bal))))
                for q, lab in (("m", "margins"), ("f", "failed rifts"), ("d", "margins minus failed rifts"),
                               ("q", "margin km minus failed-rift km")):
                    v = np.asarray(res[(w, q)], float)
                    rows.append(dict(base, quantity=lab, auc=float(np.nanmedian(v)),
                                     **(summarise_diff(v) if q == "d" else summarise(v))))
            print(f"  combined {fdef} {fr} inferred +/-{inf_sd} done ({ndraw} draws)", flush=True)
    end_share = float(e_has.mean())
    rows.append(dict(kind="age model", quantity="rifts with an end-age range in the database", auc=end_share))
    rows.append(dict(kind="age model", quantity="rifts with an onset-age range in the database", auc=float(o_has.mean())))
    for fr in ("present",):
        s = sides["KIMB"][fr]
        inferred = (s.d["age_basis"].astype(str) == "inferred").to_numpy()
        rows.append(dict(kind="age model", quantity="kimberlite localities with inferred ages",
                         auc=float(inferred.mean()), n_districts=int(s.d.loc[inferred, "cluster_id"].nunique())))
        rows.append(dict(kind="age model", quantity="distinct inferred age values",
                         auc=float(pd.Series(s.age_d[inferred]).nunique())))
        rows.append(dict(kind="age model", quantity="age balance (standardised difference), recorded ages, unweighted",
                         auc=smd(s.age_d, s.wd, s.age_r, np.ones(len(s.r)))))
        rows.append(dict(kind="age model", quantity="age balance (standardised difference), recorded ages, reweighted",
                         auc=smd(s.age_d, s.wd, s.age_r, np.maximum(s.wr, 1e-12))))
        rows.append(dict(kind="age model", quantity="age balance (standardised difference), locality-weighted kimberlites",
                         auc=smd(s.age_d, np.ones(len(s.d)), s.age_r, np.ones(len(s.r)))))

    # --------------------------------------------------------------- cap and eligibility
    for fr in ("present", "reconstructed"):
        s = sides["KIMB"][fr]
        for fname in FATES:
            for lo, hi in WINDOWS:
                for cap in CAPS:
                    (x, _, hx), (y, _, hy) = window_values(s, rifts, fsel[fname], lo, hi, max_km=cap)
                    both = dict(kind="cap", dclass="KIMB", frame=fr, window=wname(lo, hi), quantity=fname,
                                cap_km=cap if cap < 20000 else np.inf,
                                share_eligible_dep=float((s.wd * hx).sum() / s.wd.sum()),
                                share_eligible_ref=float(hy.mean()))
                    rng = np.random.default_rng(5)
                    b, be = [], []
                    for _ in range(min(a.boot, 1000)):
                        idx = draw_idx(s.groups, len(s.r), rng)
                        b.append(est(s, x, y, idx))
                        xi, yi = np.where(hx, x, np.nan), np.where(hy, y, np.nan)
                        be.append(est(s, xi, yi, idx))
                    rows.append(dict(both, treatment="censored at the range", auc=est(s, x, y), **summarise(b)))
                    rows.append(dict(both, treatment="points with a qualifying rift only",
                                     auc=est(s, np.where(hx, x, np.nan), np.where(hy, y, np.nan)), **summarise(be)))
        print(f"  cap {fr} done", flush=True)

    # ----------------------------------------------------------------- cratonic background
    from rifts.common import lonlat_to_xyz as _xyz
    from scipy.spatial import cKDTree as _Tree
    for fr in ("present", "reconstructed"):
        s = sides["KIMB"][fr]
        vs_q = np.nanquantile(s.d["vsh150"].to_numpy(float), 0.10)
        cen = s.d.groupby("cluster_id")[["lon0", "lat0"]].mean()
        tree = _Tree(_xyz(cen["lon0"], cen["lat0"]))
        near_r = tree.query(_xyz(s.r["lon0"], s.r["lat0"]))[0] <= 2 * np.sin(1000.0 / 6371.0 / 2)
        subsets = {
            "inside a craton outline": (s.d["craton_dist_km"] < 0, s.r["craton_dist_km"] < 0),
            "within 300 km of a craton outline or inside": (s.d["craton_dist_km"] < 300, s.r["craton_dist_km"] < 300),
            f"shear-wave velocity at 150 km >= {vs_q:.2f} km/s": (s.d["vsh150"] >= vs_q, s.r["vsh150"] >= vs_q),
            "inside a craton outline, within 1,000 km of a kimberlite district":
                (s.d["craton_dist_km"] < 0, (s.r["craton_dist_km"] < 0) & near_r),
        }
        for sname, (kd, kr) in subsets.items():
            kd, kr = np.asarray(kd, bool), np.asarray(kr, bool)
            sub = Side(s.d[kd], s.r[kr], s.Dd[kd], s.Dr[kr])
            for fdef in ("database phases", "rift systems (100 km)", "interior failed rifts (100 km)"):
                fs = DEFS[fdef]
                for lo, hi in WINDOWS:
                    w = wname(lo, hi)
                    (xm, im, _), (ym, _, _) = window_values(sub, rifts, fs["margin"], lo, hi)
                    (xf, i_f, _), (yf, _, _) = window_values(sub, rifts, fs["failed rift"], lo, hi)
                    grp_rift, _ = rift_groups(sub.cluster, [im, i_f])
                    for tname, groups in (("district", sub.groups), ("shared rift", grp_rift)):
                        rng = np.random.default_rng(9)
                        bm, bf = [], []
                        for _ in range(a.boot):
                            idx = draw_idx(groups, len(sub.r), rng)
                            bm.append(est(sub, xm, ym, idx))
                            bf.append(est(sub, xf, yf, idx))
                        bm, bf = np.asarray(bm), np.asarray(bf)
                        base = dict(kind="background", dclass="KIMB", frame=fr, window=w, background=sname,
                                    treatment=tname, fate_def=fdef,
                                    n_districts=int(sub.d["cluster_id"].nunique()), n_localities=len(sub.d),
                                    n_ref=len(sub.r), n_units=len(groups),
                                    age_smd_before=smd(sub.age_d, sub.wd, sub.age_r, np.ones(len(sub.r))),
                                    age_smd_after=smd(sub.age_d, sub.wd, sub.age_r, np.maximum(sub.wr, 1e-12)),
                                    craton_smd=smd(sub.d["craton_dist_km"].to_numpy(float), sub.wd,
                                                   sub.r["craton_dist_km"].to_numpy(float), np.maximum(sub.wr, 1e-12)),
                                    trench_smd=smd(sub.d["sz_dist_km"].to_numpy(float), sub.wd,
                                                   sub.r["sz_dist_km"].to_numpy(float), np.maximum(sub.wr, 1e-12)))
                        om, of = est(sub, xm, ym), est(sub, xf, yf)
                        rows.append(dict(base, quantity="margins", auc=om, **summarise(bm)))
                        rows.append(dict(base, quantity="failed rifts", auc=of, **summarise(bf)))
                        rows.append(dict(base, quantity="margins minus failed rifts", auc=om - of, **summarise_diff(bm - bf)))
            print(f"  background {fr} {sname} done", flush=True)

        # combined age uncertainty and shared rifts in the cratonic backgrounds
        for sname in ("inside a craton outline", "within 300 km of a craton outline or inside"):
            kd, kr = (np.asarray(v, bool) for v in subsets[sname])
            sub = Side(s.d[kd], s.r[kr], s.Dd[kd], s.Dr[kr])
            amin, amax = sub.d["age_min_ma"].to_numpy(float), sub.d["age_max_ma"].to_numpy(float)
            inferred = (sub.d["age_basis"].astype(str) == "inferred").to_numpy()
            has_rng = np.isfinite(amin) & np.isfinite(amax) & (amax > amin) & ~inferred
            inf_code, inf_vals = pd.factorize(np.where(inferred, sub.age_d, np.nan))
            for fdef in ("database phases", "rift systems (100 km)"):
                fs = DEFS[fdef]
                rng = np.random.default_rng(271)
                res = {(w, q): [] for w in MAIN for q in ("m", "f", "d")}
                for k in range(a.draws):
                    ag = np.where(has_rng, rng.uniform(np.where(has_rng, amin, 0), np.where(has_rng, amax, 1)),
                                  sub.age_d + rng.uniform(-MEASURED_SD, MEASURED_SD, len(sub.age_d)))
                    if len(inf_vals):
                        off = rng.uniform(-INFERRED_SD, INFERRED_SD, len(inf_vals))
                        ag = np.where(inferred & (inf_code >= 0), sub.age_d + off[np.maximum(inf_code, 0)], ag)
                    en = np.where(e_has, rng.uniform(np.where(e_has, e_lo, 0), np.where(e_has, e_hi, 1)),
                                  end0 + rng.uniform(-END_SD, END_SD, len(end0)))
                    en = np.where(active_rift, 0.0, np.maximum(en, 0.0))
                    on = np.maximum(np.where(o_has, rng.uniform(np.where(o_has, o_lo, 0), np.where(o_has, o_hi, 1)),
                                             onset0 + rng.uniform(-END_SD, END_SD, len(onset0))), en)
                    wr_k = age_weights(ag, sub.wd, sub.age_r)
                    for w in MAIN:
                        lo, hi = (float(v) for v in w.split("_"))
                        (xm, im, _), (ym, _, _) = window_values(sub, rifts, fs["margin"], lo, hi, ages_d=ag, end=en, onset=on)
                        (xf, i_f, _), (yf, _, _) = window_values(sub, rifts, fs["failed rift"], lo, hi, ages_d=ag, end=en,
                                                                  onset=on)
                        groups, _ = rift_groups(sub.cluster, [im, i_f])
                        idx = draw_idx(groups, len(sub.r), rng)
                        am, af = est(sub, xm, ym, idx, wr_k), est(sub, xf, yf, idx, wr_k)
                        res[(w, "m")].append(am)
                        res[(w, "f")].append(af)
                        res[(w, "d")].append(am - af)
                for w in MAIN:
                    base = dict(kind="background combined", dclass="KIMB", frame=fr, window=w, background=sname,
                                fate_def=fdef, treatment="age uncertainty and shared rifts (inferred +/-10 Myr)",
                                n_draws=a.draws, n_districts=int(sub.d["cluster_id"].nunique()), n_ref=len(sub.r))
                    for q, lab in (("m", "margins"), ("f", "failed rifts"), ("d", "margins minus failed rifts")):
                        v = np.asarray(res[(w, q)], float)
                        rows.append(dict(base, quantity=lab, auc=float(np.nanmedian(v)),
                                         **(summarise_diff(v) if q == "d" else summarise(v))))
            print(f"  background combined {fr} {sname} done", flush=True)

    # ------------------------------------------------------------------- matched cells
    for fr in ("present",):
        s = sides["KIMB"][fr]
        cd, cr = cells_for(s)
        keep_w = 0.0
        resid = {c: [] for c in CELL_BY}
        n_cells = 0
        for c in np.unique(cd):
            mx, my = cd == c, cr == c
            if mx.sum() < 3 or my.sum() < 5:
                continue
            n_cells += 1
            keep_w += s.wd[mx].sum()
            for col in CELL_BY:
                resid[col].append((smd(s.d.loc[mx, col].to_numpy(float), s.wd[mx], s.r.loc[my, col].to_numpy(float),
                                       np.ones(my.sum())), s.wd[mx].sum()))
        kept = s.d.loc[np.isin(cd, [c for c in np.unique(cd) if (cd == c).sum() >= 3 and (cr == c).sum() >= 5]),
                       "cluster_id"]
        multi = s.d.assign(cell=cd).groupby("cluster_id")["cell"].nunique()
        rows.append(dict(kind="cells", dclass="KIMB", frame=fr, quantity="cells used", auc=n_cells))
        rows.append(dict(kind="cells", dclass="KIMB", frame=fr, quantity="district weight retained",
                         auc=float(keep_w / s.wd.sum())))
        rows.append(dict(kind="cells", dclass="KIMB", frame=fr, quantity="districts with localities in retained cells",
                         auc=int(kept.nunique())))
        rows.append(dict(kind="cells", dclass="KIMB", frame=fr, quantity="districts spanning more than one cell",
                         auc=int((multi > 1).sum())))
        for col in CELL_BY:
            v = np.array(resid[col], float)
            rows.append(dict(kind="cells", dclass="KIMB", frame=fr, quantity=f"residual standardised difference, {col}",
                             auc=float(np.average(v[:, 0], weights=v[:, 1])) if len(v) else np.nan,
                             lo=float(np.nanmin(v[:, 0])) if len(v) else np.nan,
                             hi=float(np.nanmax(v[:, 0])) if len(v) else np.nan))
            rows.append(dict(kind="cells", dclass="KIMB", frame=fr, quantity=f"standardised difference before matching, {col}",
                             auc=smd(s.d[col].to_numpy(float), s.wd, s.r[col].to_numpy(float), np.ones(len(s.r)))))

    # -------------------------------------------------------- why the frames differ
    pid_v = None
    try:
        from rifts.riftlag import _rift_vertices
        xy, owner = _rift_vertices(rifts)
        pid_v, _, _ = mantle.assign_plate_ids(xy[:, 0], xy[:, 1])
        rift_pid = pd.Series(pid_v).groupby(owner).agg(lambda v: pd.Series(v).mode().iloc[0])
        rift_pid = rift_pid.reindex(range(len(rifts))).to_numpy()
        rift_multi = pd.Series(pid_v).groupby(owner).nunique().reindex(range(len(rifts))).to_numpy()
    except Exception as exc:                                    # plate assignment unavailable
        print(f"  plate IDs of rift vertices not assigned: {exc}")
        rift_pid = np.full(len(rifts), -1)
        rift_multi = np.full(len(rifts), np.nan)
    sp, sr = sides["KIMB"]["present"], sides["KIMB"]["reconstructed"]
    for fname in FATES:
        for lo, hi in WINDOWS:
            w = wname(lo, hi)
            (xp, ip, hp), (yp, jp, gp) = window_values(sp, rifts, fsel[fname], lo, hi)
            (xr, ir, hr), (yr, jr, gr) = window_values(sr, rifts, fsel[fname], lo, hi)
            ones = sp.wr
            dec = {"kimberlites present, reference present": wauc(xp, sp.wd, yp, ones),
                   "kimberlites at eruption, reference present": wauc(xr, sp.wd, yp, ones),
                   "kimberlites present, reference at eruption": wauc(xp, sp.wd, yr, ones),
                   "kimberlites at eruption, reference at eruption": wauc(xr, sp.wd, yr, ones)}
            for k, v in dec.items():
                rows.append(dict(kind="frames", dclass="KIMB", quantity=fname, window=w, treatment=k, auc=v))

            def classify(i0, i1, h0, h1, x0, x1, pid_pt):
                """Category of every point by its nearest qualifying rift at present (0) and at eruption
                (1): the same rift with the distance preserved (within 25 km) or changed, a different
                rift, qualifying in one frame only, or in neither. Also whether the nearest rift at
                present lies on the point's own plate."""
                cat = np.full(len(i0), "no qualifying rift in either frame", object)
                same = (i0 == i1) & h0 & h1
                cat[same & (np.abs(x1 - x0) < 25)] = "same rift, distance preserved"
                cat[same & (np.abs(x1 - x0) >= 25)] = "same rift, distance changed"
                cat[h0 & h1 & (i0 != i1)] = "different nearest rift"
                cat[h0 & ~h1] = "qualifying at present only"
                cat[~h0 & h1] = "qualifying at eruption only"
                onplate = np.where(i0 >= 0, rift_pid[np.maximum(i0, 0)] == pid_pt, False)
                return cat, onplate

            for side, (i0, i1, h0, h1, x0, x1, df, wts) in {
                    "kimberlites": (ip, ir, hp, hr, xp, xr, sp.d, sp.wd),
                    "reference points": (jp, jr, gp, gr, yp, yr, sp.r, sp.wr)}.items():
                cat, onplate = classify(i0, i1, h0, h1, x0, x1, df["pid"].to_numpy())
                for c in np.unique(cat):
                    mm = cat == c
                    rows.append(dict(kind="frame categories", dclass="KIMB", quantity=fname, window=w, treatment=side,
                                     category=c, share=float(wts[mm].sum() / wts.sum()),
                                     median_change_km=float(np.nanmedian((x1 - x0)[mm])),
                                     share_nearest_on_own_plate=float(wts[mm & onplate].sum() / max(wts[mm].sum(), 1e-12))))
            # district contributions to the change (kimberlites moved, reference fixed)
            if w in MAIN:
                ok_ = np.isfinite(yp)
                oo = np.argsort(yp[ok_])
                o, cwr = yp[ok_][oo], np.r_[0.0, np.cumsum(sp.wr[ok_][oo])]

                def F(x):
                    lo_, hi_ = np.searchsorted(o, x, "left"), np.searchsorted(o, x, "right")
                    return (cwr[lo_] + 0.5 * (cwr[hi_] - cwr[lo_])) / cwr[-1]
                cp, cr_ = sp.wd * F(xp) / sp.wd.sum(), sp.wd * F(xr) / sp.wd.sum()
                dd = pd.DataFrame({"cluster": sp.cluster, "cp": cp, "cr": cr_, "xp": xp, "xr": xr, "ip": ip, "ir": ir,
                                   "lon": sp.d["lon0"], "lat": sp.d["lat0"], "age": sp.age_d,
                                   "basis": sp.d["age_basis"].astype(str), "w": sp.wd})
                g = dd.groupby("cluster").agg(cp=("cp", "sum"), cr=("cr", "sum"), lon=("lon", "mean"), lat=("lat", "mean"),
                                              age_min=("age", "min"), age_max=("age", "max"), n=("w", "size"),
                                              basis=("basis", lambda v: "; ".join(sorted(set(v)))),
                                              km_present=("xp", "median"), km_eruption=("xr", "median"),
                                              ip=("ip", lambda v: pd.Series(v).mode().iloc[0]),
                                              ir=("ir", lambda v: pd.Series(v).mode().iloc[0]))
                g["change"] = g["cr"] - g["cp"]
                g = g.reindex(g["change"].abs().sort_values(ascending=False).index)

                def rname(j):
                    return "" if j < 0 else f"{rifts.loc[j, 'RIFTNAME']} ({rifts.loc[j, 'RIFTID']})"
                for cid, r in g.iterrows():
                    rows.append(dict(kind="district change", dclass="KIMB", quantity=fname, window=w, district=str(cid),
                                     n_localities=int(r["n"]), lon=r["lon"], lat=r["lat"], age_min_ma=r["age_min"],
                                     age_max_ma=r["age_max"], age_basis=r["basis"], contribution_present=r["cp"],
                                     contribution_eruption=r["cr"], auc=r["change"], km_present=r["km_present"],
                                     km_eruption=r["km_eruption"], rift_present=rname(int(r["ip"])),
                                     rift_eruption=rname(int(r["ir"]))))

    # ------------------------------------------------ nearest rifts of kimberlite districts
    for fr in ("present", "reconstructed"):
        s = sides["KIMB"][fr]
        for fname in FATES:
            for w in MAIN:
                lo, hi = (float(v) for v in w.split("_"))
                (x, i, h), _ = window_values(s, rifts, fsel[fname], lo, hi)
                dd = pd.DataFrame({"cluster": s.cluster, "rift": i, "w": s.wd})
                dd = dd[dd["rift"] >= 0]
                for j, g in dd.groupby("rift"):
                    rr = rifts.loc[j]
                    rows.append(dict(kind="nearest rifts", dclass="KIMB", frame=fr, quantity=fname, window=w,
                                     rift=str(rr["RIFTID"]), rift_name=str(rr["RIFTNAME"]),
                                     n_districts=int(g["cluster"].nunique()), share=float(g["w"].sum() / s.wd.sum()),
                                     rift_onset_ma=float(rr["onset_ma"]), rift_end_ma=float(rr["end_ma"]),
                                     end_young_ma=rr.get("end_young_ma", np.nan), end_old_ma=rr.get("end_old_ma", np.nan),
                                     end_strat=f"{rr.get('end_strat_young', '')} / {rr.get('end_strat_old', '')}",
                                     feature_type=str(rr["feature_type"]), rift_subtype=str(rr["rift_subtype"]),
                                     confidence=str(rr.get("confidence", "")), rift_plates=rift_multi[j],
                                     system_phase=bool(sys_flag[j])))

    # ------------------------------------------------------------------ fate classification
    for (ft, sub, act), g in rifts.assign(is_active=rifts["is_active"].astype(bool)).groupby(
            ["feature_type", "rift_subtype", "is_active"], dropna=False):
        rows.append(dict(kind="fate mapping", quantity=f"{ft} / {sub} / {'active' if act else 'not active'}",
                         treatment=str(g["fate"].iloc[0]), auc=len(g),
                         n_units=int(g["rift_system"].dropna().nunique()) if "rift_system" in g else np.nan))
    rows.append(dict(kind="fate counts", quantity="failed-rift phases of rift systems that broke up (100 km)",
                     auc=int(sys_flag.sum())))
    young = rifts[rifts["end_ma"] <= a.max_age + 250]
    for f, g in young.groupby("fate"):
        rows.append(dict(kind="fate counts", quantity=f, auc=len(g), n_units=int(g["rift_system"].dropna().nunique()),
                         share=float(g["rift_system"].notna().mean())))

    # ----------------------------------------------------------------------- diamonds
    s = sides["KIMB"]["present"]
    dm = (s.d["subtype"] == "DM").to_numpy()
    ba = (s.d["subtype"] == "NON-DM").to_numpy()
    for col in ("site_crust_km", "vsh150", "craton_dist_km"):
        v = s.d[col].to_numpy(float)
        obs = wauc(v[dm], s.wd[dm], v[ba], s.wd[ba])
        rng = np.random.default_rng(31)
        b = []
        for _ in range(a.boot):
            idx = np.concatenate([s.groups[k] for k in rng.integers(0, len(s.groups), len(s.groups))])
            sel_dm, sel_ba = idx[dm[idx]], idx[ba[idx]]
            b.append(wauc(v[sel_dm], s.wd[sel_dm], v[sel_ba], s.wd[sel_ba]))
        both = set(s.cluster[dm]) & set(s.cluster[ba])
        rows.append(dict(kind="diamond", dclass="KIMB", quantity=col, auc=obs, **summarise(b),
                         n_districts=len(set(s.cluster[dm]) | set(s.cluster[ba])), n_units=len(both),
                         treatment=f"{len(set(s.cluster[dm]))} diamondiferous, {len(set(s.cluster[ba]))} barren districts, "
                                   f"{len(both)} with both; districts resampled once for both sides"))

    # ------------------------------------------------------------------------ sliding
    for fr in ("present", "reconstructed"):
        s = sides["KIMB"][fr]
        for lo, hi in SLIDE:
            (xm, _, _), (ym, _, _) = window_values(s, rifts, fsel["margin"], lo, hi)
            (xf, _, _), (yf, _, _) = window_values(s, rifts, fsel["failed rift"], lo, hi)
            rng = np.random.default_rng(13)
            bm, bf = [], []
            for _ in range(500):
                idx = draw_idx(s.groups, len(s.r), rng)
                bm.append(est(s, xm, ym, idx))
                bf.append(est(s, xf, yf, idx))
            bm, bf = np.asarray(bm), np.asarray(bf)
            om, of = est(s, xm, ym), est(s, xf, yf)
            base = dict(kind="sliding", dclass="KIMB", frame=fr, window=wname(lo, hi))
            rows.append(dict(base, quantity="margins", auc=om, **summarise(bm)))
            rows.append(dict(base, quantity="failed rifts", auc=of, **summarise(bf)))
            rows.append(dict(base, quantity="margins minus failed rifts", auc=om - of, **summarise_diff(bm - bf)))

    out = der / f"kimberlite_fate_contrast_{a.tag}.csv"
    part = out.with_name(out.name + ".partial")
    pd.DataFrame(rows).to_csv(part, index=False, float_format="%.4g")
    part.replace(out)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    show = t[(t["kind"].isin(["contrast", "combined"])) & (t["dclass"] == "KIMB") & t["window"].isin(MAIN)
             & (t["quantity"] == "margins minus failed rifts")]
    print(show[["kind", "fate_def", "frame", "window", "treatment", "auc", "lo", "hi", "p"]].round(3).to_string(index=False))
    print(f"\nwrote {out} ({len(t)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
