"""Comparison statistics between a deposit class and its null set.

Deposits cluster in districts, and a district with fifty prospects is one geological
event, not fifty. Every deposit therefore carries a district weight (1 / district size);
the AUC is district-weighted and its confidence interval comes from resampling whole
districts. Null points are independent draws and carry unit weight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def auc(x, y, wx=None):
    """P(deposit value > null value), ties counted half; 0.5 means no separation.
    ``wx`` weights the deposit values."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    wx = np.ones(len(x)) if wx is None else np.asarray(wx, float)
    ok = np.isfinite(x) & np.isfinite(wx)
    x, wx = x[ok], wx[ok]
    y = y[np.isfinite(y)]
    if len(x) < 3 or len(y) < 3 or wx.sum() <= 0:
        return np.nan
    ys = np.sort(y)
    less = np.searchsorted(ys, x, side="left")
    leq = np.searchsorted(ys, x, side="right")
    frac = (less + 0.5 * (leq - less)) / len(y)
    return float(np.sum(wx * frac) / wx.sum())



def auc_both_weighted(a, b, wa=None, wb=None):
    """P(value from a > value from b), ties half, with district weights on BOTH sides.

    ``auc`` weights only the deposit side, because its reference points are unweighted. When
    two deposit subsets are compared with each other - diamondiferous against barren
    kimberlite, say - both sides are districts and both are weighted.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    wa = np.ones(len(a)) if wa is None else np.asarray(wa, float)
    wb = np.ones(len(b)) if wb is None else np.asarray(wb, float)
    m, n = np.isfinite(a) & np.isfinite(wa), np.isfinite(b) & np.isfinite(wb)
    a, wa, b, wb = a[m], wa[m], b[n], wb[n]
    if len(a) < 3 or len(b) < 3 or wa.sum() <= 0 or wb.sum() <= 0:
        return np.nan
    order = np.argsort(np.concatenate([a, b]), kind="mergesort")
    lab = np.concatenate([np.ones(len(a)), np.zeros(len(b))])[order]
    w = np.concatenate([wa, wb])[order]
    vals = np.concatenate([a, b])[order]
    # weight of b strictly below each position, plus half the weight of b tied with it
    wb_sorted = w * (1 - lab)
    below = np.cumsum(wb_sorted) - wb_sorted
    num = 0.0
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        tie_b = wb_sorted[i:j + 1].sum()
        base = below[i]
        num += float((w[i:j + 1] * lab[i:j + 1]).sum()) * (base + 0.5 * tie_b)
        i = j + 1
    return num / float(wa.sum() * wb.sum())


def boot_auc(x, y, n=500, seed=0):
    """AUC of ``x`` against ``y`` with a percentile interval from resampling both sets.

    For comparisons whose unit is already the independent one — a rift, not a deposit — so
    that no district weighting applies.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 10 or len(y) < 10:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = [auc(rng.choice(x, len(x)), rng.choice(y, len(y))) for _ in range(n)]
    return auc(x, y), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _strata(dep: pd.DataFrame, null: pd.DataFrame, by, n_bins):
    """Cell index of every row of ``dep`` and ``null`` in a grid of percentile bins of ``by``."""
    keys = [by] if isinstance(by, str) else list(by)
    kd = np.zeros(len(dep), int)
    kn = np.zeros(len(null), int)
    for k in keys:
        if k not in dep or k not in null:
            return None, None
        both = np.concatenate([dep[k].to_numpy(float), null[k].to_numpy(float)])
        both = both[np.isfinite(both)]
        # a matching variable can be missing for a whole class - craton-edge distance is
        # withheld before about 1 Ga - and then there is nothing to match on
        if both.size == 0:
            return None, None
        edges = np.unique(np.percentile(both, np.linspace(0, 100, n_bins + 1)))
        if len(edges) < 2:
            return None, None
        edges[0], edges[-1] = -np.inf, np.inf
        nb = len(edges) - 1
        kd = kd * nb + np.clip(np.searchsorted(edges, dep[k].to_numpy(float), "right") - 1, 0, nb - 1)
        kn = kn * nb + np.clip(np.searchsorted(edges, null[k].to_numpy(float), "right") - 1, 0, nb - 1)
    return kd, kn


def _cells(dep: pd.DataFrame, null: pd.DataFrame, col: str, by, n_bins, min_dep, min_null):
    """Per-cell deposit fractions, weights and cell sizes, for the stratified AUC and its
    bootstrap. The fraction of null values a deposit exceeds is computed against the nulls of
    its own cell, so it is the deposit's whole contribution and can be resampled directly."""
    kd, kn = _strata(dep, null, by, n_bins)
    if kd is None:
        return []
    x = dep[col].to_numpy(float)
    w = dep["w_cluster"].to_numpy(float)
    y = null[col].to_numpy(float)
    out = []
    for cell in np.unique(kd):
        di = np.nonzero((kd == cell) & np.isfinite(x) & np.isfinite(w))[0]
        yc = y[(kn == cell) & np.isfinite(y)]
        if len(di) < min_dep or len(yc) < min_null:
            continue
        ys = np.sort(yc)
        less = np.searchsorted(ys, x[di], side="left")
        leq = np.searchsorted(ys, x[di], side="right")
        out.append(((less + 0.5 * (leq - less)) / len(ys), w[di], len(di)))
    return out


def _combine(cells, idx=None):
    """Cell AUCs averaged over cells, weighted by the number of deposits in each."""
    num = den = 0.0
    for k, (f, w, n) in enumerate(cells):
        if idx is not None:
            j = idx[k]
            f, w = f[j], w[j]
        ws = w.sum()
        if ws <= 0:
            continue
        num += n * float(np.sum(w * f) / ws)
        den += n
    return num / den if den else np.nan


def stratified_auc(dep: pd.DataFrame, null: pd.DataFrame, col: str, by="sz_dist_km", n_bins=10,
                   min_dep=3, min_null=5):
    """AUC computed inside cells of ``by`` and averaged over them, weighted by deposit count.

    ``by`` is one column or several, in which case the cells are the grid of their percentile
    bins and the comparison is matched on all of them at once. Water and carbon delivery both
    fall off away from subduction zones, so a raw AUC for them largely restates distance to the
    trench; and a rift that is large and old is both likelier to hold ore and likelier to have
    been looked at. Holding those fixed shows what the measure adds.
    """
    cells = _cells(dep, null, col, by, n_bins, min_dep, min_null)
    return _combine(cells) if cells else np.nan


def strata_coverage(dep, null, col, by, n_bins=3, min_dep=3, min_null=5):
    """How much of the matched comparison actually happened: the number of cells that held
    enough of both sides, and the share of the deposit side inside them. A matched AUC resting
    on one cell, or on a third of the deposits, is not a matched comparison."""
    cells = _cells(dep, null, col, by, n_bins, min_dep, min_null)
    used = sum(n for _, _, n in cells)
    have = int(dep[col].notna().sum())
    return len(cells), (used / have if have else np.nan)


def boot_stratified_auc(dep, null, col, by, n_bins=3, n=500, seed=0, min_dep=3, min_null=5):
    """``stratified_auc`` with a percentile interval from resampling deposits within their own
    cells, which holds the matching fixed and varies only what it is measured on."""
    cells = _cells(dep, null, col, by, n_bins, min_dep, min_null)
    if not cells:
        return np.nan, np.nan, np.nan
    a = _combine(cells)
    rng = np.random.default_rng(seed)
    b = [_combine(cells, [rng.integers(0, len(f), len(f)) for f, _, _ in cells]) for _ in range(n)]
    b = np.asarray([v for v in b if np.isfinite(v)])
    if len(b) < 20:
        return a, np.nan, np.nan
    return a, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def compare(dep: pd.DataFrame, null: pd.Series, col: str, n_boot=200, seed=0, min_districts=5,
            return_boot=False):
    """``dep`` holds ``col``, ``cluster_id`` and ``w_cluster``.

    With ``return_boot`` the district-bootstrap distribution of the AUC is returned under
    ``boot``, so that a test of AUC = 0.5 uses the same resampling as the reported interval.
    """
    d = dep[[col, "cluster_id", "w_cluster"]].dropna(subset=[col])
    y = null.dropna().to_numpy(float)
    n_d = d["cluster_id"].nunique()
    if n_d < min_districts or len(y) < 20:
        return dict(n_dep=len(d), n_districts=n_d, n_null=len(y))
    x, w = d[col].to_numpy(float), d["w_cluster"].to_numpy(float)
    rng = np.random.default_rng(seed)
    a = auc(x, y, w)
    groups = [g.to_numpy() for _, g in d.groupby("cluster_id").indices.items()] \
        if False else list(d.groupby("cluster_id").indices.values())
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        boots.append(auc(x[idx], rng.choice(y, len(y)), w[idx]))
    ks = stats.ks_2samp(x, y)
    # Delivery measures are zero for most points far from subduction: a rank test on such a
    # variable is mostly a test of how often the value is non-zero, so both are reported.
    zx, zy = float(np.mean(x <= 0)), float(np.mean(y <= 0))
    nz = dict(zero_dep=zx, zero_null=zy, tie_frac=max(zx, zy))
    if min(zx, zy) < 1 and (x > 0).sum() >= 5 and (y > 0).sum() >= 5:
        nz["auc_nonzero"] = auc(x[x > 0], y[y > 0], w[x > 0])
    order = np.argsort(x)
    cw = np.cumsum(w[order]) / w.sum()
    wmed = x[order][np.searchsorted(cw, 0.5)]
    return dict(n_dep=len(d), n_districts=n_d, n_null=len(y), med_dep=np.median(x), wmed_dep=wmed,
                med_null=np.median(y), ks_D=ks.statistic, ks_p=ks.pvalue, auc=a,
                auc_lo=np.nanpercentile(boots, 2.5), auc_hi=np.nanpercentile(boots, 97.5), **nz,
                **({"boot": np.asarray(boots, float)} if return_boot else {}))


# How much exploration a point has had is measured by ``other_family_km``, the distance to the
# nearest deposit of another family (scripts/17_null_masks.py). Conditioning on it is safer than
# masking on it: a hard radius moves the null set onto delivery-rich ground, because deposits of
# every family cluster where slab-derived water and carbon are high, whereas conditioning
# compares deposits with nulls that have had the same amount of exploration and keeps the whole
# reference set. Exploration density is downstream of both the ground's prospectivity and the
# effort spent on it, so conditioning on it can only remove separation, never create it: what
# survives is safe.
EXPLORED = "other_family_km"

# Lithospheric architecture is the standing alternative explanation: deposits sit at craton
# edges, and a rift follows the edge it nucleated on. Conditioning on the distance to the
# nearest craton edge at the age of the point asks what a predictor adds beyond that
# architecture, which is the test Glerum et al. (2026) call for.
CRATON = "craton_abs_km"


def screen(features: pd.DataFrame, classes, predictors, lag=0, cls_col="dclass", null_label="NULL",
           in_model_only=True, min_districts=5):
    """District-weighted AUC screen of every predictor for every class at one lag.

    ``features`` holds ``dclass`` (class or NULL), ``null_for``, ``lag_myr``, ``in_model``,
    ``cluster_id``, ``w_cluster`` and the predictors. A finding counts as robust when its
    district-bootstrap interval excludes 0.5 (``robust``), and is marked ``tie_limited`` when
    more than 90 % of the values on one side are zero, in which case ``zero_dep``,
    ``zero_null`` and ``auc_nonzero`` carry the information instead.
    """
    f = features[features["lag_myr"] == lag]
    if in_model_only:
        f = f[f["in_model"]]
    rows = []
    for c in classes:
        d = f[f[cls_col] == c]
        n = f[(f[cls_col] == null_label) & (f["null_for"] == c)]
        if d["cluster_id"].nunique() < min_districts:
            continue
        for p in predictors:
            r = compare(d, n[p], p, min_districts=min_districts)
            if r.get("n_districts", 0) >= min_districts:
                r["auc_given_trench_distance"] = stratified_auc(d, n, p)
                if CRATON in d and CRATON in n:
                    r["auc_given_craton"] = stratified_auc(d, n, p, by=CRATON)
                    r["craton_cells"], r["craton_matched"] = strata_coverage(d, n, p, by=CRATON)
                if EXPLORED in d and EXPLORED in n:
                    r["auc_given_exploration"] = stratified_auc(d, n, p, by=EXPLORED)
                    r["auc_given_both"] = stratified_auc(d, n, p, by=("sz_dist_km", EXPLORED),
                                                         n_bins=4)
                    r["both_cells"], r["both_matched"] = strata_coverage(
                        d, n, p, by=("sz_dist_km", EXPLORED), n_bins=4)
            rows.append(dict(dclass=c, predictor=p, lag_myr=lag, **r))
    out = pd.DataFrame(rows)
    if "auc" in out:
        out["effect"] = (out["auc"] - 0.5).abs()
        out["robust"] = (out["auc_lo"] > 0.5) | (out["auc_hi"] < 0.5)
        # a result carried by ties alone (both sides almost all zero) is separated out
        out["tie_limited"] = out["tie_frac"] > 0.9
    return out


def impute_median_fold(X, tr, te):
    """Fill the gaps of a feature matrix with the column medians of the training rows ``tr`` of one
    fold, applied to the training and the held-out rows ``te`` alike, so that no value of a held-out
    row reaches training. A column with no value among the training rows is set to 0, a constant the
    forests cannot split on. Returns (X[tr], X[te])."""
    import warnings
    X = np.asarray(X, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN columns
        med = np.nanmedian(X[tr], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isfinite(X[tr]), X[tr], med), np.where(np.isfinite(X[te]), X[te], med)
