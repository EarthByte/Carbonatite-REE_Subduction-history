"""The screening frame: every measure attached to every deposit and null point.

One function builds it, so that the screen and everything measured on subsets of it - eras,
directly dated deposits, a second control class - see exactly the same columns.
"""
from __future__ import annotations

from .common import read_table
from .context import BASIN_NUMERIC, join_rift_settings

PREDICTORS = [
    "plate_speed_cm_yr", "abs_paleolat",
    "div_dist_km", "div_conv_orth", "div_obliq",
    "divcont_dist_km", "divcont_conv_orth", "divcont_obliq",
    "conv_dist_km", "conv_conv_orth", "sz_dist_km", "sz_conv_orth", "sz_bnd_orth",
    "mor_dist_km", "divcont_len_500km", "div_rate_1000km", "sz_len_1000km",
]

CONTEXT_COLS = ("rift_dist_km", "arift_dist_km", "arift_duration_myr",
                "arift_onset_minus_age_myr")

# columns that ride along with a joined table but say something about the measurement rather
# than about the place, so they are never screened as predictors
DIAGNOSTIC = {"craton_cover"}


NULL_LABEL = "NULL"


def at_lag(frame, lag=0, in_model_only=True):
    """One row per point: the screening frame carries twenty lags for every point."""
    f = frame[frame["lag_myr"] == lag]
    return f[f["in_model"]] if in_model_only and "in_model" in f else f


def within_age(frame, max_age_ma=None):
    """Points that formed within the past ``max_age_ma``; deposits and reference points alike."""
    return frame if max_age_ma is None else frame[frame["age_ma"] <= float(max_age_ma)]


def deposit_rows(frame, dclass=None, lag=0, in_model_only=True):
    """Deposits, optionally of one class.

    Note that ``null_for`` is an empty string on a deposit row rather than a missing value,
    so testing it with ``isna`` selects nothing. The class column is what separates the two
    sides: a reference point carries ``dclass == "NULL"``.
    """
    f = at_lag(frame, lag, in_model_only)
    f = f[f["dclass"] != NULL_LABEL]
    return f if dclass is None else f[f["dclass"] == dclass]


def reference_rows(frame, dclass, lag=0, in_model_only=True, include_ice=False):
    """The age-matched reference points drawn for one class, without those beneath present-day
    ice (flag ``under_ice`` from scripts/17, joined by ``load_screen_frame``): no deposit under
    an ice sheet could have entered a compilation, so such a point is no fair reference."""
    f = at_lag(frame, lag, in_model_only)
    f = f[(f["dclass"] == NULL_LABEL) & (f["null_for"] == dclass)]
    if not include_ice and "under_ice" in f:
        f = f[~f["under_ice"].fillna(False).astype(bool)]
    return f


def deposit_table(derived, tag):
    """The deposit table behind a point set: deposits_<tag> where a point set has its own (the
    kimberlite set of scripts/72), otherwise the shared compilation."""
    from pathlib import Path
    from .common import read_table
    for name in (f"deposits_{tag}", "deposits"):
        try:
            return read_table(Path(derived) / name)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"no deposit table in {derived}")


def load_screen_frame(cfg, tag, prefilter=None, verbose=True):
    """Return ``(frame, predictors, has_context)``.

    ``prefilter`` is applied to the feature table before anything is joined to it, so that a
    run on a subset of the points pays for that subset only.
    """
    import pandas as pd

    der = cfg["paths"]["derived"]
    f = read_table(der / f"features_{tag}")
    f["abs_paleolat"] = f["paleolat"].abs()
    f = f[f["draw"] == 0]
    if prefilter is not None:
        f = prefilter(f)
    if "cluster_id" not in f:          # tables written before district weights existed
        dep = deposit_table(der, tag)[["dep_id", "cluster_id", "w_cluster"]]
        f = f.merge(dep.rename(columns={"dep_id": "pt_id"}), on="pt_id", how="left")
        f["cluster_id"] = f["cluster_id"].fillna(f["pt_id"])
        f["w_cluster"] = f["w_cluster"].fillna(1.0)
    preds = [p for p in PREDICTORS if p in f]

    try:
        w = read_table(der / f"water_{tag}")
        f = f.merge(w, on="pt_id", how="left")
        preds += [c for c in w.columns if c != "pt_id"]
    except FileNotFoundError:
        pass
    try:
        # subducted carbon and water held in a diffusing mantle reservoir (scripts/90); compared,
        # not added to the screening predictors
        f = f.merge(read_table(der / f"mantle_memory_{tag}"), on="pt_id", how="left")
    except FileNotFoundError:
        pass
    try:
        # mantle-driven dynamic topography at formation (scripts/88 --source features)
        f = f.merge(read_table(der / f"dynamic_topography_{tag}"), on="pt_id", how="left")
    except FileNotFoundError:
        pass
    for name, note in ((f"point_fill_{tag}.csv", "site fill and crust"),
                       (f"craton_{tag}.csv", "craton-edge distance")):
        path = der / name
        if path.exists():
            q = pd.read_csv(path)
            f = f.merge(q, on="pt_id", how="left")
            preds += [c for c in q.columns if c != "pt_id" and c not in DIAGNOSTIC]
            if verbose:
                n = int(q[q.columns[1]].notna().sum())
                print(f"{note} joined to {n} points")
    mask = der / f"null_mask_{tag}.csv"
    if mask.exists():
        m = pd.read_csv(mask, usecols=["pt_id", "ice"]).rename(columns={"ice": "under_ice"})
        f = f.merge(m, on="pt_id", how="left")
        if verbose:
            print(f"present-day ice flag joined ({int(m['under_ice'].sum())} points beneath ice)")
    has_context = False
    try:
        ctx = read_table(der / f"context_{tag}")
        f = f.merge(ctx, on="pt_id", how="left")
        preds += [c for c in CONTEXT_COLS + tuple(BASIN_NUMERIC) if c in ctx]
        f, rift_preds = join_rift_settings(f, der, tag)
        preds += rift_preds
        has_context = True
    except FileNotFoundError:
        pass
    return f, list(dict.fromkeys(preds)), has_context
