"""Present-day geological context of each point: host basin and nearest dated rift.

These joins are made on present-day coordinates. A deposit and the rift or basin that hosts
it sit on the same plate, so their separation is frame-independent.

* Evenick (2021) basins: the polygon containing the point supplies basin type, setting and
  the basin age span.
* GloRiDa (compiled by C. Heine, unpublished): the nearest rift feature within ``max_km`` supplies
  rift onset and end ages, so a deposit age can be placed in the life of its host rift
  (syn-rift, post-rift, pre-rift) and the rift duration becomes a predictor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import chord_to_km, lonlat_to_xyz


BASIN_RENAME = {"Basin Name": "basin_name", "Basin Type": "basin_type", "Setting": "basin_setting",
                "Age Old": "basin_age_old", "Age Young": "basin_age_young", "Basin UBI": "basin_ubi",
                "Salt": "basin_salt", "Volcanics": "basin_volcanics", "Evap Numb": "basin_evap_n",
                "Poly Area": "basin_area_km2", "Mean Sed S": "_sed_straume", "Max Sed St": "_sed_straume_max",
                "Mean Sed L": "_sed_laske", "Max Sed La": "_sed_laske_max", "Moho Mean": "_moho"}
BASIN_NUMERIC = ["basin_sed_km", "basin_sed_max_km", "basin_accum_m_per_myr", "basin_crust_km",
                 "basin_stretch", "basin_area_km2", "basin_age_old", "basin_evap_n"]
UNSTRETCHED_CRUST_KM = 40.0


def basin_measures(b):
    """Fill thickness, lifetime-averaged accumulation rate and crustal thinning of each basin.

    Thickness is the present-day mean over the basin polygon, from Straume et al. (2019) where
    that model covers it and Laske et al. (2013) otherwise; both are smoothed global models.
    The accumulation rate divides it by the age of the basin's oldest sediment, so it is an
    average over the basin's whole life with no correction for compaction or for time not
    represented by rock, and it is a first-order measure of how fast a basin filled, not a
    stratigraphic rate. Crustal thickness is the depth to the Moho (Laske et al., 2013) less the
    fill, and the stretching factor is that against 40 km of unstretched continental crust.
    """
    out = b.copy()
    sed = out["_sed_straume"] if "_sed_straume" in out else None
    laske = out.get("_sed_laske")
    out["basin_sed_km"] = sed.fillna(laske) if sed is not None else laske
    out["basin_sed_source"] = np.where(sed.notna(), "straume2019", "laske2013") if sed is not None else "laske2013"
    if "_sed_straume_max" in out:
        out["basin_sed_max_km"] = out["_sed_straume_max"].fillna(out.get("_sed_laske_max"))
    age = pd.to_numeric(out.get("basin_age_old"), errors="coerce")
    out["basin_accum_m_per_myr"] = np.where(age > 0, 1000.0 * out["basin_sed_km"] / age, np.nan)
    moho = pd.to_numeric(out.get("_moho"), errors="coerce").abs()
    out["basin_crust_km"] = moho - out["basin_sed_km"]
    out["basin_stretch"] = np.where(out["basin_crust_km"] > 1.0,
                                    UNSTRETCHED_CRUST_KM / out["basin_crust_km"], np.nan)
    return out.drop(columns=[c for c in out.columns if c.startswith("_")])


def evenick_join(points: pd.DataFrame, basins_path) -> pd.DataFrame:
    """Host basin of every point, with its fill, accumulation rate, crust and evaporites."""
    import geopandas as gpd
    b = gpd.read_file(basins_path)
    b = b.rename(columns={k: v for k, v in BASIN_RENAME.items() if k in b})
    b = basin_measures(b)
    keep = list(dict.fromkeys(
        c for c in ["basin_ubi", "basin_name", "basin_type", "basin_setting", "basin_age_old",
                    "basin_age_young", "basin_salt", "basin_volcanics", "basin_sed_source"]
        + BASIN_NUMERIC if c in b))
    pts = gpd.GeoDataFrame(points[["pt_id"]].copy(),
                           geometry=gpd.points_from_xy(points["lon"], points["lat"]), crs="EPSG:4326")
    j = gpd.sjoin(pts, b.to_crs("EPSG:4326")[keep + ["geometry"]], how="left", predicate="within")
    j = j.drop_duplicates("pt_id")
    return pd.DataFrame(j[["pt_id"] + keep])


def _densify(geom, step_deg=0.05):
    out = []
    parts = getattr(geom, "geoms", [geom])
    for ln in parts:
        n = max(int(ln.length / step_deg), 1)
        out += [ln.interpolate(i / n, normalized=True).coords[0] for i in range(n + 1)]
    return np.array(out)


_GLORIDA_NAMES = {"riftid": "RIFTID", "riftname": "RIFTNAME", "gpgim_subtype": "GPGIM_SUBT",
                  "gpgim_type": "GPGIM_TYPE", "dataquality": "DATAQUALIT", "synriftvolanics": "SYNRIFTVOL",
                  "riftsystem": "RIFTSYSTEM", "georegion": "GEOREGION", "isactive": "ISACTIVE"}


def read_glorida(path):
    """GloRiDa rift features with cleaned ages.

    ``onset_ma`` = FROMAGE. ``end_ma`` = TOAGE, where the database sentinel -999 (end of rifting
    open or unknown) becomes 0 for rifts flagged active and NaN otherwise. Records whose end
    age exceeds their onset age are age-inconsistent: they keep their geometry but get NaN
    ages (``age_ok`` False) and are listed for the database author.
    """
    import geopandas as gpd
    g = gpd.read_file(path).to_crs("EPSG:4326")
    # v0.9 uses upper-case shapefile-style names, v1.0 lower case; normalise to upper case
    g = g.rename(columns={c: _GLORIDA_NAMES.get(c.lower(), c.upper()) for c in g.columns if c != "geometry"})
    g["GPGIM_TYPE"] = g["GPGIM_TYPE"].astype(str).str.replace("gpml:", "", regex=False)
    g = g[g.geometry.notna()].reset_index(drop=True)
    onset = pd.to_numeric(g["FROMAGE"], errors="coerce")
    end = pd.to_numeric(g["TOAGE"], errors="coerce")
    onset, end = onset.where(onset >= 0), end.where(end >= 0)          # -999 = open / unknown (v0.9)
    act = g["ISACTIVE"]
    active = act.astype(str).str.lower().isin(["1", "true", "1.0"])
    end = end.where(~(end.isna() & active), 0.0)
    bad = (end > onset + 1e-6)
    g["onset_ma"] = onset.where(~bad)
    g["end_ma"] = end.where(~bad)
    g["onset_unc_ma"] = ((g["FROMOLD"] - g["FROMYNG"]) / 2).where(~bad)
    g["age_ok"] = ~bad & g["onset_ma"].notna()
    return g


RIFT_TYPES = ["ContinentalRift", "ContinentalRiftBoundary", "Fault", "PassiveContinentalBoundary"]


def read_outlines(path):
    """Released rift outlines (scripts/06_rift_outlines.py) in the column layout of read_glorida."""
    import geopandas as gpd
    o = gpd.read_file(path, layer="outlines").to_crs("EPSG:4326")
    return o.rename(columns={"outline_id": "RIFTID", "rift_name": "RIFTNAME", "rift_subtype": "GPGIM_SUBT",
                             "data_quality": "DATAQUALIT", "feature_type": "GPGIM_TYPE"})


def glorida_join(points: pd.DataFrame, rifts_path, max_km=150.0, max_lead_myr=300.0, tol_myr=5.0,
                 k=None, source="structures") -> pd.DataFrame:
    """Nearest GloRiDa rift to each point, of any age and age-compatible.

    ``rift_*``      the nearest rift feature within ``max_km`` whatever its age;
    ``arift_*``     the nearest rift within ``max_km`` that was already active when the point
                    formed (onset no younger than the point's age minus ``tol_myr``) and began
                    no more than ``max_lead_myr`` before it. This is the rift that can have
                    hosted or prepared the deposit.
    """
    from scipy.spatial import cKDTree
    g = read_glorida(rifts_path) if source == "structures" else read_outlines(rifts_path)
    g = g[g["GPGIM_TYPE"].isin(RIFT_TYPES)]
    g = g.reset_index(drop=True)
    onset = g["onset_ma"].to_numpy(float)
    end = g["end_ma"].to_numpy(float)
    xy, owner = [], []
    for j, geom in enumerate(g.geometry):
        v = _densify(geom.boundary if geom.geom_type.endswith("Polygon") else geom)
        xy.append(v)
        owner.append(np.full(len(v), j))
    xy = np.vstack(xy)
    owner = np.concatenate(owner)
    if source == "outlines":
        # points inside an outline are at distance 0 from that rift
        import geopandas as gpd
        pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(points["lon"], points["lat"]), crs="EPSG:4326")
        inside = gpd.sjoin(pts, g[["geometry"]], how="inner", predicate="within")
        inside_sets = inside.groupby(level=0)["index_right"].apply(set).to_dict()
    else:
        inside_sets = {}
    tree = cKDTree(lonlat_to_xyz(xy[:, 0], xy[:, 1]))
    q = lonlat_to_xyz(points["lon"], points["lat"])
    # every outline vertex within max_km, nearest first. (A fixed number of nearest vertices,
    # as before, could stop short of max_km where outlines are dense and miss an age-compatible
    # rift behind nearer ones: 5 of 6,057 deposits, all between 127 and 149 km.)
    from .common import km_to_chord
    hits = tree.query_ball_point(q, r=float(km_to_chord(max_km)))
    d, own = [], []
    for r, h in enumerate(hits):
        h = np.asarray(h, int)
        dd = chord_to_km(np.linalg.norm(tree.data[h] - q[r], axis=1)) if len(h) else np.empty(0)
        o = np.argsort(dd, kind="mergesort")
        d.append(dd[o])
        own.append(owner[h[o]])
    ages = points["age_ma"].to_numpy(float)
    n = len(points)
    near_any = np.full(n, -1)
    near_age = np.full(n, -1)
    dist_any = np.full(n, np.nan)
    dist_age = np.full(n, np.nan)
    # How many rifts the point could have been given, and the smallest lag any of them offers.
    # A point with many candidates gets a nearer, and on average younger, rift than one with a
    # single candidate, so the count is the control for the assignment rule itself.
    n_cand = np.zeros(n, int)
    lag_min = np.full(n, np.nan)
    cand_ids = np.full(n, "", dtype=object)
    cand_km = np.full(n, "", dtype=object)
    for r in range(n):
        if r in inside_sets:                        # containing outlines first, at distance 0
            ins = np.array(sorted(inside_sets[r]))
            rest = ~np.isin(own[r], ins)
            own_r = np.concatenate([ins, own[r][rest]])
            d_r = np.concatenate([np.zeros(len(ins)), d[r][rest]])
        else:
            own_r, d_r = own[r], d[r]
        within = d_r <= max_km
        if not within.any():
            continue
        near_any[r], dist_any[r] = own_r[0], d_r[0]
        o = own_r[within]
        with np.errstate(invalid="ignore"):
            ok = (onset[o] >= ages[r] - tol_myr) & (onset[o] <= ages[r] + max_lead_myr)
        if ok.any():
            first = np.argmax(ok)
            near_age[r], dist_age[r] = o[first], d_r[within][first]
            cand = np.unique(o[ok])
            n_cand[r] = len(cand)
            lag_min[r] = np.nanmin(onset[cand] - ages[r])
            # every age-compatible outline within max_km with its distance, not only the nearest:
            # a rift beside a nearer host also satisfies the host rule for this point
            dmin = {}
            for j, dj in zip(o[ok], d_r[within][ok]):
                dmin[j] = min(dmin.get(j, np.inf), dj)
            cand_ids[r] = ";".join(str(g["RIFTID"].iat[j]) for j in dmin)
            cand_km[r] = ";".join(f"{dmin[j]:.1f}" for j in dmin)
    out = {"pt_id": points["pt_id"].to_numpy()}
    for pre, idx, dist in (("rift", near_any, dist_any), ("arift", near_age, dist_age)):
        has = idx >= 0
        pick = lambda arr: np.where(has, np.asarray(arr, object)[np.where(has, idx, 0)], None)  # noqa: E731
        num = lambda arr: np.where(has, np.asarray(arr, float)[np.where(has, idx, 0)], np.nan)  # noqa: E731
        out[f"{pre}_dist_km"] = dist
        out[f"{pre}_id"] = pick(g["RIFTID"])
        out[f"{pre}_name"] = pick(g["RIFTNAME"])
        out[f"{pre}_subtype"] = pick(g["GPGIM_SUBT"])
        out[f"{pre}_quality"] = pick(g["DATAQUALIT"].astype(str).str.upper())
        out[f"{pre}_onset_ma"] = num(onset)
        out[f"{pre}_end_ma"] = num(end)
        out[f"{pre}_duration_myr"] = out[f"{pre}_onset_ma"] - out[f"{pre}_end_ma"]
    out["arift_n_cand"] = n_cand
    out["arift_lag_min_myr"] = lag_min
    out["arift_cand_ids"] = cand_ids
    out["arift_cand_km"] = cand_km
    return pd.DataFrame(out)


CAND_COLS = ["arift_cand_ids", "arift_cand_km"]


def outlines_with_known_deposits(h, classes, host_km=150.0):
    """Outline ids that host a record of ``classes`` under the host rule (within ``host_km`` and
    age-compatible): the nearest such outline of each record (``arift_id``), which is the one a record
    is assigned to, and every other one (``arift_cand_ids``). Rifts without known deposits are the
    outlines outside this set, so that a rift beside a nearer host, or a superbasin around a hosting
    sub-basin, is never taken for a rift in which no deposit is known."""
    if not set(CAND_COLS) <= set(h.columns):
        raise SystemExit("context_full has no arift_cand_ids: rerun scripts/01b_geological_context.py --tag full")
    sel = h[h["dclass"].isin(classes) & h["arift_id"].notna()]
    ids = set(sel.loc[sel["arift_dist_km"] <= host_km, "arift_id"])
    for c, k in zip(sel["arift_cand_ids"], sel["arift_cand_km"]):
        if isinstance(c, str) and c:
            ids |= {i for i, dk in zip(c.split(";"), str(k).split(";")) if float(dk) <= host_km}
    return ids


def rift_stage(age, onset, end, tol=5.0):
    """pre-rift / syn-rift / post-rift label of an age against a rift's active span."""
    age, onset, end = map(np.asarray, (age, onset, end))
    lab = np.full(age.shape, "none", object)
    has = np.isfinite(onset)
    known_end = np.isfinite(end)
    with np.errstate(invalid="ignore"):
        lab[has & (age > onset + tol)] = "pre-rift"
        lab[has & known_end & (age <= onset + tol) & (age >= end - tol)] = "syn-rift"
        lab[has & known_end & (age < end - tol)] = "post-rift"
        lab[has & ~known_end & (age <= onset + tol)] = "after-onset"   # end of rifting unknown
    return lab


# properties of the rift a point sits in, carried onto the point: the deposit-side reading of the
# rift-level comparison. Only points with an age-compatible rift carry them, deposits and nulls
# alike, so these predictors ask what kind of rift a point sits in given that it sits in one.
RIFT_NUMERIC = ["length_rect_km", "width_rect_km", "area_km2", "n_phases", "own_rate", "own_obliq",
                "cob_obliq_deg", "cob_rate_mm_yr",
                "syn_share", "syn_share_excess", "fill_total_m", "fill_wedge_m", "fill_drape_m",
                "beta", "crust_km"]
RIFT_CATEGORICAL = ["setting", "fate", "era", "faultstyle"]


def rift_table(derived, name, tag):
    """Path of a per-rift table for ``tag``. Rift settings and kinematics describe the rifts, not
    the points, so a point set without its own copy (such as the kimberlite set of scripts/72)
    uses the tables of the full run."""
    p = Path(derived) / f"{name}_{tag}.csv"
    return p if p.exists() else Path(derived) / f"{name}_full.csv"


def join_rift_settings(f, derived, tag, verbose=True):
    """Merge each point's host-rift attributes onto it through ``arift_id``.

    Returns the table and the names of the numeric columns added. Columns already present are
    left alone, so the rift duration that comes from the context table is not duplicated.
    """
    import pandas as pd
    path = rift_table(derived, "rift_settings", tag)
    if "arift_id" not in f or not path.exists():
        return f, []
    r = pd.read_csv(path)
    kin = rift_table(derived, "rift_kinematics", tag)
    if kin.exists():
        k = pd.read_csv(kin)
        if "faultstyle" in k:
            r = r.merge(k[["outline_id", "faultstyle"]], on="outline_id", how="left")
        # the Brune segment speaks for a rift only where the rift is the block boundary
        if "cob_dist_km" in k:
            cob = k[k["cob_dist_km"] <= 100.0][["outline_id", "cob_obliq_deg", "cob_rate_mm_yr"]]
            r = r.merge(cob, on="outline_id", how="left")
    fill = Path(derived) / f"rift_fill_{tag}.csv"
    if fill.exists():
        q = pd.read_csv(fill)
        # only rifts whose transects actually fitted a wedge-and-drape shape
        q = q[q["fit_resid_rel"] < 0.5]
        r = r.merge(q[[c for c in q.columns if c == "outline_id" or c in RIFT_NUMERIC]],
                    on="outline_id", how="left")
    num = [c for c in RIFT_NUMERIC if c in r and f"arift_{c}" not in f]
    cols = num + [c for c in RIFT_CATEGORICAL if c in r and f"arift_{c}" not in f]
    r = r[["outline_id"] + cols].rename(columns={c: f"arift_{c}" for c in cols})
    f = f.merge(r, left_on="arift_id", right_on="outline_id", how="left").drop(columns=["outline_id"])
    if verbose and num:
        n = int(f.loc[f["lag_myr"] == 0, f"arift_{num[0]}"].notna().sum())
        print(f"rift attributes joined to {n} points through their age-compatible rift")
    return f, [f"arift_{c}" for c in num]
