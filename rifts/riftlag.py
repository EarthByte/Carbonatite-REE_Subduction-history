"""Distance to a preceding rift, resolved by how long before the point that rift was active.

The rift join used elsewhere asks which rift hosts a point, and answers within 150 km. The
question here is different and is asked at continental range: kimberlites are proposed to
erupt some tens of millions of years after rifting, hundreds to thousands of kilometres
inboard of it, as a wave of convective instability travels along the base of the keel. A
measure built for hosting cannot see that, because the mechanism puts the eruption far
outside any radius a host would occupy.

For every point this returns, for each lag window, the distance to the nearest rift that
ceased being active that long before the point formed, together with the migration speed
that distance and lag imply. A reference point drawn on continental crust of the same age
gets the same treatment, so the question becomes whether deposits sit closer to a rift of
a given antiquity than the crust they are compared with, and at which lag the difference
is largest.

Distances are measured in present-day coordinates by default. Within a continent that is the
frame the mechanism is stated in - migration is inboard from the rifted margin, with the
craton - and it avoids attributing to the reconstruction any error in the relative motion of
two points on the same plate. With ``reconstructed`` the rift outlines are rotated, vertex by
vertex with the plate beneath each vertex, to the age of every point and the distance is
measured to the point's reconstructed position, which is the eruption-time geometry across
plate boundaries and conjugate margins.

The distance from a point to a rift is the great-circle distance to the nearest vertex of the
rift outline (densified every 0.25 degrees), computed for every point and every rift, so the
nearest qualifying rift is exact. A point with no qualifying rift within ``max_km`` is left
empty by default; with ``censor`` it takes ``max_km`` itself, so that it stays in the
comparison as "at least this far" (ties are counted half by the AUC).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import chord_to_km, lonlat_to_xyz

# lag windows in Myr between the end of rifting and the formation of the point; the third
# straddles the 26 +/- 4 Myr lag reported for kimberlite eruption after breakup
LAG_BINS = ((0.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 70.0), (70.0, 120.0), (120.0, 250.0))


def _vertices(geom, step_deg=0.25):
    """Vertices along a geometry, densified so that a long segment is not under-sampled."""
    from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

    if isinstance(geom, (Polygon, MultiPolygon)):
        geom = geom.boundary
    parts = geom.geoms if isinstance(geom, (MultiLineString,)) else [geom]
    out = []
    for p in parts:
        if not isinstance(p, LineString) or p.is_empty:
            continue
        n = max(int(p.length / step_deg), 1)
        out.append(np.array([p.interpolate(t, normalized=True).coords[0]
                             for t in np.linspace(0, 1, n + 1)]))
    return np.vstack(out) if out else np.empty((0, 2))


def _rift_vertices(rifts, step_deg=0.25):
    """Densified outline vertices of every rift, sorted by rift, with the start index of each."""
    xy, owner = [], []
    for j, geom in enumerate(rifts.geometry):
        v = _vertices(geom, step_deg)
        if len(v):
            xy.append(v)
            owner.append(np.full(len(v), j))
    if not xy:
        raise ValueError("no rift geometry to measure against")
    xy, owner = np.vstack(xy), np.concatenate(owner)
    return xy, owner


def _min_by_rift(pxyz, vxyz, owner, n_rifts, chunk=256):
    """Great-circle distance (km) from every point to the nearest vertex of every rift."""
    from .common import EARTH_RADIUS_KM
    starts = np.r_[0, np.nonzero(np.diff(owner))[0] + 1]
    present = owner[starts]
    out = np.full((len(pxyz), n_rifts), np.inf)
    good = np.isfinite(vxyz).all(axis=1)
    for a in range(0, len(pxyz), chunk):
        p = pxyz[a:a + chunk]
        dots = np.clip(p @ np.where(good[:, None], vxyz, 0.0).T, -1.0, 1.0)
        dist = np.arccos(dots) * EARTH_RADIUS_KM
        dist[:, ~good] = np.inf
        out[a:a + chunk][:, present] = np.minimum.reduceat(dist, starts, axis=1)
    out[~np.isfinite(pxyz).all(axis=1)] = np.nan
    return out


def rift_distances(points, rifts, step_deg=0.25, reconstructed=False, model=None, age_step=1.0):
    """Distance matrix (points x rifts, km).

    Present-day mode measures from ``lon``/``lat`` to the present-day outlines. Reconstructed
    mode needs ``mlon``/``mlat`` (the point at its own age) and a plate model: the outline
    vertices are given the plate beneath each of them and rotated to each point's age, rounded
    to ``age_step`` Myr.
    """
    xy, owner = _rift_vertices(rifts, step_deg)
    n = len(rifts)
    if not reconstructed:
        return _min_by_rift(lonlat_to_xyz(points["lon"], points["lat"]), lonlat_to_xyz(xy[:, 0], xy[:, 1]), owner, n)
    if model is None:
        raise ValueError("reconstructed distances need a plate model")
    pid, begin, _ = model.assign_plate_ids(xy[:, 0], xy[:, 1])
    ages = np.round(points["age_ma"].to_numpy(float) / age_step) * age_step
    pxyz = lonlat_to_xyz(points["mlon"], points["mlat"])
    out = np.full((len(points), n), np.nan)
    for t in np.unique(ages[np.isfinite(ages)]):
        rows = np.nonzero(ages == t)[0]
        rlon, rlat = model.reconstruct_points(xy[:, 0], xy[:, 1], pid, float(t))
        rlon, rlat = np.asarray(rlon, float), np.asarray(rlat, float)
        gone = (pid <= 0) | (begin < t)                   # vertex not on a plate that exists at t
        rlon[gone], rlat[gone] = np.nan, np.nan
        out[rows] = _min_by_rift(pxyz[rows], lonlat_to_xyz(rlon, rlat), owner, n)
    return out


def lag_measures(D, ages, onset, end, max_km=3000.0, lag_bins=LAG_BINS, censor=False, prefix="rlag"):
    """Per point and lag window: distance to (and index of) the nearest rift that stopped rifting
    that long before the point, from a distance matrix; plus the nearest rift active at the
    point's age."""
    n = D.shape[0]
    lag = end[None, :] - ages[:, None]                  # positive: rifting stopped before the point
    near = np.isfinite(D) & (D <= max_km)
    cols, ids = {}, {}
    wins = [(f"{int(lo)}_{int(hi)}", (lag > lo) & (lag <= hi)) for lo, hi in lag_bins]
    wins.append(("active", (onset[None, :] >= ages[:, None]) & (end[None, :] <= ages[:, None])))
    for name, m in wins:
        ok = m & near
        dm = np.where(ok, D, np.inf)
        j = np.argmin(dm, axis=1)
        dist = dm[np.arange(n), j]
        has = np.isfinite(dist)
        val = np.where(has, dist, max_km if censor else np.nan)
        val[~np.isfinite(ages)] = np.nan
        cols[f"{prefix}_{name}_km"] = val
        cols[f"{prefix}_{name}_censored"] = ~has & np.isfinite(ages)
        ids[f"{prefix}_{name}_rift"] = np.where(has, j, -1)
    return cols, ids


def rift_lag_table(points: pd.DataFrame, rifts, max_km=3000.0, lag_bins=LAG_BINS,
                   step_deg=0.25, censor=False, reconstructed=False, model=None,
                   with_ids=False) -> pd.DataFrame:
    """One row per point: nearest preceding rift in each lag window.

    ``points`` needs ``pt_id``, ``lon``, ``lat`` and ``age_ma`` (and ``mlon``/``mlat`` when
    ``reconstructed``); ``rifts`` is a GeoDataFrame with ``onset_ma`` and ``end_ma`` in Ma. A
    rift counts as preceding a point when it stopped being active before the point formed,
    and the lag is that interval. Rifts still active at the point's age are reported as
    ``rlag_active_km``. With ``with_ids`` the outline identifier of each nearest rift is added.
    """
    rifts = rifts.reset_index(drop=True)
    D = rift_distances(points, rifts, step_deg, reconstructed=reconstructed, model=model)
    ages = points["age_ma"].to_numpy(float)
    cols, ids = lag_measures(D, ages, rifts["onset_ma"].to_numpy(float), rifts["end_ma"].to_numpy(float),
                             max_km=max_km, lag_bins=lag_bins, censor=censor)
    out = pd.DataFrame({"pt_id": points["pt_id"].to_numpy(), **cols})
    for lo, hi in lag_bins:
        key = f"{int(lo)}_{int(hi)}"
        lagv = np.full(len(out), np.nan)
        j = ids[f"rlag_{key}_rift"]
        ok = j >= 0
        lagv[ok] = rifts["end_ma"].to_numpy(float)[j[ok]] - ages[ok]
        out[f"rvel_{key}_km_myr"] = out[f"rlag_{key}_km"] / np.maximum(lagv, 1e-6)
    if with_ids:
        rid = rifts["RIFTID"].to_numpy() if "RIFTID" in rifts else rifts.index.to_numpy()
        for k, j in ids.items():
            out[k] = np.where(j >= 0, rid[np.maximum(j, 0)], None)
    return out
