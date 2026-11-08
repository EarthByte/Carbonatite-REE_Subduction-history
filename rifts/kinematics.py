"""Plate-boundary kinematics sampled at reconstructed deposit positions.

Boundaries are classified by what the plates do, not only by the feature label: before
410 Ma the 1.8 Ga model carries only ridge, subduction and transform labels, so a
continental rift exists there only as divergence between two plates on continental crust.
Every sampled boundary point carries

    conv_orth   orthogonal convergence rate, cm/yr (positive converging, negative opening)
    conv_par    boundary-parallel relative motion, cm/yr
    obliq       obliquity of relative motion to the boundary normal, degrees (0 = orthogonal)
    bnd_orth    orthogonal velocity of the boundary itself in the anchor frame, cm/yr
                (trench migration for subduction zones)
    ftype       GPlates feature type
    cont_km     smallest probe offset (km) at which both sides of the boundary lie on
                continental crust; NaN if none. Before 1 Ga the model draws plate boundaries
                offshore of the continental polygons, so a single short probe misses rifted
                margins; several offsets are therefore tested.
    cont        cont_km <= the configured margin distance
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pygplates
from scipy.spatial import cKDTree

from .common import EARTH_RADIUS_KM, chord_to_km, km_to_chord, lonlat_to_xyz


@dataclass
class BoundarySnapshot:
    time: float
    table: pd.DataFrame
    xyz: np.ndarray

    def subset_tree(self, mask):
        idx = np.nonzero(mask)[0]
        return (cKDTree(self.xyz[idx]) if len(idx) else None), idx


def _offset(xyz, normal_xyz, km):
    a = km / EARTH_RADIUS_KM
    return xyz * np.cos(a) + normal_xyz * np.sin(a)


def _pid(loc):
    """Plate ID of a TopologyPointLocation (rigid plate or network rigid block)."""
    if not loc:
        return -1
    topo = loc.located_in_resolved_boundary() or loc.located_in_resolved_network()
    return topo.get_feature().get_reconstruction_plate_id() if topo else -1


def boundary_snapshot(model, time, spacing_deg=0.25, probe_km=(50, 150, 300), delta=1.0,
                      margin_km=300) -> BoundarySnapshot:
    """Plate-boundary points at ``time``, every ``spacing_deg``: position, normal, orthogonal and parallel
    convergence (cm/yr) and obliquity, boundary-normal velocity, segment length, feature type, left and
    right plates, and for subduction zones the overriding and subducting plates. cont_km is the smallest
    of ``probe_km`` at which both sides are continental, and cont whether that is within ``margin_km``."""
    snap = model.topological_snapshot(time)
    stats = snap.calculate_plate_boundary_statistics(
        np.radians(spacing_deg), velocity_delta_time=delta,
        velocity_units=pygplates.VelocityUnits.cms_per_yr)
    rows = []
    sub_cache = {}
    for s in stats:
        over, subd = -1, -1
        if s.boundary_feature.get_feature_type().get_name() == "SubductionZone":
            key = id(s.shared_sub_segment)
            if key not in sub_cache:
                try:
                    r = s.shared_sub_segment.get_overriding_and_subducting_plates()
                except Exception:
                    r = None
                sub_cache[key] = ((r[0].get_feature().get_reconstruction_plate_id(),
                                   r[1].get_feature().get_reconstruction_plate_id()) if r else (-1, -1))
            over, subd = sub_cache[key]
        lat, lon = s.boundary_point.to_lat_lon()
        n = s.boundary_normal.to_xyz()
        lp, rp = _pid(s.left_plate), _pid(s.right_plate)
        rows.append((lon, lat, n[0], n[1], n[2],
                     s.convergence_velocity_orthogonal, s.convergence_velocity_parallel,
                     np.degrees(s.convergence_velocity_obliquity) if s.convergence_velocity else np.nan,
                     s.boundary_velocity_orthogonal,
                     s.boundary_length * EARTH_RADIUS_KM,
                     s.boundary_feature.get_feature_type().get_name(), lp, rp, over, subd))
    t = pd.DataFrame(rows, columns=["lon", "lat", "nx", "ny", "nz", "conv_orth", "conv_par", "obliq",
                                    "bnd_orth", "seg_km", "ftype", "left_pid", "right_pid",
                                    "over_pid", "sub_pid"])
    # pygplates convention: the orthogonal component is along the boundary normal and its
    # sign flips with segment direction. Re-sign so that positive always means converging.
    sm = np.array([s.convergence_velocity_signed_magnitude for s in stats])
    t["conv_orth"] = np.sign(sm) * np.abs(t["conv_orth"])
    a = np.abs(t["obliq"])
    t["obliq"] = np.where(a > 90, 180 - a, a)
    xyz = lonlat_to_xyz(t["lon"], t["lat"])
    t["cont_km"] = _continental_both_sides(model, time, xyz, t[["nx", "ny", "nz"]].to_numpy(),
                                           np.atleast_1d(probe_km))
    t["cont"] = t["cont_km"] <= margin_km
    return BoundarySnapshot(float(time), t, xyz)


def _continental_both_sides(model, time, xyz, nrm, probes_km):
    """For each boundary point, the smallest of ``probes_km`` at which the points offset to both sides
    along the boundary normal lie in continental polygons; NaN where no probe distance does."""
    out = np.full(len(xyz), np.nan)
    if not len(xyz):
        return out
    part = pygplates.PlatePartitioner(model.continental_polygons, model.rotation_model,
                                      reconstruction_time=float(time))
    for km in sorted(probes_km):
        both = np.isnan(out)
        for sign in (1.0, -1.0):
            p = _offset(xyz, nrm * sign, km)
            p /= np.linalg.norm(p, axis=1, keepdims=True)
            for i in np.nonzero(both)[0]:
                both[i] = part.partition_point(pygplates.PointOnSphere(*p[i])) is not None
        out[both] = km
    return out


CATEGORIES = {
    "div":      lambda t, thr: t["conv_orth"] < -thr["div"],
    "divcont":  lambda t, thr: (t["conv_orth"] < -thr["div"]) & t["cont"],
    "conv":     lambda t, thr: t["conv_orth"] > thr["conv"],
    "sz":       lambda t, thr: t["ftype"].eq("SubductionZone") & (t["conv_orth"] > 0),
    "mor":      lambda t, thr: t["ftype"].eq("MidOceanRidge") & ~t["cont"],
    # transform-like: strike-slip dominated relative motion faster than the convergence threshold
    "trf":      lambda t, thr: (t["conv_par"].abs() > 2 * t["conv_orth"].abs())
                               & (t["conv_par"].abs() > thr["conv"]),
    "any":      lambda t, thr: np.isfinite(t["conv_orth"]),
}


def sample_boundaries(bs: BoundarySnapshot, lon, lat, thresholds, radii_km):
    """Nearest-boundary and neighbourhood metrics for points already at time ``bs.time``."""
    out = {}
    ok = np.isfinite(lon) & np.isfinite(lat)
    q = lonlat_to_xyz(np.where(ok, lon, 0), np.where(ok, lat, 0))
    t = bs.table
    for cat, rule in CATEGORIES.items():
        tree, idx = bs.subset_tree(rule(t, thresholds).to_numpy())
        dist = np.full(len(q), np.nan)
        near = np.full(len(q), -1)
        if tree is not None:
            d, j = tree.query(q)
            dist, near = chord_to_km(d), idx[j]
        dist[~ok] = np.nan
        sel = near >= 0
        out[f"{cat}_dist_km"] = dist
        for col in ("conv_orth", "conv_par", "obliq", "bnd_orth", "cont_km"):
            v = np.full(len(q), np.nan)
            v[sel & ok] = t[col].to_numpy()[near[sel & ok]]
            out[f"{cat}_{col}"] = v
        if cat in ("div", "divcont", "conv", "sz"):
            seg = t["seg_km"].to_numpy()
            rate = np.abs(t["conv_orth"].to_numpy())
            for r in radii_km:
                length = np.zeros(len(q))
                wrate = np.full(len(q), np.nan)
                if tree is not None:
                    hits = tree.query_ball_point(q, km_to_chord(r))
                    for i, h in enumerate(hits):
                        if ok[i] and h:
                            k = idx[h]
                            length[i] = seg[k].sum()
                            wrate[i] = np.average(rate[k], weights=seg[k])
                length[~ok] = np.nan
                out[f"{cat}_len_{r}km"] = length
                out[f"{cat}_rate_{r}km"] = wrate
    # side of the nearest subduction zone: the point's own topological plate is compared with
    # the overriding and subducting plates of that trench segment (filled in by sample_points)
    tree, idx = bs.subset_tree(CATEGORIES["sz"](t, thresholds).to_numpy())
    out["_sz_over"] = np.full(len(q), -1)
    out["_sz_sub"] = np.full(len(q), -1)
    if tree is not None:
        _, j = tree.query(q)
        out["_sz_over"] = t["over_pid"].to_numpy()[idx[j]]
        out["_sz_sub"] = t["sub_pid"].to_numpy()[idx[j]]
    ftype = np.full(len(q), "", object)
    tree, idx = bs.subset_tree(np.ones(len(t), bool))
    if tree is not None:
        _, j = tree.query(q)
        ftype = t["ftype"].to_numpy()[idx[j]]
    out["any_ftype"] = np.where(ok, ftype, "")
    return pd.DataFrame(out)


def topological_plate_ids(model, time, lon, lat):
    """Plate ID of the resolved topological plate (or network) containing each point."""
    out = np.full(len(lon), -1)
    ok = np.isfinite(lon) & np.isfinite(lat)
    if ok.any():
        snap = model.topological_snapshot(time)
        locs = snap.get_point_locations([(float(b), float(a)) for a, b in zip(lon[ok], lat[ok])])
        out[ok] = [_pid(l) for l in locs]
    return out


def sample_points(points: pd.DataFrame, mantle, paleomag, lags, cfg, progress=True):
    """Sample every point at ``round(age) + lag`` for each lag.

    ``points`` needs: pid, lon, lat, age_ma, pid_pm (plate ID in the palaeomagnetic model),
    poly_begin_ma, and an identifier column ``pt_id``. With ``lags=None`` every row carries its
    own ``lag_myr``, so points with different lag sets share one pass through time.
    Returns one long table, one row per point and lag.
    """
    if lags is None:
        own = points["lag_myr"].to_numpy(float)
    b = cfg["boundaries"]
    thr = {"div": b["divergence_threshold_cm_yr"], "conv": b["convergence_threshold_cm_yr"]}
    tmax = cfg["time"]["max_age_ma"] - cfg["time"]["velocity_delta_myr"]
    base = np.rint(points["age_ma"].to_numpy())
    jobs = []
    if lags is None:
        tt = base + own
        for T in np.unique(tt[np.isfinite(tt)]):
            if 0 <= T <= tmax:
                rows_T = np.nonzero(tt == T)[0]
                for lag in np.unique(own[rows_T]):
                    jobs.append((T, lag, rows_T[own[rows_T] == lag]))
    else:
        for lag in lags:
            tt = base + lag
            for T in np.unique(tt[np.isfinite(tt)]):
                if 0 <= T <= tmax:
                    jobs.append((T, lag, np.nonzero(tt == T)[0]))
    jobs.sort(key=lambda j: j[0])
    frames = []
    snap_cache = {}
    for n, (T, lag, rows) in enumerate(jobs):
        if T not in snap_cache:
            snap_cache.clear()
            snap_cache[T] = boundary_snapshot(mantle, T, b["spacing_deg"], b["continental_probe_km"],
                                              cfg["time"]["velocity_delta_myr"], b["margin_km"])
        bs = snap_cache[T]
        p = points.iloc[rows]
        rlon, rlat = mantle.reconstruct_points(p["lon"], p["lat"], p["pid"], T)
        ve, vn, vmag, vaz = mantle.point_velocities(rlon, rlat, p["pid"], T)
        plon, plat = paleomag.reconstruct_points(p["lon"], p["lat"], p["pid_pm"], T)
        _, _, vmag_pm, _ = paleomag.point_velocities(plon, plat, p["pid_pm"], T)
        f = sample_boundaries(bs, rlon, rlat, thr, b["search_radii_km"])
        f["topo_pid"] = topological_plate_ids(mantle, T, rlon, rlat)
        side = np.full(len(f), "other", object)
        side[f["topo_pid"].to_numpy() == f["_sz_over"].to_numpy()] = "upper"
        side[f["topo_pid"].to_numpy() == f["_sz_sub"].to_numpy()] = "lower"
        side[(f["_sz_over"].to_numpy() < 0) | np.isnan(f["sz_dist_km"].to_numpy())] = "unknown"
        f["sz_side"] = side
        f = f.drop(columns=["_sz_over", "_sz_sub"])
        f.insert(0, "pt_id", p["pt_id"].to_numpy())
        f.insert(1, "lag_myr", lag)
        f.insert(2, "time_ma", T)
        f["in_model"] = T <= p["poly_begin_ma"].to_numpy()
        f["mlon"], f["mlat"] = rlon, rlat
        f["plate_speed_cm_yr"], f["plate_az"] = vmag, vaz
        f["plate_ve"], f["plate_vn"] = ve, vn
        f["paleolat"], f["paleolon"] = plat, plon
        f["plate_speed_pm_cm_yr"] = vmag_pm
        # velocity of the site relative to the nearest divergent/convergent boundary normal
        frames.append(f)
        if progress and (n % 50 == 0 or n == len(jobs) - 1):
            print(f"    {n + 1}/{len(jobs)} time steps (t = {T:.0f} Ma)", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
