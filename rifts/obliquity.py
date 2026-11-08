"""Opening rate and obliquity of a rift from its own trend and the plates on its flanks.

The plate model resolves only a minority of GloRiDa rifts as divergent plate boundaries, so
taking a rift's opening rate and obliquity from the nearest resolved boundary leaves most rifts
without either, and measures the model's own boundary geometry rather than the mapped rift. The
model does, however, carry a plate for every piece of continental crust, which allows the
measurement the rifting literature actually defines: the angle between the extension direction
and the normal to the rift trend, with 0° orthogonal opening and 90° pure strike-slip.

A rift is rarely straight, so its trend is measured along it rather than once. For each outline,
in a Lambert azimuthal equal-area projection centred on it:

1. The largest part of the outline is divided into bins along the long axis of its minimum-area
   rectangle, and the centroid of the outline in each bin gives a spine point. The local trend
   at a bin is the direction of the spine through it, and the local half-width is the spread of
   the outline about the spine there.
2. Two flank points per bin sit on the local normal, half the local width plus ``pad_km`` to
   each side, and take a plate id from the model's static polygons.
3. Where a bin's flanks belong to different plates, the relative velocity of one with respect
   to the other is taken over ``interval_myr`` ending at the time of interest, at the bin's
   reconstructed position. Trend and velocity are compared in the same frame, because the spine
   is reconstructed with the rift's own plate.

Bins are laid out along the long axis of the bounding rectangle, so a rift bent by much more
than about 45° is followed in part rather than in full, and its local half-width is overstated
where the bend is sharpest.

A rift's obliquity is the length-weighted median over its bins, and the spread between bins is
reported with it: a strongly curved rift has no single obliquity, and the spread says so. Where
the flanks of every bin belong to one plate, the model holds no relative motion across the rift
and both measures are undefined, which is reported rather than filled in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

from .outlines import _centre, _laea

MIN_BIN_KM = 25.0


def _largest(geom):
    parts = list(getattr(geom, "geoms", [geom]))
    return max(parts, key=lambda g: g.area) if parts else geom


def _boundary_points(g_km, spacing_km=10.0):
    ln = g_km.exterior
    n = max(int(np.ceil(ln.length / spacing_km)), 8)
    return np.array([ln.interpolate(x).coords[0] for x in np.linspace(0, ln.length, n)])


def spine(geom, pad_km=50.0, max_bins=9):
    """Spine points, local trends, local half-widths and flank points of one outline.

    Returns a dict of arrays in lon/lat, or ``None`` for a geometry with no area.
    """
    g = _largest(geom)
    if g.is_empty or g.area <= 0:
        return None
    lon, lat = np.asarray(g.exterior.coords).T
    lon0, lat0 = _centre(lon, lat)
    fwd, inv = _laea(lon0, lat0)
    gk = shapely.ops.transform(lambda a, b, z=None: fwd.transform(a, b), g)
    rect = shapely.minimum_rotated_rectangle(gk)
    if rect.geom_type != "Polygon":
        return None
    c = np.asarray(rect.exterior.coords)[:4]
    e = [c[(i + 1) % 4] - c[i] for i in range(2)]
    long_i = int(np.argmax([np.linalg.norm(v) for v in e]))
    u = e[long_i] / np.linalg.norm(e[long_i])
    length = float(np.linalg.norm(e[long_i]))
    mid = c.mean(axis=0)

    pts = _boundary_points(gk)
    s = (pts - mid) @ u
    n_bins = int(np.clip(np.floor(length / MIN_BIN_KM), 1, max_bins))
    edges = np.linspace(s.min(), s.max(), n_bins + 1)
    cen, half, sm = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s <= hi)
        if m.sum() < 2:
            continue
        cen.append(pts[m].mean(axis=0))
        sm.append(0.5 * (lo + hi))
        half.append(0.5 * float(np.ptp((pts[m] - pts[m].mean(axis=0)) @ np.array([-u[1], u[0]]))))
    if not cen:
        return None
    cen = np.asarray(cen)
    half = np.asarray(half)
    if len(cen) == 1:                                   # one bin: the rectangle axis is the trend
        tang = np.array([u])
    else:                                               # central differences along the spine
        tang = np.gradient(cen, np.asarray(sm), axis=0)
        tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    w = np.full(len(cen), (edges[1] - edges[0]))
    ends = cen + tang * (edges[1] - edges[0]) * 0.5
    left = cen + nrm * (half + pad_km)[:, None]
    right = cen - nrm * (half + pad_km)[:, None]
    out = {}
    for k, arr in (("c", cen), ("t", ends), ("left", left), ("right", right)):
        x, y = inv.transform(arr[:, 0], arr[:, 1])
        out[f"{k}_lon"], out[f"{k}_lat"] = np.asarray(x), np.asarray(y)
    out["w_km"] = w
    out["half_km"] = half
    return out


def rift_spines(outlines: pd.DataFrame, pad_km=50.0, max_bins=9) -> pd.DataFrame:
    """One row per bin of every rift: spine point, a point along the local trend, flank points."""
    rows = []
    for _, r in outlines.iterrows():
        sp = spine(r.geometry, pad_km, max_bins)
        if sp is None:
            continue
        for j in range(len(sp["w_km"])):
            rows.append({"outline_id": r["outline_id"], "bin": j,
                         **{k: (v[j] if hasattr(v, "__len__") else v) for k, v in sp.items()}})
    return pd.DataFrame(rows)


def _azimuth(lon1, lat1, lon2, lat2):
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = np.radians(np.asarray(lat1, float)), np.radians(np.asarray(lat2, float))
    dl = np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(y, x)) % 360.0


def _fold(a):
    """The angle between two undirected lines, 0–90°."""
    a = np.abs(np.asarray(a, float)) % 360.0
    a = np.where(a > 180.0, 360.0 - a, a)
    return np.where(a > 90.0, 180.0 - a, a)


def _wquantile(x, w, q):
    o = np.argsort(x)
    x, w = np.asarray(x)[o], np.asarray(w)[o]
    cw = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), cw, x))


def flank_plate_ids(bins: pd.DataFrame, model) -> pd.DataFrame:
    """Plate id of each spine point and of the two flank points of every bin."""
    out = bins[["outline_id", "bin"]].copy()
    for k in ("c", "left", "right"):
        out[f"pid_{k}"] = model.assign_plate_ids(bins[f"{k}_lon"].to_numpy(),
                                                 bins[f"{k}_lat"].to_numpy())[0]
    return out


def flank_kinematics(bins: pd.DataFrame, pids: pd.DataFrame, model, times: pd.Series,
                     interval_myr=5.0, anchor_plate=0) -> pd.DataFrame:
    """Opening rate and obliquity of every rift at one time, from its flank plates.

    ``bins`` comes from :func:`rift_spines`, ``pids`` from :func:`flank_plate_ids` and ``times``
    is a Series of times in Ma indexed by ``outline_id``. Returns one row per rift.
    """
    import pygplates

    b = bins.merge(pids, on=["outline_id", "bin"])
    b["time_ma"] = b["outline_id"].map(times).astype(float)
    rot = model.rotation_model
    rows = []
    for oid, g in b.groupby("outline_id", sort=False):
        t1 = float(g["time_ma"].iat[0])
        r = dict(outline_id=oid, time_ma=t1, n_bins=len(g),
                 n_bins_two_plates=int((g["pid_left"] != g["pid_right"]).sum()))
        rows.append(r)
        if not np.isfinite(t1):
            continue
        t0 = t1 + interval_myr
        ob, ra, sp, wt, tr = [], [], [], [], []
        for _, q in g.iterrows():
            pm = int(q["pid_c"])
            if pm == 0:
                continue
            rm = rot.get_rotation(t1, pm, anchor_plate_id=anchor_plate,
                                  use_identity_for_missing_plate_ids=False)
            if rm is None:
                continue
            pc = rm * pygplates.PointOnSphere(float(q["c_lat"]), float(q["c_lon"]))
            pt = rm * pygplates.PointOnSphere(float(q["t_lat"]), float(q["t_lon"]))
            (cla, clo), (tla, tlo) = pc.to_lat_lon(), pt.to_lat_lon()
            trend = float(_azimuth(clo, cla, tlo, tla))
            tr.append((trend, q["w_km"]))
            pl, pr = int(q["pid_left"]), int(q["pid_right"])
            if pl == pr or pl == 0 or pr == 0:
                continue
            stage = rot.get_rotation(t1, pl, t0, pr, anchor_plate_id=anchor_plate,
                                     use_identity_for_missing_plate_ids=False)
            if stage is None:
                continue
            v = pygplates.calculate_velocities([pc], stage, interval_myr,
                                               pygplates.VelocityUnits.cms_per_yr)[0]
            enu = pygplates.LocalCartesian.convert_from_geocentric_to_north_east_down([pc], [v])[0]
            vn, ve = enu.get_x(), enu.get_y()
            speed = float(np.hypot(ve, vn))
            sp.append(speed)
            if speed <= 1e-6:
                ra.append(0.0)
                wt.append(q["w_km"])
                continue
            d = float(_fold(np.degrees(np.arctan2(ve, vn)) % 360.0 - (trend + 90.0)))
            ob.append((d, q["w_km"]))
            ra.append(speed * np.cos(np.radians(d)))
            wt.append(q["w_km"])
        if tr:
            r["flank_trend_deg"] = _wquantile([x for x, _ in tr], [w for _, w in tr], 0.5)
        if sp:
            r["flank_speed_cm_yr"] = _wquantile(sp, wt, 0.5)
            r["flank_rate_cm_yr"] = _wquantile(ra, wt, 0.5)
            r["n_bins_moving"] = int(np.sum(np.asarray(sp) > 1e-6))
        if ob:
            x = [a for a, _ in ob]
            w = [c for _, c in ob]
            r["flank_obliq_deg"] = _wquantile(x, w, 0.5)
            r["flank_obliq_iqr_deg"] = _wquantile(x, w, 0.75) - _wquantile(x, w, 0.25)
            r["n_bins_obliq"] = len(x)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- Brune COB method
#
# The measurement of record for a rift's own opening. Where two continental blocks were still
# joined, the shared edge of their continent-ocean-boundary polygons is the rift between them,
# and the relative motion of the two blocks across that edge gives the extension rate and the
# obliquity, exactly as Brune et al. (2016, 2018) define them. This is the method of the group's
# T14 tutorial, applied here to date every GloRiDa rift that lies on such an edge while it was
# active. It reaches only rifts between the 31 blocks the polygons carry, and only as far back as
# the polygons and rotations support, which is the limit on what can be said about rift
# kinematics before the Mesozoic.

SEGMENT_ARC_DEG = 0.5
POLYGON_BUFFER_DEG = 0.5
VELOCITY_WINDOW_MYR = 1.0
MIN_VEL_MM_YR = 0.1
COMPRESSION_CLIP_MM_YR = 0.5
EARTH_RADIUS_KM = 6371.0


def _reconstruct_polygon(geom, rot, plate_id, age, anchor=0):
    """Rotate a polygon or multipolygon on plate ``plate_id`` to ``age`` (anchor plate ``anchor``); None
    where the plate has no rotation."""
    import pygplates
    from shapely.geometry import MultiPolygon, Polygon
    r = rot.get_rotation(float(age), int(plate_id), 0.0, anchor_plate_id=anchor,
                         use_identity_for_missing_plate_ids=False)
    if r is None:
        return None
    if geom.geom_type == "MultiPolygon":
        parts = [_reconstruct_polygon(p, rot, plate_id, age, anchor) for p in geom.geoms]
        parts = [p for p in parts if p is not None and not p.is_empty]
        return MultiPolygon(parts) if parts else None
    out = []
    for lon, lat in geom.exterior.coords:
        la, lo = (r * pygplates.PointOnSphere(float(lat), float(lon))).to_lat_lon()
        out.append((lo, la))
    p = Polygon(out)
    return p if p.is_valid else p.buffer(0)


def _xyz(lon, lat):
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def cob_segments(cob, model, age, anchor=0, arc_deg=SEGMENT_ARC_DEG,
                 buffer_deg=POLYGON_BUFFER_DEG, window_myr=VELOCITY_WINDOW_MYR):
    """Rift segments between every pair of continental blocks still joined at ``age``.

    Returns one row per segment: its midpoint, the orthogonal and parallel components of the
    relative velocity in mm/yr, the obliquity in degrees and the segment length. Obliquity does
    not depend on which way the normal points; the sign of the orthogonal component, which
    separates extension from convergence, is taken against the direction from one block's centre
    to the other's.
    """
    import shapely
    from shapely.geometry import LineString, MultiLineString

    rot = model.rotation_model
    rec, cen = {}, {}
    for pid, geom in cob.items():
        g = _reconstruct_polygon(geom, rot, pid, age, anchor)
        if g is not None and not g.is_empty:
            rec[pid] = g
            c = g.centroid
            cen[pid] = np.array([c.x, c.y])
    rows = []
    pids = sorted(rec)
    for i, a in enumerate(pids):
        ga = rec[a]
        for b in pids[i + 1:]:
            gb = rec[b]
            ab, bb = ga.bounds, gb.bounds
            if ab[0] > bb[2] + buffer_deg or bb[0] > ab[2] + buffer_deg \
                    or ab[1] > bb[3] + buffer_deg or bb[1] > ab[3] + buffer_deg:
                continue
            shared = ga.boundary.intersection(gb.buffer(buffer_deg))
            if shared.is_empty:
                continue
            lines = [shared] if isinstance(shared, LineString) else (
                list(shared.geoms) if isinstance(shared, MultiLineString) else
                [g for g in getattr(shared, "geoms", []) if isinstance(g, LineString)])
            stage = rot.get_rotation(float(age), int(b), float(age) + window_myr, int(a),
                                     anchor_plate_id=anchor,
                                     use_identity_for_missing_plate_ids=False)
            if stage is None or stage.represents_identity_rotation():
                continue
            pole, angle = stage.get_euler_pole_and_angle()
            omega = np.array([pole.get_x(), pole.get_y(), pole.get_z()]) * (angle / window_myr)
            out_dir = cen[b] - cen[a]
            for ln in lines:
                if ln.length < arc_deg / 2:
                    continue
                n = max(int(np.ceil(ln.length / arc_deg)), 2)
                pts = np.array([ln.interpolate(t, normalized=True).coords[0]
                                for t in np.linspace(0, 1, n + 1)])
                xyz = _xyz(pts[:, 0], pts[:, 1])
                seg = xyz[1:] - xyz[:-1]
                ln_arc = np.linalg.norm(seg, axis=1)
                keep = ln_arc > 1e-9
                if not keep.any():
                    continue
                t_hat = seg[keep] / ln_arc[keep, None]
                mid = 0.5 * (xyz[1:] + xyz[:-1])[keep]
                mid /= np.linalg.norm(mid, axis=1)[:, None]
                n_hat = np.cross(t_hat, mid)
                n_hat /= np.maximum(np.linalg.norm(n_hat, axis=1)[:, None], 1e-12)
                v = np.cross(omega[None, :], mid) * EARTH_RADIUS_KM   # rad/Myr * km = km/Myr = mm/yr
                v_orth = np.sum(v * n_hat, axis=1)
                v_par = np.sum(v * t_hat, axis=1)
                mlat = np.degrees(np.arcsin(np.clip(mid[:, 2], -1, 1)))
                mlon = np.degrees(np.arctan2(mid[:, 1], mid[:, 0]))
                # orient the normal so that positive is away from block a, towards block b
                ref = np.column_stack([out_dir[0] * np.ones(len(mlat)), out_dir[1] * np.ones(len(mlat))])
                east = np.column_stack([-np.sin(np.radians(mlon)), np.cos(np.radians(mlon)),
                                        np.zeros(len(mlat))])
                north = np.cross(mid, east)
                sign = np.sign(np.sum(n_hat * east, axis=1) * ref[:, 0]
                               + np.sum(n_hat * north, axis=1) * ref[:, 1])
                sign[sign == 0] = 1.0
                v_orth = v_orth * sign
                speed = np.hypot(v_orth, v_par)
                ok = (speed >= MIN_VEL_MM_YR) & (v_orth > -COMPRESSION_CLIP_MM_YR)
                if not ok.any():
                    continue
                rows.append(pd.DataFrame(dict(
                    age_ma=float(age), plate_a=a, plate_b=b,
                    lon=mlon[ok], lat=mlat[ok],
                    v_orth_mm_yr=v_orth[ok], v_par_mm_yr=np.abs(v_par[ok]),
                    obliq_deg=np.degrees(np.arctan2(np.abs(v_par[ok]), np.abs(v_orth[ok]))),
                    seg_len_km=(ln_arc[keep][ok] * EARTH_RADIUS_KM))))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["age_ma", "plate_a", "plate_b", "lon", "lat", "v_orth_mm_yr", "v_par_mm_yr",
                 "obliq_deg", "seg_len_km"])


def read_cob(path):
    """Brune's continent-ocean-boundary polygons, keyed by plate id."""
    import geopandas as gpd
    g = gpd.read_file(path).to_crs("EPSG:4326")
    col = "PLATEID1" if "PLATEID1" in g else g.columns[0]
    return {int(r[col]): r.geometry for _, r in g.iterrows() if r.geometry is not None}
