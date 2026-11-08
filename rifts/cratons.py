"""Distance to the nearest craton edge, measured at the age of the point.

Hoggard et al. (2020) placed about 85 % of sediment-hosted base-metal deposits within 200 km
of the present-day thick-to-thin lithosphere transition, and Shirmard et al. (2025) sharpened
that to about 120 km using REVEAL horizontal shear-wave velocity. Both are present-day
statements: the architecture is measured now and the deposit is placed on it now. Over
1.8 Gyr the continents carrying those keels have moved, collided and rifted, so the primary
measure here is made at the time each point formed, on the craton outlines of Shirmard et al.
(2025), delineated from REVEAL tomography, cut by the static polygons of the mantle-frame
plate model (scripts/44_shirmard_cratons.py) and reconstructed with the same rotation model as
everything else; the parts of each craton are merged again at every time, so that the cuts
never count as craton edges. The present-day REVEAL velocity and its horizontal gradient are
sampled beside it, at the point's present-day position, as the link to the published result.

The sign carries the information the published statements throw away: negative inside a
craton, positive outside it, so that "on the edge" and "deep in the keel" are distinguishable
rather than both being a small absolute distance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygplates
from scipy.spatial import cKDTree

from .common import EARTH_RADIUS_KM, chord_to_km, lonlat_to_xyz


@dataclass
class CratonSnapshot:
    time: float
    polygons: list
    centres: np.ndarray          # unit vectors, one per polygon
    radii: np.ndarray            # angular radius of each polygon, radians
    tree: cKDTree                # boundary points of every polygon


def load_cratons(path):
    return pygplates.FeatureCollection(str(path))


def craton_snapshot(cratons, rotation_model, time, spacing_deg=0.25, anchor_plate=0):
    """Reconstruct the craton polygons to ``time`` and index their boundaries."""
    rec = []
    pygplates.reconstruct(cratons, rotation_model, rec, float(time), anchor_plate_id=anchor_plate)
    polys, centres, radii, pts, names = [], [], [], [], []
    for r in rec:
        g = r.get_reconstructed_geometry()
        if not isinstance(g, pygplates.PolygonOnSphere):
            continue
        ll = g.to_tessellated(np.radians(spacing_deg)).to_lat_lon_array()
        xyz = lonlat_to_xyz(ll[:, 1], ll[:, 0])
        c = xyz.mean(axis=0)
        n = np.linalg.norm(c)
        if n == 0:
            continue
        c /= n
        polys.append(g)
        centres.append(c)
        radii.append(float(np.arccos(np.clip(xyz @ c, -1, 1)).max()))
        pts.append(xyz)
        names.append(r.get_feature().get_name() or "")
    if not polys:
        return None
    tree_pts = _outer_edges(polys, pts, names, spacing_deg)
    return CratonSnapshot(float(time), polys, np.array(centres), np.array(radii),
                          cKDTree(tree_pts))


def _outer_edges(polys, pts, names, spacing_deg, close_km=2.0):
    """Boundary points on the outer edge of each craton.

    A craton cut by the static polygons of the plate model (scripts/44) is carried as several
    pieces, and the cuts are not craton edges. The pieces of one craton are merged in a Lambert
    equal-area projection centred on them (closed by +/-``close_km`` to seal the cut lines),
    and only the boundary of the merged outline is kept. Where the plates carrying two pieces
    have moved apart, the gap between them is a real edge of the reconstructed craton and
    survives the merge. Unnamed features are used as they are.
    """
    out = [pts[k] for k, nm in enumerate(names) if not nm]
    for lon, lat in merged_rings(polys, names, spacing_deg, close_km):
        out.append(lonlat_to_xyz(np.asarray(lon), np.asarray(lat)))
    return np.vstack(out)


def merged_rings(polys, names, spacing_deg=0.25, close_km=2.0):
    """Outer rings (lon, lat arrays) of each named craton, its pieces merged; see
    ``_outer_edges``. Used both for the distance measure and for drawing the outlines."""
    from pyproj import Transformer
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    rings = []
    groups = {}
    for k, nm in enumerate(names):
        if nm:
            groups.setdefault(nm, []).append(k)
    for nm, ks in groups.items():
        xyz = np.vstack([lonlat_to_xyz(*polys[k].to_lat_lon_array()[:, ::-1].T) for k in ks])
        c = xyz.mean(axis=0)
        c /= np.linalg.norm(c)
        lat0, lon0 = np.degrees(np.arcsin(c[2])), np.degrees(np.arctan2(c[1], c[0]))
        crs = f"+proj=laea +lat_0={lat0} +lon_0={lon0} +R={EARTH_RADIUS_KM * 1000} +units=m"
        fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        shapes = []
        for k in ks:
            ll = polys[k].to_tessellated(np.radians(min(spacing_deg, 0.1))).to_lat_lon_array()
            x, y = fwd.transform(ll[:, 1], ll[:, 0])
            q = Polygon(np.column_stack([x, y])).buffer(0)
            if not q.is_empty:
                shapes.append(q)
        if not shapes:
            continue
        merged = unary_union(shapes).buffer(close_km * 1000).buffer(-close_km * 1000)
        parts = merged.geoms if hasattr(merged, "geoms") else [merged]
        step = np.radians(spacing_deg) * EARTH_RADIUS_KM * 1000
        for q in parts:
            # interior rings of the merge are gaps between static polygons, not craton holes
            # (the source outlines have none)
            for ring in [q.exterior]:
                seg = ring.segmentize(step)
                x, y = np.asarray(seg.xy[0]), np.asarray(seg.xy[1])
                lon, lat = inv.transform(x, y)
                rings.append((np.asarray(lon), np.asarray(lat)))
    return rings


def craton_distance(snap: CratonSnapshot, lon, lat):
    """Signed distance in km to the nearest craton edge: negative inside a craton.

    The inside test is done only against polygons whose angular extent reaches the point,
    which is what keeps it to a handful of tests per point rather than all of them.
    """
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    xyz = lonlat_to_xyz(lon, lat)
    km = chord_to_km(snap.tree.query(xyz)[0])
    cos_lim = np.cos(np.minimum(snap.radii + 1e-3, np.pi))
    dots = xyz @ snap.centres.T                       # cos of angle to each polygon centre
    inside = np.zeros(len(lon), bool)
    for i, row in enumerate(dots):
        for j in np.nonzero(row >= cos_lim)[0]:
            if snap.polygons[j].is_point_in_polygon(pygplates.PointOnSphere(float(lat[i]),
                                                                           float(lon[i]))):
                inside[i] = True
                break
    return np.where(inside, -km, km)


def sample_grid(da, lon, lat):
    """Nearest-cell value of a lon/lat DataArray, with longitude wrapped."""
    glon = np.asarray(da["lon"])
    glat = np.asarray(da["lat"])
    lo = np.asarray(lon, float)
    lo = np.where(lo < glon.min(), lo + 360, np.where(lo > glon.max(), lo - 360, lo))
    i = _nearest(glat, np.asarray(lat, float))
    j = _nearest(glon, lo)
    return np.asarray(da.values)[i, j]


def _nearest(axis, v):
    """Index of the nearest cell centre on a sorted axis."""
    k = np.clip(np.searchsorted(axis, v), 1, len(axis) - 1)
    return np.where(np.abs(v - axis[k - 1]) <= np.abs(axis[k] - v), k - 1, k)


def gradient_grid(da):
    """Horizontal gradient magnitude of a lon/lat grid, per 100 km.

    The craton edge in a velocity model is a gradient, not a contour, so the gradient is what
    the published segmentations are built on; keeping it as a field avoids committing to one
    velocity threshold.
    """
    v = np.asarray(da.values, float)
    lat = np.asarray(da["lat"], float)
    lon = np.asarray(da["lon"], float)
    dlat_km = np.gradient(lat) * np.pi / 180 * EARTH_RADIUS_KM
    dlon_km = (np.gradient(lon) * np.pi / 180 * EARTH_RADIUS_KM)[None, :] * \
        np.cos(np.radians(lat))[:, None]
    gy = np.gradient(v, axis=0) / dlat_km[:, None]
    gx = np.gradient(v, axis=1) / np.where(np.abs(dlon_km) < 1e-6, np.nan, dlon_km)
    return np.hypot(gx, gy) * 100.0
