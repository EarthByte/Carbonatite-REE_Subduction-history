"""Rift outlines from rift-structure line work.

The outline of a rift is the footprint of its mapped structures, computed so that the
structures themselves cannot be recovered from it. For each rift:

1. The structures are densified (``spacing_km``) and projected to a Lambert azimuthal
   equal-area projection centred on the rift, which removes lon/lat distortion and the
   antimeridian.
2. A length-limited concave hull (alpha shape) is formed from the Delaunay triangulation of
   the densified points, keeping triangles whose longest edge is shorter than
   ``max_edge_km``. Sub-basins further apart than that stay separate polygons, and embayments
   wider than that are excluded, so a bent or branched rift is not filled across.
3. The hull is widened by ``pad_km`` (structures bound basins; the basin fill extends beyond
   the bounding faults), merged with the padded convex hull of any part too small or too
   straight to triangulate, smoothed by an opening/closing of ``smooth_km`` and simplified to
   ``simplify_km``.

The convex hull of every rift is computed alongside for comparison.
"""
from __future__ import annotations

import numpy as np
import shapely
from pyproj import CRS, Transformer
from scipy.spatial import Delaunay
from shapely.geometry import MultiPoint, Polygon, box
from shapely.ops import transform, unary_union

from .common import lonlat_to_xyz


def _centre(lon, lat):
    v = lonlat_to_xyz(lon, lat).mean(axis=0)
    v /= np.linalg.norm(v)
    return np.degrees(np.arctan2(v[1], v[0])), np.degrees(np.arcsin(v[2]))


def _laea(lon0, lat0):
    crs = CRS.from_proj4(f"+proj=laea +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=km")
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return fwd, inv


def _densified_points(geom_km, spacing_km):
    pts = []
    for ln in getattr(geom_km, "geoms", [geom_km]):
        n = max(int(np.ceil(ln.length / spacing_km)), 1)
        d = np.linspace(0, ln.length, n + 1)
        pts.append(np.array([ln.interpolate(x).coords[0] for x in d]))
    return np.unique(np.round(np.vstack(pts), 3), axis=0)


def alpha_shape(xy, max_edge_km):
    """Union of Delaunay triangles whose longest edge is below ``max_edge_km``."""
    if len(xy) < 3:
        return Polygon()
    try:
        tri = Delaunay(xy)
    except Exception:                       # collinear or degenerate input
        return Polygon()
    t = xy[tri.simplices]
    e = np.stack([np.linalg.norm(t[:, i] - t[:, (i + 1) % 3], axis=1) for i in range(3)], axis=1)
    keep = t[e.max(axis=1) < max_edge_km]
    if not len(keep):
        return Polygon()
    polys = shapely.polygons(keep)
    return shapely.union_all(polys[shapely.area(polys) > 0])


def _to_lonlat(geom_km, inv):
    """Project a polygon from the local kilometre frame back to longitude and latitude with ``inv``, split
    at the antimeridian where it crosses it."""
    g = transform(lambda x, y, z=None: inv.transform(x, y), geom_km)
    minx, _, maxx, _ = g.bounds
    if maxx - minx <= 180:
        return shapely.make_valid(g)
    # crosses the antimeridian: work in 0-360, then split and shift back
    g360 = transform(lambda x, y, z=None: (np.where(np.asarray(x) < 0, np.asarray(x) + 360, x), y), g)
    g360 = shapely.make_valid(g360)
    west = g360.intersection(box(0, -90, 180, 90))
    east = transform(lambda x, y, z=None: (np.asarray(x) - 360, y), g360.intersection(box(180, -90, 360, 90)))
    return shapely.make_valid(unary_union([west, east]))


def rift_outline(lines_lonlat, spacing_km=5.0, max_edge_km=75.0, pad_km=10.0, smooth_km=5.0,
                 simplify_km=2.0):
    """Concave outline and convex hull of one rift, both in lon/lat, plus metrics in km²."""
    lines = shapely.line_merge(lines_lonlat) if lines_lonlat.geom_type != "LineString" else lines_lonlat
    allc = np.vstack([np.asarray(l.coords)[:, :2] for l in getattr(lines, "geoms", [lines])])
    lon0, lat0 = _centre(allc[:, 0], allc[:, 1])
    fwd, inv = _laea(lon0, lat0)
    g_km = transform(lambda x, y, z=None: fwd.transform(x, y), lines_lonlat)
    xy = _densified_points(g_km, spacing_km)

    convex = MultiPoint(xy).convex_hull.buffer(pad_km)
    core = alpha_shape(xy, max_edge_km)
    # every structure must lie inside its outline: pad the padded convex hulls of the
    # connected clusters that the triangulation could not cover
    covered = core.buffer(pad_km) if not core.is_empty else Polygon()
    missing = MultiPoint(xy).difference(covered)
    if not missing.is_empty:
        parts = [covered]
        for comp in _clusters(np.array([p.coords[0] for p in getattr(missing, "geoms", [missing])]),
                              max_edge_km):
            parts.append(MultiPoint(comp).convex_hull.buffer(pad_km))
        covered = unary_union(parts)
    concave = covered.buffer(smooth_km).buffer(-smooth_km)          # close small gaps
    concave = concave.buffer(-smooth_km / 2).buffer(smooth_km / 2)  # remove slivers
    concave = unary_union([concave, MultiPoint(xy).buffer(pad_km * 0.5)])
    concave = concave.simplify(simplify_km)

    empty = lambda poly: poly.difference(g_km.buffer(25.0)).area / max(poly.area, 1e-9)  # noqa: E731
    metrics = dict(area_concave_km2=concave.area, area_convex_km2=convex.area,
                   empty_frac_concave=empty(concave), empty_frac_convex=empty(convex),
                   n_parts=len(getattr(concave, "geoms", [concave])),
                   max_offset_km=shapely.hausdorff_distance(concave.boundary, g_km),
                   # 2A/P is the width of a thin shape; the padding adds 2*pad_km to it
                   width_eq_km=max(2 * concave.area / max(concave.length, 1e-9) - 2 * pad_km, 0.0),
                   **_rect_dims(g_km))
    return _to_lonlat(concave, inv), _to_lonlat(convex, inv), metrics


def _rect_dims(g_km):
    """Sides of the minimum-area rotated rectangle around the structures (width, length)."""
    r = shapely.minimum_rotated_rectangle(g_km)
    if r.geom_type != "Polygon":
        return dict(width_rect_km=0.0, length_rect_km=float(r.length))
    c = np.asarray(r.exterior.coords)
    a, b = np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[1])
    return dict(width_rect_km=float(min(a, b)), length_rect_km=float(max(a, b)))


def _clusters(xy, radius):
    """Groups of the points ``xy`` connected within ``radius`` (connected components)."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    if len(xy) == 1:
        return [xy]
    m = cKDTree(xy).sparse_distance_matrix(cKDTree(xy), radius)
    _, lab = connected_components(m, directed=False)
    return [xy[lab == k] for k in np.unique(lab)]
