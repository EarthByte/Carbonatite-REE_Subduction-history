#!/usr/bin/env python
"""Make the Shirmard et al. (2025) craton outlines reconstructable with the plate model.

Shirmard et al. (2025) delineated craton boundaries from the REVEAL full-waveform tomography
model: 20 present-day polygons (`Craton_Boundaries_Inferred`), distributed without plate
identifications. To measure the distance from a deposit to the nearest craton edge at the time
the deposit formed, each polygon is cut by the static polygons of the mantle-frame plate model
and every piece takes the plate identification and valid time of the static polygon it lies
in, so that it moves, appears and disappears with the crust that carries it. This replaces the
Merdith et al. craton polygons used before.

Reads the shapefile named by `paths.craton_source` in the configuration and the static
polygons of the mantle model. Writes ONE file, the path named by `paths.craton_file` (a
.gpmlz feature collection), through a .partial name, and prints the number of pieces, their
plate identifications and the area each polygon keeps after partitioning.

    python scripts/44_shirmard_cratons.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pygplates

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import load_config  # noqa: E402
from rifts.models import load_models  # noqa: E402


from shapely.ops import unary_union  # noqa: E402


def _laea(lon0, lat0):
    from pyproj import Transformer
    crs = f"+proj=laea +lat_0={lat0} +lon_0={lon0} +R=6371000 +units=m"
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return fwd, inv


def _to_shape(poly, fwd, step_deg=0.1):
    """A pygplates polygon, densified, as a shapely polygon in the projection."""
    from shapely.geometry import Polygon
    ll = poly.to_tessellated(np.radians(step_deg)).to_lat_lon_array()
    x, y = fwd.transform(ll[:, 1], ll[:, 0])
    return Polygon(np.column_stack([x, y])).buffer(0)


def _static_under(cs, inv, g, partitioner, n=20000, seed=0):
    """The present-day static polygons under a craton, found by partitioning a dense sample
    of points inside it and every vertex of its outline (partition_geometry can miss some)."""
    from shapely import contains, points
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = cs.bounds
    xs, ys = rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)
    keep = contains(cs, points(xs, ys))
    lon, lat = inv.transform(xs[keep], ys[keep])
    pts = [(float(a), float(b)) for a, b in zip(lat, lon)]
    pts += [tuple(map(float, v)) for v in g.to_lat_lon_array()]
    found, hits = {}, {}
    for la, lo in pts:
        r = partitioner.partition_point(pygplates.PointOnSphere(la, lo))
        if r is None:
            continue
        f = r.get_feature()
        key = (f.get_reconstruction_plate_id(), tuple(f.get_valid_time()),
               round(r.get_reconstructed_geometry().get_area(), 9))
        found.setdefault(key, f)
        hits[key] = hits.get(key, 0) + 1
    # most-hit first: where static polygons overlap at the present day, the one the plate
    # partitioner assigns most of the shared ground to takes it
    return [found[k] for k in sorted(found, key=lambda k: -hits[k])]


def partition(cratons, mantle):
    """Cut every craton polygon by the present-day static polygons it overlaps.

    pygplates clips a polygon into polylines, which cannot be tested for inside/outside, so
    the cut is made with shapely in a Lambert equal-area projection centred on each craton,
    where areas are exact and the antimeridian plays no role. Each piece takes the plate
    identification and valid time of its static polygon.
    """
    # static polygons with no past (valid only at 0 Ma) carry nothing back in time
    static = [f for fc in mantle.static_polygons for f in fc
              if float(f.get_valid_time()[0]) > 1.0]
    partitioner = pygplates.PlatePartitioner(static, mantle.rotation_model)
    pieces, kept = [], []
    for f in cratons:
        name = str(f.get_shapefile_attribute("Names") or f.get_name())
        for g in f.get_geometries():
            lat0, lon0 = g.get_interior_centroid().to_lat_lon()
            fwd, inv = _laea(lon0, lat0)
            cs = _to_shape(g, fwd)
            parts = []                               # [shapely polygon, static feature]
            for sf in _static_under(cs, inv, g, partitioner):
                for sg in sf.get_geometries():
                    if not isinstance(sg, pygplates.PolygonOnSphere):
                        continue
                    inter = cs.intersection(_to_shape(sg, fwd))
                    if parts:                        # ground already taken is not shared
                        inter = inter.difference(unary_union([q for q, _ in parts]))
                    for q in (inter.geoms if hasattr(inter, "geoms") else [inter]):
                        if q.geom_type == "Polygon" and q.area >= 1e6:     # 1 km2
                            parts.append([q, sf])
            covered = sum(q.area for q, _ in parts) / cs.area
            # ground of the craton not under any static polygon that has a past (gaps and
            # slivers between static polygons) joins the piece it shares most boundary with,
            # so that the pieces tile the craton exactly and no internal edge is created
            rest = cs.difference(unary_union([q for q, _ in parts])) if parts else cs
            for r in (rest.geoms if hasattr(rest, "geoms") else [rest]):
                if r.is_empty or r.geom_type != "Polygon" or not parts:
                    continue
                shared = [r.buffer(2000).intersection(q).area for q, _ in parts]
                k = int(np.argmax(shared))
                if shared[k] > 0:
                    merged = unary_union([parts[k][0], r.buffer(1)]).buffer(0)
                    parts[k][0] = max(merged.geoms, key=lambda x: x.area) \
                        if hasattr(merged, "geoms") else merged
            for q, sf in parts:
                x, y = q.exterior.xy
                lon, lat = inv.transform(np.asarray(x), np.asarray(y))
                piece = pygplates.Feature()
                piece.set_geometry(pygplates.PolygonOnSphere(list(zip(lat, lon))))
                piece.set_reconstruction_plate_id(sf.get_reconstruction_plate_id())
                b, e = sf.get_valid_time()
                piece.set_valid_time(b, e)
                piece.set_name(name)
                pieces.append(piece)
            final = sum(q.area for q, _ in parts) / cs.area
            kept.append((name, covered, final))
    return pieces, kept


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    src = Path(cfg["paths"]["craton_source"])
    out = Path(cfg["paths"]["craton_file"])
    mantle, _ = load_models(cfg)

    cratons = pygplates.FeatureCollection(str(src))
    print(f"{len(cratons)} craton polygons in {src.name}")
    pieces, kept = partition(cratons, mantle)
    pids = sorted({f.get_reconstruction_plate_id() for f in pieces})
    print(f"{len(pieces)} pieces on {len(pids)} plates")
    for name, frac, final in kept:
        if frac < 0.999 or abs(final - 1) > 0.001:
            print(f"  {name}: {frac:.1%} of its area on static polygons with a past; "
                  f"{final:.1%} after joining the remainder to its neighbours")
    begins = sorted(float(f.get_valid_time()[0]) for f in pieces)
    print(f"valid-time begin of the pieces: min {begins[0]:.0f} Ma, "
          f"median {begins[len(begins) // 2]:.0f} Ma, max {begins[-1]:.0f} Ma")

    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.stem + ".partial" + out.suffix)
    pygplates.FeatureCollection(pieces).write(str(part))
    part.replace(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
