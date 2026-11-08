"""Null (reference) point sets.

A deposit class is compared with points that could have hosted it but did not. Two
properties matter. The points must sit on crust that the plate model carries at the
sampled time, because a deposit can only be reconstructed there; and their ages must follow
the class's own age distribution, because plate-boundary geometry changes through time and
an age mismatch alone would produce a difference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pygplates

from .common import fibonacci_sphere


def continental_candidates(model, n_target, seed=0):
    """Equal-area points on present-day continental polygons, with plate IDs and polygon ages."""
    rng = np.random.default_rng(seed)
    part = pygplates.PlatePartitioner(model.continental_polygons, model.rotation_model)
    n = int(n_target / 0.3) + 1000                         # continents cover ~30-40 % of the sphere
    lon, lat = fibonacci_sphere(n)
    jitter = rng.normal(0, 0.05, (2, n))
    lon, lat = (lon + jitter[0] + 180) % 360 - 180, np.clip(lat + jitter[1], -89.9, 89.9)
    keep = np.array([part.partition_point(pygplates.PointOnSphere(float(b), float(a))) is not None
                     for a, b in zip(lon, lat)])
    lon, lat = lon[keep], lat[keep]
    pid, begin, _ = model.assign_plate_ids(lon, lat)
    ok = pid > 0
    return pd.DataFrame({"lon": lon[ok], "lat": lat[ok], "pid": pid[ok], "poly_begin_ma": begin[ok]})


def age_matched_nulls(candidates: pd.DataFrame, ages: np.ndarray, n: int, seed=0,
                      require_in_model=True, terrane_pids=None, weights=None):
    """Draw ``n`` null points whose ages follow ``ages``.

    Each drawn age is paired with a candidate location whose static polygon exists at that
    age (``require_in_model``), so nulls meet the same validity test as the deposits.
    ``terrane_pids`` restricts candidates to the plates that host the class. ``weights`` (one
    per age, e.g. one over district size) makes the drawn ages follow the weighted distribution.
    """
    rng = np.random.default_rng(seed)
    c = candidates
    if terrane_pids is not None:
        c = c[c["pid"].isin(terrane_pids)]
    ages = np.asarray(ages, float)
    ok = np.isfinite(ages)
    p = None
    if weights is not None:
        w = np.asarray(weights, float)[ok]
        p = w / w.sum()
    ages = ages[ok]
    draw = rng.choice(ages, size=n, replace=True, p=p)
    begin = c["poly_begin_ma"].to_numpy()
    rows = []
    for a in draw:
        pool = np.nonzero(begin >= a)[0] if require_in_model else np.arange(len(c))
        if len(pool):
            j = rng.choice(pool)
            rows.append((c["lon"].iat[j], c["lat"].iat[j], c["pid"].iat[j], begin[j], a))
    return pd.DataFrame(rows, columns=["lon", "lat", "pid", "poly_begin_ma", "age_ma"])


# ---------------------------------------------------------------------------
# Masks that make a null point a fair reference
#
# A null point stands for a place where a deposit of the class could have formed and been
# found. Two conditions break that. Present-day ice sheets cover crust nobody can prospect,
# so no deposit there could enter any compilation. And a region nobody has explored holds no
# deposits of any kind, so a difference between deposits and nulls there is a statement about
# exploration history. The explored footprint is taken from the deposits of *other* families
# in the same compilation, which is independent of the class under test but not of the
# predictors (deposits of every kind cluster near plate boundaries), so a masked run is a
# stress test of a result, not a replacement for the unmasked one.
# ---------------------------------------------------------------------------

# Classes that share a search history are one family, so the footprint of a class is never
# defined by deposits explorers found while looking for that class.
FAMILY = {"CD": "ZN", "MVT": "ZN", "ZNREP": "ZN", "SSPB": "ZN", "KIPU": "ZN", "ZNPB": "ZN",
          "CARB": "MANTLE", "AIREE": "MANTLE", "KIMB": "MANTLE", "LAMP": "MANTLE"}


def family(dclass: str) -> str:
    return FAMILY.get(dclass, dclass)


def ice_mask(lon, lat, paths):
    """True where a point lies under present-day ice.

    ``paths`` is one polygon file or several: Natural Earth 10 m glaciated areas (the
    Antarctic and Greenland ice sheets, ice caps and glaciers) and, because they cover
    continental crust too, the Antarctic ice shelves.
    """
    import geopandas as gpd

    if isinstance(paths, (str, Path)):
        paths = [paths]
    ice = pd.concat([gpd.read_file(p)[["geometry"]] for p in paths], ignore_index=True)
    ice = gpd.GeoDataFrame(ice, geometry="geometry", crs="EPSG:4326")
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(np.asarray(lon, float),
                                                       np.asarray(lat, float)), crs=ice.crs)
    hit = np.zeros(len(pts), bool)
    j = pts.sjoin(ice, how="inner", predicate="within").index.unique()
    hit[np.asarray(j, int)] = True
    return hit


def nearest_km(lon, lat, ref_lon, ref_lat):
    """Great-circle distance from each point to the nearest of ``ref``; inf if ``ref`` is empty."""
    from scipy.spatial import cKDTree

    from .common import chord_to_km, lonlat_to_xyz

    lon = np.asarray(lon, float)
    if not len(ref_lon):
        return np.full(len(lon), np.inf)
    d, _ = cKDTree(lonlat_to_xyz(ref_lon, ref_lat)).query(lonlat_to_xyz(lon, lat))
    return chord_to_km(d)
