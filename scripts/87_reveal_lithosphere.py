#!/usr/bin/env python
"""Thickness of the fast lithospheric lid from the REVEAL full-waveform tomography model.

REVEAL (Thrastarson et al. 2024, doi 10.1785/0120230273) gives VSV and VSH on a 0.5-degree grid
at 10 km depth spacing through the upper mantle. The isotropic (Voigt) shear-wave speed
Vs = sqrt((2 VSV^2 + VSH^2) / 3) is expressed at each depth as a percentage anomaly from its
area-weighted global mean, so that the 220 km step of the one-dimensional starting model drops
out. The lid thickness of a column is the deepest depth, counted continuously down from 60 km,
at which the anomaly stays at or above a threshold (1.0, 1.5 and 2.0 per cent; 50 km where the
column is not fast at 60 km). Its horizontal gradient (km of thickness per 100 km, after a
one-cell Gaussian smoothing) measures the steps in lithospheric thickness on which CO2-rich
magmas are found (Gibson et al. 2024; Bowman et al. 2026).

Shirmard et al. (2025, Geosci. Front. 16, 102176) analysed lithospheric thickness by tectonic age
(their Fig. 4: Hoggard et al. 2020 thickness from SL2013sv, cratons mostly 150-200 km) and clustered
REVEAL velocities at 150-200 km, with the primary aim of delineating craton outlines; they released
the outlines and velocity slices but no thickness grid. This script builds the thickness grid
directly from REVEAL.

Benchmark: against the 170 km lithosphere-asthenosphere boundary contour of Hoggard et al.
(2020) distributed with Shirmard et al. (2025, github.com/EarthByte/Craton_Boundaries), the area
of 1.5 per cent lid thicker than 230 km is recovered with recall 0.65 and precision 0.63 (written
to the benchmark table for every threshold and cut-off when the contour file is present).

Inputs: paths.reveal_upper_file (REVEAL 0-1,000 km, netCDF with depth, latitude, longitude, vsv,
vsh) and, optionally, paths.hoggard_lab170_file. Output: ONE netCDF, data/external/
reveal_lid_thickness.nc (lid_km_1p0, lid_km_1p5, lid_km_2p0 and grad_1p5 per 100 km), and a
benchmark table data/derived/reveal_lid_benchmark.csv, through .partial names.

    python scripts/87_reveal_lithosphere.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import ROOT, EARTH_RADIUS_KM, load_config  # noqa: E402

THRESHOLDS = (1.0, 1.5, 2.0)
TOP_KM, BOTTOM_KM, SHALLOW_KM = 60.0, 400.0, 50.0


def lid_thickness(dv, depth, thr):
    """Lid thickness of every column: the deepest depth of ``depth`` reached continuously from the top at
    which the anomaly ``dv`` (per cent) stays at or above ``thr``; SHALLOW_KM where the top layer is
    already below the threshold."""
    fast = dv >= thr
    alive = np.ones(fast.shape[1:], bool)
    run = np.zeros(fast.shape[1:])
    for i, z in enumerate(depth):
        alive &= fast[i]
        run = np.where(alive, z, run)
    return np.where(fast[0], run, SHALLOW_KM)


def gradient_per_100km(field, lat, lon):
    """Horizontal gradient of ``field`` in km of thickness per 100 km, after a one-cell Gaussian smoothing;
    longitude wraps, and the east-west spacing is held above 5% of its equatorial value near the poles."""
    from scipy.ndimage import gaussian_filter
    f = gaussian_filter(field, 1.0, mode=("nearest", "wrap"))
    dlat = np.radians(abs(lat[1] - lat[0])) * EARTH_RADIUS_KM
    dlon = np.radians(abs(lon[1] - lon[0])) * EARTH_RADIUS_KM * np.maximum(np.cos(np.radians(lat)), 0.05)[:, None]
    gy = np.gradient(f, axis=0) / dlat
    gx = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dlon)
    return np.hypot(gx, gy) * 100.0


def benchmark(lids, lat, lon, path):
    """Area-weighted recall, precision and Jaccard index of the lid thicker than 150, 170, 200 and 230 km,
    for each grid of ``lids``, against the inside of the 170 km lithosphere-asthenosphere boundary
    contour of Hoggard et al. (2020) at ``path``."""
    import geopandas as gpd
    from shapely import contains_xy
    from shapely.ops import polygonize, unary_union
    g = gpd.read_file(path)
    geom = unary_union(list(polygonize(unary_union(g.geometry.values))))
    LO, LA = np.meshgrid(lon, lat)
    inside = contains_xy(geom, LO, LA)
    w = np.cos(np.radians(LA))
    rows = []
    for name, th in lids.items():
        for cut in (150, 170, 200, 230):
            a = th >= cut
            tp, fp, fn = (w * (a & inside)).sum(), (w * (a & ~inside)).sum(), (w * (~a & inside)).sum()
            rows.append(dict(grid=name, cutoff_km=cut, recall=tp / (tp + fn), precision=tp / (tp + fp),
                             jaccard=tp / (tp + fp + fn)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    import xarray as xr
    src = Path(cfg["paths"]["reveal_upper_file"])
    d = xr.open_dataset(src)
    z = d["depth"].values
    k = np.where((z >= TOP_KM) & (z <= BOTTOM_KM))[0]
    lat, lon = d["latitude"].values.astype(float), d["longitude"].values.astype(float)
    # isotropic (Voigt) Vs between TOP_KM and BOTTOM_KM, as a per-cent anomaly from the area-weighted
    # mean of each depth
    vs = np.sqrt((2 * d["vsv"].isel(depth=k) ** 2 + d["vsh"].isel(depth=k) ** 2) / 3).values
    w = np.cos(np.radians(lat))[None, :, None]
    mean = (vs * w).sum(axis=(1, 2)) / (w.sum() * vs.shape[2])
    dv = (vs - mean[:, None, None]) / mean[:, None, None] * 100.0
    # one lid-thickness grid per threshold, and the gradient of the 1.5% grid
    lids = {f"lid_km_{t:.1f}".replace(".", "p"): lid_thickness(dv, z[k], t) for t in THRESHOLDS}
    out = xr.Dataset({n: (("latitude", "longitude"), v.astype("float32")) for n, v in lids.items()},
                     coords={"latitude": lat, "longitude": lon})
    out["grad_1p5"] = (("latitude", "longitude"), gradient_per_100km(lids["lid_km_1p5"], lat, lon).astype("float32"))
    out.attrs = dict(source=f"REVEAL (Thrastarson et al. 2024, doi 10.1785/0120230273), {src.name}",
                     method="deepest depth from 60 km at which the Voigt Vs anomaly from the global mean at each depth "
                            "stays at or above the threshold; 50 km where not fast at 60 km",
                     units="km; grad_1p5 in km per 100 km", written_by="scripts/87_reveal_lithosphere.py")
    dest = ROOT / "data" / "external" / "reveal_lid_thickness.nc"
    tmp = dest.with_name(dest.name + ".partial")
    out.to_netcdf(tmp)
    tmp.replace(dest)
    print(f"wrote {dest}")
    # benchmark against the 170 km contour of Hoggard et al. (2020), when the file is configured
    hp = cfg["paths"].get("hoggard_lab170_file")
    if hp and Path(hp).exists():
        b = benchmark(lids, lat, lon, hp)
        bd = Path(cfg["paths"]["derived"]) / "reveal_lid_benchmark.csv"
        bp = bd.with_name(bd.name + ".partial")
        b.to_csv(bp, index=False)
        bp.replace(bd)
        print(b.round(2).to_string())
        print(f"wrote {bd}")


if __name__ == "__main__":
    main()
