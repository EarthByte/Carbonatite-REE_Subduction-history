#!/usr/bin/env python
"""Mantle-driven dynamic topography beneath every point through time, from seven mantle-flow models.

Each model publishes its dynamic topography in the plate frame: each grid holds the dynamic topography
at a past age at the present-day coordinates of the continental blocks, every 20 Myr. A site is
therefore sampled at its present-day position at any age, without reconstruction, and its history is
the uplift or subsidence that mantle flow imposed on that piece of continent. This gives a
time-dependent trigger the rest of the workflow lacks: whether the mantle was lifting the site when
the magma formed.

Models:
  m1-m6  the six CitcomS cases of Dhungana & Flament (Deep Earth origin of the Great Unconformity,
         Zenodo 10.5281/zenodo.20603294, v1.1.0): M1 gld504, M2 gld582, M3 gld583 (the paleomagnetic-
         frame twin of M2), M4 gld486, M5 gld434, M6 gld428; 0-1000 Ma, 0.5 degrees, processed alike
         (latitude correction and removal of the global mean at each age). They are read from one
         tarball per model (tools/zenodo_zip_extract.py) in paths.dt_dhungana2025_dir.
  young  gld428 as published by Young et al. (2022, Earth Planet. Sci. Lett. 584, 117451), 0-980 Ma,
         0.25 degrees, without that processing (paths.dt_young2022_dir).

For every point of data/derived/carbonatite_features_<tag> (all classes and their reference points),
at its age t (linear interpolation between grids), and for every model:

  dt_m              dynamic topography at t (m)
  dt_uplift_20      (DT(t) - DT(t + 20 Myr)) / 20 Myr, m/Myr: positive while mantle flow lifts the site
  dt_uplift_60      the same over 60 Myr
  dt_uplift_after_20 (DT(t - 20) - DT(t)) / 20: whether uplift continued after formation
  dt_rise_200       DT(t) minus the lowest DT over the 200 Myr before t: the uplift since the last low

Plume propensity (M1-M6 only): the plumes of these forward models rise freely from the margins of,
and ridges within, each model's basal piles and are not assimilated, so they are not placed where the
geological record puts plumes; they are expected near a site statistically, in roughly the right
regions. The measure is therefore a neighbourhood and a window rather than the value at the site:

  dt_high_p500      share of the grids from 60 Myr before to 20 Myr after t in which dynamic topography
                    above the 90th percentile of that grid's continental cells lies within 500 km
  dt_rise_p500      the same for a 20-Myr uplift rate above the 90th percentile of that grid's rates
  dt_high_p1000, dt_rise_p1000   the same within 1,000 km

Thresholds are relative to each model and age, so the shares average over models of different
amplitude; the unsuffixed propensity columns are the mean over M1-M6 (a probability), and the
low-dynamic-topography measures above stay medians, because downwellings follow the imposed plate
history and are placed where the reconstruction puts subduction.

The unsuffixed columns are the median over M1-M6, the ensemble that carries the analysis; the columns
suffixed _m1 ... _m6 and _young are each model alone, and dt_uplift_20_agree is the share of the six
models that agree in sign with the ensemble median. A value needs every grid it reads; a point on an
empty grid cell takes the nearest cell with a value within 0.5 degrees, otherwise it is missing.

Output: ONE table, data/derived/dynamic_topography_<tag>.parquet, through write_table.

    python scripts/88_dynamic_topography.py --tag full
"""
import argparse
import io
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import load_config, read_table, write_table  # noqa: E402

SEARCH_DEG = 0.5
DHUNGANA = {"m1": "gld504", "m2": "gld582", "m3": "gld583", "m4": "gld486", "m5": "gld434", "m6": "gld428"}
MEASURES = ("dt_m", "dt_uplift_20", "dt_uplift_60", "dt_uplift_after_20", "dt_rise_200")
PROPENSITY = ("dt_high_p500", "dt_rise_p500", "dt_high_p1000", "dt_rise_p1000")


def open_grid(src):
    """A plate-frame grid as (lat, lon, values) from a path or from bytes (a tarball member)."""
    import xarray as xr
    if isinstance(src, (bytes, bytearray)):
        try:
            d = xr.open_dataset(io.BytesIO(src), engine="h5netcdf")
        except Exception:
            import netCDF4
            d = xr.open_dataset(xr.backends.NetCDF4DataStore(netCDF4.Dataset("grid", memory=bytes(src))))
    else:
        d = xr.open_dataset(src)
    z = d["z"]
    lat_name = next(c for c in z.dims if c in ("lat", "latitude", "y"))
    lon_name = next(c for c in z.dims if c in ("lon", "longitude", "x"))
    z = z.transpose(lat_name, lon_name)
    glat, glon, v = z[lat_name].values.astype(float), z[lon_name].values.astype(float), z.values.astype(float)
    if glat[0] > glat[-1]:
        glat, v = glat[::-1], v[::-1]
    if np.isclose(glon[-1] - glon[0], 360.0):
        glon, v = glon[:-1], v[:, :-1]
    return glat, glon, v


def young_grids(folder):
    files = {}
    for f in Path(folder).glob("gld428_PlateFrameGrid*.nc"):
        m = re.search(r"Grid(\d+)\.nc$", f.name)
        if m:
            files[int(m.group(1))] = f
    return dict(sorted(files.items()))


def tar_grids(path):
    """{age: bytes} of the plate-frame grids in one tarball."""
    out = {}
    with tarfile.open(path) as t:
        for mem in t.getmembers():
            m = re.search(r"Grid(\d+)\.nc$", mem.name)
            if m and mem.isfile():
                out[int(m.group(1))] = t.extractfile(mem).read()
    return dict(sorted(out.items()))


def sample_all(grids, lon, lat):
    """Dynamic topography of every point at every grid age (points x ages)."""
    ages = np.array(list(grids), float)
    out = np.full((len(lon), len(ages)), np.nan)
    cells = None
    for k, (age, src) in enumerate(grids.items()):
        glat, glon, v = open_grid(src)
        if cells is None:
            dlat, dlon = glat[1] - glat[0], glon[1] - glon[0]
            i0 = np.clip(np.round((lat - glat[0]) / dlat).astype(int), 0, len(glat) - 1)
            j0 = np.round(((lon - glon[0]) % 360.0) / dlon).astype(int) % len(glon)
            r = int(np.ceil(SEARCH_DEG / dlat))
            offs = sorted(((di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)),
                          key=lambda o: o[0] ** 2 + o[1] ** 2)
            cells = (i0, j0, offs, len(glat), len(glon))
        i0, j0, offs, nla, nlo = cells
        val = np.full(len(lon), np.nan)
        for di, dj in offs:
            need = ~np.isfinite(val)
            if not need.any():
                break
            ii = np.clip(i0[need] + di, 0, nla - 1)
            jj = (j0[need] + dj) % nlo
            val[need] = v[ii, jj]
        out[:, k] = val
        print(f"    {age:.0f} Ma", end="\r", flush=True)
    print()
    return ages, out


def disk_cells(lon, lat, glat, glon, radius_km):
    """Flat indices of the grid cells within radius_km of every point, and the start of each point's run."""
    from rifts.common import EARTH_RADIUS_KM
    dlat, dlon = glat[1] - glat[0], glon[1] - glon[0]
    nla, nlo = len(glat), len(glon)
    rdeg = np.degrees(radius_km / EARTH_RADIUS_KM)
    idx, starts = [], []
    for x, y in zip(lon, lat):
        starts.append(sum(len(i) for i in idx))
        ii = np.nonzero(np.abs(glat - y) <= rdeg)[0]
        half = rdeg / max(np.cos(np.radians(min(abs(y) + rdeg, 89.5))), 0.02)
        nj = min(int(np.ceil(half / dlon)), nlo // 2)
        j0 = int(np.round(((x - glon[0]) % 360.0) / dlon))
        jj = (j0 + np.arange(-nj, nj + 1)) % nlo
        I, J = np.meshgrid(ii, jj, indexing="ij")
        la1, lo1 = np.radians(glat[I]), np.radians(glon[J])
        la0, lo0 = np.radians(y), np.radians(x)
        h = np.sin((la1 - la0) / 2) ** 2 + np.cos(la0) * np.cos(la1) * np.sin((lo1 - lo0) / 2) ** 2
        keep = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1))) <= radius_km
        flat = np.unique((I[keep] * nlo + J[keep]).ravel())
        if flat.size == 0:
            flat = np.array([int(np.clip(np.round((y - glat[0]) / dlat), 0, nla - 1)) * nlo + j0 % nlo])
        idx.append(flat)
    return np.concatenate(idx), np.array(starts)


def plume_propensity(grids, lon, lat, t, radii=(500, 1000), lead=60, lag=20, pct=90):
    """Share of the grids from t + lead to t - lag Myr in which a high or a rising dynamic topography
    arrives within each radius of the point.

    Plumes in forward mantle-flow models rise freely from the margins of, and ridges within, the
    model's own basal thermochemical piles; they are not placed where the geological record puts
    plumes. A model plume is therefore expected near a site in a statistical sense, not at its exact
    place and time, and the measure is a neighbourhood and a window rather than the value at the site.
    "High" is above the pct-th percentile of all continental cells of the same grid, "rising" is a 20-Myr
    uplift rate above the pct-th percentile of that grid's uplift rates, so the thresholds are relative
    to each model and age and the measure can be averaged over models of different amplitude.
    """
    ages = np.array(list(grids), float)
    D = None
    for k, src in enumerate(grids.values()):
        glat, glon, v = open_grid(src)
        if D is None:
            D = np.full((len(ages),) + v.shape, np.nan, np.float32)
        D[k] = v
    n = len(ages)
    high = np.zeros(D.shape, bool)
    rise = np.zeros(D.shape, bool)
    valid_high = np.isfinite(D).reshape(n, -1).any(axis=1)
    valid_rise = np.zeros(n, bool)
    for k in range(n):
        f = D[k][np.isfinite(D[k])]
        if f.size:
            high[k] = D[k] > np.percentile(f, pct)
        older = np.nonzero(np.isclose(ages, ages[k] + 20))[0]
        if older.size:
            u = (D[k] - D[older[0]]) / 20.0
            fu = u[np.isfinite(u)]
            if fu.size:
                rise[k] = u > np.percentile(fu, pct)
                valid_rise[k] = True
    out = {}
    H, R = high.reshape(n, -1), rise.reshape(n, -1)
    for rad in radii:
        flat, starts = disk_cells(lon, lat, glat, glon, rad)
        hit_h = np.zeros((len(lon), n), bool)
        hit_r = np.zeros((len(lon), n), bool)
        for k in range(n):
            hit_h[:, k] = np.maximum.reduceat(H[k][flat].astype(np.int8), starts) > 0
            hit_r[:, k] = np.maximum.reduceat(R[k][flat].astype(np.int8), starts) > 0
        for name, hit, valid in (("high", hit_h, valid_high), ("rise", hit_r, valid_rise)):
            win = (ages[None, :] >= (t - lag)[:, None]) & (ages[None, :] <= (t + lead)[:, None]) & valid[None, :]
            cnt = win.sum(axis=1)
            share = np.where(cnt >= 3, (hit & win).sum(axis=1) / np.maximum(cnt, 1), np.nan)
            out[f"dt_{name}_p{rad}"] = share
    return out


def at(ages, series, t):
    """Linear interpolation of each row of ``series`` at the ages t (NaN outside the grid ages)."""
    out = np.full(len(t), np.nan)
    ok = np.isfinite(t) & (t >= ages[0]) & (t <= ages[-1])
    k = np.clip(np.searchsorted(ages, t[ok]) - 1, 0, len(ages) - 2)
    a0, a1 = ages[k], ages[k + 1]
    w = (t[ok] - a0) / (a1 - a0)
    rows = np.nonzero(ok)[0]
    out[ok] = (1 - w) * series[rows, k] + w * series[rows, k + 1]
    return out


def measures(ages, S, t):
    """Dynamic-topography measures at the sampled ages ``t`` from the site histories ``S`` (a row per site,
    a column per age of ``ages``): elevation (m), mean uplift rate over the 20 and 60 Myr before and the
    20 Myr after (m per Myr), and the rise above the lowest elevation of the preceding 200 Myr (m)."""
    dt = at(ages, S, t)
    out = {"dt_m": dt,
           "dt_uplift_20": (dt - at(ages, S, t + 20)) / 20.0,
           "dt_uplift_60": (dt - at(ages, S, t + 60)) / 60.0,
           "dt_uplift_after_20": (at(ages, S, t - 20) - dt) / 20.0}
    lows = np.full(len(t), np.nan)
    for i in np.nonzero(np.isfinite(dt) & (t + 200 <= ages[-1]))[0]:
        sel = (ages > t[i]) & (ages <= t[i] + 200)
        row = S[i, sel]
        lows[i] = np.nanmin(row) if np.isfinite(row).any() else np.nan
    out["dt_rise_200"] = dt - lows
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--source", choices=("carbonatite", "features"), default="carbonatite",
                    help="points from carbonatite_features_<tag> (scripts/83) or from every lag-0 point of "
                         "features_<tag> (the base-metal tags: full, barren, inrift)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    der = cfg["paths"]["derived"]
    # one row per point at its sampled age (lag 0, draw 0)
    if a.source == "features":
        p = read_table(der / f"features_{a.tag}")
        p = p[(p["lag_myr"] == 0) & (p["draw"] == 0)][["pt_id", "lon0", "lat0", "age_ma"]].drop_duplicates("pt_id")
    else:
        p = read_table(der / f"carbonatite_features_{a.tag}")[["pt_id", "lon0", "lat0", "age_ma"]].drop_duplicates("pt_id")
    lon, lat, t = (p[c].to_numpy(float) for c in ("lon0", "lat0", "age_ma"))
    out = pd.DataFrame({"pt_id": p["pt_id"].to_numpy()})
    # the Dhungana & Flament mantle-flow models (plate frame, 20 Myr steps) and Young et al. (2022)
    sources = {key: Path(cfg["paths"]["dt_dhungana2025_dir"]) / f"{run}_plateframe_20myr.tar"
               for key, run in DHUNGANA.items()}
    sources["young"] = cfg["paths"]["dt_young2022_dir"]
    for key, src in sources.items():
        grids = young_grids(src) if key == "young" else tar_grids(src)
        print(f"  {key}: {len(grids)} plate-frame grids, {min(grids)}-{max(grids)} Ma; {len(p)} points", flush=True)
        ages, S = sample_all(grids, lon, lat)
        for name, v in measures(ages, S, t).items():
            out[f"{name}_{key}"] = v
        if key != "young":
            for name, v in plume_propensity(grids, lon, lat, t).items():
                out[f"{name}_{key}"] = v
    # summary values: the median of the Dhungana models for each measure, their mean for the plume
    # propensities; Young et al. (2022) stays a separate column
    for name in MEASURES:
        out[name] = out[[f"{name}_{k}" for k in DHUNGANA]].median(axis=1, skipna=False)
    for name in PROPENSITY:
        out[name] = out[[f"{name}_{k}" for k in DHUNGANA]].mean(axis=1, skipna=False)
    # share of the Dhungana models whose 20 Myr uplift has the sign of the median
    sign = np.sign(out[[f"dt_uplift_20_{k}" for k in DHUNGANA]].to_numpy())
    out["dt_uplift_20_agree"] = (sign == np.sign(out["dt_uplift_20"].to_numpy())[:, None]).mean(axis=1)
    out.loc[out["dt_uplift_20"].isna(), "dt_uplift_20_agree"] = np.nan
    cols = [c for c in out.columns if c != "pt_id"]
    print(out[cols].corr(method="spearman").loc[[f"dt_uplift_20_{k}" for k in sources], "dt_uplift_20"].round(2).to_string())
    print(out[list(MEASURES) + ["dt_uplift_20_agree"] + list(PROPENSITY)].describe().round(2).to_string())
    print(f"wrote {write_table(out, der / f'dynamic_topography_{a.tag}')}")


if __name__ == "__main__":
    main()
