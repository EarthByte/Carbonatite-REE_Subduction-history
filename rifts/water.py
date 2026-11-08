"""Slab-water delivery beneath a site, from cumulative grids in the mantle frame.

The grids (one per Myr, Muller2025 mantle frame) hold time-cumulative quantities:
``mean_subducted_water`` (all water carried into subduction zones) and
``mean_slab_outflux`` (water released from the slab above 125 km, the part that can
hydrate the overlying mantle wedge and lithosphere). ``mean_stored_water`` (retained below
125 km) is not used.

For a site at mantle-frame position x(t), two measures are formed for each window T
preceding the formation age t_f:

* fixed-point: C(x(t_f), t_f) - C(x(t_f), t_f + T), water delivered beneath the mantle
  column the site occupies at formation;
* along-track: sum over t in (t_f, t_f + T] of C(x(t), t - 1) - C(x(t), t), water delivered
  beneath the lithosphere that later hosts the site, following it through the mantle frame.

The grids of each variable are split into the water source components (sediment-bound,
crust-bound, serpentinised mantle; a fourth, lithosphere_bottom, is not summed by default),
and the component grids are added at sampling time. Grid values are an areal density on a
0.2-degree mesh smoothed over 300 km; the measures below are therefore relative delivery
indices, compared between deposits and nulls on the same grids.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_AGE = re.compile(r"(\d+(?:\.\d+)?)\s*Ma", re.I)
_ANY = re.compile(r"(\d+(?:\.\d+)?)")


class WaterGrids:
    """Cumulative water grids, one file per Myr.

    Two layouts are understood.

    * Component folders (the Muller2025 grid set): ``<root>/Muller2025_<variable>_300km_<component>/
      cumulative_<short>_<component>_<age>.nc`` with the grid in variable ``z``. A variable is
      the sum of the configured components (sediment_bound, crust_bound, mantle).
    * A flat folder of files with the age in the name and the variable inside the file.

    Either folder may be replaced by one stacked file, ``<folder>.nc``, written by
    tools/stack_grids.py: ``z(age, lat, lon)`` with the same values. Where the folder exists it
    is read; otherwise the stacked file is. With ``folder_template="."`` the stacked file is
    ``<directory>.nc``.
    """

    def __init__(self, directory, pattern="*.nc", variables=("mean_subducted_water", "mean_slab_outflux"),
                 components=None, folder_template="Muller2025_{variable}_300km_{component}",
                 bad_steps=()):
        """Index the grids of every variable (and component, each in its own folder) by age. ``bad_steps``
        are the ages of spurious one-step jumps, corrected when a grid is read."""
        self.dir = Path(directory).expanduser()
        self.variables = list(variables)
        self.components = list(components or [])
        self.files = {}                      # (variable, component) -> {age: path}
        if self.components:
            for v in self.variables:
                for c in self.components:
                    d = self.dir / folder_template.format(variable=v, component=c)
                    self.files[(v, c)] = self._locate(d, pattern)
        else:
            idx = self._locate(self.dir, pattern)
            for v in self.variables:
                self.files[(v, None)] = idx
        if not any(self.files.values()):
            raise FileNotFoundError(f"no grids matching {pattern} under {self.dir}")
        # Spurious one-step jumps (see scripts/10_grid_audit.py): a step at age a is the
        # increment C(a) - C(a+1). It is replaced by the mean of the two neighbouring
        # increments, and the excess is removed from every grid younger than or equal to a.
        self.bad_steps = sorted(int(a) for a in bad_steps)

    @staticmethod
    def _index(d, pattern):
        """{age: path} of the files matching ``pattern`` in ``d``; the age is the last number in the file
        name, and files without one are skipped."""
        out = {}
        for f in sorted(Path(d).glob(pattern)):
            m = _AGE.search(f.stem) or (list(_ANY.finditer(f.stem))[-1] if _ANY.search(f.stem) else None)
            # an age embedded as the last number of the name; files without one are skipped
            if m:
                out[int(round(float(m.group(1))))] = f
        return out

    @classmethod
    def _locate(cls, d, pattern):
        """{age: path} for a folder of per-age files, or {age: (stack, index)} for <folder>.nc."""
        d = Path(d)
        if d.is_dir():
            return cls._index(d, pattern)
        stack = d if d.suffix == ".nc" and d.is_file() else Path(f"{d}.nc")
        if not stack.is_file():
            return {}
        import netCDF4
        with netCDF4.Dataset(stack) as nc:
            ages = [int(a) for a in nc["age"][:]]
        return {a: (stack, i) for i, a in enumerate(ages)}

    @staticmethod
    @lru_cache(maxsize=None)
    def _open_stack(path):
        import xarray as xr
        return xr.open_dataset(path)

    @property
    def ages(self):
        sets = [set(v) for v in self.files.values() if v]
        return np.array(sorted(set.intersection(*sets)))

    @lru_cache(maxsize=64)
    def _raw(self, var, comp, age):
        """One grid as stored (variable ``var``, component ``comp``, age ``age``), with the dimensions
        renamed lon and lat and latitude ascending."""
        import xarray as xr
        src = self.files[(var, comp)][age]
        if isinstance(src, tuple):                       # stacked file: one age of z(age, lat, lon)
            da = self._open_stack(src[0])["z"].isel(age=src[1]).load().drop_vars("age")
        else:
            with xr.open_dataset(src) as ds:
                name = "z" if "z" in ds else (var if var in ds else
                                              [k for k in ds.data_vars if ds[k].ndim == 2][0])
                da = ds[name].load()
        lon = [c for c in da.dims if c.lower().startswith(("lon", "x"))][0]
        lat = [c for c in da.dims if c.lower().startswith(("lat", "y"))][0]
        return da.rename({lon: "lon", lat: "lat"}).sortby("lat")

    @lru_cache(maxsize=16)
    def _excess(self, var, comp, a):
        g = lambda t: self._raw(var, comp, t)  # noqa: E731
        step = g(a) - g(a + 1)
        background = 0.5 * ((g(a - 1) - g(a)) + (g(a + 1) - g(a + 2)))
        return step - background

    def _grid(self, var, comp, age):
        da = self._raw(var, comp, age)
        for a in self.bad_steps:
            if age <= a and all(t in self.files[(var, comp)] for t in (a - 1, a, a + 1, a + 2)):
                da = da - self._excess(var, comp, a)
        return da

    def sample(self, age, var, lon, lat):
        """Value of ``var`` at ``age`` (rounded to 1 Myr) at each longitude and latitude, summed over
        components. Bilinear, with the nearest cell beyond the outermost cell centres; NaN if the age
        has no grid."""
        import xarray as xr
        age = int(round(age))
        lon = np.asarray(lon, float)
        lat = np.asarray(lat, float)
        total = np.zeros(len(lon))
        for (v, c), idx in self.files.items():
            if v != var:
                continue
            if age not in idx:
                return np.full(len(lon), np.nan)
            g = self._grid(v, c, age)
            x = lon % 360 if float(g.lon.max()) > 180 else lon
            val = g.interp(lon=xr.DataArray(x, dims="p"), lat=xr.DataArray(lat, dims="p")).values
            gap = ~np.isfinite(val)
            if gap.any():
                # beyond the outermost cell centres of a cell-registered grid (the dateline seam, the
                # poles): the nearest cell
                val[gap] = g.sel(lon=xr.DataArray(x[gap], dims="p"), lat=xr.DataArray(lat[gap], dims="p"),
                                 method="nearest").values
            total += val
        return total


def water_metrics(points: pd.DataFrame, mantle, grids: WaterGrids, windows, along_track=True,
                  progress=True, stock=False):
    """Water metrics for points with columns pt_id, lon, lat, pid, age_ma.

    The along-track sum walks once through grid time, sampling every point whose window
    covers that time, so each grid is read once per variable.
    """
    out = pd.DataFrame({"pt_id": points["pt_id"].to_numpy()})
    tf = np.rint(points["age_ma"].to_numpy())
    windows = sorted(windows)
    tmax_grid = grids.ages.max()
    for var in grids.variables:
        if stock:
            # cumulative amount beneath the site at formation (grids start at their own first age)
            st = np.full(len(points), np.nan)
            for a in np.unique(tf[np.isfinite(tf)]):
                rows = np.nonzero(tf == a)[0]
                p = points.iloc[rows]
                x, y = mantle.reconstruct_points(p["lon"], p["lat"], p["pid"], a)
                st[rows] = grids.sample(a, var, x, y)
            out[f"{var}_stock"] = st
        for T in windows:
            fixed = np.full(len(points), np.nan)
            for a in np.unique(tf[np.isfinite(tf)]):
                rows = np.nonzero(tf == a)[0]
                p = points.iloc[rows]
                x, y = mantle.reconstruct_points(p["lon"], p["lat"], p["pid"], a)
                fixed[rows] = grids.sample(a, var, x, y) - grids.sample(a + T, var, x, y)
            out[f"{var}_fixed_{T}myr"] = fixed
        if not along_track:
            continue
        track = {T: np.zeros(len(points)) for T in windows}
        covered = {T: np.zeros(len(points), int) for T in windows}
        t_lo = int(np.nanmin(tf)) + 1
        t_hi = int(min(np.nanmax(tf) + windows[-1], tmax_grid))
        for t in range(t_lo, t_hi + 1):
            rows = np.nonzero((tf < t) & (t <= tf + windows[-1]))[0]
            if not len(rows):
                continue
            p = points.iloc[rows]
            x, y = mantle.reconstruct_points(p["lon"], p["lat"], p["pid"], t)
            inc = grids.sample(t - 1, var, x, y) - grids.sample(t, var, x, y)
            good = np.isfinite(inc)
            for T in windows:
                sel = (t <= tf[rows] + T) & good
                track[T][rows[sel]] += inc[sel]
                covered[T][rows[sel]] += 1
            if progress and t % 100 == 0:
                print(f"    {var}: t = {t} Ma", flush=True)
        for T in windows:
            # a window that runs past the grids or off the model is reported as missing
            full = covered[T] >= np.minimum(T, tmax_grid - tf)
            out[f"{var}_track_{T}myr"] = np.where(full & (tf + T <= tmax_grid), track[T], np.nan)
    return out
