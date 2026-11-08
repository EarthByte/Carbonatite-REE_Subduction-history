#!/usr/bin/env python
"""Crustal thickness from the Earth Crustal Model 1 (ECM1; Mooney et al. 2023) as a grid.

ECM1 gives total crustal thickness (Hc: crystalline crust plus sediment, in km) on a 1 x 1 degree
mesh of cell centres (file Hc_ECM1.txt: Numb, Lon, Lat, Hc). The workflow samples grids by
nearest cell (rifts.basins.Grid), which on a 1 degree mesh of centres would put a point up to a
full degree from the value it takes. The model is therefore interpolated bilinearly, periodic in
longitude, onto a 0.1 degree grid of nodes from -180 to 180 and -90 to 90 (poles take the mean of
the cells nearest them), and written as one NetCDF file in km, which scripts/15_rift_fill.py reads
for the crust beneath every point and every rift transect, and for McKenzie's stretching factor
against 32 km of unstretched crust.

    python scripts/51_ecm1_crust.py
Reads paths.crust_source, writes paths.crust_file (config/config.yaml), through a .partial name.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import load_config  # noqa: E402

STEP = 0.1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    src, out = Path(cfg["paths"]["crust_source"]), Path(cfg["paths"]["crust_file"])
    t = pd.read_csv(src, sep=r"\s+")
    # ECM1 is a table of 1 degree cell values (Lon, Lat, Hc): arranged as a lat x lon grid
    lons, lats = np.sort(t["Lon"].unique()), np.sort(t["Lat"].unique())
    g = t.pivot(index="Lat", columns="Lon", values="Hc").loc[lats, lons].to_numpy(float)
    assert g.shape == (180, 360) and np.isfinite(g).all(), "ECM1 Hc is expected complete on 1 degree cells"

    # pad: one column either side for longitude periodicity, the pole rows for latitude
    gx = np.column_stack([g[:, -1], g, g[:, 0]])
    xx = np.r_[lons[0] - 1.0, lons, lons[-1] + 1.0]
    gy = np.vstack([np.full(gx.shape[1], g[0].mean()), gx, np.full(gx.shape[1], g[-1].mean())])
    yy = np.r_[-90.0, lats, 90.0]
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((yy, xx), gy, method="linear")
    # nodes every STEP degrees over the entire globe, interpolated bilinearly from the padded cells
    lon = np.round(np.arange(-180.0, 180.0 + STEP / 2, STEP), 6)
    lat = np.round(np.arange(-90.0, 90.0 + STEP / 2, STEP), 6)
    Y, X = np.meshgrid(lat, lon, indexing="ij")
    z = f(np.column_stack([Y.ravel(), X.ravel()])).reshape(Y.shape).astype("float32")

    import xarray as xr
    ds = xr.Dataset({"z": (("lat", "lon"), z)}, coords={"lat": lat, "lon": lon})
    ds["z"].attrs = dict(units="km", long_name="total crustal thickness (Hc)")
    ds.attrs = dict(source="Earth Crustal Model 1 (ECM1), Mooney et al. (2023), Earth-Science Reviews",
                    note="1 degree cell values interpolated bilinearly to 0.1 degree nodes by scripts/51_ecm1_crust.py")
    out.parent.mkdir(parents=True, exist_ok=True)
    # through a .partial name, so that an interrupted run never leaves a truncated grid
    part = out.with_name(out.name + ".partial")
    ds.to_netcdf(part, format="NETCDF4", engine="netcdf4", encoding={"z": {"zlib": True, "complevel": 4}})
    part.replace(out)
    print(f"ECM1 Hc {g.min():.1f}-{g.max():.1f} km on {g.shape}; wrote {out} {z.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
