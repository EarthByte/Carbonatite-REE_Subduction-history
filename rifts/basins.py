"""Sediment fill and crustal thinning of a rift, and the syn-rift share of the fill.

Two global grids carry this: Bird and Mooney's GST-1 total sediment thickness and the total
crustal thickness of the Earth Crustal Model 1 (ECM1; Mooney et al. 2023). They give, at any
point, the fill above it and the crust beneath it, and with McKenzie's (1978) stretching factor
against 32 km of unstretched continental crust they give how far that crust was thinned. Every deposit and null point takes all three, whether
or not it falls inside a mapped basin.

Total thickness is not stratigraphy, but the syn-rift and post-rift parts of a rift's fill leave
different shapes across it. Syn-rift sediment fills the accommodation the faults make, so it
thickens toward the rift axis and dies at the shoulders; post-rift sediment settles on a
thermally subsiding, widening basin and drapes across the whole width. A wedge-plus-drape fit to
thickness along a rift-normal transect separates the two, following the method of the T55
tutorial of the group's GPlately workflows. The method is first-order: it cannot see hiatus,
inversion or a post-rift depocentre that migrated, and a rift whose transects are badly fitted
says so through the residual reported with the result. Read the wedge share as a relative index
across rifts, not as a stratigraphic thickness.

McKenzie's model gives the same split independently, as physics rather than shape: initial
subsidence at rifting against thermal subsidence since, the latter still accumulating with a
62 Myr time constant, so the ratio predicted for a rift depends on its age.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# McKenzie (1978) pure shear, with the constants of the T55 tutorial
T_L_KM = 125.0          # initial lithosphere thickness
RHO_M, RHO_C, RHO_W = 3300.0, 2800.0, 1030.0
ALPHA_V = 3.28e-5       # thermal expansion coefficient (1/K)
T_M = 1333.0            # asthenospheric temperature above the surface (K)
T_C0_KM = 32.0          # unstretched continental crust
TAU_THERMAL_MYR = 62.8


def stretching_factor(crust_km, t_c0=T_C0_KM):
    """McKenzie beta = unstretched crust / present crust, undefined where crust is not resolved."""
    c = np.asarray(crust_km, float)
    return np.where(c > 0.5, t_c0 / c, np.nan)


def initial_subsidence(beta):
    """Syn-rift (air-loaded, unfilled) subsidence in metres, McKenzie (1978) eq. 4."""
    b = np.asarray(beta, float)
    a_l = T_L_KM * 1000.0
    t_c = T_C0_KM * 1000.0
    term1 = (RHO_M - RHO_C) * (t_c / a_l) * (1 - ALPHA_V * T_M * t_c / a_l)
    term2 = ALPHA_V * T_M * RHO_M / 2
    num = a_l * (term1 - term2) * (1 - 1.0 / b)
    den = RHO_M * (1 - ALPHA_V * T_M) - RHO_W
    return num / den


def thermal_subsidence(beta, t_myr):
    """Post-rift subsidence in metres, ``t_myr`` after rifting."""
    b, t = np.asarray(beta, float), np.asarray(t_myr, float)
    tl_m = T_L_KM * 1000.0
    e = 4 * tl_m * RHO_M * ALPHA_V * T_M / (np.pi ** 2 * (RHO_M - RHO_W))
    with np.errstate(divide="ignore", invalid="ignore"):
        return e * (b / np.pi) * np.sin(np.pi / b) * (1 - np.exp(-t / TAU_THERMAL_MYR))


class Grid:
    """Nearest-cell lookup into a global lon/lat grid, with the poles and dateline handled."""

    def __init__(self, path, var=None, scale=1.0):
        """Load a 2-D grid (variable ``var``, else z, else the first variable) multiplied by ``scale``,
        with latitude ascending."""
        import xarray as xr
        ds = xr.open_dataset(path)
        da = ds[var] if var else (ds["z"] if "z" in ds else list(ds.data_vars.values())[0])
        self.z = np.asarray(da.values, float) * scale
        ynm = "lat" if "lat" in da.dims else da.dims[0]
        xnm = "lon" if "lon" in da.dims else da.dims[1]
        self.y = np.asarray(da[ynm].values, float)
        self.x = np.asarray(da[xnm].values, float)
        if self.y[0] > self.y[-1]:
            self.y, self.z = self.y[::-1], self.z[::-1, :]

    def inside(self, geom):
        """Grid values whose cell centres lie inside a polygon (any part of a multipolygon)."""
        import shapely
        x0, y0, x1, y1 = geom.bounds
        xs = self.x if self.x.max() <= 180.0 else ((self.x + 180.0) % 360.0) - 180.0
        i = np.where((self.y >= y0 - 0.2) & (self.y <= y1 + 0.2))[0]
        k = np.where((xs >= x0 - 0.2) & (xs <= x1 + 0.2))[0]
        if not len(i) or not len(k):
            return np.array([])
        LO, LA = np.meshgrid(xs[k], self.y[i])
        m = shapely.contains_xy(geom, LO, LA)
        return self.z[np.ix_(i, k)][m]

    def sample(self, lon, lat):
        """Value of the cell containing each longitude and latitude (longitude wrapped to the convention of
        the grid); NaN for non-finite coordinates."""
        lon = np.asarray(lon, float)
        lat = np.asarray(lat, float)
        x = np.where(self.x.max() > 180.0, lon % 360.0, ((lon + 180.0) % 360.0) - 180.0)
        i = np.clip(np.searchsorted(self.x, x) - 1, 0, len(self.x) - 1)
        j = np.clip(np.searchsorted(self.y, lat) - 1, 0, len(self.y) - 1)
        out = self.z[j, i]
        return np.where(np.isfinite(lon) & np.isfinite(lat), out, np.nan)


def _wedge_drape(x_km, x_c, half_width_km, amp, drape):
    """Parabolic syn-rift wedge centred at ``x_c`` over a uniform post-rift drape."""
    u = (np.asarray(x_km, float) - x_c) / max(half_width_km, 1e-6)
    return amp * np.maximum(0.0, 1.0 - u ** 2) + drape


def fit_wedge_drape(x_km, thickness_m):
    """Least-squares wedge and drape of one transect, following the T55 tutorial.

    The wedge's centre, half-width and amplitude and the drape thickness are all free, because a
    rift's sedimentary depocentre is wider than its mapped structures and need not be centred on
    them. Distances are kilometres along the transect and thicknesses metres. Returns the fit and
    its residual relative to the thickness range, which says whether the shape fitted at all.
    """
    from scipy.optimize import least_squares
    t = np.asarray(thickness_m, float)
    x = np.asarray(x_km, float)
    ok = np.isfinite(t) & np.isfinite(x)
    if ok.sum() < 8:
        return None
    t, x = t[ok], x[ok]
    lo, hi = float(np.min(t)), float(np.max(t))
    span = max(hi - lo, 1.0)
    length = float(np.ptp(x))
    x0 = [float(x[np.argmax(t)]), length / 4.0, span, lo]
    bounds = ([x.min(), length / 50.0, 0.0, 0.0], [x.max(), length, np.inf, np.inf])
    try:
        r = least_squares(lambda p: t - _wedge_drape(x, *p), x0=x0, bounds=bounds, max_nfev=2000)
    except Exception:
        return None
    rms = float(np.sqrt(np.mean(r.fun ** 2)))
    return dict(x_c_km=float(r.x[0]), half_width_km=float(r.x[1]), wedge_m=float(r.x[2]),
                drape_m=float(r.x[3]), resid_rms_m=rms, resid_rel=rms / span, n=int(ok.sum()))


def transect_points(spine, factor=3.0, n=41, min_half_km=75.0):
    """Points along a rift-normal transect through every spine bin.

    The transect runs flank to flank through the spine point, ``factor`` times the local rift
    width and never shorter than ``2 * min_half_km``, so its ends sit on crust the rift did not
    thin and there is room for a depocentre wider than the mapped structures. Returns the
    longitudes, latitudes, the signed distance from the axis in kilometres, and the bin index.
    """
    c = np.column_stack([spine["c_lon"], spine["c_lat"]])
    left = np.column_stack([spine["left_lon"], spine["left_lat"]])
    right = np.column_stack([spine["right_lon"], spine["right_lat"]])
    u = np.linspace(-1.0, 1.0, n)
    half = np.asarray(spine["half_km"], float)
    lon, lat, xx, ib = [], [], [], []
    for k in range(len(c)):
        reach = max(factor * max(half[k], 1.0), min_half_km)          # km each side of the axis
        d = 0.5 * (left[k] - right[k])                                # half the flank-to-flank step
        scale = reach / max(np.hypot(*(d * [np.cos(np.radians(c[k][1])), 1.0])) * 111.19, 1e-6)
        p = c[k][None, :] + (u * scale)[:, None] * d[None, :]
        lon.append(p[:, 0])
        lat.append(p[:, 1])
        xx.append(u * reach)
        ib.append(np.full(n, k))
    return (np.concatenate(lon), np.concatenate(lat), np.concatenate(xx), np.concatenate(ib))


def rift_fill(spine, sed_grid, crust_grid, factor=3.0, n=41, min_bins=1, min_half_km=75.0):
    """Syn-rift and post-rift thickness of one rift, from wedge-plus-drape fits along it.

    Each spine bin gives one rift-normal transect; the rift's values are the length-weighted
    medians over the bins that fit. ``syn_share`` is the wedge over wedge plus drape.
    """
    lon, lat, xkm, ib = transect_points(spine, factor, n, min_half_km)
    sed = sed_grid.sample(lon, lat)
    fits = []
    for k in np.unique(ib):
        m = ib == k
        f = fit_wedge_drape(xkm[m], sed[m])
        if f:
            f["w"] = float(spine["w_km"][k])
            fits.append(f)
    if len(fits) < min_bins:
        return {}
    w = np.array([f["w"] for f in fits], float)

    def wmed(key):
        x = np.array([f[key] for f in fits], float)
        o = np.argsort(x)
        cw = np.cumsum(w[o]) - 0.5 * w[o]
        return float(np.interp(0.5 * w.sum(), cw, x[o]))

    wedge, drape = wmed("wedge_m"), wmed("drape_m")
    crust = crust_grid.sample(spine["c_lon"], spine["c_lat"])
    beta = stretching_factor(crust)
    return dict(n_transects=len(fits), fill_wedge_m=wedge, fill_drape_m=drape,
                fill_half_width_km=wmed("half_width_km"),
                fill_total_m=wedge + drape,
                syn_share=wedge / (wedge + drape) if wedge + drape > 0 else np.nan,
                fit_resid_rel=wmed("resid_rel"),
                crust_km=float(np.nanmedian(crust)), beta=float(np.nanmedian(beta)))


def mckenzie_prediction(beta, onset_ma, end_ma):
    """Predicted syn-rift and post-rift subsidence of a rift, and the syn-rift share.

    The thermal part is taken from the end of rifting to the present, so an old rift has
    completed it and a young one has not.
    """
    beta = np.asarray(beta, float)
    end = np.where(np.isfinite(end_ma), end_ma, 0.0)
    since = np.clip(np.where(np.isfinite(onset_ma), np.maximum(end, 0.0), np.nan), 0, None)
    si = initial_subsidence(beta)
    st = thermal_subsidence(beta, since)
    tot = si + st
    return pd.DataFrame(dict(pred_syn_m=si, pred_post_m=st,
                             pred_syn_share=np.where(tot > 0, si / tot, np.nan)))


# How the published thickness of each rift of known stratigraphy (config/fill_validation_sections.csv,
# column total_basis) compares with the greatest GST-1 thickness inside its outline: "total" and
# "total with volcanic rocks" are the total fill of the rift itself, "sum" the sum of the published
# components; the others are not a total fill of the rift (a partial succession, the adjacent basin,
# a younger cover or mostly volcanic infill), so no recovered share is computed for them.
COMPARABLE_TOTAL = ("total", "sum", "total with volcanic rocks")


def published_totals(sec):
    """Published total fill (low, high, in m) and its basis for each rift of known stratigraphy."""
    rows = []
    for rid, g in sec.groupby("outline_id", sort=False):
        b = g["total_basis"].iloc[0]
        if b == "sum":
            lo, hi = float(g["low_m"].sum()), float(g["high_m"].sum())
        else:
            lo, hi = float(g["low_m"].max()), float(g["high_m"].max())
        rows.append(dict(outline_id=rid, rift=g["rift"].iloc[0], basis=b, low_m=lo, high_m=hi,
                         comparable=b in COMPARABLE_TOTAL))
    return pd.DataFrame(rows).set_index("outline_id")


def rift_fill_table(der, tag="full"):
    """The fill of every rift as the rift papers use it: the components of scripts/15 (one start per
    transect) replaced, where scripts/67 has run, by its default variant (the same transects refitted
    from six starting points, the best solution kept) so that one fit serves the tables, the figures,
    the age test and the deposit comparisons. ``fit_source`` records which fit a row carries."""
    f = pd.read_csv(der / f"rift_fill_{tag}.csv")
    vf = der / f"rift_fill_validation_{tag}.csv"
    f["fit_source"] = "scripts/15"
    if vf.exists():
        v = pd.read_csv(vf)[["outline_id", "share", "drape", "wedge"]].set_index("outline_id")
        m = f["outline_id"].map(v["drape"]).notna()
        f.loc[m, "fill_drape_m"] = f.loc[m, "outline_id"].map(v["drape"]).to_numpy()
        f.loc[m, "fill_wedge_m"] = f.loc[m, "outline_id"].map(v["wedge"]).to_numpy()
        f.loc[m, "syn_share"] = f.loc[m, "outline_id"].map(v["share"]).to_numpy()
        f.loc[m, "fill_total_m"] = f.loc[m, "fill_wedge_m"] + f.loc[m, "fill_drape_m"]
        f.loc[m, "fit_source"] = "scripts/67"
    return f
