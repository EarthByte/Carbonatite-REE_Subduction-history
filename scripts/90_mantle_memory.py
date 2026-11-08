#!/usr/bin/env python
"""Subducted carbon and water held in the mantle with a finite memory: a diffusing reservoir.

The cumulative grids (scripts/03) treat the mantle beneath the plates as a closed reservoir that only
fills: carbon delivered 1.5 Gyr ago counts as much, and stays as concentrated, as carbon delivered
last year. The mantle is not closed. Material delivered by slabs is stirred by time-dependent convection,
which Richter, Daly & Nataf (1982, EPSL 60, 178) showed disperses passive material horizontally as an
effective diffusion: in internally heated convection at a Rayleigh number of 1e6 the horizontal variance
of an initially concentrated set of particles grows as sigma^2 = kappa_e t with kappa_e ~ 0.3 cm^2/s
(~950 km^2/Myr; their eqs 10-11). Following that definition, a delivery spreads to a per-axis standard
deviation sqrt(kappa_e tau): ~390 km after 150 Myr and ~710 km after 500 Myr at kappa_e = 1,000 km^2/Myr. Spreading conserves mass, so the concentration of an old
delivery falls as its footprint grows; the memory of the reservoir is finite without an extra
parameter. An optional first-order loss (export to the lower mantle, return to the surface) with
e-folding time tau_loss shortens it further. Subduction water signatures survive in the mid-ocean-ridge
mantle for at least 400 Myr, with no detectable decline of H2O/Ce with hydration age (Dixon, Müller,
Dutkiewicz & Mather, Geology, in revision), so the shortest loss time tested is 400 Myr.

The reservoir M(x, t) is marched forward in time in the mantle frame, from the start of the grids
to the present, in 1 Myr steps:

    M(t) = exp(-1/tau_loss) * G(kappa_e) * M(t + 1)  +  S(t)

where G(v) is the heat kernel on the sphere that spreads a delivery to a per-axis variance v (km^2),
and S(t) = C(t) - C(t + 1) is the delivery in the step from the cumulative grids C (spurious steps
corrected as in scripts/03). The reservoirs are held as spherical-harmonic coefficients (degree 224, the
0.4-degree Driscoll-Healy grid), in which the heat kernel is exact: each degree l is multiplied by
exp(-l(l+1) kappa_e dt / (2 R^2)) per step, so the spreading is isotropic at every latitude, crosses
the poles, and conserves mass exactly (the degree-0 term is untouched). The deliveries are already
smoothed (sigma >= 150 km), so degree 224 (half-wavelength ~45 km) resolves them without ringing.
The grids are already smoothed with Gaussians of sigma = 250 km (carbon, "500 km") and 150 km
(water, "300 km"; the FFT filter of the EarthByte water and carbon workflows uses sigma = half
the stated distance). Water deliveries are first spread to the carbon sigma (heat kernel of variance
250^2 - 150^2 km^2), so that carbon and water are treated identically: same kernel, same diffusivity,
same loss.

Fields (``var``): carbon (c_mantle set: slab mantle, crust and lithosphere), water_sub (all water
carried into subduction), water_deep (water the slab still holds below 125 km: subducted minus
released) and water_out (slab water released above 125 km).

Every field is held in the mantle reference frame: plates move over the reservoir. Water released
from the slab above 125 km stays in the uppermost convecting mantle, and the mid-ocean ridges that later
form above that mantle acquire its signature for at least 400 Myr (Dixon et al., Geology, in revision);
carbon, and the water still bound in the slab below 125 km, are carried deeper before they are released.
Little carbon is released above 125 km except from sedimentary carbonate, which was scarce on the
seafloor before pelagic calcifiers spread in the Mesozoic (before about 100 Ma); carbon_sed (the c_carbonate
set, 0-239 Ma) is that shallow carbon, kept as a sensitivity field.
The fields therefore differ in the depth of the reservoir they describe, not in its frame: water_out is
the water reservoir of the uppermost mantle, water_deep and carbon the deeper ones. The delivery
accumulated along the plate path (the "track" measures of scripts/03 and 82) records how long a piece of
lithosphere travelled over such mantle.

Spurious single-step injections (scripts/10_grid_audit.py: one-step cell increments tens to thousands
of times the typical maximum, which smoothing turns into round patches) are capped: in every step a
cell's delivery is limited to --spike-factor (default 10) times the median of the largest cell delivery
of the preceding 50 steps. The share of delivery removed is reported per variable. For every point of a
features table, at its formation age and mantle-frame position:

    mem_<var>                       kappa_e = 1,000 km^2/Myr (Richter et al. 1982), no loss (the reference)
    mem_<var>_k<kappa>_l<tau>       kappa_e in 0 (a closed reservoir that only fills), 300, 1,000, 3,000
                                    km^2/Myr; tau in none, 800, 400 Myr; and at kappa_e 1,000 also
                                    tau 200 and 100 Myr (shorter memories, kept outside the family)

Values older than the start of a grid set are missing. The reference fields are also written every
10 Myr on a 0.8-degree mesh, one stacked file per variable (z(age, lat, lon), the layout read by
rifts.water.WaterGrids), for the maps of scripts/23; the carbon field is written in addition every
1 Myr (mantle_memory_fields_carbon_1myr.nc, compressed), for the same-site histories of scripts/62.

Runtime: one pass through the grids (~1,800 steps, 7 grid files per step), about 30-60 min with the
SSD mounted. Outputs: data/derived/mantle_memory_<tag>.parquet for each --tags entry, and
data/derived/mantle_memory_fields_<var>.nc, each written through a temporary name.

    python scripts/90_mantle_memory.py --check                                # kernel test, seconds
    python scripts/90_mantle_memory.py --tags barren --start 600 --pilot      # quick check
    python scripts/90_mantle_memory.py --tags full barren inrift carb26

Needs pyshtools (conda install -c conda-forge pyshtools, or pip install pyshtools).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import load_config, read_table, write_table  # noqa: E402
from rifts.water import WaterGrids  # noqa: E402

R_KM = 6371.0
SIGMA_GRID = {"carbon": 250.0, "water": 150.0}       # sigma of the source grids (km)
KAPPAS = (0.0, 300.0, 1000.0, 3000.0)                 # kappa_e, km^2/Myr (sigma^2 = kappa_e t); 0 = closed
LOSSES = (None, 800.0, 400.0)                         # e-folding time of loss (Myr); >= 400 Myr (Dixon et al.)
# shorter memories at the reference diffusivity only, outside the family of settings: how the contrast
# depends on how long the mantle keeps what it receives
EXTRA = ((1000.0, 200.0), (1000.0, 100.0))
REF = (1000.0, None)
FIELD_STEP = 10                                       # Myr between written reference fields
FINE_VAR = "carbon"                                   # also written every 1 Myr, for scripts/62 and 66
STRIDE = 2                                            # 0.2-degree nodes -> 0.4-degree Driscoll-Healy grid


def heat(var_km2, lmax):
    """Per-degree factor of the heat kernel on the sphere that spreads a field to a per-axis variance
    var_km2: exp(-l(l+1) var / (2 R^2)), shaped to multiply a (2, l, m) coefficient array."""
    l = np.arange(lmax + 1, dtype=float)
    return np.exp(-l * (l + 1.0) * var_km2 / (2.0 * R_KM ** 2))[None, :, None]


def to_dh(da, k=STRIDE):
    """0.2-degree node grid (lat -90..90 ascending, lon -180..180) to the Driscoll-Healy grid of
    pyshtools (sampling 2): lat 90 .. -90+d descending, lon 0 .. 360-d, with d = k x 0.2 degrees."""
    z = np.nan_to_num(np.asarray(da.values, float))
    lon = np.asarray(da["lon"].values, float)
    lat = np.asarray(da["lat"].values, float)
    if np.isclose(lon[-1] - lon[0], 360.0):
        z, lon = z[:, :-1], lon[:-1]
    d = lat[1] - lat[0]
    assert np.isclose(d, 0.2) and np.isclose(lat[0], -90.0) and np.isclose(lon[0], -180.0), "expects 0.2-degree nodes"
    z = z[::-1][: len(lat) - 1:k]                       # 90, 90-0.4, ..., -89.6
    z = z[:, ::k]                                       # -180, ..., 179.6
    return np.roll(z, -z.shape[1] // 2, axis=1)         # 0, ..., 359.6


def check(lmax=224):
    """The kernel against Richter's definition: a 250 km Gaussian spread for 150 steps of 1 Myr at
    kappa_e = 1,000 km^2/Myr must reach sqrt(250^2 + 1000 x 150) = 461 km, at any latitude, with its
    mass kept."""
    import pyshtools as pysh
    n = 2 * (lmax + 1)
    la = np.radians(90.0 - 180.0 * np.arange(n) / n)[:, None]
    lo = np.radians(360.0 * np.arange(2 * n) / (2 * n))[None, :]
    w = np.cos(la) * np.ones_like(lo)
    f = heat(1000.0, lmax)
    for lat0 in (0.0, 60.0, 85.0):
        p0, q0 = np.radians(lat0), np.radians(30.0)
        dist = np.arccos(np.clip(np.sin(la) * np.sin(p0) + np.cos(la) * np.cos(p0) * np.cos(lo - q0), -1, 1)) * R_KM
        z = np.exp(-dist ** 2 / (2 * 250.0 ** 2))
        c = pysh.expand.SHExpandDH(z, sampling=2)
        for _ in range(150):
            c = c * f
        z2 = pysh.expand.MakeGridDH(c, sampling=2)
        sig = np.sqrt((z2 * w * dist ** 2).sum() / (z2 * w).sum() / 2.0)
        print(f"  latitude {lat0:4.0f}: sigma {sig:6.1f} km (expected {np.sqrt(250 ** 2 + 1000 * 150):.1f}), "
              f"mass ratio {(z2 * w).sum() / (z * w).sum():.6f}, minimum {z2.min():.1e}")


def bad_steps_for(cfg, name):
    return sorted(set(next((g.get("bad_steps", []) for g in cfg["water"]["grid_sets"] if g["name"] == name), [])))


def grid_set(cfg, name):
    gs = next(g for g in cfg["water"]["grid_sets"] if g["name"] == name)
    return WaterGrids(gs["dir"], cfg["paths"]["water_grid_glob"], gs["variables"], gs.get("components"),
                      gs.get("folder_template", "{component}"), bad_steps_for(cfg, name))


def total(grids, var, age):
    """Sum over components of one variable at one age (DataArray)."""
    out = None
    for (v, c) in grids.files:
        if v != var:
            continue
        g = grids._grid(v, c, int(age))
        out = g if out is None else out + g
    return out


def col(var, kappa, loss):
    if (kappa, loss) == REF:
        return f"mem_{var}"
    return f"mem_{var}_k{int(kappa)}_l{'none' if loss is None else int(loss)}"


def main():
    if "--check" in sys.argv:
        check()
        return 0
    import pyshtools as pysh
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--tags", nargs="+", default=["full", "barren", "inrift"])
    ap.add_argument("--carbon-set", default="c_mantle")
    ap.add_argument("--water-set", default="h2o")
    ap.add_argument("--carbonate-set", default="c_carbonate",
                    help="sedimentary carbonate carbon (0-239 Ma), the carbon released at shallow depth; '' to skip")
    ap.add_argument("--start", type=int, help="first (oldest) age to march from (default: oldest grid age)")
    ap.add_argument("--pilot", action="store_true", help="reference parameters only, no field file")
    ap.add_argument("--check", action="store_true", help="test the kernel on a synthetic delivery and stop")
    ap.add_argument("--spike-factor", type=float, default=10.0,
                    help="cap a cell's delivery in one step at this many times the recent typical maximum")
    a = ap.parse_args()
    cfg = load_config(a.config)
    der = Path(cfg["paths"]["derived"])

    pts = {}
    for tag in a.tags:
        f = read_table(der / f"features_{tag}")
        f = f[(f["lag_myr"] == 0) & (f["draw"] == 0)][["pt_id", "age_ma", "mlon", "mlat"]].drop_duplicates("pt_id")
        f = f[np.isfinite(f["age_ma"]) & np.isfinite(f["mlon"]) & np.isfinite(f["mlat"])]
        f["t"] = np.rint(f["age_ma"]).astype(int)
        pts[tag] = f.reset_index(drop=True)
        print(f"{tag}: {len(f)} points, ages {f['t'].min()}-{f['t'].max()} Ma", flush=True)

    cg, wg = grid_set(cfg, a.carbon_set), grid_set(cfg, a.water_set)
    sources = {"carbon": (cg, cg.variables[0], "carbon"),
               "water_sub": (wg, "mean_subducted_water", "water"),
               "water_deep": (wg, ("mean_subducted_water", "mean_slab_outflux"), "water"),
               "water_out": (wg, "mean_slab_outflux", "water")}
    if a.carbonate_set and any(g["name"] == a.carbonate_set for g in cfg["water"]["grid_sets"]):
        sg = grid_set(cfg, a.carbonate_set)
        sources["carbon_sed"] = (sg, sg.variables[0], "carbon")
    ages_of = {v: set(int(x) for x in g.ages) for v, (g, _, _) in sources.items()}
    starts = {v: int(g.ages.max()) for v, (g, _, _) in sources.items()}
    ends = {v: int(g.ages.min()) for v, (g, _, _) in sources.items()}
    t0 = min(max(starts.values()), a.start) if a.start else max(starts.values())
    print("grid ages:", {v: f"{ends[v]}-{starts[v]} Ma" for v in sources}, f"; marching from {t0} Ma", flush=True)

    params = [REF] if a.pilot else [(k, l) for k in KAPPAS for l in LOSSES] + list(EXTRA)
    z0 = to_dh(total(cg, cg.variables[0], ends["carbon"]))
    n = z0.shape[0]
    lmax = n // 2 - 1
    dh_lat = 90.0 - 180.0 * np.arange(n) / n                      # descending
    dh_lon = 360.0 * np.arange(2 * n) / (2 * n)
    f_step = {k: heat(k * 1.0, lmax) for k in {p[0] for p in params}}   # per-axis variance kappa_e x 1 Myr
    eq_var = SIGMA_GRID["carbon"] ** 2 - SIGMA_GRID["water"] ** 2
    f_eq = heat(eq_var, lmax)
    zero = np.zeros((2, lmax + 1, lmax + 1))
    M = {(v, p): zero.copy() for v in sources for p in params}     # spherical-harmonic coefficients
    prev = {}
    fields = {v: {} for v in sources}
    fine = {}
    out = {tag: {} for tag in a.tags}
    recent_max = {v: [] for v in sources}
    removed = {v: [0.0, 0.0] for v in sources}             # capped, total delivery
    capped_steps = []                                      # (var, age, capped, step delivery, max/cap)
    print(f"reservoirs: {len(M)} on degree {lmax} ({n} x {2 * n} grid)", flush=True)

    def grid(c):
        """Coefficients to the 0.8-degree output mesh, lat ascending, lon -180..180."""
        z = pysh.expand.MakeGridDH(c, sampling=2)[::2, ::2]
        return np.roll(z[::-1], z.shape[1] // 2, axis=1).astype("float32")

    out_lat = dh_lat[::2][::-1]
    out_lon = np.roll(dh_lon[::2], len(dh_lon[::2]) // 2)
    out_lon = np.where(out_lon >= 180.0, out_lon - 360.0, out_lon)

    def sample(c, lon, lat):
        return np.asarray(pysh.expand.MakeGridPoint(c, np.asarray(lat, float), np.asarray(lon, float) % 360.0), float)

    for t in range(t0, -1, -1):
        for v, (g, var, kind) in sources.items():
            if t > starts[v] or t < ends[v] or t not in ages_of[v]:
                continue                               # a missing age: its delivery joins the next step
            if isinstance(var, tuple):                 # deep water: subducted minus released above 125 km
                cur = to_dh(total(g, var[0], t)) - to_dh(total(g, var[1], t))
            else:
                cur = to_dh(total(g, var, t))
            if v not in prev:                          # first age of this set: the reservoir starts empty
                prev[v] = cur
                continue
            s = np.maximum(cur - prev[v], 0.0)         # delivery in the step (cumulative grids only grow)
            prev[v] = cur
            mx = float(s.max())
            if len(recent_max[v]) >= 10:
                cap = a.spike_factor * float(np.median(recent_max[v][-50:]))
                if mx > cap > 0:
                    cut = float((s - np.minimum(s, cap)).sum())
                    capped_steps.append((v, t, cut, float(s.sum()), mx / cap))
                    removed[v][0] += cut
                    s = np.minimum(s, cap)
            removed[v][1] += float(s.sum()) + 0.0
            recent_max[v].append(mx)
            cs = pysh.expand.SHExpandDH(s, sampling=2)
            if kind == "water":
                cs = cs * f_eq
            for p in params:
                kappa, loss = p
                m = M[(v, p)] * f_step[kappa]
                if loss is not None:
                    m *= np.exp(-1.0 / loss)
                M[(v, p)] = m + cs
            if not a.pilot and t % FIELD_STEP == 0:
                fields[v][t] = grid(M[(v, REF)])

        if not a.pilot and FINE_VAR in prev:                # every Myr, including ages missing from the set
            fine[t] = fields[FINE_VAR][t] if t in fields[FINE_VAR] else grid(M[(FINE_VAR, REF)])
        for tag, f in pts.items():
            sel = (f["t"] == t).to_numpy()
            if not sel.any():
                continue
            for v in sources:
                if v not in prev or t > starts[v]:
                    continue
                for p in params:
                    out[tag].setdefault(col(v, *p), np.full(len(f), np.nan))[sel] = sample(
                        M[(v, p)], f.loc[sel, "mlon"].to_numpy(), f.loc[sel, "mlat"].to_numpy())
        if t % 50 == 0:
            print(f"  {t} Ma", flush=True)

    for v, (cut, tot) in removed.items():
        print(f"  {v}: spike cap removed {100 * cut / max(cut + tot, 1e-30):.2f}% of the delivery", flush=True)
    # every capped step, so that the removal can be checked against the grid audit (scripts/10)
    cs = pd.DataFrame(capped_steps, columns=["var", "age_ma", "capped", "step_delivery", "max_over_cap"])
    cs["share_of_step"] = cs["capped"] / cs["step_delivery"]
    cs["share_of_total"] = cs["capped"] / cs["var"].map({v: c + tt for v, (c, tt) in removed.items()})
    print(cs.sort_values("capped", ascending=False).head(15).round(4).to_string(index=False), flush=True)
    if not a.pilot:
        cs.to_csv(der / "mantle_memory_capped_steps.csv", index=False)
    for tag, f in pts.items():
        res = pd.DataFrame({"pt_id": f["pt_id"].to_numpy(), **out[tag]})
        print(f"{tag}:\n{res.describe().T[['count', 'mean', '50%']].round(4).to_string()}")
        print(f"wrote {write_table(res, der / f'mantle_memory_{tag}')}")
    if not a.pilot:
        import xarray as xr
        fields = dict(fields)
        fields[f"{FINE_VAR}_1myr"] = fine
        for v, d in fields.items():
            ages = sorted(d)
            ds = xr.Dataset({"z": (("age", "lat", "lon"), np.stack([d[t] for t in ages]))},
                            coords={"age": np.array(ages, float), "lat": out_lat, "lon": out_lon})
            ds.attrs = dict(description=f"subducted {v} held in the mantle, diffusing reservoir, kappa_e "
                                        f"{REF[0]:.0f} km^2/Myr (sigma^2 = kappa_e t, Richter et al. 1982), no loss, "
                                        "spherical-harmonic heat kernel (scripts/90_mantle_memory.py)")
            dest = der / f"mantle_memory_fields_{v}.nc"
            tmp = dest.with_name(dest.name + ".partial")
            ds.to_netcdf(tmp, encoding={"z": {"zlib": True, "complevel": 4}})
            tmp.replace(dest)
            print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
