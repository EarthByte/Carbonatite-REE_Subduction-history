#!/usr/bin/env python
"""Two tests of where carbonatite sources sit: the secular record and recurrence at a site.

``secular`` (hypothesis S10, recycling of subducted carbon over Earth history). If subducted
carbon and water fertilise the sources of carbonatites, the number of carbonatite districts
formed through time should follow the global rate of subduction, after the time the slab takes
to reach and metasomatise the lithosphere. Global subduction is measured in Muller2025 every
``STEP`` Myr from the plate-boundary statistics of the resolved topologies: trench length and the
area of plate consumed (trench length times orthogonal convergence rate, km2 per year). The
counts of districts (each district once, at its mean age) are binned in ``STEP`` Myr, smoothed
over 50 Myr, and compared with the flux leading by 0 to 300 Myr (Spearman correlation). Because
both series are autocorrelated, significance comes from circular shifts of the flux series by at
least 200 Myr. The same is done for carbonatites as a share of carbonatites and kimberlites (both
intraplate magmas sampled by the same compilations, so preservation and sampling cancel to first
order), for alkaline-igneous REE deposits and for porphyry copper (the positive control, whose
record is dominated by preservation). Every record of preserved deposits grows toward the
present, and so does reconstructed trench length, so the correlations are repeated after removing
the 300 Myr running mean from both series (high-pass): only the shorter fluctuations are then
compared. Windows: 0-1800 Ma, and 0-300 Ma against 300-1800 Ma
(Banerjee et al. 2021 found recycled-carbon signatures in carbonatites younger than about 300 Ma
only).

``recurrence`` (hypotheses S3 against S1 and S6: a source fixed in the lithosphere against one
fixed in the mantle). Carbonatites recur at the same place over hundreds of Myr. If the source
is fixed in the lithosphere, recurrent districts are close in present-day (plate-fixed)
coordinates and far apart in the mantle frame at their formation ages; if it is fixed in the
mantle (a plume, the edge of a large low-shear-velocity province), they are close in the mantle
frame and spread along the plate path. For every pair of districts separated by at least
``MIN_GAP`` Myr, the distance on the plate (present day) and in the mantle frame (each district
reconstructed in Muller2025 to its own age) is counted within 100, 250 and 500 km, against a null
in which the ages are permuted among the districts of a class (positions kept; each reconstructed
to its new age), 500 times. The decisive count is of pairs close in the mantle frame but more than
twice as far apart on the plate (``mantle_only``): a track over a mantle-fixed source produces an
excess of them, and plate-frame clustering cannot. The plate-frame count against the same null
measures temporal clustering instead: neighbouring districts formed in episodes have similar ages,
so permuting ages creates more widely separated neighbours than observed. Kimberlites (with documented hotspot tracks)
are the contrast.

Inputs: data/derived/carbonatite_features_<tag> (scripts/83), the Muller2025 plate model.
Outputs (ONE file each, through a .partial name):
  data/derived/carbonatite_secular_<tag>.csv      flux and district counts per bin, and the correlations
  data/derived/carbonatite_recurrence_<tag>.csv   close-pair counts against the permutation null

    python scripts/85_carbonatite_secular_recurrence.py --tag full
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import EARTH_RADIUS_KM, load_config, lonlat_to_xyz, read_table  # noqa: E402

STEP = 10.0
MAX_AGE = 1800.0
SMOOTH_MYR = 50.0
LEADS = (0, 25, 50, 100, 150, 200, 300)
MIN_SHIFT = 200.0
HIGHPASS_MYR = 300.0
WINDOWS = ((0, 1800), (0, 300), (300, 1800))
MIN_GAP = 50.0
RADII = (100.0, 250.0, 500.0)
CLASSES = ("CARB", "AIREE", "KIMB")


def write_csv(t, path):
    tmp = path.with_name(path.name + ".partial")
    t.to_csv(tmp, index=False)
    tmp.replace(path)
    print(f"wrote {path}")


def districts(p, cls):
    """One row per district: mean age, the position of its first point, its plate."""
    d = p[(p["group"] == cls) & (p["kind"] == "deposit")].copy()
    d["cluster_id"] = d["cluster_id"].fillna(d["pt_id"])
    g = d.groupby("cluster_id")
    return pd.DataFrame({"age_ma": g["age_ma"].mean(), "lon0": g["lon0"].first(), "lat0": g["lat0"].first(),
                         "pid": g["pid"].first(), "ore": g["ore"].max()}).reset_index()


# ----------------------------------------------------------------------------- secular
def global_subduction(mantle, times):
    """Total trench length (km) and area of plate consumed (trench length times positive orthogonal
    convergence, km2 per year) in the resolved topologies of ``mantle`` at each of ``times``."""
    import pygplates
    rows = []
    for t in times:
        st = mantle.topological_snapshot(t).calculate_plate_boundary_statistics(np.radians(0.25))
        length = area = 0.0
        for s in st:
            if s.boundary_feature.get_feature_type() != pygplates.FeatureType.gpml_subduction_zone:
                continue
            seg = s.boundary_length * EARTH_RADIUS_KM
            v = s.convergence_velocity_orthogonal          # km/Myr; positive where plates converge
            length += seg
            if v is not None and np.isfinite(v) and v > 0:
                area += seg * v * 1e-6                     # km2/yr
        rows.append(dict(time_ma=float(t), trench_km=length, consumed_km2_yr=area))
        if int(t) % 200 == 0:
            print(f"    {t:.0f} Ma: trench {length:,.0f} km, {area:.2f} km2/yr", flush=True)
    return pd.DataFrame(rows)


def smooth(x, myr):
    k = max(1, int(round(myr / STEP)))
    return pd.Series(x).rolling(k, center=True, min_periods=1).mean().to_numpy()


def lagged_rho(count, flux, lead, lo, hi, times):
    """Spearman correlation of counts at t with the flux at t + lead, within lo-hi Ma."""
    from scipy.stats import spearmanr
    k = int(round(lead / STEP))
    f = np.roll(flux, -k) if k else flux
    ok = (times >= lo) & (times < hi) & (times + lead <= MAX_AGE) & np.isfinite(count) & np.isfinite(f)
    if ok.sum() < 8:
        return np.nan
    return float(spearmanr(count[ok], f[ok]).statistic)


def run_secular(p, mantle, out):
    """Counts of districts per STEP Myr bin for every class (and the carbonatite share of carbonatites and
    kimberlites), smoothed over SMOOTH_MYR, against the global subduction flux leading by each of LEADS,
    in each of WINDOWS: Spearman correlation with a p-value from at most 300 circular shifts of the flux
    of at least MIN_SHIFT Myr, smoothed and after removing the HIGHPASS_MYR running mean. Writes the
    series and the correlations to ``out``."""
    times = np.arange(0.0, MAX_AGE + STEP / 2, STEP)
    print("  global subduction through time (Muller2025) ...", flush=True)
    g = global_subduction(mantle, times)
    edges = np.append(times - STEP / 2, times[-1] + STEP / 2)
    series = {}
    for cls in CLASSES + ("PCU",):
        d = districts(p, cls) if cls != "PCU" else districts(p.assign(ore=False), cls)
        series[cls] = np.histogram(d["age_ma"], edges)[0].astype(float)
    series["CARB share of CARB+KIMB"] = series["CARB"] / np.maximum(series["CARB"] + series["KIMB"], 1)
    series["CARB share of CARB+KIMB"][(series["CARB"] + series["KIMB"]) == 0] = np.nan
    for k, v in series.items():
        g[f"n_{k}"] = v
    rows = []
    rng = np.random.default_rng(85)
    nshift = len(times)
    # circular shifts of at least MIN_SHIFT Myr from either end, for the null of the correlations
    shifts = [s for s in range(nshift) if MIN_SHIFT / STEP <= s <= nshift - MIN_SHIFT / STEP]
    for cls, v in series.items():
        c = smooth(v, SMOOTH_MYR) if "share" not in cls else pd.Series(v).rolling(
            int(SMOOTH_MYR / STEP), center=True, min_periods=1).mean().to_numpy()
        for fname, filt in (("consumed_km2_yr", "smoothed"), ("trench_km", "smoothed"),
                            ("consumed_km2_yr", "high-pass")):
            f = smooth(g[fname].to_numpy(float), SMOOTH_MYR)
            cc = c
            if filt == "high-pass":
                # remove the slow trend shared by every preserved record (more young deposits) and by
                # the model (more trench length where reconstructions are better constrained)
                f = f - smooth(f, HIGHPASS_MYR)
                cc = c - smooth(c, HIGHPASS_MYR)
            for lo, hi in WINDOWS:
                for lead in LEADS:
                    r = lagged_rho(cc, f, lead, lo, hi, times)
                    null = np.array([lagged_rho(cc, np.roll(f, s), lead, lo, hi, times)
                                     for s in rng.choice(shifts, min(300, len(shifts)), replace=False)])
                    null = null[np.isfinite(null)]
                    pv = float((np.sum(np.abs(null) >= abs(r)) + 1) / (len(null) + 1)) if np.isfinite(r) else np.nan
                    rows.append(dict(part="correlation", series=cls, flux=fname, filter=filt, window=f"{lo}-{hi} Ma",
                                     lead_myr=lead, rho=r, p_shift=pv, n_bins=int(((times >= lo) & (times < hi)).sum()),
                                     n_districts=int(np.nansum(v[(times >= lo) & (times < hi)])) if "share" not in cls else np.nan))
    t = pd.concat([g.assign(part="series"), pd.DataFrame(rows)], ignore_index=True)
    write_csv(t, out)
    best = pd.DataFrame(rows)
    best = best[best["flux"] == "consumed_km2_yr"]
    for (s, filt, w), b in best.groupby(["series", "filter", "window"], sort=False):
        i = b["rho"].abs().idxmax()
        print(f"  {s:24s} {filt:9s} {w:12s} strongest at lead {b.loc[i, 'lead_myr']:>3.0f} Myr: "
              f"rho {b.loc[i, 'rho']:+.2f}, p {b.loc[i, 'p_shift']:.3f}")


# -------------------------------------------------------------------------- recurrence
def mantle_positions(mantle, lon, lat, pid, age):
    """Each present-day point reconstructed to its own age (rounded to 1 Myr) in the mantle frame."""
    out = np.full((len(lon), 2), np.nan)
    ar = np.round(age).astype(float)
    for a in np.unique(ar):
        i = np.nonzero(ar == a)[0]
        x, y = mantle.reconstruct_points(lon[i], lat[i], pid[i], a)
        out[i, 0], out[i, 1] = x, y
    return out


def pair_distances(lon, lat):
    xyz = lonlat_to_xyz(lon, lat)
    c = np.clip(xyz @ xyz.T, -1, 1)
    return np.arccos(c) * EARTH_RADIUS_KM


def close_counts(Dp, Dm, age, iu):
    """For district pairs at least MIN_GAP Myr apart: the number close on the plate (present day) and in
    the mantle frame within each of RADII, the number close in the mantle frame but more than twice the
    radius apart on the plate, and the number of pairs with mantle positions."""
    gap = np.abs(age[:, None] - age[None, :])[iu]
    dp, dm = Dp[iu], Dm[iu]
    sel = gap >= MIN_GAP
    out = {}
    for r in RADII:
        out[f"plate_{r:.0f}"] = int(np.sum(sel & (dp <= r)))
        out[f"mantle_{r:.0f}"] = int(np.sum(sel & np.isfinite(dm) & (dm <= r)))
        # close in the mantle but not on the plate: the signature of a track over a fixed source
        out[f"mantle_only_{r:.0f}"] = int(np.sum(sel & np.isfinite(dm) & (dm <= r) & (dp > 2 * r)))
    out["pairs"] = int(np.sum(sel & np.isfinite(dm)))
    return out


def run_recurrence(p, mantle, out, n_perm):
    """Close-pair counts of every class (close_counts) against a null in which ages are permuted among the
    districts of the class, ``n_perm`` times, each district reconstructed to its new age. Mantle-frame
    counts of the null are scaled to the observed number of pairs with mantle positions. Writes observed
    and null counts, ratio and one-sided p-values to ``out``."""
    rows = []
    rng = np.random.default_rng(85)
    for cls in CLASSES:
        d = districts(p, cls)
        d = d[np.isfinite(d["age_ma"]) & (d["pid"].fillna(0) > 0)].reset_index(drop=True)
        lon, lat = d["lon0"].to_numpy(float), d["lat0"].to_numpy(float)
        pid, age = d["pid"].astype(int).to_numpy(), d["age_ma"].to_numpy(float)
        Dp = pair_distances(lon, lat)
        iu = np.triu_indices(len(d), 1)

        def counts(a):
            m = mantle_positions(mantle, lon, lat, pid, a)
            ok = np.isfinite(m).all(axis=1)
            Dm = np.full_like(Dp, np.nan)
            Dm[np.ix_(ok, ok)] = pair_distances(m[ok, 0], m[ok, 1])
            return close_counts(Dp, Dm, a, iu)

        obs = counts(age)
        null = []
        for k in range(n_perm):
            null.append(counts(rng.permutation(age)))
            if (k + 1) % 100 == 0:
                print(f"    {cls}: permutation {k + 1} of {n_perm}", flush=True)
        null = pd.DataFrame(null)
        for key, v in obs.items():
            nv = null[key].to_numpy(float)
            if key.startswith("mantle"):
                # permuted ages put some districts outside the plate model at their new age; scale the
                # null to the observed number of pairs with mantle positions
                nv = nv * obs["pairs"] / null["pairs"].to_numpy(float)
            rows.append(dict(cls=cls, measure=key, observed=v, null_mean=float(nv.mean()),
                             null_lo=float(np.percentile(nv, 2.5)), null_hi=float(np.percentile(nv, 97.5)),
                             ratio=float(v / nv.mean()) if nv.mean() > 0 else np.nan,
                             p_excess=float((np.sum(nv >= v) + 1) / (len(nv) + 1)),
                             p_deficit=float((np.sum(nv <= v) + 1) / (len(nv) + 1)), n_districts=len(d)))
        r = pd.DataFrame([x for x in rows if x["cls"] == cls]).set_index("measure")
        print(f"  {cls} ({len(d)} districts, pairs >= {MIN_GAP:.0f} Myr apart): " + "; ".join(
            f"{m} {r.loc[m, 'observed']:.0f} vs {r.loc[m, 'null_mean']:.0f} (x{r.loc[m, 'ratio']:.2f})"
            for m in ("plate_250", "mantle_250", "mantle_only_250")))
    write_csv(pd.DataFrame(rows), out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--parts", nargs="*", default=["secular", "recurrence"], choices=["secular", "recurrence"])
    ap.add_argument("--perm", type=int, default=500)
    a = ap.parse_args()
    cfg = load_config(a.config)
    der = cfg["paths"]["derived"]
    from rifts.models import load_models
    mantle, _ = load_models(cfg)
    p = read_table(der / f"carbonatite_features_{a.tag}")
    if "secular" in a.parts:
        run_secular(p, mantle, der / f"carbonatite_secular_{a.tag}.csv")
    if "recurrence" in a.parts:
        print("  recurrence in the plate and mantle frames ...", flush=True)
        run_recurrence(p, mantle, der / f"carbonatite_recurrence_{a.tag}.csv", a.perm)


if __name__ == "__main__":
    main()
