#!/usr/bin/env python
"""Features of the carbonatite and alkaline rare-earth mineral systems, one row per point.

For carbonatites (CARB), alkaline-igneous rare-earth deposits (AIREE), kimberlites (KIMB, the
contrasting intraplate magma) and porphyry copper (PCU, the subduction control), and the
age-matched reference points of each, this step gathers every measured property into one table
and labels each by the stage of the mineral system it describes (``FEATURES`` below):

  source      what the lithosphere and mantle beneath the site had received before the magma
              formed: slab water and carbon along the plate path (scripts/03 and 82) and the time on
              an overriding plate near a trench, also at a stationary trench (scripts/82)
  lithosphere the architecture the melt rose through: craton-edge distance at the age of the
              point, shear-wave velocity at 150 km and its horizontal gradient, crustal thickness
              and apparent stretching factor, the thickness of the fast lid in REVEAL and its
              gradient (scripts/87), and whether the site lies in a fertilised domain of
              Spandler et al. (2026) formed before it (Proterozoic lithosphere around cratons)
  trigger     what could have started melting: mantle-driven dynamic topography of the site at its age,
              its rate of uplift over the 20 and 60 Myr before and the uplift since its last low in
              the 200 Myr before (median of six mantle-flow models of Dhungana & Flament, scripts/88), distance to
              continental divergence in the plate
              model at formation and its nearest approach over the 20 Myr before (Muller2025
              topologies; set to missing for points older than 300 Ma, before which the model
              does not resolve continental divergence), distance to a large igneous province
              emplaced before or after, to the edge of a
              large low-shear-velocity province (agreed by at least three of four modern
              tomographic families, held fixed in the mantle frame; rifts/llsvp.py), also signed
              (negative inside), plate speed and its change in direction, distance to the trench
              and to a divergent boundary
  upgrading   what could have enriched or preserved a system after it formed: Myr spent within the
              humid tropics (palaeolatitude within 25 degrees, Cao2024 palaeomagnetic frame) from
              formation to the present and over the last 100 Myr, where lateritic weathering forms
              the supergene REE and Nb ores of carbonatites (Mount Weld, Araxa, Tomtor)
  exposure    whether a system of that age could be found and mined today: present-day sediment
              thickness at the site (GST-1), whether the site is under less than 1 km of sediment (within reach of mining), and exploration
              density (distance to the nearest deposit of another family)

It also marks endowment (``ore_flags``): ``ore`` for carbonatites within 10 km of a deposit with
reported tonnage and REE or Nb grade (Berger et al. 2009) or of a magmatic REE ore deposit of
Spandler et al. (2026, Table S1b), and for every alkaline-igneous REE deposit;
``ree_mineralised`` adds carbonatite-hosted REE occurrences (Orris and Grauch 2002); ``ore_ree`` and
``ore_nb_only`` split the ore class by commodity (REE grade or magmatic REE ore deposit; niobium grade only).

Output: ONE table, data/derived/carbonatite_features_<tag>.parquet, through a .partial name,
with a companion data/derived/carbonatite_feature_stages_<tag>.csv naming the stage of every column.

    python scripts/83_carbonatite_features.py --tag full
"""
import argparse
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import ROOT, EARTH_RADIUS_KM, load_config, lonlat_to_xyz, read_table, write_table  # noqa: E402

CLASSES = ("CARB", "AIREE", "KIMB", "PCU")
MINEABLE_KM = 1.0
ORE_RADIUS_KM = 10.0
DIVCONT_LAGS = (0, 5, 10, 15, 20)                    # Myr before formation
DIVCONT_MAX_AGE = 300.0                              # continental divergence resolved after ~300 Ma
LIP_WINDOWS = ((0, 20), (20, 70), (-40, 0))          # Myr before the point (negative: after it)
TROPICS_DEG = 25.0                                   # humid-tropical weathering belt (|palaeolatitude|)
TROPICS_STEP_MYR = 5.0

FEATURES = {
    # source
    "h2o_mean_slab_outflux_track_100myr": "source", "h2o_mean_slab_outflux_track_400myr": "source",
    "h2o_mean_subducted_water_track_400myr": "source", "c_mantle_subducted_carbon_track_800myr": "source",
    "c_mantle_subducted_carbon_stock": "source", "c_organic_subducted_carbon_stock": "source",
    "out_sed_path_full": "source", "out_crust_path_full": "source", "out_mantle_path_full": "source",
    "out_total_path_full": "source", "out_total_path_1200myr": "source", "out_total_path_lag_myr": "source",
    "sub_total_path_full": "source", "c_mantle_path_full": "source", "c_mantle_path_lag_myr": "source",
    "over_900km_myr": "source", "over_900km_since_myr": "source", "over_500km_myr": "source",
    "over_900km_since_censored_myr": "source", "over_900km_episode": "source",
    "over_900km_stable_myr": "source", "over_900km_stable_since_myr": "source",
    # slab water carried past 125 km (subducted minus slab outflux, i.e. the water the thermal-parameter
    # lookup of the water workflow keeps in the slab at 4.4 GPa): the cold-slab water that can reach the
    # transition zone (Shirey et al. 2021, 2024); fixed = mantle frame, where it is stored, track = plate path
    "h2o_deep_fixed_400myr": "source", "h2o_deep_track_400myr": "source", "deep_total_path_full": "source",
    # subducted carbon and slab water held in diffusing mantle-frame reservoirs (scripts/90, reference
    # model): water released above 125 km in the uppermost mantle, carbon deeper; the parameter grid,
    # deep and total water are sensitivity
    "mem_carbon": "source", "mem_water_out": "source",
    "h2o_deep_fixed_100myr": "sensitivity", "h2o_deep_track_100myr": "sensitivity",
    "h2o_deep_fixed_200myr": "sensitivity", "h2o_deep_track_200myr": "sensitivity",
    # the fertilised domains of Spandler et al. (2026) mark Proterozoic lithosphere around cratons and
    # carry no information on the amount of slab input (scripts/81), so they describe architecture
    "fml_primed": "lithosphere", "fml_age_gap_myr": "lithosphere",
    # lithosphere
    "craton_dist_km": "lithosphere", "vsh150": "lithosphere", "vsh_grad150": "lithosphere",
    "site_crust_km": "lithosphere", "site_beta": "lithosphere",
    # thickness of the fast lid in REVEAL and its horizontal gradient (scripts/87)
    "lid_km": "lithosphere", "lid_grad": "lithosphere",
    "lid_km_1p0": "sensitivity", "lid_km_2p0": "sensitivity",
    # trigger
    "divcont_dist_km": "trigger", "divcont_prior20_km": "trigger",
    # mantle-driven dynamic topography of the site: median of the six mantle-flow cases of Dhungana &
    # Flament (scripts/88); each case alone and Young et al. (2022) are added below as sensitivity
    "dt_m": "trigger", "dt_uplift_20": "trigger", "dt_uplift_60": "trigger", "dt_rise_200": "trigger",
    "dt_uplift_after_20": "sensitivity", "dt_uplift_20_agree": "sensitivity",
    # plume propensity: share of the 80 Myr around formation in which a high or rising dynamic topography
    # reaches within 500 km, averaged over the six cases (model plumes are free, so statistical only)
    "dt_high_p500": "trigger", "dt_rise_p500": "trigger",
    "dt_high_p1000": "sensitivity", "dt_rise_p1000": "sensitivity",
    "lip_before_0_20_km": "trigger", "lip_before_20_70_km": "trigger", "lip_after_0_40_km": "trigger",
    "llsvp_edge_km": "trigger", "llsvp_signed_km": "trigger", "plate_speed_cm_yr": "trigger", "plate_turn_10myr_deg": "trigger",
    "sz_dist_km": "trigger", "div_dist_km": "trigger", "abs_paleolat": "trigger",
    # upgrading after formation (lateritic weathering in the humid tropics)
    "tropics_myr": "upgrading", "tropics_share": "upgrading", "tropics_last100_myr": "upgrading",
    # the LLSVP edge at other levels of agreement between tomographic models (compared, not ranked)
    "llsvp_edge_v2_km": "sensitivity", "llsvp_edge_v4_km": "sensitivity",
    # exposure and mineability
    "site_sed_m": "exposure", "mineable": "exposure",
    "other_family_km": "exposure",
}


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def base_points(der, tag, classes):
    """One row per point of ``classes`` and of their reference points at the sampled age (lag 0, draw 0),
    with ``group`` (the class, or the class a reference point was drawn for), ``kind`` (deposit or
    reference), the absolute palaeolatitude and the change in plate-motion direction over the 10 Myr
    before formation."""
    f = read_table(der / f"features_{tag}")
    f0 = f[(f["lag_myr"] == 0) & (f["draw"] == 0)]
    keep = f0["dclass"].isin(classes) | f0["null_for"].isin(classes)
    cols = ["pt_id", "dclass", "null_for", "age_ma", "lon0", "lat0", "pid", "mlon", "mlat", "in_model",
            "cluster_id", "w_cluster", "source", "subtype", "plate_speed_cm_yr", "plate_az", "sz_dist_km",
            "div_dist_km", "paleolat"]
    p = f0.loc[keep, [c for c in cols if c in f0]].drop_duplicates("pt_id").reset_index(drop=True)
    p["group"] = np.where(p["dclass"].isin(classes), p["dclass"], p["null_for"])
    p["kind"] = np.where(p["dclass"].isin(classes), "deposit", "reference")
    p["abs_paleolat"] = p["paleolat"].abs()
    # change in the direction of plate motion over the 10 Myr before formation
    f10 = f[(f["lag_myr"] == 10) & (f["draw"] == 0)][["pt_id", "plate_az"]].rename(columns={"plate_az": "az10"})
    p = p.merge(f10, on="pt_id", how="left")
    d = (p["plate_az"] - p["az10"] + 180) % 360 - 180
    p["plate_turn_10myr_deg"] = d.abs()
    return p.drop(columns=["az10"])


def present_distance(points, geoms, step_deg=0.25):
    """Great-circle distance (km) from present-day points to each geometry (points x geometries)."""
    from rifts.riftlag import _min_by_rift, _rift_vertices
    xy, owner = _rift_vertices(geoms, step_deg)
    return _min_by_rift(lonlat_to_xyz(points["lon0"], points["lat0"]), lonlat_to_xyz(xy[:, 0], xy[:, 1]),
                        owner, len(geoms))


def nearest_in_window(D, ages, event_age, lo, hi, max_km=5000.0):
    """Distance to the nearest geometry whose event age lies lo-hi Myr before each point."""
    lag = event_age[None, :] - ages[:, None]
    m = (lag > lo) & (lag <= hi) & np.isfinite(D)
    d = np.where(m, D, np.inf).min(axis=1)
    return np.where(np.isfinite(d), np.minimum(d, max_km), max_km)


def divergence_features(p, der, tag):
    """Distance to continental divergence in Muller2025 at formation and over the 20 Myr before.

    ``divcont_dist_km`` of the feature table (features_<tag>: a divergent boundary with continental
    crust on both sides within 300 km) at lag 0, and its minimum over lags 0-20 Myr. Missing for
    points older than ``DIVCONT_MAX_AGE``, where the plate model carries too little continental
    divergence for a distance to mean rifting.
    """
    f = read_table(der / f"features_{tag}")
    f = f[(f["draw"] == 0) & f["lag_myr"].isin(DIVCONT_LAGS) & f["pt_id"].isin(p["pt_id"])]
    w = f.pivot_table(index="pt_id", columns="lag_myr", values="divcont_dist_km", aggfunc="first")
    out = pd.DataFrame({"divcont_dist_km": w.get(0), "divcont_prior20_km": w.min(axis=1)}).reset_index()
    out = p[["pt_id", "age_ma"]].merge(out, on="pt_id", how="left")
    old = out["age_ma"] > DIVCONT_MAX_AGE
    out.loc[old, ["divcont_dist_km", "divcont_prior20_km"]] = np.nan
    return out[["divcont_dist_km", "divcont_prior20_km"]].reset_index(drop=True)


def lip_features(p, root):
    """Distance (present-day positions, km, capped at 5,000) to the nearest large igneous province of
    Johansson et al. (2018) emplaced in each of LIP_WINDOWS before or after the point, and the distance
    to the edge of the LLSVPs in the mantle frame at the age of the point (agreed by three of four
    tomographic families; two and four of four as sensitivity), also signed (negative inside)."""
    bt = load_module("bt", "74_kimberlite_breakup_tests.py")
    lips = bt.read_lips(root / "data" / "external" / "Johansson2018_LIPs_v2" / "Johansson_etal_2018_EarthByte_LIPs_v2.shp",
                        2000.0).reset_index(drop=True)
    D = present_distance(p, lips)
    ages, la = p["age_ma"].to_numpy(float), lips["onset_ma"].to_numpy(float)
    out = {}
    for lo, hi in LIP_WINDOWS:
        name = f"lip_before_{lo}_{hi}_km" if lo >= 0 else f"lip_after_{-hi}_{-lo}_km"
        out[name] = nearest_in_window(D, ages, la, lo, hi)
    # LLSVPs agreed by at least three of four independent tomographic families (rifts/llsvp.py);
    # two and four of four are kept as a sensitivity test, outside the ranked models
    from rifts.llsvp import Provinces
    mlon, mlat = p["mlon"].to_numpy(float), p["mlat"].to_numpy(float)
    for votes, name in ((3, ""), (2, "_v2"), (4, "_v4")):
        sd = Provinces(root, votes).signed_distance(mlon, mlat)
        if not name:
            out["llsvp_signed_km"] = sd
        out[f"llsvp_edge{name}_km"] = np.abs(sd)
    return pd.DataFrame(out)


def spandler_features(p, root):
    sp = load_module("sp81", "81_spandler_reassessment.py")
    polys = sp.load_polygons(sp.SPANDLER / "Extended Dataset 2")
    pa = sp.polygon_ages(p["lat0"], p["lon0"], polys)
    older = [[a for a in x if a > t] for x, t in zip(pa, p["age_ma"])]
    return pd.DataFrame({"fml_primed": [bool(o) for o in older],
                         "fml_age_gap_myr": [max(o) - t if o else np.nan for o, t in zip(older, p["age_ma"])]})


def lid_features(p, root):
    """Present-day fast-lid thickness of REVEAL (1.5% threshold; 1.0 and 2.0% as sensitivity) and its
    gradient at each point, from data/external/reveal_lid_thickness.nc (scripts/87)."""
    import xarray as xr
    f = root / "data" / "external" / "reveal_lid_thickness.nc"
    if not f.exists():
        print("  reveal_lid_thickness.nc not found: run scripts/87 first")
        return pd.DataFrame(index=p.index)
    g = xr.open_dataset(f)
    lat, lon = g["latitude"].values, g["longitude"].values
    i = np.clip(np.round((p["lat0"].to_numpy(float) - lat[0]) / (lat[1] - lat[0])).astype(int), 0, len(lat) - 1)
    lo = ((p["lon0"].to_numpy(float) - lon[0]) % 360.0)
    j = np.clip(np.round(lo / (lon[1] - lon[0])).astype(int), 0, len(lon) - 1)
    return pd.DataFrame({"lid_km": g["lid_km_1p5"].values[i, j], "lid_grad": g["grad_1p5"].values[i, j],
                         "lid_km_1p0": g["lid_km_1p0"].values[i, j], "lid_km_2p0": g["lid_km_2p0"].values[i, j]},
                        index=p.index)


def tropical_features(p, cfg):
    """Time each point spent in the humid tropics between its formation and the present.

    The point moves with its plate in the palaeomagnetic frame (Cao2024, which shares the plate IDs
    of Muller2025); its palaeolatitude is sampled every ``TROPICS_STEP_MYR`` from the formation age
    to the present. Returns Myr within ``TROPICS_DEG`` of the equator, that time as a share of the
    age, and the Myr within the tropics over the last 100 Myr (when most preserved laterite formed).
    """
    from rifts.models import load_models
    _, pm = load_models(cfg)
    lon, lat = p["lon0"].to_numpy(float), p["lat0"].to_numpy(float)
    pid, age = p["pid"].fillna(0).astype(int).to_numpy(), p["age_ma"].to_numpy(float)
    trop, last, seen = np.zeros(len(p)), np.zeros(len(p)), np.zeros(len(p))
    step = TROPICS_STEP_MYR
    for t in np.arange(0.0, np.nanmax(age) + step, step):
        on = age >= t
        if not on.any():
            continue
        _, plat = pm.reconstruct_points(lon[on], lat[on], pid[on], t)
        # the interval this sample stands for, cut at the formation age
        dt = np.minimum(age[on] - t, step / 2) + np.where(t > 0, step / 2, 0.0)
        ok = np.isfinite(plat)
        idx = np.nonzero(on)[0]
        seen[idx[ok]] += dt[ok]
        inside = ok & (np.abs(plat) <= TROPICS_DEG)
        trop[idx[inside]] += dt[inside]
        if t < 100:
            last[idx[inside]] += dt[inside]
        elif t == 100:                                  # only the half of the interval after 100 Ma
            last[idx[inside]] += np.minimum(dt[inside], step / 2)
    full = seen >= 0.9 * np.maximum(age, step)
    return pd.DataFrame({"tropics_myr": np.where(full, trop, np.nan),
                         "tropics_share": np.where(full, trop / np.maximum(seen, 1e-9), np.nan),
                         "tropics_last100_myr": np.where(seen >= np.minimum(age, 100) * 0.9, last, np.nan)})


def _near(p, q, km):
    from scipy.spatial import cKDTree
    q = q[np.isfinite(q).all(axis=1)]
    if not len(q):
        return np.zeros(len(p), bool)
    tree = cKDTree(lonlat_to_xyz(q[:, 0], q[:, 1]))
    chord = 2 * np.sin(km / EARTH_RADIUS_KM / 2)
    return np.array([len(h) > 0 for h in tree.query_ball_point(lonlat_to_xyz(p["lon0"], p["lat0"]), chord)])


def ore_flags(p, root, cfg):
    """Two levels of rare-earth (or niobium) endowment for carbonatites.

    ``ore``: a deposit with reported tonnage and REE or Nb grade (USGS carbonatite deposits,
    Berger et al. 2009) or a magmatic REE ore deposit of Spandler et al. (2026, Table S1b), within
    ``ORE_RADIUS_KM``; every alkaline-igneous REE deposit counts as ore.
    ``ree_mineralised``: the same, or an REE occurrence of the USGS REE table (Orris and Grauch
    2002) hosted by carbonatite, within the same radius.
    """
    raw = cfg["paths"]["raw_data"]
    u = pd.read_csv(raw / "usgs_carbonatite" / "main.csv", encoding="latin-1")
    u = u[(u["ree"].fillna(0) > 0) | (u["niobium"].fillna(0) > 0)]
    deposits = [u[["longitude", "latitude"]].to_numpy(float)]
    sp = load_module("sp81b", "81_spandler_reassessment.py")
    try:
        _, ree = sp.spandler_tables(sp.SPANDLER)
        deposits.append(ree[["lon", "lat"]].to_numpy(float))
    except FileNotFoundError:
        pass
    r = pd.read_csv(raw / "usgs_ree" / "main.csv", encoding="latin-1")
    r = r[r["deptype"].astype(str).str.lower().str.startswith("carb")]
    occ = r[["longitude", "latitude"]].to_numpy(float)
    dep = (p["kind"] == "deposit").to_numpy()
    carb, alk = (p["group"] == "CARB").to_numpy(), (p["group"] == "AIREE").to_numpy()
    # a record of the USGS carbonatite table counts as ore only through the REE or niobium grade of a
    # deposit within ORE_RADIUS_KM, itself included (it lies 0 km from itself); a USGS record without such a
    # grade, of which the table holds some, is not ore (review 2026-11-09: two records were flagged ore by
    # their source alone, so that the ore class exceeded the union of its REE and niobium parts)
    near_dep = _near(p, np.vstack(deposits), ORE_RADIUS_KM)
    near_any = near_dep | _near(p, occ, ORE_RADIUS_KM)
    # the ore class split by commodity (review 2026-11-08): ``ore_ree`` within ORE_RADIUS_KM of a USGS
    # carbonatite deposit with an REE grade or of a magmatic REE ore deposit of Spandler et al.;
    # ``ore_nb_only`` within that radius of a USGS carbonatite deposit with a niobium grade and no REE
    # grade, and not of any REE deposit. A USGS record is within 0 km of itself, so it needs no own clause.
    ree_dep = [u.loc[u["ree"].fillna(0) > 0, ["longitude", "latitude"]].to_numpy(float)] + deposits[1:]
    near_ree = _near(p, np.vstack(ree_dep), ORE_RADIUS_KM)
    nb_only = u[(u["niobium"].fillna(0) > 0) & ~(u["ree"].fillna(0) > 0)]
    near_nb = _near(p, nb_only[["longitude", "latitude"]].to_numpy(float), ORE_RADIUS_KM) & ~near_ree
    flags = dict(ore=dep & (alk | (carb & near_dep)), ree_mineralised=dep & (alk | (carb & near_any)),
                 ore_ree=dep & carb & near_ree, ore_nb_only=dep & carb & near_nb)
    return flags


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--classes", nargs="*", default=list(CLASSES))
    a = ap.parse_args()
    cfg = load_config(a.config)
    der = cfg["paths"]["derived"]
    p = base_points(der, a.tag, a.classes)
    print(f"{len(p)} points: " + ", ".join(f"{g} {n}" for g, n in p.groupby(['group', 'kind']).size().items()))
    nm = pd.read_csv(der / f"null_mask_{a.tag}.csv")[["pt_id", "ice", "other_family_km"]]
    p = p.merge(nm, on="pt_id", how="left")
    p = p.merge(pd.read_csv(der / f"craton_{a.tag}.csv"), on="pt_id", how="left")
    p = p.merge(pd.read_csv(der / f"point_fill_{a.tag}.csv"), on="pt_id", how="left")
    p = p.merge(read_table(der / f"water_{a.tag}"), on="pt_id", how="left")
    try:
        p = p.merge(read_table(der / f"mantle_memory_{a.tag}"), on="pt_id", how="left")
        # the reservoir without the cap on single-step deliveries (scripts/90 --spike-factor 1e9 --pilot
        # --out-suffix _nocap), as a sensitivity measure (review 2026-11-08)
        nocap = der / f"mantle_memory_{a.tag}_nocap.parquet"
        if nocap.exists():
            nc = read_table(der / f"mantle_memory_{a.tag}_nocap")
            nc = nc[["pt_id"] + [c for c in ("mem_carbon", "mem_water_out") if c in nc]]
            p = p.merge(nc.rename(columns={c: f"{c}_nocap" for c in nc.columns if c != "pt_id"}), on="pt_id", how="left")
        else:
            print("  mantle_memory_<tag>_nocap not found: the uncapped-reservoir sensitivity is skipped")
    except FileNotFoundError:
        print("  mantle_memory not found: run scripts/90 for the diffusing-reservoir measures")
    try:
        p = p.merge(read_table(der / f"dynamic_topography_{a.tag}"), on="pt_id", how="left")
    except FileNotFoundError:
        print("  dynamic_topography not found: run scripts/88 first for the mantle-flow measures")
    try:
        p = p.merge(read_table(der / f"primer_history_{a.tag}"), on="pt_id", how="left")
    except FileNotFoundError:
        print("  primer_history not found: run scripts/82 first for the along-path measures")
    for fr in ("fixed", "track"):
        for T in (100, 200, 400):
            sub, out = f"h2o_mean_subducted_water_{fr}_{T}myr", f"h2o_mean_slab_outflux_{fr}_{T}myr"
            if sub in p and out in p:
                p[f"h2o_deep_{fr}_{T}myr"] = p[sub] - p[out]
    if "sub_total_path_full" in p and "out_total_path_full" in p:
        p["deep_total_path_full"] = p["sub_total_path_full"] - p["out_total_path_full"]
    p["mineable"] = (p["site_sed_m"] <= MINEABLE_KM * 1000).astype(float).where(p["site_sed_m"].notna())
    print("  continental divergence in the plate model ...", flush=True)
    p = pd.concat([p, divergence_features(p, der, a.tag)], axis=1)
    print("  large igneous provinces and LLSVPs ...", flush=True)
    p = pd.concat([p, lip_features(p, ROOT)], axis=1)
    print("  fertilised domains of Spandler et al. (2026) ...", flush=True)
    p = pd.concat([p, spandler_features(p, ROOT)], axis=1)
    print("  REVEAL lid thickness and gradient ...", flush=True)
    p = pd.concat([p, lid_features(p, ROOT)], axis=1)
    print("  time in the humid tropics after formation (Cao2024) ...", flush=True)
    p = pd.concat([p, tropical_features(p, cfg)], axis=1)
    for name, flag in ore_flags(p, ROOT, cfg).items():
        p[name] = flag
    dd = p[p["kind"] == "deposit"]
    print("  ore-bearing: " + ", ".join(f"{g} {int(n)}" for g, n in dd.groupby("group")["ore"].sum().items())
          + f"; carbonatites with REE ore {int(dd['ore_ree'].sum())}, with niobium ore only {int(dd['ore_nb_only'].sum())}")
    features = dict(FEATURES)
    features.update({c: "sensitivity" for c in p.columns
                     if re.match(r"dt_(m|uplift_20|uplift_60|uplift_after_20|rise_200|high_p500|rise_p500|high_p1000|rise_p1000)_(m[1-6]|young)$", c)
                     or re.match(r"mem_(carbon|carbon_sed|water_\w+)_k\d+_l\w+$", c)
                     or c in ("mem_water_sub", "mem_water_deep", "mem_carbon_sed", "mem_carbon_nocap", "mem_water_out_nocap")})
    stages = pd.DataFrame([(k, v, k in p.columns) for k, v in features.items()], columns=["feature", "stage", "present"])
    stages.to_csv(der / f"carbonatite_feature_stages_{a.tag}.csv", index=False)
    print(f"wrote {write_table(p, der / f'carbonatite_features_{a.tag}')}")
    miss = stages.loc[~stages["present"], "feature"].tolist()
    if miss:
        print("  features not available in this run: " + ", ".join(miss))


if __name__ == "__main__":
    main()
