#!/usr/bin/env python
"""Markdown tables of the carbonatite supplement, written from the derived tables so that every
number in them comes from the workflow.

  t1   every primary measure, by stage of the mineral system: the probability that the value
       beneath a district exceeds that beneath reference lithosphere of the same age, with its
       95% interval, for carbonatites, ore-bearing carbonatites, alkaline-silicate REE districts,
       kimberlites and porphyry copper (scripts/84, part compare); † marks results that survive
       Holm correction within the class
  t2   the same measures at equal craton-edge distance and exploration density, and the two
       direct comparisons: ore-bearing against barren carbonatites, and alkaline-silicate REE
       districts against carbonatites reweighted to their ages
  t3   sensitivity: the twelve settings of the mantle reservoir (scripts/90), slab water carried
       below 125 km over shorter windows, each mantle-flow model alone, the lid thresholds and the
       LLSVP outlines at two and four of four models
  t4   the fertilised domains of Spandler et al. (2026) (scripts/81): share of districts and
       reference positions inside a domain, at equal craton-edge distance, exploration density
       and shear-wave gradient, and their REE deposits split by host
  t5   isotopes and lanthanum against subducted volatiles (scripts/86)
  t6   numbers through time against subduction (scripts/85): correlations and recurrence

    python tools/carbonatite_tables.py t1 [--tag carb26]
    python tools/carbonatite_tables.py all > /tmp/carbonatite_tables.md
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLASSES = [("CARB", "Carbonatites"), ("CARB ore", "Ore-bearing carbonatites"),
           ("AIREE", "Alkaline-silicate REE"), ("KIMB", "Kimberlites"), ("PCU", "Porphyry Cu")]
DIRECT = [("CARB ore vs barren", "Ore-bearing against barren carbonatites"),
          ("AIREE vs CARB, age-matched", "Alkaline-silicate REE against carbonatites of the same ages")]
EQUAL = "auc at equal craton-edge distance and exploration density"
STAGES = [("source", "What the mantle and lithosphere had received"),
          ("lithosphere", "Lithosphere the melt rose through"),
          ("trigger", "What could have started melting"),
          ("upgrading", "Upgrading after formation"),
          ("exposure", "Exposure and exploration")]

LABEL = {
    # source
    "h2o_mean_slab_outflux_track_100myr": "Slab water released above 125 km, along the plate path, 100 Myr before",
    "h2o_mean_slab_outflux_track_400myr": "Slab water released above 125 km, along the plate path, 400 Myr before",
    "h2o_mean_subducted_water_track_400myr": "All subducted water, along the plate path, 400 Myr before",
    "c_mantle_subducted_carbon_track_800myr": "Subducted carbon, along the plate path, 800 Myr before",
    "c_mantle_subducted_carbon_stock": "Subducted carbon summed beneath the mantle position since 1800 Ma",
    "c_organic_subducted_carbon_stock": "Subducted organic carbon summed beneath the mantle position",
    "out_sed_path_full": "Slab water released above 125 km from sediment, entire plate path",
    "out_crust_path_full": "Slab water released above 125 km from oceanic crust, entire plate path",
    "out_mantle_path_full": "Slab water released above 125 km from serpentinised mantle, entire plate path",
    "out_total_path_full": "Slab water released above 125 km, entire plate path",
    "out_total_path_1200myr": "Slab water released above 125 km, plate path, 1,200 Myr before",
    "out_total_path_lag_myr": "Mean time before formation at which the slab water arrived (Myr)",
    "sub_total_path_full": "All subducted water, entire plate path",
    "c_mantle_path_full": "Subducted carbon, entire plate path",
    "c_mantle_path_lag_myr": "Mean time before formation at which the subducted carbon arrived (Myr)",
    "over_900km_myr": "Time on an overriding plate within 900 km of a trench (Myr)",
    "over_900km_since_myr": "Time since the last such episode of at least 100 Myr (Myr)",
    "over_900km_since_censored_myr": "Time since the last such episode, or the length of the covered history where there is none (Myr)",
    "over_900km_episode": "Share of districts with such an episode",
    "over_500km_myr": "Time on an overriding plate within 500 km of a trench (Myr)",
    "over_900km_stable_myr": "Time within 900 km of a stationary trench (Myr)",
    "over_900km_stable_since_myr": "Time since the last episode at a stationary trench (Myr)",
    "h2o_deep_fixed_400myr": "Slab water carried below 125 km, beneath the mantle position, 400 Myr before",
    "h2o_deep_track_400myr": "Slab water carried below 125 km, along the plate path, 400 Myr before",
    "deep_total_path_full": "Slab water carried below 125 km, entire plate path",
    "mem_carbon": "Subducted carbon held in the mantle reservoir",
    "mem_water_out": "Slab water released above 125 km held in the mantle reservoir",
    "out_total_path_400myr": "Slab water released above 125 km, plate path, 400 Myr before",
    "out_total_path_800myr": "Slab water released above 125 km, plate path, 800 Myr before",
    "c300_path_full": "Subducted carbon (300 km smoothing), entire plate path",
    # lithosphere
    "fml_primed": "Inside a fertilised domain formed before the site (share)",
    "fml_age_gap_myr": "Age of that domain before the site (Myr)",
    "craton_dist_km": "Distance to a craton edge at the age of the site (km)",
    "vsh150": "Shear-wave velocity at 150 km",
    "vsh_grad150": "Horizontal gradient of shear-wave velocity at 150 km",
    "site_crust_km": "Crustal thickness (km)",
    "site_beta": "Apparent stretching factor",
    "lid_km": "Thickness of the fast lithospheric lid (km)",
    "lid_grad": "Horizontal gradient of lid thickness",
    # trigger
    "divcont_dist_km": "Distance to continental divergence at formation (km)",
    "divcont_prior20_km": "Closest continental divergence over the 20 Myr before (km)",
    "dt_m": "Dynamic topography at formation (m)",
    "dt_uplift_20": "Change in dynamic topography over the 20 Myr before (m)",
    "dt_uplift_60": "Change in dynamic topography over the 60 Myr before (m)",
    "dt_rise_200": "Rise in dynamic topography since its lowest point in the 200 Myr before (m)",
    "dt_high_p500": "Share of the 80 Myr around formation with high dynamic topography within 500 km",
    "dt_rise_p500": "Share of the 80 Myr around formation with rising dynamic topography within 500 km",
    "lip_before_0_20_km": "Distance to a large igneous province emplaced 0–20 Myr before (km)",
    "lip_before_20_70_km": "Distance to a large igneous province emplaced 20–70 Myr before (km)",
    "lip_after_0_40_km": "Distance to a large igneous province emplaced 0–40 Myr after (km)",
    "llsvp_edge_km": "Distance to the edge of a large low-shear-velocity province (km)",
    "llsvp_signed_km": "Signed distance to that edge, negative inside (km)",
    "plate_speed_cm_yr": "Plate speed (cm/yr)",
    "plate_turn_10myr_deg": "Change in direction of plate motion over 10 Myr (°)",
    "sz_dist_km": "Trench distance at formation (km)",
    "div_dist_km": "Distance to a divergent plate boundary at formation (km)",
    "abs_paleolat": "Palaeolatitude (°, absolute)",
    # upgrading
    "tropics_myr": "Time within 25° of the equator since formation (Myr)",
    "tropics_share": "Share of the time since formation within 25° of the equator",
    "tropics_last100_myr": "Time within 25° of the equator over the past 100 Myr (Myr)",
    # exposure
    "site_sed_m": "Present-day sediment thickness (m)",
    "mineable": "Under less than 1 km of sediment (share)",
    "other_family_km": "Distance to the nearest deposit of another family (km)",
}

MEM_VAR = {"mem_carbon": "subducted carbon", "mem_carbon_sed": "carbon of subducted sedimentary carbonate",
           "mem_water_out": "slab water released above 125 km", "mem_water_deep": "slab water carried below 125 km",
           "mem_water_sub": "all subducted water"}


def sens_label(f):
    """Readable label of a sensitivity measure."""
    m = re.fullmatch(r"(mem_[a-z_]+?)(?:_k(\d+)_l(none|\d+))?", f)
    if m and m.group(1) in MEM_VAR:
        k, loss = m.group(2) or "1000", m.group(3) or "none"
        held = ("closed reservoir" if loss == "none" else "no spreading") if k == "0" \
            else f"diffusivity {int(k):,} km² Myr⁻¹"
        held += ", no loss" if loss == "none" else f", loss over {int(loss):,} Myr"
        return f"Reservoir, {MEM_VAR[m.group(1)]}; {held}"
    m = re.fullmatch(r"h2o_deep_(fixed|track)_(\d+)myr", f)
    if m:
        where = "beneath the mantle position" if m.group(1) == "fixed" else "along the plate path"
        return f"Slab water carried below 125 km, {where}, {m.group(2)} Myr before"
    m = re.fullmatch(r"(dt_[a-z_0-9]+?)_(m\d|young)", f)
    if m:
        base = {"dt_high_p1000": "Share of the 80 Myr with high dynamic topography within 1,000 km",
                "dt_rise_p1000": "Share of the 80 Myr with rising dynamic topography within 1,000 km",
                "dt_uplift_after_20": "Change in dynamic topography over the 20 Myr after (m)"}.get(m.group(1)) \
            or LABEL.get(m.group(1), m.group(1))
        model = "Young et al. (2022)" if m.group(2) == "young" else f"mantle-flow case {m.group(2)[1:]}"
        return f"{base}; {model}"
    fixed = {"dt_uplift_after_20": "Change in dynamic topography over the 20 Myr after (m)",
             "dt_uplift_20_agree": "Models agreeing on uplift over the 20 Myr before (share)",
             "dt_high_p1000": "Share of the 80 Myr with high dynamic topography within 1,000 km",
             "dt_rise_p1000": "Share of the 80 Myr with rising dynamic topography within 1,000 km",
             "lid_km_1p0": "Lid thickness, 1.0% threshold (km)", "lid_km_2p0": "Lid thickness, 2.0% threshold (km)",
             "llsvp_edge_v2_km": "Distance to an LLSVP edge, two of four models (km)",
             "llsvp_edge_v4_km": "Distance to an LLSVP edge, four of four models (km)"}
    return fixed.get(f, f)


def span(lo, hi):
    """A 95% interval; 'to' where a bound is negative, so that the dash cannot read as a minus."""
    sep = " to " if min(lo, hi) < 0 else "–"
    return f"{lo:.2f}{sep}{hi:.2f}".replace("-", "−")


def num(v):
    """A median to four significant figures, with a true minus sign."""
    return "–" if not np.isfinite(v) else f"{v:.4g}".replace("-", "−")


def cell(r, col="auc", lo="lo", hi="hi", holm=True, n_class=None):
    """One table cell. Where the measure is defined for fewer districts than the class holds (a
    subduction episode the reconstruction does not contain, a dynamic-topography history older than
    the models, a window running past the grids), the number with a value follows as 'n = ...'."""
    if r is None or not np.isfinite(r[col]):
        return "–"
    s = f"{r[col]:.2f}"
    if lo and np.isfinite(r[lo]):
        s += f" ({span(r[lo], r[hi])})"
    if holm and np.isfinite(r.get("p_holm", np.nan)) and r["p_holm"] < 0.05:
        s += " †"
    nd = r.get("n_districts", np.nan)
    if n_class is not None and np.isfinite(nd) and int(nd) < n_class:
        s += f" [n = {int(nd)}]"
    return s


def compare(der, tag):
    d = pd.read_csv(der / f"carbonatite_systems_{tag}.csv")
    return d[d["part"] == "compare"]


def counts(c, sets):
    return {s: int(c.loc[c["set"] == s, "n_districts"].max()) for s, _ in sets}


def _measure_table(c, sets, feats, label, col="auc", lo="lo", hi="hi", holm=True):
    idx = c.set_index(["set", "feature"])
    n = counts(c, sets)
    out = ["| Measure | " + " | ".join(f"{lab} ({n[s]})" for s, lab in sets) + " |",
           "|---|" + "---|" * len(sets)]
    for f in feats:
        row = [idx.loc[(s, f)] if (s, f) in idx.index else None for s, _ in sets]
        out.append(f"| {label(f)} | " + " | ".join(cell(r, col, lo, hi, holm, n[s]) for r, (s, _) in zip(row, sets)) + " |")
    return out


def t1(der, tag):
    c = compare(der, tag)
    out = []
    for stage, title in STAGES:
        feats = list(dict.fromkeys(c.loc[(c["set"] == "CARB") & (c["stage"] == stage), "feature"]))
        body = _measure_table(c, CLASSES, feats, lambda f: LABEL.get(f, f))
        out += ([] if not out else [""]) + [f"*{title}*", ""] + body
    return "\n".join(out)


def t2(der, tag):
    c = compare(der, tag)
    out = []
    for stage, title in STAGES:
        feats = list(dict.fromkeys(c.loc[(c["set"] == "CARB") & (c["stage"] == stage), "feature"]))
        a = _measure_table(c, CLASSES, feats, lambda f: LABEL.get(f, f), col=EQUAL, lo=None, holm=False)
        b = _measure_table(c, DIRECT, feats, lambda f: LABEL.get(f, f))
        # one table per stage: equal-condition columns, then the direct comparisons
        rows = [x + " " + y.split("|", 2)[2].strip() for x, y in zip(a, b)]
        rows[1] = "|---|" + "---|" * (len(CLASSES) + len(DIRECT))
        out += ([] if not out else [""]) + [f"*{title}*", ""] + rows
    # the alkaline-silicate class by era against the carbonatites of its era (scripts/84; review 2026-10-18)
    # set names: "AIREE <era> vs CARB <era>, age-matched" and "AIREE <setting> vs CARB, age-matched"
    eras = [s_ for s_ in c["set"].unique() if s_.startswith("AIREE ") and " vs CARB" in s_ and "age-matched" in s_
            and "mineable" not in s_ and s_ != "AIREE vs CARB, age-matched"]
    if eras:
        feats = [f for f in ("over_900km_since_myr", "over_900km_since_censored_myr", "over_900km_episode",
                             "over_900km_myr", "sz_dist_km", "dt_rise_200") if f in set(c["feature"])]
        names = {"Phanerozoic": "Phanerozoic", "Proterozoic": "Proterozoic", "rift": "Rift and anorogenic setting",
                 "post-collisional": "Post-collisional and post-subduction setting"}
        order = ["Phanerozoic", "Proterozoic", "rift", "post-collisional"]
        eras = sorted(eras, key=lambda s_: order.index(s_.split(" ")[1]) if s_.split(" ")[1] in order else 9)
        sets_ = [(s_, names.get(s_.split(" ")[1], s_.split(" ")[1]) + ": alkaline-silicate REE against carbonatites of the same ages")
                 for s_ in eras]
        out += ["", "*Alkaline-silicate REE districts against carbonatite districts, by era and by tectonic setting*", ""]
        out += _measure_table(c, sets_, feats, lambda f: LABEL.get(f, f))
    return "\n".join(out)


def t3(der, tag):
    c = compare(der, tag)
    feats = list(dict.fromkeys(c.loc[(c["set"] == "CARB") & (c["stage"] == "sensitivity"), "feature"]))
    order = (lambda f: (0 if f.startswith("mem_") else 1 if f.startswith("h2o_deep") else
                        2 if f.startswith("dt_") else 3, f))
    return "\n".join(_measure_table(c, CLASSES, sorted(feats, key=order), sens_label))


GROUP = {"CARB": "Carbonatites", "AIREE": "Alkaline-silicate REE", "KIMB": "Kimberlites", "PCU": "Porphyry Cu"}
DOMAIN = {"primed": "formed before the site", "primed_le1800": "formed before the site, no older than 1800 Ma",
          "primed_within400": "formed within 400 Myr before", "primed_within800": "formed within 800 Myr before",
          "primed_within1200": "formed within 1,200 Myr before"}
COND = {"craton_dist_km": "craton-edge distance at the age of the site",
        "craton_present_km": "present-day craton-edge distance",
        "other_family_km": "exploration density", "vsh_grad150": "shear-wave gradient at 150 km"}


def t4(der, tag):
    s = pd.read_csv(der / f"spandler_reassessment_{tag}.csv")
    sh = s[s["kind"] == "share"]
    eq = s[s["kind"] == "share at equal"].set_index(["group", "measure", "condition"])
    conds = list(COND)
    out = ["| Class | Domain | Districts | Inside (95% interval) | Reference (95% interval) | "
           + " | ".join(f"Expected at equal {COND[k]}" for k in conds) + " |",
           "|---|---|---:|---|---|" + "---:|" * len(conds)]
    for _, r in sh.iterrows():
        exp = [f"{eq.loc[(r['group'], r['measure'], k), 'share_ref']:.2f}"
               if (r["group"], r["measure"], k) in eq.index else "–" for k in conds]
        out.append(f"| {GROUP[r['group']]} | {DOMAIN[r['measure']]} | {int(r['n'])} | "
                   f"{r['share_dep']:.2f} ({r['dep_lo']:.2f}–{r['dep_hi']:.2f}) | "
                   f"{r['share_ref']:.2f} ({r['ref_lo']:.2f}–{r['ref_hi']:.2f}) | " + " | ".join(exp) + " |")
    h = s[s["kind"] == "spandler by host"]
    out += ["", "| Deposits of Spandler et al. (2026) | Domain | Deposits | Inside | Expected, all positions "
                "(difference, 95% interval) | Expected at equal present-day craton-edge distance (difference) |",
            "|---|---|---:|---:|---|---|"]
    for (g, m), x in h.groupby(["group", "measure"], sort=False):
        a = x[x["condition"].isna()].iloc[0]
        b = x[x["condition"] == "craton_present_km"].iloc[0]
        out.append(f"| {g} | {DOMAIN[m]} | {int(a['n'])} | {a['share_dep']:.2f} | "
                   f"{a['share_ref']:.2f} ({a['diff_lo']:+.2f} to {a['diff_hi']:+.2f}) | "
                   f"{b['share_ref']:.2f} ({b['diff_lo']:+.2f} to {b['diff_hi']:+.2f}) |")
    pv = s[s["kind"] == "polygon vs volatiles"]
    key = {"out_total_path_full": "slab water released above 125 km, entire path",
           "sub_total_path_full": "all subducted water, entire path",
           "c_mantle_path_full": "subducted carbon, entire path",
           "out_sed_path_full": "slab water from sediment, entire path",
           "path_cover_to_ma": "length of the reconstructed history (Ma)",
           "h2o_mean_subducted_water_fixed_100myr": "subducted water in the mantle reservoir, 100 Myr before",
           "h2o_mean_subducted_water_track_400myr": "subducted water in the mantle reservoir along the path, 400 Myr before",
           "c_mantle_subducted_carbon_fixed_100myr": "subducted carbon in the mantle reservoir, 100 Myr before",
           "c_mantle_subducted_carbon_track_400myr": "subducted carbon in the mantle reservoir along the path, 400 Myr before",
           "c_mantle_subducted_carbon_stock": "subducted carbon in the mantle reservoir at the age of the point"}
    pv = pv[pv["measure"].isin(key)]
    if len(pv):
        out += ["", "| Reference points | Delivery measure | Domain | Inside / outside (n) | Probability that a point inside "
                    "a domain received more |", "|---|---|---|---:|---:|"]
        for _, r in pv.iterrows():
            dom = "formed before the point" if r["condition"] == "primed" else "formed within 800 Myr before the point"
            out.append(f"| {r['group']} | {key[r['measure']]} | {dom} | {int(r['n'])} / {int(r['n_ref'])} | {r['auc']:.2f} |")
    return "\n".join(out).replace("+-", "−").replace("-0.", "−0.")


ISO = {"d_eNd": "Departure of initial εNd from depleted mantle", "eNd_i": "Initial εNd",
       "d_sr": "Departure of initial ⁸⁷Sr/⁸⁶Sr from depleted mantle", "sr_i": "Initial ⁸⁷Sr/⁸⁶Sr",
       "pb206_204": "²⁰⁶Pb/²⁰⁴Pb", "la": "La (ppm)"}
ISO_Q = {"out_total_path_full", "out_sed_path_full", "sub_total_path_full", "c_mantle_path_full",
         "out_total_path_400myr", "over_900km_myr", "craton_dist_km", "vsh_grad150"}


def t5(der, tag):
    i = pd.read_csv(der / f"carbonatite_isotopes_{tag}.csv")
    d = i[(i["part"] == "dose") & (i["variant"] == "all samples") & i["quantity"].isin(ISO_Q)]
    out = ["| Class | Measure | Against | Districts | Rank correlation (95% interval) | Spatial blocks | "
           "Age held constant | Age and lithosphere held constant |",
           "|---|---|---|---:|---|---|---:|---:|"]
    for _, r in d.iterrows():
        lab = LABEL.get(r["quantity"], r["quantity"].replace("_", " "))
        out.append(f"| {GROUP[r['cls']]} | {ISO[r['measure']]} | {lab} | {int(r['n_districts'])} | "
                   f"{r['rho']:.2f} ({span(r['lo'], r['hi'])}) | {span(r['block_lo'], r['block_hi'])} | "
                   f"{r['rho_age_partialled']:.2f} | {r['rho_age_lithosphere_partialled']:.2f} |")
    sp = i[(i["part"] == "split") & (i["variant"] == "all samples")]
    out += ["", "| Class | Measure | Split | Districts | Probability (95% interval) | Median, group | Median, others |",
            "|---|---|---|---:|---|---:|---:|"]
    for _, r in sp.iterrows():
        out.append(f"| {GROUP[r['cls']]} | {ISO[r['measure']]} | {r['quantity']} | {int(r['n_districts'])} | "
                   f"{r['auc']:.2f} ({span(r['lo'], r['hi'])}) | {num(r['median_dep'])} | {num(r['median_ref'])} |")
    return "\n".join(out).replace("-0.", "−0.")


SERIES = {"CARB": "Carbonatites", "AIREE": "Alkaline-silicate REE", "KIMB": "Kimberlites", "PCU": "Porphyry Cu",
          "CARB share of CARB+KIMB": "Carbonatite share of carbonatites and kimberlites"}
FLUX = {"consumed_km2_yr": "consumption of oceanic lithosphere", "trench_km": "trench length"}
RECUR = {"plate": "same plate position", "mantle": "same mantle position", "mantle_only": "same mantle position only",
         "pairs": "pairs within 500 km"}


def t6(der, tag):
    s = pd.read_csv(der / f"carbonatite_secular_{tag}.csv")
    c = s[s["part"] == "correlation"]
    out = ["| Series | Against | Filter | Window | Districts | " +
           # scripts/85 applies a lead as round(lead / 10 Myr) bins, so the nominal 25 Myr lead is 20 Myr
           " | ".join(f"Lag {int(round(x / 10) * 10)} Myr" for x in sorted(c["lead_myr"].unique())) + " |",
           "|---|---|---|---|---:|" + "---|" * c["lead_myr"].nunique()]
    for (se, fl, fi, w), x in c.groupby(["series", "flux", "filter", "window"], sort=False):
        x = x.sort_values("lead_myr")
        cells = ["–" if not np.isfinite(r.rho) else f"{r.rho:.2f} ({r.p_shift:.2f})".replace("-", "−")
                 for r in x.itertuples()]
        nd = x["n_districts"].max()
        nd = f"{int(nd)}" if np.isfinite(nd) else "–"
        out.append(f"| {SERIES[se]} | {FLUX[fl]} | {fi} | {w.replace('-', '–')} | {nd} | " + " | ".join(cells) + " |")
    r = pd.read_csv(der / f"carbonatite_recurrence_{tag}.csv")
    out += ["", "| Class | Recurrence | Observed | Expected (95% range) | Ratio | Probability of excess | Probability of deficit |",
            "|---|---|---:|---|---:|---:|---:|"]
    for x in r.itertuples():
        kind, _, km = x.measure.rpartition("_")
        lab = RECUR["pairs"] if x.measure == "pairs" else f"{RECUR[kind]}, within {km} km"
        out.append(f"| {GROUP[x.cls]} | {lab} | {x.observed:,} | {x.null_mean:,.1f} ({x.null_lo:,.1f}–{x.null_hi:,.1f}) | "
                   f"{x.ratio:.2f} | {x.p_excess:.3f} | {x.p_deficit:.3f} |")
    return "\n".join(out).replace("-0.", "−0.")


TABLES = {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5, "t6": t6}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", choices=list(TABLES) + ["all"])
    ap.add_argument("--tag", default="carb26")
    ap.add_argument("--derived", default=None, help="derived-data folder (default: config paths.derived)")
    a = ap.parse_args()
    if a.derived:
        der = Path(a.derived)
    else:
        from rifts.common import load_config
        der = load_config(None)["paths"]["derived"]
    for name in (TABLES if a.table == "all" else [a.table]):
        if a.table == "all":
            print(f"<!-- {name} -->")
        print(TABLES[name](der, a.tag))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
