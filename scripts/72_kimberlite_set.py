#!/usr/bin/env python
"""Kimberlite point set built on measured ages: the Gernon et al. (2023) compilation.

The deposit compilation (scripts/01) takes its kimberlites from Giuliani & Pearson (2019), in
which half of the localities younger than 400 Ma carry an age inferred from a cited range rather
than measured, many of them sharing one inferred value. The combined database of Gernon et al.
(2023; Zenodo 7849142, CC-BY 4.0) gives every locality a measured age with the mineral dated and
the isotopic system. This step builds a separate deposit table for the kimberlite paper from it,
so that the shared compilation, and every number of the other papers, stays as it is:

  kimberlites   every Gernon et al. locality with a finite eruption age, except helium exposure
                ages (a surface exposure age, not an eruption age); age ranges are not given in
                the release, so each age carries an assumed analytical uncertainty of 2% (at
                least 1 Myr) in age_min_ma / age_max_ma; diamond status is carried from the
                nearest Giuliani & Pearson locality within 25 km (rifts.deposits)
  control       the porphyry copper deposits of the shared compilation, unchanged

Plate IDs, validity at the age and districts (50 km) are assigned exactly as in scripts/01.

Output: ONE table, data/derived/deposits_kimb23.parquet, through a temporary name. Then:

    python scripts/72_kimberlite_set.py
    python scripts/02_sample_kinematics.py --deposits deposits_kimb23 --classes KIMB PCU \\
        --nulls-per-class 2000 --null-weights district --tag kimb23
    python scripts/01b_geological_context.py --tag kimb23
    python scripts/15_rift_fill.py --tag kimb23
    python scripts/17_null_masks.py --tag kimb23
    python scripts/20_craton.py --tag kimb23
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import load_config, read_table, write_table  # noqa: E402
from rifts.deposits import (BUILTIN_SOURCES, assign_clusters, carry_diamond_status,  # noqa: E402
                            read_kimberlites_gernon2023, read_kimberlites_gp2019)
from rifts.models import load_models  # noqa: E402

ANALYTICAL = 0.02            # assumed relative uncertainty of a measured age without a stated error
ANALYTICAL_MIN_MYR = 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--tag", default="kimb23")
    a = ap.parse_args()
    cfg = load_config(a.config)
    raw, der = Path(cfg["paths"]["raw_data"]), cfg["paths"]["derived"]

    # Gernon et al. (2023): measured eruption ages, with the dating method and the mineral dated
    g = read_kimberlites_gernon2023(raw / BUILTIN_SOURCES["gernon2023_kimberlite"][0])
    src = pd.read_csv(raw / BUILTIN_SOURCES["gernon2023_kimberlite"][0], encoding="latin-1")
    g["method"] = src["Method"].astype(str).to_numpy()
    g["mineral"] = src["Notes_ref_"].astype(str).to_numpy()
    n0 = len(g)
    g = g[np.isfinite(g["age_ma"]) & (g["age_ma"] > 0) & (g["method"] != "He-Exposure")]
    g = g[np.isfinite(g["lon"]) & np.isfinite(g["lat"]) & (g["lat"].abs() <= 90)]
    g = g[g["age_ma"] <= cfg["time"]["max_age_ma"]].copy()
    # the release gives no age ranges: an assumed analytical uncertainty of 2% (at least 1 Myr)
    err = np.maximum(ANALYTICAL * g["age_ma"], ANALYTICAL_MIN_MYR)
    g["age_min_ma"], g["age_max_ma"] = g["age_ma"] - err, g["age_ma"] + err
    print(f"Gernon et al. (2023): {n0} localities, {len(g)} kept (finite eruption age within "
          f"{cfg['time']['max_age_ma']} Ma, helium exposure ages excluded)")

    # diamond status from the nearest Giuliani & Pearson (2019) locality; anything else is 'unknown'
    gp = read_kimberlites_gp2019(raw / BUILTIN_SOURCES["gp2019_kimberlite"][0])
    g = carry_diamond_status(g.reset_index(drop=True), [gp])
    g.loc[~g["subtype"].isin(("DM", "NON-DM")), "subtype"] = "unknown"

    # plate IDs in the mantle-frame and the palaeomagnetic model; localities on no plate are dropped
    mantle, paleomag = load_models(cfg)
    g["pid"], g["poly_begin_ma"], _ = mantle.assign_plate_ids(g["lon"], g["lat"])
    g["pid_pm"], g["poly_begin_pm_ma"], _ = paleomag.assign_plate_ids(g["lon"], g["lat"])
    g["in_model_at_age"] = g["age_ma"] <= g["poly_begin_ma"]
    g = g[g["pid"] > 0].reset_index(drop=True)
    g["dep_id"] = [f"KIMB23_{i:05d}" for i in range(len(g))]
    for col in ("name", "country", "subtype", "age_basis", "source", "source_id"):
        g[col] = g[col].fillna("").astype(str)
    g = assign_clusters(g, cfg["deposits"]["cluster_radius_km"])

    # the porphyry copper control, unchanged from the shared compilation
    pcu = read_table(der / "deposits")
    pcu = pcu[pcu["dclass"] == "PCU"]
    cols = [c for c in pcu.columns if c in g.columns]
    out = pd.concat([g[cols + ["method", "mineral"]], pcu[cols]], ignore_index=True)
    path = write_table(out, der / f"deposits_{a.tag}")
    k = out[out["dclass"] == "KIMB"]
    young = k[k["age_ma"] <= 400]
    print(f"wrote {path}: {len(k)} kimberlites in {k['cluster_id'].nunique()} districts "
          f"({len(young)} in {young['cluster_id'].nunique()} districts within 400 Ma), "
          f"{len(pcu)} porphyry copper deposits")
    print("diamond status:", k["subtype"].value_counts().to_dict())
    print("methods:", k["method"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
