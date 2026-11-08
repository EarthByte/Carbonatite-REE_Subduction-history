#!/usr/bin/env python
"""Point set of the carbonatite paper, with carbonatites from Humphreys-Williams & Zahirovic (2021).

The shared compilation (scripts/01) takes its carbonatite occurrences from Woolley & Kjarsgaard
(2008): 527 located occurrences, 274 with a single age and no error. Humphreys-Williams & Zahirovic
(2021, Elements 17, 339; Zenodo 10.5281/zenodo.5968095, CC BY 4.0) revised that catalogue: 593
located occurrences under the same catalogue numbers, 376 dated, each age with an error and a
reference, many of them replacing older K-Ar or whole-rock ages with U-Pb (Kovdor 365 Ma rather than
500 Ma; Prairie Lake 1,164 Ma rather than 1,023 Ma). This step builds a separate deposit table for the
carbonatite paper from it, so that the shared compilation and every number of the other papers stays
as it is:

  carbonatites  every Humphreys-Williams & Zahirovic occurrence identified as a carbonatite ('c';
                the 32 'c?' and one '?' are left out) with an age, the carbonatites of the USGS
                carbonatite database and the carbonatite-hosted REE deposits of Orris & Grauch
                (2002) with an age; duplicates within the dedup radius are merged in that order of
                priority, after records without an age are removed, so that an undated record
                never displaces a dated one
  alkaline REE  the alkaline-igneous REE deposits of Orris & Grauch (2002), deduplicated against the
                carbonatites as in the shared compilation (one carbonatite family)
  kimberlites   the measured-age set of the kimberlite paper (deposits_kimb23, scripts/72)
  control       the porphyry copper deposits of the shared compilation, unchanged

Ages beyond config time.max_age_ma are left out. Plate IDs, validity at the age and districts
(50 km) are assigned exactly as in scripts/01.

Output: ONE table, data/derived/deposits_<tag>.parquet, through write_table. Then:

    python scripts/89_carbonatite_set.py
    python scripts/02_sample_kinematics.py --deposits deposits_carb26 --classes CARB AIREE KIMB PCU \\
        --nulls-per-class 2000 --null-weights district --tag carb26
    python scripts/01b_geological_context.py --tag carb26
    python scripts/03_water.py --tag carb26
    python scripts/15_rift_fill.py --tag carb26
    python scripts/17_null_masks.py --tag carb26
    python scripts/20_craton.py --tag carb26
    python scripts/82_primer_history.py --tag carb26
    python scripts/83_carbonatite_features.py --tag carb26
    python scripts/88_dynamic_topography.py --tag carb26
    python scripts/83_carbonatite_features.py --tag carb26
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import load_config, read_table, write_table  # noqa: E402
from rifts.deposits import (BUILTIN_SOURCES, IntervalLookup, assign_clusters, deduplicate,  # noqa: E402
                            read_carbonatites_hwz2021, read_usgs_carbonatite, read_usgs_ree)
from rifts.models import load_models  # noqa: E402

PRIORITY = ["hwz2021_carbonatite", "usgs_carbonatite", "usgs_ree"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--tag", default="carb26")
    ap.add_argument("--kimberlites", default="deposits_kimb23", help="deposit table the kimberlites come from")
    a = ap.parse_args()
    cfg = load_config(a.config)
    raw, der = Path(cfg["paths"]["raw_data"]), cfg["paths"]["derived"]
    max_age = cfg["time"]["max_age_ma"]

    # carbonatites and REE deposits, in order of priority for deduplication
    parts = [read_carbonatites_hwz2021(raw / BUILTIN_SOURCES["hwz2021_carbonatite"][0]),
             read_usgs_carbonatite(raw / BUILTIN_SOURCES["usgs_carbonatite"][0]),
             read_usgs_ree(raw / BUILTIN_SOURCES["usgs_ree"][0], IntervalLookup())]
    for d in parts:
        print(f"  {d['source'].iloc[0]:20s} {len(d):5d} records, {int(np.isfinite(d['age_ma']).sum())} with an age")
    c = pd.concat(parts, ignore_index=True)
    c["lon"], c["lat"] = pd.to_numeric(c["lon"], errors="coerce"), pd.to_numeric(c["lat"], errors="coerce")
    c = c[np.isfinite(c["lon"]) & np.isfinite(c["lat"]) & (c["lat"].abs() <= 90)]
    # undated records go before deduplication, so that they never displace a dated one
    c = c[np.isfinite(c["age_ma"]) & (c["age_ma"] > 0) & (c["age_ma"] <= max_age)].copy()
    c["age_min_ma"] = c["age_min_ma"].fillna(c["age_ma"])
    c["age_max_ma"] = c["age_max_ma"].fillna(c["age_ma"])
    c = deduplicate(c.reset_index(drop=True), PRIORITY, cfg["deposits"]["dedup_radius_km"])
    print("after deduplication:", c.groupby(["dclass", "source"]).size().to_dict())

    # plate ids in both frames and validity at the age, as in scripts/01; records off every plate go
    mantle, paleomag = load_models(cfg)
    c["pid"], c["poly_begin_ma"], _ = mantle.assign_plate_ids(c["lon"], c["lat"])
    c["pid_pm"], c["poly_begin_pm_ma"], _ = paleomag.assign_plate_ids(c["lon"], c["lat"])
    c["in_model_at_age"] = c["age_ma"] <= c["poly_begin_ma"]
    c = c[c["pid"] > 0].reset_index(drop=True)
    c["dep_id"] = [f"{k}26_{i:05d}" for i, k in enumerate(c["dclass"])]
    for col in ("name", "country", "subtype", "age_basis", "source", "source_id"):
        c[col] = c[col].fillna("").astype(str)

    # kimberlites of the kimberlite paper and porphyry copper of the shared compilation, then districts
    kimb = read_table(der / a.kimberlites)
    kimb = kimb[kimb["dclass"] == "KIMB"]
    shared = read_table(der / "deposits")
    pcu = shared[shared["dclass"] == "PCU"]
    cols = [x for x in pcu.columns if x in c.columns and x in kimb.columns and x not in ("cluster_id", "w_cluster")]
    out = pd.concat([c[cols + ["age_ref"]], kimb[cols], pcu[cols]], ignore_index=True)
    out = assign_clusters(out, cfg["deposits"]["cluster_radius_km"])
    path = write_table(out, der / f"deposits_{a.tag}")

    s = out.groupby("dclass").agg(n=("dep_id", "size"), districts=("cluster_id", "nunique"),
                                  in_model=("in_model_at_age", "sum"), age_med=("age_ma", "median"),
                                  age_max=("age_ma", "max"))
    print(s.to_string())
    old = shared[shared["dclass"].isin(("CARB", "AIREE"))]
    print("shared compilation for comparison:", old.groupby("dclass").agg(
        n=("dep_id", "size"), districts=("cluster_id", "nunique")).to_dict("index"))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
