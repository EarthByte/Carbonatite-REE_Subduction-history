#!/usr/bin/env python
"""Harmonise the deposit compilations into data/derived/deposits and assign plate IDs.

Plate IDs and static-polygon begin ages are assigned separately in each plate model.
A deposit whose age exceeds the begin age of its static polygon is not carried by the model
at its formation age; it keeps its row but is flagged (``in_model`` in later steps).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import load_config, write_table  # noqa: E402
from rifts.deposits import assign_clusters, build_deposit_table  # noqa: E402
from rifts.models import load_models  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    print("reading sources")
    dep = build_deposit_table(cfg["paths"]["raw_data"], cfg["deposits"]["source_priority"],
                              cfg["deposits"]["dedup_radius_km"])
    dep = dep[np.isfinite(dep["age_ma"]) & (dep["age_ma"] <= cfg["time"]["max_age_ma"])].copy()
    mantle, paleomag = load_models(cfg)
    print("assigning plate IDs")
    dep["pid"], dep["poly_begin_ma"], _ = mantle.assign_plate_ids(dep["lon"], dep["lat"])
    dep["pid_pm"], dep["poly_begin_pm_ma"], _ = paleomag.assign_plate_ids(dep["lon"], dep["lat"])
    dep["in_model_at_age"] = dep["age_ma"] <= dep["poly_begin_ma"]
    dep = dep[dep["pid"] > 0]
    dep = assign_clusters(dep.reset_index(drop=True), cfg["deposits"]["cluster_radius_km"])
    out = write_table(dep.reset_index(drop=True), cfg["paths"]["derived"] / "deposits")
    print(f"wrote {out}")
    s = dep.groupby("dclass").agg(n=("dep_id", "size"), districts=("cluster_id", "nunique"), in_model=("in_model_at_age", "sum"),
                                  age_med=("age_ma", "median"), age_max=("age_ma", "max"))
    print(s.to_string())


if __name__ == "__main__":
    main()
