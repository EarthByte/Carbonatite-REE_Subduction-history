#!/usr/bin/env python
"""Download the open deposit compilations into data/raw.

Each source is extracted into its own sub-folder, keeping only the files the workflow reads.
Existing files are left untouched; use --force to fetch again.
"""
import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import load_config  # noqa: E402

SOURCES = {
    # name: (url, {member-in-zip: target-name}) ; a None mapping means save the file as is
    "hoggard2020.xls": ("https://osf.io/download/r5mfz/", None),
    "usgs_sedznpb": ("https://mrdata.usgs.gov/sedznpb/sedznpb-csv.zip", {"main.csv": "main.csv"}),
    "usgs_sedexmvt": ("https://mrdata.usgs.gov/sedexmvt/sedexmvt-csv.zip", {"main.csv": "main.csv"}),
    "usgs_sedcu": ("https://mrdata.usgs.gov/sedcu/sedcu-csv.zip", {"main.csv": "main.csv"}),
    "usgs_sir20105090z": ("https://mrdata.usgs.gov/sir20105090z/sir20105090z-csv.zip",
                          {"sedcu_deps_pros.csv": "sedcu_deps_pros.csv"}),
    "usgs_carbonatite": ("https://mrdata.usgs.gov/carbonatite/carbonatite-csv.zip", {"main.csv": "main.csv"}),
    "usgs_ree": ("https://mrdata.usgs.gov/ree/ree-csv.zip", {"main.csv": "main.csv"}),
    "usgs_potash": ("https://mrdata.usgs.gov/potash/potash-csv.zip", {"deposit.csv": "deposit.csv"}),
    "usgs_phosphate": ("https://mrdata.usgs.gov/phosphate/phosphate-csv.zip", {"main.csv": "main.csv"}),
}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Muller_rifts/0.1"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    raw = load_config()["paths"]["raw_data"]
    raw.mkdir(parents=True, exist_ok=True)
    for name, (url, members) in SOURCES.items():
        if members is None:
            target = raw / name
            if target.exists() and not args.force:
                print(f"have  {name}")
                continue
            target.write_bytes(fetch(url))
            print(f"saved {name}")
            continue
        folder = raw / name
        if all((folder / t).exists() for t in members.values()) and not args.force:
            print(f"have  {name}")
            continue
        folder.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(fetch(url))) as z:
            for info in z.infolist():
                base = Path(info.filename).name
                if base in members:
                    (folder / members[base]).write_bytes(z.read(info))
        print(f"saved {name}")


if __name__ == "__main__":
    main()
