"""Configuration, paths and spherical-geometry helpers shared by every step."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

EARTH_RADIUS_KM = 6371.009
ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    """Read config/config.yaml (or ``path``) and make every relative path under 'paths', except *_glob
    patterns, absolute under the repository root."""
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    for key, val in cfg["paths"].items():
        if isinstance(val, str) and not key.endswith("_glob"):
            p = Path(val).expanduser()
            cfg["paths"][key] = p if p.is_absolute() else ROOT / p
    return cfg


def lonlat_to_xyz(lon, lat):
    lon = np.radians(np.asarray(lon, float))
    lat = np.radians(np.asarray(lat, float))
    c = np.cos(lat)
    return np.column_stack([c * np.cos(lon), c * np.sin(lon), np.sin(lat)])


def chord_to_km(chord):
    """Unit-sphere chord length to great-circle distance in km."""
    chord = np.clip(np.asarray(chord, float), 0.0, 2.0)
    return 2.0 * np.arcsin(chord / 2.0) * EARTH_RADIUS_KM


def km_to_chord(km):
    return 2.0 * np.sin(np.asarray(km, float) / EARTH_RADIUS_KM / 2.0)


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def fibonacci_sphere(n: int):
    """Near-equal-area points on the sphere (lon, lat in degrees)."""
    i = np.arange(n) + 0.5
    lat = np.degrees(np.arcsin(1 - 2 * i / n))
    lon = np.degrees((np.pi * (1 + 5 ** 0.5) * i) % (2 * np.pi)) - 180.0
    return lon, lat


def write_table(df, path: Path):
    """Write one table as one file (parquet if pyarrow is present, else gzipped CSV).

    Each workflow step writes a single output file, written to a temporary name and
    renamed on completion so an interrupted run never replaces a good file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        final = path.with_suffix(".parquet")
        tmp = final.with_name(final.name + ".partial")
        df.to_parquet(tmp, index=False)
    except ImportError:
        final = path.with_suffix(".csv.gz")
        tmp = final.with_name(final.name + ".partial")
        df.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(final)
    write_provenance(final)
    return final


def write_provenance(path):
    """Library versions next to an output (``<name>.versions.json``); geometry and plate-model
    libraries can change results at the level of individual boundary cases."""
    import json
    import platform
    from importlib import metadata
    v = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "scipy", "shapely", "pyproj", "geopandas", "pyogrio",
                "pygplates", "gplately", "plate-model-manager"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    try:
        import shapely
        v["geos"] = shapely.geos_version_string
    except Exception:
        pass
    Path(path).with_name(Path(path).name + ".versions.json").write_text(json.dumps(v, indent=1))


def read_table(path: Path):
    import pandas as pd
    path = Path(path)
    for cand in (path.with_suffix(".parquet"), path.with_suffix(".csv.gz"), path):
        if cand.exists():
            return pd.read_parquet(cand) if cand.suffix == ".parquet" else pd.read_csv(cand)
    raise FileNotFoundError(path)
