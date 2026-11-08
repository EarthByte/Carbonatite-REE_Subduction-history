"""Harmonise deposit compilations into one table with a common schema.

Schema (one row per deposit):
    dep_id, name, country, lon, lat, dclass, subtype, age_ma, age_min_ma, age_max_ma,
    age_basis, metal_mt, ore_mt, source, source_id

``dclass`` codes

A class is one ore model, not one commodity: an association averaged over models that form
differently means nothing. The USGS sediment-hosted Zn-Pb table carries a conventional deposit
type, and it is that field, not the commodity, that sets the class below.

    SSC   sediment-hosted stratiform Cu          CD    clastic-dominated Zn-Pb (SEDEX)
    MVT   Mississippi Valley-type Zn-Pb          ZNREP Zn skarn and polymetallic replacement
                                                       (magmatic-hydrothermal contrast class)
    ZNPB  sediment-hosted Zn-Pb, model not given SSPB  sandstone-hosted Pb
    KIPU  Kipushi-type carbonate-hosted Cu-Zn
    CARB  carbonatite                            AIREE alkaline-igneous REE
    KIMB  kimberlite                             LAMP  lamproite
    NICU  magmatic Ni-Cu-PGE                     IOCG  iron oxide-Cu-Au
    VMS   volcanogenic massive sulfide           PCU   porphyry Cu (contrast class)
    POT   potash / evaporite                     PHOS  phosphorite
``age_basis``
    mineralisation (dated ore), compiled (the deposit age as the source database compiles it,
    Singer et al. 2009), host (host-rock or stratigraphic age), inferred, interval (a named
    time-scale interval converted to its midpoint).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, haversine_km

INTERVALS = ROOT / "data" / "reference" / "macrostrat_intervals.csv"
SCHEMA = ["dep_id", "name", "country", "lon", "lat", "dclass", "subtype", "age_ma",
          "age_min_ma", "age_max_ma", "age_basis", "metal_mt", "ore_mt", "source", "source_id"]


# --------------------------------------------------------------------------- ages
class IntervalLookup:
    """Named geological intervals (Macrostrat) to numeric ages."""

    ALIASES = {"early proterozoic": "paleoproterozoic", "middle proterozoic": "mesoproterozoic",
               "late proterozoic": "neoproterozoic", "tertiary": "paleogene-neogene"}

    def __init__(self, path: Path = INTERVALS):
        d = pd.read_csv(path)
        d["key"] = d["name"].str.lower().str.strip()
        d = d.sort_values("int_type").drop_duplicates("key")
        self.table = d.set_index("key")[["t_age", "b_age"]]
        self.table.loc["paleogene-neogene"] = (2.58, 66.0)

    def _one(self, text: str):
        """(top, base) in Ma of one interval name, after aliases. Early or lower, middle, and late or upper
        take the oldest, middle and youngest third of the interval. None if the name is unknown."""
        k = text.lower().strip().rstrip("?").strip()
        k = self.ALIASES.get(k, k)
        if k in self.table.index:
            return tuple(self.table.loc[k])
        for pre, frac in (("early ", (0.667, 1.0)), ("lower ", (0.667, 1.0)), ("middle ", (0.333, 0.667)),
                          ("late ", (0.0, 0.333)), ("upper ", (0.0, 0.333))):
            if k.startswith(pre) and k[len(pre):] in self.table.index:
                t, b = self.table.loc[k[len(pre):]]
                return t + frac[0] * (b - t), t + frac[1] * (b - t)
        return None

    def __call__(self, text):
        """Return (age_ma, min_ma, max_ma) for a text such as 'Early Mississippian' or
        'Permian-Triassic'; NaNs when nothing is recognised."""
        if not isinstance(text, str) or not text.strip():
            return (np.nan,) * 3
        parts = [p for p in re.split(r"\s*(?:-|–|—| to | or |/)\s*", text) if p.strip()]
        spans = [s for s in (self._one(p) for p in parts) if s is not None]
        if not spans:
            return (np.nan,) * 3
        lo = min(s[0] for s in spans)
        hi = max(s[1] for s in spans)
        return 0.5 * (lo + hi), lo, hi


_NUM_MA = re.compile(r"(\d+(?:\.\d+)?)\s*(?:±\s*(\d+(?:\.\d+)?))?\s*(?:[-–]\s*(\d+(?:\.\d+)?))?\s*(Ma|Ga|m\.y\.)", re.I)


def parse_age_text(text, lookup: IntervalLookup):
    """Numeric ages in free text take precedence over interval names."""
    if not isinstance(text, str):
        return (np.nan, np.nan, np.nan, None)
    m = _NUM_MA.search(text.replace("Â", ""))
    if m:
        scale = 1000.0 if m.group(4).lower() == "ga" else 1.0
        a = float(m.group(1)) * scale
        if m.group(3):
            b = float(m.group(3)) * scale
            return (0.5 * (a + b), min(a, b), max(a, b), "mineralisation")
        err = float(m.group(2)) * scale if m.group(2) else 0.0
        return (a, a - err, a + err, "mineralisation")
    age, lo, hi = lookup(text)
    return (age, lo, hi, "interval" if np.isfinite(age) else None)


def _num(x):
    return pd.to_numeric(x, errors="coerce")


# ------------------------------------------------------------------------ readers
def read_hoggard2020(path: Path) -> pd.DataFrame:
    """Hoggard et al. (2020, Nat. Geosci.) supplementary table, seven sheets."""
    sheets = {"PbZn-CD": "CD", "PbZn-MVT": "MVT", "Cu-sed": "SSC", "Magmatic Ni": "NICU",
              "VMS": "VMS", "Cu-por": "PCU", "IOCG": "IOCG"}
    metal_col = {"CD": "Zn (Mt)", "MVT": "Zn (Mt)", "SSC": "Cu (Mt)", "NICU": "Ni (Mt)",
                 "VMS": "Cu (Mt)", "PCU": "Cu (Mt)", "IOCG": "Cu (Mt)"}
    out = []
    for sheet, cls in sheets.items():
        d = pd.read_excel(path, sheet_name=sheet)
        d.columns = [str(c).strip() for c in d.columns]
        lon = d.filter(regex=r"^Lon").iloc[:, 0]
        lat = d.filter(regex=r"^Lat").iloc[:, 0]
        age = _num(d.get("Age (Ga)")) * 1000.0
        unc = _num(d.get("Age Uncertainty (Ga)", pd.Series(np.nan, index=d.index))) * 1000.0
        atype = d.get("Age Type", pd.Series("", index=d.index)).astype(str).str.lower()
        basis = np.where(atype.str.startswith("direct"), "mineralisation",
                         np.where(atype.str.contains("host"), "host", "inferred"))
        ore = d.filter(regex=r"^Ore").iloc[:, 0] if d.filter(regex=r"^Ore").shape[1] else np.nan
        out.append(pd.DataFrame({
            "name": d["Deposit"], "country": d.get("Country"), "lon": _num(lon), "lat": _num(lat),
            "dclass": cls, "subtype": "", "age_ma": age,
            "age_min_ma": age - unc.fillna(0), "age_max_ma": age + unc.fillna(0),
            "age_basis": basis, "metal_mt": _num(d.get(metal_col[cls])), "ore_mt": _num(ore),
            "source": "hoggard2020", "source_id": sheet + ":" + d.index.astype(str)}))
    return pd.concat(out, ignore_index=True)


def read_usgs_sir20105090z(path: Path) -> pd.DataFrame:
    """Zientek et al. (2014) global sediment-hosted Cu sites; ages derive from host units."""
    d = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    d = d[d["dep_type"].str.startswith("Sediment-hosted", na=False)]
    return pd.DataFrame({
        "name": d["name"], "country": d["country"], "lon": d["longitude"], "lat": d["latitude"],
        "dclass": "SSC", "subtype": d["dep_subtype"].fillna(""), "age_ma": _num(d["age_ma"]),
        "age_min_ma": np.nan, "age_max_ma": np.nan, "age_basis": "host",
        "metal_mt": _num(d["con_cu_t"]) / 1e6, "ore_mt": _num(d["tonnage_mt"]),
        "source": "usgs_sir20105090z", "source_id": d["gmrap_id"].astype(str),
        "_status": d["sitestatus"]})


#: the table's conventional deposit type, which is the ore model, mapped to a class. Zn skarn
#: and polymetallic replacement are magmatic-hydrothermal and belong with the contrast classes,
#: not with the rift-hosted ores; SEDEX is the same model as the clastic-dominated class.
ZNPB_MODEL = {"SEDEX": "CD", "MVT": "MVT", "ZnSkarn": "ZNREP", "POLYREPL": "ZNREP",
              "SSPb": "SSPB", "Kipushi": "KIPU"}


def read_usgs_sedznpb(path: Path) -> pd.DataFrame:
    """Singer et al. (2009) sediment-hosted Zn-Pb, split by its conventional deposit type.

    ``subtype`` keeps the host-lithology code (CA carbonate, SH shale, ML mixed clastic,
    SS sandstone), which is the second split the class supports.
    """
    d = pd.read_csv(path, encoding="latin1", low_memory=False)
    zn = _num(d["oreton"]) * _num(d["zngrd"]) / 100.0
    model = (d["prevtype"].fillna("").astype(str).str.split("[;,]").str[0].str.strip()
             .map(ZNPB_MODEL).fillna("ZNPB"))
    return pd.DataFrame({
        "name": d["depname"], "country": d["country"], "lon": d["longitude"], "lat": d["latitude"],
        "dclass": model, "subtype": d["deptype"], "age_ma": _num(d["agemy"]),
        # `depage`/`agemy` is the age of the deposit as the database compiles it (a skarn's
        # intrusion age, an ore age, or a host age where nothing else is known), not a host
        # age by construction
        "age_min_ma": np.nan, "age_max_ma": np.nan, "age_basis": "compiled",
        "metal_mt": zn, "ore_mt": _num(d["oreton"]),
        "source": "usgs_sedznpb", "source_id": d["rec_id"].astype(str)})


def read_usgs_sedexmvt(path: Path, lookup: IntervalLookup) -> pd.DataFrame:
    """Taylor et al. (2009) CD and MVT deposits; mineralisation age where reported."""
    d = pd.read_csv(path, encoding="latin1", low_memory=False)
    rows = []
    for _, r in d.iterrows():
        a, lo, hi, basis = parse_age_text(r["minagemeth"], lookup)
        if not np.isfinite(a):
            a, lo, hi = lookup(r["hostage"])
            basis = "host"
        rows.append((a, lo, hi, basis))
    ages = pd.DataFrame(rows, columns=["age_ma", "age_min_ma", "age_max_ma", "age_basis"])
    cls = d["deptype"].map({"CD": "CD", "MVT": "MVT"}).fillna("ZNPB")   # deptype is the model here
    return pd.concat([pd.DataFrame({
        "name": d["depname"], "country": d["country"], "lon": d["longitude"], "lat": d["latitude"],
        "dclass": cls, "subtype": d["tectset"].fillna(""),
        "metal_mt": _num(d["znton"]) / 1e6, "ore_mt": _num(d["oreton"]),
        "source": "usgs_sedexmvt", "source_id": d["rec_id"].astype(str)}), ages], axis=1)


def read_usgs_sedcu(path: Path) -> pd.DataFrame:
    """Cox et al. (2003) sediment-hosted Cu grade-tonnage deposits."""
    d = pd.read_csv(path, encoding="latin1", low_memory=False)
    return pd.DataFrame({
        "name": d["depname"], "country": d["country"], "lon": d["longitude"], "lat": d["latitude"],
        "dclass": "SSC", "subtype": d["deptype"], "age_ma": _num(d["agemy"]),
        "age_min_ma": np.nan, "age_max_ma": np.nan, "age_basis": "host",
        "metal_mt": _num(d["cuton"]) / 1e6, "ore_mt": _num(d["oreton"]),
        "source": "usgs_sedcu", "source_id": d["rec_id"].astype(str)})


def read_usgs_carbonatite(path: Path) -> pd.DataFrame:
    """USGS carbonatite compilation as rows of the deposit schema (class CARB, ages of mineralisation in
    Ma)."""
    d = pd.read_csv(path, encoding="latin1", low_memory=False)
    return pd.DataFrame({
        "name": d["depname"], "country": d["country"], "lon": d["longitude"], "lat": d["latitude"],
        "dclass": "CARB", "subtype": d["deptype"].fillna(""), "age_ma": _num(d["agemy"]),
        "age_min_ma": np.nan, "age_max_ma": np.nan, "age_basis": "mineralisation",
        "metal_mt": np.nan, "ore_mt": _num(d["oreton"]),
        "source": "usgs_carbonatite", "source_id": d["rec_id"].astype(str)})


def read_usgs_ree(path: Path, lookup: IntervalLookup) -> pd.DataFrame:
    """Orris & Grauch (2002) REE deposits; igneous carbonatite and alkaline types.

    Carbonatite-hosted REE joins the carbonatite class; alkaline-igneous REE is a separate
    class, because it shares neither the rift-timing nor the carbon association of carbonatites.
    """
    d = pd.read_csv(path, encoding="latin1", low_memory=False)
    d = d[d["deptype"].isin(["carb", "carb-re", "alk-ig"])]
    ages = pd.DataFrame([parse_age_text(t, lookup) for t in d["age_info"]],
                        columns=["age_ma", "age_min_ma", "age_max_ma", "age_basis"], index=d.index)
    return pd.concat([pd.DataFrame({
        "name": d["depname"], "country": d["country"], "lon": d["longitude"], "lat": d["latitude"],
        "dclass": np.where(d["deptype"] == "alk-ig", "AIREE", "CARB"),
        "subtype": d["deptype"], "metal_mt": np.nan, "ore_mt": np.nan,
        "source": "usgs_ree", "source_id": d["rec_id"].astype(str)}), ages], axis=1)


def read_usgs_potash(path: Path, lookup: IntervalLookup) -> pd.DataFrame:
    """Orris et al. (2014) potash; ages are the span of max_age and min_age stage names."""
    d = pd.read_csv(path, encoding="latin1", low_memory=False)
    rows = []
    for mx, mn in zip(d["max_age"], d["min_age"]):
        a1 = lookup(str(mx).split("-")[-1]) if isinstance(mx, str) else (np.nan,) * 3
        a2 = lookup(str(mn).split("-")[-1]) if isinstance(mn, str) else a1
        # neither interval name resolves for some records; that is a missing age, not a warning
        los = [v for v in (a1[1], a2[1]) if np.isfinite(v)]
        his = [v for v in (a1[2], a2[2]) if np.isfinite(v)]
        lo = min(los) if los else np.nan
        hi = max(his) if his else np.nan
        rows.append((0.5 * (lo + hi), lo, hi, "interval"))
    ages = pd.DataFrame(rows, columns=["age_ma", "age_min_ma", "age_max_ma", "age_basis"])
    return pd.concat([pd.DataFrame({
        "name": d["site_name"], "country": d["country"], "lon": d["longitude"], "lat": d["latitude"],
        "dclass": "POT", "subtype": d["basin_type"].fillna(""), "metal_mt": np.nan,
        "ore_mt": _num(d["rr_ore_mt"]), "source": "usgs_potash",
        "source_id": d["rec_no"].astype(str)}), ages], axis=1)


def read_usgs_phosphate(path: Path, lookup: IntervalLookup) -> pd.DataFrame:
    """Marine phosphate deposits of the USGS compilation as rows of the deposit schema (class PHOS), dated
    from the geological interval of their minimum age through ``lookup``."""
    d = pd.read_csv(path, encoding="latin1", low_memory=False)
    d = d[d["dep_type"].str.startswith("Marine", na=False)]
    ages = pd.DataFrame([lookup(t) for t in d["min_age_g"]],
                        columns=["age_ma", "age_min_ma", "age_max_ma"], index=d.index)
    ages["age_basis"] = "interval"
    return pd.concat([pd.DataFrame({
        "name": d["site_name"], "country": d["country"], "lon": d["longitude"], "lat": d["latitude"],
        "dclass": "PHOS", "subtype": d["sub_type"].fillna(""), "metal_mt": np.nan, "ore_mt": np.nan,
        "source": "usgs_phosphate", "source_id": d["rec_no"].astype(str)}), ages], axis=1)


def read_generic(path: Path, source: str, mapping: dict, dclass: str | None = None) -> pd.DataFrame:
    """Loader for compilations added later (Shirmard et al. 2026, kimberlite and carbonatite
    age compilations). ``mapping`` maps schema names to the file's column names."""
    p = Path(path)
    d = pd.read_excel(p) if p.suffix in (".xls", ".xlsx") else pd.read_csv(p, low_memory=False)
    out = pd.DataFrame({k: d[v] for k, v in mapping.items() if v in d})
    if dclass:
        out["dclass"] = dclass
    out["source"] = source
    out["source_id"] = d.index.astype(str)
    return out


def read_kimberlites_gp2019(path: Path) -> pd.DataFrame:
    """Giuliani & Pearson (2019, Elements 15) kimberlite localities, screened from Faure (2010).

    ``Age1_Ma`` is a measured age (method in ``Age1_Methode``); where it is missing, the age
    estimated from the cited age range (column ``Age estimated RDM``) is used and flagged
    ``inferred``.
    """
    d = pd.read_excel(path, header=2)
    a1 = _num(d["Age1_Ma"])
    est = _num(d["Age estimated RDM"])
    err = _num(d["Age1_Error_Plus_Minus"]).fillna(0)
    age = a1.fillna(est)
    basis = np.where(a1.notna(), "mineralisation", np.where(est.notna(), "inferred", None))
    typ = d["Type"].astype(str)
    return pd.DataFrame({
        "name": d["Name_Field"].fillna(d["Name"]), "country": d["Country"],
        "lon": _num(d["Longitude"]), "lat": _num(d["Latitude"]), "dclass": "KIMB",
        "subtype": np.where(typ.str.contains(r"\?"), "uncertain type", d["Diamonds"].astype(str)),
        "age_ma": age, "age_min_ma": np.where(a1.notna(), a1 - err, age),
        "age_max_ma": np.where(a1.notna(), a1 + err, age), "age_basis": basis,
        "metal_mt": np.nan, "ore_mt": np.nan, "source": "gp2019_kimberlite",
        "source_id": d.index.astype(str)})


def read_kimberlites_gernon2023(path: Path) -> pd.DataFrame:
    """Gernon et al. (2023) combined kimberlite database (Zenodo 7849142, CC-BY 4.0).

    Every one of the 1,141 localities carries a measured age with the mineral dated and the
    isotopic system, which is what the Giuliani & Pearson compilation lacks for most of its
    records. Diamond status is not in this table and is carried across from Giuliani & Pearson
    afterwards (``carry_diamond_status``), so the class keeps both its ages and its sub-models.
    """
    d = pd.read_csv(path, encoding="latin-1")
    age = _num(d["Age"])
    return pd.DataFrame({
        "name": d["Location"].astype(str) + " " + d["ID"].astype(str), "country": d["Country"],
        "lon": _num(d["Longitude"]), "lat": _num(d["Latitude"]), "dclass": "KIMB",
        "subtype": d["Method"].astype(str) + " on " + d["Notes_ref_"].astype(str),
        "age_ma": age, "age_min_ma": age, "age_max_ma": age, "age_basis": "mineralisation",
        "metal_mt": np.nan, "ore_mt": np.nan, "source": "gernon2023_kimberlite",
        "source_id": d["Source"].astype(str) + "_" + d["ID"].astype(str)})


def read_carbonatites_wk2008(path: Path) -> pd.DataFrame:
    """Woolley & Kjarsgaard (2008) global carbonatite occurrences (527; about half dated)."""
    d = pd.read_excel(path)
    return pd.DataFrame({
        "name": d["Locality"], "country": d["Country"], "lon": _num(d["Long"]), "lat": _num(d["Lat"]),
        "dclass": "CARB", "subtype": "carbonatite occurrence", "age_ma": _num(d["Age"]),
        "age_min_ma": np.nan, "age_max_ma": np.nan, "age_basis": "mineralisation",
        "metal_mt": np.nan, "ore_mt": np.nan, "source": "wk2008_carbonatite",
        "source_id": d["ID Number"].astype(str)})


HWZ_ANALYTICAL, HWZ_ANALYTICAL_MIN_MYR = 0.02, 1.0


def read_carbonatites_hwz2021(path: Path, include_uncertain: bool = False) -> pd.DataFrame:
    """Humphreys-Williams & Zahirovic (2021) carbonatite compilation (Zenodo 5968095, CC BY 4.0).

    The located occurrences of their GPlates shapefile (593, Woolley & Kjarsgaard 2008 catalogue
    numbers, 376 dated with an error and a reference). ``carbonatit`` flags the identification:
    'c' a carbonatite, 'c?' uncertain, '?' unknown; only 'c' is kept unless include_uncertain.
    An age of zero is undated. The age range is the stated error, or 2% (at least 1 Myr) where the
    stated error is zero.
    """
    import geopandas as gpd
    d = gpd.read_file(f"zip://{path}!1_CarbonatitesShapefile_WithAgeConstraints/carbonatites_gplates.shp")
    flag = d["carbonatit"].fillna("").str.strip()
    d = d[flag.isin(("c", "c?") if include_uncertain else ("c",))]
    age = _num(d["Age_ma"]).where(_num(d["Age_ma"]) > 0)
    err = _num(d["Error_ma"]).fillna(0.0)
    err = err.where(err > 0, np.maximum(HWZ_ANALYTICAL * age, HWZ_ANALYTICAL_MIN_MYR))
    return pd.DataFrame({
        "name": d["Name"], "country": d["Country"], "lon": _num(d["Long"]), "lat": _num(d["Lat"]),
        "dclass": "CARB", "subtype": "carbonatite occurrence", "age_ma": age,
        "age_min_ma": age - err, "age_max_ma": age + err, "age_basis": "mineralisation",
        "metal_mt": np.nan, "ore_mt": np.nan, "source": "hwz2021_carbonatite",
        "source_id": d["Occurrence"].astype(str), "age_ref": d["reference"].fillna("").astype(str)}).reset_index(drop=True)


def read_lamproites_georoc(path: Path) -> pd.DataFrame:
    """Dated lamproite samples (GEOROC extract, Aug 2023); one row per sample location."""
    d = pd.read_csv(path)
    d = d.drop_duplicates(["Longitude", "Latitude", "Age"])
    return pd.DataFrame({
        "name": "", "country": "", "lon": _num(d["Longitude"]), "lat": _num(d["Latitude"]),
        "dclass": "LAMP", "subtype": "lamproite", "age_ma": _num(d["Age"]),
        "age_min_ma": np.nan, "age_max_ma": np.nan, "age_basis": "mineralisation",
        "metal_mt": np.nan, "ore_mt": np.nan, "source": "georoc_lamproite",
        "source_id": d.index.astype(str)})


# ------------------------------------------------------------------------ assembly
BUILTIN_SOURCES = {
    "hoggard2020": ("hoggard2020.xls", read_hoggard2020, False),
    "usgs_sir20105090z": ("usgs_sir20105090z/sedcu_deps_pros.csv", read_usgs_sir20105090z, False),
    "usgs_sedznpb": ("usgs_sedznpb/main.csv", read_usgs_sedznpb, False),
    "usgs_sedexmvt": ("usgs_sedexmvt/main.csv", read_usgs_sedexmvt, True),
    "usgs_sedcu": ("usgs_sedcu/main.csv", read_usgs_sedcu, False),
    "usgs_carbonatite": ("usgs_carbonatite/main.csv", read_usgs_carbonatite, False),
    "usgs_ree": ("usgs_ree/main.csv", read_usgs_ree, True),
    "usgs_potash": ("usgs_potash/deposit.csv", read_usgs_potash, True),
    "usgs_phosphate": ("usgs_phosphate/main.csv", read_usgs_phosphate, True),
    "gp2019_kimberlite": ("magmatic/kimberlites_Giuliani_Pearson_2019.xlsx", read_kimberlites_gp2019, False),
    "gernon2023_kimberlite": ("magmatic/kimberlites_Gernon_etal_2023.csv", read_kimberlites_gernon2023, False),
    "wk2008_carbonatite": ("magmatic/carbonatites_Woolley_Kjarsgaard_2008.xls", read_carbonatites_wk2008, False),
    # not in config source_priority: the shared compilation keeps Woolley & Kjarsgaard (2008); the
    # carbonatite paper builds its own set from this source (scripts/89)
    "hwz2021_carbonatite": ("magmatic/HumphreysWilliams_Zahirovic_2021/1_CarbonatitesShapefile_WithAgeConstraints.zip",
                            read_carbonatites_hwz2021, False),
    "georoc_lamproite": ("magmatic/lamproites_georoc_2023.csv", read_lamproites_georoc, False),
}


DIAMOND_STATUS = ("DM", "NON-DM")


def carry_diamond_status(df: pd.DataFrame, parts: list[pd.DataFrame], radius_km: float = 25.0):
    """Give each surviving kimberlite the diamond status of the nearest Giuliani & Pearson
    record within ``radius_km``.

    Deduplication keeps the record from the higher-priority source and drops the other, which
    for kimberlites would mean trading the diamondiferous / non-diamondiferous split for
    measured ages. The status is a property of the pipe rather than of the compilation, so it
    is carried across instead of lost.
    """
    from scipy.spatial import cKDTree

    from .common import km_to_chord, lonlat_to_xyz

    ref = pd.concat([p for p in parts if "source" in p and
                     (p["source"] == "gp2019_kimberlite").any()], ignore_index=True) \
        if parts else pd.DataFrame()
    if not len(ref):
        return df
    ref = ref[ref["subtype"].isin(DIAMOND_STATUS)]
    need = (df["dclass"] == "KIMB") & ~df["subtype"].isin(DIAMOND_STATUS)
    if not len(ref) or not need.any():
        return df
    tree = cKDTree(lonlat_to_xyz(ref["lon"], ref["lat"]))
    sub = df[need]
    d, j = tree.query(lonlat_to_xyz(sub["lon"], sub["lat"]))
    hit = d <= km_to_chord(radius_km)
    idx = sub.index[hit]
    df.loc[idx, "subtype"] = ref["subtype"].to_numpy()[j[hit]]
    print(f"  diamond status carried to {len(idx)} kimberlite records from Giuliani & Pearson")
    return df


def deduplicate(df: pd.DataFrame, priority: list[str], radius_km: float) -> pd.DataFrame:
    """Within each class, drop records lying within ``radius_km`` of a record from a
    higher-priority source. Records from the same source are never merged."""
    rank = {s: i for i, s in enumerate(priority)}
    df = df.assign(_rank=df["source"].map(rank).fillna(len(rank))).sort_values("_rank")
    keep = np.ones(len(df), bool)
    # one duplicate-detection group per commodity family, so the same deposit reported by two
    # sources under different models is still caught
    zn_family = {"ZNPB": "ZN", "CD": "ZN", "MVT": "ZN", "ZNREP": "ZN", "SSPB": "ZN", "KIPU": "ZN"}
    for _, g in df.groupby(df["dclass"].replace(zn_family).replace({"AIREE": "CARB"})):
        idx = g.index.to_numpy()
        lon, lat, src = g["lon"].to_numpy(), g["lat"].to_numpy(), g["source"].to_numpy()
        pos = {k: i for i, k in enumerate(df.index)}
        for j in range(len(g)):
            if not keep[pos[idx[j]]]:
                continue
            d = haversine_km(lon[j], lat[j], lon[j + 1:], lat[j + 1:])
            for k in np.nonzero((d < radius_km) & (src[j + 1:] != src[j]))[0]:
                keep[pos[idx[j + 1 + k]]] = False
    return df[keep].drop(columns="_rank")


def assign_clusters(df: pd.DataFrame, radius_km: float) -> pd.DataFrame:
    """District clusters: deposits of one class linked by chains of separations below
    ``radius_km``. ``w_cluster`` = 1 / cluster size gives every district one vote."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    from .common import km_to_chord, lonlat_to_xyz
    cid = np.empty(len(df), object)
    for c, idx in df.groupby("dclass").indices.items():
        xyz = lonlat_to_xyz(df["lon"].to_numpy()[idx], df["lat"].to_numpy()[idx])
        m = cKDTree(xyz).sparse_distance_matrix(cKDTree(xyz), km_to_chord(radius_km))
        _, lab = connected_components(m, directed=False)
        cid[idx] = [f"{c}_c{v:04d}" for v in lab]
    df = df.assign(cluster_id=cid)
    df["w_cluster"] = 1.0 / df.groupby("cluster_id")["cluster_id"].transform("size")
    return df


def build_deposit_table(raw_dir: Path, priority: list[str], radius_km: float,
                        extra: list[pd.DataFrame] | None = None) -> pd.DataFrame:
    """The shared deposit compilation. Every source named in ``priority`` is read from ``raw_dir`` (with
    the ``extra`` tables), rows without valid coordinates are dropped, missing age bounds are set to the
    age, duplicates within ``radius_km`` are merged by source priority (deduplicate), diamond status is
    carried over, and ids <class>_<n> are assigned."""
    lookup = IntervalLookup()
    parts = []
    for name, (rel, fn, needs_lookup) in BUILTIN_SOURCES.items():
        if name not in priority:
            # only the sources named in config deposits.source_priority enter the shared compilation;
            # the others are read by the paper-specific sets (scripts/72, 89)
            print(f"  [not in source_priority] {name}")
            continue
        p = Path(raw_dir) / rel
        if not p.exists():
            print(f"  [skip] {name}: {p} not found")
            continue
        d = fn(p, lookup) if needs_lookup else fn(p)
        print(f"  {name:20s} {len(d):6d} records")
        parts.append(d)
    parts += extra or []
    df = pd.concat(parts, ignore_index=True)
    df["lon"], df["lat"] = _num(df["lon"]), _num(df["lat"])
    df = df[np.isfinite(df["lon"]) & np.isfinite(df["lat"]) & (df["lat"].abs() <= 90)]
    df["age_min_ma"] = df["age_min_ma"].fillna(df["age_ma"])
    df["age_max_ma"] = df["age_max_ma"].fillna(df["age_ma"])
    df = deduplicate(df.reset_index(drop=True), priority, radius_km)
    df = carry_diamond_status(df, parts)
    df["dep_id"] = [f"{c}_{i:05d}" for i, c in enumerate(df["dclass"])]
    for col in ("name", "country", "subtype", "age_basis", "source", "source_id"):
        df[col] = df[col].fillna("").astype(str)
    extra_cols = [c for c in df.columns if c not in SCHEMA]
    return df[SCHEMA + extra_cols].reset_index(drop=True)
