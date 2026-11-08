"""Self-tests that need no external data beyond the PlateModelManager download.

Run:  python -m pytest -q tests
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rifts.common import chord_to_km, haversine_km, km_to_chord, lonlat_to_xyz  # noqa: E402
from rifts.deposits import IntervalLookup, parse_age_text  # noqa: E402
from rifts.stats import auc  # noqa: E402


def test_distance_round_trip():
    a, b = lonlat_to_xyz([0, 10], [0, 20])
    d = chord_to_km(np.linalg.norm(a - b))
    assert abs(d - haversine_km(0, 0, 10, 20)) < 1e-6
    assert abs(chord_to_km(km_to_chord(1234.5)) - 1234.5) < 1e-6


def test_age_parsing():
    lk = IntervalLookup()
    assert abs(lk("Permian")[0] - 275.5) < 3
    a, lo, hi = lk("Early Cretaceous-Late Cretaceous")
    assert lo < 70 and hi > 140
    assert parse_age_text("349 ± 12 Ma (K-Ar, mica)", lk)[:3] == (349.0, 337.0, 361.0)
    assert parse_age_text("38.3 to 50.5 Ma", lk)[3] == "mineralisation"
    assert parse_age_text("Middle Proterozoic-- 1580 Ma", lk)[0] == 1580.0


def test_auc_matches_mann_whitney():
    from scipy import stats
    rng = np.random.default_rng(0)
    x, y = rng.normal(1, 1, 40), np.round(rng.normal(0, 1, 70), 1)
    u = stats.mannwhitneyu(x, y).statistic / (len(x) * len(y))
    assert abs(auc(x, y) - u) < 1e-12
    # duplicating a value with half weight each leaves the weighted AUC unchanged
    x2 = np.append(x, x[0])
    w2 = np.append(np.ones(len(x)), 1.0)
    w2[0] = w2[-1] = 0.5
    assert abs(auc(x2, y, w2) - auc(x, y)) < 1e-12


def test_water_fixed_and_track(tmp_path):
    """Synthetic cumulative grids C(t) = (100 - t) * f(lat): delivery over T Myr is T * f."""
    import xarray as xr
    from rifts.water import WaterGrids, water_metrics

    lon = np.arange(-180, 180.1, 1.0)
    lat = np.arange(-90, 90.1, 1.0)
    f = np.cos(np.radians(lat))[:, None] * np.ones(len(lon))
    for t in range(0, 101):
        ds = xr.Dataset({"mean_slab_outflux": (("lat", "lon"), (100 - t) * f)}, coords={"lat": lat, "lon": lon})
        ds.to_netcdf(tmp_path / f"water_{t}Ma.nc")
    g = WaterGrids(tmp_path, "*.nc", ["mean_slab_outflux"])
    assert len(g.ages) == 101

    class Still:                      # a model in which nothing moves
        def reconstruct_points(self, lon, lat, pid, t):
            return np.asarray(lon, float), np.asarray(lat, float)

    pts = pd.DataFrame({"pt_id": ["a"], "lon": [10.0], "lat": [60.0], "pid": [1], "age_ma": [20.0]})
    w = water_metrics(pts, Still(), g, [30], along_track=True)
    assert abs(w["mean_slab_outflux_fixed_30myr"].iat[0] - 30 * 0.5) < 1e-6
    assert abs(w["mean_slab_outflux_track_30myr"].iat[0] - 30 * 0.5) < 1e-6


def test_water_component_layout(tmp_path):
    import xarray as xr
    from rifts.water import WaterGrids
    lon = np.arange(-180, 180.1, 2.0)
    lat = np.arange(-90, 90.1, 2.0)
    for comp, k in (("crust_bound", 1.0), ("mantle", 2.0)):
        d = tmp_path / f"Muller2025_mean_slab_outflux_300km_{comp}"
        d.mkdir()
        for t in (0, 1):
            xr.Dataset({"z": (("lat", "lon"), np.full((len(lat), len(lon)), k * (2 - t)))},
                       coords={"lat": lat, "lon": lon}).to_netcdf(d / f"cumulative_slab_outflux_{comp}_{t}.nc")
    g = WaterGrids(tmp_path, "*.nc", ["mean_slab_outflux"], ["crust_bound", "mantle"])
    assert list(g.ages) == [0, 1]
    assert np.allclose(g.sample(0, "mean_slab_outflux", [10.0], [5.0]), 6.0)


def test_outline_bent_rift_and_antimeridian():
    from shapely.geometry import LineString, MultiLineString
    from rifts.outlines import rift_outline
    # an L-shaped rift: two 500 km arms; the concave outline must not fill the corner
    bent = MultiLineString([LineString([(10, 0), (10, 4.5)]), LineString([(10, 0), (14.5, 0)]),
                            LineString([(10.3, 0.2), (10.3, 4.3)]), LineString([(10.2, 0.3), (14.3, 0.3)])])
    conc, conv, m = rift_outline(bent)
    assert m["area_concave_km2"] < 0.5 * m["area_convex_km2"]
    from shapely.geometry import Point
    assert not conc.covers(Point(13.0, 3.0))       # the empty corner of the L stays outside
    assert m["empty_frac_concave"] < m["empty_frac_convex"]
    for ln in bent.geoms:                       # every structure lies inside its outline
        assert conc.buffer(1e-6).covers(ln)
    # a rift straddling 180 degrees comes back split, not wrapped around the globe
    wrap = MultiLineString([LineString([(179.0, 60), (179.5, 61)]), LineString([(-179.6, 60.2), (-179.0, 61)])])
    conc, _, _ = rift_outline(wrap)
    assert conc.bounds[2] - conc.bounds[0] > 300      # pieces on both sides of the antimeridian
    assert all((p.bounds[2] - p.bounds[0]) < 5 for p in conc.geoms)


def test_bad_step_correction(tmp_path):
    """A spurious jump at 5 Ma is removed from every grid at or after it; stock and windows
    then match a series without the jump."""
    import xarray as xr
    from rifts.water import WaterGrids
    lon = np.arange(-180, 180.1, 2.0)
    lat = np.arange(-90, 90.1, 2.0)
    for t in range(0, 11):
        c = (10 - t) * 1.0 + (50.0 if t <= 5 else 0.0)          # 1 per Myr plus a jump of 50 at 5 Ma
        xr.Dataset({"z": (("lat", "lon"), np.full((len(lat), len(lon)), c))},
                   coords={"lat": lat, "lon": lon}).to_netcdf(tmp_path / f"cumulative_x_{t}.nc")
    raw = WaterGrids(tmp_path, "*.nc", ["x"], ["."], "{component}")
    fixed = WaterGrids(tmp_path, "*.nc", ["x"], ["."], "{component}", bad_steps=[5])
    s = lambda g, t: float(g.sample(t, "x", [0.0], [0.0])[0])  # noqa: E731
    assert s(raw, 3) - s(raw, 7) == 54.0
    assert abs(s(fixed, 3) - s(fixed, 7) - 4.0) < 1e-9
    assert abs(s(fixed, 0) - 10.0) < 1e-9 and abs(s(fixed, 8) - 2.0) < 1e-9


def test_zero_inflation_and_trench_stratification():
    """A delivery measure that is a pure function of trench distance loses its apparent signal
    once distance is held fixed; a measure with its own signal keeps it. Variables that are
    almost all zeros are flagged."""
    from rifts.stats import screen
    rng = np.random.default_rng(2)
    rows = []
    for i in range(600):
        dist = abs(rng.normal(500, 300))
        rows.append(dict(pt_id=f"d{i}", dclass="X", null_for="", lag_myr=0, in_model=True, w_cluster=1.0,
                         cluster_id=f"c{i % 60}", sz_dist_km=dist, distance_only=np.exp(-dist / 800),
                         own_signal=np.exp(-dist / 800) + 0.3 * rng.random(),
                         mostly_zero=(abs(rng.normal()) if i < 60 else 0.0)))
    for i in range(1200):
        dist = abs(rng.normal(2000, 800))
        rows.append(dict(pt_id=f"n{i}", dclass="NULL", null_for="X", lag_myr=0, in_model=True, w_cluster=1.0,
                         cluster_id=f"n{i}", sz_dist_km=dist, distance_only=np.exp(-dist / 800),
                         own_signal=np.exp(-dist / 800), mostly_zero=(abs(rng.normal()) if i < 24 else 0.0)))
    r = screen(pd.DataFrame(rows), ["X"], ["distance_only", "own_signal", "mostly_zero"]).set_index("predictor")
    assert r.loc["distance_only", "auc"] > 0.9
    assert r.loc["distance_only", "auc_given_trench_distance"] < 0.65
    assert r.loc["own_signal", "auc_given_trench_distance"] > 0.85
    assert r.loc["mostly_zero", "tie_limited"] and not r.loc["own_signal", "tie_limited"]


def test_survivors_rule():
    """A sign flip, a collapse under conditioning and a small class are all rejected;
    a result that keeps its separation is kept."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s11", Path(__file__).resolve().parents[1] / "scripts" / "11_survivors.py")
    s11 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s11)
    res = pd.DataFrame([
        # keeps 0.95 of its separation on 40 districts -> survives
        dict(dclass="A", predictor="p", lag_myr=0, n_districts=40, auc=0.70,
             auc_given_trench_distance=0.69, robust=True, tie_limited=False),
        # trench distance accounts for it
        dict(dclass="B", predictor="p", lag_myr=0, n_districts=40, auc=0.88,
             auc_given_trench_distance=0.54, robust=True, tie_limited=False),
        # sign flip
        dict(dclass="C", predictor="p", lag_myr=0, n_districts=40, auc=0.64,
             auc_given_trench_distance=0.34, robust=True, tie_limited=False),
        # too few districts
        dict(dclass="D", predictor="p", lag_myr=0, n_districts=8, auc=0.70,
             auc_given_trench_distance=0.69, robust=True, tie_limited=False),
        # right lag only
        dict(dclass="A", predictor="p", lag_myr=100, n_districts=40, auc=0.70,
             auc_given_trench_distance=0.69, robust=True, tie_limited=False),
    ])
    keep = s11.survivors(res)
    assert list(keep["dclass"]) == ["A"]
    assert len(keep) == 1


def test_matching_removes_a_confound():
    """Two sets that differ only because one is drawn from larger, older rifts separate on a
    variable correlated with size, and stop separating once the comparison is made inside cells
    of size and age."""
    from rifts.stats import boot_stratified_auc, strata_coverage
    rng = np.random.default_rng(3)
    n_null, n_dep = 1200, 80
    area_n, age_n = rng.lognormal(9, 1.2, n_null), rng.uniform(0, 600, n_null)
    area_d, age_d = rng.lognormal(10.5, 1.0, n_dep), rng.uniform(150, 600, n_dep)
    f = lambda area, age: np.log(area) + 0.002 * age + rng.normal(0, 0.3, len(area))  # noqa: E731
    null = pd.DataFrame(dict(area_km2=area_n, onset_ma=age_n, v=f(area_n, age_n)))
    dep = pd.DataFrame(dict(area_km2=area_d, onset_ma=age_d, v=f(area_d, age_d), w_cluster=1.0))
    raw = auc(dep["v"], null["v"])
    assert raw > 0.8
    m, lo, hi = boot_stratified_auc(dep, null, "v", by=("onset_ma", "area_km2"), n_bins=3)
    assert lo < m < hi
    assert m < raw - 0.1                       # most of the separation is the confound
    finer, _, _ = boot_stratified_auc(dep, null, "v", by=("onset_ma", "area_km2"), n_bins=5)
    assert finer < m                           # and finer cells remove more of it
    n_cells, frac = strata_coverage(dep, null, "v", ("onset_ma", "area_km2"), 3)
    assert n_cells >= 4 and frac > 0.8


def test_spine_follows_a_bent_rift():
    """The spine of an L-shaped rift turns with it, and the flank points stay on opposite
    sides. Bins are laid out along the bounding rectangle's long axis, so a right-angle bend is
    followed in part, not in full."""
    from shapely.geometry import LineString
    from rifts.obliquity import _azimuth, spine
    # a rift running north for 3 degrees then east for 3, 40 km wide
    line = LineString([(0.0, 0.0), (0.0, 3.0), (3.0, 3.0)]).buffer(0.2)
    sp = spine(line, pad_km=50.0, max_bins=9)
    assert sp is not None and len(sp["w_km"]) >= 4
    trend = _azimuth(sp["c_lon"], sp["c_lat"], sp["t_lon"], sp["t_lat"])
    turn = np.abs(((trend[-1] - trend[0]) + 180) % 360 - 180)
    assert turn > 20                                   # the trend follows the bend
    d = np.hypot(sp["left_lon"] - sp["right_lon"], sp["left_lat"] - sp["right_lat"])
    assert np.all(d > 0.5)                             # flanks are on opposite sides


def test_join_rift_settings_leaves_existing_columns_alone(tmp_path):
    from rifts.context import join_rift_settings
    der = tmp_path
    pd.DataFrame({"outline_id": ["R1", "R2"], "width_rect_km": [100.0, 200.0],
                  "own_obliq": [10.0, 80.0], "setting": ["a", "b"]}).to_csv(der / "rift_settings_t.csv",
                                                                            index=False)
    f = pd.DataFrame({"pt_id": [1, 2, 3], "lag_myr": [0, 0, 0], "arift_id": ["R1", "R2", None],
                      "arift_width_rect_km": [1.0, 2.0, 3.0]})
    out, num = join_rift_settings(f, der, "t", verbose=False)
    assert "arift_own_obliq" in num and "arift_width_rect_km" not in num
    assert list(out["arift_width_rect_km"]) == [1.0, 2.0, 3.0]          # not overwritten
    assert list(out["arift_own_obliq"].fillna(-1)) == [10.0, 80.0, -1.0]


def test_wedge_drape_reproduces_the_tutorial_case_studies():
    """The syn-rift share of the North Sea Central Graben and the Gulf of Suez, against the
    values the T55 tutorial reports for the same grids (0.59 and 0.50)."""
    import geopandas
    from rifts.basins import Grid, rift_fill
    from rifts.obliquity import spine
    root = Path(__file__).resolve().parents[1]
    sed_p = root / "data/external/Bird_Mooney_GST14_WGS84.nc"
    cru_p = root / "data/external/crustal-thickness-gmt-surface.nc"
    shp = root / "data/external/Evenick2021_GlobalBasins.shp"
    if not (sed_p.exists() and cru_p.exists() and shp.exists()):
        import pytest
        pytest.skip("thickness grids not present")
    sed, cru = Grid(sed_p), Grid(cru_p, scale=1e-3)
    b = geopandas.read_file(shp)
    for name, expected in (("Central North Sea", 0.59), ("Gulf of Suez", 0.50)):
        g = b[b["Basin Name"].astype(str).str.strip() == name]
        m = rift_fill(spine(g.geometry.iloc[0], pad_km=50.0, max_bins=9), sed, cru, n=81)
        assert abs(m["syn_share"] - expected) < 0.06, (name, m["syn_share"])
