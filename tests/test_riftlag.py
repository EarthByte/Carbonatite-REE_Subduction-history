"""The rift-lag measure: distance to the nearest rift that ceased a given interval earlier.

Two rifts of known age and position, three points of known age, and distances checked against
haversine, so that a change to the binning or to the active/preceding distinction cannot pass
unnoticed.
"""
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import LineString
from rifts.riftlag import rift_lag_table
from rifts.common import haversine_km

rifts = gpd.GeoDataFrame({
    "onset_ma": [100.0, 300.0],
    "end_ma":   [ 80.0, 250.0],
    "name": ["A", "B"],
}, geometry=[LineString([(0, -10), (0, 10)]), LineString([(20, -10), (20, 10)])], crs="EPSG:4326")

pts = pd.DataFrame({"pt_id": [1, 2, 3], "lon": [5.0, 5.0, 5.0], "lat": [0.0, 0.0, 0.0],
                    "age_ma": [60.0, 90.0, 10.0]})
t = rift_lag_table(pts, rifts, max_km=4000.0, step_deg=0.25)
exp_A = haversine_km(5, 0, 0, 0); exp_B = haversine_km(5, 0, 20, 0)

ok = []
r0 = t.iloc[0]      # age 60: A ended 80 -> lag 20 (bin 10-20); B ended 250 -> lag 190 (bin 120-250)
ok.append(("A in 10-20 bin at right distance", abs(r0["rlag_10_20_km"] - exp_A) < 5))
ok.append(("velocity = d/lag", abs(r0["rvel_10_20_km_myr"] - exp_A / 20) < 0.5))
ok.append(("B in 120-250 bin", abs(r0["rlag_120_250_km"] - exp_B) < 15))
ok.append(("best preceding is A", abs(r0["rlag_best_km"] - exp_A) < 5 and abs(r0["rlag_best_myr"] - 20) < 0.01))
r1 = t.iloc[1]      # age 90: A is active (100 >= 90 >= 80)
ok.append(("A counted as active at 90 Ma", abs(r1["rlag_active_km"] - exp_A) < 5))
ok.append(("A not counted as preceding at 90 Ma", not np.isfinite(r1["rlag_0_10_km"]) and not np.isfinite(r1["rlag_10_20_km"])))
r2 = t.iloc[2]      # age 10: A ended 80 -> lag 70 (bin 40-70 excluded, 70-120 includes 70? lag=70 -> bin (40,70] )
ok.append(("A at lag 70 lands in the 40-70 window", abs(r2["rlag_40_70_km"] - exp_A) < 5))
def test_rift_lag_table():
    for name, passed in ok:
        assert passed, name


if __name__ == "__main__":
    for name, passed in ok:
        print(("PASS  " if passed else "FAIL  ") + name)
