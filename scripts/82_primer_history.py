#!/usr/bin/env python
"""Subduction history along the plate path of each deposit and reference point, back to 1.8 Ga.

The primer of carbonatite and alkaline rare-earth systems is held to be lithospheric mantle
fertilised by slab-derived fluids and melts long before the magmas form (Spandler et al. 2026,
and the work they review). This step measures that history for every point of the chosen
classes and their reference points, following the point with its plate through the mantle
frame from its formation age back to the oldest grid (1,800 Ma) or to the time its crustal
block enters the plate model, whichever is younger.

Part ``water`` (needs the grid series on the SSD):

* ``<series>_path_full``: delivery integrated along the plate path over the entire covered
  history; ``<series>_path_<T>myr``: over the T Myr before formation (missing where the window
  runs past the covered history);
* ``<series>_path_lag_myr``: delivery-weighted mean time before formation at which it arrived
  (positive increments only);
* empty grid cells along the path count as no delivery;
* series are slab water released above 125 km by component (``out_sed``, ``out_crust``,
  ``out_mantle``) and in total (``out_total``), all subducted water (``sub_total``), and
  subducted carbon (``c_mantle``, 500 km smoothing, 0-1,800 Ma; ``c300``, 300 km, 0-1,500 Ma).
* ``path_cover_to_ma``: the oldest time integrated.

Part ``topo`` (resolved topologies of the mantle-frame model, every ``--step`` Myr):

* ``over_<D>km_myr``: Myr the point spent on the overriding plate within D km of a subduction
  zone before formation (D = 500, 700, 900 and 1,100 km);
* ``over_<D>km_longest_myr``: the longest continuous such episode;
* ``over_<D>km_since_myr``: time between formation and the end of the most recent episode
  lasting at least 100 Myr (the criterion of Spandler et al. 2026), missing if none;
* ``over_<D>km_first_myr``: time between formation and the start of the oldest such episode;
* ``over_<D>km_stable_myr``, ``_stable_longest_myr``, ``_stable_since_myr``: the same for episodes
  of at least 100 Myr at a stationary trench, one whose distance from the point varies by less than
  ``--stable-km`` (500 km) over the episode: the fixed-trench criterion of Spandler et al. (2026).
* ``over_<D>km_since_L<L>myr`` and ``over_<D>km_episode_L<L>myr``: the time since, and the presence of,
  the most recent episode of at least L = 50 or 150 Myr, a sensitivity test of the 100 Myr criterion;
* ``over_<D>km_after_myr``: how long after formation the point stayed on the overriding plate within D km,
  read forwards from the first step at or after formation (0 if it was not there at that step; at most
  100 Myr), which tells whether a lag of one step marks an episode that ended or one still under way.

Writes ONE table, data/derived/primer_history_<tag> (parquet), through a .partial file. With
``--parts`` a run of one part keeps the other part's columns from the existing table.

    python scripts/82_primer_history.py --tag full --max-points 300 --out-tag pilot
    python scripts/82_primer_history.py --tag full
"""
import argparse
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import chord_to_km, load_config, lonlat_to_xyz, read_table, write_table  # noqa: E402

CLASSES = ("CARB", "AIREE", "KIMB", "PCU")
WINDOWS = (100, 200, 400, 800, 1200)
INBOARD_KM = (500, 700, 900, 1100)
MIN_EPISODE_MYR = 100
SENS_EPISODE_MYR = (50, 150)  # shorter and longer qualifying episodes, a sensitivity test (review 2026-11-09)
AFTER_MYR = 100               # how far after formation the persistence of the setting is followed
SEARCH_KM = 2500.0            # trench search radius; distances beyond it are not needed


def point_list(cfg, tag, classes, max_points=None, seed=1):
    """The points of ``classes`` and their reference points at their sampled age (lag 0, draw 0), one row
    per point with class, age, present-day position and plate id; ``max_points`` draws a random subset
    for pilot runs."""
    f = read_table(cfg["paths"]["derived"] / f"features_{tag}")
    f = f[(f["lag_myr"] == 0) & (f["draw"] == 0)]
    keep = f["dclass"].isin(classes) | f["null_for"].isin(classes)
    p = (f.loc[keep, ["pt_id", "dclass", "null_for", "age_ma", "lon0", "lat0", "pid"]]
         .drop_duplicates("pt_id").rename(columns={"lon0": "lon", "lat0": "lat"}).reset_index(drop=True))
    if max_points and len(p) > max_points:
        p = p.sample(max_points, random_state=seed).reset_index(drop=True)
    return p


def block_begin(mantle, p):
    """Age at which each point's crustal block enters the model (no integration before it)."""
    try:
        _, begin, _ = mantle.assign_plate_ids(p["lon"].to_numpy(), p["lat"].to_numpy())
        return np.asarray(begin, float)
    except Exception:                                    # older model wrapper: no limit
        return np.full(len(p), np.inf)


def water_series(cfg):
    """(name, WaterGrids, variable) for every series sampled."""
    from rifts.water import WaterGrids
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location("w03", Path(__file__).with_name("03_water.py"))
    w03 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w03)
    glob = cfg["paths"]["water_grid_glob"]
    sets = {g["name"]: g for g in cfg["water"]["grid_sets"]}
    out = []
    h = sets["h2o"]
    tmpl = h.get("folder_template", "{component}")
    for comp, short in (("sediment_bound", "sed"), ("crust_bound", "crust"), ("mantle", "mantle")):
        g = WaterGrids(h["dir"], glob, ["mean_slab_outflux"], [comp], tmpl, w03.bad_steps_for(cfg, "h2o"))
        out.append((f"out_{short}", g, "mean_slab_outflux"))
    # total outflux is the sum of the three components, formed at accumulation (no second read)
    out.append(("out_total", None, ("out_sed", "out_crust", "out_mantle")))
    g = WaterGrids(h["dir"], glob, ["mean_subducted_water"], h["components"], tmpl, w03.bad_steps_for(cfg, "h2o"))
    out.append(("sub_total", g, "mean_subducted_water"))
    for name, short in (("c_mantle", "c_mantle"), ("c300_mantle", "c300")):
        if name in sets:
            s = sets[name]
            g = WaterGrids(s["dir"], glob, s["variables"], s.get("components"), s.get("folder_template", "{component}"),
                           w03.bad_steps_for(cfg, name))
            out.append((short, g, s["variables"][0]))
    return out


def run_water(p, mantle, series, windows, begin, tmax=None, progress=True):
    """Integrate every slab-input series along the plate path of each point, in 1 Myr steps from its
    formation age back to the oldest grid, the entry of its block into the model or ``tmax``. The
    increment of a step is the difference of the cumulative grid between consecutive ages at the
    reconstructed position; empty cells count as zero. Returns the integral over the entire covered
    history and over each of ``windows`` (missing where fewer than 98% of the steps were covered or the
    window runs past the covered history), the delivery-weighted mean lag of the positive increments,
    and the oldest time integrated."""
    tf = np.rint(p["age_ma"].to_numpy()).astype(int)
    lon, lat, pid = p["lon"].to_numpy(), p["lat"].to_numpy(), p["pid"].to_numpy()
    res = {}
    acc = {}
    tmax_of = {n: int(g.ages.max()) for n, g, _ in series if g is not None}
    for name, g, var in series:
        tm = tmax_of[name] if g is not None else min(tmax_of[c] for c in var)
        acc[name] = dict(full=np.zeros(len(p)), lagsum=np.zeros(len(p)), possum=np.zeros(len(p)), cover=np.zeros(len(p), int),
                         win={T: np.zeros(len(p)) for T in windows}, tmax=tm)
    t_hi = max(a["tmax"] for a in acc.values())
    if tmax:
        t_hi = min(t_hi, int(tmax))
    cover_to = np.minimum(np.minimum(begin, t_hi), np.inf)
    t0 = _time.time()
    for t in range(int(tf.min()) + 1, t_hi + 1):
        rows = np.nonzero((tf < t) & (t <= cover_to))[0]
        if not len(rows):
            continue
        x, y = mantle.reconstruct_points(lon[rows], lat[rows], pid[rows], t)
        incs = {}
        for name, g, var in series:
            a = acc[name]
            if t > a["tmax"]:
                continue
            if g is None:
                inc = sum(incs[c] for c in var)
            else:
                inc = g.sample(t - 1, var, x, y) - g.sample(t, var, x, y)
            # a cell the grid leaves empty received nothing from that component: count it as zero,
            # so that a few empty cells along a path of 1,000 Myr do not discard the entire path
            pos = np.isfinite(x) & np.isfinite(y)
            inc = np.where(pos & ~np.isfinite(inc), 0.0, inc)
            incs[name] = inc
            good = np.isfinite(inc)
            r = rows[good]
            a["full"][r] += inc[good]
            a["lagsum"][r] += np.clip(inc[good], 0, None) * (t - tf[r])
            a["possum"][r] += np.clip(inc[good], 0, None)
            a["cover"][r] += 1
            for T in windows:
                s = r[t <= tf[r] + T]
                a["win"][T][s] += inc[good][t <= tf[r] + T]
        if progress and t % 50 == 0:
            print(f"  water t = {t} Ma ({len(rows)} points, {(_time.time() - t0) / 60:.1f} min)", flush=True)
    for name, a in acc.items():
        top = np.minimum(cover_to, a["tmax"])
        span = np.clip(top - tf, 0, None)
        complete = a["cover"] >= span * 0.98
        res[f"{name}_path_full"] = np.where(complete & (span > 0), a["full"], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            res[f"{name}_path_lag_myr"] = np.where(a["possum"] > 0, a["lagsum"] / a["possum"], np.nan)
        for T in windows:
            res[f"{name}_path_{T}myr"] = np.where(complete & (tf + T <= top), a["win"][T], np.nan)
    res["path_cover_to_ma"] = np.minimum(cover_to, t_hi)
    return pd.DataFrame(res)


class Trenches:
    """Subduction-zone points with their overriding plate, from the resolved topologies."""

    def __init__(self, model, spacing_deg=0.5):
        import pygplates
        self.pg, self.model, self.spacing = pygplates, model, np.radians(spacing_deg)

    def snapshot(self, t):
        """The resolved topologies at ``t``, a KD-tree of the subduction-zone points (tessellated every
        ``spacing_deg``) and the overriding plate of each point; tree and plates are None when no
        subduction zone is resolved."""
        from scipy.spatial import cKDTree
        snap = self.model.topological_snapshot(float(t))
        lon, lat, over = [], [], []
        for sec in snap.get_resolved_topological_sections():
            if sec.get_topological_section_feature().get_feature_type() != self.pg.FeatureType.gpml_subduction_zone:
                continue
            for seg in sec.get_shared_sub_segments():
                try:
                    r = seg.get_overriding_and_subducting_plates()
                except Exception:
                    r = None
                if not r:
                    continue
                op = r[0].get_feature().get_reconstruction_plate_id()
                g = seg.get_resolved_geometry()
                if g is None:
                    continue
                for pt in g.to_tessellated(self.spacing).get_points():
                    la, lo = pt.to_lat_lon()
                    lon.append(lo), lat.append(la), over.append(op)
        if not lon:
            return snap, None, None
        return snap, cKDTree(lonlat_to_xyz(np.array(lon), np.array(lat))), np.array(over)


def topo_pids(snap, x, y):
    """Plate id of the resolved boundary or network containing each reconstructed point (``x`` longitude,
    ``y`` latitude) in ``snap``; -1 outside every topology or for a missing position."""
    out = np.full(len(x), -1)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.any():
        locs = snap.get_point_locations([(float(b), float(a)) for a, b in zip(x[ok], y[ok])])
        vals = []
        for loc in locs:
            topo = (loc.located_in_resolved_boundary() or loc.located_in_resolved_network()) if loc else None
            vals.append(topo.get_feature().get_reconstruction_plate_id() if topo else -1)
        out[ok] = vals
    return out


def run_topo(p, mantle, begin, step, inboard, tmax=1800, progress=True, stable_km=500.0,
             sens_myr=SENS_EPISODE_MYR, after_myr=AFTER_MYR):
    """Time each point spent on the overriding plate near a trench before formation, every ``step`` Myr
    back to ``tmax`` or the entry of its block into the model, and whether that setting persisted after
    formation. At each step the point is reconstructed, its topological plate found, and its distance taken
    to the nearest trench point whose overriding plate is that plate. Per distance of ``inboard``: total
    time, longest episode, the time since the end of the most recent episode of at least MIN_EPISODE_MYR and
    since the start of the oldest, the censored lag and episode flag, the same totals for episodes at a
    stationary trench (distance range at most ``stable_km``), the time since the end of the most recent
    episode of at least each of ``sens_myr`` (with its flag), and the time for which the point stayed within
    that distance after formation, read forwards from the first step at or after formation over at most
    ``after_myr`` (review 2026-11-09: a lag of one step can mean that the episode ended just before
    formation or that it was still under way)."""
    tf = p["age_ma"].to_numpy(float)
    lon, lat, pid = p["lon"].to_numpy(), p["lat"].to_numpy(), p["pid"].to_numpy()
    # steps from the present back to tmax: a step older than formation belongs to the history before it, a
    # step at or younger than formation (back to after_myr later) shows whether the setting persisted
    times = np.arange(0.0, tmax + step / 2, step, dtype=float)
    hist = {D: np.zeros((len(p), len(times)), bool) for D in inboard}
    after = {D: np.zeros((len(p), len(times)), bool) for D in inboard}
    dist = np.full((len(p), len(times)), np.inf)          # distance to the point's own trench
    valid = np.zeros((len(p), len(times)), bool)          # steps before formation within the block's life
    valid_after = np.zeros((len(p), len(times)), bool)    # steps at or after formation, up to after_myr
    tr = Trenches(mantle)
    t0 = _time.time()
    for k, t in enumerate(times):
        before_rows = (tf < t) & (t <= begin)
        after_rows = (t <= tf) & (t >= tf - after_myr)
        rows = np.nonzero(before_rows | after_rows)[0]
        if not len(rows):
            continue
        valid[before_rows, k] = True
        valid_after[after_rows, k] = True
        snap, tree, over = tr.snapshot(t)
        if tree is None:
            continue
        x, y = mantle.reconstruct_points(lon[rows], lat[rows], pid[rows], t)
        tp = topo_pids(snap, x, y)
        ok = np.isfinite(x) & (tp >= 0)
        if not ok.any():
            continue
        q = lonlat_to_xyz(x[ok], y[ok])
        # nearest trench point whose overriding plate is the point's own plate
        rmax = max(max(inboard), SEARCH_KM)
        hits = tree.query_ball_point(q, 2 * np.sin(rmax / 6371.0 / 2))
        dmin = np.full(ok.sum(), np.inf)
        for i, h in enumerate(hits):
            if not h:
                continue
            h = np.asarray(h)
            mine = h[over[h] == tp[ok][i]]
            if len(mine):
                dmin[i] = chord_to_km(np.min(np.linalg.norm(tree.data[mine] - q[i], axis=1)))
        r_ok = rows[ok]
        dist[r_ok, k] = dmin
        is_before = before_rows[r_ok]
        for D in inboard:
            near = dmin <= D
            hist[D][r_ok[is_before], k] = near[is_before]
            after[D][r_ok[~is_before], k] = near[~is_before]
        if progress and k % 20 == 0:
            print(f"  topology t = {t:.0f} Ma ({len(rows)} points, {(_time.time() - t0) / 60:.1f} min)", flush=True)
    out = {}
    for D in inboard:
        h = hist[D]
        total = h.sum(1) * step
        longest = np.zeros(len(p))
        since = np.full(len(p), np.nan)
        first = np.full(len(p), np.nan)
        st_total, st_longest, st_since = np.zeros(len(p)), np.zeros(len(p)), np.full(len(p), np.nan)
        since_s = {L: np.full(len(p), np.nan) for L in sens_myr}
        after_d = np.full(len(p), np.nan)
        for i in range(len(p)):
            # persistence after formation: consecutive steps within D km, from the first step at or after
            # formation towards the present (zero if the point was not within D km at that step)
            ka = np.nonzero(valid_after[i])[0][::-1]
            if len(ka):
                seq = after[D][i, ka]
                after_d[i] = (len(seq) if seq.all() else int(np.argmin(seq))) * step
            row = h[i]
            if not row.any():
                continue
            # runs of consecutive True steps (times increase with k: older)
            edges = np.diff(np.concatenate([[0], row.astype(int), [0]]))
            starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1
            lens = (ends - starts + 1) * step
            longest[i] = lens.max()
            long_ = lens >= MIN_EPISODE_MYR
            if long_.any():
                since[i] = times[starts[long_].min()] - tf[i]      # youngest end of a long episode
                first[i] = times[ends[long_].max()] - tf[i]        # oldest start of a long episode
            for L in sens_myr:
                long_l = lens >= L
                if long_l.any():
                    since_s[L][i] = times[starts[long_l].min()] - tf[i]
            # an episode at a stationary trench: the distance from the point to its trench varies by
            # less than stable_km over the episode (the plates carry the point, so a trench fixed to
            # the continent keeps its distance; a retreating or advancing one does not)
            rng = np.array([np.ptp(dist[i, a:b + 1]) for a, b in zip(starts, ends)])
            stable = long_ & (rng <= stable_km)
            if stable.any():
                st_total[i] = lens[stable].sum()
                st_longest[i] = lens[stable].max()
                st_since[i] = times[starts[stable].min()] - tf[i]
        out[f"over_{D:g}km_myr"] = total
        out[f"over_{D:g}km_longest_myr"] = longest
        out[f"over_{D:g}km_since_myr"] = since
        out[f"over_{D:g}km_first_myr"] = first
        # the lag is undefined for a point that never had an episode of at least MIN_EPISODE_MYR, and
        # such points are most of some classes; the censored lag sets it to the length of the covered
        # history (the point has had no episode for at least that long), and the flag records whether
        # an episode exists, so the two can be compared without conditioning on having one
        out[f"over_{D:g}km_since_censored_myr"] = np.where(np.isfinite(since), since, valid.sum(1) * step)
        out[f"over_{D:g}km_episode"] = np.isfinite(since).astype(float)
        out[f"over_{D:g}km_stable_myr"] = st_total
        out[f"over_{D:g}km_stable_longest_myr"] = st_longest
        out[f"over_{D:g}km_stable_since_myr"] = st_since
        for L in sens_myr:
            out[f"over_{D:g}km_since_L{L:g}myr"] = since_s[L]
            out[f"over_{D:g}km_episode_L{L:g}myr"] = np.isfinite(since_s[L]).astype(float)
        out[f"over_{D:g}km_after_myr"] = after_d
    out["topo_cover_myr"] = valid.sum(1) * step
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--tag", default="full", help="features table to read the points from")
    ap.add_argument("--out-tag", help="tag of the output table (default: --tag)")
    ap.add_argument("--classes", nargs="*", default=list(CLASSES))
    ap.add_argument("--parts", nargs="*", default=["water", "topo"], choices=["water", "topo"])
    ap.add_argument("--windows", nargs="*", type=int, default=list(WINDOWS))
    ap.add_argument("--inboard-km", nargs="*", type=float, default=list(INBOARD_KM))
    ap.add_argument("--stable-km", type=float, default=500.0,
                    help="largest change in point-trench distance over a stationary-trench episode")
    ap.add_argument("--step", type=float, default=5.0, help="topology step (Myr)")
    ap.add_argument("--tmax", type=float, default=None, help="oldest time integrated (pilot runs)")
    ap.add_argument("--max-points", type=int, help="random subset of points (pilot runs)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    from rifts.models import load_models
    mantle, _ = load_models(cfg)
    p = point_list(cfg, a.tag, a.classes, a.max_points)
    begin = block_begin(mantle, p)
    print(f"{len(p)} points ({p['dclass'].isin(a.classes).sum()} deposits) of {', '.join(a.classes)}")
    out = p[["pt_id"]].copy()
    if "water" in a.parts:
        series = water_series(cfg)
        print("water series: " + ", ".join(f"{n} ({int(g.ages.min())}-{int(g.ages.max())} Ma)" if g is not None
                                            else f"{n} (sum)" for n, g, _ in series))
        out = pd.concat([out, run_water(p, mantle, series, a.windows, begin, a.tmax)], axis=1)
    if "topo" in a.parts:
        out = pd.concat([out, run_topo(p, mantle, begin, a.step, a.inboard_km, a.tmax or 1800,
                                       stable_km=a.stable_km)], axis=1)
    target = cfg["paths"]["derived"] / f"primer_history_{a.out_tag or a.tag}"
    # a run of one part keeps the columns of the other part from the existing table
    if set(a.parts) != {"water", "topo"}:
        try:
            old = read_table(target)
            keep = [c for c in old.columns if c == "pt_id" or c not in out.columns]
            out = old[keep].merge(out, on="pt_id", how="outer")
        except FileNotFoundError:
            pass
    print(f"wrote {write_table(out, target)}")


if __name__ == "__main__":
    main()
