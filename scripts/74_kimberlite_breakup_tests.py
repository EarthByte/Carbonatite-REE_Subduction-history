#!/usr/bin/env python
"""Kimberlites and continental breakup: does the timing of breakup matter, and what else could
explain where kimberlites erupt?

scripts/71 found that kimberlites erupted closer than continental crust of the same age to
margins whose rifting ended 20-70 Myr before eruption, and no closer to failed rifts. This step
asks five further questions of the same points, all at the positions of eruption (mantle frame)
unless stated, with the same district weights, age balance and estimator (the probability that a
kimberlite lies farther than a reference point; below 0.5, closer):

  timing       Is it the age of the margins or only their position? The margin outlines are held
               fixed and their ages reassigned at random (onset and end together) among margins
               whose rifting ended in the past 550 Myr ("margins shuffled"), and among all rifts
               of that age range of every fate ("all rifts shuffled"); and all margin ages are
               shifted by a constant from -100 to +100 Myr. The observed estimate is set against
               the distribution of the reassigned ones, for continental crust and for crust inside
               cratons.
  power        Would the failed-rift test have seen an attraction as strong as that of margins?
               Synthetic kimberlite sets are drawn from the reference points, favouring points
               with a rift of the window within 1,000 km by the factor that reproduces the
               kimberlite share for margins (and, as a check, by a distance kernel exp(-d/L) with L
               reproducing the margin estimate); the same favouring is applied to failed rifts. The
               share of synthetic sets whose estimate falls below the 2.5th percentile of
               unfavoured sets is the power of the test.
  extension    Rifts active at eruption, rifts of any age that had begun before eruption, and
               failed rifts restricted to grabens (rift_subtype RiftGraben), by window.
  alternatives Large igneous provinces (Johansson et al. 2018 polygons; seamount chains, oceanic
               ridges, plateaus, rises, volcanic passive margins, seaward-dipping reflectors and
               breakup basalts left out, so that the provinces are not the margins again) emplaced a
               given interval before eruption, reconstructed with their plates; signed distance
               from the edge of the large low-shear-velocity provinces (agreed by at least three
               of four modern tomographic families, rifts/llsvp.py; held fixed in the mantle frame;
               negative inside) and its absolute value;
               plate speed in the mantle frame at eruption. The margin estimate is repeated within
               tertiles of each, and the LIP estimate within tertiles of margin distance.
  migration    Within provinces, the reconstructed distance of each kimberlite to the nearest
               margin whose rifting ended 0-150 Myr before eruption against the time since that
               margin's rifting ended (and since it began): the slope in km per Myr, weighted by
               district, with intervals from resampling districts.

Output: ONE long table, data/derived/kimberlite_breakup_tests_<tag>.csv, written through a
.partial name.

    python scripts/74_kimberlite_breakup_tests.py --tag kimb23 --perm 100 --boot 200 --synth 200   # pilot
    python scripts/74_kimberlite_breakup_tests.py --tag kimb23
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import EARTH_RADIUS_KM, load_config, lonlat_to_xyz  # noqa: E402
from rifts.context import rift_table, RIFT_TYPES, read_outlines  # noqa: E402
from rifts.frames import deposit_table, deposit_rows, load_screen_frame, reference_rows, within_age  # noqa: E402
from rifts.riftlag import rift_distances  # noqa: E402

_spec = importlib.util.spec_from_file_location("fc", Path(__file__).with_name("71_kimberlite_fate_contrast.py"))
fc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fc)
wauc, Side, nearest, draw_idx, strat_auc, wname = fc.wauc, fc.Side, fc.nearest, fc.draw_idx, fc.strat_auc, fc.wname

MAX_KM = 3000.0
REACH_KM = 1000.0
WINDOWS = fc.WINDOWS
MAIN = ((20.0, 40.0), (40.0, 70.0))
LIP_WINDOWS = ((0.0, 20.0), (20.0, 40.0), (40.0, 70.0), (70.0, 120.0), (0.0, 70.0))
SHUFFLE_MAX_END = 550.0
MIGRATION_KM = 3000.0
PARTS = {"timing": ("timing", "timing offset", "timing lag"), "power": ("power",),
         "extension": ("extension",), "alternatives": ("alternatives", "stratified", "nearest lips"),
         "migration": ("migration", "migration district", "breakup events")}
OFFSETS = np.arange(-100, 101, 5)
LIP_EXCLUDE = ("METER", "SEAMOUNT", "RIDGE", "PLATEAU", "RISE", "VOLCANIC PASSIVE", "BANK", "GUYOT", "ISLAND",
               "HOTSPOT", "TRACK", "CHAIN", "ABYSSAL", "MARGIN", "HAWAII", "KERGUELEN", "MANIHIKI", "ONTONG",
               "REFLECTOR", "BREAKUP", "BAHAMAS", "BASIN, ", "CAMEROON", "ARCHIPELAGO", "SAMOA", "CHAGOS",
               "OFFSHORE BRAZIL", "SHALLOW INTRUSIONS", "EXPLORA")
# polygons of one province carry several names in the compilation; a province is one event, so
# its polygons share one age when ages are reassigned and are reported together
LIP_FAMILY = (("SIBERIAN", "Siberian Traps"), ("OSLO", "Oslo"), ("EMEISHAN", "Emeishan"),
              ("BRITISH TERTIARY", "North Atlantic"), ("UNITED KINGDOM", "North Atlantic"),
              ("N. ATLANTIC VOLCANIC", "North Atlantic"), ("E-GREEN", "North Atlantic"),
              ("BAFFIN BAY", "North Atlantic"), ("PANAMA", "Caribbean"), ("GUAYAPE", "Caribbean"),
              ("CARIBBEAN", "Caribbean"), ("KAROO", "Karoo-Ferrar"), ("TRANSANTARCTIC", "Karoo-Ferrar"),
              ("DUFEK", "Karoo-Ferrar"), ("THERON", "Karoo-Ferrar"), ("WHICHAWAY", "Karoo-Ferrar"),
              ("DRONNING MAUD", "Karoo-Ferrar"), ("HIGH ARCTIC", "High Arctic"), ("ETHIOPIAN", "Afro-Arabian"),
              ("YEMEN", "Afro-Arabian"), ("PARANA", "Parana-Etendeka"), ("ETENDEKA", "Parana-Etendeka"),
              ("LOVEJOY", "Columbia River"), ("COLUMBIA RIVER", "Columbia River"), ("YAKUTSK", "Yakutsk-Vilyui"),
              ("VILNBASIN", "Yakutsk-Vilyui"), ("EAST EUROPEAN PLATFORM", "Kola-Dnieper"),
              ("CENTRAL ATLANTIC", "Central Atlantic"), ("KALKARINDJI", "Kalkarindji"), ("SEYCHEL", "Seychelles"))


def lip_family(label, age):
    u = label.upper()
    for key, fam in LIP_FAMILY:
        if key in u:
            return fam
    return f"{label.strip()[:48]} ({age:g} Ma)"
PROVINCES = {   # present-day boxes: lon_min, lon_max, lat_min, lat_max
    "southern Africa (Kaapvaal-Zimbabwe)": (14, 36, -35, -15),
    "Angola-Congo": (10, 32, -15, 5),
    "West Africa": (-20, 10, 0, 20),
    "Brazil": (-70, -34, -35, 5),
    "eastern North America": (-100, -55, 25, 52),
    "northern North America (Slave, Superior)": (-125, -60, 52, 75),
    "Siberia": (95, 130, 55, 75),
    "India": (70, 90, 10, 30),
    "Australia": (110, 155, -40, -10),
    "Baltica": (20, 50, 55, 70),
}


def summarise(b, null=None):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    if not len(b):
        return dict(lo=np.nan, hi=np.nan)
    return dict(lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)))


def reach(x, w, km=REACH_KM):
    ok = np.isfinite(x)
    return float(np.sum(w[ok] * (x[ok] <= km)) / max(np.sum(w[ok]), 1e-12))


# ----------------------------------------------------------------------- data set-up
def load(cfg, tag, max_age):
    """The kimberlite frame of ``tag`` up to ``max_age``, with the deep slab water (subducted minus slab
    outflux) where it was sampled and the age bounds and subtype of every record; the dated rift
    outlines with fate and type; the large igneous provinces younger than max_age + 150 Myr; and the
    LLSVP provinces."""
    der = cfg["paths"]["derived"]
    root = Path(__file__).resolve().parents[1]
    frame, _, _ = load_screen_frame(cfg, tag, verbose=False)
    frame = within_age(frame, max_age)
    # slab water carried past 125 km (subducted minus slab outflux; the water workflow keeps it in the
    # slab with a thermal-parameter lookup at 4.4 GPa): the cold-slab water that can reach the transition
    # zone (Shirey et al. 2021, 2024). Present when scripts/03 has run for the tag.
    for fr in ("fixed", "track"):
        for T in (100, 200, 400):
            s_, o_ = f"h2o_mean_subducted_water_{fr}_{T}myr", f"h2o_mean_slab_outflux_{fr}_{T}myr"
            if s_ in frame and o_ in frame:
                frame[f"h2o_deep_{fr}_{T}myr"] = frame[s_] - frame[o_]
    dep = deposit_table(der, tag)
    frame = frame.merge(dep[["dep_id", "subtype", "age_min_ma", "age_max_ma"]].rename(columns={"dep_id": "pt_id"}),
                        on="pt_id", how="left", suffixes=("", "_dep"))
    rifts = read_outlines(der / "rift_outlines.gpkg")
    rifts = rifts[rifts["GPGIM_TYPE"].isin(RIFT_TYPES)]
    rifts = rifts[np.isfinite(rifts["onset_ma"]) & np.isfinite(rifts["end_ma"])]
    st = pd.read_csv(rift_table(der, "rift_settings", tag)).drop_duplicates("outline_id")
    rifts = rifts.merge(st[["outline_id", "fate", "feature_type", "rift_subtype"]], left_on="RIFTID",
                        right_on="outline_id", how="left").reset_index(drop=True)
    lips = read_lips(root / "data" / "external" / "Johansson2018_LIPs_v2" / "Johansson_etal_2018_EarthByte_LIPs_v2.shp",
                     max_age + 150)
    llsvp = read_llsvp(root)
    return frame, rifts, lips, llsvp


def read_lips(path, max_age):
    """Johansson et al. (2018) LIP polygons no older than ``max_age``, without the oceanic plateaus,
    seamounts, ridges, hotspot tracks and other features named in LIP_EXCLUDE and without unnamed
    polygons. Each is dated by FROMAGE and assigned a LIP family."""
    import geopandas as gpd
    g = gpd.read_file(path)
    g = g[(g["FROMAGE"] <= max_age) & g.geometry.notna()]
    # polygons without a NAME carry their name in DESCR
    label = g["NAME"].where(g["NAME"].fillna("").str.strip() != "", g["DESCR"]).fillna("").astype(str)
    g = g[~label.str.upper().apply(lambda v: any(k in v for k in LIP_EXCLUDE))].copy()
    g = g[label.loc[g.index].str.strip() != ""]
    g["onset_ma"] = g["end_ma"] = g["FROMAGE"].astype(float)
    g["lip_name"] = label.loc[g.index]
    g["lip"] = [lip_family(lb, a) for lb, a in zip(g["lip_name"], g["FROMAGE"])]
    return g.reset_index(drop=True)


def read_llsvp(root, votes=3):
    """LLSVPs agreed by at least ``votes`` of four modern tomographic families (rifts/llsvp.py)."""
    from rifts.llsvp import Provinces
    return Provinces(root, votes)


def llsvp_distance(prov, lon, lat):
    """Signed great-circle distance (km) from each point to the nearest LLSVP edge; negative inside."""
    return prov.signed_distance(np.asarray(lon, float), np.asarray(lat, float))


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--tag", default="kimb23")
    ap.add_argument("--max-age", type=float, default=400.0)
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--synth", type=int, default=1000)
    ap.add_argument("--parts", nargs="*", default=list(PARTS), choices=list(PARTS),
                    help="run only these parts; the rows of the other parts are kept from the existing table")
    a = ap.parse_args()
    cfg = load_config(a.config)
    der = cfg["paths"]["derived"]

    frame, rifts, lips, llsvp = load(cfg, a.tag, a.max_age)
    from rifts.models import load_models
    mantle, _ = load_models(cfg)
    onset0, end0 = rifts["onset_ma"].to_numpy(float), rifts["end_ma"].to_numpy(float)
    fate = rifts["fate"].to_numpy()
    is_m, is_f = fate == "margin", fate == "failed rift"
    is_graben = is_f & rifts["rift_subtype"].eq("RiftGraben").to_numpy()
    any_fate = np.isin(fate, ["margin", "failed rift", "active"])
    print(f"{len(rifts)} rifts: {is_m.sum()} margins, {is_f.sum()} failed rifts ({is_graben.sum()} grabens); "
          f"{len(lips)} LIP polygons in {lips['lip'].nunique()} provinces")

    sides, Dlip = {}, {}
    for cls in ("KIMB", "PCU"):
        d, r = deposit_rows(frame, cls), reference_rows(frame, cls)
        both = pd.concat([d, r], ignore_index=True)
        pts = pd.DataFrame({"pt_id": both["pt_id"].to_numpy(), "lon": both["lon0"].to_numpy(float),
                            "lat": both["lat0"].to_numpy(float), "mlon": both["mlon"].to_numpy(float),
                            "mlat": both["mlat"].to_numpy(float), "age_ma": both["age_ma"].to_numpy(float)})
        D = rift_distances(pts, rifts, reconstructed=True, model=mantle)
        L = rift_distances(pts, lips, reconstructed=True, model=mantle)
        nd = len(d)
        sides[cls] = Side(d, r, D[:nd], D[nd:])
        Dlip[cls] = (L[:nd], L[nd:])
        s = sides[cls]
        s.llsvp_d = llsvp_distance(llsvp, s.d["mlon"].to_numpy(float), s.d["mlat"].to_numpy(float))
        s.llsvp_r = llsvp_distance(llsvp, s.r["mlon"].to_numpy(float), s.r["mlat"].to_numpy(float))
        print(f"{cls}: {nd} localities in {d['cluster_id'].nunique()} districts, {len(r)} reference points", flush=True)

    rows = []
    K = sides["KIMB"]
    crat_d = (K.d["craton_dist_km"] < 0).to_numpy()
    crat_r = (K.r["craton_dist_km"] < 0).to_numpy()
    KC = Side(K.d[crat_d], K.r[crat_r], K.Dd[crat_d], K.Dr[crat_r])

    def vals(s, sel, lo, hi, on=onset0, en=end0, active=False, D=None, cap=MAX_KM):
        Dd, Dr = (s.Dd, s.Dr) if D is None else D
        xd = nearest(Dd, s.age_d, on, en, sel, lo, hi, cap, active)
        xr = nearest(Dr, s.age_r, on, en, sel, lo, hi, cap, active)
        return xd, xr

    def begun(s, sel, D=None, cap=MAX_KM):
        """Nearest rift of ``sel`` whose rifting had begun by the point's age (any end)."""
        out = []
        for Dm, ages in (((s.Dd if D is None else D[0]), s.age_d), ((s.Dr if D is None else D[1]), s.age_r)):
            m = (onset0[None, :] >= ages[:, None]) & sel[None, :] & np.isfinite(Dm) & (Dm <= cap)
            dm = np.where(m, Dm, np.inf).min(axis=1)
            has = np.isfinite(dm)
            out.append((np.where(has, dm, cap), None, has))
        return out

    def boot_est(s, x, y, n, seed, cells=None):
        """Bootstrap over districts and reference points: ``n`` resampled AUCs (within cells when ``cells``
        is given) and, for every resample, the share of kimberlites and of reference points within
        REACH_KM."""
        rng = np.random.default_rng(seed)
        b, rd, rr = [], [], []
        for _ in range(n):
            idx = draw_idx(s.groups, len(s.r), rng)
            if cells is None:
                b.append(wauc(x[idx[0]], s.wd[idx[0]], y[idx[1]], s.wr[idx[1]]))
            else:
                b.append(strat_auc(x[idx[0]], s.wd[idx[0]], cells[0][idx[0]], y[idx[1]], s.wr[idx[1]], cells[1][idx[1]]))
            rd.append(reach(x[idx[0]], s.wd[idx[0]]))
            rr.append(reach(y[idx[1]], s.wr[idx[1]]))
        return np.asarray(b), np.asarray(rd), np.asarray(rr)

    def comparison(kind, quantity, s, xd, xr, cls="KIMB", seed=1, cells=None, **extra):
        """Add one result row: the observed AUC of the kimberlite distances ``xd`` against the reference
        distances ``xr`` (both nearest() tuples), its bootstrap interval, the shares within REACH_KM,
        the counts, and the median distance of kimberlites with a qualifying feature. Returns the
        observed AUC."""
        x, y = xd[0], xr[0]
        obs = wauc(x, s.wd, y, s.wr) if cells is None else strat_auc(x, s.wd, cells[0], y, s.wr, cells[1])
        b, rd, rr = boot_est(s, x, y, a.boot, seed, cells)
        rows.append(dict(kind=kind, dclass=cls, quantity=quantity, auc=obs, **summarise(b),
                         reach_dep=reach(x, s.wd), reach_dep_lo=summarise(rd)["lo"], reach_dep_hi=summarise(rd)["hi"],
                         reach_ref=reach(y, s.wr), n_districts=int(pd.Series(s.cluster).nunique()), n_ref=len(s.r),
                         median_dep_km=float(np.nanmedian(np.where(xd[2], x, np.nan))) if xd[2] is not None else np.nan,
                         **extra))
        return obs

    # ------------------------------------------------------------------- 1 timing
    if "timing" in a.parts:
        pool_m = is_m & (end0 <= SHUFFLE_MAX_END) & (end0 > 0)
        pool_all = any_fate & (fate != "active") & (end0 <= SHUFFLE_MAX_END) & (end0 > 0)
        for bname, s in (("continental crust", K), ("crust inside cratons", KC)):
            for lo, hi in MAIN:
                w = wname(lo, hi)
                xd, xr = vals(s, is_m, lo, hi)
                obs = wauc(xd[0], s.wd, xr[0], s.wr)
                obs_reach = reach(xd[0], s.wd) - reach(xr[0], s.wr)
                for pname, pool in (("margins shuffled", pool_m), ("all rifts shuffled", pool_all)):
                    rng = np.random.default_rng(74)
                    ids = np.nonzero(pool)[0]
                    null, null_reach = [], []
                    for _ in range(a.perm):
                        p = rng.permutation(ids)
                        on, en = onset0.copy(), end0.copy()
                        on[ids], en[ids] = onset0[p], end0[p]
                        pd_, pr_ = vals(s, is_m, lo, hi, on, en)
                        null.append(wauc(pd_[0], s.wd, pr_[0], s.wr))
                        null_reach.append(reach(pd_[0], s.wd) - reach(pr_[0], s.wr))
                    null, null_reach = np.asarray(null), np.asarray(null_reach)
                    rows.append(dict(kind="timing", dclass="KIMB", background=bname, window=w, quantity=pname, auc=obs,
                                     null_median=float(np.nanmedian(null)), **summarise(null),
                                     p=float((1 + np.sum(null <= obs)) / (1 + np.isfinite(null).sum())),
                                     reach_excess=obs_reach, null_reach_median=float(np.nanmedian(null_reach)),
                                     null_reach_lo=float(np.nanpercentile(null_reach, 2.5)),
                                     null_reach_hi=float(np.nanpercentile(null_reach, 97.5)),
                                     p_reach=float((1 + np.sum(null_reach >= obs_reach)) / (1 + len(null_reach))),
                                     n_perm=a.perm, n_pool=int(len(ids))))
                for off in OFFSETS:
                    en = np.where(is_m, np.maximum(end0 + off, 0.0), end0)
                    on = np.where(is_m, np.maximum(onset0 + off, en), onset0)
                    pd_, pr_ = vals(s, is_m, lo, hi, on, en)
                    rows.append(dict(kind="timing offset", dclass="KIMB", background=bname, window=w, offset_myr=int(off),
                                     auc=wauc(pd_[0], s.wd, pr_[0], s.wr),
                                     reach_dep=reach(pd_[0], s.wd), reach_ref=reach(pr_[0], s.wr)))
                print(f"  timing {bname} {w}: observed {obs:.3f}", flush=True)
        # the same for large igneous provinces: ages reassigned among provinces (all polygons of one
        # province move together), outlines fixed
        lip_on0 = lips["onset_ma"].to_numpy(float)
        lip_code, lip_names = pd.factorize(lips["lip"])
        lip_age = pd.Series(lip_on0).groupby(lip_code).median().to_numpy()
        for lo, hi in ((0.0, 20.0), (0.0, 70.0), (20.0, 40.0), (40.0, 70.0)):
            w = wname(lo, hi)
            xd, xr = vals(K, np.ones(len(lips), bool), lo, hi, lip_on0, lip_on0, D=Dlip["KIMB"])
            obs = wauc(xd[0], K.wd, xr[0], K.wr)
            rng = np.random.default_rng(75)
            null = []
            for _ in range(a.perm):
                ages = rng.permutation(lip_age)[lip_code]
                pd_, pr_ = vals(K, np.ones(len(lips), bool), lo, hi, ages, ages, D=Dlip["KIMB"])
                null.append(wauc(pd_[0], K.wd, pr_[0], K.wr))
            null = np.asarray(null)
            rows.append(dict(kind="timing", dclass="KIMB", background="continental crust", window=w,
                             quantity="large igneous provinces shuffled", auc=obs, null_median=float(np.nanmedian(null)),
                             **summarise(null), p=float((1 + np.sum(null <= obs)) / (1 + np.isfinite(null).sum())),
                             n_perm=a.perm, n_pool=int(len(lip_age))))
        # a continuous version: the estimate for margins whose rifting ended lag..lag+30 Myr before
        for lag in range(0, 151, 5):
            xd, xr = vals(K, is_m, float(lag), float(lag + 30))
            b, _, _ = boot_est(K, xd[0], xr[0], min(a.boot, 300), 5)
            rows.append(dict(kind="timing lag", dclass="KIMB", window=wname(lag, lag + 30), quantity="margins",
                             auc=wauc(xd[0], K.wd, xr[0], K.wr), **summarise(b),
                             reach_dep=reach(xd[0], K.wd), reach_ref=reach(xr[0], K.wr)))

    # ------------------------------------------------------------------- 2 power
    if "power" in a.parts:
        rng = np.random.default_rng(7)
        n_syn = int(pd.Series(K.cluster).nunique())
        base_p = K.wr / K.wr.sum()

        def synth_auc(y, prob, n):
            out = []
            for _ in range(n):
                pick = rng.choice(len(y), n_syn, replace=True, p=prob)
                out.append(wauc(y[pick], np.ones(n_syn), y, K.wr))
            return np.asarray(out)

        for lo, hi in MAIN:
            w = wname(lo, hi)
            (xm, _, _), (ym, _, hm) = vals(K, is_m, lo, hi)
            obs_share = reach(xm, K.wd)
            near_m = ym <= REACH_KM
            p_near = float(np.sum(base_p[near_m]))
            # favouring factor k: k*p/(k*p + 1 - p) = observed share
            k = obs_share * (1 - p_near) / max(p_near * (1 - obs_share), 1e-9)
            # kernel length reproducing the margin estimate
            obs_m = wauc(xm, K.wd, ym, K.wr)
            best = None
            for Lk in (100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000):
                pr = base_p * np.exp(-ym / Lk)
                pr /= pr.sum()
                m_ = float(np.median(synth_auc(ym, pr, 60)))
                if best is None or abs(m_ - obs_m) < abs(best[1] - obs_m):
                    best = (Lk, m_)
            Lk = best[0]
            null_m = synth_auc(ym, base_p, a.synth)
            for fname, sel in (("margins", is_m), ("failed rifts", is_f), ("failed grabens", is_graben)):
                (xf, _, _), (yf, _, _) = vals(K, sel, lo, hi)
                null = null_m if fname == "margins" else synth_auc(yf, base_p, a.synth)
                thr = np.nanpercentile(null, 2.5)
                near = yf <= REACH_KM
                pr_r = base_p * np.where(near, k, 1.0)
                pr_r /= pr_r.sum()
                pr_k = base_p * np.exp(-yf / Lk)
                pr_k /= pr_k.sum()
                for model, pr in (("reach factor", pr_r), ("distance kernel", pr_k)):
                    sa = synth_auc(yf, pr, a.synth)
                    rows.append(dict(kind="power", dclass="KIMB", window=w, quantity=fname, treatment=model,
                                     auc=float(np.nanmedian(sa)), **summarise(sa), power=float(np.mean(sa < thr)),
                                     null_median=float(np.nanmedian(null)), null_lo=float(thr),
                                     observed=wauc(xf, K.wd, yf, K.wr), reach_ref=float(np.sum(base_p[near])),
                                     synthetic_reach=float(np.sum(pr[near])), factor=k, kernel_km=Lk))
            print(f"  power {w}: factor {k:.2f}, kernel {Lk} km", flush=True)

    # --------------------------------------------------------------- 3 extension
    if "extension" in a.parts:
        for cls in ("KIMB", "PCU"):
            s = sides[cls]
            comparison("extension", "all rifts active at eruption", s, *vals(s, any_fate, 0, 0, active=True), cls=cls)
            comparison("extension", "rifts that became margins, active at eruption", s, *vals(s, is_m, 0, 0, active=True), cls=cls)
            comparison("extension", "rifts that failed, active at eruption", s, *vals(s, is_f, 0, 0, active=True), cls=cls)
            comparison("extension", "rifts of any age begun before eruption", s, *begun(s, any_fate), cls=cls)
            comparison("extension", "failed rifts of any age begun before eruption", s, *begun(s, is_f), cls=cls)
            comparison("extension", "failed grabens of any age begun before eruption", s, *begun(s, is_graben), cls=cls)
            for lo, hi in WINDOWS:
                comparison("extension", "failed grabens", s, *vals(s, is_graben, lo, hi), cls=cls, window=wname(lo, hi))
                comparison("extension", "failed rifts", s, *vals(s, is_f, lo, hi), cls=cls, window=wname(lo, hi))
                comparison("extension", "margins", s, *vals(s, is_m, lo, hi), cls=cls, window=wname(lo, hi))
            print(f"  extension {cls} done", flush=True)
        for lo, hi in MAIN:
            comparison("extension", "failed grabens", KC, *vals(KC, is_graben, lo, hi), window=wname(lo, hi),
                       background="crust inside cratons")

    # ------------------------------------------------------------- 4 alternatives
    if "alternatives" in a.parts:
        lip_on = lips["onset_ma"].to_numpy(float)
        lip_all = np.ones(len(lips), bool)
        for cls in ("KIMB", "PCU"):
            s = sides[cls]
            for lo, hi in LIP_WINDOWS:
                comparison("alternatives", "large igneous provinces", s,
                           *vals(s, lip_all, lo, hi, lip_on, lip_on, D=Dlip[cls]), cls=cls, window=wname(lo, hi))
            for q, xd, xr in (("signed distance from LLSVP edge", s.llsvp_d, s.llsvp_r),
                              ("distance from LLSVP edge", np.abs(s.llsvp_d), np.abs(s.llsvp_r)),
                              ("plate speed", s.d["plate_speed_cm_yr"].to_numpy(float), s.r["plate_speed_cm_yr"].to_numpy(float)),
                              ("trench distance", s.d["sz_dist_km"].to_numpy(float), s.r["sz_dist_km"].to_numpy(float))):
                comparison("alternatives", q, s, (xd, None, None), (xr, None, None), cls=cls,
                           median_ref=float(np.nanmedian(xr)), median_dep=float(np.nanmedian(xd)))
            print(f"  alternatives {cls} done", flush=True)
        for lo, hi in LIP_WINDOWS:
            comparison("alternatives", "large igneous provinces", KC,
                       *vals(KC, lip_all, lo, hi, lip_on, lip_on, D=(Dlip["KIMB"][0][crat_d], Dlip["KIMB"][1][crat_r])),
                       window=wname(lo, hi), background="crust inside cratons")
        # the provinces nearest to kimberlite districts at eruption, 0-20 and 0-70 Myr before
        for lo, hi in ((0.0, 20.0), (0.0, 70.0)):
            (x, i, h), _ = vals(K, lip_all, lo, hi, lip_on, lip_on, D=Dlip["KIMB"])
            near = h & (x <= REACH_KM)
            dd = pd.DataFrame({"cluster": K.cluster[near], "lip": lips["lip"].to_numpy()[i[near]],
                               "lip_age": lip_on[i[near]], "w": K.wd[near], "age": K.age_d[near]})
            for name, g in dd.groupby("lip"):
                rows.append(dict(kind="nearest lips", dclass="KIMB", window=wname(lo, hi), quantity=name,
                                 auc=float(g["lip_age"].iloc[0]), n_districts=int(g["cluster"].nunique()),
                                 share=float(g["w"].sum() / K.wd.sum()), age_min_ma=float(g["age"].min()),
                                 age_max_ma=float(g["age"].max())))
        # inside LLSVP share
        rows.append(dict(kind="alternatives", dclass="KIMB", quantity="share above an LLSVP",
                         reach_dep=float(np.sum(K.wd * (K.llsvp_d < 0)) / K.wd.sum()),
                         reach_ref=float(np.sum(K.wr * (K.llsvp_r < 0)) / K.wr.sum())))
        # stratified: margins within tertiles of each alternative, and LIPs within tertiles of margin distance
        from scipy.stats import rankdata  # noqa: F401

        def tertile_cells(a_d, a_r):
            edges = np.nanquantile(np.r_[a_d, a_r], [1 / 3, 2 / 3])
            return np.searchsorted(edges, a_d), np.searchsorted(edges, a_r)

        (ld, _, _), (lr, _, _) = vals(K, lip_all, 0, 70, lip_on, lip_on, D=Dlip["KIMB"])
        strata = {"LIP (0-70 Myr) distance": (ld, lr),
                  "distance from LLSVP edge": (np.abs(K.llsvp_d), np.abs(K.llsvp_r)),
                  "plate speed": (K.d["plate_speed_cm_yr"].to_numpy(float), K.r["plate_speed_cm_yr"].to_numpy(float))}
        for lo, hi in MAIN:
            w = wname(lo, hi)
            xd, xr = vals(K, is_m, lo, hi)
            for sname, (ad, ar) in strata.items():
                comparison("stratified", "margins", K, xd, xr, cells=tertile_cells(ad, ar), window=w, background=sname)
            md, mr = xd[0], xr[0]
            for lw in ((0.0, 70.0), (20.0, 40.0), (40.0, 70.0)):
                comparison("stratified", "large igneous provinces", K,
                           *vals(K, lip_all, lw[0], lw[1], lip_on, lip_on, D=Dlip["KIMB"]),
                           cells=tertile_cells(md, mr), window=wname(*lw), background=f"margin ({w} Myr) distance")
        print("  stratified done", flush=True)

    # --------------------------------------------------------------- 5 migration
    if "migration" in a.parts:
        lag_end = end0[None, :] - K.age_d[:, None]
        m = is_m[None, :] & (lag_end > 0) & (lag_end <= 150) & np.isfinite(K.Dd) & (K.Dd <= MIGRATION_KM)
        dm = np.where(m, K.Dd, np.inf)
        j = np.argmin(dm, axis=1)
        dist = dm[np.arange(len(j)), j]
        ok = np.isfinite(dist)
        delay_end = np.where(ok, end0[j] - K.age_d, np.nan)
        delay_on = np.where(ok, onset0[j] - K.age_d, np.nan)
        lon0, lat0 = K.d["lon0"].to_numpy(float), K.d["lat0"].to_numpy(float)
        prov = np.full(len(K.d), "", object)
        for pname, (x0, x1, y0, y1) in PROVINCES.items():
            prov[(lon0 >= x0) & (lon0 <= x1) & (lat0 >= y0) & (lat0 <= y1) & (prov == "")] = pname
        rifts_name = rifts["RIFTNAME"].astype(str).to_numpy()

        def wslope(x, y, w):
            ok_ = np.isfinite(x) & np.isfinite(y) & (w > 0)
            x, y, w = x[ok_], y[ok_], w[ok_]
            if len(x) < 3 or np.ptp(x) == 0:
                return np.nan
            mx, my = np.average(x, weights=w), np.average(y, weights=w)
            return float(np.sum(w * (x - mx) * (y - my)) / np.sum(w * (x - mx) ** 2))

        for pname in list(PROVINCES) + ["all provinces"]:
            sel = (prov == pname) if pname != "all provinces" else (prov != "")
            sel &= ok
            if (pd.Series(K.cluster[sel]).nunique() < 4):
                continue
            groups = list(pd.Series(np.nonzero(sel)[0]).groupby(K.cluster[sel]).apply(lambda v: v.to_numpy()))
            for dname, delay in (("since rifting ended", delay_end), ("since rifting began", delay_on)):
                obs = wslope(delay[sel], dist[sel], K.wd[sel])
                rng = np.random.default_rng(3)
                b = []
                for _ in range(a.boot):
                    idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
                    b.append(wslope(delay[idx], dist[idx], K.wd[idx]))
                from scipy.stats import spearmanr
                rho = spearmanr(delay[sel], dist[sel]).correlation if sel.sum() > 3 else np.nan
                nearest_m = pd.Series(rifts_name[j[sel]]).value_counts().head(3)
                rows.append(dict(kind="migration", dclass="KIMB", quantity=pname, treatment=dname, auc=obs, **summarise(b),
                                 rho=rho, n_districts=int(pd.Series(K.cluster[sel]).nunique()), n_localities=int(sel.sum()),
                                 delay_min=float(np.nanmin(delay[sel])), delay_max=float(np.nanmax(delay[sel])),
                                 median_dep_km=float(np.nanmedian(dist[sel])), age_min_ma=float(K.age_d[sel].min()),
                                 age_max_ma=float(K.age_d[sel].max()),
                                 rift_eruption="; ".join(f"{k} ({v})" for k, v in nearest_m.items())))
        # per-district points for a figure
        dd = pd.DataFrame({"cluster": K.cluster, "province": prov, "delay_end": delay_end, "delay_on": delay_on,
                           "dist": dist, "age": K.age_d, "w": K.wd, "rift": np.where(ok, rifts_name[j], "")})
        g = dd[dd["province"] != ""].groupby("cluster").agg(province=("province", "first"), delay_end=("delay_end", "median"),
                                                            delay_on=("delay_on", "median"), dist=("dist", "median"),
                                                            age=("age", "median"), w=("w", "sum"),
                                                            rift=("rift", lambda v: pd.Series(v).mode().iloc[0]))
        for cid, r in g.iterrows():
            rows.append(dict(kind="migration district", dclass="KIMB", district=str(cid), quantity=r["province"],
                             delay_end=r["delay_end"], delay_on=r["delay_on"], median_dep_km=r["dist"], age_min_ma=r["age"],
                             share=r["w"], rift_eruption=r["rift"]))
        print("  migration done", flush=True)

        # effective number of breakup events: margins nearest to kimberlite districts (20-70 Myr)
        for lo, hi in MAIN:
            (x, i, h), _ = vals(K, is_m, lo, hi)
            ids = pd.Series(i[h])
            rows.append(dict(kind="breakup events", dclass="KIMB", window=wname(lo, hi), quantity="margins nearest a district",
                             auc=int(ids.nunique()), n_districts=int(pd.Series(K.cluster[h]).nunique())))

    out = der / f"kimberlite_breakup_tests_{a.tag}.csv"
    part = out.with_name(out.name + ".partial")
    new_t = pd.DataFrame(rows)
    if set(a.parts) != set(PARTS) and out.exists():
        kinds = [k for p_ in a.parts for k in PARTS[p_]]
        old_t = pd.read_csv(out, low_memory=False)
        new_t = pd.concat([old_t[~old_t["kind"].isin(kinds)], new_t], ignore_index=True)
    rows = new_t.to_dict("records")
    new_t.to_csv(part, index=False, float_format="%.4g")
    part.replace(out)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    cols = ["kind", "background", "window", "quantity", "treatment", "auc", "lo", "hi", "null_median", "p", "power",
            "reach_dep", "reach_ref"]
    show = t[t["kind"].isin(["timing", "power", "extension", "alternatives", "stratified", "migration"])]
    print(show[[c for c in cols if c in show]].round(3).to_string(index=False))
    print(f"\nwrote {out} ({len(t)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
