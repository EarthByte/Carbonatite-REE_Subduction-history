#!/usr/bin/env python
"""Figures of the carbonatite and alkaline-silicate rare-earth paper (Nature Geoscience artwork rules).

Reads the tables of the carbonatite workflow for one point set (default carb26):
data/derived/carbonatite_features_<tag>, carbonatite_systems_<tag>.csv, spandler_reassessment_<tag>.csv,
carbonatite_secular_<tag>.csv and carbonatite_isotope_districts_<tag>.csv, with
data/external/reveal_lid_thickness.nc, the craton outlines of Shirmard et al. (2025) and the fertilised
domains of Spandler et al. (2026). No titles; panel letters and data identifiers only; every figure is
checked for overlapping and undersized lettering before it is written.

  carbonatite_fig1.pdf  districts over the thickness of the fast lithospheric lid, craton outlines and
                        the fertilised domains (present-day positions)
  (Fig. 2, the reconstructed slab-water maps with craton outlines and districts, is drawn by
   scripts/23_reconstruction_maps.py --paper carbonatite)
  carbonatite_fig3.pdf  subducted water and carbon along the plate path against reference lithosphere,
                        by window, for every class
  carbonatite_fig4.pdf  shares inside the fertilised domains, and the REE deposits of Spandler et al.
                        by host rock against the share expected at equal craton-edge distance
  carbonatite_fig5.pdf  the two rare-earth systems: time since the last subduction episode, trench
                        distance, dynamic-topography rise, numbers through time against subduction, and
                        neodymium enrichment with and without sediment-bound slab water (writes
                        carbonatite_nd_split_<tag>.csv)
  carbonatite_fig6.pdf  sketch of the results (the paper's closing figure)
  carbonatite_figS1.pdf lid thickness and shear-velocity gradient beneath the classes (--only 7)

--only takes the figure numbers 1, 3, 4, 5 and 6, and 7 for Supplementary Fig. 1.

    python scripts/91_carbonatite_figures.py --tag carb26
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifts.common import ROOT, load_config, read_table  # noqa: E402
from rifts.figures import assert_no_text_overlap, panel_letters, use_journal, width_mm  # noqa: E402
from rifts.figures import drawn_texts  # noqa: E402
from rifts.maps import CLASS_COLOUR, prepare_axes, projection, wrapped_polygons  # noqa: E402

MM = 1 / 25.4
LABEL = {"CARB": "carbonatites", "CARB ore": "ore-bearing carbonatites", "AIREE": "alkaline-silicate REE",
         "KIMB": "kimberlites", "PCU": "porphyry copper (control)"}
COLOUR = {"CARB": CLASS_COLOUR["CARB"], "CARB ore": "#8c5a00", "AIREE": CLASS_COLOUR["AIREE"],
          "KIMB": CLASS_COLOUR["KIMB"], "PCU": CLASS_COLOUR["PCU"]}
MARK = {"CARB": "o", "CARB ore": "s", "AIREE": "D", "KIMB": "^", "PCU": "X"}
REF_COLOUR = "#9a9a9a"
LS = {"CARB": "-", "AIREE": (0, (5, 1.5)), "KIMB": (0, (5, 1.5, 1, 1.5))}
SPANDLER = ROOT / "Papers" / "Spandler_carbonatites_subduction_lithospheric_mantle_fertilisation"


def save(fig, out, name):
    assert_no_text_overlap(fig)
    path = out / name
    fig.savefig(path, bbox_inches="tight", bbox_extra_artists=drawn_texts(fig))
    plt.close(fig)
    print(f"wrote {path}")


def wecdf(ax, x, w, **kw):
    ok = np.isfinite(x) & np.isfinite(w)
    x, w = x[ok], w[ok]
    if not len(x):
        return
    o = np.argsort(x)
    ax.step(x[o], np.cumsum(w[o]) / w.sum(), where="post", **kw)


def sides(p, group):
    d = p[(p["group"] == group) & (p["kind"] == "deposit")]
    r = p[(p["group"] == group) & (p["kind"] == "reference")]
    return d, r


def districts(p, group):
    d = p[(p["group"] == group) & (p["kind"] == "deposit")]
    return d.groupby("cluster_id").agg(lon=("lon0", "first"), lat=("lat0", "first"), ore=("ore", "max")).reset_index()


def polygons_of_spandler():
    """The fertilised-domain polygons of Spandler et al. (2026) at present-day positions, split at the
    antimeridian; files that are missing are skipped."""
    import pygplates
    files = ["Li_Polygons_on_Cao_0-900_km.gpml", "mesopro_polygons_0-900_km.gpml",
             "neopro_polygons_0-900_km.gpml", "phanero_polygons_0-900_km.gpml"]
    out = []
    for f in files:
        path = SPANDLER / "Extended Dataset 2" / "data" / "mantle_fertilisation_polygons" / f
        if not path.exists():
            continue
        for ft in pygplates.FeatureCollection(str(path)):
            for g in ft.get_all_geometries():
                ll = pygplates.PolygonOnSphere(g).to_lat_lon_array()
                out += wrapped_polygons(ll[:, 1], ll[:, 0])
    return out


def draw_polys(ax, polys, tr, **kw):
    """Draw ``polys`` on ``ax``: as cartopy geometries with the data transform ``tr``, or as patches on a
    plain longitude-latitude axis."""
    if tr is not None:
        ax.add_geometries(polys, crs=tr, **kw)
        return
    from matplotlib.patches import Polygon
    for g in polys:
        for part in getattr(g, "geoms", [g]):
            ax.add_patch(Polygon(np.asarray(part.exterior.coords), closed=True, **kw))


def fig1(p, out):
    """Sources are cited in the caption, not in the key: an author name without its reference number is an
    incomplete citation (review 2026-11-14). Fig. 1: carbonatite (with and without REE or Nb ore), alkaline-silicate REE and kimberlite districts
    over the REVEAL lid thickness (1.5% threshold), the craton outlines of Shirmard et al. (2025) and
    the fertilised domains of Spandler et al. (2026), at present-day positions."""
    import geopandas as gpd
    import xarray as xr
    proj, tr = projection()
    fig = plt.figure(figsize=(width_mm("double") * MM, 105 * MM))
    ax = fig.add_subplot(1, 1, 1, projection=proj) if proj is not None else fig.add_subplot(1, 1, 1)
    prepare_axes(ax, proj)
    lid = xr.open_dataset(ROOT / "data" / "external" / "reveal_lid_thickness.nc")["lid_km_1p5"]
    lid = lid.coarsen(latitude=2, longitude=2, boundary="trim").mean()
    kw = dict(transform=tr) if tr is not None else {}
    from matplotlib.colors import LinearSegmentedColormap
    greys = LinearSegmentedColormap.from_list("lid", plt.get_cmap("Greys")(np.linspace(0.02, 0.55, 256)))
    m = ax.pcolormesh(lid["longitude"], lid["latitude"], lid.values, cmap=greys, vmin=50, vmax=300,
                      shading="auto", rasterized=True, zorder=0.6, **kw)
    cr = gpd.read_file(ROOT / "data" / "external" / "Craton_Boundaries_Shirmard2025" / "Craton_Boundaries_Inferred.shp")
    cpolys = []
    for g in cr.geometry:
        for part in getattr(g, "geoms", [g]):
            x, y = np.asarray(part.exterior.coords).T
            cpolys += wrapped_polygons(x, y)
    draw_polys(ax, cpolys, tr, facecolor="none", edgecolor="#1f1f1f", linewidth=0.6, zorder=2)
    sp = polygons_of_spandler()
    draw_polys(ax, sp, tr, facecolor="none", edgecolor="#4393c3", linewidth=0.3, hatch="////", zorder=1.5)
    for grp, size in (("KIMB", 4), ("CARB", 14), ("AIREE", 16)):
        d = districts(p, grp)
        face = [COLOUR[grp] if (grp != "CARB" or o) else "white" for o in d["ore"]] if grp == "CARB" else COLOUR[grp]
        ax.scatter(d["lon"], d["lat"], s=size, marker=MARK[grp], c=face, edgecolors=COLOUR[grp] if grp != "KIMB" else "none",
                   linewidths=0.6, zorder=4, label=None, **kw)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], ls="", marker="o", mfc=COLOUR["CARB"], mec=COLOUR["CARB"], ms=4,
                label="carbonatite district with REE or Nb ore"),
         Line2D([], [], ls="", marker="o", mfc="white", mec=COLOUR["CARB"], ms=4, label="carbonatite district without recorded ore"),
         Line2D([], [], ls="", marker="D", mfc=COLOUR["AIREE"], mec=COLOUR["AIREE"], ms=4, label="alkaline-silicate REE district"),
         Line2D([], [], ls="", marker="^", mfc=COLOUR["KIMB"], mec="none", ms=3, label="kimberlite district"),
         Line2D([], [], color="#1f1f1f", lw=0.6, label="craton outline"),
         Patch(facecolor="none", edgecolor="#4393c3", hatch="////", lw=0.35, label="subduction-fertilised domain")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.45, -0.07))
    cax = fig.add_axes([0.92, 0.25, 0.012, 0.5])
    cb = fig.colorbar(m, cax=cax, extend="both")
    cb.set_label("thickness of the fast lithospheric lid (km)")
    save(fig, out, "carbonatite_fig1.pdf")


def compare_rows(sysdf, sets, feats):
    c = sysdf[(sysdf["part"] == "compare") & sysdf["set"].isin(sets) & sysdf["feature"].isin(feats)]
    return c.set_index(["set", "feature"])


FIG2_MEASURES = [
    ("out_total_path_full", "slab water released above 125 km, entire path"),
    ("h2o_mean_slab_outflux_track_400myr", "slab water released above 125 km, 400 Myr"),
    ("h2o_mean_slab_outflux_track_100myr", "slab water released above 125 km, 100 Myr"),
    ("out_sed_path_full", "sediment-bound water, entire path"),
    ("out_crust_path_full", "crust-bound water, entire path"),
    ("out_mantle_path_full", "serpentinite-bound water, entire path"),
    ("sub_total_path_full", "all subducted water, entire path"),
    ("deep_total_path_full", "water carried below 125 km, entire path"),
    ("c_mantle_path_full", "subducted carbon, entire path"),
    ("c_mantle_subducted_carbon_track_800myr", "subducted carbon, 800 Myr"),
    ("mem_carbon", "carbon held in the mantle reservoir"),
    ("mem_water_out", "water held in the uppermost-mantle reservoir"),
]


def fig3(sysdf, out):
    """Fig. 3 (Fig. 2 until v11): probability that the slab input beneath a district exceeds that beneath reference
    lithosphere of the same age, for every measure of FIG2_MEASURES and every class, with 95% intervals
    (carbonatite_systems compare rows)."""
    sets = ["CARB", "CARB ore", "AIREE", "KIMB", "PCU"]
    c = compare_rows(sysdf, sets, [m for m, _ in FIG2_MEASURES])
    meas = [(m, lab) for m, lab in FIG2_MEASURES if any((s, m) in c.index for s in sets)]
    fig, ax = plt.subplots(figsize=(width_mm("double") * MM, (22 + 9 * len(meas)) * MM))
    y0 = np.arange(len(meas))[::-1]
    for k, s in enumerate(sets):
        off = (k - 2) * 0.15
        for (m, _), y in zip(meas, y0):
            if (s, m) not in c.index:
                continue
            r = c.loc[(s, m)]
            ax.errorbar(r["auc"], y + off, xerr=[[r["auc"] - r["lo"]], [r["hi"] - r["auc"]]], fmt=MARK[s], ms=3.2,
                        color=COLOUR[s], elinewidth=0.6, capsize=0, label=LABEL[s] if m == meas[0][0] else None)
    ax.axvline(0.5, color="#666666", lw=0.5, ls=(0, (2, 2)))
    ax.set_yticks(y0)
    # where a window leaves districts out (it runs past the covered history), the row label carries the
    # numbers of carbonatite and alkaline-silicate districts that enter (review 2026-11-08)
    total = {s: max(c.loc[s, "n_districts"]) if s in c.index.get_level_values(0) else 0 for s in ("CARB", "AIREE")}
    labels = []
    for m, lab in meas:
        n = [int(c.loc[(s, m), "n_districts"]) if (s, m) in c.index else None for s in ("CARB", "AIREE")]
        short = [x is not None and x < total[s] for x, s in zip(n, ("CARB", "AIREE"))]
        labels.append(lab + (f" (n = {n[0]}, {n[1]})" if any(short) else ""))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(meas) - 0.4)
    ax.set_xlim(0.25, 0.9)
    ax.set_xlabel("probability that the value beneath a district exceeds that beneath\nreference lithosphere of the same age (0.5: no difference)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    save(fig, out, "carbonatite_fig3.pdf")


def fig4(sp, out, rc=None):
    """Fig. 4 (Fig. 3 until v11): (a) share of districts inside a fertilised domain against reference positions, the share
    expected at equal craton-edge distance and, where scripts/96 has run, the share expected under joint
    matching on craton-edge distance, shear-wave gradient and exploration density; (b) the REE deposits of
    Spandler et al. by host rock inside domains formed before them, within each age gap, against the
    expected share."""
    fig, axes = plt.subplots(1, 2, figsize=(width_mm("double") * MM, 62 * MM))
    a, b = axes
    groups = ["CARB", "AIREE", "KIMB", "PCU"]
    s = sp[(sp["kind"] == "share") & (sp["measure"] == "primed")].set_index("group")
    e = sp[(sp["kind"] == "share at equal") & (sp["measure"] == "primed") &
           (sp["condition"] == "craton_present_km")].set_index("group")
    x = np.arange(len(groups))
    # joint matching (scripts/96): the expected share is in median_ref of the "domains" rows
    j = None
    if rc is not None:
        jj = rc[(rc["part"] == "domains") & (rc["measure"] == "primed")].set_index("set")
        if set(groups) <= set(jj.index):
            j = jj
    bw = 0.2 if j is not None else 0.25
    off = [-1.5 * bw, -0.5 * bw, 0.5 * bw, 1.5 * bw] if j is not None else [-bw, 0.0, bw]
    a.bar(x + off[0], [s.loc[g, "share_dep"] for g in groups], bw, color=[COLOUR[g] for g in groups])
    a.errorbar(x + off[0], [s.loc[g, "share_dep"] for g in groups],
               yerr=[[s.loc[g, "share_dep"] - s.loc[g, "dep_lo"] for g in groups],
                     [s.loc[g, "dep_hi"] - s.loc[g, "share_dep"] for g in groups]], fmt="none", ecolor="#222222", elinewidth=0.6)
    a.bar(x + off[1], [s.loc[g, "share_ref"] for g in groups], bw, color="#d9d9d9", edgecolor="#7f7f7f", lw=0.5, hatch="////")
    a.bar(x + off[2], [e.loc[g, "share_ref"] for g in groups], bw, color="white", edgecolor="#7f7f7f", lw=0.8)
    if j is not None:
        a.bar(x + off[3], [j.loc[g, "median_ref"] for g in groups], bw, color="white", edgecolor="#222222", lw=0.8,
              hatch="....")
    from matplotlib.patches import Patch
    # the district bars take the colour of each class; their key entry is a hatched neutral swatch, which
    # the colour-safety check of rifts.figures tells apart from the grey and white reference entries
    a.legend(handles=[Patch(facecolor="white", edgecolor="#222222", lw=1.2, hatch="xx", label="districts (bars in the colour of the class)"),
                      Patch(facecolor="#d9d9d9", edgecolor="#7f7f7f", hatch="////", lw=0.5,
                            label="reference positions of the same ages"),
                      Patch(facecolor="white", edgecolor="#7f7f7f", lw=0.8, label="expected at equal craton-edge distance")]
             + ([Patch(facecolor="white", edgecolor="#222222", lw=0.8, hatch="....",
                       label="expected at equal craton-edge distance,\nshear-wave gradient and exploration density")]
                if j is not None else []),
             frameon=False, loc="upper left")
    a.set_xticks(x)
    a.set_xticklabels(["carbonatites", "alkaline-\nsilicate REE", "kimberlites", "porphyry\ncopper"])
    a.set_ylabel("share inside a fertilised domain")
    a.set_ylim(0, 1.3 if j is None else 1.6)      # room for the key above the bars
    a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    h = sp[(sp["kind"] == "spandler by host") & (sp["condition"] == "craton_present_km")]
    hosts = [("REE ore, carbonatite", "carbonatite-hosted REE ore"), ("REE ore, alkaline-silicate", "alkaline-silicate REE ore")]
    meas = [("primed", "any age"), ("primed_within400", "≤400 Myr"), ("primed_within800", "≤800 Myr"),
            ("primed_within1200", "≤1,200 Myr")]
    xm = np.arange(len(meas))
    for k, (grp, lab) in enumerate(hosts):
        r = h[h["group"] == grp].set_index("measure")
        col = COLOUR["CARB"] if "carbonatite" in grp else COLOUR["AIREE"]
        off = (k - 0.5) * 0.36
        b.bar(xm + off - 0.08, [r.loc[m, "share_dep"] for m, _ in meas], 0.16, color=col,
              label=f"{lab} (n = {int(r['n'].iloc[0])})")
        b.bar(xm + off + 0.08, [r.loc[m, "share_ref"] for m, _ in meas], 0.16, color="white", edgecolor=col, lw=0.8)
    b.set_xticks(xm)
    b.set_xticklabels([lab for _, lab in meas])
    b.set_xlabel("domain formed before the deposit, within")
    b.set_ylabel("share of deposits (filled) and expected (open)")
    b.set_ylim(0, 1.2)
    b.legend(frameon=False, loc="upper right")
    panel_letters(axes)
    fig.tight_layout()
    save(fig, out, "carbonatite_fig4.pdf")


def figS1(p, out):
    """Supplementary Fig. 1 (Fig. 5 until v10): weighted cumulative distributions of lid thickness and of the shear-wave-speed gradient at
    150 km beneath carbonatite, alkaline-silicate REE and kimberlite districts and reference
    lithosphere."""
    fig, axes = plt.subplots(1, 3, figsize=(width_mm("double") * MM, 58 * MM), sharey=True)
    # the lid is a floored proxy (50 km wherever the lithosphere is not fast at 60 km), so the shear-wave
    # speed at 150 km is shown as well (review 2026-11-08)
    for ax, col, xlab in ((axes[0], "lid_km", "thickness of the fast lithospheric lid (km)"),
                          (axes[1], "vsh150", "shear-wave speed at 150 km (km/s)"),
                          (axes[2], "vsh_grad150", "horizontal gradient of shear-wave speed\nat 150 km (km/s per 100 km)")):
        for grp in ("CARB", "AIREE", "KIMB"):
            d, _ = sides(p, grp)
            wecdf(ax, d[col].to_numpy(float), d["w_cluster"].to_numpy(float), color=COLOUR[grp], lw=1.2, ls=LS[grp],
                  label=LABEL[grp])
        _, r = sides(p, "CARB")
        wecdf(ax, r[col].to_numpy(float), np.ones(len(r)), color=REF_COLOUR, lw=1.0, ls=(0, (2, 1.5)),
              label="reference lithosphere")
        ax.set_xlabel(xlab)
    axes[0].set_ylabel("cumulative share of districts")
    axes[2].legend(frameon=False, loc="lower right")     # the only panel with an empty lower right
    panel_letters(axes)
    fig.tight_layout()
    save(fig, out, "carbonatite_figS1.pdf")


def subduction_panels(axes, p, sec):
    """Panels a-d of Fig. 5: weighted cumulative distributions for alkaline-silicate REE districts,
    carbonatite districts and reference lithosphere of (a) the time since the last subduction episode,
    (b) trench distance and (c) the rise of dynamic topography over 200 Myr, and (d) the high-pass
    alkaline-silicate REE record against the consumption of oceanic lithosphere through time."""
    a, b, c, d = axes
    ca, ra = sides(p, "AIREE")
    cc, rc = sides(p, "CARB")
    for ax, col, xlab, xmax in ((a, "over_900km_since_myr", "time since the last subduction episode\nbeneath the overriding plate (Myr)", 1500),
                                (b, "sz_dist_km", "distance to a subduction zone\nat formation (km)", 6000),
                                (c, "dt_rise_200", "rise of dynamic topography over the\npreceding 200 Myr (m)", None)):
        wecdf(ax, ca[col].to_numpy(float), ca["w_cluster"].to_numpy(float), color=COLOUR["AIREE"], lw=1.2, label=LABEL["AIREE"])
        wecdf(ax, cc[col].to_numpy(float), cc["w_cluster"].to_numpy(float), color=COLOUR["CARB"], lw=1.2, label=LABEL["CARB"])
        wecdf(ax, ra[col].to_numpy(float), np.ones(len(ra)), color=REF_COLOUR, lw=1.0, ls=(0, (2, 1.5)),
              label="reference lithosphere")
        ax.set_xlabel(xlab)
        if xmax:
            ax.set_xlim(0, xmax)
    a.set_ylabel("cumulative share")
    # one key for panels a-c, above them, since the narrow panels leave no corner free of data; panel a holds
    # only the districts with a reconstructed episode (19 and 106), which the caption gives
    a.figure.legend(*a.get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
                    bbox_to_anchor=(0.5, 1.0))
    s = sec[sec["part"] == "series"].sort_values("time_ma")
    t = s["time_ma"].to_numpy(float)

    def hp(v):
        v = pd.Series(v).rolling(5, center=True, min_periods=1).mean()
        return (v - v.rolling(30, center=True, min_periods=1).mean()).to_numpy()
    for name, col, lab in (("n_AIREE", COLOUR["AIREE"], "alkaline-silicate REE"),
                           ("consumed_km2_yr", "#222222", "consumption of oceanic lithosphere")):
        v = hp(s[name].to_numpy(float))
        d.plot(t, v / np.nanstd(v), color=col, lw=1.0, label=lab)
    d.set_xlim(1000, 0)
    d.set_ylim(-3.5, 6.5)
    d.set_xlabel("age (Ma)")
    d.set_ylabel("departure from the 300 Myr\nrunning mean (standard deviations)")
    d.legend(frameon=False, loc="upper left")


ND_SPLIT = 1e-6       # t/m²: below this the delivery is the numerical tail of the smoothing kernel (data gap 1e-7 to 1e-6)


def nd_split(iso, threshold=ND_SPLIT, n_boot=5000, seed=0):
    """Neodymium enrichment of carbonatite districts that did and did not receive sediment-bound slab water
    along their plate paths: median of each group, the difference of medians with a bootstrap interval
    (districts resampled within each group), for all districts and for those younger and older than 300 Ma,
    and the same split at other thresholds. Returns (districts, table)."""
    d = iso[(iso["cls"] == "CARB") & (iso["variant"] == "all samples")].dropna(subset=["d_eNd", "out_sed_path_full"]).copy()
    d["enrich"] = -d["d_eNd"]
    rng = np.random.default_rng(seed)
    rows = []
    for thr in sorted({threshold, 1e-9, 1e-7, 1e-5}):
        rec = d["out_sed_path_full"] > thr
        for lab, sel in (("all", np.ones(len(d), bool)), ("younger than 300 Ma", d["age_ma"] < 300),
                         ("300 Ma and older", d["age_ma"] >= 300)):
            a = d.loc[sel & rec, "enrich"].to_numpy()
            b = d.loc[sel & ~rec, "enrich"].to_numpy()
            bs = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b))) for _ in range(n_boot)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(dict(threshold_t_m2=thr, group=lab, n_received=len(a), n_none=len(b),
                             median_received=np.median(a), median_none=np.median(b),
                             difference=np.median(a) - np.median(b), lo=lo, hi=hi))
    d["received"] = d["out_sed_path_full"] > threshold
    return d, pd.DataFrame(rows)


def nd_panel(ax, d):
    """Panel e of Fig. 5: neodymium enrichment relative to the depleted mantle of carbonatite districts
    whose lithosphere did and did not receive sediment-bound slab water along its plate path, for all
    districts and for those younger and older than 300 Ma; boxes span the quartiles, bars mark the
    medians, numbers give the districts."""
    from matplotlib.lines import Line2D
    none_c, rec_c = REF_COLOUR, CLASS_COLOUR["CD"]
    groups = [("all", d), ("<300 Ma", d[d["age_ma"] < 300]), ("≥300 Ma", d[d["age_ma"] >= 300])]
    rng = np.random.default_rng(1)
    ticks = []
    for i, (lab, s) in enumerate(groups):
        for j, (rec, c) in enumerate(((False, none_c), (True, rec_c))):
            v = s.loc[s["received"] == rec, "enrich"].to_numpy()
            x = i * 2.6 + j
            q1, m, q3 = np.percentile(v, [25, 50, 75])
            ax.add_patch(plt.Rectangle((x - 0.32, q1), 0.64, q3 - q1, facecolor=c, alpha=0.18, edgecolor="none", zorder=1))
            ax.plot([x - 0.32, x + 0.32], [m, m], color=c, lw=1.6, zorder=3, solid_capstyle="butt")
            ax.scatter(x + rng.uniform(-0.22, 0.22, len(v)), v, s=6, color=c, edgecolors="white", linewidths=0.3, zorder=2)
            ax.text(x, -1.6, f"{len(v)}", ha="center", va="top", color=c)
        ticks.append((i * 2.6 + 0.5, lab))
    ax.set_xticks([t for t, _ in ticks])
    ax.set_xticklabels([l for _, l in ticks])
    ax.tick_params(axis="x", length=0, pad=11)
    ax.set_ylim(-3.6, 29.5)
    ax.set_xlim(-0.7, 2 * 2.6 + 1.7)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.spines["left"].set_bounds(0, 22.5)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.set_ylabel("neodymium enrichment relative to\ndepleted mantle (ε units)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=none_c, ms=3, label="no sediment-bound slab water"),
                       Line2D([], [], marker="o", ls="", color=rec_c, ms=3, label="sediment-bound slab water")],
              loc="upper left", frameon=False, handletextpad=0.2, borderaxespad=0.1)


def fig5(p, sec, iso, out, der, tag):
    """Fig. 5: the two rare-earth systems. (a-d) subduction history of alkaline-silicate REE and
    carbonatite districts (subduction_panels); (e) neodymium enrichment of carbonatite districts with and
    without sediment-bound slab water (nd_panel). Writes carbonatite_nd_split_<tag>.csv as well."""
    d, tab = nd_split(iso)
    tab.to_csv(der / f"carbonatite_nd_split_{tag}.csv", index=False)
    print(tab[tab["threshold_t_m2"] == ND_SPLIT].round(2).to_string(index=False))
    fig = plt.figure(figsize=(width_mm("double") * MM, 118 * MM))
    gs = fig.add_gridspec(2, 6, hspace=0.6, wspace=0.9, top=0.9, bottom=0.1, left=0.08, right=0.99)
    a = fig.add_subplot(gs[0, 0:2])
    b = fig.add_subplot(gs[0, 2:4], sharey=a)
    c = fig.add_subplot(gs[0, 4:6], sharey=a)
    d_ax = fig.add_subplot(gs[1, 0:4])
    e = fig.add_subplot(gs[1, 4:6])
    for ax in (b, c):
        ax.tick_params(labelleft=False)
    subduction_panels((a, b, c, d_ax), p, sec)
    nd_panel(e, d)
    panel_letters([a, b, c, d_ax, e])
    save(fig, out, "carbonatite_fig5.pdf")


def _rounded(pts, n=4):
    """Chaikin corner cutting: a closed polygon with its corners rounded off, for the sketch."""
    p = np.asarray(pts, float)
    for _ in range(n):
        q = np.roll(p, -1, axis=0)
        p = np.column_stack([0.75 * p + 0.25 * q, 0.25 * p + 0.75 * q]).reshape(-1, 2)
    return p


def fig6(out):
    """Fig. 6: sketch of the results. Carbonatites rise at the lateral seismic-velocity contrast at the
    edge of a cratonic keel, from mantle that holds sediment recycled by ancient subduction; alkaline-
    silicate REE systems form inboard of an active arc late in long-lived subduction, about half as far
    from the trench as reference lithosphere. Schematic, not to scale; distances and times are the
    medians of the paper. Drawn at the 89 mm single-column width and wider than tall, so that the 7 pt
    lettering (the Nature maximum) reads at a useful size against the artwork; shapes have rounded corners
    and are clipped to a rounded frame, every label lies inside the frame, and each label sits on its
    object or is tied to it by a thin leader line (review 2026-11-13). Saturation follows importance: pale
    setting, stronger colour for the keel edge, the velocity contrast, the slab water and the recycled
    sediment."""
    from matplotlib.patches import Polygon, FancyArrowPatch, FancyBboxPatch
    carb, airee, pcu = COLOUR["CARB"], COLOUR["AIREE"], COLOUR["PCU"]
    sky, mantle, lith, keel, crust = "#eef4f8", "#f7f0e6", "#ebdcc0", "#b07a3c", "#c3c8cc"
    ocean, oplate, slab, old = "#a9d0ee", "#8592a0", "#5f7386", "#dcd6e2"
    wedge, sed, step = "#4fa3e3", "#174a93", "#d62839"
    grey = "#555555"
    k = 0.72                     # vertical squeeze of everything below the surface
    X0, X1, Y0, Y1 = 0, 100, -36, 25
    fig, ax = plt.subplots(figsize=(width_mm("single") * MM, 58 * MM))
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.axis("off")
    frame = FancyBboxPatch((X0, Y0), X1 - X0, Y1 - Y0, boxstyle="round,pad=0,rounding_size=3.5",
                           facecolor="none", edgecolor="none", zorder=0)
    ax.add_patch(frame)

    def sq(pts):
        return [(x, y * k if y < 0 else y) for x, y in pts]

    def poly(pts, fc, z, rounded=True, n=4, **kw):
        q = _rounded(sq(pts), n) if rounded else sq(pts)
        pa = Polygon(q, closed=True, facecolor=fc, edgecolor="none", zorder=z, **kw)
        ax.add_patch(pa)
        pa.set_clip_path(frame)

    def arrow(a, b, c, lw=1.6, **kw):
        a, b = sq([a])[0], sq([b])[0]
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, color=c, lw=lw, zorder=4, **kw))

    def volcano(x, c):
        ax.add_patch(Polygon(_rounded([(x - 2.6, -0.2), (x + 2.6, -0.2), (x + 0.5, 2.8), (x - 0.5, 2.8)], 3),
                             closed=True, facecolor=c, edgecolor="none", zorder=5))

    def text(x, y, s, **kw):
        ax.text(x, y * k if y < 0 else y, s, **kw)

    def tag(s, xy_text, xy_obj, c, **kw):
        """A label tied to its object by a thin leader line in the label's colour."""
        ax.annotate(s, xy=sq([xy_obj])[0], xytext=sq([xy_text])[0], color=c, zorder=6, **kw,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.6, shrinkA=1.5, shrinkB=0.5))

    # sky and mantle fill the frame; shapes reach beyond it and are clipped, so only inner corners round
    poly([(-10, 0), (110, 0), (110, 40), (-10, 40)], sky, 0, rounded=False)
    poly([(-10, -80), (110, -80), (110, 0), (-10, 0)], mantle, 0.1, rounded=False)
    # continental lithosphere: a cratonic keel at the left, its edge at x ~ 30, thinner towards the arc
    poly([(-10, -1), (86.5, -1), (86.5, -6), (78, -7), (60, -8), (38, -9), (33, -13), (31, -22), (27, -27), (-10, -27)], lith, 2)
    poly([(-10, -2), (25, -2), (29, -8), (31, -20), (27, -25), (-10, -25)], keel, 2.1)
    poly([(-10, 0), (86.5, 0), (86.5, -2.2), (-10, -2.2)], crust, 2.2, rounded=False)
    text(11, -13, "cratonic\nkeel", ha="center", va="center", color="white")
    ax.plot(*zip(*sq([(29.0, -6), (31.2, -12), (31.7, -21), (29.8, -25.5)])), color=step, lw=1.6,
            ls=(0, (3, 1.6)), zorder=3, dash_capstyle="round")
    tag("velocity\ncontrast", (40, -24), (31.6, -17.5), step, ha="left", va="center")
    # an ancient slab and the sediment it left in the lithosphere at the keel edge
    poly([(41, -15), (45, -15), (30, -48), (26, -48)], old, 1, alpha=0.9)
    tag("ancient slab", (44, -40), (35.5, -33), "#6a5f78", ha="left", va="center")
    rng = np.random.default_rng(3)
    dx, dy = rng.uniform(25, 35, 40), rng.uniform(-24, -15, 40)
    dots = ax.scatter(dx, dy * k, s=2.4, color=sed, zorder=3.2, linewidths=0)
    dots.set_clip_path(frame)
    tag("recycled\nsediment", (12, -38), (26, -22.5), sed, ha="center", va="center")
    arrow((31.1, -20), (30.2, 0.4), carb, connectionstyle="arc3,rad=0.15")
    # active margin: oceanic plate, retreating slab, hydrated wedge, arc
    poly([(86.5, 0), (150, 0), (150, 1.8), (86.5, 1.8)], ocean, 2.3, rounded=False)
    poly([(86.5, 0), (150, 0), (150, -5), (89, -5)], oplate, 2.4, rounded=False)
    poly([(85, 1.5), (91, -3), (71, -70), (64, -70)], slab, 1.5, n=2)
    arrow((96, -2.4), (91, -2.4), "white", lw=1.0)
    arrow((82, -28), (90, -28), grey, lw=1.2)
    text(86.5, -31, "slab\nretreat", ha="center", va="top", color=grey)
    poly([(55, -9), (78, -7.5), (84, -6), (76, -21), (68, -23), (56, -15)], wedge, 1.8)
    text(68, -14.5, "slab water\nand carbon", ha="center", va="center", color="white")
    # volcanoes, each with its name and the time since subduction beneath it (medians)
    arrow((62, -9.5), (60.3, 0.4), airee)
    for x, c, name, when in ((30.1, carb, "carbonatite", "460 ± 40 Myr\nafter subduction"),
                             (58, airee, "alkaline-silicate REE", "5 Myr after\nsubduction")):
        volcano(x if c == carb else 60, c)
        t = ax.text(x, 3.6, when, ha="center", va="bottom", color=c)
        ax.annotate(name, xy=(0.5, 1.0), xycoords=t, xytext=(0, 3.0), textcoords="offset points",
                    ha="center", va="bottom", color=c, fontweight="bold")
    volcano(80, pcu)
    ax.text(79.5, 3.6, "porphyry\ncopper", ha="center", va="bottom", color=pcu)
    # distance from the trench at formation (medians), and the trench itself
    ax.add_patch(FancyArrowPatch((86.5, 18.5), (60, 18.5), arrowstyle="<|-|>", mutation_scale=6, color=airee, lw=0.9, zorder=4))
    ax.text(73.2, 19.3, "1,200 km", ha="center", va="bottom", color=airee)
    ax.plot([86.5, 86.5], [2.4, 17.8], color=grey, lw=0.6, ls=(0, (2, 1.5)), zorder=3)
    ax.text(88, 14.0, "trench", ha="left", va="center", color=grey)
    fig.tight_layout(pad=0.1)
    save(fig, out, "carbonatite_fig6.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--tag", default="carb26")
    ap.add_argument("--only", nargs="*", type=int)
    a = ap.parse_args()
    cfg = load_config(a.config)
    der = Path(cfg["paths"]["derived"])
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    use_journal("ngeo")
    p = read_table(der / f"carbonatite_features_{a.tag}")
    # the districts and reference points compared in scripts/84: not beneath present-day ice and on a block the
    # plate model holds at their age, so that every figure shows the population of the statistics
    p = p[~p["ice"].fillna(False).astype(bool) & p["in_model"].fillna(True).astype(bool)]
    sysdf = pd.read_csv(der / f"carbonatite_systems_{a.tag}.csv", low_memory=False)
    todo = set(a.only or (1, 3, 4, 5, 6, 7))
    if 1 in todo:
        fig1(p, out)
    # Fig. 2, the reconstructed slab-water maps, is drawn by scripts/23 (--paper carbonatite)
    if 3 in todo:
        fig3(sysdf, out)
    if 4 in todo:
        rc = der / f"carbonatite_review_checks_{a.tag}.csv"
        fig4(pd.read_csv(der / f"spandler_reassessment_{a.tag}.csv"), out,
             pd.read_csv(rc) if rc.exists() else None)
    if 5 in todo:
        fig5(p, pd.read_csv(der / f"carbonatite_secular_{a.tag}.csv"),
             pd.read_csv(der / f"carbonatite_isotope_districts_{a.tag}.csv"), out, der, a.tag)
    if 6 in todo:
        fig6(out)
    if 7 in todo:
        figS1(p, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
