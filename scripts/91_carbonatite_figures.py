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
  carbonatite_fig2.pdf  subducted water and carbon along the plate path against reference lithosphere,
                        by window, for every class
  carbonatite_fig3.pdf  shares inside the fertilised domains, and the REE deposits of Spandler et al.
                        by host rock against the share expected at equal craton-edge distance
  carbonatite_fig4.pdf  the two rare-earth systems: time since the last subduction episode, trench
                        distance, dynamic-topography rise, and numbers through time against subduction
  carbonatite_fig5.pdf  lid thickness and shear-velocity gradient beneath the classes
  carbonatite_fig6.pdf  neodymium enrichment against sediment-bound slab water along the plate path

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
    """Fig. 1: carbonatite (with and without REE or Nb ore), alkaline-silicate REE and kimberlite districts
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
         Line2D([], [], ls="", marker="o", mfc="white", mec=COLOUR["CARB"], ms=4, label="carbonatite district without ore"),
         Line2D([], [], ls="", marker="D", mfc=COLOUR["AIREE"], mec=COLOUR["AIREE"], ms=4, label="alkaline-silicate REE district"),
         Line2D([], [], ls="", marker="^", mfc=COLOUR["KIMB"], mec="none", ms=3, label="kimberlite district"),
         Line2D([], [], color="#1f1f1f", lw=0.6, label="craton outline (Shirmard et al.)"),
         Patch(facecolor="none", edgecolor="#4393c3", hatch="////", lw=0.35, label="fertilised domain (Spandler et al.)")]
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


def fig2(sysdf, out):
    """Fig. 2: probability that the slab input beneath a district exceeds that beneath reference
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
    save(fig, out, "carbonatite_fig2.pdf")


def fig3(sp, out):
    """Fig. 3: (a) share of districts inside a fertilised domain against reference positions and the share
    expected at equal craton-edge distance, (b) the REE deposits of Spandler et al. by host rock inside
    domains formed before them, within each age gap, against the expected share."""
    fig, axes = plt.subplots(1, 2, figsize=(width_mm("double") * MM, 62 * MM))
    a, b = axes
    groups = ["CARB", "AIREE", "KIMB", "PCU"]
    s = sp[(sp["kind"] == "share") & (sp["measure"] == "primed")].set_index("group")
    e = sp[(sp["kind"] == "share at equal") & (sp["measure"] == "primed") &
           (sp["condition"] == "craton_present_km")].set_index("group")
    x = np.arange(len(groups))
    a.bar(x - 0.25, [s.loc[g, "share_dep"] for g in groups], 0.25, color=[COLOUR[g] for g in groups])
    a.errorbar(x - 0.25, [s.loc[g, "share_dep"] for g in groups],
               yerr=[[s.loc[g, "share_dep"] - s.loc[g, "dep_lo"] for g in groups],
                     [s.loc[g, "dep_hi"] - s.loc[g, "share_dep"] for g in groups]], fmt="none", ecolor="#222222", elinewidth=0.6)
    a.bar(x, [s.loc[g, "share_ref"] for g in groups], 0.25, color="#d9d9d9", edgecolor="#7f7f7f", lw=0.5, hatch="////")
    a.bar(x + 0.25, [e.loc[g, "share_ref"] for g in groups], 0.25, color="white", edgecolor="#7f7f7f", lw=0.8)
    from matplotlib.patches import Patch
    # the district bars take the colour of each class; their key entry is a hatched neutral swatch, which
    # the colour-safety check of rifts.figures tells apart from the grey and white reference entries
    a.legend(handles=[Patch(facecolor="white", edgecolor="#222222", lw=1.2, hatch="xx", label="districts (bars in the colour of the class)"),
                      Patch(facecolor="#d9d9d9", edgecolor="#7f7f7f", hatch="////", lw=0.5,
                            label="reference positions of the same ages"),
                      Patch(facecolor="white", edgecolor="#7f7f7f", lw=0.8, label="expected at equal craton-edge distance")],
             frameon=False, loc="upper left")
    a.set_xticks(x)
    a.set_xticklabels(["carbonatites", "alkaline-\nsilicate REE", "kimberlites", "porphyry\ncopper"])
    a.set_ylabel("share inside a fertilised domain")
    a.set_ylim(0, 1.3)
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
    save(fig, out, "carbonatite_fig3.pdf")


def fig4(p, sec, out):
    """Fig. 4: weighted cumulative distributions for alkaline-silicate REE districts, carbonatite districts
    and reference lithosphere of (a) the time since the last subduction episode, (b) trench distance,
    (c) the rise of dynamic topography over 200 Myr, and (d) the high-pass alkaline-silicate REE record
    against the consumption of oceanic lithosphere through time."""
    fig, axes = plt.subplots(2, 2, figsize=(width_mm("double") * MM, 110 * MM))
    (a, b), (c, d) = axes
    ca, ra = sides(p, "AIREE")
    cc, rc = sides(p, "CARB")
    for ax, col, xlab, xmax in ((a, "over_900km_since_myr", "time since the last subduction episode beneath\nthe overriding plate (Myr)", 1500),
                                (b, "sz_dist_km", "distance to a subduction zone at formation (km)", 6000),
                                (c, "dt_rise_200", "rise of dynamic topography over the preceding\n200 Myr (m; median of six mantle-flow models)", None)):
        wecdf(ax, ca[col].to_numpy(float), ca["w_cluster"].to_numpy(float), color=COLOUR["AIREE"], lw=1.2, label=LABEL["AIREE"])
        wecdf(ax, cc[col].to_numpy(float), cc["w_cluster"].to_numpy(float), color=COLOUR["CARB"], lw=1.2, label=LABEL["CARB"])
        wecdf(ax, ra[col].to_numpy(float), np.ones(len(ra)), color=REF_COLOUR, lw=1.0, ls=(0, (2, 1.5)),
              label="reference lithosphere")
        ax.set_xlabel(xlab)
        ax.set_ylabel("cumulative share")
        if xmax:
            ax.set_xlim(0, xmax)
    a.legend(frameon=False, loc="lower right")
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
    panel_letters(axes.ravel())
    fig.tight_layout()
    save(fig, out, "carbonatite_fig4.pdf")


def fig5(p, out):
    """Fig. 5: weighted cumulative distributions of lid thickness and of the shear-wave-speed gradient at
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
    axes[0].legend(frameon=False, loc="lower right")
    panel_letters(axes)
    fig.tight_layout()
    save(fig, out, "carbonatite_fig5.pdf")


def fig6(iso, out):
    """Fig. 6: neodymium enrichment relative to the depleted mantle of carbonatite districts against the
    sediment-bound slab water delivered along the plate path, coloured by age."""
    d = iso[(iso["cls"] == "CARB") & (iso["variant"] == "all samples")].dropna(subset=["d_eNd", "out_sed_path_full"])
    fig, ax = plt.subplots(figsize=(width_mm("single") * MM, 70 * MM))
    sc = ax.scatter(d["out_sed_path_full"].clip(lower=1e-7), -d["d_eNd"], c=d["age_ma"], cmap="viridis", s=10,
                    edgecolors="#333333", linewidths=0.3, vmin=0, vmax=1500)
    ax.set_xscale("log")
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "none" if np.isclose(v, 1e-7) else f"{v:g}"))
    ax.set_xlabel("sediment-bound slab water delivered\nalong the plate path (t/m²)")
    ax.set_ylabel("neodymium enrichment relative to\ndepleted mantle (ε units)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("age (Ma)")
    fig.tight_layout()
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
    p = p[~p["ice"].fillna(False).astype(bool)]
    sysdf = pd.read_csv(der / f"carbonatite_systems_{a.tag}.csv", low_memory=False)
    todo = set(a.only or range(1, 7))
    if 1 in todo:
        fig1(p, out)
    if 2 in todo:
        fig2(sysdf, out)
    if 3 in todo:
        fig3(pd.read_csv(der / f"spandler_reassessment_{a.tag}.csv"), out)
    if 4 in todo:
        fig4(p, pd.read_csv(der / f"carbonatite_secular_{a.tag}.csv"), out)
    if 5 in todo:
        fig5(p, out)
    if 6 in todo:
        fig6(pd.read_csv(der / f"carbonatite_isotope_districts_{a.tag}.csv"), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
