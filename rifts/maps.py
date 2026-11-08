"""Reconstructed maps: continents, plate boundaries, a gridded field and deposits.

One module serves the figure panels and the animation frames, so that a frame of the video
and a panel of the figure are the same map drawn at different times. Projection is Robinson
where cartopy is installed and equirectangular otherwise; both take longitude and latitude, so
nothing else in the drawing code changes.
"""
from __future__ import annotations

import numpy as np
import pygplates

# class colours from the Okabe-Ito and Tol colour-blind-safe sets. No figure asks colour alone
# to separate classes: maps give each class its own marker (CLASS_MARKER) and line plots their
# own dash pattern (CLASS_LS), and rifts.figures checks every key in greyscale and under
# simulated protanopia, deuteranopia and tritanopia before a figure is written.
CLASS_COLOUR = {"SSC": "#D55E00", "CD": "#0072B2", "MVT": "#56B4E9", "ZNREP": "#CC79A7",
                "CARB": "#E69F00", "KIMB": "#009E73", "LAMP": "#999933", "AIREE": "#882255",
                "POT": "#8C510A", "PHOS": "#44AA99", "PCU": "#333333", "VMS": "#999999"}
CLASS_MARKER = {"SSC": "o", "CD": "s", "MVT": "D", "ZNREP": "^", "CARB": "o", "KIMB": "o",
                "LAMP": "P", "AIREE": "v", "POT": "s", "PHOS": "^", "PCU": "X", "VMS": "h"}
# dotted is reserved for reference points, which every comparison figure draws
CLASS_LS = {"SSC": "-", "CD": (0, (5, 1.5)), "MVT": (0, (5, 1.5, 1, 1.5)),
            "ZNREP": (0, (2, 1.2)), "PCU": (0, (9, 2.5)), "CARB": "-", "KIMB": "-",
            "AIREE": (0, (3, 1, 1, 1, 1, 1)), "LAMP": (0, (5, 1.5)), "POT": (0, (5, 1.5)),
            "PHOS": (0, (2, 1.2)), "VMS": (0, (5, 1.5, 1, 1.5))}
CLASS_LABEL = {"SSC": "sediment-hosted Cu", "CD": "clastic-dominated Zn–Pb", "MVT": "MVT Zn–Pb",
               "ZNREP": "Zn skarn and replacement", "CARB": "carbonatite", "KIMB": "kimberlite",
               "LAMP": "lamproite", "AIREE": "alkaline-igneous REE", "POT": "potash",
               "PHOS": "phosphorite", "PCU": "porphyry Cu", "VMS": "VMS"}
BOUNDARY_STYLE = {"SubductionZone": dict(color="#222222", lw=0.9),
                  "MidOceanRidge": dict(color="#c1272d", lw=0.6),
                  "other": dict(color="#b3b3b3", lw=0.4)}


def projection():
    """Robinson if cartopy is available, else None for plain lon/lat axes."""
    try:
        import cartopy.crs as ccrs
        return ccrs.Robinson(central_longitude=0), ccrs.PlateCarree()
    except Exception:
        return None, None


# meridians every 90°: the flat pole line of a Robinson map is half the map's width, and at
# journal lettering sizes labels any closer together collide on a panel 80 mm wide
LON_TICKS = (-180, -90, 0, 90, 180)
LAT_TICKS = (-60, -30, 0, 30, 60)
GRATICULE = dict(color="#b8b8b8", linewidth=0.25, alpha=0.8)


def _deg(v, axis):
    if v == 0:
        return "0°"
    return f"{abs(v):g}°{('E' if v > 0 else 'W') if axis == 'lon' else ('N' if v > 0 else 'S')}"


def prepare_axes(ax, proj, labels=True, fontsize=None):
    """``proj`` is None for plain lon/lat axes, anything else for a cartopy axes.

    Every map carries a graticule with latitude labels on its left edge and longitude labels
    on its bottom edge, in the lettering size of the journal the figure is drawn for.
    """
    import matplotlib as mpl
    fs = fontsize or mpl.rcParams["font.size"]
    if proj is None:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal")
        if labels:
            ax.set_xticks(LON_TICKS)
            ax.set_yticks(LAT_TICKS)
            ax.set_xticklabels([_deg(v, "lon") for v in LON_TICKS], fontsize=fs)
            ax.set_yticklabels([_deg(v, "lat") for v in LAT_TICKS], fontsize=fs)
            ax.tick_params(length=0, pad=1.5)
            ax.grid(True, **GRATICULE)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        ax.set_global()
        gl = ax.gridlines(draw_labels=["left", "bottom"] if labels else False,
                          xlocs=list(LON_TICKS), ylocs=list(LAT_TICKS), zorder=0.5, **GRATICULE)
        if labels:
            from matplotlib.ticker import FuncFormatter
            # the ±180° meridians are the map's edge and need no label
            gl.xformatter = FuncFormatter(lambda v, _: "" if abs(v) >= 179.9 else _deg(round(v), "lon"))
            gl.yformatter = FuncFormatter(lambda v, _: _deg(round(v), "lat"))
            gl.xlabel_style = gl.ylabel_style = dict(size=fs, color="#333333")
            gl.xpadding = gl.ypadding = 2
            gl.rotate_labels = False
    for s in ax.spines.values():
        s.set_linewidth(0.4)


def lowest_text_y(fig):
    """Figure-fraction height of the lowest lettering on any map, latitude and longitude labels
    included, so that a key hung below the maps can clear them. Layout managers do not see the
    graticule labels, which cartopy places at draw time."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    ys = [1.0]
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        ys.append(ax.get_window_extent(r).transformed(inv).y0)
        for gl in [c for c in ax.get_children() if type(c).__name__ == "Gridliner"]:
            for t in getattr(gl, "label_artists", []):
                if hasattr(t, "get_text") and t.get_visible() and t.get_text().strip():
                    ys.append(t.get_window_extent(r).transformed(inv).y0)
    return min(ys)


def _segments(lon, lat):
    """Split a polyline where it crosses the antimeridian, for plain lon/lat axes."""
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    cut = np.nonzero(np.abs(np.diff(lon)) > 180)[0] + 1
    return zip(np.split(lon, cut), np.split(lat, cut))


def draw_line(ax, lon, lat, tr, **kw):
    """A polyline, split where it crosses the antimeridian so that it is not drawn the long
    way round the map. Projections do not do this for us."""
    for x, y in _segments(lon, lat):
        if len(x) < 2:
            continue
        ax.plot(x, y, transform=tr, **kw) if tr is not None else ax.plot(x, y, **kw)


def wrapped_polygons(lon, lat, max_span=350.0, min_area_deg2=1e-3):
    """Polygon rings in [-180, 180], split at the antimeridian.

    A ring is first unwrapped so that it is continuous in longitude, then cut against the
    three copies of the map. A ring that encircles a pole is closed through that pole first,
    since in longitude and latitude it never closes on itself. Slivers below ``min_area_deg2``
    are dropped: a degenerate ring a few hundredths of a degree across carries no information
    at map scale and projects into a path that can fill the frame.
    """
    from shapely.affinity import translate
    from shapely.geometry import Polygon, box

    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    if len(lon) < 3:
        return []
    step = (np.diff(np.concatenate([lon, lon[:1]])) + 180.0) % 360.0 - 180.0
    u = np.concatenate([[lon[0]], lon[0] + np.cumsum(step)[:-1]])
    # a ring encircles a pole when its longitude changes by a full turn; a ring that merely
    # spans most of the globe does not, and closing that one through the pole would fill the map
    if abs(step.sum()) >= max_span:
        pole = 90.0 if lat.mean() > 0 else -90.0
        u = np.concatenate([u, [u[-1], u[0]]])
        lat = np.concatenate([lat, [pole, pole]])
    p = Polygon(np.column_stack([u, lat]))
    if not p.is_valid:
        p = p.buffer(0)
    # the clip stops a hair short of the antimeridian: an edge lying exactly on it projects
    # onto the map boundary, where the fill can flip to the outside of the map
    eps = 1e-6
    out = []
    for k in (-1, 0, 1):
        q = p.intersection(box(-180 + eps + 360 * k, -90, 180 - eps + 360 * k, 90))
        if not q.is_empty and q.area >= min_area_deg2:
            out.append(translate(q, xoff=-360 * k))
    return out


# continents are a backdrop: a near-white fill and soft outlines leave deposit symbols legible
LAND, LAND_EDGE, LAND_OUTLINE = "#f6f5f2", "#dcd8d1", "#9d978d"


def plot_continents(ax, model, time, tr, colour=LAND, edge=LAND_EDGE, fill=True,
                    zorder=1, lw=0.2):
    """Reconstructed continental polygons at ``time``, merged into landmasses and filled (or outlined only
    when not ``fill``)."""
    from shapely.ops import unary_union

    rec = []
    pygplates.reconstruct(model.continental_polygons, model.rotation_model, rec, float(time),
                          anchor_plate_id=model.anchor_plate)
    polys = []
    for r in rec:
        ll = r.get_reconstructed_geometry().to_lat_lon_array()
        polys += wrapped_polygons(ll[:, 1], ll[:, 0])
    if not polys:
        return
    # the continental polygons are a mosaic of blocks; merging them leaves the outline of the
    # landmass rather than a tangle of internal joins
    merged = unary_union([p.buffer(0) for p in polys])
    parts = list(getattr(merged, "geoms", [merged]))
    face = colour if fill else "none"
    if tr is not None:
        ax.add_geometries(parts, crs=tr, facecolor=face, edgecolor=edge, linewidth=lw,
                          zorder=zorder)
    else:
        for part in parts:
            x, y = np.asarray(part.exterior.coords).T
            ax.fill(x, y, facecolor=face, edgecolor=edge, linewidth=lw, zorder=zorder)


def boundary_halo(lw):
    """A light outline under a boundary line, for fields whose dark end hides dark lines."""
    from matplotlib import patheffects
    return [patheffects.Stroke(linewidth=lw + 1.1, foreground="white", alpha=0.85),
            patheffects.Normal()]


def boundary_style(name):
    """Line style of a plate boundary for the journal the figure is drawn for. Subduction zones
    are told from mid-ocean ridges by width as well as colour, because red and black are close
    for red-blind readers; a journal with a minimum line weight (GSA, 1-2 pt) gets the same
    three widths scaled into its range rather than all raised to the minimum."""
    from rifts import figures
    st = dict(BOUNDARY_STYLE.get(name, BOUNDARY_STYLE["other"]))
    spec = figures._SPEC or {}
    if spec.get("line_pt"):
        lo, hi = spec["line_pt"]
        ws = [v["lw"] for v in BOUNDARY_STYLE.values()]
        f = lo / min(ws)
        st["lw"] = min(st["lw"] * f, hi)
    return st


def plot_boundaries(ax, model, time, tr, halo=False, zorder=3):
    """Resolved plate boundaries, subduction zones heavier than the rest. ``halo`` draws each
    line over a light outline, for maps whose colour field is dark where lines run; ``zorder``
    places the lines above everything a map has already drawn when passed high enough."""
    snap = model.topological_snapshot(time)
    for section in snap.get_resolved_topological_sections():
        name = section.get_topological_section_feature().get_feature_type().get_name()
        style = boundary_style(name)
        for sub in section.get_shared_sub_segments():
            ll = sub.get_resolved_geometry().to_lat_lon_array()
            if len(ll) > 1:
                extra = dict(path_effects=boundary_halo(style["lw"])) if halo else {}
                draw_line(ax, ll[:, 1], ll[:, 0], tr, zorder=zorder, **style, **extra)


# carbon and water are different quantities, so they never share a colour scale
# Water has one colour scale in every figure of the base-metal paper (maps, videos and the
# line art of the concept figure): viridis, with line art drawn from the same scale. Carbon
# stays on magma.
WATER_CMAP = "viridis"
WATER_COLOUR, WATER_DARK = "#21918c", "#3b528b"    # viridis at 0.5 and 0.25, for line art
FIELD_CMAP = {"subducted_carbon": "magma_r",
              "mean_subducted_water": WATER_CMAP,
              "mean_slab_outflux": WATER_CMAP}


def field_cmap(variable):
    """Colour map for a volatile field: magma for carbon, viridis for water."""
    if variable in FIELD_CMAP:
        return FIELD_CMAP[variable]
    return WATER_CMAP if ("water" in variable or "outflux" in variable) else "magma_r"


def plot_grid(ax, da, tr, cmap="magma_r", norm=None, alpha=0.9, max_cols=900):
    """A global field as a warped raster.

    The delivery grids are 0.2 degrees, 1.6 million cells, and drawing them cell by cell in a
    projected frame costs more than everything else in the figure together. Coarsening to the
    resolution the page can show and warping the raster gives the same picture in a fraction
    of the time.
    """
    lon = np.asarray(da["lon"], float)
    lat = np.asarray(da["lat"], float)
    z = np.asarray(da.values, float)
    if lon.max() > 180:                              # 0-360 grids, shifted for plotting
        k = np.argsort(((lon + 180) % 360) - 180)
        lon = ((lon + 180) % 360 - 180)[k]
        z = z[:, k]
    step = max(int(np.ceil(len(lon) / max_cols)), 1)
    if step > 1:
        n_lat = (len(lat) // step) * step
        n_lon = (len(lon) // step) * step
        # blocks with no data at all (outside the grid's coverage) stay empty and are drawn
        # transparent; numpy warns about each of them, which says nothing about the figure
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            z = np.nanmean(z[:n_lat, :n_lon].reshape(n_lat // step, step, n_lon // step, step),
                           axis=(1, 3))
        lon = lon[:n_lon].reshape(-1, step).mean(axis=1)
        lat = lat[:n_lat].reshape(-1, step).mean(axis=1)
    kw = dict(cmap=cmap, norm=norm, alpha=alpha, zorder=2, rasterized=True)
    extent = [lon[0], lon[-1], lat[0], lat[-1]]
    origin = "lower" if lat[1] > lat[0] else "upper"
    if tr is not None:
        return ax.imshow(z, transform=tr, extent=extent, origin=origin, interpolation="bilinear",
                         **kw)
    return ax.imshow(z, extent=extent, origin=origin, interpolation="bilinear", aspect="auto",
                     **kw)


def plot_deposits(ax, lon, lat, classes, tr, size=9, edge=0.25, alpha=1.0, zorder=5,
                  colour=None, label=True, edgecolor="white"):
    """Scatter the deposits by class, with the marker, colour and legend label of the class, or circles of
    one ``colour``."""
    classes = np.asarray(classes)
    for c in np.unique(classes):
        m = classes == c
        kw = dict(s=size, marker=CLASS_MARKER.get(c, "o") if colour is None else "o",
                  facecolor=colour or CLASS_COLOUR.get(c, "#000000"),
                  edgecolor=edgecolor, linewidth=edge, alpha=alpha, zorder=zorder,
                  label=CLASS_LABEL.get(c, c) if label else None)
        if tr is not None:
            ax.scatter(np.asarray(lon)[m], np.asarray(lat)[m], transform=tr, **kw)
        else:
            ax.scatter(np.asarray(lon)[m], np.asarray(lat)[m], **kw)


def age_label(ax, time, x=0.015, y=0.04, fontsize=None, ha="left"):
    import matplotlib as mpl
    ax.text(x, y, f"{time:.0f} Ma", transform=ax.transAxes, fontsize=fontsize or mpl.rcParams["font.size"],
            fontweight="bold", va="bottom", ha=ha, zorder=20,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.75))


def deposit_layers(model, dep, time, window=25.0):
    """Deposits forming at ``time`` and those already in place, reconstructed to ``time``.

    A deposit exists at ``time`` when its age is at least ``time``; the ones whose age falls
    within ``window`` of it are the ones forming then, and are drawn in full colour over the
    faded record of everything older.
    """
    import numpy as np

    a = dep["age_ma"].to_numpy(float)
    have = a >= time
    if not have.any():
        return (np.array([]),) * 3 + (np.array([]),) * 3
    d = dep[have]
    lon, lat = model.reconstruct_points(d["lon"], d["lat"], d["pid"], float(time))
    fresh = (d["age_ma"].to_numpy(float) <= time + window) & np.isfinite(lon)
    old = ~fresh & np.isfinite(lon)
    cls = d["dclass"].to_numpy()
    return (lon[fresh], lat[fresh], cls[fresh], lon[old], lat[old], cls[old])


def active_outlines(outlines, time):
    """Outlines of the rifts active at ``time``: begun no later than it, ended no earlier."""
    return outlines[(outlines["onset_ma"] >= time) & (outlines["end_ma"] <= time)]


def reconstruct_outlines(model, gdf, time):
    """Rotate each rift outline with the plate beneath its representative point.

    An active rift sits within one plate until it breaks it, so a single rotation per outline
    is exact for most and a close approximation for the rest at the scale of a global map.
    """
    import numpy as np

    if not len(gdf):
        return []
    if "pid" in gdf:                               # assigned once by with_plate_ids
        pids = gdf["pid"].to_numpy()
    else:
        rp = gdf.geometry.representative_point()
        pids, _, _ = model.assign_plate_ids(rp.x.to_numpy(), rp.y.to_numpy())
    out = []
    for geom, pid in zip(gdf.geometry, pids):
        if pid <= 0:
            continue
        rot = model.rotation_model.get_rotation(float(time), int(pid),
                                                anchor_plate_id=model.anchor_plate)
        for poly in getattr(geom, "geoms", [geom]):
            ll = np.asarray(poly.exterior.coords)
            if len(ll) < 4:
                continue
            arr = (rot * pygplates.PolygonOnSphere([(la, lo) for lo, la in ll])).to_lat_lon_array()
            out += wrapped_polygons(arr[:, 1], arr[:, 0])
    return out


def with_plate_ids(model, outlines):
    """Attach the plate beneath each outline's representative point, once.

    Assigning plate IDs builds a plate partitioner, which is slow; an animation that did it for
    every frame would spend most of its time there.
    """
    rp = outlines.geometry.representative_point()
    out = outlines.copy()
    out["pid"] = model.assign_plate_ids(rp.x.to_numpy(), rp.y.to_numpy())[0]
    return out


RIFT_FACE, RIFT_EDGE = "#cab2d6", "#6a3d9a"

# Major rift settings (rifts/riftsets.py), one style for every figure that shows them: face,
# hatch, key label. The hues avoid both continuous scales of the paper, plasma (rift age,
# Fig. 1a) and viridis (water, Figs. 3 and 5), and the orange and blue deposit symbols:
# back-arc and arc rifts light blue, craton-edge rifts brown (the colour of the craton
# outlines on the maps), rifts near divergent boundaries grey, and the minor settings hatched
# or dotted. The figure scripts check the key for greyscale and colour-vision separation.
SETTING_STYLE = {
    "overriding plate near trench": ("#88CCEE", None, "overriding plate near a trench (back-arc and arc)"),
    "craton edge": ("#8C510A", None, "craton edge (within 200 km)"),
    "near divergent boundary": ("#888888", None, "near a divergent boundary"),
    "other plate boundary": ("#BBBBBB", "//////", "subducting plate near a trench, or near a transform"),
    "plate interior away from craton edges": ("#ffffff", "xxxx", "plate interior away from craton edges"),
    "setting not resolved": ("#ffffff", "......", "setting not resolved"),
}
# on the reconstructed maps only the two settings the paper contrasts are coloured; every other
# rift is drawn white, so that the field beneath stays readable
MAP_RIFT_STYLE = {
    "overriding plate near trench": ("#88CCEE", "#08306b", "back-arc and arc rift"),
    "craton edge": ("#8C510A", "#4a2a04", "craton-edge rift"),
    "other": ("#ffffff", "#6e6e6e", "other rift"),
}
CRATON_EDGE = dict(color="#3f2a12", lw=0.8, ls=(0, (3.0, 1.4)))


def plot_rifts(ax, model, outlines, time, tr, zorder=2.5, by_setting=False):
    """The rifts active at ``time``, reconstructed and shaded; with ``by_setting`` (and a
    ``major_setting`` column), back-arc and craton-edge rifts in their own colours."""
    act = active_outlines(outlines, time)
    if not by_setting or "major_setting" not in act:
        polys = reconstruct_outlines(model, act, time)
        if polys and tr is not None:
            ax.add_geometries(polys, crs=tr, facecolor=RIFT_FACE, edgecolor=RIFT_EDGE,
                              linewidth=0.35, zorder=zorder)
        return len(polys)
    n = 0
    key = act["major_setting"].where(act["major_setting"].isin(list(MAP_RIFT_STYLE)), "other")
    # other rifts first, so that the two coloured settings are drawn on top
    for k, name in enumerate(("other", "craton edge", "overriding plate near trench")):
        polys = reconstruct_outlines(model, act[key == name], time)
        face, edge, _ = MAP_RIFT_STYLE[name]
        if polys and tr is not None:
            # a firm edge keeps small rifts visible at the scale of a global panel; the
            # back-arc rifts get a white halo and a heavier edge so that they stand out
            # against the blue-green of the water field
            if name == "overriding plate near trench":
                ax.add_geometries(polys, crs=tr, facecolor="none", edgecolor="white",
                                  linewidth=2.4, zorder=zorder + 0.01 * k)
                ax.add_geometries(polys, crs=tr, facecolor=face, edgecolor=edge, linewidth=1.2,
                                  zorder=zorder + 0.01 * k + 0.005)
            else:
                ax.add_geometries(polys, crs=tr, facecolor=face, edgecolor=edge, linewidth=0.9,
                                  zorder=zorder + 0.01 * k)
        n += len(polys)
    return n


def plot_cratons(ax, model, cratons, time, tr, zorder=3.6):
    """Craton outlines (present-day outlines moving with their plates), thin dashed brown."""
    if cratons is None or tr is None:
        return 0
    from .cratons import merged_rings
    rec = []
    pygplates.reconstruct(cratons, model.rotation_model, rec, float(time),
                          anchor_plate_id=model.anchor_plate)
    geoms, names = [], []
    for r in rec:
        g = r.get_reconstructed_geometry()
        if isinstance(g, pygplates.PolygonOnSphere):
            geoms.append(g)
            names.append(r.get_feature().get_name() or "")
    polys = []
    # cratons cut by the static polygons are drawn as merged outlines, without the cut lines
    for lon, lat in merged_rings(geoms, names, spacing_deg=0.1):
        polys += wrapped_polygons(lon, lat)
    for g, nm in zip(geoms, names):
        if not nm:
            ll = g.to_lat_lon_array()
            polys += wrapped_polygons(ll[:, 1], ll[:, 0])
    if polys:
        ax.add_geometries(polys, crs=tr, facecolor="none", edgecolor=CRATON_EDGE["color"],
                          linewidth=CRATON_EDGE["lw"], linestyle=CRATON_EDGE["ls"], zorder=zorder)
    return len(polys)


def draw_frame(ax, model, time, tr, dep=None, field=None, norm=None, cmap="magma_r",
               window=25.0, legend=False, mesh_out=None, max_cols=900, rifts=None,
               halo=False, cratons=None, rifts_by_setting=False, emphasise_new=False):
    """``legend`` is kept for callers; the legend is drawn at figure level by the caller.
    ``halo`` outlines the plate boundaries in white, for the dark end of the water colour map."""
    """One reconstructed map: grid, continents, plate boundaries, deposits, age label."""
    prepare_axes(ax, None if tr is None else True)
    mesh = None
    plot_continents(ax, model, time, tr, fill=True, zorder=1)
    if field is not None:
        # partly transparent so that continents and craton outlines read through it
        mesh = plot_grid(ax, field, tr, cmap=cmap, norm=norm, alpha=0.62, max_cols=max_cols)
    if rifts is not None:
        plot_rifts(ax, model, rifts, time, tr, by_setting=rifts_by_setting)
    plot_continents(ax, model, time, tr, fill=False, zorder=3.5, edge=LAND_OUTLINE, lw=0.3)
    plot_cratons(ax, model, cratons, time, tr)
    plot_boundaries(ax, model, time, tr, halo=halo)
    if dep is not None and len(dep):
        nl, na, nc, ol, oa, oc = deposit_layers(model, dep, time, window)
        # ``emphasise_new``: the deposits forming at the time shown are the ones that test the
        # association with the field, so they are drawn large and opaque and the older record
        # as small pale dots
        old_kw = dict(size=1.0, alpha=0.22, colour="#8a8a8a") if emphasise_new else \
            dict(size=1.6, alpha=0.3, colour="#3b3b3b")
        new_kw = dict(size=26, edge=0.6, alpha=0.95) if emphasise_new else dict(size=17, edge=0.4, alpha=0.68)
        if len(ol):
            plot_deposits(ax, ol, oa, oc, tr, edge=0.0, zorder=4, label=False, **old_kw)
        if len(nl):
            # translucent with a thin dark outline: clusters darken where deposits overlap, and
            # the field beneath shows through
            plot_deposits(ax, nl, na, nc, tr, zorder=6, edgecolor="#1a1a1a", **new_kw)
    # lower right, clear of the latitude labels on the left edge at any lettering size
    age_label(ax, time, x=0.985, ha="right")
    if mesh_out is not None:
        mesh_out.append(mesh)
    return mesh
