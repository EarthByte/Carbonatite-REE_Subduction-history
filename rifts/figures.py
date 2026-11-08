"""Figure helpers: journal lettering rules, panel letters and text-collision detection.

Figures carry no titles; only panel letters and data identifiers appear in the artwork.
``assert_no_text_overlap`` is called by every figure script before saving.

Each paper's figures follow the artwork rules of its journal, applied at the size the figure
will be printed. A figure wider than the journal's widest column is reduced to fit, and its
lettering with it, so every size check is made on the reduced size:

  EPSL (Elsevier)    widths 90 / 140 / 190 mm; Elsevier's floor is 7 pt (sub- and superscripts
                     6 pt), but EPSL requires figures legible at A4 without enlarging and rejects
                     text disproportionately small for the image, and Elsevier allows ~10 pt for
                     dense artwork; lettering is therefore set at 9 pt, never below 8 pt, panel
                     letters 10 pt bold
  Nature Geoscience  widths 89 / 120 / 183 mm; lettering 5-7 pt (held here at 6.5-7 pt), panel
                     letters 8 pt bold lower case
  Geology (GSA)      widths 89 / 181 mm; lettering 7-12 pt (set at 9 pt, never below 8 pt, as
                     for EPSL), part labels 13-16 pt, capitals; lines in graphs, maps and keys
                     between 1 and 2 pt, which ``assert_no_text_overlap`` applies to every
                     stroked line, outline and tick before checking the figure

A script declares its journal with ``use_journal`` before it draws; the check then knows the
rules. Mathtext exponents ("km s$^{-1}$") are set at about 70 per cent of the lettering and
fall below every minimum, so labels use Unicode superscripts ("km/s") instead, and the check
rejects mathtext scripts that would print too small.
"""
from __future__ import annotations

import string

import matplotlib as mpl

MIN_FONT_PT = 6.0      # used only when a script has not declared its journal

JOURNALS = {
    "epsl": dict(name="Earth and Planetary Science Letters", widths_mm=(90, 140, 190),
                 font_pt=9.0, min_pt=8.0, min_script_pt=6.0, max_pt=None,
                 letter_pt=10.0, upper=False),
    "ngeo": dict(name="Nature Geoscience", widths_mm=(89, 120, 183),
                 font_pt=7.0, min_pt=6.5, min_script_pt=6.5, max_pt=7.0,
                 letter_pt=8.0, upper=False),
    "geology": dict(name="Geology", widths_mm=(89, 181),
                    font_pt=9.0, min_pt=8.0, min_script_pt=7.0, max_pt=12.0,
                    letter_pt=13.0, upper=True, line_pt=(1.0, 2.0)),
}
# Science Advances (AAAS, checked 2026-10-09): online only, figures at one column (89 mm) or
# full width (183 mm); Science family lettering 6-8 pt at print size, panel letters bold capitals
JOURNALS["scienceadvances"] = dict(name="Science Advances", widths_mm=(89, 120, 183),
                                   font_pt=8.0, min_pt=7.0, min_script_pt=6.0, max_pt=12.0,
                                   letter_pt=10.0, upper=True, line_pt=(0.75, 1.5))
# GSA Bulletin follows the GSA figure rules shared with Geology
JOURNALS["gsab"] = dict(JOURNALS["geology"], name="GSA Bulletin")
# Gondwana Research is an Elsevier journal: Elsevier artwork rules, as for EPSL
JOURNALS["gr"] = dict(JOURNALS["epsl"], name="Gondwana Research")
# Basin Research (Wiley): figures are supplied as separate files for review; the Elsevier sizes and
# lettering floor stand in until the post-acceptance artwork rules are applied
JOURNALS["basinresearch"] = dict(JOURNALS["epsl"], name="Basin Research")
PAPER_JOURNAL = {"basemetals": "epsl", "magmatic": "basinresearch", "kimberlite": "scienceadvances",
                 "carbonatite": "ngeo"}
_SPEC = None           # the journal declared by the running script

# Arial or Helvetica, which all three journals ask for; Liberation Sans is metrically the same
mpl.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]})
mpl.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "pdf.fonttype": 42, "savefig.dpi": 600,
                     # the margin a tight save adds counts towards the printed width
                     "savefig.pad_inches": 0.02})


def use_journal(key):
    """Declare the journal a figure is drawn for (a journal key or a paper name) and set the
    default lettering to its size. Returns the journal's rules."""
    global _SPEC
    _SPEC = dict(JOURNALS[PAPER_JOURNAL.get(key, key)], key=PAPER_JOURNAL.get(key, key))
    fs = _SPEC["font_pt"]
    mpl.rcParams.update({"font.size": fs, "axes.titlesize": fs, "axes.labelsize": fs,
                         "xtick.labelsize": fs, "ytick.labelsize": fs, "legend.fontsize": fs,
                         "legend.title_fontsize": fs})
    if _SPEC.get("line_pt"):
        lo = _SPEC["line_pt"][0]
        mpl.rcParams.update({"axes.linewidth": lo, "xtick.major.width": lo, "ytick.major.width": lo,
                             "xtick.minor.width": lo, "ytick.minor.width": lo,
                             "lines.linewidth": max(lo, 1.5), "patch.linewidth": lo,
                             "hatch.linewidth": lo, "legend.frameon": False})
    return _SPEC


def _enforce_line_weights(fig, r):
    """Bring every stroked line of the figure within the journal's line-weight range at print
    size (GSA: 1-2 pt in graphs, maps and keys): data lines, outlines of filled shapes, plate
    boundaries, graticules, axis spines and ticks, and the lines in keys. Strokes of width zero
    (fills drawn without an outline) are left alone, and so are the outlines of point symbols,
    which are symbols rather than lines."""
    import numpy as np
    from matplotlib.collections import Collection, PathCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text
    lo_pt, hi_pt = _SPEC["line_pt"]
    scale, _ = print_scale(fig, r)
    lo, hi = lo_pt / scale, hi_pt / scale

    def clamp(w):
        return w if w == 0 else min(max(w, lo), hi)

    for art in fig.findobj():
        if isinstance(art, Text):
            continue
        if isinstance(art, Line2D):
            if art.get_linestyle() not in ("None", "none", "", " "):
                art.set_linewidth(clamp(art.get_linewidth()))
            elif art.get_marker() in (0, 1, 2, 3, "|", "_"):      # axis ticks
                art.set_markeredgewidth(clamp(art.get_markeredgewidth()))
        elif isinstance(art, Patch):
            if art.get_edgecolor()[3] > 0 and art.get_linewidth() > 0:
                art.set_linewidth(clamp(art.get_linewidth()))
        elif isinstance(art, Collection) and not (isinstance(art, PathCollection)
                                                  and len(art.get_offsets()) > 0):
            w = np.atleast_1d(art.get_linewidths())
            if len(w) and np.any(w > 0):
                art.set_linewidths([clamp(float(x)) for x in w])
    for ax in fig.axes:
        for sp in ax.spines.values():
            if sp.get_visible() and sp.get_linewidth() > 0:
                sp.set_linewidth(clamp(sp.get_linewidth()))


def width_mm(which="double"):
    """The journal's column width in mm: 'single', 'double' (the widest) or a middle width."""
    w = (_SPEC or JOURNALS["epsl"])["widths_mm"]
    return {"single": w[0], "double": w[-1]}.get(which, w[len(w) // 2])


def panel_letters(axes, x=0.0, y=1.03, upper=None, size=None, va="bottom", labels=None):
    """Bold panel letters in the journal's style: GSA capitals, Nature and Elsevier lower case.

    ``labels``, one per panel, adds a short data identifier after the letter in normal weight (the
    subset a panel shows, such as a mapping-status category or a deposit class), never a title: the
    description stays in the caption."""
    spec = _SPEC or {}
    upper = spec.get("upper", False) if upper is None else upper
    size = size or spec.get("letter_pt", 8.0)
    letters = string.ascii_uppercase if upper else string.ascii_lowercase
    labels = list(labels) if labels is not None else []
    for i, (ax, letter) in enumerate(zip(axes, letters)):
        t = ax.text(x, y, letter, transform=ax.transAxes, fontsize=size, fontweight="bold",
                    va=va)
        t.set_gid("panel-letter")
        if i < len(labels) and labels[i]:
            lab = ax.annotate(labels[i], xy=(1, 0), xycoords=t, xytext=(3, 0), textcoords="offset points",
                              fontsize=size, va="bottom", ha="left")
            lab.set_gid("panel-label")


def _drawn_texts(fig):
    """Text artists that are actually drawn: visible, non-empty, on visible axes and, for
    tick labels, at ticks inside the view interval."""
    out = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = sorted(lim)
            # minor tick labels too: a log axis labels its minor ticks by default, and those clash
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if lo - 1e-9 * abs(hi - lo) <= tick.get_loc() <= hi + 1e-9 * abs(hi - lo):
                    out += [t for t in (tick.label1, tick.label2) if t.get_visible() and t.get_text().strip()]
            out += [t for t in (axis.label, axis.get_offset_text()) if t.get_visible() and t.get_text().strip()]
        out += [t for t in ax.texts if t.get_visible() and t.get_text().strip()]
        out += [t for t in (ax.title, ax._left_title, ax._right_title)
                if t.get_visible() and t.get_text().strip()]
        leg = ax.get_legend()
        if leg:
            out += [t for t in leg.get_texts() if t.get_text().strip()]
        # latitude and longitude labels of cartopy maps live on the gridliner, not the axes
        for gl in [c for c in ax.get_children() if type(c).__name__ == "Gridliner"]:
            out += [t for t in getattr(gl, "label_artists", [])
                    if hasattr(t, "get_text") and t.get_visible() and t.get_text().strip()]
    out += [t for t in fig.texts if t.get_visible() and t.get_text().strip()]
    for leg in getattr(fig, "legends", []):          # figure-level legends sit outside the axes
        out += [t for t in leg.get_texts() if t.get_text().strip()]
    return out


def _legends_over_data(fig, step_px=2.0, pad_px=1.0):
    """Legends drawn inside an axes that sit on top of that axes' data.

    Text crossing text is one failure; a legend crossing a curve is the other, and it is the
    one that recurs, because the empty corner of a panel depends on where the data happen to
    fall. Every visible line and marker set is sampled in display space - segments densified,
    so that a line crossing the legend between two vertices is still caught - and any sample
    inside a legend box is a clash.
    """
    import numpy as np
    from matplotlib.collections import LineCollection, PathCollection

    r = fig.canvas.get_renderer()
    clashes = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if not ax.get_visible() or leg is None or not leg.get_visible():
            continue
        box = leg.get_window_extent(r).padded(-pad_px)
        clip = ax.get_window_extent(r)
        samples = []
        for ln in ax.lines:
            if not ln.get_visible():
                continue
            xy = np.column_stack(ln.get_data(orig=False)).astype(float)
            xy = xy[np.all(np.isfinite(xy), axis=1)]
            if len(xy) == 0:
                continue
            pts = ln.get_transform().transform(xy)
            if len(pts) > 1 and ln.get_linestyle() not in ("None", "none", " ", ""):
                seg = np.diff(pts, axis=0)
                n = np.maximum(1, (np.hypot(seg[:, 0], seg[:, 1]) / step_px).astype(int))
                pts = np.vstack([pts[i] + np.outer(np.linspace(0, 1, k, endpoint=False), seg[i])
                                 for i, k in enumerate(n)] + [pts[-1:]])
            samples.append((ln.get_label(), pts))
        for col in ax.collections:
            if not col.get_visible():
                continue
            if isinstance(col, PathCollection) and len(col.get_offsets()):
                samples.append((col.get_label(), col.get_offset_transform().transform(col.get_offsets())))
            elif isinstance(col, LineCollection):
                for seg in col.get_segments():
                    if len(seg) > 1:
                        samples.append((col.get_label(), col.get_transform().transform(np.asarray(seg))))
        for label, pts in samples:
            inside_axes = ((pts[:, 0] >= clip.x0) & (pts[:, 0] <= clip.x1)
                           & (pts[:, 1] >= clip.y0) & (pts[:, 1] <= clip.y1))
            hit = inside_axes & (pts[:, 0] > box.x0) & (pts[:, 0] < box.x1) \
                & (pts[:, 1] > box.y0) & (pts[:, 1] < box.y1)
            if hit.any():
                clashes.append((ax.get_label() or "axes", str(label)))
    return clashes


def _text_outside_boxes(fig, pad_px=0.5):
    """Text anchored inside a drawn box (a rectangle or rounded box) that spills out of it.

    Diagram labels are placed inside boxes by hand, and a label that grows past its box is the
    commonest way a figure goes wrong without any two pieces of text touching.
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle
    r = fig.canvas.get_renderer()
    patches = []
    for ax in fig.axes:
        for p in ax.patches:
            if isinstance(p, (Rectangle, FancyBboxPatch)) and p.get_visible():
                patches.append(p.get_window_extent(r))
    out = []
    for t in _drawn_texts(fig):
        if not t.get_text().strip():
            continue
        tb = t.get_window_extent(r)
        anchor = t.get_transform().transform(t.get_position())
        inside = [b for b in patches if b.contains(*anchor)]
        if not inside:
            continue
        host = min(inside, key=lambda b: b.width * b.height)
        if not (host.x0 - pad_px <= tb.x0 and tb.x1 <= host.x1 + pad_px
                and host.y0 - pad_px <= tb.y0 and tb.y1 <= host.y1 + pad_px):
            out.append(t.get_text().replace("\n", " ")[:60])
    return out


def print_scale(fig, renderer=None):
    """Factor by which the figure is reduced when printed: 1 unless it is wider than the widest
    column of the journal, in which case it is reduced to that width."""
    r = renderer or fig.canvas.get_renderer()
    tight = fig.get_tightbbox(r).width
    # a tight save adds its margin; a figure whose artwork stays inside its own frame is at
    # most as wide as that frame, however it is saved
    w_mm = min(tight + 2 * mpl.rcParams["savefig.pad_inches"],
               max(tight, fig.get_figwidth())) * 25.4
    return min(1.0, width_mm("double") / w_mm), w_mm


def _check_print_size(fig, texts, r):
    """Lettering at the size it will be printed, against the journal's rules."""
    spec = _SPEC or dict(name="unspecified journal", min_pt=MIN_FONT_PT,
                         min_script_pt=MIN_FONT_PT, max_pt=None)
    scale, w_mm = print_scale(fig, r) if _SPEC else (1.0, None)
    small, script, large = [], [], []
    for t in texts:
        size = t.get_fontsize() * scale
        text = t.get_text()
        if size < spec["min_pt"] - 1e-6:
            small.append(f"{text[:40]!r} {size:.1f} pt")
        if "$" in text and ("^" in text or "_" in text) and size * 0.7 < spec["min_script_pt"]:
            script.append(text[:40])
        if (spec.get("max_pt") and t.get_gid() != "panel-letter"
                and size > spec["max_pt"] + 1e-6):
            large.append(f"{text[:40]!r} {size:.1f} pt")
    where = f" at print size (figure {w_mm:.0f} mm wide, scaled by {scale:.2f})" if w_mm else ""
    if small:
        raise ValueError(f"{spec['name']}: lettering below {spec['min_pt']} pt{where}: {small[:5]}")
    if script:
        raise ValueError(f"{spec['name']}: mathtext sub/superscripts print below "
                         f"{spec['min_script_pt']} pt; write units as km/s, t/m² instead: {script[:5]}")
    if large:
        raise ValueError(f"{spec['name']}: lettering above {spec['max_pt']} pt{where}: {large[:5]}")


# ---- colour accessibility --------------------------------------------------------------------
# Continuous fields use perceptually uniform maps; these rainbow-type maps distort data and are
# unreadable with colour-vision deficiency, and are refused wherever a figure uses them.
RAINBOW_CMAPS = {"jet", "rainbow", "gist_rainbow", "hsv", "turbo", "nipy_spectral", "gist_ncar",
                 "Spectral", "prism", "flag", "gist_stern", "brg", "CMRmap", "cubehelix_r"}
MIN_GREY_DL = 15.0      # L* separation two keyed entries of one style need in greyscale
MIN_CVD_DE = 12.0       # CIELAB separation they need under each simulated deficiency
# Machado et al. (2009) matrices, severity 1, applied to linear RGB
_CVD = {
    "protanopia": ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216),
                   (-0.003882, -0.048116, 1.051998)),
    "deuteranopia": ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413),
                     (-0.011820, 0.042940, 0.968881)),
    "tritanopia": ((1.255528, -0.076749, -0.178779), (-0.078411, 0.930809, 0.147602),
                   (0.004733, 0.691367, 0.303900)),
}


def _linear(rgb):
    import numpy as np
    c = np.asarray(rgb, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lab_linear(lin):
    import numpy as np
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = m @ np.clip(lin, 0, 1) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def colour_views(colour, alpha=1.0):
    """CIELAB of a colour as printed on white, in normal vision and under each simulated
    deficiency; a partly transparent colour is first blended with the white page."""
    import numpy as np
    from matplotlib.colors import to_rgba
    r, g, b, a = to_rgba(colour)
    a = a * (1.0 if alpha is None else alpha)
    lin = _linear(np.array([r, g, b]) * a + (1 - a))
    out = {"normal": _lab_linear(lin)}
    for name, m in _CVD.items():
        out[name] = _lab_linear(np.array(m) @ lin)
    return out


def _key_entries(fig):
    """(text, kind, style, colour, alpha) for every entry of every key in the figure."""
    from matplotlib.collections import PathCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    legends += list(getattr(fig, "legends", []))
    out = []
    for leg in legends:
        handles = getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", [])
        for h, t in zip(handles, leg.get_texts()):
            label = t.get_text()
            if isinstance(h, Line2D):
                ls = h.get_linestyle()
                marker = h.get_marker() if h.get_marker() not in (None, "None", "", " ") else None
                if marker is not None and str(h.get_markerfacecolor()).lower() == "none":
                    marker = f"{marker} open"
                colour = h.get_color() if ls not in ("None", "none", "", " ") else (
                    h.get_markerfacecolor() if h.get_markerfacecolor() not in ("none", "None")
                    else h.get_markeredgecolor())
                # the dash pattern itself: every custom pattern reports its style as "--"
                dash = getattr(h, "_unscaled_dash_pattern", None)
                dash = (None if ls in ("None", "none", "", " ") else
                        str(dash[1]) if dash is not None and dash[1] is not None else ls)
                style = ("line", dash, marker, round(float(h.get_linewidth()) / 0.6))
                out.append((label, style, colour, h.get_alpha()))
            elif isinstance(h, Patch):
                face = h.get_facecolor()
                colour = face if face[3] > 0 else h.get_edgecolor()
                # an outline and a filled patch differ in form, as do solid and dashed outlines
                out.append((label, ("patch", h.get_hatch() or None, face[3] > 0, str(h.get_linestyle())), colour, None))
            elif isinstance(h, PathCollection):
                fc = h.get_facecolor()
                colour = fc[0] if len(fc) and fc[0][3] > 0 else h.get_edgecolor()[0]
                path = h.get_paths()[0] if len(h.get_paths()) else None
                shape = None if path is None else len(path.vertices)
                out.append((label, ("marker", shape), colour, None))
    return out


def _check_colour(fig):
    """Colour rules shared by EPSL, Nature and GSA: no rainbow maps, and every pair of entries
    in a key that looks alike in form (same line style, marker or hatch) must stay apart in
    greyscale and under protanopia, deuteranopia and tritanopia."""
    import numpy as np
    bad = []
    for ax in fig.axes:
        for art in list(ax.images) + list(ax.collections):
            if hasattr(art, "get_cmap") and getattr(art, "get_array", lambda: None)() is not None:
                name = art.get_cmap().name.removesuffix("_r")
                if name in RAINBOW_CMAPS:
                    bad.append(f"rainbow colour map '{art.get_cmap().name}'")
    if bad:
        raise ValueError("; ".join(sorted(set(bad))) + ": use a perceptually uniform map")
    entries = _key_entries(fig)
    clashes = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            (ti, si, ci, ai), (tj, sj, cj, aj) = entries[i], entries[j]
            if si != sj or ti == tj:
                continue
            vi, vj = colour_views(ci, ai), colour_views(cj, aj)
            dl = abs(vi["normal"][0] - vj["normal"][0])
            de = min(float(np.linalg.norm(vi[k] - vj[k])) for k in _CVD)
            if dl < MIN_GREY_DL or de < MIN_CVD_DE:
                clashes.append(f"'{ti}' / '{tj}' (grey dL {dl:.0f}, colour-blind dE {de:.0f})")
    if clashes:
        raise ValueError("key entries told apart by colour alone and not safely: "
                         + "; ".join(clashes[:6])
                         + ". Give them different line styles, markers or hatching, or colours "
                           "that differ in lightness")


def assert_no_text_overlap(fig, pad_px=1.0):
    """Raise if any two visible text elements overlap, any text breaks the journal's size rules
    at print size, any legend inside an axes sits on that axes' data, any label spills out of
    the box that holds it, or colour breaks the accessibility rules (``_check_colour``)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    if _SPEC and _SPEC.get("line_pt"):
        _enforce_line_weights(fig, r)
        fig.canvas.draw()
    texts = _drawn_texts(fig)
    _check_print_size(fig, texts, r)
    boxes = [(t, t.get_window_extent(r).expanded(1, 1).padded(pad_px)) for t in texts]
    clashes = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                clashes.append((boxes[i][0].get_text(), boxes[j][0].get_text()))
    if clashes:
        raise ValueError(f"{len(clashes)} overlapping text pairs, e.g. {clashes[:5]}")
    on_data = _legends_over_data(fig)
    if on_data:
        raise ValueError(f"legend drawn over data in {len(on_data)} places, e.g. {on_data[:5]}")
    spill = _text_outside_boxes(fig)
    if spill:
        raise ValueError(f"text runs outside its box in {len(spill)} places, e.g. {spill[:5]}")
    _check_colour(fig)


def drawn_texts(fig):
    """Every text the figure draws (tick, graticule and legend labels included), to pass as
    ``bbox_extra_artists`` to ``savefig(..., bbox_inches="tight")``: the tight bounding box of a
    cartopy axes leaves out its graticule labels, which are then cut at the page edge."""
    fig.canvas.draw()
    return _drawn_texts(fig)
