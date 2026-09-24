"""Vector illustrations for Figure 1b: the skin cross-section and the sensor.

Both are drawn in inch coordinates with an (ox, oy) offset, so the same
functions draw into the composite figure and into standalone assets.

Skin (1.28 x 0.82 in, the size of the former placeholder)
    epidermis over dermis with rete ridges, a coiled eccrine gland, a straight
    dermal duct and a helical epidermal duct opening at a pore, a capillary
    looping beneath the gland, and the patch whose inlet sits over the pore.
    The capillary is labelled C_source and the gland H_bio, so the drawing is
    the physical counterpart of C_in = H_bio{C_source} directly below it.

Sensor (1.78 x 0.86 in)
    top view of a planar three-electrode cell with traces and contact pads,
    and a zoomed cross-section of the working electrode: substrate, WE,
    selective layer and the channel fluid (C_f). The bracket on the selective
    layer marks where the first-order lag tau_s arises (reaction-diffusion).

Colours follow Figure 1b: transport teal, sensing purple, biology green, and
sweat drawn in the "young" end of the sample-age colour map.
Lettering >= 5.2 pt (math subscripts 4.9 pt); strokes >= 0.6 pt.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, Wedge, Circle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "revision_2026-09-19" / "figure1b_illustrations"

SKIN_SIZE = (1.28, 0.82)
SENSOR_SIZE = (1.78, 0.86)
SKIN_OUTLET_Y = 0.70 + 0.05     # centre of the patch lumen, local in

BLACK, RULE, GRAY = "#202020", "#4A4A4A", "#8A8A8A"
BIO, DEV, SEN = "#3A7D44", "#1B6F87", "#5B3A99"
SWEAT = "#DDF0F3"                 # young end of the age map
PARCEL = "#5E97A6"                # mid-age parcel, as in the channel

EPI, EPI_SC, DERMIS = "#EFCDBB", "#E0AE97", "#F7E6DC"
SKIN_LINE, SKIN_TEXT = "#B9826C", "#6B4A3E"
VESSEL_WALL, VESSEL_LUMEN, VESSEL_TEXT = "#B24A4A", "#EBB1AC", "#A33B3B"
GLAND_WALL = BIO
PATCH_FILL = "#E8EEF2"

WE_FILL, CE_FILL, RE_FILL = "#8E7CC3", "#C9C0E4", "#9A9A9A"
TRACE, SUBSTRATE, LAYER, FLUID = "#BDBDBD", "#D5D5D5", "#E4DEF3", "#E3F1F4"

FS_LABEL, FS_MATH, FS_SMALL = 5.5, 7.0, 5.2

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ------------------------------------------------------------------ helpers
def _bezier(p0, p1, p2, p3, n=160):
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = ((1 - t) ** 3 * np.array(p0) + 3 * (1 - t) ** 2 * t * np.array(p1)
           + 3 * (1 - t) * t ** 2 * np.array(p2) + t ** 3 * np.array(p3))
    return pts[:, 0], pts[:, 1]


def _tube(ax, x, y, wall, lumen, w_out, w_in, z, clip=None):
    """A hollow tube: a wide wall stroke with a narrower lumen stroke on top."""
    for colour, lw, dz in ((wall, w_out, 0.0), (lumen, w_in, 0.1)):
        ln = Line2D(x, y, color=colour, lw=lw, solid_capstyle="round",
                    solid_joinstyle="round", zorder=z + dz)
        ax.add_line(ln)
        if clip is not None:
            ln.set_clip_path(clip)


def _text(ax, x, y, s, size=FS_LABEL, colour=BLACK, **kw):
    kw.setdefault("ha", "left")
    kw.setdefault("va", "center")
    return ax.text(x, y, s, fontsize=size, color=colour, zorder=20, **kw)


# --------------------------------------------------------------------- skin
def draw_skin(ax, ox=0.0, oy=0.0):
    X = lambda v: ox + np.asarray(v)          # noqa: E731
    Y = lambda v: oy + np.asarray(v)          # noqa: E731

    x0, x1, y0, surf = 0.02, 1.26, 0.02, 0.70
    block = FancyBboxPatch((X(x0), Y(y0)), x1 - x0, surf - y0,
                           boxstyle="round,pad=0,rounding_size=0.035",
                           fc=DERMIS, ec="none", zorder=1)
    ax.add_patch(block)
    clip = block

    # epidermis with rete ridges, then stratum corneum
    xs = np.linspace(x0, x1, 300)
    boundary = 0.52 + 0.022 * np.sin(2 * np.pi * xs / 0.22)
    epi = Polygon(np.column_stack([X(np.r_[xs, xs[::-1]]),
                                   Y(np.r_[boundary, np.full_like(xs, surf)])]),
                  closed=True, fc=EPI, ec="none", zorder=2)
    ax.add_patch(epi)
    epi.set_clip_path(clip)
    sc = Rectangle((X(x0), Y(surf - 0.04)), x1 - x0, 0.04, fc=EPI_SC,
                   ec="none", zorder=3)
    ax.add_patch(sc)
    sc.set_clip_path(clip)
    for x, y, c, lw in ((xs, boundary, SKIN_LINE, 0.6),):
        ln = Line2D(X(x), Y(y), color=c, lw=lw, alpha=0.55, zorder=3)
        ax.add_line(ln)
        ln.set_clip_path(clip)
    ax.add_line(Line2D(X([x0 + 0.02, x1 - 0.02]), Y([surf, surf]),
                       color=SKIN_LINE, lw=0.7, zorder=4))

    # capillary looping beneath the gland
    ax1, ay1 = _bezier((x0 - 0.02, 0.13), (0.28, 0.02), (0.60, 0.00), (0.76, 0.13))
    ax2, ay2 = _bezier((0.76, 0.13), (0.88, 0.23), (1.08, 0.20), (x1 + 0.02, 0.26))
    _tube(ax, X(np.r_[ax1, ax2]), Y(np.r_[ay1, ay2]), VESSEL_WALL,
          VESSEL_LUMEN, 3.6, 2.2, z=5, clip=clip)

    # coiled secretory gland, dermal duct and helical epidermal duct
    # a horizontal helix of three turns reads unambiguously as a coiled tube
    cx, cy, turns = 0.45, 0.185, 3.0
    t = np.linspace(0.0, 2.0 * np.pi * turns, 600)
    gx = cx - 0.09 + 0.18 * t / t[-1] - 0.034 * np.cos(t)
    gy = cy + 0.046 * np.sin(t)
    pore_x = 0.60
    dx1, dy1 = _bezier((gx[-1], gy[-1]), (gx[-1] + 0.035, gy[-1] + 0.06),
                       (pore_x, 0.38), (pore_x, 0.50))
    yh = np.linspace(0.50, surf, 120)
    hx = pore_x + 0.022 * np.sin(2 * np.pi * (yh - 0.50) / 0.065)
    hx[-12:] = np.linspace(hx[-12], pore_x, 12)       # open straight at the pore
    _tube(ax, X(np.r_[gx, dx1, hx]), Y(np.r_[gy, dy1, yh]), GLAND_WALL,
          SWEAT, 2.6, 1.4, z=6, clip=clip)

    # patch on the skin surface, inlet over the pore, lumen running right
    p0, p1, pb, pt = 0.40, SKIN_SIZE[0], surf, surf + 0.105
    lb, lt = surf + 0.030, surf + 0.070
    in0, in1 = pore_x - 0.028, pore_x + 0.028
    ax.add_patch(Rectangle((X(p0), Y(pb)), p1 - p0, pt - pb, fc=PATCH_FILL,
                           ec="none", zorder=7))
    ax.add_patch(Rectangle((X(in0), Y(lb)), p1 - in0, lt - lb, fc=SWEAT,
                           ec="none", zorder=8))
    ax.add_patch(Rectangle((X(in0), Y(pb)), in1 - in0, lb - pb + 0.001,
                           fc=SWEAT, ec="none", zorder=8))
    for xs_, ys_ in (([p0, p1], [pt, pt]),                    # top face
                     ([p0, p0], [pb, pt]),                    # left end
                     ([p0, in0], [pb, pb]), ([in1, p1], [pb, pb]),  # base, gap
                     ([in0, in0], [pb, lt]), ([in0, p1], [lt, lt]),  # lumen
                     ([in1, in1], [pb, lb]), ([in1, p1], [lb, lb])):
        ax.add_line(Line2D(X(xs_), Y(ys_), color=RULE if ys_[0] in (pt,)
                           or xs_[0] == p0 else DEV,
                           lw=0.6, solid_capstyle="butt", zorder=9))
    for cxv in (0.86, 1.00, 1.14):                           # flow direction
        ymid = 0.5 * (lb + lt)
        ax.add_line(Line2D(X([cxv - 0.012, cxv, cxv - 0.012]),
                           Y([ymid + 0.012, ymid, ymid - 0.012]),
                           color=DEV, lw=0.7, solid_joinstyle="miter",
                           zorder=10))

    # labels
    _text(ax, X(0.05), Y(0.755), "patch", colour=RULE)
    _text(ax, X(0.06), Y(0.588), "epidermis", colour=SKIN_TEXT)
    _text(ax, X(0.06), Y(0.44), "dermis", colour=SKIN_TEXT)
    _text(ax, X(0.05), Y(0.305), r"$C_{\mathrm{source}}$", size=FS_MATH,
          colour=VESSEL_TEXT)
    _text(ax, X(0.69), Y(0.345), r"$H_{\mathrm{bio}}$", size=FS_MATH,
          colour=BIO)
    return {"pore_xy": (ox + pore_x, oy + surf),
            "patch_outlet_xy": (ox + p1, oy + 0.5 * (lb + lt))}


# ------------------------------------------------------------------- sensor
def draw_sensor(ax, ox=0.0, oy=0.0):
    X = lambda v: ox + np.asarray(v)          # noqa: E731
    Y = lambda v: oy + np.asarray(v)          # noqa: E731

    # ---- top view ----------------------------------------------------------
    ax.add_patch(FancyBboxPatch((X(0.05), Y(0.12)), 0.74, 0.72,
                                boxstyle="round,pad=0,rounding_size=0.04",
                                fc="white", ec=RULE, lw=0.7, zorder=1))
    cx, cy = 0.42, 0.575
    # silver traces start beneath each electrode and end on separate pads
    trace_w = 0.03 * 72.0                           # 0.03 in, in points
    a_re = np.deg2rad(242.0)
    re_x = cx + 0.185 * np.cos(a_re)
    re_y = cy + 0.185 * np.sin(a_re)
    traces = (
        ([cx, cx], [cy - 0.105, 0.20]),                           # WE
        ([0.59, 0.59], [cy + 0.07, 0.20]),                        # CE
        ([re_x, re_x, 0.26, 0.26], [re_y, 0.33, 0.27, 0.20]),     # RE
    )
    for xs_, ys_ in traces:
        ax.add_line(Line2D(X(xs_), Y(ys_), color=TRACE, lw=trace_w,
                           solid_capstyle="butt", solid_joinstyle="round",
                           zorder=2))
    for px in (0.26, cx, 0.59):
        ax.add_patch(Rectangle((X(px - 0.04), Y(0.15)), 0.08, 0.06,
                               fc=TRACE, ec=RULE, lw=0.6, zorder=2.5))
    ax.add_patch(Wedge((X(cx), Y(cy)), 0.21, 22, 158, width=0.05, fc=CE_FILL,
                       ec=SEN, lw=0.6, zorder=3))
    ax.add_patch(Wedge((X(cx), Y(cy)), 0.21, 198, 250, width=0.05, fc=RE_FILL,
                       ec=RULE, lw=0.6, zorder=3))
    ax.add_patch(Circle((X(cx), Y(cy)), 0.11, fc=WE_FILL, ec=SEN, lw=0.7,
                        zorder=4))
    _text(ax, X(cx), Y(cy), "WE", size=6.0, colour="white", ha="center",
          fontweight="bold")
    _text(ax, X(0.09), Y(0.765), "CE", size=6.0, colour=SEN)
    _text(ax, X(0.10), Y(0.405), "RE", size=6.0, colour=RULE)
    _text(ax, X(0.42), Y(0.035), "top view", size=FS_SMALL, colour=GRAY,
          ha="center")

    # ---- zoomed cross-section of the working electrode ----------------------
    f0, f1, g0, g1 = 1.00, 1.58, 0.14, 0.82
    sub, we, lay = (g0, 0.24), (0.24, 0.32), (0.32, 0.45)
    ax.add_patch(Rectangle((X(f0), Y(sub[0])), f1 - f0, sub[1] - sub[0],
                           fc=SUBSTRATE, ec="none", zorder=2))
    ax.add_patch(Rectangle((X(f0), Y(we[0])), f1 - f0, we[1] - we[0],
                           fc=WE_FILL, ec="none", zorder=2))
    ax.add_patch(Rectangle((X(f0), Y(lay[0])), f1 - f0, lay[1] - lay[0],
                           fc=LAYER, ec="none", zorder=2))
    ax.add_patch(Rectangle((X(f0), Y(lay[1])), f1 - f0, g1 - lay[1],
                           fc=FLUID, ec="none", zorder=2))
    ax.add_patch(Rectangle((X(f0), Y(g0)), f1 - f0, g1 - g0, fill=False,
                           ec=RULE, lw=0.7, zorder=6))
    for yv in (we[0], we[1], lay[1]):
        ax.add_line(Line2D(X([f0, f1]), Y([yv, yv]), color=RULE, lw=0.6,
                           alpha=0.6, zorder=5))
    # analyte: dense in the fluid, sparser across the selective layer
    rng = np.random.default_rng(11)
    gx_, gy_ = np.meshgrid(np.linspace(f0 + 0.08, f1 - 0.07, 5),
                           np.linspace(lay[1] + 0.06, g1 - 0.17, 3))
    n_col = gx_.shape[1]
    for i, (x, y) in enumerate(zip(gx_.ravel(), gy_.ravel())):
        row, col = divmod(i, n_col)
        if row == gx_.shape[0] - 1 and col < 2:     # keep clear of the C_f label
            continue
        x += rng.uniform(-0.025, 0.025)
        y += rng.uniform(-0.02, 0.02) + (0.03 if i % 2 else 0.0)
        ax.add_patch(Circle((X(x), Y(y)), 0.016, fc=PARCEL, ec="white",
                            lw=0.6, zorder=7))
    _text(ax, X(f0 + 0.04), Y(g1 - 0.09), r"$C_{\mathrm{f}}$", size=FS_MATH,
          colour=DEV)
    _text(ax, X(f0 + 0.04), Y(0.385), "selective layer", size=FS_SMALL,
          colour=SEN)
    _text(ax, X(f0 + 0.04), Y(0.28), "WE", size=FS_SMALL, colour="white",
          fontweight="bold")
    _text(ax, X(0.5 * (f0 + f1)), Y(0.035), "cross-section", size=FS_SMALL,
          colour=GRAY, ha="center")
    # tau_s bracket on the selective layer
    bx = f1 + 0.03
    ax.add_line(Line2D(X([bx - 0.015, bx, bx, bx - 0.015]),
                       Y([we[1], we[1], lay[1], lay[1]]), color=SEN, lw=0.7,
                       zorder=8))
    _text(ax, X(bx + 0.02), Y(0.5 * (we[1] + lay[1])), r"$\tau_{\mathrm{s}}$",
          size=FS_MATH, colour=SEN)
    # zoom lines from the WE disc to the section
    for ang, corner in ((60, (f0, g1)), (-60, (f0, g0))):
        a = np.deg2rad(ang)
        ax.add_line(Line2D(X([cx + 0.11 * np.cos(a), corner[0]]),
                           Y([cy + 0.11 * np.sin(a), corner[1]]), color=GRAY,
                           lw=0.6, ls=(0, (2, 1.6)), zorder=0.5))
    return {}


# ------------------------------------------------------------------- export
def _canvas(size):
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, size[0])
    ax.set_ylim(0, size[1])
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def export(folder: Path = OUT) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    written = {}
    for stem, size, fn in (("fig1b_skin_illustration", SKIN_SIZE, draw_skin),
                           ("fig1b_sensor_illustration", SENSOR_SIZE,
                            draw_sensor)):
        fig, ax = _canvas(size)
        info = fn(ax)
        for ext in ("pdf", "svg"):
            fig.savefig(folder / (stem + "." + ext), transparent=True)
        fig.savefig(folder / (stem + ".png"), dpi=600, transparent=True)
        # a white-background PNG for quick viewing
        fig.savefig(folder / (stem + "_preview.png"), dpi=600,
                    facecolor="white")
        plt.close(fig)
        written[stem] = {"size_in": list(size), **{k: list(v) for k, v in info.items()}}
    return written


if __name__ == "__main__":
    for k, v in export().items():
        print(k, v)
