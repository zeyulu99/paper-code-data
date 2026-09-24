"""Figure 3 - sampling topology fixes what a readout can mean.

Three rows, five columns (one per sampling archetype), one computation:

  a  schematic of each topology; fluid parcels are coloured by sample age on
     the absolute scale used throughout (young -> old), and the purple bar
     marks where the readout is taken;
  b  the sample-age weighting E(theta) of one readout, declared (not fitted)
     and plotted with theta = 0 (now) on the left, as in Figure 1b;
  c  what each topology can report for the SAME input C_in(t): the input is
     convolved with (or windowed by) the weighting in row b.

Row c replaces the former text rows. The "evidence required" list belongs to
Table 3 (tab:minimum-reporting); the claim names follow the manuscript's
conclusion (continuously ordered signal, history-dependent distributed
average, cycle-resolved value, ordered window average, time-integrated
endpoint). The former footer ("electronic sampling is not fluid renewal") is
drawn in the fill-measure-clear column: electronic samples between cycles
repeat the held value.

All weightings are schematic declarations, written to CSV with the outputs.
Format: 7.00 in wide; lettering >= 4.5 pt (math subscripts included);
strokes >= 0.6 pt. Backend: Python / matplotlib.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Polygon, Rectangle)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "revision_2026-09-19" / "figure3"
OUT.mkdir(parents=True, exist_ok=True)
STEM = "fig3_topology"

# ------------------------------------------------------------------ style
BLACK, GRAY, RULE, LIGHT = "#202020", "#8A8A8A", "#4A4A4A", "#BDBDBD"
DEV, SEN = "#1B6F87", "#5B3A99"
AGE_CMAP = LinearSegmentedColormap.from_list("age", ["#BFE4EB", "#0E3B4A"])
THETA_MAX = 130.0                         # min, absolute colour scale

# the author's original Figure 3 palette, one colour per topology
TOPO = {
    "through": dict(line="#0030B4", fill="#CCD8FC"),
    "wick": dict(line="#0C5A30", fill="#CCDED2"),
    "fmc": dict(line="#5A36D2", fill="#E2DCF6"),
    "chrono": dict(line="#F2541B", fill="#FCEADE"),
    "accum": dict(line="#202020", fill="#D6D6D6"),
}
USE_ORIGINAL_ROW_A = True
ORIG_A = ROOT / "outputs" / "revision_2026-09-19" / "figure3" / "original_row_a"
FLUID_BG = "#EEF6F8"

FS_COL, FS_ROW, FS_LABEL, FS_TICK, FS_NOTE, FS_MATH = 7.0, 6.5, 6.5, 6.0, 5.6, 7.0
LW_OUT, LW_REF = 1.3, 0.8

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
    "font.size": FS_LABEL,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ----------------------------------------------------------------- layout
FIG_W, FIG_H = 7.00, 5.10
X0_COL, COL_W, COL_GAP = 0.66, 1.216, 0.05
ROW_A = (3.62, 4.62)            # y0, y1 of the schematic row (in)
ROW_B = (2.34, 3.02)            # axes y0, y1
ROW_C = (0.42, 1.52)            # axes y0, y1
AX_PAD_L, AX_PAD_R = 0.28, 0.04
KEYS = ["through", "wick", "fmc", "chrono", "accum"]
TITLES = {
    "through": "Continuous\nthrough-flow",
    "wick": "Distributed\nwick",
    "fmc": "Fill–measure–\nclear",
    "chrono": "Chronological\nreservoirs",
    "accum": "Accumulation\nendpoint",
}
CLAIMS = {
    "through": "Continuously\nordered signal",
    "wick": "History-dependent\ndistributed average",
    "fmc": "Cycle-resolved\nvalue",
    "chrono": "Ordered\nwindow average",
    "accum": "Time-integrated\nendpoint",
}

# ------------------------------------------------------------------ model
DT = 0.1
T_END = 125.0
t = np.arange(0.0, 150.0 + DT, DT)
theta = np.arange(0.0, THETA_MAX + DT, DT)

P_THROUGH = dict(kind="gamma", k=3.0, mean_min=3.0)
P_WICK = dict(kind="gamma", k=1.5, mean_min=20.0)
P_FMC = dict(period_min=15.0, fill_min=10.0, measure_delay_min=2.0,
             f_res=0.15, first_readout_min=12.0)
P_CHRONO = dict(window_min=30.0, n=4, readout_after_min=120.0, shown=3)
P_ACCUM = dict(start_min=0.0, end_min=120.0, flow_trend=0.8)


def c_input(tt):
    """Two events of different size on a small baseline."""
    g = lambda mu, fwhm, a: a * np.exp(-0.5 * ((tt - mu) / (fwhm / 2.355)) ** 2)  # noqa: E731
    return 0.10 + g(20.0, 10.0, 0.90) + g(75.0, 10.0, 0.55)


def gamma_pdf(th, k, mean):
    from math import gamma as G
    scale = mean / k
    y = np.where(th > 0, th ** (k - 1) * np.exp(-th / scale), 0.0)
    y = y / (G(k) * scale ** k)
    return y / np.trapezoid(y, th)


def box(th, a, b, mass=1.0):
    y = ((th >= a) & (th <= b)).astype(float)
    return mass * y / np.trapezoid(y, th)


def weightings():
    fm = P_FMC
    # event_reset_update: C_n = (1 - f) new_n + f C_(n-1), so the cycle that
    # is k periods old carries weight (1 - f) f**k
    f = fm["f_res"]
    w_fmc = np.zeros_like(theta)
    for k_old in range(6):
        a = fm["measure_delay_min"] + k_old * fm["period_min"]
        if a >= THETA_MAX:
            break
        w_fmc += box(theta, a, a + fm["fill_min"], (1.0 - f) * f ** k_old)
    w_fmc /= np.trapezoid(w_fmc, theta)
    ch = P_CHRONO
    windows = []
    for i in range(ch["n"]):                     # W1 collected first = oldest
        start = i * ch["window_min"]             # collection start time
        windows.append((ch["readout_after_min"] - (start + ch["window_min"]),
                        ch["readout_after_min"] - start))
    sel = windows[ch["shown"] - 1]
    w_chrono = box(theta, *sel)
    ac = P_ACCUM
    span = (0.0, ac["end_min"] - ac["start_min"])
    w_accum = box(theta, *span)
    trend = []
    for s in (+1, -1):                            # rising / falling sweat rate
        q = 1.0 + s * ac["flow_trend"] * (0.5 - theta / span[1])
        trend.append(np.where((theta >= span[0]) & (theta <= span[1]), q, 0.0)
                     / np.trapezoid(np.where((theta >= span[0]) &
                                             (theta <= span[1]), q, 0.0), theta))
    return {
        "through": gamma_pdf(theta, P_THROUGH["k"], P_THROUGH["mean_min"]),
        "wick": gamma_pdf(theta, P_WICK["k"], P_WICK["mean_min"]),
        "fmc": w_fmc,
        "chrono": w_chrono,
        "accum": w_accum,
        "_chrono_windows": windows,
        "_accum_trend": trend,
    }


def reportable(w):
    """Apply each weighting to the same input."""
    cin = c_input(t)
    hist = lambda tn, ww: float(np.trapezoid(ww * c_input(tn - theta), theta))  # noqa: E731
    out = {"t": t, "c_in": cin}
    # continuous outputs: history integral, with the baseline before t = 0
    for key in ("through", "wick"):
        out[key] = np.array([hist(tn, w[key]) for tn in t])
    fm = P_FMC
    tn = np.arange(fm["first_readout_min"], T_END + 1e-9, fm["period_min"])
    out["fmc_t"] = tn
    out["fmc_v"] = np.array([hist(x, w["fmc"]) for x in tn])
    ch = P_CHRONO
    segs = []
    for i in range(ch["n"]):
        a = i * ch["window_min"]
        b = a + ch["window_min"]
        m = (t >= a) & (t <= b)
        segs.append((a, b, float(np.mean(cin[m]))))
    out["chrono"] = segs
    ac = P_ACCUM
    m = (t >= ac["start_min"]) & (t <= ac["end_min"])
    vals = [float(np.mean(cin[m]))]
    for s in (+1, -1):
        q = 1.0 + s * ac["flow_trend"] * ((t[m] - ac["start_min"])
                                          / (ac["end_min"] - ac["start_min"]) - 0.5)
        vals.append(float(np.sum(q * cin[m]) / np.sum(q)))
    out["accum"] = (ac["start_min"], ac["end_min"], vals[0], min(vals), max(vals))
    return out


# --------------------------------------------------------------- drawing
def col_x(i):
    return X0_COL + i * (COL_W + COL_GAP)


def age_colour(th):
    return AGE_CMAP(float(np.clip(th / THETA_MAX, 0.0, 1.0)))


def parcel(cv, x, y, th, r=0.024, alpha=1.0):
    cv.add_patch(Circle((x, y), r, fc=age_colour(th), ec="white", lw=0.6,
                        alpha=alpha, zorder=6))


def arrow(cv, p0, p1, colour=RULE, lw=0.8, rad=0.0, ms=6, ls="-"):
    cv.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=colour, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=5,
                                 connectionstyle="arc3,rad=%g" % rad))


def electrode(cv, x0, x1, y):
    cv.add_patch(Rectangle((x0, y), x1 - x0, 0.026, fc=SEN, ec="none",
                           zorder=5))


def note(cv, x, y, s, size=FS_NOTE, colour=RULE, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    cv.text(x, y, s, fontsize=size, color=colour, zorder=8, **kw)


def draw_through(cv, ox, oy, rng):
    yb, yt, x0, x1 = oy + 0.36, oy + 0.62, ox + 0.14, ox + 1.10
    cv.add_patch(Rectangle((x0, yb), x1 - x0, yt - yb, fc=FLUID_BG, ec="none",
                           zorder=1))
    for y in (yb, yt):
        cv.plot([x0, x1], [y, y], color=RULE, lw=0.9, zorder=3)
    zx0, zx1 = ox + 0.50, ox + 0.80
    cv.add_patch(Rectangle((zx0, yb - 0.05), zx1 - zx0, yt - yb + 0.10,
                           fill=False, ec=DEV, lw=0.8, ls=(0, (3, 2)), zorder=4))
    note(cv, 0.5 * (zx0 + zx1), yt + 0.14, r"$V_{\mathrm{eff}}$", size=FS_MATH,
         colour=DEV)
    electrode(cv, zx0 + 0.05, zx1 - 0.05, yb)
    for i, x in enumerate(np.linspace(x0 + 0.06, x1 - 0.06, 11)):
        frac = (x - x0) / (x1 - x0)
        th = max(0.0, 6.0 * frac + rng.normal(0, 1.0))
        parcel(cv, x, yb + 0.07 + ((i * 0.41) % 1.0) * (yt - yb - 0.14), th)
    arrow(cv, (ox + 0.00, 0.5 * (yb + yt)), (x0 - 0.02, 0.5 * (yb + yt)))
    arrow(cv, (x1 + 0.02, 0.5 * (yb + yt)), (ox + 1.20, 0.5 * (yb + yt)))


def draw_wick(cv, ox, oy, rng):
    yb, yt, x0, x1, front = oy + 0.36, oy + 0.62, ox + 0.16, ox + 1.14, ox + 0.96
    cv.add_patch(Rectangle((x0, yb), front - x0, yt - yb, fc=FLUID_BG,
                           ec="none", zorder=1))
    cv.add_patch(Rectangle((x0, yb), x1 - x0, yt - yb, fill=False, ec=RULE,
                           lw=0.8, zorder=3))
    # fibre texture
    for x in np.arange(x0 + 0.03, x1, 0.06):
        cv.plot([x, x + 0.03], [yb + 0.02, yt - 0.02], color=LIGHT, lw=0.6,
                zorder=2)
    cv.plot([front, front], [yb - 0.04, yt + 0.04], color=RULE, lw=0.8,
            ls=(0, (2.5, 1.8)), zorder=4)
    note(cv, front, yt + 0.13, "wetting\nfront", linespacing=1.2)
    electrode(cv, ox + 0.46, ox + 0.70, yb)
    for i, x in enumerate(np.linspace(x0 + 0.05, front - 0.05, 12)):
        frac = (x - x0) / (front - x0)
        th = float(np.clip(10 + 70 * frac + rng.normal(0, 22), 1, THETA_MAX))
        parcel(cv, x, yb + 0.07 + ((i * 0.37) % 1.0) * (yt - yb - 0.14), th)
    for y in np.linspace(yb + 0.06, yt - 0.06, 3):
        arrow(cv, (ox + 0.00, y), (x0 - 0.02, y), ms=5, lw=0.7)
    note(cv, ox + 0.06, yb - 0.10, "skin", ha="center")


def cup(cv, x0, y0, w, h, level, parcels, rng, electrode_on=False):
    cv.add_patch(Rectangle((x0, y0), w, h * level, fc=FLUID_BG, ec="none",
                           zorder=1))
    cv.plot([x0, x0, x0 + w, x0 + w], [y0 + h, y0, y0, y0 + h], color=RULE,
            lw=0.9, zorder=3, solid_joinstyle="miter")
    if electrode_on:
        electrode(cv, x0 + 0.05, x0 + w - 0.05, y0)
    for th, (fx, fy) in parcels:
        parcel(cv, x0 + fx * w, y0 + fy * h, th, r=0.021)


def draw_fmc(cv, ox, oy, rng):
    w, h, y0 = 0.28, 0.36, oy + 0.30
    xs = [ox + 0.06, ox + 0.46, ox + 0.86]
    young = lambda: float(rng.uniform(2, 12))  # noqa: E731
    fill = [(young(), (0.25, 0.30)), (young(), (0.65, 0.38)),
            (young(), (0.45, 0.52)), (28.0, (0.30, 0.10)), (30.0, (0.72, 0.10))]
    meas = [(young(), (0.22, 0.30)), (young(), (0.55, 0.62)),
            (young(), (0.80, 0.40)), (young(), (0.40, 0.78)),
            (young(), (0.70, 0.80)), (27.0, (0.30, 0.10)), (29.0, (0.74, 0.10))]
    clear = [(28.0, (0.30, 0.10)), (30.0, (0.72, 0.10))]
    cup(cv, xs[0], y0, w, h, 0.62, fill, rng)
    cup(cv, xs[1], y0, w, h, 0.95, meas, rng, electrode_on=True)
    cup(cv, xs[2], y0, w, h, 0.18, clear, rng)
    for x, lab in zip(xs, ("fill", "measure", "clear")):
        note(cv, x + w / 2, y0 + h + 0.09, lab)
    for a, b in ((xs[0] + w, xs[1]), (xs[1] + w, xs[2])):
        arrow(cv, (a + 0.02, y0 + h / 2), (b - 0.02, y0 + h / 2), ms=5, lw=0.7)
    arrow(cv, (xs[2] + w / 2, y0 - 0.04), (xs[0] + w / 2, y0 - 0.04),
          rad=-0.25, ls=(0, (2.5, 1.8)), lw=0.7, ms=5)
    note(cv, xs[2] + w / 2, y0 + 0.19, r"$f_{\mathrm{res}}$", size=FS_MATH)


def draw_chrono(cv, ox, oy, rng):
    w, h, y0 = 0.22, 0.36, oy + 0.30
    ages = [105.0, 75.0, 45.0, 15.0]
    for i, th0 in enumerate(ages):
        x0 = ox + 0.08 + i * 0.28
        ps = [(float(np.clip(th0 + rng.normal(0, 8), 0, THETA_MAX)), (fx, fy))
              for fx, fy in ((0.3, 0.2), (0.7, 0.32), (0.4, 0.5), (0.72, 0.68),
                             (0.3, 0.78))]
        cup(cv, x0, y0, w, h, 0.95, ps, rng, electrode_on=(i == 2))
        note(cv, x0 + w / 2, y0 + h + 0.09, r"$W_{%d}$" % (i + 1), size=FS_MATH)
    arrow(cv, (ox + 0.10, y0 - 0.08), (ox + 1.16, y0 - 0.08), lw=0.7, ms=5)
    note(cv, ox + 0.63, y0 - 0.16, "collection order")


def draw_accum(cv, ox, oy, rng):
    x0, x1, y0, y1 = ox + 0.24, ox + 0.98, oy + 0.30, oy + 0.66
    cv.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.04",
                                fc=FLUID_BG, ec=RULE, lw=0.9, zorder=2))
    for i in range(16):
        fx, fy = rng.uniform(0.08, 0.92), rng.uniform(0.15, 0.85)
        parcel(cv, x0 + fx * (x1 - x0), y0 + fy * (y1 - y0),
               float(rng.uniform(0, 120)), r=0.021)
    arrow(cv, (ox + 0.02, 0.5 * (y0 + y1)), (x0 - 0.02, 0.5 * (y0 + y1)))
    electrode(cv, x1 + 0.05, x1 + 0.17, 0.5 * (y0 + y1) - 0.013)
    note(cv, x1 + 0.11, y1 + 0.09, "readout", size=FS_NOTE)
    note(cv, 0.5 * (x0 + x1), y0 - 0.11,
         "collected over " + r"$t_0$" + chr(8211) + r"$t_1$", size=FS_MATH)


DRAW = {"through": draw_through, "wick": draw_wick, "fmc": draw_fmc,
        "chrono": draw_chrono, "accum": draw_accum}


def gradient_fill(ax, x, y):
    verts = np.column_stack([np.r_[x[0], x, x[-1]], np.r_[0.0, y, 0.0]])
    clip = Polygon(verts, closed=True, fc="none", ec="none",
                   transform=ax.transData)
    ax.add_patch(clip)
    img = ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=AGE_CMAP,
                    extent=(0, THETA_MAX, 0, float(np.max(y)) * 1.001),
                    aspect="auto", origin="lower", interpolation="bilinear",
                    zorder=1)
    img.set_clip_path(clip)


def plot_weighting(ax, key, w, first):
    """E(theta) of one readout, drawn against sample time before the readout.

    x = -theta, so the oldest sample is on the left and the readout ("now")
    on the right, the same direction as row c and as the author's original.
    """
    col = TOPO[key]
    x = -theta
    y = w[key] / np.max(w[key])
    dashed = key == "accum"
    ax.fill_between(x, 0, y, color=col["fill"], lw=0, zorder=1)
    ax.plot(x, y, color=col["line"], lw=0.9 if dashed else 1.0,
            ls=(0, (2.5, 1.8)) if dashed else "-", zorder=3)
    if key == "chrono":
        for i, (a, b) in enumerate(w["_chrono_windows"]):
            shown = i == P_CHRONO["shown"] - 1
            if not shown:
                ax.add_patch(Rectangle((-b, 0), b - a, 1.0, fill=False,
                                       ec=col["line"], lw=0.6, alpha=0.55,
                                       ls=(0, (2, 1.6)), zorder=2))
            ax.text(-0.5 * (a + b), 1.08, r"$W_{%d}$" % (i + 1), ha="center",
                    va="bottom", fontsize=FS_MATH,
                    color=col["line"] if shown else GRAY)
    if key == "fmc":
        fm = P_FMC
        a = fm["measure_delay_min"] + fm["period_min"]
        ax.text(-(a + fm["fill_min"]) - 2.0,
                fm["f_res"] + 0.06, r"$f_{\mathrm{res}}$",
                ha="right", va="bottom", fontsize=FS_MATH, color=col["line"])
    if key == "accum":
        ax.text(-60, 0.50, "?", ha="center", va="center", fontsize=9.0,
                color=col["line"], zorder=5)
        ax.text(-60, 1.08, "shape unknown", ha="center", va="bottom",
                fontsize=FS_NOTE, color=RULE)
    ax.set_xlim(-THETA_MAX, 5)
    ax.set_ylim(0, 1.45)
    ax.set_yticks([0, 0.5, 1.0])
    if first:
        ax.set_ylabel("Relative weight\n(peak = 1)", fontsize=FS_TICK,
                      labelpad=1.5, linespacing=1.2)
    ax.set_xticks([-120, -60, 0])
    ax.set_xticklabels(["\u2212120", "\u221260", "now"])
    ax.tick_params(pad=1.2)
    if first:
        ax.set_xlabel("Time before\nreadout (min)", fontsize=FS_TICK,
                      labelpad=1.0, linespacing=1.2)


def plot_reportable(ax, key, r, first):
    col = TOPO[key]["line"]
    ax.plot(r["t"], r["c_in"], color=GRAY, lw=LW_REF, ls=(0, (3, 2)), zorder=2)
    if key in ("through", "wick"):
        ax.plot(r["t"], r[key], color=col, lw=LW_OUT, zorder=3)
    elif key == "fmc":
        tn, v = r["fmc_t"], r["fmc_v"]
        for i, (x, y) in enumerate(zip(tn, v)):
            nxt = tn[i + 1] if i + 1 < len(tn) else T_END
            es = np.arange(x + 2.5, nxt - 0.5, 2.5)
            ax.plot(es, np.full_like(es, y), ls="none", marker="o", ms=1.4,
                    color=GRAY, zorder=3)
        ax.plot(tn, v, ls="none", marker="o", ms=3.2, mfc=col, mec="white",
                mew=0.6, zorder=4)
        ax.legend(handles=[
            Line2D([], [], ls="none", marker="o", ms=3.2, mfc=col,
                   mec="white", mew=0.6, label="cycle readout"),
            Line2D([], [], ls="none", marker="o", ms=1.6, color=GRAY,
                   label="electronic\nsample")],
            loc="upper right", fontsize=FS_NOTE, frameon=False,
            handlelength=0.8, handletextpad=0.3, labelspacing=0.5,
            borderaxespad=0.1, borderpad=0.1)
    elif key == "chrono":
        for a, b, v in r["chrono"]:
            ax.plot([a, b], [v, v], color=col, lw=LW_OUT + 0.4,
                    solid_capstyle="butt", zorder=3)
            for xe in (a, b):
                ax.plot([xe, xe], [v - 0.04, v + 0.04], color=col, lw=0.7,
                        zorder=3)
    elif key == "accum":
        a, b, v, lo, hi = r["accum"]
        ax.plot([a, b], [v, v], color=col, lw=LW_OUT + 0.4,
                solid_capstyle="butt", zorder=3)
    ax.set_xlim(0, T_END)
    ax.set_ylim(0, 1.12)
    ax.set_xticks([0, 60, 120])
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(pad=1.2)
    if first:
        ax.set_xlabel("Time (min)", fontsize=FS_TICK, labelpad=1.0,
                      loc="left")
        ax.set_ylabel("Concentration (a.u.)", fontsize=FS_TICK, labelpad=1.5)
        # the encoding is the same in every column, so it is keyed once
        ax.legend(handles=[
            Line2D([], [], color=GRAY, lw=LW_REF, ls=(0, (3, 2)),
                   label="input"),
            Line2D([], [], color=col, lw=LW_OUT, label="reported")],
            loc="upper right", fontsize=FS_NOTE, frameon=False,
            handlelength=1.6, handletextpad=0.3, labelspacing=0.3,
            borderaxespad=0.1, borderpad=0.1)


def draw_original_row_a(cv, key, ox):
    """Place the author's original row-a artwork, centred in its cell."""
    im = plt.imread(ORIG_A / (key + ".png"))
    h_px, w_px = im.shape[:2]
    box_w, box_h = COL_W - 0.10, ROW_A[1] - ROW_A[0]
    k = min(box_w / w_px, box_h / h_px)
    w_in, h_in = w_px * k, h_px * k
    x0 = ox + 0.5 * (COL_W - w_in)
    y0 = ROW_A[0] + 0.5 * (box_h - h_in)
    cv.imshow(im, extent=(x0, x0 + w_in, y0, y0 + h_in), zorder=2,
              interpolation="lanczos")


def _save(fig, folder, stem):
    for ext in ("pdf", "svg"):
        fig.savefig(folder / (stem + "." + ext), transparent=True)
    fig.savefig(folder / (stem + ".png"), dpi=600, transparent=True)
    plt.close(fig)


def export_individual(w, r):
    """Rows b and c as standalone assets at final (7.0 in figure) size."""
    folder = OUT / "individual"
    folder.mkdir(exist_ok=True)
    aw = COL_W - AX_PAD_L - AX_PAD_R
    bh, chh = ROW_B[1] - ROW_B[0], ROW_C[1] - ROW_C[0]
    bm = (0.46, 0.12, 0.42, 0.34)          # left, right, bottom, top
    cm = (0.36, 0.06, 0.30, 0.42)
    rec = {}
    for i, key in enumerate(KEYS, start=1):
        W, H = bm[0] + aw + bm[1], bm[2] + bh + bm[3]
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([bm[0] / W, bm[2] / H, aw / W, bh / H])
        ax.set_facecolor("none")
        plot_weighting(ax, key, w, first=(i == 1))
        _save(fig, folder, "b%d_%s" % (i, key))
        rec["b%d_%s" % (i, key)] = {"canvas_in": [round(W, 3), round(H, 3)],
                                     "axes_offset_in": [bm[0], bm[2]],
                                     "axes_size_in": [round(aw, 3), bh]}
        W, H = cm[0] + aw + cm[1], cm[2] + chh + cm[3]
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([cm[0] / W, cm[2] / H, aw / W, chh / H])
        ax.set_facecolor("none")
        plot_reportable(ax, key, r, first=(i == 1))
        fig.text((cm[0] + aw / 2) / W, (cm[2] + chh + 0.08) / H, CLAIMS[key],
                 ha="center", va="bottom", fontsize=FS_LABEL, color=BLACK,
                 linespacing=1.2)
        _save(fig, folder, "c%d_%s" % (i, key))
        rec["c%d_%s" % (i, key)] = {"canvas_in": [round(W, 3), round(H, 3)],
                                     "axes_offset_in": [cm[0], cm[2]],
                                     "axes_size_in": [round(aw, 3), chh]}
    return rec


def main():
    rng = np.random.default_rng(3)
    w = weightings()
    r = reportable(w)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    cv = fig.add_axes([0, 0, 1, 1], zorder=0)
    cv.set_xlim(0, FIG_W)
    cv.set_ylim(0, FIG_H)
    cv.set_aspect("equal")
    cv.axis("off")

    # column titles and row separators
    for i, key in enumerate(KEYS):
        x = col_x(i)
        cv.text(x + COL_W / 2, FIG_H - 0.08, TITLES[key], ha="center",
                va="top", fontsize=FS_COL, fontweight="bold", color=BLACK,
                linespacing=1.2)
        if i:
            cv.plot([x - COL_GAP / 2] * 2, [0.12, FIG_H - 0.40], color=LIGHT,
                    lw=0.6, zorder=0)
    for y in (3.44, 1.98):
        cv.plot([0.06, FIG_W - 0.04], [y, y], color=LIGHT, lw=0.6, zorder=0)

    rows = (("a", "Topology", ROW_A, ROW_A[1] + 0.10),
            ("b", r"Age weighting $E(\theta)$", ROW_B, ROW_B[1] + 0.28),
            ("c", "Reportable quantity", ROW_C, ROW_C[1] + 0.34))
    for letter, name, (y0r, y1r), ytop in rows:
        cv.text(0.05, ytop, letter, ha="left", va="top", fontsize=8.0,
                fontweight="bold", color=BLACK)
        cv.text(0.12, 0.5 * (y0r + y1r), name, ha="center", va="center",
                fontsize=FS_ROW, color=BLACK, rotation=90)

    for i, key in enumerate(KEYS):
        x = col_x(i)
        if USE_ORIGINAL_ROW_A:
            draw_original_row_a(cv, key, x)
        else:
            DRAW[key](cv, x, ROW_A[0], rng)
        axb = fig.add_axes([(x + AX_PAD_L) / FIG_W, ROW_B[0] / FIG_H,
                            (COL_W - AX_PAD_L - AX_PAD_R) / FIG_W,
                            (ROW_B[1] - ROW_B[0]) / FIG_H])
        axb.set_facecolor("none")
        plot_weighting(axb, key, w, first=(i == 0))
        axc = fig.add_axes([(x + AX_PAD_L) / FIG_W, ROW_C[0] / FIG_H,
                            (COL_W - AX_PAD_L - AX_PAD_R) / FIG_W,
                            (ROW_C[1] - ROW_C[0]) / FIG_H])
        axc.set_facecolor("none")
        plot_reportable(axc, key, r, first=(i == 0))
        cv.text(x + COL_W / 2, ROW_C[1] + 0.08, CLAIMS[key], ha="center",
                va="bottom", fontsize=FS_LABEL, color=BLACK, linespacing=1.2)

    if not USE_ORIGINAL_ROW_A:
        bx0, bx1, by = col_x(2) + 0.30, col_x(2) + COL_W - 0.22, 3.53
        cv.imshow(np.linspace(0, 1, 128)[None, :], cmap=AGE_CMAP,
                  aspect="auto", extent=(bx0, bx1, by - 0.02, by + 0.02),
                  interpolation="bilinear", zorder=5)
        cv.add_patch(Rectangle((bx0, by - 0.02), bx1 - bx0, 0.04, fill=False,
                               ec=RULE, lw=0.6, zorder=6))
        note(cv, bx0 - 0.04, by, "young", ha="right")
        note(cv, bx1 + 0.04, by, "old", ha="left")
    cv.set_xlim(0, FIG_W)
    cv.set_ylim(0, FIG_H)

    for ext in ("pdf", "svg"):
        fig.savefig(OUT / (STEM + "." + ext), facecolor="white")
    fig.savefig(OUT / (STEM + ".png"), dpi=600, facecolor="white")
    plt.close(fig)

    with (OUT / "fig3_weightings.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["theta_min"] + KEYS)
        for i in range(0, len(theta), 5):
            wr.writerow(["%.1f" % theta[i]] + ["%.6f" % w[k][i] for k in KEYS])
    with (OUT / "fig3_reportable.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["time_min", "c_in", "through", "wick"])
        for i in range(0, len(t), 5):
            wr.writerow(["%.1f" % t[i], "%.6f" % r["c_in"][i],
                         "%.6f" % r["through"][i], "%.6f" % r["wick"][i]])
    record = {
        "status": "framework draft; weightings are schematic declarations",
        "input": "baseline 0.10 + pulses at 20 min (FWHM 10, 0.90) and 75 min (FWHM 10, 0.55)",
        "through_flow": P_THROUGH, "wick": P_WICK, "fill_measure_clear": P_FMC,
        "chronological": P_CHRONO, "accumulation": P_ACCUM,
        "fmc_readouts": [[float(a), float(b)] for a, b in zip(r["fmc_t"], r["fmc_v"])],
        "chrono_window_means": r["chrono"],
        "accumulation_mean_and_range": r["accum"][2:],
    }
    (OUT / "fig3_parameters.json").write_text(json.dumps(record, indent=2),
                                              encoding="utf-8")
    assets = export_individual(w, r)
    (OUT / "individual" / "assets.json").write_text(
        json.dumps(assets, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
