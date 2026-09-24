"""Figure 4 - decision-first validation and the evidence-gated claim ladder.

Card layout: both rows use the same visual unit, a light rounded card with a
short title inside and one graphic, joined by arrows. Everything explanatory
belongs to the caption.

a  Set tolerance -> Characterize components -> Predict held-out inputs ->
   Interpret in context.
   - tolerance: threshold crossing with Delta t_max and Delta A_max (Fig. 1a);
   - components: one test per stage of the Fig. 1b chain, coloured as in
     Fig. 1b (paired reference for H_bio, tracer for E(theta), concentration
     step for H_sensor); dashed grey = applied input or reference;
   - held-out inputs: the Fig. 2 pulse, ramp and sinusoid; prediction from
     preregistered parameters, "measured" points illustrative;
   - context: forearm with the planar sensor of Fig. 1b; site, sweat rate,
     paired reference and cohort.
b  Five claim levels, each shown by a glyph of the evidence it rests on;
   arrows name the evidence needed for the next level.

Format: 7.00 x 3.05 in; lettering >= 4.5 pt (math subscripts included);
strokes >= 0.6 pt. Backend: Python / matplotlib.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Polygon, Rectangle, Wedge)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "python"))
from run_evidence_closure_analysis import canonical_waveform  # noqa: E402
from transport_fidelity_models import (  # noqa: E402
    first_order_filter,
    transport_sensor_response,
)

OUT = ROOT / "outputs" / "revision_2026-09-21" / "figure4"
OUT.mkdir(parents=True, exist_ok=True)
STEM = "fig4_validation"

# ------------------------------------------------------------------ style
BLACK, GRAY, RULE, LIGHT = "#202020", "#8A8A8A", "#4A4A4A", "#C8C8C8"
BIO, DEV, SEN = "#3A7D44", "#1B6F87", "#5B3A99"
PRED, TOL, REF_RED = "#1B6F87", "#D6E4EE", "#C9585B"
CARD_FILL, CARD_EDGE = "#F7F8FA", "#D5D9DE"
LEVEL_FILL = ["#F7F8FA", "#EFF2F5", "#E6EBF0", "#DDE4EA", "#D3DCE4"]
SKIN_FILL, SKIN_EDGE = "#F3DDCD", "#B9826C"

FS_TITLE, FS_LABEL, FS_SMALL, FS_MATH, FS_AXIS = 7.0, 6.5, 6.0, 7.0, 6.5
LW = 1.2

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
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FIG_W, FIG_H = 7.00, 3.05
X_LEFT, X_RIGHT = 0.16, 6.94
# row a
A_Y0, A_Y1 = 1.55, 2.95
A_W = [1.18, 1.90, 1.68, 1.58]
A_GAP = (X_RIGHT - X_LEFT - sum(A_W)) / 3
A_TITLES = ["Set tolerance", "Characterize components",
            "Predict held-out inputs", "Interpret in context"]
# row b
B_GAP = 0.50
B_W = (X_RIGHT - X_LEFT - 4 * B_GAP) / 5
B_Y0, B_H = 0.52, 0.46


def a_cards():
    xs, x = [], X_LEFT
    for w in A_W:
        xs.append((x, x + w))
        x += w + A_GAP
    return xs


# ---------------------------------------------------------------- helpers
FRAME = [0.0, 0.0, 7.00, 3.05]      # ox, oy, width, height of the canvas


def new_canvas(x0, y0, x1, y1):
    fig = plt.figure(figsize=(x1 - x0, y1 - y0))
    fig.patch.set_facecolor("white")
    cv = fig.add_axes([0, 0, 1, 1], zorder=0)
    cv.set_xlim(x0, x1)
    cv.set_ylim(y0, y1)
    cv.set_aspect("equal")
    cv.axis("off")
    FRAME[:] = [x0, y0, x1 - x0, y1 - y0]
    return fig, cv


def axes_in(fig, x0, y0, w, h):
    ox, oy, W, H = FRAME
    ax = fig.add_axes([(x0 - ox) / W, (y0 - oy) / H, w / W, h / H])
    ax.set_facecolor("none")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def tc_labels(ax, ylabel=True):
    ax.text(1.03, 0.0, r"$t$", transform=ax.transAxes, ha="left",
            va="center", fontsize=FS_AXIS)
    if ylabel:
        ax.text(0.0, 1.04, r"$C$", transform=ax.transAxes, ha="center",
                va="bottom", fontsize=FS_AXIS)


def arrow(cv, p0, p1, colour=RULE, lw=0.9, ms=7):
    cv.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=colour, shrinkA=0, shrinkB=0,
                                 zorder=6))


def text(cv, x, y, s, size=FS_LABEL, colour=BLACK, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    cv.text(x, y, s, fontsize=size, color=colour, zorder=8, **kw)


def card(cv, x0, x1, y0, y1, fill=CARD_FILL):
    cv.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                fc=fill, ec=CARD_EDGE, lw=0.6, zorder=1))


def _bez(p0, p1, p2, p3, n=40):
    tt = np.linspace(0, 1, n)[:, None]
    return ((1 - tt) ** 3 * np.array(p0) + 3 * (1 - tt) ** 2 * tt * np.array(p1)
            + 3 * (1 - tt) * tt ** 2 * np.array(p2) + tt ** 3 * np.array(p3))


# -------------------------------------------------------- card contents
def key_line(cv, x, y, label, **line_kw):
    cv.plot([x, x + 0.16], [y, y], **line_kw)
    text(cv, x + 0.20, y, label, size=FS_SMALL, ha="left")


def tolerance(fig, cv, x0, x1):
    ax = axes_in(fig, x0 + 0.17, 1.84, x1 - x0 - 0.30, 0.78)
    t = np.linspace(0, 60, 1201)
    c = 0.10 + 0.85 * np.exp(-0.5 * ((t - 30) / 7.5) ** 2)
    thr = 0.55
    ts = float(t[int(np.argmax(c >= thr))])
    peak = float(c.max())
    dt, da = 4.0, 0.09
    ax.fill_between([ts - dt, ts + dt], -0.05, 0.72, color=TOL, lw=0, zorder=0)
    ax.fill_between([24, 36], peak - da, peak + da, color=TOL, lw=0, zorder=0)
    ax.axhline(thr, color=GRAY, lw=0.7, ls=(0, (2.5, 1.8)), zorder=1)
    ax.plot(t, c, color=BLACK, lw=LW, zorder=3)
    ax.plot([ts], [thr], "o", ms=3.2, mfc=BLACK, mec="white", mew=0.6,
            zorder=4)
    ax.annotate("", xy=(ts + dt, 0.02), xytext=(ts - dt, 0.02),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=BLACK,
                                shrinkA=0, shrinkB=0, mutation_scale=5))
    ax.text(ts + dt + 1.5, 0.02, r"$\Delta t_{\max}$", ha="left", va="center",
            fontsize=FS_MATH)
    ax.annotate("", xy=(38, peak + da), xytext=(38, peak - da),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=BLACK,
                                shrinkA=0, shrinkB=0, mutation_scale=5))
    ax.text(39.5, peak, r"$\Delta A_{\max}$", ha="left", va="center",
            fontsize=FS_MATH)
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.05, 1.12)
    tc_labels(ax)
    key_line(cv, 0.5 * (x0 + x1) - 0.30, 1.66, "threshold", color=GRAY,
             lw=0.7, ls=(0, (2.5, 1.8)))


def components(fig, cv, x0, x1):
    comps = [(r"$H_{\mathrm{bio}}$", "reference", BIO, "bio"),
             (r"$E(\theta)$", "tracer", DEV, "tracer"),
             (r"$H_{\mathrm{sensor}}$", "step", SEN, "step")]
    w, gap = 0.46, 0.14
    start = x0 + 0.5 * (x1 - x0 - (3 * w + 2 * gap)) + 0.02
    t = np.linspace(0, 40, 801)
    ref_kw = dict(color=GRAY, lw=0.8, ls=(0, (2.5, 1.8)))
    for k, (sym, test, col, kind) in enumerate(comps):
        sx = start + k * (w + gap)
        ax = axes_in(fig, sx, 2.00, w, 0.46)
        if kind == "bio":
            ref = 0.10 + 0.85 * np.exp(-0.5 * ((t - 14) / 4.0) ** 2)
            out = first_order_filter(t * 60, ref, 4.0 * 60)
            out = 0.10 + 0.75 * (out - 0.10) / (out.max() - 0.10)
        elif kind == "tracer":
            ref = np.where((t >= 5) & (t <= 6.5), 0.95, 0.05)
            out = transport_sensor_response(t * 60, ref, topology="gamma_rtd",
                                            transport_time_s=6.0 * 60,
                                            sensor_tau_s=None, rtd_shape=3.0)
            out = 0.05 + (out - 0.05) / (out.max() - 0.05) * 0.55
        else:
            ref = np.where(t >= 8, 0.9, 0.1)
            out = first_order_filter(t * 60, ref, 4.0 * 60)
        ax.plot(t, ref, **ref_kw)
        ax.plot(t, out, color=col, lw=LW)
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 1.1)
        tc_labels(ax, ylabel=(k == 0))
        text(cv, sx + w / 2, 2.62, sym, size=FS_MATH + 0.5, colour=col)
        text(cv, sx + w / 2, 1.86, test, size=FS_SMALL, colour=col)
    kx = 0.5 * (x0 + x1) - 0.52
    key_line(cv, kx, 1.66, "input", color=GRAY, lw=0.8, ls=(0, (2.5, 1.8)))
    cv.plot([kx + 0.56, kx + 0.64], [1.66, 1.66], color=BIO, lw=LW)
    cv.plot([kx + 0.64, kx + 0.72], [1.66, 1.66], color=DEV, lw=LW)
    text(cv, kx + 0.80 + 0.04, 1.66, "response", size=FS_SMALL, ha="left")
    # multicolour key swatch: one segment per component colour
    cv.plot([kx + 0.72, kx + 0.80], [1.66, 1.66], color=SEN, lw=LW)


def heldout(fig, cv, x0, x1, rng):
    pre = dict(tau_f=2.0, tau_s=1.0)
    true = dict(tau_f=2.4, tau_s=1.2)
    cases = [("pulse", (48, 80)), ("ramp", (30, 56)), ("sinusoid", (140, 170))]
    w, gap = 0.42, 0.11
    start = x0 + 0.5 * (x1 - x0 - (3 * w + 2 * gap)) + 0.02
    for k, (kind, (a, b)) in enumerate(cases):
        ts, u = canonical_waveform(kind, 10.0)
        tm = ts / 60.0
        kw = dict(topology="gamma_rtd", rtd_shape=3.0)
        pred = transport_sensor_response(ts, u, transport_time_s=pre["tau_f"] * 60,
                                         sensor_tau_s=pre["tau_s"] * 60, **kw)
        meas = transport_sensor_response(ts, u, transport_time_s=true["tau_f"] * 60,
                                         sensor_tau_s=true["tau_s"] * 60, **kw)
        m = (tm >= a) & (tm <= b)
        sx = start + k * (w + gap)
        ax = axes_in(fig, sx, 1.90, w, 0.52)
        ax.plot(tm[m], pred[m], color=PRED, lw=LW)
        idx = np.where(m)[0][::30]
        ax.plot(tm[idx], meas[idx] + rng.normal(0, 0.025, len(idx)), "o",
                ms=1.8, mfc=BLACK, mec="none")
        ax.set_xlim(a, b)
        ax.set_ylim(-0.08, 1.12)
        tc_labels(ax, ylabel=(k == 0))
        text(cv, sx + w / 2, 2.58, kind, size=FS_SMALL)
    ky = 1.66
    kx = x0 + 0.5 * (x1 - x0) - 0.58
    key_line(cv, kx, ky, "prediction", color=PRED, lw=LW)
    cv.plot([kx + 0.80], [ky], "o", ms=1.8, mfc=BLACK, mec="none")
    text(cv, kx + 0.86, ky, "measured", size=FS_SMALL, ha="left")


def forearm(cv, A, B, yo):
    W = B - 0.34
    Y = lambda v: v + yo  # noqa: E731
    pts = np.vstack([
        _bez((A + 0.05, Y(2.71)), (A + 0.35, Y(2.75)), (W - 0.25, Y(2.74)), (W, Y(2.72))),
        _bez((W, Y(2.72)), (W + 0.05, Y(2.76)), (W + 0.12, Y(2.84)), (W + 0.17, Y(2.81))),
        _bez((W + 0.17, Y(2.81)), (W + 0.20, Y(2.79)), (W + 0.17, Y(2.74)), (W + 0.15, Y(2.72))),
        _bez((W + 0.15, Y(2.72)), (W + 0.22, Y(2.72)), (B - 0.04, Y(2.70)), (B, Y(2.66))),
        _bez((B, Y(2.66)), (B + 0.02, Y(2.63)), (B - 0.01, Y(2.59)), (B - 0.05, Y(2.59))),
        _bez((B - 0.05, Y(2.59)), (W + 0.20, Y(2.56)), (W + 0.10, Y(2.54)), (W, Y(2.55))),
        _bez((W, Y(2.55)), (W - 0.25, Y(2.52)), (A + 0.35, Y(2.42)), (A + 0.05, Y(2.38))),
        _bez((A + 0.05, Y(2.38)), (A - 0.03, Y(2.44)), (A - 0.03, Y(2.66)), (A + 0.05, Y(2.71))),
    ])
    cv.add_patch(Polygon(pts, closed=True, fc=SKIN_FILL, ec=SKIN_EDGE, lw=0.8,
                         joinstyle="round", zorder=3))
    cv.plot([W + 0.02, W + 0.03], [Y(2.57), Y(2.70)], color=SKIN_EDGE, lw=0.6,
            zorder=4)
    for yf in (2.62, 2.66):
        cv.plot([B - 0.16, B - 0.05], [Y(yf), Y(yf) + 0.004], color=SKIN_EDGE,
                lw=0.6, zorder=4)
    px, py, pw, ph = 0.5 * (A + W) - 0.15, Y(2.555), 0.30, 0.15
    cv.add_patch(FancyBboxPatch((px, py), pw, ph,
                                boxstyle="round,pad=0,rounding_size=0.03",
                                fc="white", ec=RULE, lw=0.7, zorder=5))
    ex, ey = px + pw / 2, py + ph / 2
    cv.add_patch(Circle((ex, ey), 0.030, fc="#8E7CC3", ec=SEN, lw=0.6,
                        zorder=6))
    cv.add_patch(Wedge((ex, ey), 0.055, 40, 320, width=0.013, fc="#C9C0E4",
                       ec=SEN, lw=0.6, zorder=6))


def icon_site(cv, x, y):
    body = np.vstack([_bez((x - 0.080, y - 0.075), (x - 0.080, y - 0.005),
                           (x - 0.040, y + 0.010), (x, y + 0.010)),
                      _bez((x, y + 0.010), (x + 0.040, y + 0.010),
                           (x + 0.080, y - 0.005), (x + 0.080, y - 0.075))])
    cv.add_patch(Polygon(body, closed=True, fc="white", ec=RULE, lw=0.7,
                         zorder=6, joinstyle="round"))
    cv.add_patch(Circle((x, y + 0.055), 0.036, fc="white", ec=RULE, lw=0.7,
                        zorder=6))
    cv.add_patch(Circle((x + 0.058, y - 0.035), 0.019, fc=SEN, ec="white",
                        lw=0.6, zorder=7))


def icon_drop(cv, x, y):
    drop = np.vstack([_bez((x, y + 0.085), (x + 0.020, y + 0.050),
                           (x + 0.060, y + 0.005), (x + 0.060, y - 0.030)),
                      _bez((x + 0.060, y - 0.030), (x + 0.060, y - 0.070),
                           (x + 0.030, y - 0.085), (x, y - 0.085)),
                      _bez((x, y - 0.085), (x - 0.030, y - 0.085),
                           (x - 0.060, y - 0.070), (x - 0.060, y - 0.030)),
                      _bez((x - 0.060, y - 0.030), (x - 0.060, y + 0.005),
                           (x - 0.020, y + 0.050), (x, y + 0.085))])
    cv.add_patch(Polygon(drop, closed=True, fc="#BFE4EB", ec=DEV, lw=0.7,
                         zorder=6, joinstyle="round"))
    cv.plot([x - 0.030, x - 0.036], [y - 0.010, y - 0.045], color="white",
            lw=1.0, solid_capstyle="round", zorder=7)


def icon_tube(cv, x, y):
    cv.add_patch(FancyBboxPatch((x - 0.028, y - 0.085), 0.056, 0.150,
                                boxstyle="round,pad=0,rounding_size=0.026",
                                fc="white", ec=RULE, lw=0.7, zorder=6))
    cv.add_patch(FancyBboxPatch((x - 0.021, y - 0.078), 0.042, 0.075,
                                boxstyle="round,pad=0,rounding_size=0.019",
                                fc=REF_RED, ec="none", zorder=6.5))
    cv.add_patch(Rectangle((x - 0.036, y + 0.060), 0.072, 0.030, fc=RULE,
                           ec="none", zorder=7))


def icon_group(cv, x, y):
    for dx, dy, sc, fc in ((-0.055, 0.012, 0.80, "#E4E4E4"),
                           (0.055, 0.012, 0.80, "#E4E4E4"),
                           (0.0, 0.0, 1.0, "#CFCFCF")):
        cx, cy = x + dx, y + dy
        body = np.vstack([
            _bez((cx - 0.060 * sc, cy - 0.075), (cx - 0.060 * sc, cy - 0.015),
                 (cx - 0.030 * sc, cy), (cx, cy)),
            _bez((cx, cy), (cx + 0.030 * sc, cy),
                 (cx + 0.060 * sc, cy - 0.015), (cx + 0.060 * sc, cy - 0.075))])
        z = 7 if dx == 0 else 6
        cv.add_patch(Polygon(body, closed=True, fc=fc, ec=RULE, lw=0.6,
                             zorder=z, joinstyle="round"))
        cv.add_patch(Circle((cx, cy + 0.040 * sc + 0.005), 0.028 * sc, fc=fc,
                            ec=RULE, lw=0.6, zorder=z))


def placeholder(cv, x0, y0, w, h, label=None):
    cv.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle="round,pad=0,rounding_size=0.03",
                                fc="white", ec=GRAY, lw=0.6,
                                ls=(0, (2.5, 1.8)), zorder=2))
    if label:
        text(cv, x0 + w / 2, y0 + h / 2, label, size=5.2, colour=GRAY,
             style="italic", linespacing=1.15)


# card 4 artwork is drawn in BioRender; these boxes fix its footprint
CONTEXT_ART = {"forearm": None, "icons": []}


def context(cv, x0, x1):
    ax0, ay0, aw, ah = x0 + 0.10, 2.14, x1 - x0 - 0.20, 0.56
    placeholder(cv, ax0, ay0, aw, ah, "forearm with patch\n(BioRender)")
    CONTEXT_ART["forearm"] = (ax0, ay0, aw, ah)
    labels = ("site", "sweat\nrate", "reference", "cohort")
    slot = (x1 - x0 - 0.12) / 4
    CONTEXT_ART["icons"] = []
    for i, lab in enumerate(labels):
        ix = x0 + 0.06 + slot * (i + 0.5)
        placeholder(cv, ix - 0.09, 1.86, 0.18, 0.18)
        CONTEXT_ART["icons"].append((ix - 0.09, 1.86, 0.18, 0.18))
        text(cv, ix, 1.70, lab, size=FS_SMALL, linespacing=1.1)


# ------------------------------------------------------------- panel b
LEVELS = ["Detectable", "Component-\ncharacterized", "Dynamically\npredictive",
          "Physiologically\ninterpretable", "Clinically\nuseful"]
EVIDENCE = ["component\ntests", "held-out\nprediction", "on-body\nreference",
            "decision\nvalidation"]


def glyph(ax, k, rng):
    x = np.linspace(0, 1, 400)
    if k == 0:
        noise = np.convolve(0.05 * rng.standard_normal(x.size), np.ones(4) / 4,
                            mode="same")
        y = 0.22 + noise + 0.55 * np.exp(-0.5 * ((x - 0.55) / 0.035) ** 2)
        ax.fill_between([0, 1], 0.12, 0.32, color="#E2E2E2", lw=0, zorder=0)
        ax.plot(x, y, color=BLACK, lw=0.8)
    elif k == 1:
        for xv, yv, col in ((0.42, 0.78, BIO), (0.60, 0.50, DEV),
                            (0.48, 0.22, SEN)):
            ax.plot([xv - 0.16, xv + 0.16], [yv, yv], color=col, lw=1.0)
            for xe in (xv - 0.16, xv + 0.16):
                ax.plot([xe, xe], [yv - 0.06, yv + 0.06], color=col, lw=1.0)
            ax.plot([xv], [yv], "o", ms=3.4, mfc=col, mec="white", mew=0.6)
    elif k == 2:
        y = 0.12 + 0.72 * (np.exp(-0.5 * ((x - 0.30) / 0.08) ** 2)
                           + 0.8 * np.exp(-0.5 * ((x - 0.72) / 0.08) ** 2))
        ax.plot(x, y, color=PRED, lw=LW)
        idx = np.arange(10, 400, 26)
        ax.plot(x[idx], y[idx] + 0.035 * rng.standard_normal(idx.size), "o",
                ms=1.8, mfc=BLACK, mec="none")
    elif k == 3:
        ref = 0.15 + 0.65 * np.exp(-0.5 * ((x - 0.45) / 0.14) ** 2)
        dev = 0.15 + 0.62 * np.exp(-0.5 * ((x - 0.49) / 0.15) ** 2)
        ax.plot(x, ref, color=REF_RED, lw=0.9, ls=(0, (2.5, 1.8)))
        ax.plot(x, dev, color=BLACK, lw=LW)
    else:
        c = 0.10 + 0.75 * np.exp(-0.5 * ((x - 0.55) / 0.13) ** 2)
        thr = 0.50
        xs = float(x[int(np.argmax(c >= thr))])
        ax.fill_between([xs - 0.07, xs + 0.07], 0.0, 0.95, color="white",
                        alpha=0.8, lw=0, zorder=0)
        ax.axhline(thr, color=GRAY, lw=0.7, ls=(0, (2.5, 1.8)))
        ax.plot(x, c, color=BLACK, lw=LW)
        ax.plot([xs], [thr], "o", ms=3.0, mfc=BLACK, mec="white", mew=0.6)
        ax.plot([0.74, 0.80, 0.92], [0.80, 0.70, 0.92], color=BIO, lw=1.3,
                solid_capstyle="round", solid_joinstyle="round")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, 1.0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def ladder(fig, cv, rng):
    for k in range(5):
        cx0 = X_LEFT + k * (B_W + B_GAP)
        card(cv, cx0, cx0 + B_W, B_Y0 - 0.06, B_Y0 + B_H + 0.06,
             fill=LEVEL_FILL[k])
        glyph(axes_in(fig, cx0 + 0.10, B_Y0, B_W - 0.20, B_H), k, rng)
        text(cv, cx0 + B_W / 2, 0.26, LEVELS[k], size=FS_LABEL,
             fontweight="bold", linespacing=1.15)
        if k < 4:
            a, b = cx0 + B_W + 0.05, cx0 + B_W + B_GAP - 0.05
            ym = B_Y0 + B_H / 2
            arrow(cv, (a, ym - 0.10), (b, ym - 0.10))
            text(cv, 0.5 * (a + b), ym + 0.05, EVIDENCE[k], size=FS_SMALL,
                 colour=RULE, linespacing=1.15)


def draw_card(fig, cv, i, x0, x1, rng):
    card(cv, x0, x1, A_Y0, A_Y1)
    text(cv, 0.5 * (x0 + x1), A_Y1 - 0.14, A_TITLES[i], size=FS_TITLE,
         fontweight="bold")
    if i == 0:
        tolerance(fig, cv, x0, x1)
    elif i == 1:
        components(fig, cv, x0, x1)
    elif i == 2:
        heldout(fig, cv, x0, x1, rng)
    else:
        context(cv, x0, x1)


def draw_row_a(fig, cv, cards, seed=5):
    rng = np.random.default_rng(seed)
    for i, (x0, x1) in enumerate(cards):
        draw_card(fig, cv, i, x0, x1, rng)
        if i < 3:
            ym = 0.5 * (A_Y0 + A_Y1)
            arrow(cv, (x1 + 0.025, ym), (cards[i + 1][0] - 0.025, ym), ms=6)
    cv.text(0.02, FIG_H - 0.02, "a", ha="left", va="top", fontsize=8.0,
            fontweight="bold")


def save(fig, stem, folder=OUT, transparent=False):
    for ext in ("pdf", "svg"):
        fig.savefig(folder / (stem + "." + ext), facecolor="white",
                    transparent=transparent)
    fig.savefig(folder / (stem + ".png"), dpi=600, facecolor="white",
                transparent=transparent)
    plt.close(fig)


def main():
    cards = a_cards()

    fig, cv = new_canvas(0.0, 0.0, FIG_W, FIG_H)
    draw_row_a(fig, cv, cards)
    ladder(fig, cv, np.random.default_rng(6))
    cv.text(0.02, 1.36, "b", ha="left", va="top", fontsize=8.0,
            fontweight="bold")
    save(fig, STEM)

    fig, cv = new_canvas(0.0, A_Y0 - 0.04, FIG_W, FIG_H)
    draw_row_a(fig, cv, cards)
    save(fig, "fig4a_panel")

    card_dir = OUT / "panel_a_cards"
    card_dir.mkdir(exist_ok=True)
    names = ["a1_set_tolerance", "a2_characterize_components",
             "a3_predict_heldout", "a4_context_placeholder"]
    rng = np.random.default_rng(5)
    for i, ((x0, x1), name) in enumerate(zip(cards, names)):
        fig, cv = new_canvas(x0 - 0.01, A_Y0 - 0.01, x1 + 0.01, A_Y1 + 0.01)
        draw_card(fig, cv, i, x0, x1, rng)
        save(fig, name, card_dir, transparent=True)

    import json
    layout = {"figure_in": [FIG_W, FIG_H],
              "panel_a_cards_in": [[round(x0, 3), A_Y0, round(x1, 3), A_Y1]
                                   for x0, x1 in cards],
              "card4_biorender_boxes_in": {
                  "forearm_x_y_w_h": [round(v, 3) for v in CONTEXT_ART["forearm"]],
                  "icons_x_y_w_h": [[round(v, 3) for v in b]
                                    for b in CONTEXT_ART["icons"]]}}
    (OUT / "fig4_layout.json").write_text(json.dumps(layout, indent=2),
                                          encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
