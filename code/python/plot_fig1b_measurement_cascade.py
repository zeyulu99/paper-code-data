"""Figure 1b - the measurement cascade, computed rather than sketched.

Top row, the physical chain: skin and gland (placeholder for the author's
illustration) -> device channel with its hydrodynamically coupled sensing zone
-> planar electrodes on the channel floor.

Bottom row, the same chain as signals on one shared time axis and one shared
concentration axis, so delay, attenuation and dispersion are visible:
    C_in  --E(theta)-->  C_f  --H_sensor-->  S
Each plot shows its own input dashed and its own output solid.

Both stages are rows of the released 1,620-point canonical grid:
    transport only : pulse, T = 10 min, distributed RTD (k = 3), tau_f = 5, tau_s = 0
    full chain     : pulse, T = 10 min, distributed RTD (k = 3), tau_f = 5, tau_s = 5
main() asserts agreement with that CSV before drawing.

H_bio is analyte-, site- and subject-dependent and is not modelled here, so no
C_source curve is drawn; drawing one would silently assert a transfer function.

Format  : 7.00 x 2.60 in, lettering >= 4.5 pt including math subscripts,
          strokes >= 0.6 pt.
Backend : Python / matplotlib.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch, Rectangle,
                                Wedge, Circle)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "python"))
from run_evidence_closure_analysis import (  # noqa: E402
    canonical_metrics,
    canonical_waveform,
)
import fig1b_illustrations as illus  # noqa: E402
from transport_fidelity_models import (  # noqa: E402
    first_order_filter,
    gamma_rtd_kernel,
    transport_sensor_response,
)

OUT = ROOT / "outputs" / "revision_2026-09-17" / "figure1b"
OUT.mkdir(parents=True, exist_ok=True)
GRID = (ROOT / "outputs" / "revision_2026-08-31_figure2_data_package"
        / "02_canonical_extended_grid_1620.csv")
STEM = "fig1b_measurement_cascade"

# ----------------------------------------------------------------- parameters
WAVEFORM = "pulse"
T_MIN = 10.0            # FWHM of the canonical pulse
TOPOLOGY = "distributed_RTD"
RTD_SHAPE = 3.0
TAU_F = 5.0             # mean residence time of the coupled zone, min
TAU_S = 5.0             # first-order sensor time constant, min
VIEW_MIN, VIEW_MAX = 45.0, 105.0
X_TICKS = [50, 75, 100]
THETA_MAX = 20.0

FIG_W, FIG_H = 7.00, 2.60

BLACK = "#202020"
GRAY = "#8A8A8A"
RULE = "#4A4A4A"
BIO, BIO_BG = "#3A7D44", "#EFF6F0"
DEV, DEV_BG = "#1B6F87", "#EBF3F6"
SEN, SEN_BG = "#5B3A99", "#F2EFF8"
AGE_CMAP = LinearSegmentedColormap.from_list("age", ["#BFE4EB", "#0E3B4A"])

COLS = {  # x0, x1 in inches
    "bio": (0.04, 1.98),
    "dev": (2.06, 5.06),
    "sen": (5.14, 6.96),
}
PLOT_W, PLOT_H, PLOT_B = 1.40, 0.78, 0.40
PLOT_X = {"in": 0.50, "f": 2.92, "s": 5.46}
PLACEHOLDER = (0.14, 1.50, 1.42, 2.32)   # x0, y0, x1, y1 in inches
KERNEL_BOX = (4.36, 1.70, 0.60, 0.50)    # x0, y0, w, h in inches
SENSOR_ORIGIN = (5.16, 1.46)             # lower-left of the sensor drawing

FS_HEAD, FS_TITLE, FS_AXIS, FS_TICK, FS_NOTE = 7.0, 7.0, 6.5, 6.0, 6.0
LW_CURVE, LW_REF, LW_THIN = 1.4, 0.9, 0.7

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
    "font.size": FS_AXIS,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ------------------------------------------------------------------- model
def simulate():
    time_s, c_in = canonical_waveform(WAVEFORM, T_MIN)
    c_f = transport_sensor_response(time_s, c_in, topology="gamma_rtd",
                                    transport_time_s=TAU_F * 60.0,
                                    sensor_tau_s=None, rtd_shape=RTD_SHAPE)
    s = first_order_filter(time_s, c_f, TAU_S * 60.0)
    chain = transport_sensor_response(time_s, c_in, topology="gamma_rtd",
                                      transport_time_s=TAU_F * 60.0,
                                      sensor_tau_s=TAU_S * 60.0,
                                      rtd_shape=RTD_SHAPE)
    if float(np.max(np.abs(chain - s))) > 1e-9:
        raise SystemExit("staged and one-shot cascades disagree")
    dt_s = float(time_s[1] - time_s[0])
    kernel = gamma_rtd_kernel(dt_s, TAU_F * 60.0, RTD_SHAPE) * 60.0  # per min
    theta = np.arange(len(kernel)) * dt_s / 60.0
    return time_s, c_in, c_f, s, theta, kernel


def metrics(time_s, ref, out):
    m = canonical_metrics(WAVEFORM, T_MIN, time_s, ref, out)
    return {k: float(m[k]) for k in
            ("amplitude_retention", "feature_delay_min", "nrmse", "pearson_r")}


def assert_matches_grid(stage_metrics: dict) -> list[str]:
    with GRID.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    checked = []
    for label, tau_s in (("transport", 0.0), ("chain", TAU_S)):
        match = [r for r in rows
                 if r["waveform_type"] == WAVEFORM and float(r["T_min"]) == T_MIN
                 and r["topology"] == TOPOLOGY
                 and float(r["tau_transport_min"]) == TAU_F
                 and float(r["tau_sensor_min"]) == tau_s]
        if len(match) != 1:
            raise SystemExit("no unique grid row for the %s stage" % label)
        row = match[0]
        for key, col, tol in (("amplitude_retention", "amplitude_retention", 5e-3),
                              ("nrmse", "NRMSE", 5e-3),
                              ("pearson_r", "pearson_r", 5e-3),
                              ("feature_delay_min", "feature_delay_min", 0.06)):
            if abs(stage_metrics[label][key] - float(row[col])) > tol:
                raise SystemExit("%s %s drifted: %.4f vs grid %.4f"
                                 % (label, key, stage_metrics[label][key],
                                    float(row[col])))
        checked.append("%s: tau_f=%g, tau_s=%g" % (label, TAU_F, tau_s))
    return checked


def moments(t_min, y):
    area = float(np.trapezoid(y, t_min))
    centroid = float(np.trapezoid(t_min * y, t_min) / area)
    var = float(np.trapezoid((t_min - centroid) ** 2 * y, t_min) / area)
    return area, centroid, var ** 0.5


# ------------------------------------------------------------------ drawing
def col_background(cv, key, colour, bg, title):
    x0, x1 = COLS[key]
    cv.add_patch(FancyBboxPatch((x0, 0.04), x1 - x0, 2.52,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                fc=bg, ec=colour, lw=0.6, zorder=0))
    cv.text(0.5 * (x0 + x1), 2.445, title, ha="center", va="center",
            fontsize=FS_HEAD, fontweight="bold", color=colour, zorder=3)


def arrow(cv, p0, p1, colour=RULE, lw=0.9, style="-|>", rad=0.0, ls="-"):
    cv.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=7,
                                 lw=lw, color=colour, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=4,
                                 connectionstyle="arc3,rad=%g" % rad))


def draw_placeholder(cv):
    x0, y0, x1, y1 = PLACEHOLDER
    cv.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.04",
                                fc="#FFFFFF", ec=GRAY, lw=0.7,
                                linestyle=(0, (3, 2)), zorder=2))
    cv.text(0.5 * (x0 + x1), 0.5 * (y0 + y1) + 0.07,
            "Skin and sweat-gland\nillustration", ha="center", va="center",
            fontsize=FS_NOTE, color=GRAY, zorder=3, linespacing=1.2)
    cv.text(0.5 * (x0 + x1), 0.5 * (y0 + y1) - 0.17,
            "(placeholder, %.2f x %.2f in)" % (x1 - x0, y1 - y0),
            ha="center", va="center", fontsize=5.0, color=GRAY, zorder=3,
            style="italic")


CHANNEL = dict(cx0=2.22, cx1=3.80, yb=1.78, yt=2.08, zx0=2.80, zx1=3.26)
AGE_BAR_Y = 1.535
PARTICLE_SEED = 7


def draw_channel(cv, flow_arrows=True, age_scale=True):
    """Channel, coupled zone, electrodes and age-coloured fluid parcels."""
    c = CHANNEL
    cx0, cx1, yb, yt, zx0, zx1 = (c[k] for k in
                                  ("cx0", "cx1", "yb", "yt", "zx0", "zx1"))
    rng = np.random.default_rng(PARTICLE_SEED)
    # walls
    for y in (yb, yt):
        cv.plot([cx0, cx1], [y, y], color=RULE, lw=0.9, zorder=3,
                solid_capstyle="butt")
    # coupled sensing zone
    cv.add_patch(Rectangle((zx0, yb - 0.06), zx1 - zx0, (yt - yb) + 0.12,
                           fill=False, ec=DEV, lw=0.8, linestyle=(0, (3, 2)),
                           zorder=4))
    cv.text(0.5 * (zx0 + zx1), yt + 0.10, r"$V_{\mathrm{eff}}$",
            ha="center", va="bottom", fontsize=FS_TITLE, color=DEV, zorder=5)
    # planar electrodes on the floor, colour-linked to the sensor column and
    # named, so the link does not rest on colour alone
    cv.add_patch(Rectangle((zx0 + 0.07, yb), zx1 - zx0 - 0.14, 0.028,
                           fc=SEN, ec="none", zorder=5))
    cv.text(0.5 * (zx0 + zx1), yb - 0.09, "electrodes", ha="center",
            va="top", fontsize=FS_NOTE, color=SEN, zorder=5)
    # fluid parcels: age grows downstream and spreads with distance;
    # two old parcels are held up in the corners of the coupled zone
    xs = np.linspace(cx0 + 0.08, cx1 - 0.08, 15)
    for i, x in enumerate(xs):
        frac = (x - cx0) / (cx1 - cx0)
        age = np.clip(frac * 0.85 + rng.normal(0, 0.05 + 0.18 * frac), 0, 1)
        y = yb + 0.07 + ((i * 0.37) % 1.0) * (yt - yb - 0.14)
        cv.add_patch(Circle((x, y), 0.027, fc=AGE_CMAP(age), ec="white",
                            lw=0.6, zorder=6))
    for x, y in ((zx0 + 0.05, yt - 0.05), (zx1 - 0.05, yb + 0.07)):
        cv.add_patch(Circle((x, y), 0.027, fc=AGE_CMAP(0.97), ec="white",
                            lw=0.6, zorder=6))
    if flow_arrows:
        # sweat leaves the patch lumen and enters the enlarged channel
        y_out = PLACEHOLDER[1] + illus.SKIN_OUTLET_Y
        x_bend = 1.80
        cv.plot([PLACEHOLDER[2] + 0.02, x_bend, x_bend],
                [y_out, y_out, 1.93], color=RULE, lw=0.9, zorder=4,
                solid_joinstyle="round", solid_capstyle="butt")
        arrow(cv, (x_bend, 1.93), (cx0 - 0.02, 1.93), colour=RULE)
        cv.text(x_bend - 0.05, 0.5 * (y_out + 1.93), r"$Q(t)$", ha="right",
                va="center", fontsize=FS_TITLE, color=BLACK, zorder=5)
        arrow(cv, (cx1 + 0.02, 1.93), (cx1 + 0.22, 1.93), colour=RULE)
    if age_scale:
        bx0, bx1, by = 2.62, 3.44, AGE_BAR_Y
        grad = np.linspace(0, 1, 128)[None, :]
        cv.imshow(grad, extent=(bx0, bx1, by - 0.022, by + 0.022),
                  cmap=AGE_CMAP, aspect="auto", zorder=5,
                  interpolation="bilinear")
        cv.add_patch(Rectangle((bx0, by - 0.022), bx1 - bx0, 0.044,
                               fill=False, ec=RULE, lw=0.6, zorder=6))
        cv.text(bx0 - 0.05, by, "young", ha="right", va="center",
                fontsize=FS_NOTE, color=RULE)
        cv.text(bx1 + 0.05, by, "old", ha="left", va="center",
                fontsize=FS_NOTE, color=RULE)
    return (zx1, yt + 0.06)


def draw_zone_to_kernel_arrow(cv, zone_anchor):
    """Curved arrow from the coupled zone to the E(theta) inset."""
    x0, y0, _, h = KERNEL_BOX
    arrow(cv, zone_anchor, (x0 - 0.18, y0 + h - 0.02), colour=DEV, lw=0.7,
          rad=-0.35)


def draw_electrodes(cv):
    sx0, sx1, sy0, sy1 = 5.44, 6.66, 1.60, 2.28
    cv.add_patch(FancyBboxPatch((sx0, sy0), sx1 - sx0, sy1 - sy0,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                fc="#FFFFFF", ec=RULE, lw=0.7, zorder=2))
    cx, cy = 5.84, 1.94
    cv.add_patch(Circle((cx, cy), 0.13, fc="#8E7CC3", ec=SEN, lw=0.7, zorder=3))
    cv.add_patch(Wedge((cx, cy), 0.25, 40, 320, width=0.06, fc="#C9C0E4",
                       ec=SEN, lw=0.6, zorder=3))
    cv.add_patch(Wedge((cx, cy), 0.25, -24, 24, width=0.06, fc="#9A9A9A",
                       ec=RULE, lw=0.6, zorder=3))
    cv.text(cx, cy, "WE", ha="center", va="center", fontsize=FS_NOTE,
            color="white", fontweight="bold", zorder=5)
    labels = (("CE", (cx + 0.08, cy + 0.225), 2.14),
              ("RE", (cx + 0.24, cy - 0.06), 1.74))
    for text, (px, py), ty in labels:
        tx = 6.30
        cv.plot([px, tx - 0.04], [py, ty], color=RULE, lw=0.6, zorder=4)
        cv.text(tx, ty, text, ha="left", va="center", fontsize=FS_NOTE,
                color=BLACK, zorder=5)
    cv.text(0.5 * (sx0 + sx1), sy0 - 0.07, "planar electrodes, channel floor",
            ha="center", va="top", fontsize=5.4, color=RULE, zorder=5)


def gradient_fill(ax, x, y, vmax):
    """Fill under a curve with the age colour map, as one clipped image."""
    from matplotlib.patches import Polygon
    verts = np.column_stack([np.r_[x[0], x, x[-1]], np.r_[0.0, y, 0.0]])
    clip = Polygon(verts, closed=True, fc="none", ec="none",
                   transform=ax.transData)
    ax.add_patch(clip)
    img = ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=AGE_CMAP,
                    extent=(0, vmax, 0, float(np.max(y))), aspect="auto",
                    origin="lower", interpolation="bilinear", zorder=1)
    img.set_clip_path(clip)


def plot_kernel(ax, theta, kernel):
    ax.set_facecolor("none")
    keep = theta <= THETA_MAX
    th, ke = theta[keep], kernel[keep]
    gradient_fill(ax, th, ke, THETA_MAX)
    ax.plot(th, ke, color=DEV, lw=1.0, zorder=2)
    ax.axvline(TAU_F, color=BLACK, lw=0.6, ls=(0, (2, 1.6)), zorder=3)
    ax.text(TAU_F + 0.6, ke.max() * 1.02, r"$\tau_{\mathrm{f}}$",
            ha="left", va="bottom", fontsize=FS_TITLE, color=BLACK)
    ax.set_xlim(0, THETA_MAX)
    ax.set_ylim(0, ke.max() * 1.30)
    ax.set_xticks([0, 10, 20])
    ax.set_yticks([])
    ax.set_xlabel(r"Age $\theta$ (min)", fontsize=FS_TICK, labelpad=1.0)
    ax.set_ylabel(r"$E(\theta)$", fontsize=FS_TICK, labelpad=1.5)
    ax.tick_params(pad=1.2)


def draw_kernel(fig, cv, theta, kernel, zone_anchor):
    x0, y0, w, h = KERNEL_BOX
    ax = fig.add_axes([x0 / FIG_W, y0 / FIG_H, w / FIG_W, h / FIG_H])
    plot_kernel(ax, theta, kernel)
    draw_zone_to_kernel_arrow(cv, zone_anchor)
    return ax


def style_signal(ax, show_ylabel):
    ax.set_facecolor("none")
    ax.set_xlim(VIEW_MIN, VIEW_MAX)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(X_TICKS)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("Time (min)", fontsize=FS_AXIS, labelpad=1.5)
    if show_ylabel:
        ax.set_ylabel("Concentration\n(normalized)", fontsize=FS_AXIS,
                      labelpad=1.5, linespacing=1.3)
    ax.tick_params(pad=1.3)


LBL_IN = r"$C_{\mathrm{in}}$"
LBL_F = r"$C_{\mathrm{f}}$"
LBL_S = r"$S$"
FS_LEGEND = 7.0   # mathtext subscripts render at 0.7x, so 7.0 pt -> 4.9 pt


def plot_stage(ax, key, t, c_in, c_f, s):
    """Draw one stage: its input dashed, its output solid, both labelled."""
    style_signal(ax, key == "in")
    note_x = VIEW_MAX - 0.02 * (VIEW_MAX - VIEW_MIN)
    dash = (0, (3.2, 2.2))
    if key == "in":
        ax.plot(t, c_in, color=BLACK, lw=LW_CURVE, label=LBL_IN)
    elif key == "f":
        ax.plot(t, c_in, color=GRAY, lw=LW_REF, ls=dash, label=LBL_IN)
        ax.plot(t, c_f, color=DEV, lw=LW_CURVE, label=LBL_F)
        ax.text(note_x, 1.08, "delay, dispersion,\ncarryover", ha="right",
                va="top", fontsize=FS_NOTE, color=DEV, linespacing=1.15)
    else:
        ax.plot(t, c_f, color=DEV, lw=LW_REF, ls=dash, label=LBL_F)
        ax.plot(t, s, color=SEN, lw=LW_CURVE, label=LBL_S)
        ax.text(note_x, 1.08, "added sensor\nlag", ha="right", va="top",
                fontsize=FS_NOTE, color=SEN, linespacing=1.15)
    ax.legend(loc="upper left", fontsize=FS_LEGEND, frameon=False,
              handlelength=1.3, handletextpad=0.3, labelspacing=0.25,
              borderaxespad=0.15, borderpad=0.1)


def signal_axes(fig, key):
    return fig.add_axes([PLOT_X[key] / FIG_W, PLOT_B / FIG_H,
                         PLOT_W / FIG_W, PLOT_H / FIG_H])


STAGE_TITLES = {
    "in": (r"$C_{\mathrm{in}}=H_{\mathrm{bio}}\{C_{\mathrm{source}}\}$", BIO),
    "f": (r"$C_{\mathrm{f}}=E\ast C_{\mathrm{in}}$", DEV),
    "s": (r"$S=H_{\mathrm{sensor}}\{C_{\mathrm{f}}\}$", SEN),
}
STAGE_NAMES = {"in": "C_in", "f": "C_f", "s": "S"}
SIGNAL_MARGINS = (0.46, 0.06, 0.30, 0.06)   # left, right, bottom, top (in)
KERNEL_MARGINS = (0.22, 0.20, 0.30, 0.14)


def save_asset(fig, folder, stem):
    for ext in ("pdf", "svg"):
        fig.savefig(folder / (stem + "." + ext), transparent=True)
    fig.savefig(folder / (stem + ".png"), dpi=600, transparent=True)
    plt.close(fig)


def export_individual(t, c_in, c_f, s, theta, kernel):
    """Standalone assets at the composite's exact physical size.

    Every signal asset uses the same margins, so the axes boxes coincide
    when the files are placed at 100 % and aligned on their left edges.
    """
    folder = OUT / "individual"
    folder.mkdir(exist_ok=True)
    ml, mr, mb, mt = SIGNAL_MARGINS
    size = (ml + PLOT_W + mr, mb + PLOT_H + mt)
    for idx, key in enumerate(("in", "f", "s"), start=1):
        fig = plt.figure(figsize=size)
        ax = fig.add_axes([ml / size[0], mb / size[1],
                           PLOT_W / size[0], PLOT_H / size[1]])
        plot_stage(ax, key, t, c_in, c_f, s)
        save_asset(fig, folder, "b%d_signal_%s" % (idx, STAGE_NAMES[key]))

    _, _, w, h = KERNEL_BOX
    kl, kr, kb, kt = KERNEL_MARGINS
    ksize = (kl + w + kr, kb + h + kt)
    fig = plt.figure(figsize=ksize)
    ax = fig.add_axes([kl / ksize[0], kb / ksize[1], w / ksize[0], h / ksize[1]])
    plot_kernel(ax, theta, kernel)
    save_asset(fig, folder, "b4_kernel_E_theta")

    fig = plt.figure(figsize=(1.42, 0.20))
    cv = fig.add_axes([0, 0, 1, 1])
    cv.set_xlim(0, 1.42)
    cv.set_ylim(0, 0.20)
    cv.axis("off")
    bx0, bx1 = 0.30, 1.12   # same 0.82 in bar as the composite
    cv.imshow(np.linspace(0, 1, 128)[None, :], cmap=AGE_CMAP, aspect="auto",
              extent=(bx0, bx1, 0.078, 0.122), interpolation="bilinear")
    cv.add_patch(Rectangle((bx0, 0.078), bx1 - bx0, 0.044, fill=False,
                           ec=RULE, lw=0.6))
    cv.text(bx0 - 0.04, 0.10, "young", ha="right", va="center",
            fontsize=FS_NOTE, color=RULE)
    cv.text(bx1 + 0.04, 0.10, "old", ha="left", va="center",
            fontsize=FS_NOTE, color=RULE)
    save_asset(fig, folder, "b5_age_scale")

    for idx, key in enumerate(("in", "f", "s"), start=6):
        text, colour = STAGE_TITLES[key]
        fig = plt.figure(figsize=(1.60, 0.22))
        fig.text(0.5, 0.5, text, ha="center", va="center",
                 fontsize=FS_TITLE, color=colour)
        save_asset(fig, folder, "b%d_title_%s" % (idx, STAGE_NAMES[key]))

    # -- schematic assets, drawn in the composite's inch coordinates -------
    def canvas(x0, x1, y0, y1):
        f = plt.figure(figsize=(x1 - x0, y1 - y0))
        c = f.add_axes([0, 0, 1, 1])
        c.set_xlim(x0, x1)
        c.set_ylim(y0, y1)
        c.set_aspect("equal")
        c.axis("off")
        return f, c

    boxes = {
        "b9_channel_schematic": (2.16, 3.86, 1.58, 2.34),
        "b10_channel_with_curved_arrow": (2.16, 4.26, 1.58, 2.44),
        "b11_curved_arrow": (3.20, 4.26, 2.08, 2.44),
        "b12_label_Q": (0.0, 0.40, 0.0, 0.20),
    }
    f, c = canvas(*boxes["b9_channel_schematic"])
    draw_channel(c, flow_arrows=False, age_scale=False)
    save_asset(f, folder, "b9_channel_schematic")

    f, c = canvas(*boxes["b10_channel_with_curved_arrow"])
    anchor = draw_channel(c, flow_arrows=False, age_scale=False)
    draw_zone_to_kernel_arrow(c, anchor)
    save_asset(f, folder, "b10_channel_with_curved_arrow")

    f, c = canvas(*boxes["b11_curved_arrow"])
    draw_zone_to_kernel_arrow(c, (CHANNEL["zx1"], CHANNEL["yt"] + 0.06))
    save_asset(f, folder, "b11_curved_arrow")

    f, c = canvas(*boxes["b12_label_Q"])
    c.text(0.20, 0.10, r"$Q(t)$", ha="center", va="center",
           fontsize=FS_TITLE, color=BLACK)
    save_asset(f, folder, "b12_label_Q")

    return folder, size, ksize, boxes


def main() -> None:
    time_s, c_in, c_f, s, theta, kernel = simulate()
    t = time_s / 60.0

    stage = {"transport": metrics(time_s, c_in, c_f),
             "chain": metrics(time_s, c_in, s),
             "sensor_only": metrics(time_s, c_f, s)}
    checked = assert_matches_grid(stage)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    cv = fig.add_axes([0, 0, 1, 1], zorder=0)
    cv.set_xlim(0, FIG_W)
    cv.set_ylim(0, FIG_H)
    cv.set_aspect("equal")
    cv.axis("off")

    col_background(cv, "bio", BIO, BIO_BG, "Biology")
    col_background(cv, "dev", DEV, DEV_BG, "Device fluidics")
    col_background(cv, "sen", SEN, SEN_BG, "Sensor")

    illus.draw_skin(cv, PLACEHOLDER[0], PLACEHOLDER[1])
    zone_anchor = draw_channel(cv)
    illus.draw_sensor(cv, *SENSOR_ORIGIN)
    draw_kernel(fig, cv, theta, kernel, zone_anchor)

    # -- signal row --------------------------------------------------------
    for key in ("in", "f", "s"):
        plot_stage(signal_axes(fig, key), key, t, c_in, c_f, s)
    for key, (text, colour) in STAGE_TITLES.items():
        cv.text(PLOT_X[key] + 0.5 * PLOT_W, PLOT_B + PLOT_H + 0.07, text,
                ha="center", va="bottom", fontsize=FS_TITLE, color=colour,
                zorder=5)

    ymid = PLOT_B + 0.5 * PLOT_H
    arrow(cv, (PLOT_X["in"] + PLOT_W + 0.06, ymid), (PLOT_X["f"] - 0.26, ymid))
    arrow(cv, (PLOT_X["f"] + PLOT_W + 0.06, ymid), (PLOT_X["s"] - 0.26, ymid))

    cv.text(0.10, 2.53, "b", ha="left", va="top", fontsize=8.0,
            fontweight="bold", color=BLACK, zorder=6)

    for ext in ("pdf", "svg"):
        fig.savefig(OUT / (STEM + "." + ext), facecolor="white")
    fig.savefig(OUT / (STEM + ".png"), dpi=600, facecolor="white")
    plt.close(fig)

    folder, sig_size, ker_size, schematic_boxes = export_individual(
        t, c_in, c_f, s, theta, kernel)

    # -- records -----------------------------------------------------------
    step = max(1, int(round(0.25 / (t[1] - t[0]))))
    with (OUT / "fig1b_plotted_values.csv").open("w", newline="",
                                                 encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time_min", "C_in", "C_f", "S"])
        for i in range(0, len(t), step):
            w.writerow(["%.2f" % t[i], "%.6f" % c_in[i], "%.6f" % c_f[i],
                        "%.6f" % s[i]])
    with (OUT / "fig1b_kernel_E_theta.csv").open("w", newline="",
                                                 encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["theta_min", "E_per_min"])
        for i in range(0, len(theta), step):
            if theta[i] > 3 * THETA_MAX:
                break
            w.writerow(["%.2f" % theta[i], "%.6f" % kernel[i]])

    mom = {name: moments(t, y) for name, y in
           (("C_in", c_in), ("C_f", c_f), ("S", s))}
    k_area = float(np.trapezoid(kernel, theta))
    k_mean = float(np.trapezoid(theta * kernel, theta))
    record = {
        "panel": "Figure 1b",
        "provenance": "rows of 02_canonical_extended_grid_1620.csv",
        "grid_rows_checked": checked,
        "waveform": {"kind": WAVEFORM, "fwhm_T_min": T_MIN},
        "kernel": {"topology": TOPOLOGY, "shape_k": RTD_SHAPE,
                   "mean_residence_tau_f_min": TAU_F,
                   "numerical_area": k_area, "numerical_mean_min": k_mean},
        "sensor": {"model": "first order, g linear", "tau_s_min": TAU_S},
        "stage_metrics": stage,
        "moments_area_centroid_sd": {k: {"area": v[0], "centroid_min": v[1],
                                         "sd_min": v[2]}
                                     for k, v in mom.items()},
        "placeholder_region_in": dict(zip(("x0", "y0", "x1", "y1"),
                                          PLACEHOLDER)),
        "figure_size_in": [FIG_W, FIG_H],
        "individual_assets": {
            "folder": folder.relative_to(ROOT).as_posix(),
            "signal_asset_size_in": [round(v, 3) for v in sig_size],
            "signal_axes_offset_in": {"left": SIGNAL_MARGINS[0],
                                      "bottom": SIGNAL_MARGINS[2]},
            "signal_axes_size_in": [PLOT_W, PLOT_H],
            "kernel_asset_size_in": [round(v, 3) for v in ker_size],
            "kernel_axes_offset_in": {"left": KERNEL_MARGINS[0],
                                      "bottom": KERNEL_MARGINS[2]},
            "kernel_axes_size_in": [KERNEL_BOX[2], KERNEL_BOX[3]],
            "schematic_canvas_in_composite_coords": {
                k: {"x0": v[0], "x1": v[1], "y0": v[2], "y1": v[3]}
                for k, v in schematic_boxes.items() if k != "b12_label_Q"},
        },
    }
    (OUT / "fig1b_parameters_and_metrics.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")

    print("grid agreement verified: " + "; ".join(checked))
    print("kernel area %.4f, mean %.3f min (tau_f %.1f)" % (k_area, k_mean, TAU_F))
    for k, (a, c, sd) in mom.items():
        print("%-5s area %.4f  centroid %.3f  sd %.3f" % (k, a, c, sd))
    print("centroid shifts: transport %+.3f (expect %.2f), sensor %+.3f (expect %.2f)"
          % (mom["C_f"][1] - mom["C_in"][1], TAU_F,
             mom["S"][1] - mom["C_f"][1], TAU_S))
    for k, m in stage.items():
        print("%-12s amp %.3f  delay %+.2f  nrmse %.3f  r %.3f"
              % (k, m["amplitude_retention"], m["feature_delay_min"],
                 m["nrmse"], m["pearson_r"]))


if __name__ == "__main__":
    main()
