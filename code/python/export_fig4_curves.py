"""Figure 4 curves: shared drawing helpers and standalone assets.

The draw_* helpers put one curve set into a given axes and return the legend
entries, so the standalone assets written here and the composite layout in
plot_fig4_layout.py draw exactly the same curves.

Each data-style plot has axis titles centred along the axes ("Time",
"Concentration"), as in Figures 1-3; no tick numbers (schematic, arbitrary
units). Each standalone plot carries its own legend.

Panel a assets
  a1_tolerance                threshold crossing, Delta t_max, Delta A_max
  a2_1_bio_reference          blood (dashed) vs sweat, H_bio
  a2_2_transport_tracer       inlet tracer (dashed) vs outlet, E(theta)
  a2_3_sensor_step            concentration step (dashed) vs sensor, H_sensor
  a3_1_pulse / a3_2_ramp / a3_3_sinusoid
                              prediction (line) vs measured (points)
Panel b assets
  b1-b5 level glyphs, pictograms without axes.

PDF / SVG / PNG (600 dpi), transparent background.
Lettering >= 4.5 pt (math subscripts included); strokes >= 0.6 pt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_fig4_validation as F  # noqa: E402  (shared style and glyphs)
import matplotlib.pyplot as plt  # noqa: E402
from run_evidence_closure_analysis import canonical_waveform  # noqa: E402
from transport_fidelity_models import (  # noqa: E402
    first_order_filter,
    transport_sensor_response,
)

OUT = F.OUT / "individual"
OUT.mkdir(parents=True, exist_ok=True)

DASH = (0, (2.5, 1.8))
REF_KW = dict(color=F.GRAY, lw=0.8, ls=DASH)
COMPONENTS = [
    # key, symbol, colour, reference label, response label, asset name
    ("bio", r"$H_{\mathrm{bio}}$", F.BIO, "blood", "sweat", "a2_1_bio_reference"),
    ("tracer", r"$E(\theta)$", F.DEV, "inlet", "outlet", "a2_2_transport_tracer"),
    ("step", r"$H_{\mathrm{sensor}}$", F.SEN, "step", "sensor", "a2_3_sensor_step"),
]
HELDOUT = [("pulse", (48, 80), "a3_1_pulse"), ("ramp", (30, 56), "a3_2_ramp"),
           ("sinusoid", (140, 170), "a3_3_sinusoid")]


# ------------------------------------------------------------ helpers
def style_axes(ax, fs=6.5, ylabel=True, xlabel=True):
    ax.set_facecolor("none")
    ax.set_xticks([])
    ax.set_yticks([])
    if xlabel:
        ax.set_xlabel("Time", fontsize=fs, labelpad=2.0)
    if ylabel:
        ax.set_ylabel("Concentration", fontsize=fs, labelpad=2.0)


def draw_tolerance(ax):
    t = np.linspace(0, 60, 1201)
    c = 0.10 + 0.85 * np.exp(-0.5 * ((t - 30) / 7.5) ** 2)
    thr = 0.55
    ts = float(t[int(np.argmax(c >= thr))])
    peak = float(c.max())
    dt, da = 4.0, 0.09
    ax.fill_between([ts - dt, ts + dt], -0.05, 0.72, color=F.TOL, lw=0, zorder=0)
    ax.fill_between([24, 36], peak - da, peak + da, color=F.TOL, lw=0, zorder=0)
    ax.axhline(thr, color=F.GRAY, lw=0.7, ls=DASH, zorder=1)
    ax.plot(t, c, color=F.BLACK, lw=F.LW, zorder=3)
    ax.plot([ts], [thr], "o", ms=3.2, mfc=F.BLACK, mec="white", mew=0.6,
            zorder=4)
    ax.annotate("", xy=(ts + dt, 0.12), xytext=(ts - dt, 0.12),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=F.BLACK,
                                shrinkA=0, shrinkB=0, mutation_scale=5))
    ax.text(ts + dt + 1.5, 0.12, r"$\Delta t_{\max}$", ha="left", va="center",
            fontsize=F.FS_MATH)
    ax.annotate("", xy=(38, peak + da), xytext=(38, peak - da),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=F.BLACK,
                                shrinkA=0, shrinkB=0, mutation_scale=5))
    ax.text(39.5, peak, r"$\Delta A_{\max}$", ha="left", va="center",
            fontsize=F.FS_MATH)
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.05, 1.12)
    return [("threshold", "line", dict(color=F.GRAY, lw=0.7, ls=DASH))]


def component_curves():
    t = np.linspace(0, 40, 801)
    blood = 0.10 + 0.85 * np.exp(-0.5 * ((t - 14) / 4.0) ** 2)
    sweat = first_order_filter(t * 60, blood, 4.0 * 60)
    sweat = 0.10 + 0.75 * (sweat - 0.10) / (sweat.max() - 0.10)
    inlet = np.where((t >= 5) & (t <= 6.5), 0.95, 0.05)
    outlet = transport_sensor_response(t * 60, inlet, topology="gamma_rtd",
                                       transport_time_s=6.0 * 60,
                                       sensor_tau_s=None, rtd_shape=3.0)
    outlet = 0.05 + (outlet - 0.05) / (outlet.max() - 0.05) * 0.55
    step = np.where(t >= 8, 0.9, 0.1)
    sensor = first_order_filter(t * 60, step, 4.0 * 60)
    return t, {"bio": (blood, sweat), "tracer": (inlet, outlet),
               "step": (step, sensor)}


def draw_component(ax, key):
    t, curves = component_curves()
    ref, out = curves[key]
    _, sym, col, lref, lout, _ = next(c for c in COMPONENTS if c[0] == key)
    ax.plot(t, ref, **REF_KW)
    ax.plot(t, out, color=col, lw=F.LW)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 1.1)
    return [(lref, "line", REF_KW), (lout, "line", dict(color=col, lw=F.LW))]


def draw_heldout(ax, kind, window, rng):
    pre = dict(tau_f=2.0, tau_s=1.0)
    true = dict(tau_f=2.4, tau_s=1.2)
    ts, u = canonical_waveform(kind, 10.0)
    tm = ts / 60.0
    kw = dict(topology="gamma_rtd", rtd_shape=3.0)
    pred = transport_sensor_response(ts, u, transport_time_s=pre["tau_f"] * 60,
                                     sensor_tau_s=pre["tau_s"] * 60, **kw)
    meas = transport_sensor_response(ts, u, transport_time_s=true["tau_f"] * 60,
                                     sensor_tau_s=true["tau_s"] * 60, **kw)
    a, b = window
    m = (tm >= a) & (tm <= b)
    ax.plot(tm[m], pred[m], color=F.PRED, lw=F.LW)
    idx = np.where(m)[0][::30]
    ax.plot(tm[idx], meas[idx] + rng.normal(0, 0.025, len(idx)), "o",
            ms=1.9, mfc=F.BLACK, mec="none")
    ax.set_xlim(a, b)
    ax.set_ylim(-0.08, 1.12)
    return [("prediction", "line", dict(color=F.PRED, lw=F.LW)),
            ("measured", "dot", dict(ms=1.9, mfc=F.BLACK, mec="none"))]


def draw_key(fig, cv, entries, xc, y, fs=6.0, limits=None):
    """One-row legend centred at (xc, y) in canvas inches, spaced by the
    measured text widths; shifted inside `limits` (x0, x1) if given."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    widths = []
    for lab, _, _ in entries:
        probe = cv.text(0, 0, lab, fontsize=fs)
        widths.append(probe.get_window_extent(renderer=rend).width / fig.dpi)
        probe.remove()
    handle, pad, gap = 0.14, 0.04, 0.12
    total = sum(handle + pad + w for w in widths) + gap * (len(entries) - 1)
    x = xc - total / 2
    if limits is not None:
        x = min(max(x, limits[0]), limits[1] - total)
    for (lab, kind, kw), w in zip(entries, widths):
        if kind == "line":
            cv.plot([x, x + handle], [y, y], **kw)
        else:
            cv.plot([x + handle / 2], [y], "o", **kw)
        cv.text(x + handle + pad, y, lab, ha="left", va="center", fontsize=fs,
                color=F.BLACK, zorder=8)
        x += handle + pad + w + gap


# ------------------------------------------------------- standalone assets
W, H = 1.42, 1.30
AX = (0.36, 0.26, 0.98, 0.64)
TITLE_Y, KEY_Y = 1.20, 1.05


def asset(draw, title=None, colour=F.BLACK):
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([AX[0] / W, AX[1] / H, AX[2] / W, AX[3] / H])
    style_axes(ax)
    entries = draw(ax)
    cv = fig.add_axes([0, 0, 1, 1], zorder=-1)
    cv.set_xlim(0, W)
    cv.set_ylim(0, H)
    cv.axis("off")
    if title:
        cv.text(AX[0] + AX[2] / 2, TITLE_Y, title, ha="center", va="center",
                fontsize=7.0, color=colour)
    draw_key(fig, cv, entries, AX[0] + AX[2] / 2, KEY_Y, limits=(0.03, W - 0.03))
    return fig


def save(fig, name, folder=OUT):
    for ext in ("pdf", "svg"):
        fig.savefig(folder / (name + "." + ext), transparent=True)
    fig.savefig(folder / (name + ".png"), dpi=600, transparent=True)
    plt.close(fig)


def export_all():
    save(asset(draw_tolerance), "a1_tolerance")
    for key, sym, col, _, _, name in COMPONENTS:
        save(asset(lambda ax, k=key: draw_component(ax, k), sym, col), name)
    rng = np.random.default_rng(5)
    for kind, window, name in HELDOUT:
        save(asset(lambda ax, k=kind, w=window: draw_heldout(ax, k, w, rng),
                   kind), name)
    rng = np.random.default_rng(6)
    gw, gh = F.B_W - 0.20, F.B_H
    names = ["b1_detectable", "b2_component_characterized",
             "b3_dynamically_predictive", "b4_physiologically_interpretable",
             "b5_clinically_useful"]
    for k, name in enumerate(names):
        fig = plt.figure(figsize=(gw + 0.08, gh + 0.08))
        ax = fig.add_axes([0.04 / (gw + 0.08), 0.04 / (gh + 0.08),
                           gw / (gw + 0.08), gh / (gh + 0.08)])
        ax.set_facecolor("none")
        ax.set_xticks([])
        ax.set_yticks([])
        F.glyph(ax, k, rng)
        save(fig, name)


if __name__ == "__main__":
    export_all()
    print("wrote", OUT)
