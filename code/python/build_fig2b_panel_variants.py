"""Complete Fig 2b panel mock-ups in two styles, for side-by-side choice.

v2  dashed reference only: behind each stage, a faint dashed copy of that
    stage's own input (C_in behind C_f, C_f behind S), plus the input-peak line.
v3  v2 plus the amplitude retention A and peak delay of each stage, printed
    beside the curve.

Both are assembled as the full panel, with column titles, row labels and time
axes, so they can be compared as they would appear in the figure. Curves come
from the released panel data; nothing is recomputed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / "outputs" / "figures_final" / "figure2_placement_ready" / "data"
OUT = PROJECT / "outputs" / "revision_2026-09-23" / "figure2b_contrast"
BLACK, GRAY, BLUE, PURPLE = "#151515", "#8C8C8C", "#245DB8", "#6B2AA6"
OFFSET = {"C_in": 2.4, "C_f": 1.2, "S": 0.0}
DASH = (0, (2.2, 1.6))
COLUMNS = [("plug_flow", "Plug flow"), ("distributed_RTD", "Distributed RTD"),
           ("well_mixed", "Well mixed")]
W, H = 5.10, 2.95          # panel size in inches
LEFT, RIGHT, TOP, BOTTOM = 0.62, 0.06, 0.34, 0.40
GAP = 0.16


def build(annotate: bool):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    col_w = (W - LEFT - RIGHT - 2 * GAP) / 3
    ax_h = H - TOP - BOTTOM
    axes = []
    for i, (topology, title) in enumerate(COLUMNS):
        x0 = LEFT + i * (col_w + GAP)
        ax = fig.add_axes([x0 / W, BOTTOM / H, col_w / W, ax_h / H])
        axes.append(ax)
        d = pd.read_csv(SRC / f"fig2b_{topology}_T10.csv")
        t = d.time_min.to_numpy()
        y = {k: d[k].to_numpy() for k in ("C_in", "C_f", "S")}
        amp_in = y["C_in"].max() - y["C_in"].min()
        tp = {k: t[v.argmax()] for k, v in y.items()}

        ax.axvline(tp["C_in"], color=GRAY, lw=0.9, ls=DASH, zorder=1)
        for stage, ref in (("C_f", "C_in"), ("S", "C_f")):
            ax.plot(t, y[ref] + OFFSET[stage], color=GRAY, lw=1.1, ls=DASH, zorder=2)
        for key, color in (("C_in", BLACK), ("C_f", BLUE), ("S", PURPLE)):
            ax.plot(t, y[key] + OFFSET[key], color=color, lw=2.0, zorder=3)
            ax.plot([tp[key]], [y[key].max() + OFFSET[key]], "o", ms=3.6,
                    mfc=color, mec="white", mew=0.7, zorder=4)
        if annotate:
            for key, color in (("C_f", BLUE), ("S", PURPLE)):
                a = (y[key].max() - y[key].min()) / amp_in
                dt = tp[key] - tp["C_in"]
                ax.text(t[-1], OFFSET[key] + 1.04,
                        f"A = {a:.2f}\n$\\Delta t$ = {dt:+.1f} min",
                        ha="right", va="top", fontsize=6.6, color=color,
                        linespacing=1.3, zorder=5)
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.set_ylim(-0.10, 3.62)
        ax.axis("off")
        ax.set_title(title, fontsize=8.5, fontweight="bold", color=BLACK, pad=4)
        # time axis
        ax.annotate("", xy=(0.98, -0.055), xytext=(0.02, -0.055),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color=BLACK,
                                    shrinkA=0, shrinkB=0, mutation_scale=7))
        ax.text(0.5, -0.135, "Time", transform=ax.transAxes, ha="center",
                va="top", fontsize=8.0, color=BLACK)

    labels = [("$C_{in}(t)$", "C_in", BLACK), ("$C_{f}(t)$", "C_f", BLUE),
              ("$S(t)$", "S", PURPLE)]
    for text, key, color in labels:
        yy = (BOTTOM + ax_h * (OFFSET[key] + 0.45) / 3.62) / H
        fig.text((LEFT - 0.10) / W, yy, text, ha="right", va="center",
                 fontsize=9.0, color=color)
    return fig


def main():
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    for sub in ("png", "pdf", "svg"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    for tag, annotate in (("v2_reference", False), ("v3_annotated", True)):
        fig = build(annotate)
        stem = f"fig2b_panel_{tag}"
        fig.savefig(OUT / "png" / f"{stem}.png", dpi=600, facecolor="white")
        fig.savefig(OUT / "pdf" / f"{stem}.pdf", facecolor="white")
        fig.savefig(OUT / "svg" / f"{stem}.svg", facecolor="white")
        plt.close(fig)
    print("wrote", OUT / "png")


if __name__ == "__main__":
    main()
