"""Fig 2b variants that make the topology differences visible.

v2  dashed reference: each stage carries a faint dashed copy of its own input
    (C_in behind C_f, C_f behind S), the convention already used in Figs 1b and 4b,
    plus the input-peak line kept from the current panel.
v3  v2 plus the amplitude retention A and peak delay of each stage, printed
    next to the curve.

Canvas (2.55 x 2.75 in), axis limits and colors match the panels currently in
figs.pptx, so each file is a drop-in replacement for its column.

Curves are read from the released panel data, so nothing is recomputed.
Canvas and colors match build_figure2_placement_ready.export_b.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / "outputs" / "figures_final" / "figure2_placement_ready" / "data"
OUT = PROJECT / "outputs" / "revision_2026-09-23" / "figure2b_contrast"
BLACK, GRAY, BLUE, PURPLE = "#151515", "#8C8C8C", "#245DB8", "#6B2AA6"
OFFSET = {"C_in": 2.4, "C_f": 1.2, "S": 0.0}
DASH = (0, (2.2, 1.6))


def style():
    plt.rcParams.update({"font.family": "Arial", "font.size": 8.0,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})


def panel(topology: str, annotate: bool):
    d = pd.read_csv(SRC / f"fig2b_{topology}_T10.csv")
    t = d.time_min.to_numpy()
    y = {k: d[k].to_numpy() for k in ("C_in", "C_f", "S")}
    amp_in = y["C_in"].max() - y["C_in"].min()
    t_peak = {k: t[v.argmax()] for k, v in y.items()}

    fig, ax = plt.subplots(figsize=(2.55, 2.75))
    ax.axvline(t_peak["C_in"], color=GRAY, lw=0.9, ls=DASH, zorder=1)
    # faint dashed copy of the input to each stage, on that stage's baseline
    for stage, ref in (("C_f", "C_in"), ("S", "C_f")):
        ax.plot(t, y[ref] + OFFSET[stage], color=GRAY, lw=1.2, ls=DASH, zorder=2)
    for key, color in (("C_in", BLACK), ("C_f", BLUE), ("S", PURPLE)):
        ax.plot(t, y[key] + OFFSET[key], color=color, lw=2.2, zorder=3)
        ax.plot([t_peak[key]], [y[key].max() + OFFSET[key]], "o", ms=4.0,
                mfc=color, mec="white", mew=0.8, zorder=4)

    if annotate:
        for key, color in (("C_f", BLUE), ("S", PURPLE)):
            a = (y[key].max() - y[key].min()) / amp_in
            dt = t_peak[key] - t_peak["C_in"]
            ax.text(t[-1], OFFSET[key] + 1.02, f"A = {a:.2f}\n$\\Delta t$ = {dt:+.1f} min",
                    ha="right", va="top", fontsize=7.0, color=color, linespacing=1.25)
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_ylim(-0.08, 3.5)
    ax.axis("off")
    return fig


def main():
    style()
    for sub in ("png", "pdf", "svg"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    for topology in ("plug_flow", "distributed_RTD", "well_mixed"):
        for tag, annotate in (("v2_reference", False), ("v3_annotated", True)):
            fig = panel(topology, annotate)
            stem = f"fig2b_{topology}_T10_{tag}"
            fig.savefig(OUT / "png" / f"{stem}.png", dpi=600, transparent=True)
            fig.savefig(OUT / "pdf" / f"{stem}.pdf", transparent=True)
            fig.savefig(OUT / "svg" / f"{stem}.svg", transparent=True)
            plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
