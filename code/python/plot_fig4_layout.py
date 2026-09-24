"""Figure 4, layout v5: panel a as a 2 x 2 grid, panel b as a vertical ladder.

a  four numbered cards in reading order (1 2 / 3 4); every curve comes from
   export_fig4_curves, so the composite and the standalone assets agree.
   Card 4 holds placeholders for the author's BioRender artwork.
b  claim levels stacked bottom (Detectable) to top (Clinically useful); the
   upward arrows name the evidence for the next level and the stage of
   panel a that supplies it.

Format: 7.00 x 3.28 in; lettering >= 4.5 pt (math subscripts included);
strokes >= 0.6 pt. Backend: Python / matplotlib.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_fig4_validation as F  # noqa: E402
import export_fig4_curves as C  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = F.OUT
STEM = "fig4_layout_v5"
FIG_W, FIG_H = 7.00, 3.28

# panel a grid
A_X0, A_X1, A_GAP_X = 0.20, 4.52, 0.14
A_Y0, A_Y1, A_GAP_Y = 0.12, 3.18, 0.14
CARD_W = (A_X1 - A_X0 - A_GAP_X) / 2
CARD_H = (A_Y1 - A_Y0 - A_GAP_Y) / 2
TITLES = ["Set tolerance", "Characterize components",
          "Predict held-out inputs", "Interpret in context"]
FS_AX = 6.0

# panel b ladder
B_X0, BOX_W = 4.80, 0.90
BOX_H = 0.42
B_Y0, B_Y1 = A_Y0, A_Y1
B_GAP = (B_Y1 - B_Y0 - 5 * BOX_H) / 4
LEVELS = ["Detectable", "Component-\ncharacterized", "Dynamically\npredictive",
          "Physiologically\ninterpretable", "Clinically\nuseful"]
EVIDENCE = ["+ component tests (2)", "+ held-out prediction (3)",
            "+ on-body reference (4)", "+ decision validation (1)"]

FRAME = [0.0, 0.0, FIG_W, FIG_H]


def axes_in(fig, x0, y0, w, h):
    ox, oy, W, H = FRAME
    return fig.add_axes([(x0 - ox) / W, (y0 - oy) / H, w / W, h / H])


def card_box(i):
    col, row = i % 2, i // 2
    x0 = A_X0 + col * (CARD_W + A_GAP_X)
    y1 = A_Y1 - row * (CARD_H + A_GAP_Y)
    return x0, y1 - CARD_H, x0 + CARD_W, y1


def card_frame(cv, i):
    x0, y0, x1, y1 = card_box(i)
    F.card(cv, x0, x1, y0, y1)
    cv.text(x0 + 0.10, y1 - 0.13, "%d" % (i + 1), ha="left", va="center",
            fontsize=F.FS_TITLE, fontweight="bold", color=F.GRAY, zorder=8)
    cv.text(x0 + 0.24, y1 - 0.13, TITLES[i], ha="left", va="center",
            fontsize=F.FS_TITLE, fontweight="bold", color=F.BLACK, zorder=8)
    return x0, y0, x1, y1


def three_axes(fig, x0, y0, x1, y1):
    w, gap = 0.48, 0.16
    # centred in the card, nudged right to leave room for the y title
    start = x0 + 0.5 * (x1 - x0 - (3 * w + 2 * gap)) + 0.06
    top, bottom = y1 - 0.46, y0 + 0.44
    axs = []
    for k in range(3):
        ax = axes_in(fig, start + k * (w + gap), bottom, w, top - bottom)
        C.style_axes(ax, fs=FS_AX, ylabel=(k == 0))
        axs.append(ax)
    return axs, [start + k * (w + gap) + w / 2 for k in range(3)], top


def card1(fig, cv):
    x0, y0, x1, y1 = card_frame(cv, 0)
    ax = axes_in(fig, x0 + 0.52, y0 + 0.44, x1 - x0 - 0.90, CARD_H - 0.78)
    C.style_axes(ax, fs=FS_AX)
    entries = C.draw_tolerance(ax)
    C.draw_key(fig, cv, entries, 0.5 * (x0 + x1), y0 + 0.11)


def card2(fig, cv):
    x0, y0, x1, y1 = card_frame(cv, 1)
    axs, centres, top = three_axes(fig, x0, y0, x1, y1)
    for ax, xc, (key, sym, col, *_rest) in zip(axs, centres, C.COMPONENTS):
        C.draw_component(ax, key)
        cv.text(xc, top + 0.10, sym, ha="center", va="center",
                fontsize=F.FS_MATH, color=col, zorder=8)
    # key: dashed applied input, response in the three component colours
    fig.canvas.draw()
    kx, ky = 0.5 * (x0 + x1) - 0.55, y0 + 0.11
    cv.plot([kx, kx + 0.14], [ky, ky], **C.REF_KW)
    cv.text(kx + 0.18, ky, "input or reference", ha="left", va="center",
            fontsize=6.0, color=F.BLACK, zorder=8)
    sx = kx + 0.98
    for j, col in enumerate((F.BIO, F.DEV, F.SEN)):
        cv.plot([sx + 0.05 * j, sx + 0.05 * (j + 1)], [ky, ky], color=col,
                lw=F.LW, solid_capstyle="butt")
    cv.text(sx + 0.19, ky, "response", ha="left", va="center", fontsize=6.0,
            color=F.BLACK, zorder=8)


def card3(fig, cv, rng):
    x0, y0, x1, y1 = card_frame(cv, 2)
    axs, centres, top = three_axes(fig, x0, y0, x1, y1)
    entries = None
    for ax, xc, (kind, window, _name) in zip(axs, centres, C.HELDOUT):
        entries = C.draw_heldout(ax, kind, window, rng)
        cv.text(xc, top + 0.10, kind, ha="center", va="center", fontsize=6.0,
                color=F.BLACK, zorder=8)
    C.draw_key(fig, cv, entries, 0.5 * (x0 + x1), y0 + 0.11)


def placeholder(cv, x0, y0, w, h, label=None):
    cv.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle="round,pad=0,rounding_size=0.03",
                                fc="white", ec=F.GRAY, lw=0.6,
                                ls=(0, (2.5, 1.8)), zorder=2))
    if label:
        cv.text(x0 + w / 2, y0 + h / 2, label, ha="center", va="center",
                fontsize=5.2, color=F.GRAY, style="italic", zorder=8,
                linespacing=1.15)


def card4(cv):
    x0, y0, x1, y1 = card_frame(cv, 3)
    boxes = {}
    arm = (x0 + 0.16, y0 + 0.62, x1 - x0 - 0.32, y1 - y0 - 0.62 - 0.30)
    placeholder(cv, *arm, label="forearm with patch\n(BioRender)")
    boxes["forearm"] = arm
    labels = ("site", "sweat rate", "reference", "cohort")
    slot = (x1 - x0 - 0.20) / 4
    boxes["icons"] = []
    for i, lab in enumerate(labels):
        xc = x0 + 0.10 + slot * (i + 0.5)
        b = (xc - 0.10, y0 + 0.27, 0.20, 0.20)
        placeholder(cv, *b)
        boxes["icons"].append(b)
        cv.text(xc, y0 + 0.14, lab, ha="center", va="center", fontsize=6.0,
                color=F.BLACK, zorder=8)
    return boxes


def ladder(fig, cv, rng):
    for k in range(5):
        y0 = B_Y0 + k * (BOX_H + B_GAP)
        F.card(cv, B_X0, B_X0 + BOX_W, y0, y0 + BOX_H, fill=F.LEVEL_FILL[k])
        ax = axes_in(fig, B_X0 + 0.08, y0 + 0.05, BOX_W - 0.16, BOX_H - 0.10)
        ax.set_facecolor("none")
        ax.set_xticks([])
        ax.set_yticks([])
        F.glyph(ax, k, rng)
        cv.text(B_X0 + BOX_W + 0.10, y0 + BOX_H / 2, LEVELS[k], ha="left",
                va="center", fontsize=F.FS_LABEL, fontweight="bold",
                color=F.BLACK, zorder=8, linespacing=1.15)
        if k < 4:
            xa = B_X0 + BOX_W / 2
            ya, yb = y0 + BOX_H + 0.03, y0 + BOX_H + B_GAP - 0.03
            cv.add_patch(FancyArrowPatch((xa, ya), (xa, yb), arrowstyle="-|>",
                                         mutation_scale=6, lw=0.9,
                                         color=F.RULE, shrinkA=0, shrinkB=0,
                                         zorder=6))
            cv.text(xa + 0.10, 0.5 * (ya + yb), EVIDENCE[k], ha="left",
                    va="center", fontsize=6.0, color=F.RULE, zorder=8)


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    cv = fig.add_axes([0, 0, 1, 1], zorder=0)
    cv.set_xlim(0, FIG_W)
    cv.set_ylim(0, FIG_H)
    cv.set_aspect("equal")
    cv.axis("off")

    card1(fig, cv)
    card2(fig, cv)
    card3(fig, cv, np.random.default_rng(5))
    boxes = card4(cv)
    ladder(fig, cv, np.random.default_rng(6))
    cv.text(0.02, FIG_H - 0.02, "a", ha="left", va="top", fontsize=8.0,
            fontweight="bold")
    cv.text(B_X0 - 0.18, FIG_H - 0.02, "b", ha="left", va="top", fontsize=8.0,
            fontweight="bold")

    for ext in ("pdf", "svg"):
        fig.savefig(OUT / (STEM + "." + ext), facecolor="white")
    fig.savefig(OUT / (STEM + ".png"), dpi=600, facecolor="white")
    plt.close(fig)

    layout = {"figure_in": [FIG_W, FIG_H],
              "cards_x0_y0_x1_y1": [[round(v, 3) for v in card_box(i)]
                                    for i in range(4)],
              "card4_biorender": {k: ([round(v, 3) for v in b] if k == "forearm"
                                      else [[round(v, 3) for v in bb] for bb in b])
                                  for k, b in boxes.items()},
              "ladder_boxes_x0_y0_w_h": [[B_X0, round(B_Y0 + k * (BOX_H + B_GAP), 3),
                                          BOX_W, BOX_H] for k in range(5)]}
    (OUT / (STEM + "_layout.json")).write_text(json.dumps(layout, indent=2),
                                               encoding="utf-8")
    print("wrote", OUT / STEM, "card %.2f x %.2f in" % (CARD_W, CARD_H))


if __name__ == "__main__":
    main()
