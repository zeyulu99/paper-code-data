"""Figure 1c - exchanging transport and sensor time constants leaves the output
unchanged, exactly.

A well-mixed sensing zone has an exponential sample-age distribution, and a
first-order sensor has an exponential impulse response. Both are models the
manuscript already uses. Two pathways that swap the two time constants,
    A: tau_f = 5 min, tau_s = 1 min
    B: tau_f = 1 min, tau_s = 5 min
share the transfer function 1 / [(1 + 5s)(1 + s)], so S_A = S_B for every
input. Both pathways are rows of the released 1,620-point canonical grid; the
script checks them against it, and also reports that every well-mixed
label-exchanged pair in that grid agrees to machine precision whereas
distributed-RTD and plug-flow pairs do not.

Both stages are drawn as unit-area impulse responses on one shared axis, so
"broad transport with a fast sensor" and "narrow transport with a slow
sensor" are the same two shapes in swapped positions.

The input pulse and the time axis are those of Figure 1b.

Format  : 7.00 x 1.95 in, lettering >= 4.5 pt including math subscripts,
          strokes >= 0.6 pt. Individual assets are written without the
          straight connector arrows.
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
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "python"))
from run_evidence_closure_analysis import (  # noqa: E402
    canonical_metrics,
    canonical_waveform,
)
from transport_fidelity_models import transport_sensor_response  # noqa: E402
import plot_fig1b_measurement_cascade as f1b  # noqa: E402  (shared style)

OUT = ROOT / "outputs" / "revision_2026-09-17" / "figure1c"
OUT.mkdir(parents=True, exist_ok=True)
GRID = f1b.GRID
STEM = "fig1c_label_exchange"

# ----------------------------------------------------------------- parameters
WAVEFORM, T_MIN = f1b.WAVEFORM, f1b.T_MIN
TAU_BROAD, TAU_NARROW = 5.0, 1.0
PATHWAYS = {
    "A": dict(tau_f=TAU_BROAD, tau_s=TAU_NARROW),
    "B": dict(tau_f=TAU_NARROW, tau_s=TAU_BROAD),
}
VIEW_MIN, VIEW_MAX, X_TICKS = f1b.VIEW_MIN, f1b.VIEW_MAX, f1b.X_TICKS
K_MIN, K_MAX = -0.8, 15.0

FIG_W, FIG_H = 7.00, 1.95
BLACK, GRAY, RULE = f1b.BLACK, f1b.GRAY, f1b.RULE
DEV, SEN = f1b.DEV, f1b.SEN
PATH_A = "#8FA9BF"   # drawn thick and light
PATH_B = "#202020"   # drawn thin and dashed, on top

IN_BOX = (0.52, 0.52, 1.00, 0.82)            # x0, y0, w, h (in)
K_W, K_H = 0.95, 0.42
K_X = {"transport": 2.28, "sensor": 3.66}
K_Y = {"A": 1.08, "B": 0.42}
OUT_BOX = (5.36, 0.42, 1.50, 1.08)

FS_TITLE, FS_AXIS, FS_TICK, FS_NOTE = 7.0, 6.5, 6.0, 6.0
FS_MATH = 7.0        # subscripts render at 0.7x -> 4.9 pt
LW_CURVE, LW_REF = 1.4, 0.9
DASH = (0, (3.2, 2.2))


# ------------------------------------------------------------------- model
def simulate():
    time_s, c_in = canonical_waveform(WAVEFORM, T_MIN)
    out = {}
    for name, p in PATHWAYS.items():
        out[name] = transport_sensor_response(
            time_s, c_in, topology="cstr", transport_time_s=p["tau_f"] * 60.0,
            sensor_tau_s=p["tau_s"] * 60.0)
    return time_s, c_in, out


def check_grid(stats):
    with GRID.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    cols = [("amplitude_retention", "amplitude_retention"), ("nrmse", "NRMSE"),
            ("pearson_r", "pearson_r"), ("feature_delay_min", "feature_delay_min")]
    for name, p in PATHWAYS.items():
        match = [r for r in rows if r["waveform_type"] == WAVEFORM
                 and float(r["T_min"]) == T_MIN and r["topology"] == "well_mixed"
                 and float(r["tau_transport_min"]) == p["tau_f"]
                 and float(r["tau_sensor_min"]) == p["tau_s"]]
        if len(match) != 1:
            raise SystemExit("no unique grid row for pathway %s" % name)
        for key, col in cols:
            tol = 0.06 if key == "feature_delay_min" else 5e-3
            if abs(stats[name][key] - float(match[0][col])) > tol:
                raise SystemExit("pathway %s %s drifted from the grid" % (name, key))

    # every label-exchanged pair in the released grid, by topology
    index = {}
    for r in rows:
        key = (r["topology"], r["waveform_type"], float(r["T_min"]),
               float(r["tau_transport_min"]), float(r["tau_sensor_min"]))
        index[key] = np.array([float(r[c]) for _, c in cols])
    summary = {}
    for topo in ("well_mixed", "distributed_RTD", "plug_flow"):
        worst, n = 0.0, 0
        for (tp, w, T, a, b), v in index.items():
            if tp != topo or b == 0.0 or not a < b:
                continue
            other = index.get((tp, w, T, b, a))
            if other is None:
                continue
            n += 1
            worst = max(worst, float(np.max(np.abs(v - other))))
        summary[topo] = {"pairs": n, "max_abs_metric_difference": worst}
    return summary


def exp_kernel(x, tau):
    y = np.where(x >= 0, np.exp(-np.clip(x, 0, None) / tau) / tau, 0.0)
    return y


# ------------------------------------------------------------------ drawing
def plot_input(ax, t, c_in):
    f1b.style_signal(ax, True)
    ax.set_ylim(0, 1.12)
    ax.plot(t, c_in, color=BLACK, lw=LW_CURVE, label=f1b.LBL_IN)
    ax.legend(loc="upper right", fontsize=FS_MATH, frameon=False,
              handlelength=1.3, handletextpad=0.3, borderaxespad=0.15,
              borderpad=0.1)


def plot_kernel(ax, stage, tau, bottom_row):
    colour = DEV if stage == "transport" else SEN
    x = np.linspace(K_MIN, K_MAX, 1600)
    y = exp_kernel(x, tau)
    ax.set_facecolor("none")
    # fill only where the kernel is visibly non-zero (6 tau: 0.25 % of peak)
    fill = x <= min(K_MAX, 6.0 * tau)
    ax.fill_between(x[fill], 0, y[fill], color=colour, alpha=0.18, lw=0)
    ax.plot(x, y, color=colour, lw=1.1)
    ax.set_xlim(K_MIN, K_MAX)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([])
    ax.set_xticks([0, 5, 10, 15])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(pad=1.2, labelsize=FS_TICK)
    if bottom_row:
        ax.set_xlabel(r"Age $\theta$ (min)" if stage == "transport"
                      else "Time (min)", fontsize=FS_TICK, labelpad=1.0)
    else:
        ax.set_xticklabels([])
    sym = r"\tau_{\mathrm{f}}" if stage == "transport" else r"\tau_{\mathrm{s}}"
    ax.text(K_MAX, 1.14, r"$%s$ = %g min" % (sym, tau), ha="right", va="top",
            fontsize=FS_MATH, color=colour)


def plot_output(ax, t, c_in, out, show_ylabel=False):
    f1b.style_signal(ax, show_ylabel)
    ax.plot(t, c_in, color=GRAY, lw=LW_REF, ls=DASH, label=f1b.LBL_IN)
    ax.plot(t, out["A"], color=PATH_A, lw=3.0, solid_capstyle="round",
            label=r"$S_{\mathrm{A}}$")
    ax.plot(t, out["B"], color=PATH_B, lw=1.0, ls=(0, (2.6, 1.8)),
            label=r"$S_{\mathrm{B}}$")
    ax.legend(loc="upper right", fontsize=FS_MATH, frameon=False,
              handlelength=1.5, handletextpad=0.3, labelspacing=0.2,
              borderaxespad=0.15, borderpad=0.1)
    ax.text(VIEW_MAX - 0.03 * (VIEW_MAX - VIEW_MIN), 0.50,
            r"$S_{\mathrm{A}} \equiv S_{\mathrm{B}}$", ha="right",
            va="center", fontsize=8.0, color=BLACK)


def add_axes_in(fig, box):
    x0, y0, w, h = box
    return fig.add_axes([x0 / FIG_W, y0 / FIG_H, w / FIG_W, h / FIG_H])


def line(cv, xs, ys, lw=0.9):
    cv.add_line(Line2D(xs, ys, color=RULE, lw=lw, solid_capstyle="butt",
                       zorder=4))


def main() -> None:
    time_s, c_in, out = simulate()
    t = time_s / 60.0
    stats = {k: {kk: float(vv) for kk, vv in canonical_metrics(
        WAVEFORM, T_MIN, time_s, c_in, v).items() if kk != "feature_definition"}
        for k, v in out.items()}
    grid_pairs = check_grid(stats)
    max_diff = float(np.max(np.abs(out["A"] - out["B"])))
    if max_diff > 1e-9:
        raise SystemExit("pathways differ: %.3e" % max_diff)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    cv = fig.add_axes([0, 0, 1, 1], zorder=0)
    cv.set_xlim(0, FIG_W)
    cv.set_ylim(0, FIG_H)
    cv.set_aspect("equal")
    cv.axis("off")

    plot_input(add_axes_in(fig, IN_BOX), t, c_in)
    for row in ("A", "B"):
        p = PATHWAYS[row]
        for stage, tau in (("transport", p["tau_f"]), ("sensor", p["tau_s"])):
            ax = add_axes_in(fig, (K_X[stage], K_Y[row], K_W, K_H))
            plot_kernel(ax, stage, tau, bottom_row=(row == "B"))
    plot_output(add_axes_in(fig, OUT_BOX), t, c_in, out)

    # column headers
    head_y = 1.66
    heads = ((IN_BOX[0] + IN_BOX[2] / 2, "Input", BLACK),
             (K_X["transport"] + K_W / 2, r"Transport, $E(\theta)$", DEV),
             (K_X["sensor"] + K_W / 2, r"Sensor, $h(t)$", SEN),
             (OUT_BOX[0] + OUT_BOX[2] / 2, "Output", BLACK))
    for x, text, colour in heads:
        cv.text(x, head_y, text, ha="center", va="bottom", fontsize=FS_TITLE,
                fontweight="bold", color=colour)

    # pathway labels
    for row, colour in (("A", PATH_A), ("B", PATH_B)):
        cv.text(K_X["transport"] - 0.20, K_Y[row] + K_H / 2, row,
                ha="center", va="center", fontsize=8.0, fontweight="bold",
                color="#5F7D95" if row == "A" else PATH_B)

    # connectors: split, stage-to-stage, merge
    ya, yb = K_Y["A"] + K_H / 2, K_Y["B"] + K_H / 2
    ymid = IN_BOX[1] + IN_BOX[3] / 2
    xs = 1.70
    line(cv, [IN_BOX[0] + IN_BOX[2] + 0.06, xs], [ymid, ymid])
    line(cv, [xs, xs], [yb, ya])
    for y in (ya, yb):
        f1b.arrow(cv, (xs, y), (K_X["transport"] - 0.32, y))
        f1b.arrow(cv, (K_X["transport"] + K_W + 0.08, y),
                  (K_X["sensor"] - 0.08, y))
        line(cv, [K_X["sensor"] + K_W + 0.08, 4.86], [y, y])
    line(cv, [4.86, 4.86], [yb, ya])
    omid = OUT_BOX[1] + OUT_BOX[3] / 2
    f1b.arrow(cv, (4.86, omid), (OUT_BOX[0] - 0.24, omid))

    cv.text(0.06, 1.91, "c", ha="left", va="top", fontsize=8.0,
            fontweight="bold", color=BLACK)

    for ext in ("pdf", "svg"):
        fig.savefig(OUT / (STEM + "." + ext), facecolor="white")
    fig.savefig(OUT / (STEM + ".png"), dpi=600, facecolor="white")
    plt.close(fig)

    assets = export_individual(t, c_in, out)

    # records
    step = max(1, int(round(0.25 / (t[1] - t[0]))))
    with (OUT / "fig1c_plotted_values.csv").open("w", newline="",
                                                 encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time_min", "C_in", "S_A", "S_B"])
        for i in range(0, len(t), step):
            w.writerow(["%.2f" % t[i], "%.6f" % c_in[i], "%.6f" % out["A"][i],
                        "%.6f" % out["B"][i]])
    record = {
        "panel": "Figure 1c",
        "construction": "well-mixed transport (exponential E) and first-order "
                        "sensor (exponential h) with exchanged time constants; "
                        "transfer function 1/[(1+5s)(1+s)] for both pathways",
        "pathways": PATHWAYS,
        "waveform": {"kind": WAVEFORM, "fwhm_T_min": T_MIN},
        "max_abs_S_A_minus_S_B": max_diff,
        "pathway_metrics_vs_input": stats,
        "released_grid_label_exchange_pairs": grid_pairs,
        "individual_assets": assets,
    }
    (OUT / "fig1c_parameters_and_metrics.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")

    print("max |S_A - S_B| = %.3e" % max_diff)
    for k, v in stats.items():
        print("pathway %s: amp %.3f delay %+.2f nrmse %.3f r %.3f"
              % (k, v["amplitude_retention"], v["feature_delay_min"],
                 v["nrmse"], v["pearson_r"]))
    for k, v in grid_pairs.items():
        print("grid %-16s pairs %2d  max diff %.2e"
              % (k, v["pairs"], v["max_abs_metric_difference"]))


def export_individual(t, c_in, out):
    folder = OUT / "individual"
    folder.mkdir(exist_ok=True)
    rec = {}

    def canvas(w, h, ml, mb):
        f = plt.figure(figsize=(ml + w + 0.08, mb + h + 0.06))
        W, H = f.get_size_inches()
        return f, f.add_axes([ml / W, mb / H, w / W, h / H])

    f, ax = canvas(IN_BOX[2], IN_BOX[3], 0.46, 0.30)
    plot_input(ax, t, c_in)
    f1b.save_asset(f, folder, "c1_input_C_in")
    rec["c1_input_C_in"] = {"axes_offset_in": [0.46, 0.30],
                            "axes_size_in": list(IN_BOX[2:])}

    idx = 2
    for row in ("A", "B"):
        p = PATHWAYS[row]
        for stage, tau in (("transport", p["tau_f"]), ("sensor", p["tau_s"])):
            f, ax = canvas(K_W, K_H, 0.06, 0.30)
            plot_kernel(ax, stage, tau, bottom_row=(row == "B"))
            name = "c%d_pathway%s_%s" % (idx, row, stage)
            f1b.save_asset(f, folder, name)
            rec[name] = {"axes_offset_in": [0.06, 0.30],
                         "axes_size_in": [K_W, K_H]}
            idx += 1

    f, ax = canvas(OUT_BOX[2], OUT_BOX[3], 0.30, 0.30)
    plot_output(ax, t, c_in, out)
    f1b.save_asset(f, folder, "c6_output_S")
    rec["c6_output_S"] = {"axes_offset_in": [0.30, 0.30],
                          "axes_size_in": list(OUT_BOX[2:])}
    return rec


if __name__ == "__main__":
    main()
