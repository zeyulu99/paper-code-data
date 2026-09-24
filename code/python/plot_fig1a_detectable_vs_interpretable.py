"""Figure 1a - detectability and temporal fidelity are separate requirements.

Panel a is not a separate illustration: both devices are rows of the released
1,620-point canonical grid that produces the Figure 2 operating envelopes. The
waveform comes from run_evidence_closure_analysis.canonical_waveform and the
four metrics from canonical_metrics, so the numbers quoted in the caption are
the same numbers tabulated in the Supporting Information. main() asserts that
agreement against 02_canonical_extended_grid_1620.csv before drawing.

The two devices share the sensor time constant and differ only in fluidic
design, which is what makes the panel carry the Perspective's thesis: the
binding constraint is transport, not detection and not sensor kinetics.

Layout  : three plots only. The verdict is one marker line under each device
          title; the requirement lists that have no visual counterpart
          (selectivity, calibration stability, sample age) belong to the
          caption, not to the artwork.
Format  : 7.00 x 2.00 in, all lettering >= 5.8 pt, all strokes >= 0.6 pt
          (ACS Sensors floors are 4.5 pt and 0.5 pt).
Backend : Python / matplotlib.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "python"))
from run_evidence_closure_analysis import (  # noqa: E402
    canonical_metrics,
    canonical_waveform,
)
from transport_fidelity_models import transport_sensor_response  # noqa: E402

OUT = ROOT / "outputs" / "revision_2026-09-16" / "figure1a"
OUT.mkdir(parents=True, exist_ok=True)
GRID = (ROOT / "outputs" / "revision_2026-08-31_figure2_data_package"
        / "02_canonical_extended_grid_1620.csv")
STEM = "fig1a_detectable_vs_interpretable"

# ----------------------------------------------------------------- parameters
WAVEFORM = "pulse"
T_MIN = 30.0         # full width at half maximum of the canonical pulse
BASELINE = 0.30      # display baseline; a constant offset leaves all four
PEAK = 1.00          # metrics unchanged, so the grid rows still apply
LOD = 0.10           # a.u., below the baseline: detection is not the constraint

# Both devices are rows of the released grid. They share tau_s, so the only
# difference between them is the fluidic design.
TAU_SENSOR = 1.0
DEVICE_A = dict(label="Device A", topology="distributed_RTD", tau_f=2.0,
                tau_s=TAU_SENSOR, shape=3.0)
DEVICE_B = dict(label="Device B", topology="well_mixed", tau_f=15.0,
                tau_s=TAU_SENSOR, shape=None)

TOPOLOGY_ARG = {"plug_flow": "plug", "well_mixed": "cstr",
                "distributed_RTD": "gamma_rtd"}

# strict criteria, verbatim from the grid's strict_criterion field
CRIT = dict(amp=0.90, nrmse=0.10, r=0.95, delay=max(1.0, T_MIN / 4.0))

VIEW_MIN, VIEW_MAX = 40.0, 190.0
X_TICKS = [50, 100, 150]

BLACK = "#202020"
GRAY = "#8A8A8A"
RULE = "#4A4A4A"
BLUE = "#1657A6"      # Device A
ORANGE = "#C1440E"    # Device B (deliberately not the red used for pathway B in 1c)
LOD_FILL = "#ECECEC"

FIG_W, FIG_H = 7.00, 2.00
PW, PH, PB = 1.76, 0.92, 0.50
X_F, X_A, X_B = 0.46, 2.88, 4.86

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 6.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.transparent": False,
})

FS_AXIS, FS_TITLE, FS_NOTE, FS_TAG = 6.5, 7.0, 6.0, 5.8
LW_CURVE, LW_TARGET, LW_THIN = 1.5, 0.9, 0.7

DELTA = "Δ"


# ------------------------------------------------------------------- helpers
def fx(x_in: float) -> float:
    return x_in / FIG_W


def fy(y_in: float) -> float:
    return y_in / FIG_H


def axes_in(fig, x, y, w, h):
    return fig.add_axes([fx(x), fy(y), fx(w), fy(h)])


def to_display(y_unit: np.ndarray) -> np.ndarray:
    """Map the canonical 0-to-1 waveform onto the displayed concentration."""
    return BASELINE + (PEAK - BASELINE) * np.asarray(y_unit, dtype=float)


def device_output(time_s: np.ndarray, inlet: np.ndarray, device: dict) -> np.ndarray:
    kwargs = {}
    if device["shape"] is not None:
        kwargs["rtd_shape"] = device["shape"]
    return transport_sensor_response(
        time_s, inlet, topology=TOPOLOGY_ARG[device["topology"]],
        transport_time_s=device["tau_f"] * 60.0,
        sensor_tau_s=device["tau_s"] * 60.0, **kwargs)


def fwhm(t: np.ndarray, y: np.ndarray) -> float:
    half = y.min() + 0.5 * (y.max() - y.min())
    idx = np.where(y >= half)[0]
    return float(t[idx[-1]] - t[idx[0]])


def evaluate(time_s, inlet, measured, device) -> dict:
    m = canonical_metrics(WAVEFORM, T_MIN, time_s, inlet, measured)
    return {
        "amplitude_retention": float(m["amplitude_retention"]),
        "feature_delay_min": float(m["feature_delay_min"]),
        "nrmse": float(m["nrmse"]),
        "pearson_r": float(m["pearson_r"]),
        "display_peak": float(to_display(measured).max()),
        "peak_over_lod": float(to_display(measured).max() / LOD),
        "fwhm_min": fwhm(time_s / 60.0, measured),
        "pass_strict": bool(
            m["amplitude_retention"] >= CRIT["amp"]
            and m["nrmse"] <= CRIT["nrmse"]
            and m["pearson_r"] >= CRIT["r"]
            and abs(m["feature_delay_min"]) <= CRIT["delay"]),
    }


def assert_matches_grid(stats: dict) -> list[dict]:
    """Fail loudly if panel a has drifted away from the released Figure 2 grid."""
    import csv

    with GRID.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    checked = []
    for device in (DEVICE_A, DEVICE_B):
        match = [r for r in rows
                 if r["waveform_type"] == WAVEFORM
                 and float(r["T_min"]) == T_MIN
                 and r["topology"] == device["topology"]
                 and float(r["tau_transport_min"]) == device["tau_f"]
                 and float(r["tau_sensor_min"]) == device["tau_s"]]
        if len(match) != 1:
            raise SystemExit("no unique grid row for %s" % device["label"])
        row = match[0]
        s = stats[device["label"]]
        for key, column, tol in (("amplitude_retention", "amplitude_retention", 5e-3),
                                 ("nrmse", "NRMSE", 5e-3),
                                 ("pearson_r", "pearson_r", 5e-3),
                                 ("feature_delay_min", "feature_delay_min", 0.06)):
            released = float(row[column])
            if abs(s[key] - released) > tol:
                raise SystemExit(
                    "%s %s drifted: panel a %.4f vs released grid %.4f"
                    % (device["label"], key, s[key], released))
        released_pass = row["pass_strict"].strip().lower() == "true"
        if released_pass != s["pass_strict"]:
            raise SystemExit("%s verdict disagrees with the released grid"
                             % device["label"])
        checked.append({"device": device["label"], "grid_row": row["grid"],
                        "released_pass_strict": released_pass})
    return checked


def mark_at(fig, x_in: float, y_in: float, ok: bool, colour: str,
            side_in: float = 0.050, lw: float = 1.0) -> None:
    """Draw a check or cross of a given physical size, centred on (x_in, y_in).

    Arial has no U+2713/U+2717, so both glyphs are drawn as explicit segments.
    """
    dx, dy = fx(side_in), fy(side_in)
    x, y = fx(x_in), fy(y_in)
    if ok:
        xs = [x - dx * 0.5, x - dx * 0.12, x + dx * 0.5]
        ys = [y + dy * 0.05, y - dy * 0.40, y + dy * 0.55]
        fig.add_artist(Line2D(xs, ys, color=colour, lw=lw,
                              solid_capstyle="round", solid_joinstyle="round",
                              transform=fig.transFigure, zorder=6))
    else:
        for xs, ys in (([x - dx * 0.45, x + dx * 0.45],
                        [y - dy * 0.45, y + dy * 0.45]),
                       ([x - dx * 0.45, x + dx * 0.45],
                        [y + dy * 0.45, y - dy * 0.45])):
            fig.add_artist(Line2D(xs, ys, color=colour, lw=lw,
                                  solid_capstyle="round",
                                  transform=fig.transFigure, zorder=6))


def text_width_in(fig, label: str, fontsize: float) -> float:
    """Rendered width of a string, in inches."""
    probe = fig.text(0.0, 0.0, label, fontsize=fontsize)
    fig.canvas.draw()
    width = probe.get_window_extent(
        renderer=fig.canvas.get_renderer()).width / fig.dpi
    probe.remove()
    return float(width)


def verdict_subtitle(fig, x_centre: float, y_in: float, colour: str,
                     interpretable: bool) -> None:
    """One-line 'detectable / interpretable' verdict under a device title."""
    items = [("detectable", True), ("interpretable", interpretable)]
    mark_w, text_gap, item_gap = 0.055, 0.050, 0.135
    widths = [text_width_in(fig, label, FS_TAG) for label, _ in items]
    total = sum(widths) + len(items) * (text_gap + mark_w) + item_gap
    cursor = x_centre - 0.5 * total
    for (label, ok), w in zip(items, widths):
        fig.text(fx(cursor), fy(y_in), label, ha="left", va="center",
                 fontsize=FS_TAG, color=colour)
        mark_at(fig, cursor + w + text_gap + 0.5 * mark_w, y_in, ok, colour)
        cursor += w + text_gap + mark_w + item_gap


def style_plot(ax, show_ylabel: bool, show_yticklabels: bool):
    ax.set_xlim(VIEW_MIN, VIEW_MAX)
    ax.set_ylim(0, 1.30)
    ax.set_xticks(X_TICKS)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("Time (min)", fontsize=FS_AXIS, labelpad=1.8)
    if show_ylabel:
        ax.set_ylabel("Concentration (a.u.)", fontsize=FS_AXIS, labelpad=2.0)
    if not show_yticklabels:
        ax.set_yticklabels([])
    ax.tick_params(pad=1.5)
    ax.axhspan(0, LOD, color=LOD_FILL, lw=0, zorder=0)
    ax.axhline(LOD, color=GRAY, lw=LW_THIN, ls=(0, (3.2, 2.2)), zorder=1)
    # the label fits in the gap between the LOD line and the resting baseline
    ax.text(VIEW_MAX - 0.02 * (VIEW_MAX - VIEW_MIN), LOD + 0.035, "LOD",
            ha="right", va="bottom", fontsize=FS_NOTE, color=GRAY)


# ---------------------------------------------------------------------- build
def main() -> None:
    time_s, unit_inlet = canonical_waveform(WAVEFORM, T_MIN)
    t = time_s / 60.0

    out_a = device_output(time_s, unit_inlet, DEVICE_A)
    out_b = device_output(time_s, unit_inlet, DEVICE_B)

    stats = {"Device A": evaluate(time_s, unit_inlet, out_a, DEVICE_A),
             "Device B": evaluate(time_s, unit_inlet, out_b, DEVICE_B)}
    checked = assert_matches_grid(stats)

    c_in, y_a, y_b = (to_display(unit_inlet), to_display(out_a), to_display(out_b))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")

    ax_f = axes_in(fig, X_F, PB, PW, PH)
    ax_a = axes_in(fig, X_A, PB, PW, PH)
    ax_b = axes_in(fig, X_B, PB, PW, PH)

    # -- underlying feature ------------------------------------------------
    style_plot(ax_f, True, True)
    ax_f.plot(t, c_in, color=BLACK, lw=LW_CURVE, zorder=3)

    # -- device A ----------------------------------------------------------
    style_plot(ax_a, False, True)
    ax_a.plot(t, c_in, color=GRAY, lw=LW_TARGET, ls=(0, (3.2, 2.2)), zorder=2)
    ax_a.plot(t, y_a, color=BLUE, lw=LW_CURVE, zorder=3)
    ax_a.text(VIEW_MAX - 0.03 * (VIEW_MAX - VIEW_MIN), 1.27, "Measured",
              ha="right", va="top", fontsize=FS_NOTE, color=BLUE)
    ax_a.text(VIEW_MAX - 0.03 * (VIEW_MAX - VIEW_MIN), 1.13, "Target",
              ha="right", va="top", fontsize=FS_NOTE, color=GRAY)
    verdict_subtitle(fig, X_A + 0.5 * PW, 1.475, BLUE, True)

    # -- device B ----------------------------------------------------------
    style_plot(ax_b, False, False)
    ax_b.plot(t, c_in, color=GRAY, lw=LW_TARGET, ls=(0, (3.2, 2.2)), zorder=2)
    ax_b.plot(t, y_b, color=ORANGE, lw=LW_CURVE, zorder=3)
    verdict_subtitle(fig, X_B + 0.5 * PW, 1.475, ORANGE, False)

    pk_ref = float(t[np.argmax(c_in)])
    pk_obs = float(t[np.argmax(y_b)])
    ref_peak = float(c_in.max())
    amp_obs = float(y_b.max())

    # peak delay, measured above both peaks
    y_dt = 1.11
    for xv, y_from, cv in ((pk_ref, ref_peak, GRAY), (pk_obs, amp_obs, ORANGE)):
        ax_b.plot([xv, xv], [y_from, y_dt], color=cv, lw=0.6,
                  ls=(0, (1.6, 1.6)), zorder=1)
    ax_b.annotate("", xy=(pk_obs, y_dt), xytext=(pk_ref, y_dt),
                  arrowprops=dict(arrowstyle="<->", lw=LW_THIN, color=BLACK,
                                  shrinkA=0, shrinkB=0, mutation_scale=5))
    ax_b.text(0.5 * (pk_ref + pk_obs), y_dt + 0.035, DELTA + "t", ha="center",
              va="bottom", fontsize=FS_NOTE, color=BLACK)

    # amplitude loss, measured between the two peak levels
    x_da = pk_obs + 0.16 * (VIEW_MAX - VIEW_MIN)
    ax_b.plot([pk_ref, x_da], [ref_peak, ref_peak], color=GRAY, lw=0.6,
              ls=(0, (1.6, 1.6)), zorder=1)
    ax_b.plot([pk_obs, x_da], [amp_obs, amp_obs], color=ORANGE, lw=0.6,
              ls=(0, (1.6, 1.6)), zorder=1)
    ax_b.annotate("", xy=(x_da, ref_peak), xytext=(x_da, amp_obs),
                  arrowprops=dict(arrowstyle="<->", lw=LW_THIN, color=BLACK,
                                  shrinkA=0, shrinkB=0, mutation_scale=5))
    ax_b.text(x_da + 0.02 * (VIEW_MAX - VIEW_MIN), 0.5 * (ref_peak + amp_obs),
              DELTA + "A", ha="left", va="center", fontsize=FS_NOTE, color=BLACK)

    # waveform distortion, anchored on the trailing shoulder of the output
    level = BASELINE + 0.42 * (amp_obs - BASELINE)
    tail = t > pk_obs
    j = int(np.argmin(np.abs(y_b[tail] - level)))
    x_tail = float(t[tail][j])
    ax_b.annotate("broadened", xy=(x_tail, level),
                  xytext=(x_tail + 0.035 * (VIEW_MAX - VIEW_MIN), level + 0.17),
                  ha="left", va="center", fontsize=FS_NOTE, color=ORANGE,
                  arrowprops=dict(arrowstyle="-", lw=0.6, color=ORANGE,
                                  shrinkA=1.5, shrinkB=1.0))

    # -- flow from the feature to the two devices --------------------------
    fig.add_artist(FancyArrowPatch(
        (fx(X_F + PW + 0.10), fy(PB + 0.5 * PH)),
        (fx(X_F + PW + 0.40), fy(PB + 0.5 * PH)),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7,
        lw=0.9, color=RULE, shrinkA=0, shrinkB=0))

    bx0, bx1, by = X_A, X_B + PW, 1.735
    fig.add_artist(Line2D([fx(bx0), fx(bx1)], [fy(by), fy(by)], color=RULE,
                          lw=LW_THIN, transform=fig.transFigure))
    for xv in (bx0, bx1):
        fig.add_artist(Line2D([fx(xv), fx(xv)], [fy(by), fy(by - 0.06)],
                              color=RULE, lw=LW_THIN, transform=fig.transFigure))
    fig.text(fx(0.5 * (bx0 + bx1)), fy(by + 0.05),
             "Same feature, same sensor kinetics, different fluidic design",
             ha="center", va="bottom", fontsize=FS_AXIS, color=BLACK)

    for xc, label, colour in ((X_F + 0.5 * PW, "Underlying feature", BLACK),
                              (X_A + 0.5 * PW, "Device A", BLUE),
                              (X_B + 0.5 * PW, "Device B", ORANGE)):
        fig.text(fx(xc), fy(1.575), label, ha="center", va="bottom",
                 fontsize=FS_TITLE, color=colour, fontweight="bold")

    fig.text(fx(0.045), fy(1.925), "a", ha="left", va="top",
             fontsize=8.0, fontweight="bold", color=BLACK)

    for ext in ("pdf", "svg"):
        fig.savefig(OUT / (STEM + "." + ext), facecolor="white")
    fig.savefig(OUT / (STEM + ".png"), dpi=600, facecolor="white")
    plt.close(fig)

    # -- machine-readable record ------------------------------------------
    step = max(1, int(round(0.25 / (t[1] - t[0]))))
    lines = ["time_min,c_true,c_measured_device_A,c_measured_device_B,lod_au\n"]
    for i in range(0, len(t), step):
        lines.append("%.2f,%.6f,%.6f,%.6f,%.2f\n"
                     % (t[i], c_in[i], y_a[i], y_b[i], LOD))
    (OUT / "fig1a_plotted_values.csv").write_text("".join(lines), encoding="utf-8")

    record = {
        "panel": "Figure 1a",
        "provenance": "rows of outputs/revision_2026-08-31_figure2_data_package/"
                      "02_canonical_extended_grid_1620.csv",
        "grid_rows_checked": checked,
        "waveform": {
            "kind": WAVEFORM, "fwhm_T_min": T_MIN,
            "definition": "run_evidence_closure_analysis.canonical_waveform",
            "display_baseline_au": BASELINE, "display_peak_au": PEAK,
            "note": "a constant display offset leaves all four metrics unchanged",
        },
        "limit_of_detection_au": LOD,
        "plot_window_min": [VIEW_MIN, VIEW_MAX],
        "device_A": {**DEVICE_A, **stats["Device A"]},
        "device_B": {**DEVICE_B, **stats["Device B"]},
        "strict_criteria": CRIT,
        "reference_fwhm_min": fwhm(t, unit_inlet),
    }
    (OUT / "fig1a_parameters_and_metrics.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")

    print("grid agreement verified for rows: %s"
          % ", ".join(c["grid_row"] for c in checked))
    for name in ("Device A", "Device B"):
        s = stats[name]
        print("%s: amp=%.3f dt=%+.2f nrmse=%.3f r=%.3f | peak=%.3f (%.1f x LOD) "
              "fwhm=%.1f | strict=%s"
              % (name, s["amplitude_retention"], s["feature_delay_min"],
                 s["nrmse"], s["pearson_r"], s["display_peak"],
                 s["peak_over_lod"], s["fwhm_min"],
                 "PASS" if s["pass_strict"] else "FAIL"))
    print("reference fwhm=%.1f min; wrote %s" % (fwhm(t, unit_inlet), OUT))


if __name__ == "__main__":
    main()
