"""Build and compare honest visualization strategies for revised Figure 2.

This is a candidate-generation script. It does not overwrite manuscript art.
It compares representative conditions and plotting layouts before a final
placement-ready export is selected.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

from run_evidence_closure_analysis import canonical_metrics, canonical_waveform
from transport_fidelity_models import gamma_rtd_kernel, transport_sensor_response


PROJECT = Path(r"E:\Projects_im\6.sweat_sensor_perspective")
PACKAGE = PROJECT / "outputs" / "revision_2026-08-31_figure2_data_package"
OUTPUT = PROJECT / "outputs" / "figure_working" / "figure2_strategy_comparison_2026-09-09"
PNG = OUTPUT / "png"
DATA = OUTPUT / "data"

CANONICAL_EXTENDED = PACKAGE / "02_canonical_extended_grid_1620.csv"
EMPIRICAL_INPUTS = PACKAGE / "05_empirical_waveforms_model_inputs.csv"
EMPIRICAL_RUNS = PACKAGE / "08_empirical_continuous_only_1155.csv"

BLACK = "#151515"
GRAY = "#8C8C8C"
BLUE = "#245DB8"
PURPLE = "#6B2AA6"
ORANGE = "#E87522"
TEAL = "#138A8A"
RED = "#C73A3A"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 1.1,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No data for {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_candidate(fig: plt.Figure, stem: str) -> Path:
    PNG.mkdir(parents=True, exist_ok=True)
    path = PNG / f"{stem}.png"
    fig.savefig(path, dpi=350, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    return path


def fft_convolve_prefix(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    full_length = len(signal) + len(kernel) - 1
    nfft = 1 << (full_length - 1).bit_length()
    result = np.fft.irfft(np.fft.rfft(signal, nfft) * np.fft.rfft(kernel, nfft), nfft)
    return result[: len(signal)]


def first_order_fast(time_s: np.ndarray, values: np.ndarray, tau_s: float) -> np.ndarray:
    dt = float(time_s[1] - time_s[0])
    decay = float(np.exp(-dt / tau_s))
    centered = np.asarray(values, dtype=float) - float(values[0])
    shifted = np.empty_like(centered)
    shifted[0] = 0.0
    shifted[1:] = centered[:-1]
    impulse = (1.0 - decay) * decay ** np.arange(len(centered), dtype=float)
    return fft_convolve_prefix(shifted, impulse) + float(values[0])


def transport_fast(
    time_s: np.ndarray,
    inlet: np.ndarray,
    topology: str,
    transport_min: float,
    sensor_min: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    transport_s = transport_min * 60.0
    if topology == "plug":
        fluid = np.interp(time_s - transport_s, time_s, inlet, left=inlet[0], right=inlet[-1])
    elif topology == "cstr":
        fluid = first_order_fast(time_s, inlet, transport_s)
    elif topology == "gamma_rtd":
        dt = float(time_s[1] - time_s[0])
        kernel = gamma_rtd_kernel(dt, transport_s, shape=3.0)
        baseline = float(inlet[0])
        fluid = fft_convolve_prefix(inlet - baseline, kernel * dt) + baseline
    else:
        raise ValueError(topology)
    sensor = first_order_fast(time_s, fluid, sensor_min * 60.0) if sensor_min > 0 else fluid.copy()
    return fluid, sensor


def strict_pass(kind: str, timescale: float, metrics: dict) -> bool:
    return bool(
        metrics["amplitude_retention"] >= 0.90
        and metrics["nrmse"] <= 0.10
        and metrics["pearson_r"] >= 0.95
        and abs(metrics["feature_delay_min"]) <= max(1.0, timescale / 4.0)
    )


def clean_axes(ax: plt.Axes, show_ticks: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.grid(False)


def representative_trace(kind: str, timescale: float, topology: str, transport_min: float = 5.0):
    time_s, inlet = canonical_waveform(kind, timescale)
    fluid, sensor = transport_fast(time_s, inlet, topology, transport_min, sensor_min=1.0)
    active = np.flatnonzero(np.maximum.reduce([inlet, fluid, sensor]) > 0.002)
    margin_before = int(8.0 / 0.05)
    margin_after = int(12.0 / 0.05)
    lo = max(0, int(active[0]) - margin_before)
    hi = min(len(time_s), int(active[-1]) + margin_after)
    time_min = time_s[lo:hi] / 60.0
    return time_min - time_min[0], inlet[lo:hi], fluid[lo:hi], sensor[lo:hi]


def plot_b_candidate(timescale: float, topology: str) -> tuple[Path, dict]:
    time, inlet, fluid, sensor = representative_trace("pulse", timescale, topology)
    fig, ax = plt.subplots(figsize=(2.55, 2.75))
    ax.plot(time, inlet + 2.4, color=BLACK, linewidth=2.2)
    ax.plot(time, fluid + 1.2, color=BLUE, linewidth=2.2)
    ax.plot(time, sensor, color=PURPLE, linewidth=2.2)
    ax.set_ylim(-0.08, 3.5)
    ax.set_xlim(float(time[0]), float(time[-1]))
    ax.axis("off")
    stem = f"candidate_b_T{timescale:g}_{topology}"
    path = save_candidate(fig, stem)
    rows = [
        {
            "time_min": f"{x:.6f}",
            "C_in": f"{a:.10f}",
            "C_f": f"{b:.10f}",
            "S": f"{c:.10f}",
            "T_min": f"{timescale:g}",
            "tau_transport_min": "5",
            "tau_sensor_min": "1",
            "topology": topology,
        }
        for x, a, b, c in zip(time, inlet, fluid, sensor)
    ]
    write_csv(DATA / f"{stem}.csv", rows)
    return path, {
        "candidate": stem,
        "T_min": timescale,
        "topology": topology,
        "fluid_max_abs_change": float(np.max(np.abs(fluid - inlet))),
        "sensor_max_abs_change": float(np.max(np.abs(sensor - inlet))),
    }


def dense_boundary_grid() -> list[dict]:
    anchors = np.asarray([0.05, 0.10, 0.25, 0.50, 0.75, 1.0, 2.0, 3.0, 3.7, 5.0, 10.0, 15.0])
    dense = np.geomspace(0.05, 15.0, 61)
    transport_values = np.unique(np.round(np.concatenate([dense, anchors]), 8))
    rows = []
    for kind in ["pulse", "ramp", "sinusoid"]:
        for timescale in [2.0, 5.0, 10.0, 30.0, 60.0]:
            time_s, inlet = canonical_waveform(kind, timescale)
            for topology in ["plug", "cstr", "gamma_rtd"]:
                for transport_min in transport_values:
                    _, sensor = transport_fast(time_s, inlet, topology, float(transport_min), sensor_min=1.0)
                    metrics = canonical_metrics(kind, timescale, time_s, inlet, sensor)
                    passed = strict_pass(kind, timescale, metrics)
                    rows.append(
                        {
                            "waveform_type": kind,
                            "T_min": f"{timescale:g}",
                            "topology": topology,
                            "tau_transport_min": f"{transport_min:.8g}",
                            "tau_sensor_min": "1",
                            "amplitude_retention": f"{metrics['amplitude_retention']:.10g}",
                            "NRMSE": f"{metrics['nrmse']:.10g}",
                            "pearson_r": f"{metrics['pearson_r']:.10g}",
                            "feature_delay_min": f"{metrics['feature_delay_min']:.10g}",
                            "pass_strict": "PASS" if passed else "FAIL",
                        }
                    )
    return rows


def envelope(rows: list[dict], topology: str, waveform: str) -> list[float | None]:
    values = []
    for timescale in [2.0, 5.0, 10.0, 30.0, 60.0]:
        passing = [
            float(row["tau_transport_min"])
            for row in rows
            if row["topology"] == topology
            and row["waveform_type"] == waveform
            and float(row["T_min"]) == timescale
            and row["pass_strict"].upper() == "PASS"
        ]
        values.append(max(passing) if passing else None)
    return values


def log_tick(value: float, _position: int) -> str:
    return f"{value:g}" if value >= 1 else f"{value:.2g}"


def format_envelope_axes(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.set_xlim(-0.45, 4.45)
    ax.set_ylim(0.025, 18.0)
    ax.set_xticks(np.arange(5), ["2", "5", "10", "30", "60"])
    ticks = [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(log_tick))
    ax.tick_params(which="minor", left=False, bottom=False)
    clean_axes(ax, show_ticks=True)


def plot_c_topology(rows: list[dict], topology: str, prefix: str) -> Path:
    styles = {"pulse": (BLACK, "o"), "ramp": (BLUE, "s"), "sinusoid": (ORANGE, "^")}
    offsets = {"pulse": -0.075, "ramp": 0.0, "sinusoid": 0.075}
    x = np.arange(5, dtype=float)
    fig, ax = plt.subplots(figsize=(3.0, 2.55))
    for waveform in ["pulse", "ramp", "sinusoid"]:
        color, marker = styles[waveform]
        values = envelope(rows, topology, waveform)
        y = np.asarray([np.nan if value is None else value for value in values])
        ax.plot(x, y, color=color, marker=marker, linewidth=1.45, markersize=5.6)
        missing = np.asarray([value is None for value in values])
        if np.any(missing):
            ax.scatter(x[missing] + offsets[waveform], np.full(np.sum(missing), 0.034), marker="x", color=color, s=40, linewidths=1.5, clip_on=False)
    format_envelope_axes(ax)
    ax.set_xlabel(r"Feature timescale, $T$ (min)")
    ax.set_ylabel(r"Largest passing $\tau_{\mathrm{transport}}$ (min)")
    return save_candidate(fig, f"{prefix}_{topology}")


def plot_c_waveform(rows: list[dict], waveform: str) -> Path:
    styles = {"plug": (BLACK, "o"), "cstr": (BLUE, "s"), "gamma_rtd": (ORANGE, "^")}
    offsets = {"plug": -0.08, "cstr": 0.0, "gamma_rtd": 0.08}
    x = np.arange(5, dtype=float)
    fig, ax = plt.subplots(figsize=(3.0, 2.55))
    for topology in ["plug", "cstr", "gamma_rtd"]:
        color, marker = styles[topology]
        values = envelope(rows, topology, waveform)
        y = np.asarray([np.nan if value is None else value for value in values])
        ax.plot(x + offsets[topology], y, color=color, marker=marker, linewidth=1.4, markersize=5.4)
        missing = np.asarray([value is None for value in values])
        if np.any(missing):
            ax.scatter(x[missing] + offsets[topology], np.full(np.sum(missing), 0.034), marker="x", color=color, s=38, linewidths=1.4, clip_on=False)
    format_envelope_axes(ax)
    ax.set_xlabel(r"Feature timescale, $T$ (min)")
    ax.set_ylabel(r"Largest passing $\tau_{\mathrm{transport}}$ (min)")
    return save_candidate(fig, f"candidate_c_dense_by_waveform_{waveform}")


def group_inputs(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        result[row["trace_id"]].append(row)
    for trace in result.values():
        trace.sort(key=lambda row: float(row["time_min"]))
    return result


def normalize_to_source(source: np.ndarray, values: np.ndarray) -> np.ndarray:
    return (values - float(np.min(source))) / float(np.ptp(source))


def plot_d_reference(trace_id: str, trace: list[dict[str, str]], source_name: str) -> Path:
    time = np.asarray([float(row["time_min"]) for row in trace])
    values = np.asarray([float(row["signal"]) for row in trace])
    normalized = normalize_to_source(values, values)
    fig, ax = plt.subplots(figsize=(3.1, 2.0))
    ax.plot(time, normalized, color=BLACK, linewidth=2.1)
    clean_axes(ax)
    ax.set_xlim(float(time[0]), float(time[-1]))
    ax.set_ylim(-0.08, 1.08)
    stem = f"candidate_d_{source_name}_representative_original"
    path = save_candidate(fig, stem)
    write_csv(
        DATA / f"{stem}.csv",
        [
            {"trace_id": trace_id, "time_min": f"{x:.6f}", "signal": f"{raw:.10g}", "signal_normalized": f"{norm:.10g}"}
            for x, raw, norm in zip(time, values, normalized)
        ],
    )
    return path


def plot_d_transport_series(
    trace_id: str,
    trace: list[dict[str, str]],
    source_name: str,
    transport_values: list[float],
    run_rows: list[dict[str, str]],
) -> tuple[Path, list[dict]]:
    time = np.asarray([float(row["time_min"]) for row in trace])
    source = np.asarray([float(row["signal"]) for row in trace])
    colors = [TEAL, BLUE, ORANGE, RED]
    fig, ax = plt.subplots(figsize=(3.25, 2.0))
    ax.plot(time, normalize_to_source(source, source), color=GRAY, linewidth=1.6, linestyle=(0, (4, 3)), zorder=1)
    data_rows = []
    metrics_summary = []
    for color, transport_min in zip(colors, transport_values):
        modeled = transport_sensor_response(
            time * 60.0,
            source,
            topology="cstr",
            transport_time_s=transport_min * 60.0,
            sensor_tau_s=None,
        )
        normalized = normalize_to_source(source, modeled)
        ax.plot(time, normalized, color=color, linewidth=1.8, zorder=2)
        matching = [
            row
            for row in run_rows
            if row["waveform_id"] == trace_id
            and row["topology"] == "well_mixed"
            and float(row["sensor_tau_min"]) == 0.0
            and abs(float(row["tau_transport_or_window_min"]) - transport_min) < 1e-9
        ]
        if len(matching) != 1:
            raise ValueError(f"Expected one empirical run for {trace_id}, tau={transport_min}")
        run = matching[0]
        metrics_summary.append(
            {
                "trace_id": trace_id,
                "tau_transport_min": transport_min,
                "pass_strict": run["pass_strict"],
                "NRMSE": float(run["nrmse"]),
                "pearson_r": float(run["pearson_r"]),
                "feature_delay_min": float(run["feature_delay_min"]),
            }
        )
        for x, raw, norm in zip(time, modeled, normalized):
            data_rows.append(
                {
                    "trace_id": trace_id,
                    "time_min": f"{x:.6f}",
                    "tau_transport_min": f"{transport_min:g}",
                    "modeled_signal": f"{raw:.10g}",
                    "modeled_normalized_to_source": f"{norm:.10g}",
                    "pass_strict": run["pass_strict"],
                }
            )
    clean_axes(ax)
    ax.set_xlim(float(time[0]), float(time[-1]))
    ax.set_ylim(-0.08, 1.08)
    stem = f"candidate_d_{source_name}_same_trace_transport_series"
    path = save_candidate(fig, stem)
    write_csv(DATA / f"{stem}.csv", data_rows)
    return path, metrics_summary


def main() -> None:
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    inventory = []
    b_metrics = []
    for timescale in [30.0, 10.0]:
        for topology in ["plug", "cstr", "gamma_rtd"]:
            path, metrics = plot_b_candidate(timescale, topology)
            inventory.append(str(path))
            b_metrics.append(metrics)

    dense_rows = dense_boundary_grid()
    write_csv(DATA / "canonical_dense_boundary_grid_tau_sensor_1min.csv", dense_rows)
    original_rows = read_csv(CANONICAL_EXTENDED)
    original_standardized = [
        {
            "waveform_type": row["waveform_type"],
            "T_min": row["T_min"],
            "topology": {"plug_flow": "plug", "well_mixed": "cstr", "distributed_RTD": "gamma_rtd"}[row["topology"]],
            "tau_transport_min": row["tau_transport_min"],
            "pass_strict": "PASS" if row["pass_strict"].strip().upper() in {"TRUE", "PASS", "1"} else "FAIL",
        }
        for row in original_rows
        if float(row["tau_sensor_min"]) == 1.0
    ]
    for topology in ["plug", "cstr", "gamma_rtd"]:
        inventory.append(str(plot_c_topology(original_standardized, topology, "candidate_c_discrete")))
        inventory.append(str(plot_c_topology(dense_rows, topology, "candidate_c_dense")))
    for waveform in ["pulse", "ramp", "sinusoid"]:
        inventory.append(str(plot_c_waveform(dense_rows, waveform)))

    input_rows = read_csv(EMPIRICAL_INPUTS)
    run_rows = read_csv(EMPIRICAL_RUNS)
    traces = group_inputs(input_rows)
    representatives = {
        "saha": ("Saha_2024_Fig3b_Subject1", [0.5, 3.7, 5.0, 10.0]),
        "choi": ("Choi_2020_C", [0.05, 1.0, 2.0, 5.0]),
    }
    d_metrics = []
    for source_name, (trace_id, transport_values) in representatives.items():
        inventory.append(str(plot_d_reference(trace_id, traces[trace_id], source_name)))
        path, summary = plot_d_transport_series(trace_id, traces[trace_id], source_name, transport_values, run_rows)
        inventory.append(str(path))
        d_metrics.extend(summary)

    discrete_vs_dense = []
    for topology in ["plug", "cstr", "gamma_rtd"]:
        for waveform in ["pulse", "ramp", "sinusoid"]:
            discrete = envelope(original_standardized, topology, waveform)
            dense = envelope(dense_rows, topology, waveform)
            for timescale, old, new in zip([2, 5, 10, 30, 60], discrete, dense):
                discrete_vs_dense.append(
                    {
                        "topology": topology,
                        "waveform": waveform,
                        "T_min": timescale,
                        "discrete_boundary_min": old,
                        "dense_boundary_min": new,
                    }
                )

    report = {
        "status": "candidate_comparison_complete",
        "candidate_images": inventory,
        "b_visibility_metrics": b_metrics,
        "c_discrete_vs_dense": discrete_vs_dense,
        "d_representative_metrics": d_metrics,
        "scientific_constraints": [
            "No independent axis rescaling between topology candidates.",
            "No artificial curve offsets except the declared vertical staging of C_in, C_f, and S in panel b.",
            "Panel c dense boundaries are recalculated model outputs, not interpolated artwork.",
            "Panel d compares the same source trace across transport conditions on one source-based normalization.",
        ],
    }
    (OUTPUT / "comparison_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(OUTPUT), "images": len(inventory), "dense_rows": len(dense_rows)}, indent=2))


if __name__ == "__main__":
    main()
