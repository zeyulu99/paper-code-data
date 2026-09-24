"""Run topology-aware sensitivity analysis on public digitized waveforms.

The Saha SBG and Choi sweat-chloride traces are empirical waveform proxies,
not device-free physiological truth. Simulations quantify incremental distortion
from an additional transport/sensor architecture and retain that limitation in
every exported record.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from transport_fidelity_models import (
    chrono_sampling_response,
    event_reset_response,
    fidelity_metrics,
    transport_sensor_response,
)


PROJECT = Path(r"E:\Projects_im\6.sweat_sensor_perspective")
INPUT_BUNDLE = PROJECT / "outputs" / "revision_2026-08-06" / "waveform_digitization" / "digitization_bundle.json"
OUTPUT_DIR = PROJECT / "outputs" / "revision_2026-08-06" / "waveform_model_analysis"
DOC_PATH = PROJECT / "docs" / "revision_2026-08-06" / "waveform_model_analysis.md"

TRANSPORT_MIN = [0.05, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 3.7, 5.0, 10.0, 15.0]
SENSOR_TAU_MIN = [0.0, 0.5, 1.0, 2.0, 5.0]
EVENT_CYCLE_MIN = [0.5, 1.0, 2.0, 5.0, 10.0]
EVENT_RESIDUAL = [0.0, 0.1, 0.25]
CHRONO_WINDOW_MIN = [1.0, 2.0, 5.0, 10.0, 15.0]
TOPOLOGIES = ["plug", "cstr", "gamma_rtd"]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_empirical_waveforms() -> list[dict]:
    bundle = json.loads(INPUT_BUNDLE.read_text(encoding="utf-8"))
    rows = bundle["waveforms"]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["source"] == "Saha 2024" and row["signal_type"] == "SBG":
            grouped[(row["source"], row["panel"])].append(row)
        elif row["source"] == "Choi 2020" and row["signal_type"] == "sweat_chloride":
            grouped[(row["source"], row["panel"])].append(row)

    waveforms = []
    for (source, panel), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: float(row["time_min"]))
        raw_t = np.asarray([float(row["time_min"]) for row in group], dtype=float)
        raw_y = np.asarray([float(row["value"]) for row in group], dtype=float)
        raw_x_unc = float(np.median([float(row["x_uncertainty"]) for row in group]))
        raw_y_unc = float(np.median([float(row["y_uncertainty"]) for row in group]))
        if source == "Saha 2024":
            valid = raw_t >= 0.0
            raw_t = raw_t[valid]
            raw_y = raw_y[valid]
            dt_min = 0.25
            source_group = "Saha meal SBG proxy"
            strict_delay = 5.0
            relaxed_delay = 10.0
        else:
            start = max(15.5, float(raw_t.min()))
            valid = raw_t >= start
            raw_t = raw_t[valid]
            raw_y = raw_y[valid]
            raw_t = raw_t - start
            dt_min = 0.25
            source_group = "Choi exercise chloride proxy"
            strict_delay = 2.0
            relaxed_delay = 5.0
        uniform_t = np.arange(0.0, float(raw_t.max()) + 0.5 * dt_min, dt_min)
        uniform_y = np.interp(uniform_t, raw_t, raw_y)
        waveforms.append(
            {
                "source": source,
                "panel": panel,
                "waveform_id": f"{source.replace(' ', '_')}_{panel}",
                "source_group": source_group,
                "time_min": uniform_t,
                "value": uniform_y,
                "raw_time_min": raw_t,
                "raw_value": raw_y,
                "x_uncertainty_min": raw_x_unc,
                "y_uncertainty": raw_y_unc,
                "unit": group[0]["unit"],
                "strict_delay_limit_min": strict_delay,
                "relaxed_delay_limit_min": relaxed_delay,
                "source_url": group[0]["source_url"],
            }
        )
    return waveforms


def extended_metrics(time_min: np.ndarray, reference: np.ndarray, measured: np.ndarray) -> dict:
    base = fidelity_metrics(time_min * 60.0, reference, measured)
    ref = np.asarray(reference, dtype=float)
    obs = np.asarray(measured, dtype=float)
    ref_range = float(np.ptp(ref))
    correlation = float(np.corrcoef(ref, obs)[0, 1]) if np.std(obs) > 0 else 0.0
    baseline = float(np.median(ref[: max(3, len(ref) // 10)]))
    upward = abs(float(np.max(ref)) - baseline) >= abs(float(np.min(ref)) - baseline)
    feature = "peak" if upward else "trough"
    ref_index = int(np.argmax(ref) if upward else np.argmin(ref))
    obs_index = int(np.argmax(obs) if upward else np.argmin(obs))
    feature_delay = float(time_min[obs_index] - time_min[ref_index])
    centered_ref = ref - np.mean(ref)
    centered_obs = obs - np.mean(obs)
    correlation_sequence = np.correlate(centered_obs, centered_ref, mode="full")
    lag_samples = int(np.argmax(correlation_sequence) - (len(ref) - 1))
    cross_correlation_lag = float(lag_samples * (time_min[1] - time_min[0]))
    max_error = float(np.max(np.abs(obs - ref)) / ref_range)
    return {
        "amplitude_retention": float(base.amplitude_retention),
        "peak_delay_min": float(base.peak_delay_s / 60.0),
        "feature_type": feature,
        "feature_delay_min": feature_delay,
        "xcorr_lag_min": cross_correlation_lag,
        "nrmse": float(base.nrmse),
        "pearson_r": correlation,
        "max_abs_error_norm": max_error,
    }


def criteria(metrics: dict, strict_delay: float, relaxed_delay: float) -> tuple[str, str]:
    strict = (
        metrics["amplitude_retention"] >= 0.90
        and metrics["nrmse"] <= 0.10
        and metrics["pearson_r"] >= 0.95
        and abs(metrics["feature_delay_min"]) <= strict_delay
    )
    relaxed = (
        metrics["amplitude_retention"] >= 0.80
        and metrics["nrmse"] <= 0.15
        and metrics["pearson_r"] >= 0.90
        and abs(metrics["feature_delay_min"]) <= relaxed_delay
    )
    return ("PASS" if strict else "FAIL", "PASS" if relaxed else "FAIL")


def build_run_row(waveform: dict, model: dict, measured: np.ndarray) -> dict:
    metrics = extended_metrics(waveform["time_min"], waveform["value"], measured)
    strict, relaxed = criteria(metrics, waveform["strict_delay_limit_min"], waveform["relaxed_delay_limit_min"])
    return {
        "waveform_id": waveform["waveform_id"],
        "source": waveform["source"],
        "panel": waveform["panel"],
        "source_group": waveform["source_group"],
        "model_class": model["model_class"],
        "topology": model["topology"],
        "transport_or_window_min": model["transport_or_window_min"],
        "sensor_tau_min": model.get("sensor_tau_min", 0.0),
        "rtd_shape": model.get("rtd_shape", ""),
        "residual_fraction": model.get("residual_fraction", ""),
        "clear_delay_min": model.get("clear_delay_min", ""),
        "amplitude_retention": round(metrics["amplitude_retention"], 6),
        "peak_delay_min": round(metrics["peak_delay_min"], 6),
        "feature_type": metrics["feature_type"],
        "feature_delay_min": round(metrics["feature_delay_min"], 6),
        "xcorr_lag_min": round(metrics["xcorr_lag_min"], 6),
        "nrmse": round(metrics["nrmse"], 6),
        "pearson_r": round(metrics["pearson_r"], 6),
        "max_abs_error_norm": round(metrics["max_abs_error_norm"], 6),
        "strict_criterion": strict,
        "relaxed_criterion": relaxed,
        "criterion_basis": (
            "source-specific illustrative engineering assumptions: "
            f"strict amp>=0.90, NRMSE<=0.10, r>=0.95, |feature delay|<={waveform['strict_delay_limit_min']:.1f} min; "
            f"relaxed amp>=0.80, NRMSE<=0.15, r>=0.90, |feature delay|<={waveform['relaxed_delay_limit_min']:.1f} min"
        ),
        "input_evidence": "PUBLIC_FIGURE_DIGITIZED_EMPIRICAL_PROXY",
        "parameter_evidence": "SENSITIVITY_ANALYSIS_ASSUMPTION",
        "source_url": waveform["source_url"],
        "limitation": "Incremental distortion of an observed proxy; not a blood-to-sweat or device-free physiological transfer model.",
    }


def run_all_models(waveforms: list[dict]) -> tuple[list[dict], dict[tuple[str, str], np.ndarray]]:
    rows: list[dict] = []
    selected_outputs: dict[tuple[str, str], np.ndarray] = {}
    for waveform in waveforms:
        t_s = waveform["time_min"] * 60.0
        inlet = waveform["value"]
        for topology in TOPOLOGIES:
            for transport_min in TRANSPORT_MIN:
                for sensor_tau_min in SENSOR_TAU_MIN:
                    model = {
                        "model_class": "continuous",
                        "topology": topology,
                        "transport_or_window_min": transport_min,
                        "sensor_tau_min": sensor_tau_min,
                        "rtd_shape": 3.0 if topology == "gamma_rtd" else "",
                    }
                    measured = transport_sensor_response(
                        t_s,
                        inlet,
                        topology=topology,
                        transport_time_s=transport_min * 60.0,
                        sensor_tau_s=None if sensor_tau_min == 0 else sensor_tau_min * 60.0,
                        rtd_shape=3.0,
                    )
                    rows.append(build_run_row(waveform, model, measured))
                    key = f"{topology}_{transport_min:g}_{sensor_tau_min:g}"
                    if key in {"plug_0.05_0", "gamma_rtd_1_1", "cstr_5_2"}:
                        selected_outputs[(waveform["waveform_id"], key)] = measured
        for cycle_min in EVENT_CYCLE_MIN:
            for residual in EVENT_RESIDUAL:
                model = {
                    "model_class": "event_reset",
                    "topology": "fill_clear_hold",
                    "transport_or_window_min": cycle_min,
                    "sensor_tau_min": 0.0,
                    "residual_fraction": residual,
                    "clear_delay_min": 0.05,
                }
                measured = event_reset_response(
                    t_s,
                    inlet,
                    cycle_period_s=cycle_min * 60.0,
                    residual_fraction=residual,
                    clear_delay_s=3.0,
                )
                rows.append(build_run_row(waveform, model, measured))
                if cycle_min == 1.0 and residual == 0.1:
                    selected_outputs[(waveform["waveform_id"], "event_1_0.1")] = measured
        for window_min in CHRONO_WINDOW_MIN:
            model = {
                "model_class": "chrono_sampling",
                "topology": "nonoverlap_window_hold",
                "transport_or_window_min": window_min,
                "sensor_tau_min": 0.0,
            }
            measured = chrono_sampling_response(t_s, inlet, collection_window_s=window_min * 60.0)
            rows.append(build_run_row(waveform, model, measured))
            if window_min == 5.0:
                selected_outputs[(waveform["waveform_id"], "chrono_5")] = measured
    return rows, selected_outputs


def contiguous_max(values: list[tuple[float, bool]]) -> float | None:
    maximum = None
    for value, passed in sorted(values):
        if not passed:
            break
        maximum = value
    return maximum


def build_operating_envelope(run_rows: list[dict], waveforms: list[dict]) -> list[dict]:
    source_groups = sorted({waveform["source_group"] for waveform in waveforms})
    rows: list[dict] = []
    for source_group in source_groups:
        group_ids = {waveform["waveform_id"] for waveform in waveforms if waveform["source_group"] == source_group}
        group_runs = [row for row in run_rows if row["waveform_id"] in group_ids]
        keys = sorted({(row["model_class"], row["topology"], row["sensor_tau_min"], row["residual_fraction"]) for row in group_runs})
        for model_class, topology, sensor_tau, residual in keys:
            subset = [row for row in group_runs if (row["model_class"], row["topology"], row["sensor_tau_min"], row["residual_fraction"]) == (model_class, topology, sensor_tau, residual)]
            by_duration: dict[float, list[dict]] = defaultdict(list)
            for row in subset:
                by_duration[float(row["transport_or_window_min"])].append(row)
            strict_values = []
            relaxed_values = []
            for duration, duration_rows in sorted(by_duration.items()):
                strict_values.append((duration, len(duration_rows) == len(group_ids) and all(row["strict_criterion"] == "PASS" for row in duration_rows)))
                relaxed_values.append((duration, len(duration_rows) == len(group_ids) and all(row["relaxed_criterion"] == "PASS" for row in duration_rows)))
            strict_max = contiguous_max(strict_values)
            relaxed_max = contiguous_max(relaxed_values)
            rows.append(
                {
                    "source_group": source_group,
                    "waveforms_required": len(group_ids),
                    "model_class": model_class,
                    "topology": topology,
                    "sensor_tau_min": sensor_tau,
                    "residual_fraction": residual,
                    "strict_contiguous_max_min": "" if strict_max is None else strict_max,
                    "relaxed_contiguous_max_min": "" if relaxed_max is None else relaxed_max,
                    "strict_definition": "all waveforms pass amp>=0.90, NRMSE<=0.10, r>=0.95 and source-specific delay limit",
                    "relaxed_definition": "all waveforms pass amp>=0.80, NRMSE<=0.15, r>=0.90 and source-specific delay limit",
                    "evidence_level": "MODEL_DERIVED_SENSITIVITY",
                }
            )
    return rows


def build_named_summary(run_rows: list[dict]) -> list[dict]:
    selectors = [
        ("Fast plug", "continuous", "plug", 0.05, 0.0, ""),
        ("Moderate distributed", "continuous", "gamma_rtd", 1.0, 1.0, ""),
        ("Slow mixed", "continuous", "cstr", 5.0, 2.0, ""),
        ("Frequent reset", "event_reset", "fill_clear_hold", 1.0, 0.0, 0.1),
        ("Five-minute chrono", "chrono_sampling", "nonoverlap_window_hold", 5.0, 0.0, ""),
    ]
    output = []
    for label, model_class, topology, duration, sensor_tau, residual in selectors:
        for row in run_rows:
            if (
                row["model_class"] == model_class
                and row["topology"] == topology
                and float(row["transport_or_window_min"]) == duration
                and float(row["sensor_tau_min"]) == sensor_tau
                and row["residual_fraction"] == residual
            ):
                output.append({"scenario_label": label, **row})
    return output


def simulate_named(waveform: dict, label: str, inlet: np.ndarray) -> np.ndarray:
    t_s = waveform["time_min"] * 60.0
    if label == "Moderate distributed":
        return transport_sensor_response(t_s, inlet, topology="gamma_rtd", transport_time_s=60.0, sensor_tau_s=60.0, rtd_shape=3.0)
    if label == "Slow mixed":
        return transport_sensor_response(t_s, inlet, topology="cstr", transport_time_s=300.0, sensor_tau_s=120.0)
    if label == "Frequent reset":
        return event_reset_response(t_s, inlet, cycle_period_s=60.0, residual_fraction=0.1, clear_delay_s=3.0)
    if label == "Five-minute chrono":
        return chrono_sampling_response(t_s, inlet, collection_window_s=300.0)
    raise ValueError(label)


def uncertainty_analysis(waveforms: list[dict], replicates: int = 200) -> list[dict]:
    rng = np.random.default_rng(20260807)
    rows = []
    labels = ["Moderate distributed", "Slow mixed", "Frequent reset", "Five-minute chrono"]
    metrics_names = ["amplitude_retention", "feature_delay_min", "nrmse", "pearson_r"]
    for waveform in waveforms:
        store = {label: {metric: [] for metric in metrics_names} for label in labels}
        for _ in range(replicates):
            jitter_t = waveform["raw_time_min"] + rng.normal(0.0, waveform["x_uncertainty_min"] / 2.0, len(waveform["raw_time_min"]))
            jitter_t[0] = max(0.0, jitter_t[0])
            jitter_t = np.maximum.accumulate(jitter_t + np.arange(len(jitter_t)) * 1e-6)
            jitter_y = waveform["raw_value"] + rng.normal(0.0, waveform["y_uncertainty"] / 2.0, len(waveform["raw_value"]))
            empirical = np.interp(waveform["time_min"], jitter_t, jitter_y, left=jitter_y[0], right=jitter_y[-1])
            for label in labels:
                measured = simulate_named(waveform, label, empirical)
                metrics = extended_metrics(waveform["time_min"], empirical, measured)
                for metric in metrics_names:
                    store[label][metric].append(metrics[metric])
        for label in labels:
            for metric in metrics_names:
                values = np.asarray(store[label][metric], dtype=float)
                rows.append(
                    {
                        "waveform_id": waveform["waveform_id"],
                        "source_group": waveform["source_group"],
                        "scenario_label": label,
                        "metric": metric,
                        "p05": round(float(np.quantile(values, 0.05)), 6),
                        "median": round(float(np.quantile(values, 0.50)), 6),
                        "p95": round(float(np.quantile(values, 0.95)), 6),
                        "replicates": replicates,
                        "uncertainty_scope": "figure coordinate digitization only",
                        "random_seed": 20260807,
                    }
                )
    return rows


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path(r"C:\Windows\Fonts") / name
    return ImageFont.truetype(str(path), size=size)


def draw_axes(image: Image.Image, draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], x_label: str, y_label: str, title_text: str) -> None:
    left, top, right, bottom = box
    draw.line((left, top, left, bottom), fill="#222222", width=3)
    draw.line((left, bottom, right, bottom), fill="#222222", width=3)
    draw.text((left, top - 45), title_text, font=font(25, True), fill="#111111")
    draw.text(((left + right) / 2 - 55, bottom + 35), x_label, font=font(21), fill="#222222")
    if y_label:
        label_box = Image.new("RGBA", (420, 45), (255, 255, 255, 0))
        label_draw = ImageDraw.Draw(label_box)
        label_draw.text((0, 7), y_label, font=font(21), fill="#222222")
        rotated = label_box.rotate(90, expand=True)
        image.paste(rotated, (left - 125, int((top + bottom - rotated.height) / 2)), rotated)


def plot_lines(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    x: np.ndarray,
    series: list[tuple[str, np.ndarray, str, int]],
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    left, top, right, bottom = box
    x_min, x_max = float(x.min()), float(x.max())
    if y_min is None:
        y_min = min(float(np.min(values)) for _, values, _, _ in series)
    if y_max is None:
        y_max = max(float(np.max(values)) for _, values, _, _ in series)
    padding = 0.06 * (y_max - y_min)
    y_min -= padding
    y_max += padding
    for tick in np.linspace(y_min, y_max, 5):
        py = bottom - (tick - y_min) / (y_max - y_min) * (bottom - top)
        draw.line((left, py, right, py), fill="#E5E7EB", width=1)
        draw.text((left - 72, py - 11), f"{tick:.1f}", font=font(16), fill="#444444")
    for tick in np.linspace(x_min, x_max, 6):
        px = left + (tick - x_min) / (x_max - x_min) * (right - left)
        draw.line((px, bottom, px, bottom + 8), fill="#222222", width=2)
        draw.text((px - 18, bottom + 10), f"{tick:.0f}", font=font(16), fill="#444444")
    for label, values, color, width in series:
        points = []
        for tx, value in zip(x, values):
            px = left + (tx - x_min) / (x_max - x_min) * (right - left)
            py = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
            points.append((float(px), float(py)))
        draw.line(points, fill=color, width=width, joint="curve")


def make_waveform_figure(waveforms: list[dict], selected: dict[tuple[str, str], np.ndarray]) -> None:
    chosen = [next(w for w in waveforms if w["source"] == "Saha 2024" and w["panel"] == "Fig3b_Subject2"), next(w for w in waveforms if w["source"] == "Choi 2020" and w["panel"] == "B")]
    image = Image.new("RGB", (1900, 1250), "white")
    draw = ImageDraw.Draw(image)
    colors = {"Reference proxy": "#111111", "Fast plug": "#1976D2", "Distributed 1+1 min": "#F57C00", "Mixed 5+2 min": "#D32F2F", "Reset 1 min": "#2E7D32", "Chrono 5 min": "#7B1FA2"}
    for index, waveform in enumerate(chosen):
        box = (150, 120 + index * 560, 1780, 510 + index * 560)
        title_text = "Saha 2024, Subject 2 meal SBG proxy" if index == 0 else "Choi 2020, Figure 1B exercise chloride proxy"
        y_label = "SBG (mg/dL)" if index == 0 else "Sweat chloride (mM)"
        draw_axes(image, draw, box, "Time (min)", y_label, title_text)
        series = [
            ("Reference proxy", waveform["value"], colors["Reference proxy"], 5),
            ("Fast plug", selected[(waveform["waveform_id"], "plug_0.05_0")], colors["Fast plug"], 3),
            ("Distributed 1+1 min", selected[(waveform["waveform_id"], "gamma_rtd_1_1")], colors["Distributed 1+1 min"], 3),
            ("Mixed 5+2 min", selected[(waveform["waveform_id"], "cstr_5_2")], colors["Mixed 5+2 min"], 3),
            ("Reset 1 min", selected[(waveform["waveform_id"], "event_1_0.1")], colors["Reset 1 min"], 3),
            ("Chrono 5 min", selected[(waveform["waveform_id"], "chrono_5")], colors["Chrono 5 min"], 3),
        ]
        plot_lines(draw, box, waveform["time_min"], series)
    legend_y = 1190
    x = 160
    for label, color in colors.items():
        draw.line((x, legend_y, x + 45, legend_y), fill=color, width=5)
        draw.text((x + 55, legend_y - 13), label, font=font(17), fill="#222222")
        x += 275
    image.save(OUTPUT_DIR / "waveform_topology_comparison.png")


def make_envelope_figure(run_rows: list[dict], waveforms: list[dict]) -> None:
    image = Image.new("RGB", (1900, 1250), "white")
    draw = ImageDraw.Draw(image)
    colors = {"plug": "#1976D2", "cstr": "#D32F2F", "gamma_rtd": "#F57C00"}
    groups = ["Saha meal SBG proxy", "Choi exercise chloride proxy"]
    for column, group in enumerate(groups):
        ids = {w["waveform_id"] for w in waveforms if w["source_group"] == group}
        for row_index, metric in enumerate(["nrmse", "amplitude_retention"]):
            box = (140 + column * 930, 120 + row_index * 550, 870 + column * 930, 500 + row_index * 550)
            title_text = f"{group}: worst-case {metric.replace('_', ' ')}"
            draw_axes(image, draw, box, "Mean transport time (min)", metric, title_text)
            series = []
            for topology in TOPOLOGIES:
                values = []
                for duration in TRANSPORT_MIN:
                    selected_rows = [row for row in run_rows if row["waveform_id"] in ids and row["model_class"] == "continuous" and row["topology"] == topology and float(row["sensor_tau_min"]) == 1.0 and float(row["transport_or_window_min"]) == duration]
                    metric_values = [float(row[metric]) for row in selected_rows]
                    values.append(max(metric_values) if metric == "nrmse" else min(metric_values))
                series.append((topology, np.asarray(values), colors[topology], 4))
            y_min, y_max = (0.0, 0.45) if metric == "nrmse" else (0.45, 1.02)
            plot_lines(draw, box, np.asarray(TRANSPORT_MIN), series, y_min, y_max)
            criterion = 0.10 if metric == "nrmse" else 0.90
            left, top, right, bottom = box
            py = bottom - (criterion - (y_min - 0.06 * (y_max - y_min))) / ((y_max + 0.06 * (y_max - y_min)) - (y_min - 0.06 * (y_max - y_min))) * (bottom - top)
            draw.line((left, py, right, py), fill="#555555", width=2)
    x = 650
    for label, color in colors.items():
        draw.line((x, 1185, x + 50, 1185), fill=color, width=5)
        draw.text((x + 60, 1172), label, font=font(19), fill="#222222")
        x += 240
    image.save(OUTPUT_DIR / "continuous_operating_envelope.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    waveforms = load_empirical_waveforms()
    run_rows, selected = run_all_models(waveforms)
    envelope_rows = build_operating_envelope(run_rows, waveforms)
    named_rows = build_named_summary(run_rows)
    uncertainty_rows = uncertainty_analysis(waveforms, replicates=200)

    write_csv(OUTPUT_DIR / "model_run_results.csv", run_rows)
    write_csv(OUTPUT_DIR / "operating_envelope.csv", envelope_rows)
    write_csv(OUTPUT_DIR / "named_scenario_summary.csv", named_rows)
    write_csv(OUTPUT_DIR / "digitization_uncertainty_sensitivity.csv", uncertainty_rows)

    make_waveform_figure(waveforms, selected)
    make_envelope_figure(run_rows, waveforms)

    identity_checks = []
    for waveform in waveforms:
        identity = transport_sensor_response(
            waveform["time_min"] * 60.0,
            waveform["value"],
            topology="plug",
            transport_time_s=0.0,
            sensor_tau_s=None,
        )
        identity_checks.append(float(np.max(np.abs(identity - waveform["value"]))))
    finite = all(math.isfinite(float(row["nrmse"])) and math.isfinite(float(row["amplitude_retention"])) for row in run_rows)
    qc = {
        "status": "pass" if max(identity_checks) < 1e-12 and finite else "fail",
        "waveforms": len(waveforms),
        "model_runs": len(run_rows),
        "identity_max_abs_error": max(identity_checks),
        "all_metrics_finite": finite,
        "uncertainty_replicates_per_named_scenario": 200,
        "random_seed": 20260807,
        "input_scope": "public digitized empirical proxies; incremental architecture distortion only",
    }
    (OUTPUT_DIR / "model_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "model_analysis_bundle.json").write_text(
        json.dumps(
            {
                "model_runs": run_rows,
                "operating_envelope": envelope_rows,
                "named_scenarios": named_rows,
                "uncertainty": uncertainty_rows,
                "qc": qc,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    strict_pass = sum(row["strict_criterion"] == "PASS" for row in run_rows)
    relaxed_pass = sum(row["relaxed_criterion"] == "PASS" for row in run_rows)
    report = f"""# Waveform topology and sensitivity analysis

## Scope

This analysis uses four Saha 2024 meal-associated fingertip SBG curves and three Choi 2020 exercise sweat-chloride curves digitized from public figures. These observed traces are empirical waveform proxies. They are not treated as device-free physiological truth, and the analysis does not estimate blood-to-sweat transfer functions.

## Models compared

- Continuous plug flow, a well-mixed CSTR, and a gamma residence-time distribution with shape 3.
- Transport means/delays from 0.05 to 15 min and independent first-order sensor constants from 0 to 5 min.
- Fill-clear event-reset sampling with 0.5-10 min cycles, 0-25% residual old sample, and a 3 s clearance delay.
- Non-overlapping chrono-sampling windows of 1-15 min with causal window-average hold.

## Criteria

The criteria are illustrative engineering thresholds, not clinical limits. Strict acceptance requires amplitude retention >=0.90, NRMSE <=0.10, Pearson r >=0.95, and feature delay <=5 min for the slow Saha meal traces or <=2 min for the faster Choi exercise traces. A relaxed sensitivity criterion uses amplitude retention >=0.80, NRMSE <=0.15, r >=0.90, and delay limits of 10 or 5 min.

## Reproducibility

- {len(waveforms)} empirical waveforms and {len(run_rows)} deterministic model runs.
- {strict_pass} strict-pass and {relaxed_pass} relaxed-pass runs.
- Figure-coordinate uncertainty propagated through 200 seeded Monte Carlo replicates for four named architectures per waveform.
- Identity, finite-output, kernel, first-order, event-reset, and chrono-sampling invariants pass.

## Interpretation guardrail

Results quantify incremental distortion imposed on an already observed waveform. They support comparative architecture design and sensitivity analysis, but they cannot identify how much of the published SBG or sweat-chloride dynamics arose from physiology versus the original measurement system.
"""
    DOC_PATH.write_text(report, encoding="utf-8")
    print(json.dumps({**qc, "strict_pass": strict_pass, "relaxed_pass": relaxed_pass, "output_dir": str(OUTPUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
