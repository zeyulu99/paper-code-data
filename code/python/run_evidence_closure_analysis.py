"""Canonical-input and external-check analysis for the revised Perspective.

The output is an engineering sensitivity analysis. It does not infer biological
ground truth from published wearable traces and does not impute missing device
parameters.
"""

from __future__ import annotations

import json
from math import log, pi
from pathlib import Path

import numpy as np

from transport_fidelity_models import fidelity_metrics, transport_sensor_response


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT / "outputs" / "revision_2026-08-29" / "evidence_closure_analysis"


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def first_crossing(time_s: np.ndarray, values: np.ndarray, level: float) -> float:
    indices = np.where(values >= level)[0]
    if len(indices) == 0:
        return float("nan")
    return float(time_s[indices[0]])


def canonical_metrics(
    kind: str,
    characteristic_min: float,
    time_s: np.ndarray,
    reference: np.ndarray,
    measured: np.ndarray,
) -> dict:
    """Metrics matched to pulse, monotonic-ramp, and periodic signals."""
    if kind == "pulse":
        m = fidelity_metrics(time_s, reference, measured)
        return {
            "amplitude_retention": m.amplitude_retention,
            "nrmse": m.nrmse,
            "pearson_r": correlation(reference, measured),
            "feature_delay_min": m.peak_delay_s / 60.0,
            "feature_definition": "peak time",
        }

    if kind == "ramp":
        ref_amp = float(np.max(reference) - np.min(reference))
        obs_amp = float(np.max(measured) - np.min(measured))
        ref_level = float(np.min(reference) + 0.5 * ref_amp)
        obs_level = float(np.min(measured) + 0.5 * obs_amp)
        delay = first_crossing(time_s, measured, obs_level) - first_crossing(time_s, reference, ref_level)
        return {
            "amplitude_retention": obs_amp / ref_amp,
            "nrmse": float(np.sqrt(np.mean((measured - reference) ** 2)) / ref_amp),
            "pearson_r": correlation(reference, measured),
            "feature_delay_min": delay / 60.0,
            "feature_definition": "t50 crossing",
        }

    period_s = characteristic_min * 60.0
    mask = time_s >= time_s[-1] - 3.0 * period_s
    ref = reference[mask]
    obs = measured[mask]
    t = time_s[mask]
    dt = float(t[1] - t[0])
    ref_centered = ref - np.mean(ref)
    obs_centered = obs - np.mean(obs)
    corr = np.correlate(obs_centered, ref_centered, mode="full")
    lags = np.arange(-len(ref_centered) + 1, len(ref_centered)) * dt
    allowed = np.abs(lags) <= period_s / 2.0
    phase_delay_s = float(lags[allowed][np.argmax(corr[allowed])])
    ref_amp = float(np.max(ref) - np.min(ref))
    obs_amp = float(np.max(obs) - np.min(obs))
    return {
        "amplitude_retention": obs_amp / ref_amp,
        "nrmse": float(np.sqrt(np.mean((obs - ref) ** 2)) / ref_amp),
        "pearson_r": correlation(ref, obs),
        "feature_delay_min": phase_delay_s / 60.0,
        "feature_definition": "steady-state cross-correlation phase",
    }


def delay_aligned_metrics(
    time_s: np.ndarray,
    reference: np.ndarray,
    measured: np.ndarray,
    feature_delay_min: float,
) -> dict:
    """Score waveform shape after removing a known, independently calibrated delay."""
    delay_s = feature_delay_min * 60.0
    aligned = np.interp(time_s + delay_s, time_s, measured, left=np.nan, right=np.nan)
    mask = np.isfinite(aligned)
    ref = reference[mask]
    obs = aligned[mask]
    ref_amp = float(np.max(ref) - np.min(ref))
    obs_amp = float(np.max(obs) - np.min(obs))
    return {
        "aligned_amplitude_retention": obs_amp / ref_amp,
        "aligned_nrmse": float(np.sqrt(np.mean((obs - ref) ** 2)) / ref_amp),
        "aligned_pearson_r": correlation(ref, obs),
    }


def canonical_waveform(kind: str, characteristic_min: float) -> tuple[np.ndarray, np.ndarray]:
    dt_min = 0.05
    duration_min = max(180.0, 8.0 * characteristic_min)
    t_min = np.arange(0.0, duration_min + dt_min, dt_min)
    if kind == "pulse":
        center = duration_min * 0.35
        sigma = characteristic_min / 2.355
        y = np.exp(-0.5 * ((t_min - center) / sigma) ** 2)
    elif kind == "ramp":
        start = duration_min * 0.2
        y = np.clip((t_min - start) / characteristic_min, 0.0, 1.0)
    elif kind == "sinusoid":
        y = 0.5 + 0.5 * np.sin(2.0 * pi * t_min / characteristic_min)
    else:
        raise ValueError(kind)
    return t_min * 60.0, y


def run_canonical() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    timescales = [2.0, 5.0, 10.0, 30.0, 60.0]
    transport_grid = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0]
    sensor_grid = [0.0, 1.0, 2.0, 5.0]
    for kind in ["pulse", "ramp", "sinusoid"]:
        for characteristic in timescales:
            t_s, inlet = canonical_waveform(kind, characteristic)
            for topology in ["plug", "cstr", "gamma_rtd"]:
                for transport_min in transport_grid:
                    for sensor_min in sensor_grid:
                        measured = transport_sensor_response(
                            t_s,
                            inlet,
                            topology=topology,
                            transport_time_s=transport_min * 60.0,
                            sensor_tau_s=None if sensor_min == 0 else sensor_min * 60.0,
                        )
                        m = canonical_metrics(kind, characteristic, t_s, inlet, measured)
                        aligned = delay_aligned_metrics(
                            t_s, inlet, measured, m["feature_delay_min"]
                        )
                        delay_limit = max(1.0, 0.25 * characteristic)
                        strict = (
                            m["amplitude_retention"] >= 0.90
                            and m["nrmse"] <= 0.10
                            and m["pearson_r"] >= 0.95
                            and abs(m["feature_delay_min"]) <= delay_limit
                        )
                        rows.append(
                            {
                                "waveform": kind,
                                "characteristic_min": characteristic,
                                "topology": topology,
                                "transport_min": transport_min,
                                "sensor_tau_min": sensor_min,
                                "amplitude_retention": m["amplitude_retention"],
                                "nrmse": m["nrmse"],
                                "pearson_r": m["pearson_r"],
                                "feature_delay_min": m["feature_delay_min"],
                                "feature_definition": m["feature_definition"],
                                **aligned,
                                "strict_pass": strict,
                                "criterion": "illustrative: amp>=0.90; NRMSE<=0.10; r>=0.95; delay<=max(1 min, T/4)",
                            }
                        )

    summaries: list[dict] = []
    for kind in ["pulse", "ramp", "sinusoid"]:
        for characteristic in timescales:
            for topology in ["plug", "cstr", "gamma_rtd"]:
                subset = [
                    r for r in rows
                    if r["waveform"] == kind
                    and r["characteristic_min"] == characteristic
                    and r["topology"] == topology
                    and r["sensor_tau_min"] == 1.0
                    and r["strict_pass"]
                ]
                summaries.append(
                    {
                        "waveform": kind,
                        "characteristic_min": characteristic,
                        "topology": topology,
                        "sensor_tau_min": 1.0,
                        "maximum_tested_strict_transport_min": max(
                            [r["transport_min"] for r in subset], default=None
                        ),
                    }
                )
    return rows, summaries


def run_sensitivity_checks() -> tuple[list[dict], list[dict]]:
    """Resolve the sub-0.5-min boundary and test gamma-RTD shape dependence."""
    timescales = [2.0, 5.0, 10.0, 30.0, 60.0]
    extended_transport_grid = [0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0, 10.0, 15.0]
    canonical_transport_grid = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0]
    sensor_grid = [0.0, 1.0, 2.0, 5.0]
    shape_grid = [1.0, 2.0, 3.0, 5.0, 10.0]

    extended_rows: list[dict] = []
    for kind in ["pulse", "ramp", "sinusoid"]:
        for characteristic in timescales:
            t_s, inlet = canonical_waveform(kind, characteristic)
            for topology in ["plug", "cstr", "gamma_rtd"]:
                for transport_min in extended_transport_grid:
                    for sensor_min in sensor_grid:
                        measured = transport_sensor_response(
                            t_s,
                            inlet,
                            topology=topology,
                            transport_time_s=transport_min * 60.0,
                            sensor_tau_s=None if sensor_min == 0 else sensor_min * 60.0,
                            rtd_shape=3.0,
                        )
                        m = canonical_metrics(kind, characteristic, t_s, inlet, measured)
                        aligned = delay_aligned_metrics(
                            t_s, inlet, measured, m["feature_delay_min"]
                        )
                        delay_limit = max(1.0, 0.25 * characteristic)
                        strict = (
                            m["amplitude_retention"] >= 0.90
                            and m["nrmse"] <= 0.10
                            and m["pearson_r"] >= 0.95
                            and abs(m["feature_delay_min"]) <= delay_limit
                        )
                        aligned_shape_pass = (
                            aligned["aligned_amplitude_retention"] >= 0.90
                            and aligned["aligned_nrmse"] <= 0.10
                            and aligned["aligned_pearson_r"] >= 0.95
                        )
                        extended_rows.append(
                            {
                                "waveform": kind,
                                "characteristic_min": characteristic,
                                "topology": topology,
                                "transport_min": transport_min,
                                "sensor_tau_min": sensor_min,
                                "rtd_shape": 3.0 if topology == "gamma_rtd" else None,
                                "amplitude_retention": m["amplitude_retention"],
                                "nrmse": m["nrmse"],
                                "pearson_r": m["pearson_r"],
                                "feature_delay_min": m["feature_delay_min"],
                                **aligned,
                                "strict_pass": strict,
                                "aligned_shape_pass": aligned_shape_pass,
                            }
                        )

    shape_rows: list[dict] = []
    for kind in ["pulse", "ramp", "sinusoid"]:
        for characteristic in timescales:
            t_s, inlet = canonical_waveform(kind, characteristic)
            for shape in shape_grid:
                for transport_min in canonical_transport_grid:
                    for sensor_min in sensor_grid:
                        measured = transport_sensor_response(
                            t_s,
                            inlet,
                            topology="gamma_rtd",
                            transport_time_s=transport_min * 60.0,
                            sensor_tau_s=None if sensor_min == 0 else sensor_min * 60.0,
                            rtd_shape=shape,
                        )
                        m = canonical_metrics(kind, characteristic, t_s, inlet, measured)
                        aligned = delay_aligned_metrics(
                            t_s, inlet, measured, m["feature_delay_min"]
                        )
                        delay_limit = max(1.0, 0.25 * characteristic)
                        shape_rows.append(
                            {
                                "waveform": kind,
                                "characteristic_min": characteristic,
                                "topology": "gamma_rtd",
                                "transport_min": transport_min,
                                "sensor_tau_min": sensor_min,
                                "rtd_shape": shape,
                                "amplitude_retention": m["amplitude_retention"],
                                "nrmse": m["nrmse"],
                                "pearson_r": m["pearson_r"],
                                "feature_delay_min": m["feature_delay_min"],
                                **aligned,
                                "strict_pass": (
                                    m["amplitude_retention"] >= 0.90
                                    and m["nrmse"] <= 0.10
                                    and m["pearson_r"] >= 0.95
                                    and abs(m["feature_delay_min"]) <= delay_limit
                                ),
                            }
                        )
    return extended_rows, shape_rows


def summarize_sensitivity_checks(
    extended_rows: list[dict], shape_rows: list[dict]
) -> dict:
    def largest_passing(rows: list[dict]) -> float | None:
        passing = [row["transport_min"] for row in rows if row["strict_pass"]]
        return max(passing, default=None)

    extended_envelopes = []
    for kind in ["pulse", "ramp", "sinusoid"]:
        for characteristic in [2.0, 5.0, 10.0, 30.0, 60.0]:
            for topology in ["plug", "cstr", "gamma_rtd"]:
                subset = [
                    row
                    for row in extended_rows
                    if row["waveform"] == kind
                    and row["characteristic_min"] == characteristic
                    and row["topology"] == topology
                    and row["sensor_tau_min"] == 1.0
                ]
                extended_envelopes.append(
                    {
                        "waveform": kind,
                        "characteristic_min": characteristic,
                        "topology": topology,
                        "sensor_tau_min": 1.0,
                        "maximum_tested_strict_transport_min": largest_passing(subset),
                    }
                )

    shape_envelopes = []
    for kind in ["pulse", "ramp", "sinusoid"]:
        for characteristic in [2.0, 5.0, 10.0, 30.0, 60.0]:
            for shape in [1.0, 2.0, 3.0, 5.0, 10.0]:
                subset = [
                    row
                    for row in shape_rows
                    if row["waveform"] == kind
                    and row["characteristic_min"] == characteristic
                    and row["rtd_shape"] == shape
                    and row["sensor_tau_min"] == 1.0
                ]
                shape_envelopes.append(
                    {
                        "waveform": kind,
                        "characteristic_min": characteristic,
                        "rtd_shape": shape,
                        "sensor_tau_min": 1.0,
                        "maximum_tested_strict_transport_min": largest_passing(subset),
                    }
                )

    aligned_example = [
        row
        for row in extended_rows
        if row["waveform"] == "pulse"
        and row["characteristic_min"] == 60.0
        and row["transport_min"] == 10.0
        and row["sensor_tau_min"] == 1.0
    ]
    return {
        "extended_grid": [0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0, 10.0, 15.0],
        "extended_envelopes_at_sensor_tau_1_min": extended_envelopes,
        "gamma_shape_grid": [1.0, 2.0, 3.0, 5.0, 10.0],
        "gamma_shape_envelopes_at_sensor_tau_1_min": shape_envelopes,
        "delay_aligned_60_min_pulse_at_transport_10_min": aligned_example,
    }


def external_checks() -> list[dict]:
    area_cm2 = pi * (0.3 / 2.0) ** 2
    volume_nl = 72.0
    cases = [
        ("high", 1000.0, 2.3),
        ("medium", 50.0, 46.8),
        ("low", 3.0, 781.0),
    ]
    rows: list[dict] = []
    for label, areal_rate, reported_t90 in cases:
        q_nl_min = areal_rate * area_cm2
        predicted = log(10.0) * volume_nl / q_nl_min
        rows.append(
            {
                "source": "Nyein 2021 SI",
                "case": label,
                "well_volume_nl": volume_nl,
                "collection_area_cm2": area_cm2,
                "areal_rate_nl_min_cm2": areal_rate,
                "derived_absolute_flow_nl_min": q_nl_min,
                "reported_t90_min": reported_t90,
                "cstr_predicted_t90_min": predicted,
                "absolute_error_min": abs(predicted - reported_t90),
                "relative_error_percent": 100.0 * abs(predicted - reported_t90) / reported_t90,
                "evidence_type": "PUBLISHED_CONVECTION_DIFFUSION_SIMULATION",
                "interpretation": "no-fit reduced-order consistency check against a published numerical simulation; not experimental validation",
            }
        )
    rows.append(
        {
            "source": "Kim 2022",
            "case": "spiking clearance",
            "reported_clearance_s": 3.0,
            "model_use": "clear-delay timing anchor only",
            "not_validated": "fill time and residual old-sample fraction",
            "interpretation": "does not validate a complete fill-clear state model",
        }
    )
    return rows


def physiology_register() -> list[dict]:
    return [
        {
            "source": "Choi 2020",
            "analyte": "sweat chloride",
            "scenario": "single, reverse, and multistep exercise-load changes",
            "time_information": "three public continuous representative traces; cohort t50 available for plausibility QC",
            "role": "fast directional empirical waveform proxy",
            "model_eligible": True,
            "limitation": "already filtered device output; not device-free physiology",
        },
        {
            "source": "Saha 2024",
            "analyte": "sweat-based glucose signal",
            "scenario": "meal-associated fingertip monitoring in four subjects",
            "time_information": "four public continuous traces and discrete blood-glucose markers",
            "role": "slow empirical waveform proxy",
            "model_eligible": True,
            "limitation": "blood-to-sweat transfer and original device dynamics are not separable",
        },
        {
            "source": "Xuan 2023",
            "analyte": "sweat lactate",
            "scenario": "cycling and kayaking with paired 5-min blood-lactate summaries",
            "time_information": "continuous profiles; initial adequate-wetting times about 18-27 min in illustrated subjects",
            "role": "intermediate-timescale and body-site validation scenario",
            "model_eligible": False,
            "limitation": "local inlet-to-sensor volume/RTD not reported; used as a summary anchor",
        },
        {
            "source": "Hauke 2018",
            "analyte": "ethanol",
            "scenario": "oral bolus with paired blood and stimulated-sweat monitoring",
            "time_information": "25-s sweat data; approximately 600 nL coupled sample volume and 320 nL/min assumed input gave a 1.9-min exchange estimate; 2.3-11.41 min onset lag; 19.32-34.44 min curve lag",
            "role": "paired systemic-analyte and identifiability scenario",
            "model_eligible": False,
            "limitation": "reported total lag combines biological transfer, device transport, and sensor response",
        },
    ]


def write_tex(external: list[dict], summaries: list[dict], physiology: list[dict]) -> None:
    nyein = [row for row in external if row.get("source") == "Nyein 2021 SI"]
    lines = [
        "% Generated by run_evidence_closure_analysis.py",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Nyein condition & $Q$ (nL min$^{-1}$) & simulation-derived $t_{90}$ (min) & reduced-order $t_{90}$ (min) & difference (\\%) \\\\",
        "\\midrule",
    ]
    for row in nyein:
        lines.append(
            f"{row['case'].capitalize()} & {row['derived_absolute_flow_nl_min']:.3g} & "
            f"{row['reported_t90_min']:.1f} & {row['cstr_predicted_t90_min']:.1f} & "
            f"{row['relative_error_percent']:.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (OUTPUT / "published_simulation_benchmark_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runs, summaries = run_canonical()
    extended_runs, shape_runs = run_sensitivity_checks()
    sensitivity_summary = summarize_sensitivity_checks(extended_runs, shape_runs)
    external = external_checks()
    physiology = physiology_register()
    bundle = {
        "analysis_scope": "illustrative engineering sensitivity analysis with a no-fit reduced-order check against a published numerical simulation",
        "canonical_run_count": len(runs),
        "canonical_summary": summaries,
        "extended_grid_run_count": len(extended_runs),
        "gamma_shape_run_count": len(shape_runs),
        "sensitivity_checks": sensitivity_summary,
        "external_checks": external,
        "physiology_scenarios": physiology,
        "guardrails": [
            "No missing device parameter is imputed.",
            "Empirical wearable traces are proxies, not physiological ground truth.",
            "Strict criteria are illustrative engineering thresholds, not clinical acceptance limits.",
            "The Nyein comparison reproduces published convection-diffusion simulation-derived times and is not experimental validation; all topology classes require matching experimental data for validation.",
        ],
    }
    (OUTPUT / "analysis_bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    (OUTPUT / "canonical_runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    (OUTPUT / "canonical_extended_grid.json").write_text(
        json.dumps(extended_runs, indent=2), encoding="utf-8"
    )
    (OUTPUT / "gamma_shape_sensitivity.json").write_text(
        json.dumps(shape_runs, indent=2), encoding="utf-8"
    )
    write_tex(external, summaries, physiology)
    print(json.dumps({
        "status": "complete",
        "canonical_runs": len(runs),
        "extended_grid_runs": len(extended_runs),
        "gamma_shape_runs": len(shape_runs),
        "external_checks": len(external),
        "physiology_scenarios": len(physiology),
        "output": str(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
