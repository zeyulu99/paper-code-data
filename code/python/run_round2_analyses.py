"""Analyses requested by the simulated second-round review (2026-09-22).

A. Dense-grid envelopes at tau_s = 0 and 1 min, with the binding criterion of
   each boundary, and a delay-aligned (shape-only) envelope.
B. Device budget: largest coupled volume V_eff = tau_f,max * q'' * A for the
   areal sweat rates reported by Nyein et al. (3, 50, 1000 nL min-1 cm-2,
   3 mm well, A = 0.0707 cm2), and placement of the 72 nL well itself.
C. Time-varying flux Q(t): a well-mixed zone versus fill--measure--clear on the
   same input, for steady, burst, and declining flux with equal mean.
D. Fill--measure--clear and chronological-window results of the existing
   1,295-row empirical stress test, summarized per trace group.

All parameters are declared for illustration. Outputs go to
outputs/revision_2026-09-22/round2_analyses/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_figure2_strategies import strict_pass, transport_fast
from run_evidence_closure_analysis import (
    canonical_metrics,
    canonical_waveform,
    delay_aligned_metrics,
)

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "outputs" / "revision_2026-09-22" / "round2_analyses"
OUT.mkdir(parents=True, exist_ok=True)

ANCHORS = np.asarray([0.05, 0.10, 0.25, 0.50, 0.75, 1.0, 2.0, 3.0, 3.7, 5.0, 10.0, 15.0])
TAU_F = np.unique(np.round(np.concatenate([np.geomspace(0.05, 15.0, 61), ANCHORS]), 8))
KINDS = ["pulse", "ramp", "sinusoid"]
TIMESCALES = [2.0, 5.0, 10.0, 30.0, 60.0]
TOPOLOGIES = ["plug", "cstr", "gamma_rtd"]
TOPO_NAME = {"plug": "plug flow", "cstr": "well mixed", "gamma_rtd": "gamma RTD"}


def failed_criteria(kind, T, m):
    out = []
    if m["amplitude_retention"] < 0.90:
        out.append("amplitude")
    if m["nrmse"] > 0.10:
        out.append("NRMSE")
    if m["pearson_r"] < 0.95:
        out.append("r")
    if abs(m["feature_delay_min"]) > max(1.0, T / 4.0):
        out.append("delay")
    return out


def shape_pass(m, a):
    return bool(m["amplitude_retention"] >= 0.90 and a["aligned_nrmse"] <= 0.10
                and a["aligned_pearson_r"] >= 0.95)


# ------------------------------------------------------------------ A
def analysis_a():
    rows = []
    for tau_s in (0.0, 1.0):
        for kind in KINDS:
            for T in TIMESCALES:
                time_s, inlet = canonical_waveform(kind, T)
                for topo in TOPOLOGIES:
                    for tf in TAU_F:
                        _, s = transport_fast(time_s, inlet, topo, float(tf), sensor_min=tau_s)
                        m = canonical_metrics(kind, T, time_s, inlet, s)
                        if kind == "sinusoid":
                            per = T * 60.0
                            mask = time_s >= time_s[-1] - 3.0 * per
                            a = delay_aligned_metrics(time_s[mask], inlet[mask], s[mask], m["feature_delay_min"])
                        else:
                            a = delay_aligned_metrics(time_s, inlet, s, m["feature_delay_min"])
                        rows.append({
                            "tau_s_min": tau_s, "waveform": kind, "T_min": T, "topology": topo,
                            "tau_f_min": float(tf), **{k: m[k] for k in ("amplitude_retention", "nrmse", "pearson_r", "feature_delay_min")},
                            **a, "pass_strict": strict_pass(kind, T, m),
                            "pass_shape_only": shape_pass(m, a),
                            "failed": "+".join(failed_criteria(kind, T, m)),
                        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "A_dense_grid_tau_s_0_and_1.csv", index=False)

    summ = []
    for (ts, kind, T, topo), g in df.groupby(["tau_s_min", "waveform", "T_min", "topology"]):
        g = g.sort_values("tau_f_min")
        rec = {"tau_s_min": ts, "waveform": kind, "T_min": T, "topology": topo}
        for col, tag in (("pass_strict", "strict"), ("pass_shape_only", "shape")):
            ok = g[col].to_numpy()
            if ok[0]:
                # contiguous envelope: passing run from the smallest tested value
                n = int(np.argmin(ok)) if not ok.all() else len(ok)
                last = g.tau_f_min.iloc[n - 1]
                above = g[(g.tau_f_min > last)]
                rec[f"{tag}_max_tau_f"] = last
                rec[f"{tag}_next_fail"] = above.tau_f_min.min() if len(above) else None
                if tag == "strict" and len(above):
                    rec["binding_criterion"] = above.iloc[0]["failed"]
            else:
                rec[f"{tag}_max_tau_f"] = None
                rec[f"{tag}_next_fail"] = g.tau_f_min.min()
                if tag == "strict":
                    rec["binding_criterion"] = g.iloc[0]["failed"] + " (at 0.05 min)"
        summ.append(rec)
    s = pd.DataFrame(summ)
    s.to_csv(OUT / "A_envelope_summary.csv", index=False)
    # consistency with the released dense grid at tau_s = 1
    ref = pd.read_csv(PROJECT / "outputs/figure_working/figure2_strategy_comparison_2026-09-09/data/canonical_dense_boundary_grid_tau_sensor_1min.csv")
    ref["p"] = ref.pass_strict.astype(str).str.upper().eq("PASS")
    mine = df[df.tau_s_min == 1.0]
    mism = 0
    for (kind, T, topo, tf), p in zip(mine[["waveform", "T_min", "topology", "tau_f_min"]].itertuples(index=False), mine.pass_strict):
        r = ref[(ref.waveform_type == kind) & (ref.T_min == T) & (ref.topology == topo) & (np.isclose(ref.tau_transport_min, tf))]
        if len(r) == 1 and bool(r.p.iloc[0]) != bool(p):
            mism += 1
    return s, mism


# ------------------------------------------------------------------ B
AREA_CM2 = np.pi * 0.15 ** 2          # 3 mm diameter well
STEADY_DELAY = {}                      # peak delay calibrated at steady flux
FLUX = {"low (3)": 3.0, "medium (50)": 50.0, "high (1000)": 1000.0}   # nL min-1 cm-2
V_WELL = 72.0                          # nL


def analysis_b(summary):
    s1 = summary[(summary.tau_s_min == 1.0) & (summary.topology == "cstr")]
    rows = []
    for kind, T in [("pulse", 10.0), ("pulse", 30.0), ("pulse", 60.0), ("ramp", 10.0),
                    ("ramp", 30.0), ("ramp", 60.0), ("sinusoid", 30.0), ("sinusoid", 60.0)]:
        r = s1[(s1.waveform == kind) & (s1.T_min == T)].iloc[0]
        tf = r.strict_max_tau_f
        rec = {"waveform": kind, "T_min": T, "tau_f_max_min": tf}
        for lab, q in FLUX.items():
            Q = q * AREA_CM2
            rec[f"V_eff_max_nL_{lab}"] = None if tf is None or np.isnan(tf) else tf * Q
        rows.append(rec)
    budget = pd.DataFrame(rows)
    budget.to_csv(OUT / "B_volume_budget.csv", index=False)
    well = []
    for lab, q in FLUX.items():
        Q = q * AREA_CM2
        tau = V_WELL / Q
        ok = budget[budget.tau_f_max_min >= tau][["waveform", "T_min"]].apply(lambda x: f"{x.iloc[0]} {x.iloc[1]:g}", axis=1).tolist()
        well.append({"flux": lab, "Q_nL_min": Q, "tau_f_min": tau, "t90_min": np.log(10) * tau, "cells_passed": ok})
    pd.DataFrame(well).to_csv(OUT / "B_nyein_well_placement.csv", index=False)
    return budget, well


# ------------------------------------------------------------------ C
def analysis_c():
    dt = 0.05
    t = np.arange(0.0, 300.0 + dt, dt)
    cin = 0.1 + 0.9 * np.exp(-0.5 * ((t - 80.0) / (30.0 / 2.355)) ** 2) \
        + 0.6 * np.exp(-0.5 * ((t - 200.0) / (30.0 / 2.355)) ** 2)
    V = V_WELL
    qmean = 50.0
    scenarios = {
        "steady": np.full_like(t, qmean),
        "burst (10 min on/off, 9:1)": np.where((t // 10) % 2 == 0, 90.0, 10.0),
        "declining (95 to 5)": 95.0 - 90.0 * t / t[-1],
    }
    f_res = 0.15
    rows, traces = [], {"t_min": t, "c_in": cin}
    for name, qflux in scenarios.items():
        Q = qflux * AREA_CM2                      # nL/min
        # well-mixed through-flow, exact update per step
        c = np.empty_like(t); c[0] = cin[0]
        for i in range(1, len(t)):
            k = Q[i - 1] / V
            c[i] = cin[i - 1] + (c[i - 1] - cin[i - 1]) * np.exp(-k * dt)
        traces[f"through_{name}"] = c
        delays = []
        for centre in (80.0, 200.0):
            w = (t > centre - 40) & (t < centre + 80)
            delays.append(float(t[w][np.argmax(c[w])] - centre))
        calib = STEADY_DELAY.setdefault(0, delays[0]) if name == "steady" else STEADY_DELAY[0]
        # fill--measure--clear: chamber of volume V fills at Q(t); readout at fill end
        vol, start, prev = 0.0, 0, cin[0]
        reads = []
        for i in range(1, len(t)):
            vol += Q[i - 1] * dt
            if vol >= V:
                wts = Q[start:i] * dt
                fresh = float(np.sum(wts * cin[start:i]) / np.sum(wts))
                val = f_res * prev + (1 - f_res) * fresh
                reads.append((t[start], t[i], val, fresh))
                prev, vol, start = val, 0.0, i
        win = np.array([r[1] - r[0] for r in reads])
        err = np.array([abs(r[2] - r[3]) for r in reads])
        rows.append({
            "scenario": name, "mean_flux_nL_min_cm2": float(np.mean(qflux)),
            "through_peak_delay_event1_min": delays[0], "through_peak_delay_event2_min": delays[1],
            "delay_calibrated_at_steady_flux_min": calib,
            "timing_error_event1_min": delays[0] - calib, "timing_error_event2_min": delays[1] - calib,
            "through_amp_retention_event1": float(c[(t > 40) & (t < 160)].max() - 0.1) / 0.9,
            "fmc_n_readouts": len(reads), "fmc_window_min_median": float(np.median(win)),
            "fmc_window_min_max": float(win.max()), "fmc_max_residual_error": float(err.max()),
        })
        traces[f"fmc_{name}"] = reads
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "C_time_varying_flux.csv", index=False)
    oka = [{"site": site, "phase": ph, "flux_nL_min_cm2": q * 1000.0,
            "tau_f_72nL_well_min": V / (q * 1000.0 * AREA_CM2)}
           for site, vals in {"forehead": (0.66, 1.85, 2.37), "upper arm": (0.37, 0.99, 1.29)}.items()
           for ph, q in zip(("onset-FP1", "FP1-FP2", "FP2-exhaustion"), vals)]
    pd.DataFrame(oka).to_csv(OUT / "C_okawara_exercise_flux.csv", index=False)
    return df


# ------------------------------------------------------------------ D
def analysis_d():
    e = pd.read_csv(PROJECT / "outputs/revision_2026-08-31_figure2_data_package/07_empirical_stress_test_full_1295.csv")
    e["group"] = e.source_group.str.split().str[0]
    rows = []
    fmc = e[e.topology == "fill_clear_hold"]
    for (g, per, fr), h in fmc.groupby(["group", "tau_transport_or_window_min", "residual_fraction"]):
        rows.append({"class": "fill-measure-clear", "group": g, "cycle_min": per, "f_res": fr,
                     "n_traces": len(h), "n_strict_pass": int((h.pass_strict == "PASS").sum()),
                     "median_amp": h.amplitude_retention.median(), "median_nrmse": h.nrmse.median(),
                     "median_delay_min": h.feature_delay_min.median()})
    ch = e[e.topology == "nonoverlap_window_hold"]
    for (g, w), h in ch.groupby(["group", "tau_transport_or_window_min"]):
        rows.append({"class": "chronological window", "group": g, "cycle_min": w, "f_res": None,
                     "n_traces": len(h), "n_strict_pass": int((h.pass_strict == "PASS").sum()),
                     "median_amp": h.amplitude_retention.median(), "median_nrmse": h.nrmse.median(),
                     "median_delay_min": h.feature_delay_min.median()})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "D_fmc_chrono_empirical.csv", index=False)
    return d


if __name__ == "__main__":
    summary, mism = analysis_a()
    budget, well = analysis_b(summary)
    c = analysis_c()
    d = analysis_d()
    meta = {"dense_grid_mismatches_vs_released_tau_s_1": mism, "area_cm2": AREA_CM2,
            "n_tau_f": int(len(TAU_F))}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(meta)
    print(summary.to_string())
    print(budget.to_string()); print(pd.DataFrame(well).to_string())
    print(c.to_string()); print(d.to_string())
