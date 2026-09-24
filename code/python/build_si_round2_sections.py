"""Insert the round-2 analyses into the SI (writes supporting_information_v4.tex).

All numbers are read from outputs/revision_2026-09-22/round2_analyses/*.csv,
so the SI tables cannot drift from the computed values.
"""
from pathlib import Path

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[2]
A = P / "outputs/revision_2026-09-22/round2_analyses"
SRC = P / "manuscript/revision_evidence_complete/supporting_information_figures_revised.tex"
DST = SRC.with_name("supporting_information_v4.tex")
s = SRC.read_text(encoding="utf-8")
BS = "\\"


def f(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if x >= 15.0 - 1e-9:
        return r"$\geq$15"
    return f"{x:.{nd}f}"


summ = pd.read_csv(A / "A_envelope_summary.csv")
topo_order = [("cstr", "WM"), ("gamma_rtd", "G"), ("plug", "P")]
rows = []
for kind, lab in [("pulse", "Pulse"), ("ramp", "Ramp"), ("sinusoid", "Sinusoid")]:
    for T in [2.0, 5.0, 10.0, 30.0, 60.0]:
        cells = []
        for ts in (0.0, 1.0):
            g = summ[(summ.tau_s_min == ts) & (summ.waveform == kind) & (summ.T_min == T)]
            cells.append(" / ".join(f(g[g.topology == t].strict_max_tau_f.iloc[0]) for t, _ in topo_order))
        g1 = summ[(summ.tau_s_min == 1.0) & (summ.waveform == kind) & (summ.T_min == T)]
        cells.append(" / ".join(f(g1[g1.topology == t].shape_max_tau_f.iloc[0]) for t, _ in topo_order))
        bc = summ[(summ.tau_s_min == 0.0) & (summ.waveform == kind) & (summ.T_min == T) & (summ.topology == "cstr")].binding_criterion.iloc[0]
        bc = "--" if not isinstance(bc, str) else bc.replace(" (at 0.05 min)", "")
        rows.append(f"{lab} & {T:g} & {cells[0]} & {cells[1]} & {cells[2]} & {bc} {BS}{BS}")
tab_env = "\n".join(rows)

bud = pd.read_csv(A / "B_volume_budget.csv")
brow = "\n".join(
    f"{r.waveform.capitalize()} & {r.T_min:g} & {r.tau_f_max_min:.2f} & {r['V_eff_max_nL_low (3)']:.2f} & {r['V_eff_max_nL_medium (50)']:.1f} & {r['V_eff_max_nL_high (1000)']:.0f} {BS}{BS}"
    for _, r in bud.iterrows())
well = pd.read_csv(A / "B_nyein_well_placement.csv")
oka = pd.read_csv(A / "C_okawara_exercise_flux.csv")
okrow = "\n".join(f"{r.site.capitalize()} & {r.phase.replace('-', '--')} & {r.flux_nL_min_cm2:.0f} & {r.tau_f_72nL_well_min:.2f} {BS}{BS}" for _, r in oka.iterrows())
c = pd.read_csv(A / "C_time_varying_flux.csv")
crow = "\n".join(
    f"{r.scenario.replace('(', '(').replace('to', 'to')} & {r.through_peak_delay_event1_min:.2f} / {r.through_peak_delay_event2_min:.2f} & {r.timing_error_event1_min:+.2f} / {r.timing_error_event2_min:+.2f} & {r.through_amp_retention_event1:.2f} & {r.fmc_window_min_median:.1f} ({r.fmc_window_min_max:.1f}) {BS}{BS}"
    for _, r in c.iterrows())
d = pd.read_csv(A / "D_fmc_chrono_empirical.csv")
drows = []
for cls in ["fill-measure-clear", "chronological window"]:
    for grp in ["Saha", "Choi"]:
        h = d[(d["class"] == cls) & (d.group == grp)]
        if cls == "fill-measure-clear":
            for per, hh in h.groupby("cycle_min"):
                cells = " & ".join(f"{int(x.n_strict_pass)}/{int(x.n_traces)}" for _, x in hh.sort_values("f_res").iterrows())
                drows.append(f"FMC & {grp} & {per:g} & {cells} {BS}{BS}")
        else:
            for _, x in h.iterrows():
                drows.append(f"Window & {grp} & {x.cycle_min:g} & \\multicolumn{{3}}{{c}}{{{int(x.n_strict_pass)}/{int(x.n_traces)}}} {BS}{BS}")
drow = "\n".join(drows)

SEC_CRIT = r"""
\section*{Timing criteria, sensor lag, and delay-aligned envelopes}

Table~\ref{tab:crit} extends the dense-grid envelope to $\tau_{\mathrm{s}}=0$ and reports, for each cell, the criterion that fails first above the boundary. The binding criterion is correlation for pulses, feature delay for ramps, and NRMSE for sinusoids. Two closed forms explain most boundaries. For a ramp, the $t_{50}$ delay is approximately $\tau_{\mathrm{f}}+\tau_{\mathrm{s}}$, so $\tau_{\mathrm{f,max}}\approx T/4-\tau_{\mathrm{s}}$ (13.6 min against 14 min predicted at $T=60$ min and $\tau_{\mathrm{s}}=1$ min). For a sinusoid, a pure phase shift $\varphi$ gives NRMSE $=\sin(\varphi/2)/\sqrt{2}$, so NRMSE $\leq0.10$ implies a delay of at most $0.045T$, which is stricter than the $T/4$ delay limit. A first-order sensor with $\tau_{\mathrm{s}}=1$ min alone introduces a phase lag $\arctan(2\pi\tau_{\mathrm{s}}/T)$ of 0.56 rad at $T=10$ min, above the 0.28 rad allowed. The absence of any passing condition for 2--10 min sinusoids at $\tau_{\mathrm{s}}=1$ min is therefore set by sensor lag; without sensor lag, transport alone allows 0.06--0.45 min.

The delay-aligned envelope removes the independently known feature delay before scoring amplitude, NRMSE, and correlation, and applies no delay limit. It increases the tolerable $\tau_{\mathrm{f}}$ 1.5- to 10-fold for well-mixed and gamma transport and removes the limit for plug flow within the tested range. Most of the strict envelope therefore reflects uncorrected timing error. That error is correctable only when the delay is stable; because $\tau_{\mathrm{f}}=V_{\mathrm{eff}}/Q$, it changes with sweat rate (see the time-varying-flux section).

\begin{table}[htbp]
\caption{Largest contiguous passing $\tau_{\mathrm{f}}$ (min) on the 71-value dense grid, listed as well mixed / gamma RTD ($k=3$) / plug flow. Strict: all four criteria. Aligned: amplitude, NRMSE, and correlation after removal of the known feature delay. Binding: criterion failing first above the well-mixed strict boundary at $\tau_{\mathrm{s}}=0$. A dash means no tested value passed; $\geq$15 means the whole tested range passed.}
\label{tab:crit}
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrcccl}
\toprule
\textbf{Input} & $T$ (min) & \textbf{Strict}, $\tau_{\mathrm{s}}=0$ & \textbf{Strict}, $\tau_{\mathrm{s}}=1$ min & \textbf{Aligned}, $\tau_{\mathrm{s}}=1$ min & \textbf{Binding} \\
\midrule
TABLE_ENV
\bottomrule
\end{tabular}
\end{table}
""".replace("TABLE_ENV", tab_env)

w = {r.flux: r for _, r in well.iterrows()}
SEC_BUDGET = r"""
\section*{Coupled-volume budget and reported sweat rates}

Inverting $\tau_{\mathrm{f}}=V_{\mathrm{eff}}/Q$ with $Q=q''A$ gives the largest coupled volume that meets an envelope, $V_{\mathrm{eff,max}}=\tau_{\mathrm{f,max}}\,q''A$. Table~\ref{tab:budget} evaluates it for the well-mixed envelope at $\tau_{\mathrm{s}}=1$ min, a 3 mm collection area ($A=0.0707$ cm$^2$), and the three areal sweat rates of the Nyein et al. simulations.\cite{si9} The 72 nL well of that study corresponds to $\tau_{\mathrm{f}}=$ TAU_HI, TAU_MED, and TAU_LO min at 1000, 50, and 3 nL min$^{-1}$ cm$^{-2}$. It meets the 30 and 60 min pulse, 10--60 min ramp, and 60 min sinusoid envelopes only at the highest rate. For comparison, the assumption-based 1.9 min exchange time of Hauke et al.\cite{si6} lies inside the 30 min pulse envelope but outside the 10 min one.

Reported areal sweat rates span about three orders of magnitude. Passive fingertip secretion is approximately 3--10 nL min$^{-1}$ cm$^{-2}$.\cite{si3} During constant-load exercise at 25\% of peak power (17 participants for the upper arm and 15 for the forehead), local sweat rate passed through two inflection points, about 4--5 min and a further 11--13 min after sweating onset, with the phase means in Table~\ref{tab:okawara}.\cite{si7} At these exercise rates the same 72 nL well would be renewed within 0.4--2.8 min, so for slow features transport is rarely limiting during exercise. Within one bout, however, the forehead rate rose 3.6-fold, and a well-mixed delay calibrated in one phase would not hold in the next.

\begin{table}[htbp]
\caption{Largest coupled volume $V_{\mathrm{eff,max}}$ (nL) meeting the strict well-mixed envelope at $\tau_{\mathrm{s}}=1$ min over a 3 mm collection area.}
\label{tab:budget}
\centering
\scriptsize
\begin{tabular}{lrrrrr}
\toprule
\textbf{Input} & $T$ (min) & $\tau_{\mathrm{f,max}}$ (min) & \shortstack{3 nL min$^{-1}$ cm$^{-2}$} & \shortstack{50 nL min$^{-1}$ cm$^{-2}$} & \shortstack{1000 nL min$^{-1}$ cm$^{-2}$} \\
\midrule
TABLE_BUD
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\caption{Local sweat-rate phases during constant-load exercise (phase means from Okawara et al.;\cite{si7} mg cm$^{-2}$ min$^{-1}$ converted to nL min$^{-1}$ cm$^{-2}$ at 1 mg = 1 $\mu$L) and the resulting mean residence time of a 72 nL well-mixed zone over a 3 mm area.}
\label{tab:okawara}
\centering
\scriptsize
\begin{tabular}{llrr}
\toprule
\textbf{Site} & \textbf{Phase} & \textbf{Rate} (nL min$^{-1}$ cm$^{-2}$) & $\tau_{\mathrm{f}}$ (min) \\
\midrule
TABLE_OKA
\bottomrule
\end{tabular}
\end{table}
""".replace("TABLE_BUD", brow).replace("TABLE_OKA", okrow).replace(
    "TAU_HI", f"{w['high (1000)'].tau_f_min:.2f}").replace(
    "TAU_MED", f"{w['medium (50)'].tau_f_min:.1f}").replace(
    "TAU_LO", f"{w['low (3)'].tau_f_min:.0f}")

SEC_QT = r"""
\section*{Time-varying sweat flux}

All envelopes above assume steady flow. To test this, a 72 nL well-mixed zone over a 3 mm area was driven by the same mean flux, 50 nL min$^{-1}$ cm$^{-2}$, in three ways: steady; 10 min bursts alternating between 90 and 10 nL min$^{-1}$ cm$^{-2}$; and a linear decline from 95 to 5 nL min$^{-1}$ cm$^{-2}$ over 300 min. The input carried two 30 min pulses centered at 80 and 200 min. The well-mixed zone was integrated exactly for $dC_{\mathrm{f}}/dt=[Q(t)/V_{\mathrm{eff}}](C_{\mathrm{in}}-C_{\mathrm{f}})$. A fill--measure--clear chamber of the same volume was filled at $Q(t)$ and read when full, with residual fraction 0.15. Table~\ref{tab:qt} compares both with a delay calibrated under steady flux.

Bursts shorter than the residence time were averaged by the well and shifted the peak delay by no more than 0.5 min. A slow decline shifted it by $-2.2$ and $+1.9$ min for the two events and raised amplitude retention from 0.66 to 0.76, so a single calibration misassigns timing as the flux drifts. Under the same decline, fill--measure--clear windows lengthened from 11 to 38 min. Each window is bounded by its own fill events, however, so the sample age of every readout is recorded rather than inferred.

\begin{table}[htbp]
\caption{Steady versus time-varying flux at the same mean (50 nL min$^{-1}$ cm$^{-2}$). Delays are peak delays of the well-mixed zone for the two events; timing error is relative to the steady-flux calibration. FMC window: median (maximum) fill time of the fill--measure--clear chamber.}
\label{tab:qt}
\centering
\scriptsize
\begin{tabular}{lcccc}
\toprule
\textbf{Flux scenario} & \textbf{Delay} (min) & \textbf{Timing error} (min) & \textbf{Amplitude retention} & \textbf{FMC window} (min) \\
\midrule
TABLE_QT
\bottomrule
\end{tabular}
\end{table}
""".replace("TABLE_QT", crow)

SEC_FMC_RES = r"""The fill--measure--clear and chronological runs use the models defined under Topology definitions. Table~\ref{tab:fmc} reports how many traces in each group pass the strict criterion. Fill--measure--clear preserved every Saha trace up to 5 min cycles at all residual fractions, but the faster Choi traces only up to 1--2 min. Chronological windows preserved the Saha traces up to 2 min (and three of four at 5 min) and the Choi traces up to 1 min. Because the two source traces do not constrain fill, clearance, or window parameters independently, these results describe the sampling requirement set by each waveform, not the performance of any device.

\begin{table}[htbp]
\caption{Strict-criterion passes (passing traces / traces) for fill--measure--clear (FMC) cycles at residual fractions 0, 0.1, and 0.25, and for chronological windows, in the empirical stress test.}
\label{tab:fmc}
\centering
\scriptsize
\begin{tabular}{llrccc}
\toprule
\textbf{Class} & \textbf{Traces} & \textbf{Cycle or window} (min) & $f_{\mathrm{res}}=0$ & 0.1 & 0.25 \\
\midrule
TABLE_FMC
\bottomrule
\end{tabular}
\end{table}
""".replace("TABLE_FMC", drow)

EQ_FMC = r"""\textbf{Fill--measure--clear or reset.} A discrete volume is filled, measured, and displaced or cleared. Required quantities are fill-time distribution, measurement window, clearance time, and residual old-sample fraction. This physical operation is distinct from electronic duty cycling. The $n$th readout is modeled as $C_n=f_{\mathrm{res}}C_{n-1}+(1-f_{\mathrm{res}})\bar{C}_{\mathrm{in},n}$, where $\bar{C}_{\mathrm{in},n}$ is the flow-weighted mean input over the $n$th fill window, so a sample of age $k$ cycles carries weight $(1-f_{\mathrm{res}})f_{\mathrm{res}}^{k}$.

\textbf{Chronological reservoirs.} Successive chambers isolate nominal collection windows. Each value is a windowed sample whose age distribution depends on chamber fill, transition, and isolation. It should not be labelled an instantaneous continuous measurement. The value of window $k$ is modeled as $C_k=\int_{W_k}Q(t)C_{\mathrm{in}}(t)\,dt\big/\int_{W_k}Q(t)\,dt$, which reduces to the time average for constant flow."""

R = [
    (r"\title{Supporting Information\\[0.5em]\large From Sensitivity to Information Fidelity in Wearable Sweat Sensing}",
     r"\title{Supporting Information\\[0.5em]\large From Sensitivity to Temporal Fidelity in Wearable Sweat Sensing}"),
    (r"""\textbf{Fill--measure--clear or reset.} A discrete volume is filled, measured, and displaced or cleared. Required quantities are fill-time distribution, measurement window, clearance time, and residual old-sample fraction. This physical operation is distinct from electronic duty cycling.

\textbf{Chronological reservoirs.} Successive chambers isolate nominal collection windows. Each value is a windowed sample whose age distribution depends on chamber fill, transition, and isolation. It should not be labelled an instantaneous continuous measurement.""", EQ_FMC),
    ("The fill--measure--clear and chronological runs are included in the released grid and are not analysed in the main text, because the two source traces do not constrain fill, clearance, or window parameters independently.",
     "\n\n" + SEC_FMC_RES),
    (r"\section*{Worked decision-to-tolerance example}", SEC_CRIT + "\n" + r"\section*{Worked decision-to-tolerance example}"),
    ("The close numerical agreement is expected when the simulated well behaves approximately as a well-mixed volume and both descriptions use the same geometry, flow, and replacement definition.",
     "The close numerical agreement is expected when the simulated well behaves approximately as a well-mixed volume and both descriptions use the same geometry, flow, and replacement definition. The check concerns the 90\\% replacement time only, not the shape of the residence-time distribution, and the largest difference (1.97\\%) lies within the rounding of the published 2.3 min."),
    (r"\section*{Numerical implementation and checks}", SEC_BUDGET + "\n" + SEC_QT + "\n" + r"\section*{Numerical implementation and checks}"),
    (r"and \path{analysis_bundle.json}.",
     r"and \path{analysis_bundle.json}. Second-round analyses: \path{run_round2_analyses.py}, producing the dense envelopes at $\tau_{\mathrm{s}}=0$ and 1 min with binding criteria and delay-aligned scores, the coupled-volume budget, the time-varying-flux comparison, and the fill--measure--clear and chronological summaries."),
]
for a, b in R:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
DST.write_text(s, encoding="utf-8", newline="")
print("wrote", DST)
