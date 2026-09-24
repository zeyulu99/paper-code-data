"""Export the selected placement-ready Figure 2 subplot assets.

The output remains unassembled so that the author can arrange the panels in
PowerPoint. Every plotted line has a machine-readable CSV counterpart.
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
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter


PROJECT = Path(r"E:\Projects_im\6.sweat_sensor_perspective")
SOURCE_ASSETS = PROJECT / "outputs" / "figures_final" / "figure2_individual_subplots" / "data"
CANDIDATES = PROJECT / "outputs" / "figure_working" / "figure2_strategy_comparison_2026-09-09"
EMPIRICAL = PROJECT / "outputs" / "revision_2026-08-31_figure2_data_package" / "05_empirical_waveforms_model_inputs.csv"
OUTPUT = PROJECT / "outputs" / "figures_final" / "figure2_placement_ready"

BLACK = "#151515"
GRAY = "#8C8C8C"
BLUE = "#245DB8"
PURPLE = "#6B2AA6"
ORANGE = "#E87522"
TEAL = "#138A8A"
RED = "#C73A3A"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 1.1,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def clean_axes(ax: plt.Axes, ticks: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.grid(False)


def save_all(fig: plt.Figure, stem: str) -> dict[str, str]:
    png_path = OUTPUT / "png" / f"{stem}.png"
    svg_path = OUTPUT / "svg" / f"{stem}.svg"
    pdf_path = OUTPUT / "pdf" / f"{stem}.pdf"
    tiff_path = OUTPUT / "tiff" / f"{stem}.tiff"
    for path in [png_path, svg_path, pdf_path, tiff_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight", pad_inches=0.04, pil_kwargs={"compression": "tiff_lzw"})
    paths = {"png": str(png_path), "svg": str(svg_path), "pdf": str(pdf_path), "tiff": str(tiff_path)}
    plt.close(fig)
    return paths


def export_a() -> list[dict]:
    inventory = []
    for waveform in ["pulse", "ramp", "sinusoid"]:
        source = SOURCE_ASSETS / f"fig2a_{waveform}_input.csv"
        rows = read_csv(source)
        x = np.asarray([float(row["time_min"]) for row in rows])
        y = np.asarray([float(row["C_in_normalized"]) for row in rows])
        fig, ax = plt.subplots(figsize=(2.3, 1.65))
        ax.plot(x, y, color=BLACK, linewidth=2.2, solid_capstyle="round")
        clean_axes(ax)
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_ylim(-0.05, 1.08)
        stem = f"fig2a_{waveform}"
        files = save_all(fig, stem)
        data_path = OUTPUT / "data" / f"{stem}.csv"
        write_csv(data_path, rows)
        inventory.append({"stem": stem, "role": f"Canonical {waveform} input", "files": files, "data": str(data_path)})
    return inventory


def export_b() -> list[dict]:
    inventory = []
    mapping = {"plug": "plug_flow", "cstr": "well_mixed", "gamma_rtd": "distributed_RTD"}
    for source_topology, display_topology in mapping.items():
        source = CANDIDATES / "data" / f"candidate_b_T10_{source_topology}.csv"
        rows = read_csv(source)
        x = np.asarray([float(row["time_min"]) for row in rows])
        inlet = np.asarray([float(row["C_in"]) for row in rows])
        fluid = np.asarray([float(row["C_f"]) for row in rows])
        sensor = np.asarray([float(row["S"]) for row in rows])
        fig, ax = plt.subplots(figsize=(2.55, 2.75))
        ax.plot(x, inlet + 2.4, color=BLACK, linewidth=2.2)
        ax.plot(x, fluid + 1.2, color=BLUE, linewidth=2.2)
        ax.plot(x, sensor, color=PURPLE, linewidth=2.2)
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_ylim(-0.08, 3.5)
        ax.axis("off")
        stem = f"fig2b_{display_topology}_T10"
        files = save_all(fig, stem)
        data_path = OUTPUT / "data" / f"{stem}.csv"
        write_csv(data_path, rows)
        inventory.append(
            {
                "stem": stem,
                "role": f"Pulse propagation through {display_topology}",
                "condition": "T=10 min; tau_transport=5 min; tau_sensor=1 min; gamma k=3 where applicable",
                "files": files,
                "data": str(data_path),
            }
        )
    return inventory


def envelope(rows: list[dict[str, str]], topology: str, waveform: str) -> list[float | None]:
    values = []
    for timescale in [2.0, 5.0, 10.0, 30.0, 60.0]:
        passing = [
            float(row["tau_transport_min"])
            for row in rows
            if row["topology"] == topology
            and row["waveform_type"] == waveform
            and float(row["T_min"]) == timescale
            and row["pass_strict"] == "PASS"
        ]
        values.append(max(passing) if passing else None)
    return values


def log_tick(value: float, _position: int) -> str:
    return f"{value:g}" if value >= 1 else f"{value:.2g}"


def export_c() -> list[dict]:
    dense_source = CANDIDATES / "data" / "canonical_dense_boundary_grid_tau_sensor_1min.csv"
    rows = read_csv(dense_source)
    inventory = []
    styles = {"plug": (BLACK, "o"), "cstr": (BLUE, "s"), "gamma_rtd": (ORANGE, "^")}
    offsets = {"plug": -0.08, "cstr": 0.0, "gamma_rtd": 0.08}
    x = np.arange(5, dtype=float)
    for waveform in ["pulse", "ramp", "sinusoid"]:
        fig, ax = plt.subplots(figsize=(3.0, 2.55))
        data_rows = []
        for topology in ["plug", "cstr", "gamma_rtd"]:
            color, marker = styles[topology]
            values = envelope(rows, topology, waveform)
            y = np.asarray([np.nan if value is None else value for value in values])
            ax.plot(x + offsets[topology], y, color=color, marker=marker, linewidth=1.4, markersize=5.4)
            missing = np.asarray([value is None for value in values])
            if np.any(missing):
                ax.scatter(
                    x[missing] + offsets[topology],
                    np.full(np.sum(missing), 0.034),
                    marker="x",
                    color=color,
                    s=38,
                    linewidths=1.4,
                    clip_on=False,
                )
            for timescale, value in zip([2, 5, 10, 30, 60], values):
                data_rows.append(
                    {
                        "waveform_type": waveform,
                        "topology": topology,
                        "T_min": timescale,
                        "largest_tested_passing_tau_transport_min": "" if value is None else f"{value:.8g}",
                        "status": "NO_TESTED_PASS" if value is None else "PASSING_BOUNDARY",
                        "tau_sensor_min": 1,
                    }
                )
        positive_boundaries = [
            float(row["largest_tested_passing_tau_transport_min"])
            for row in data_rows
            if row["largest_tested_passing_tau_transport_min"] != ""
        ]
        if np.any(np.asarray(positive_boundaries) <= 0):
            raise ValueError("Log-scale operating-envelope boundaries must be positive")
        ax.set_yscale("log")
        ax.set_xlim(-0.45, 4.45)
        ax.set_ylim(0.025, 18.0)
        ax.set_xticks(x, ["2", "5", "10", "30", "60"])
        ticks = [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(log_tick))
        ax.tick_params(which="minor", left=False, bottom=False)
        clean_axes(ax, ticks=True)
        ax.set_xlabel(r"Feature timescale, $T$ (min)")
        ax.set_ylabel(r"Largest passing $\tau_{\mathrm{transport}}$ (min)")
        stem = f"fig2c_{waveform}_dense_topology_comparison"
        files = save_all(fig, stem)
        data_path = OUTPUT / "data" / f"{stem}.csv"
        write_csv(data_path, data_rows)
        inventory.append(
            {
                "stem": stem,
                "role": f"Dense operating-envelope comparison for {waveform}",
                "condition": "71 transport-time values; tau_sensor=1 min; strict joint criterion",
                "files": files,
                "data": str(data_path),
            }
        )
    return inventory


def empirical_trace(trace_id: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [row for row in read_csv(EMPIRICAL) if row["trace_id"] == trace_id]
    rows.sort(key=lambda row: float(row["time_min"]))
    return (
        np.asarray([float(row["time_min"]) for row in rows]),
        np.asarray([float(row["signal"]) for row in rows]),
    )


def normalize_to_source(source: np.ndarray, values: np.ndarray) -> np.ndarray:
    return (values - float(np.min(source))) / float(np.ptp(source))


def export_d() -> list[dict]:
    inventory = []
    settings = {
        "saha": {
            "trace_id": "Saha_2024_Fig3b_Subject1",
            "values": [0.5, 3.7, 5.0, 10.0],
            "boundary": 3.7,
        },
        "choi": {
            "trace_id": "Choi_2020_C",
            "values": [0.05, 1.0, 2.0, 5.0],
            "boundary": 1.0,
        },
    }
    colors = [TEAL, BLUE, ORANGE, RED]
    for source_name, setting in settings.items():
        trace_id = setting["trace_id"]
        time, source_values = empirical_trace(trace_id)
        normalized_source = normalize_to_source(source_values, source_values)

        fig, ax = plt.subplots(figsize=(3.1, 2.0))
        ax.plot(time, normalized_source, color=BLACK, linewidth=2.1)
        clean_axes(ax)
        ax.set_xlim(float(time[0]), float(time[-1]))
        ax.set_ylim(-0.08, 1.08)
        stem = f"fig2d_{source_name}_representative_original"
        files = save_all(fig, stem)
        data_path = OUTPUT / "data" / f"{stem}.csv"
        write_csv(
            data_path,
            [
                {"trace_id": trace_id, "time_min": f"{x:.6f}", "signal": f"{raw:.10g}", "signal_normalized": f"{norm:.10g}"}
                for x, raw, norm in zip(time, source_values, normalized_source)
            ],
        )
        inventory.append({"stem": stem, "role": f"Representative original {source_name} proxy", "files": files, "data": str(data_path)})

        candidate_path = CANDIDATES / "data" / f"candidate_d_{source_name}_same_trace_transport_series.csv"
        modeled_rows = read_csv(candidate_path)
        grouped: dict[float, list[dict[str, str]]] = defaultdict(list)
        for row in modeled_rows:
            grouped[float(row["tau_transport_min"])].append(row)
        fig, ax = plt.subplots(figsize=(3.25, 2.0))
        ax.plot(time, normalized_source, color=GRAY, linewidth=1.6, linestyle=(0, (4, 3)), zorder=1)
        for color, transport_min in zip(colors, setting["values"]):
            group = sorted(grouped[transport_min], key=lambda row: float(row["time_min"]))
            modeled_time = np.asarray([float(row["time_min"]) for row in group])
            modeled = np.asarray([float(row["modeled_normalized_to_source"]) for row in group])
            ax.plot(modeled_time, modeled, color=color, linewidth=1.8, zorder=2)
        clean_axes(ax)
        ax.set_xlim(float(time[0]), float(time[-1]))
        ax.set_ylim(-0.08, 1.08)
        stem = f"fig2d_{source_name}_transport_series"
        files = save_all(fig, stem)
        data_path = OUTPUT / "data" / f"{stem}.csv"
        write_csv(data_path, modeled_rows)
        inventory.append(
            {
                "stem": stem,
                "role": f"Same {source_name} trace across passing and failing well-mixed transport conditions",
                "condition": f"strict group boundary={setting['boundary']:g} min; tau_sensor=0",
                "files": files,
                "data": str(data_path),
            }
        )
    return inventory


def export_legends() -> list[dict]:
    inventory = []
    legend_specs = {
        "fig2c_topology_legend": [
            (BLACK, "o", "Plug flow", "-"),
            (BLUE, "s", "Well mixed", "-"),
            (ORANGE, "^", "Distributed RTD", "-"),
        ],
        "fig2d_saha_transport_legend": [
            (GRAY, None, "Original", "--"),
            (TEAL, None, "0.5 min (pass)", "-"),
            (BLUE, None, "3.7 min (boundary)", "-"),
            (ORANGE, None, "5 min (fail)", "-"),
            (RED, None, "10 min (fail)", "-"),
        ],
        "fig2d_choi_transport_legend": [
            (GRAY, None, "Original", "--"),
            (TEAL, None, "0.05 min (pass)", "-"),
            (BLUE, None, "1 min (boundary)", "-"),
            (ORANGE, None, "2 min (fail)", "-"),
            (RED, None, "5 min (fail)", "-"),
        ],
    }
    for stem, specs in legend_specs.items():
        fig, ax = plt.subplots(figsize=(2.3, 1.05 if "topology" in stem else 1.45))
        handles = [
            Line2D([0], [0], color=color, marker=marker, linestyle=linestyle, linewidth=1.7, markersize=5.4, label=label)
            for color, marker, label, linestyle in specs
        ]
        ax.legend(handles=handles, loc="center left", frameon=False, handlelength=2.3, borderaxespad=0, labelspacing=0.45)
        ax.axis("off")
        files = save_all(fig, stem)
        inventory.append({"stem": stem, "role": "Standalone assembly legend", "files": files})
    return inventory


def main() -> None:
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inventory = []
    inventory.extend(export_a())
    inventory.extend(export_b())
    inventory.extend(export_c())
    inventory.extend(export_d())
    inventory.extend(export_legends())
    manifest = {
        "figure": "Figure 2 placement-ready independent assets",
        "paper_type": "Perspective",
        "assembled": False,
        "selected_strategy": {
            "panel_a": "Exact canonical model definitions",
            "panel_b": "T=10 min representative pulse for visible topology contrast",
            "panel_c": "Dense recalculated boundary grid, faceted by waveform to expose topology overlap and differences",
            "panel_d": "One declared representative trace per source; original, passing boundary, first failure, and stronger failure on a common normalization",
        },
        "inventory": inventory,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme = """# Figure 2 placement-ready assets

This directory contains the selected independent assets after strategy comparison. No assembled manuscript figure is created.

- `png/`, `svg/`, `pdf/`, and `tiff/` contain matching exports.
- `data/` contains the plotted values for each scientific subplot.
- Panel b uses T=10 min, tau_transport=5 min, and tau_sensor=1 min.
- Panel c uses 71 recalculated transport-time values from 0.05 to 15 min at tau_sensor=1 min and is faceted by waveform.
- Panel d uses Saha Subject 1 and Choi panel C as declared representative display traces. Group boundaries remain based on all four Saha and all three Choi traces.
- Standalone legend assets are supplied for PowerPoint assembly.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(OUTPUT), "assets": len(inventory)}, indent=2))


if __name__ == "__main__":
    main()
