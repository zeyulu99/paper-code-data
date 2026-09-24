"""Digitize public waveform panels used in the sweat-sensor Perspective revision.

The script operates on 300-dpi PNG renderings of the public PDFs.  It keeps the
figure-derived traces separate from directly reported scalar values and records
the coordinate calibration and pixel-scale uncertainty for every output row.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(r"E:\Projects_im\6.sweat_sensor_perspective")
IMAGE_DIR = PROJECT / "tmp" / "pdfs" / "digitization_stage"
OUTPUT_DIR = PROJECT / "outputs" / "revision_2026-08-06" / "waveform_digitization"
DOC_PATH = PROJECT / "docs" / "revision_2026-08-06" / "public_figure_digitization_method.md"

SAHA_IMAGE = IMAGE_DIR / "saha-07.png"
CHOI_IMAGE = IMAGE_DIR / "choi-02.png"

SAHA_URL = "https://doi.org/10.1002/advs.202405518"
CHOI_URL = "https://doi.org/10.1038/s41598-020-64406-5"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def map_x(pixel: float, x_left: float, x_right: float, x_min: float, x_max: float) -> float:
    return x_min + (pixel - x_left) * (x_max - x_min) / (x_right - x_left)


def map_y(pixel: float, y_top: float, y_bottom: float, y_min: float, y_max: float) -> float:
    return y_max - (pixel - y_top) * (y_max - y_min) / (y_bottom - y_top)


def pixel_x(value: float, x_left: float, x_right: float, x_min: float, x_max: float) -> float:
    return x_left + (value - x_min) * (x_right - x_left) / (x_max - x_min)


def pixel_y(value: float, y_top: float, y_bottom: float, y_min: float, y_max: float) -> float:
    return y_top + (y_max - value) * (y_bottom - y_top) / (y_max - y_min)


def cluster_centers(values: np.ndarray, max_gap: int = 2) -> list[float]:
    if values.size == 0:
        return []
    values = np.unique(np.sort(values.astype(int)))
    clusters: list[list[int]] = [[int(values[0])]]
    for value in values[1:]:
        if int(value) - clusters[-1][-1] <= max_gap:
            clusters[-1].append(int(value))
        else:
            clusters.append([int(value)])
    return [float(np.median(cluster)) for cluster in clusters if len(cluster) <= 18]


def interpolate_short_gaps(values: list[float | None], max_gap: int) -> tuple[list[float | None], list[bool]]:
    result = list(values)
    interpolated = [False] * len(values)
    i = 0
    while i < len(result):
        if result[i] is not None:
            i += 1
            continue
        start = i
        while i < len(result) and result[i] is None:
            i += 1
        end = i - 1
        gap = end - start + 1
        if start > 0 and i < len(result) and gap <= max_gap:
            left = float(result[start - 1])
            right = float(result[i])
            for k in range(gap):
                fraction = (k + 1) / (gap + 1)
                result[start + k] = left + fraction * (right - left)
                interpolated[start + k] = True
    return result, interpolated


def trace_black_curve(
    rgb: np.ndarray,
    bounds: tuple[int, int, int, int],
    axis: tuple[float, float, float, float],
    expected_start: float,
    times: list[float],
) -> tuple[list[float | None], list[bool]]:
    x_left, x_right, y_top, y_bottom = bounds
    x_min, x_max, y_min, y_max = axis
    mask = np.all(rgb < 120, axis=2)
    # Remove borders, ticks, and the bold subject label in the upper-right corner.
    mask[: y_top + 7, :] = False
    mask[y_bottom - 6 :, :] = False
    mask[:, : x_left + 7] = False
    mask[:, x_right - 6 :] = False
    mask[y_top : min(y_top + 68, y_bottom), x_left + 245 : x_right] = False

    output: list[float | None] = []
    previous_y = pixel_y(expected_start, y_top, y_bottom, y_min, y_max)
    for time in times:
        x = int(round(pixel_x(time, x_left, x_right, x_min, x_max)))
        candidates: list[float] = []
        for half_width in (2, 4, 7):
            ys, _ = np.where(mask[y_top + 7 : y_bottom - 6, x - half_width : x + half_width + 1])
            candidates = cluster_centers(ys + y_top + 7)
            if candidates:
                break
        if not candidates:
            output.append(None)
            continue
        choice = min(candidates, key=lambda value: abs(value - previous_y))
        if abs(choice - previous_y) > 48:
            output.append(None)
            continue
        output.append(map_y(choice, y_top, y_bottom, y_min, y_max))
        previous_y = choice
    return interpolate_short_gaps(output, max_gap=3)


def find_red_markers(
    rgb: np.ndarray,
    bounds: tuple[int, int, int, int],
    axis: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    x_left, x_right, y_top, y_bottom = bounds
    x_min, x_max, y_min, y_max = axis
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    red = (r > 175) & (r - g > 45) & (r - b > 35) & (g < 180)
    red[: y_top + 6, :] = False
    red[y_bottom - 6 :, :] = False
    red[:, : x_left + 6] = False
    red[:, x_right - 8 :] = False

    column_counts = red[y_top:y_bottom, x_left:x_right].sum(axis=0)
    peak_columns = np.where(column_counts >= 8)[0] + x_left
    groups: list[list[int]] = []
    for x in peak_columns:
        if not groups or x - groups[-1][-1] > 8:
            groups.append([int(x)])
        else:
            groups[-1].append(int(x))

    markers: list[tuple[float, float, float, float]] = []
    for group in groups:
        center_x = int(round(float(np.median(group))))
        x0, x1 = max(x_left + 6, center_x - 8), min(x_right - 8, center_x + 9)
        ys, xs = np.where(red[y_top + 6 : y_bottom - 6, x0:x1])
        if len(ys) < 15:
            continue
        absolute_x = xs + x0
        absolute_y = ys + y_top + 6
        # Marker rings produce a compact two-dimensional cluster; reject long dashes.
        if absolute_y.max() - absolute_y.min() < 7:
            continue
        px = float(np.median(absolute_x))
        py = float(np.median(absolute_y))
        markers.append((map_x(px, x_left, x_right, x_min, x_max), map_y(py, y_top, y_bottom, y_min, y_max), px, py))

    # Consolidate any double detections around one circular marker.
    consolidated: list[tuple[float, float, float, float]] = []
    for marker in sorted(markers):
        if consolidated and marker[0] - consolidated[-1][0] < 3.0:
            previous = consolidated.pop()
            consolidated.append(tuple((a + b) / 2 for a, b in zip(previous, marker)))
        else:
            consolidated.append(marker)
    return consolidated


def trace_colored_curve(
    rgb: np.ndarray,
    bounds: tuple[int, int, int, int],
    axis: tuple[float, float, float, float],
    color: str,
    times: list[float],
) -> tuple[list[float | None], list[bool]]:
    x_left, x_right, y_top, y_bottom = bounds
    x_min, x_max, y_min, y_max = axis
    r, g, b = [rgb[:, :, index].astype(int) for index in range(3)]
    if color == "red":
        mask = (r > 170) & (r - g > 45) & (r - b > 35)
    elif color == "blue":
        mask = (b > 145) & (b - r > 45) & (b - g > 20)
    elif color == "green":
        mask = (g > 85) & (g - r > 25) & (g - b > 15)
    else:
        raise ValueError(color)
    mask[: y_top + 2, :] = False
    mask[y_bottom - 2 :, :] = False
    mask[:, : x_left + 2] = False
    mask[:, x_right - 2 :] = False

    values: list[float | None] = []
    for time in times:
        x = int(round(pixel_x(time, x_left, x_right, x_min, x_max)))
        ys, _ = np.where(mask[y_top + 2 : y_bottom - 2, x - 2 : x + 3])
        if len(ys) == 0:
            values.append(None)
            continue
        py = float(np.median(ys + y_top + 2))
        values.append(map_y(py, y_top, y_bottom, y_min, y_max))
    return interpolate_short_gaps(values, max_gap=4)


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def linear_interpolate(times: list[float], values: list[float | None], query: float) -> float | None:
    pairs = [(t, v) for t, v in zip(times, values) if v is not None]
    if not pairs or query < pairs[0][0] or query > pairs[-1][0]:
        return None
    for (t0, v0), (t1, v1) in zip(pairs[:-1], pairs[1:]):
        if t0 <= query <= t1:
            if t1 == t0:
                return float(v0)
            return float(v0) + (query - t0) * (float(v1) - float(v0)) / (t1 - t0)
    return float(pairs[-1][1])


def first_half_crossing(times: list[float], values: list[float | None], step_time: float, before: float, after: float) -> float | None:
    target = (before + after) / 2
    increasing = after > before
    for t0, t1, v0, v1 in zip(times[:-1], times[1:], values[:-1], values[1:]):
        if t1 < step_time or v0 is None or v1 is None:
            continue
        crossed = (v0 <= target <= v1) if increasing else (v0 >= target >= v1)
        if crossed and v1 != v0:
            return t0 + (target - v0) * (t1 - t0) / (v1 - v0)
    return None


def median_window(times: list[float], values: list[float | None], start: float, end: float) -> float:
    selected = [float(v) for t, v in zip(times, values) if start <= t <= end and v is not None]
    return float(np.median(selected))


def protocol_phase(panel: str, time: float) -> tuple[str, float | None, bool]:
    if time < 15:
        return "55 W warm-up / sensor stabilization", 55, False
    if panel == "B":
        if time < 35:
            return "100 W plateau", 100, True
        if time < 55:
            return "200 W plateau", 200, True
        return "55 W recovery", 55, True
    if panel == "C":
        if time < 35:
            return "200 W plateau", 200, True
        if time < 55:
            return "100 W plateau", 100, True
        return "55 W recovery", 55, True
    if panel == "D":
        if time < 25:
            return "100 W step", 100, True
        if time < 35:
            return "125 W step", 125, True
        if time < 45:
            return "150 W step", 150, True
        if time < 55:
            return "175 W step", 175, True
        if time < 65:
            return "200 W step", 200, True
        return "200 W late step", 200, True
    raise ValueError(panel)


def draw_overlay(
    image: Image.Image,
    panels: list[dict],
    rows: list[dict],
    output: Path,
    source_key: str,
) -> None:
    boxes = [panel["bounds"] for panel in panels]
    left = min(box[0] for box in boxes) - 35
    top = min(box[2] for box in boxes) - 35
    right = max(box[1] for box in boxes) + 35
    bottom = max(box[3] for box in boxes) + 35
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    draw = ImageDraw.Draw(crop)
    panel_map = {panel["panel"]: panel for panel in panels}
    for row in rows:
        if row.get("source") != source_key or row.get("value") in (None, ""):
            continue
        panel = panel_map[row["panel"]]
        x_left, x_right, y_top, y_bottom = panel["bounds"]
        x_min, x_max, y_min, y_max = panel["axis"]
        px = pixel_x(float(row["time_min"]), x_left, x_right, x_min, x_max) - left
        py = pixel_y(float(row["value"]), y_top, y_bottom, y_min, y_max) - top
        color = "#00A6D6" if row.get("signal_type") in ("SBG", "sweat_chloride") else "#FFD400"
        radius = 2 if row.get("signal_type") != "BG_reference" else 4
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=color, width=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saha_pil = Image.open(SAHA_IMAGE).convert("RGB")
    choi_pil = Image.open(CHOI_IMAGE).convert("RGB")
    saha_rgb = np.asarray(saha_pil)
    choi_rgb = np.asarray(choi_pil)

    saha_panels = [
        {"panel": "Fig3b_Subject1", "subject": "Subject 1", "bounds": (1222, 1696, 373, 654), "axis": (-15.0, 130.0, 90.0, 150.0), "expected_start": 97.0, "reported_pr": 0.94},
        {"panel": "Fig3b_Subject2", "subject": "Subject 2", "bounds": (1223, 1707, 771, 1068), "axis": (-15.0, 130.0, 90.0, 150.0), "expected_start": 98.0, "reported_pr": 0.85},
        {"panel": "Fig3b_Subject3", "subject": "Subject 3", "bounds": (1214, 1706, 1167, 1454), "axis": (-15.0, 130.0, 75.0, 150.0), "expected_start": 97.0, "reported_pr": 0.92},
        {"panel": "Fig3b_Subject4", "subject": "Subject 4", "bounds": (1212, 1700, 1575, 1850), "axis": (-15.0, 130.0, 75.0, 180.0), "expected_start": 105.0, "reported_pr": 0.80},
    ]
    choi_panels = [
        {"panel": "B", "bounds": (790, 1197, 1003, 1410), "axis": (0.0, 70.0, 0.0, 30.0), "color": "red", "y_uncertainty": 0.5},
        {"panel": "C", "bounds": (1339, 1746, 1009, 1415), "axis": (0.0, 70.0, 0.0, 50.0), "color": "blue", "y_uncertainty": 0.8},
        {"panel": "D", "bounds": (1887, 2293, 1011, 1418), "axis": (0.0, 70.0, 0.0, 60.0), "color": "green", "y_uncertainty": 1.0},
    ]

    waveform_rows: list[dict] = []
    saha_times = [float(value) for value in range(-14, 131, 2)]
    for panel in saha_panels:
        values, interpolated = trace_black_curve(
            saha_rgb, panel["bounds"], panel["axis"], panel["expected_start"], saha_times
        )
        x_left, x_right, y_top, y_bottom = panel["bounds"]
        x_min, x_max, y_min, y_max = panel["axis"]
        x_uncertainty = 2.0 * (x_max - x_min) / (x_right - x_left)
        y_uncertainty = 3.0 * (y_max - y_min) / (y_bottom - y_top)
        for time, value, was_interpolated in zip(saha_times, values, interpolated):
            if value is None:
                continue
            waveform_rows.append({
                "source": "Saha 2024",
                "doi": "10.1002/advs.202405518",
                "figure": "Figure 3b(i)",
                "panel": panel["panel"],
                "subject": panel["subject"],
                "signal_type": "SBG",
                "time_min": round(time, 2),
                "value": round(float(value), 2),
                "unit": "mg/dL",
                "protocol_phase": "post-meal" if time >= 0 else "fasting baseline",
                "exercise_load_W": "",
                "valid_for_model": "YES" if time >= 0 else "BASELINE_ONLY",
                "interpolated_short_gap": "YES" if was_interpolated else "NO",
                "x_uncertainty": round(x_uncertainty, 3),
                "x_uncertainty_unit": "min",
                "y_uncertainty": round(y_uncertainty, 3),
                "y_uncertainty_unit": "mg/dL",
                "evidence_level": "PUBLIC_FIGURE_DIGITIZED",
                "source_url": SAHA_URL,
                "notes": "No smoothing; black SBG line sampled every 2 min from the 300-dpi PDF rendering.",
            })
        for time, value, _, _ in find_red_markers(saha_rgb, panel["bounds"], panel["axis"]):
            waveform_rows.append({
                "source": "Saha 2024",
                "doi": "10.1002/advs.202405518",
                "figure": "Figure 3b(i)",
                "panel": panel["panel"],
                "subject": panel["subject"],
                "signal_type": "BG_reference",
                "time_min": round(time, 2),
                "value": round(value, 2),
                "unit": "mg/dL",
                "protocol_phase": "post-meal" if time >= 0 else "fasting baseline",
                "exercise_load_W": "",
                "valid_for_model": "REFERENCE",
                "interpolated_short_gap": "NO",
                "x_uncertainty": round(x_uncertainty, 3),
                "x_uncertainty_unit": "min",
                "y_uncertainty": round(y_uncertainty, 3),
                "y_uncertainty_unit": "mg/dL",
                "evidence_level": "PUBLIC_FIGURE_DIGITIZED",
                "source_url": SAHA_URL,
                "notes": "Marker center detected from the red BG reference ring; time is calibrated from the plotted x coordinate.",
            })

    choi_times = [round(value * 0.5, 2) for value in range(0, 141)]
    for panel in choi_panels:
        values, interpolated = trace_colored_curve(
            choi_rgb, panel["bounds"], panel["axis"], panel["color"], choi_times
        )
        x_left, x_right, y_top, y_bottom = panel["bounds"]
        x_min, x_max, y_min, y_max = panel["axis"]
        x_uncertainty = 2.0 * (x_max - x_min) / (x_right - x_left)
        for time, value, was_interpolated in zip(choi_times, values, interpolated):
            if value is None:
                continue
            phase, load, valid = protocol_phase(panel["panel"], time)
            waveform_rows.append({
                "source": "Choi 2020",
                "doi": "10.1038/s41598-020-64406-5",
                "figure": f"Figure 1{panel['panel']}",
                "panel": panel["panel"],
                "subject": "Representative subject (identity not reported)",
                "signal_type": "sweat_chloride",
                "time_min": time,
                "value": round(float(value), 3),
                "unit": "mM",
                "protocol_phase": phase,
                "exercise_load_W": load,
                "valid_for_model": "YES" if valid else "NO_STABILIZATION",
                "interpolated_short_gap": "YES" if was_interpolated else "NO",
                "x_uncertainty": round(x_uncertainty, 3),
                "x_uncertainty_unit": "min",
                "y_uncertainty": panel["y_uncertainty"],
                "y_uncertainty_unit": "mM",
                "evidence_level": "PUBLIC_FIGURE_DIGITIZED",
                "source_url": CHOI_URL,
                "notes": "Published trace was already processed with the reported 1-min median filter; no additional smoothing applied.",
            })

    waveform_rows.sort(key=lambda row: (row["source"], row["panel"], row["signal_type"], float(row["time_min"])))
    waveform_fields = list(waveform_rows[0].keys())
    write_csv(OUTPUT_DIR / "public_figure_digitized_waveforms.csv", waveform_fields, waveform_rows)
    write_csv(
        OUTPUT_DIR / "saha_2024_digitized_waveforms.csv",
        waveform_fields,
        [row for row in waveform_rows if row["source"] == "Saha 2024"],
    )
    write_csv(
        OUTPUT_DIR / "choi_2020_digitized_waveforms.csv",
        waveform_fields,
        [row for row in waveform_rows if row["source"] == "Choi 2020"],
    )

    qc_rows: list[dict] = []
    for panel in saha_panels:
        subject_rows = [row for row in waveform_rows if row["panel"] == panel["panel"]]
        sbg_rows = [row for row in subject_rows if row["signal_type"] == "SBG"]
        bg_rows = [row for row in subject_rows if row["signal_type"] == "BG_reference"]
        sbg_times = [float(row["time_min"]) for row in sbg_rows]
        sbg_values = [float(row["value"]) for row in sbg_rows]
        paired_bg: list[float] = []
        paired_sbg: list[float] = []
        for row in bg_rows:
            estimate = linear_interpolate(sbg_times, sbg_values, float(row["time_min"]))
            if estimate is not None:
                paired_bg.append(float(row["value"]))
                paired_sbg.append(estimate)
        calculated = pearson(paired_bg, paired_sbg)
        qc_rows.append({
            "source": "Saha 2024",
            "panel": panel["panel"],
            "metric": "Pearson r: digitized SBG vs BG marker pairs",
            "digitized_value": round(calculated, 3),
            "reported_or_expected": panel["reported_pr"],
            "absolute_difference": round(abs(calculated - panel["reported_pr"]), 3),
            "tolerance": 0.08,
            "status": "PASS" if abs(calculated - panel["reported_pr"]) <= 0.08 else "REVIEW",
            "notes": f"{len(paired_bg)} digitized BG/SBG pairs; Figure 3b(ii) reports Pr={panel['reported_pr']:.2f}.",
            "source_url": SAHA_URL,
        })

    choi_values: dict[str, tuple[list[float], list[float | None]]] = {}
    for panel in choi_panels:
        rows = [row for row in waveform_rows if row["source"] == "Choi 2020" and row["panel"] == panel["panel"]]
        choi_values[panel["panel"]] = ([float(row["time_min"]) for row in rows], [float(row["value"]) for row in rows])
    for panel_id, direction, expected, tolerance, first_window, second_window in [
        ("B", "upstep", 6.5, 3.3, (24, 33), (47, 54)),
        ("C", "downstep", 6.7, 3.2, (27, 34), (58, 68)),
    ]:
        times, values = choi_values[panel_id]
        before = median_window(times, values, *first_window)
        after = median_window(times, values, *second_window)
        crossing = first_half_crossing(times, values, 35.0, before, after)
        t50 = None if crossing is None else crossing - 35.0
        difference = None if t50 is None else abs(t50 - expected)
        qc_rows.append({
            "source": "Choi 2020",
            "panel": panel_id,
            "metric": f"Representative Figure 1 {direction} t50",
            "digitized_value": "" if t50 is None else round(t50, 3),
            "reported_or_expected": expected,
            "absolute_difference": "" if difference is None else round(difference, 3),
            "tolerance": tolerance,
            "status": "PASS" if difference is not None and difference <= tolerance else "REVIEW",
            "notes": "Compared with the reported cohort mean; the Figure 1 trace is one representative subject, so equality is not expected.",
            "source_url": CHOI_URL,
        })

    qc_fields = list(qc_rows[0].keys())
    write_csv(OUTPUT_DIR / "digitization_qc.csv", qc_fields, qc_rows)

    manifest_rows = []
    for panel in saha_panels:
        x_left, x_right, y_top, y_bottom = panel["bounds"]
        x_min, x_max, y_min, y_max = panel["axis"]
        manifest_rows.append({
            "source": "Saha 2024", "figure_panel": panel["panel"], "rendered_page": "PDF page 7 at 300 dpi",
            "x_left_px": x_left, "x_right_px": x_right, "y_top_px": y_top, "y_bottom_px": y_bottom,
            "x_min": x_min, "x_max": x_max, "x_unit": "min", "y_min": y_min, "y_max": y_max, "y_unit": "mg/dL",
            "extraction": "black-line continuity + red marker centers", "source_url": SAHA_URL,
        })
    for panel in choi_panels:
        x_left, x_right, y_top, y_bottom = panel["bounds"]
        x_min, x_max, y_min, y_max = panel["axis"]
        manifest_rows.append({
            "source": "Choi 2020", "figure_panel": f"Figure 1{panel['panel']}", "rendered_page": "PDF page 2 at 300 dpi",
            "x_left_px": x_left, "x_right_px": x_right, "y_top_px": y_top, "y_bottom_px": y_bottom,
            "x_min": x_min, "x_max": x_max, "x_unit": "min", "y_min": y_min, "y_max": y_max, "y_unit": "mM",
            "extraction": f"{panel['color']} line color segmentation", "source_url": CHOI_URL,
        })
    write_csv(OUTPUT_DIR / "digitization_manifest.csv", list(manifest_rows[0].keys()), manifest_rows)

    (OUTPUT_DIR / "digitization_bundle.json").write_text(
        json.dumps(
            {
                "waveforms": waveform_rows,
                "quality_control": qc_rows,
                "manifest": manifest_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    draw_overlay(saha_pil, saha_panels, waveform_rows, OUTPUT_DIR / "qc_overlay_saha_figure3.png", "Saha 2024")
    draw_overlay(choi_pil, choi_panels, waveform_rows, OUTPUT_DIR / "qc_overlay_choi_figure1.png", "Choi 2020")

    report = """# Public-figure waveform digitization method

## Scope

This stage digitizes only public figures. No author request was made. The output is not described as raw data, deposited data, or author-supplied data.

## Sources and panels

- Saha et al. (2024), Figure 3b(i): continuous fingertip sweat-based glucose (SBG) lines and discrete blood-glucose (BG) markers for healthy Subjects 1-4.
- Choi et al. (2020), Figure 1B-D: representative continuous sweat-chloride traces under single-step, reverse-step, and multistep exercise-load protocols.

## Procedure

1. Render the public PDF page to PNG at 300 dpi with Poppler.
2. Calibrate each axis from the plotted frame and tick labels. Calibration bounds are stored in `digitization_manifest.csv`.
3. Segment Saha BG markers by red color and recover SBG lines from the continuous black path. Sample the SBG path every 2 min.
4. Segment the Choi red, blue, and green traces and sample them every 0.5 min. The published trace already uses the authors' 1-min median filter; no additional smoothing is applied.
5. Interpolate only short raster gaps, flag every interpolated row, and retain the pre-15-min Choi interval as stabilization data that is excluded from model fitting.
6. Assign coordinate uncertainty from approximately two x pixels and three y pixels for Saha, and use conservative panel-specific chloride uncertainty of 0.5-1.0 mM for Choi.

## Quality control

- Saha: compare Pearson correlations from digitized SBG/BG pairs with the Pr values printed in Figure 3b(ii).
- Choi: compare representative-trace t50 estimates with the reported cohort means. The comparison is a plausibility check; a representative subject need not equal the cohort mean.
- Inspect cyan/yellow overlays against the public figure before using the CSVs in a model.

## Use limitation

The digitized curves support waveform scenarios and sensitivity analysis. They must not be used as subject-level raw data for inferential statistics, and raster-level precision must not be presented as biological measurement precision.
"""
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(report, encoding="utf-8")

    summary = {
        "waveform_rows": len(waveform_rows),
        "saha_rows": sum(row["source"] == "Saha 2024" for row in waveform_rows),
        "choi_rows": sum(row["source"] == "Choi 2020" for row in waveform_rows),
        "qc_pass": sum(row["status"] == "PASS" for row in qc_rows),
        "qc_review": sum(row["status"] == "REVIEW" for row in qc_rows),
        "output_dir": str(OUTPUT_DIR),
    }
    (OUTPUT_DIR / "digitization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
