"""Topology-aware transport and sensor response models for the Perspective.

These functions define the revision's computational contract. They do not turn
unreported device geometry into evidence; inputs must come from a cited source
or be labeled as a sensitivity-analysis assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gamma

import numpy as np


@dataclass(frozen=True)
class FidelityMetrics:
    amplitude_retention: float
    peak_delay_s: float
    nrmse: float


def _check_series(time_s: np.ndarray, values: np.ndarray) -> float:
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    if time_s.ndim != 1 or values.ndim != 1 or len(time_s) != len(values):
        raise ValueError("time_s and values must be one-dimensional and equal length")
    if len(time_s) < 2:
        raise ValueError("at least two samples are required")
    steps = np.diff(time_s)
    if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-12):
        raise ValueError("time_s must be strictly increasing and uniformly sampled")
    return float(steps[0])


def first_order_filter(time_s: np.ndarray, input_signal: np.ndarray, tau_s: float) -> np.ndarray:
    """Exact zero-order-hold recursion for dY/dt=(U-Y)/tau."""
    dt = _check_series(time_s, input_signal)
    if tau_s <= 0:
        raise ValueError("tau_s must be positive")
    u = np.asarray(input_signal, dtype=float)
    y = np.empty_like(u)
    y[0] = u[0]
    decay = np.exp(-dt / tau_s)
    for i in range(1, len(u)):
        y[i] = decay * y[i - 1] + (1.0 - decay) * u[i - 1]
    return y


def pure_delay(time_s: np.ndarray, input_signal: np.ndarray, delay_s: float) -> np.ndarray:
    """Plug-flow approximation with a pure transport delay."""
    _check_series(time_s, input_signal)
    if delay_s < 0:
        raise ValueError("delay_s cannot be negative")
    t = np.asarray(time_s, dtype=float)
    u = np.asarray(input_signal, dtype=float)
    return np.interp(t - delay_s, t, u, left=u[0], right=u[-1])


def gamma_rtd_kernel(dt_s: float, mean_residence_s: float, shape: float = 3.0, support_factor: float = 10.0) -> np.ndarray:
    """Return a normalized gamma residence-time-distribution kernel."""
    if dt_s <= 0 or mean_residence_s <= 0 or shape <= 0 or support_factor <= 0:
        raise ValueError("RTD parameters must be positive")
    scale = mean_residence_s / shape
    theta = np.arange(0.0, support_factor * mean_residence_s + dt_s, dt_s)
    safe_theta = np.maximum(theta, np.finfo(float).eps)
    kernel = safe_theta ** (shape - 1.0) * np.exp(-safe_theta / scale)
    kernel /= gamma(shape) * scale**shape
    kernel /= np.sum(kernel) * dt_s
    return kernel


def rtd_convolution(time_s: np.ndarray, input_signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve an inlet concentration with a normalized RTD kernel."""
    dt = _check_series(time_s, input_signal)
    kernel = np.asarray(kernel, dtype=float)
    if kernel.ndim != 1 or len(kernel) < 2 or np.any(kernel < 0):
        raise ValueError("kernel must be a nonnegative one-dimensional array")
    normalized = kernel / (np.sum(kernel) * dt)
    baseline = float(np.asarray(input_signal)[0])
    centered = np.asarray(input_signal, dtype=float) - baseline
    filtered = np.convolve(centered, normalized * dt, mode="full")[: len(centered)]
    return filtered + baseline


def transport_sensor_response(
    time_s: np.ndarray,
    inlet: np.ndarray,
    *,
    topology: str,
    transport_time_s: float,
    sensor_tau_s: float | None = None,
    rtd_shape: float = 3.0,
) -> np.ndarray:
    """Apply a topology-specific transport model and optional sensor kinetics."""
    dt = _check_series(time_s, inlet)
    topology = topology.lower()
    if topology == "cstr":
        fluid = first_order_filter(time_s, inlet, transport_time_s)
    elif topology == "plug":
        fluid = pure_delay(time_s, inlet, transport_time_s)
    elif topology == "gamma_rtd":
        kernel = gamma_rtd_kernel(dt, transport_time_s, rtd_shape)
        fluid = rtd_convolution(time_s, inlet, kernel)
    else:
        raise ValueError("topology must be 'cstr', 'plug', or 'gamma_rtd'")
    return fluid if sensor_tau_s is None else first_order_filter(time_s, fluid, sensor_tau_s)


def event_reset_update(old_concentration: float, new_concentration: float, residual_fraction: float) -> float:
    """Concentration immediately after a fill-clear event."""
    if not 0.0 <= residual_fraction <= 1.0:
        raise ValueError("residual_fraction must lie in [0, 1]")
    return residual_fraction * old_concentration + (1.0 - residual_fraction) * new_concentration


def event_reset_response(
    time_s: np.ndarray,
    input_signal: np.ndarray,
    cycle_period_s: float,
    residual_fraction: float,
    clear_delay_s: float = 0.0,
) -> np.ndarray:
    """Return a causal sample-and-hold trace for repeated fill-clear events.

    The sampled concentration is updated once per cycle. ``residual_fraction``
    represents old sample that remains after each event, while ``clear_delay_s``
    optionally delays the newly sampled inlet concentration.
    """
    dt = _check_series(time_s, input_signal)
    if cycle_period_s < dt:
        raise ValueError("cycle_period_s must be at least one sample interval")
    if clear_delay_s < 0:
        raise ValueError("clear_delay_s cannot be negative")
    if not 0.0 <= residual_fraction <= 1.0:
        raise ValueError("residual_fraction must lie in [0, 1]")

    t = np.asarray(time_s, dtype=float)
    source = np.asarray(input_signal, dtype=float)
    if clear_delay_s > 0:
        source = pure_delay(t, source, clear_delay_s)
    output = np.empty_like(source)
    output[0] = source[0]
    next_event_s = t[0] + cycle_period_s
    for i in range(1, len(source)):
        output[i] = output[i - 1]
        if t[i] + 0.5 * dt >= next_event_s:
            output[i] = event_reset_update(output[i - 1], source[i], residual_fraction)
            next_event_s += cycle_period_s
    return output


def chrono_sampling_response(
    time_s: np.ndarray,
    input_signal: np.ndarray,
    collection_window_s: float,
) -> np.ndarray:
    """Return causal non-overlapping collection-window averages.

    Each completed reservoir/window is averaged and then held until the next
    window closes. The first value is held during the initial collection window.
    """
    dt = _check_series(time_s, input_signal)
    if collection_window_s < dt:
        raise ValueError("collection_window_s must be at least one sample interval")
    t = np.asarray(time_s, dtype=float)
    source = np.asarray(input_signal, dtype=float)
    output = np.empty_like(source)
    output[0] = source[0]
    window_start = 0
    next_close_s = t[0] + collection_window_s
    held_value = source[0]
    for i in range(1, len(source)):
        if t[i] + 0.5 * dt >= next_close_s:
            held_value = float(np.mean(source[window_start : i + 1]))
            window_start = i + 1
            next_close_s += collection_window_s
        output[i] = held_value
    return output


def fidelity_metrics(time_s: np.ndarray, reference: np.ndarray, measured: np.ndarray) -> FidelityMetrics:
    """Return amplitude retention, peak delay, and range-normalized RMSE."""
    _check_series(time_s, reference)
    _check_series(time_s, measured)
    t = np.asarray(time_s, dtype=float)
    ref = np.asarray(reference, dtype=float)
    obs = np.asarray(measured, dtype=float)
    ref_amp = float(np.max(ref) - np.min(ref))
    obs_amp = float(np.max(obs) - np.min(obs))
    if ref_amp <= 0:
        raise ValueError("reference must have nonzero amplitude")
    return FidelityMetrics(
        amplitude_retention=obs_amp / ref_amp,
        peak_delay_s=float(t[np.argmax(obs)] - t[np.argmax(ref)]),
        nrmse=float(np.sqrt(np.mean((obs - ref) ** 2)) / ref_amp),
    )
