"""Numerical invariants for the frozen transport-fidelity model."""

from __future__ import annotations

import json

import numpy as np

from transport_fidelity_models import (
    chrono_sampling_response,
    event_reset_update,
    event_reset_response,
    fidelity_metrics,
    first_order_filter,
    gamma_rtd_kernel,
    pure_delay,
    transport_sensor_response,
)


def main() -> None:
    t = np.arange(0.0, 600.0 + 0.5, 0.5)
    step = (t >= 30.0).astype(float)
    tau = 40.0
    cstr = first_order_filter(t, step, tau)
    after_step = np.where(t >= 30.0)[0]
    t90 = t[after_step[np.argmax(cstr[after_step] >= 0.9)]] - 30.0
    expected_t90 = np.log(10.0) * tau
    assert abs(t90 - expected_t90) <= 1.0

    delayed = pure_delay(t, step, 20.0)
    assert t[np.argmax(delayed >= 0.5)] == 50.0

    dt = t[1] - t[0]
    kernel = gamma_rtd_kernel(dt, mean_residence_s=40.0, shape=3.0)
    assert abs(np.sum(kernel) * dt - 1.0) < 1e-10

    pulse = np.exp(-0.5 * ((t - 150.0) / 30.0) ** 2)
    cascade = transport_sensor_response(
        t, pulse, topology="gamma_rtd", transport_time_s=40.0, sensor_tau_s=20.0
    )
    metrics = fidelity_metrics(t, pulse, cascade)
    assert 0.0 < metrics.amplitude_retention < 1.0
    assert metrics.peak_delay_s > 0.0
    assert metrics.nrmse > 0.0

    reset = event_reset_update(1.0, 0.0, residual_fraction=0.1)
    assert abs(reset - 0.1) < 1e-12

    reset_trace = event_reset_response(t, step, cycle_period_s=10.0, residual_fraction=0.25)
    step_index = int(np.where(t >= 30.0)[0][0])
    assert abs(reset_trace[step_index] - 0.75) < 1e-12

    chrono = chrono_sampling_response(t, step, collection_window_s=20.0)
    close_40 = int(np.where(t >= 40.0)[0][0])
    assert 0.45 <= chrono[close_40] <= 0.55

    failed_guard = False
    try:
        first_order_filter(t, step, 0.0)
    except ValueError:
        failed_guard = True
    assert failed_guard

    print(
        json.dumps(
            {
                "status": "pass",
                "cstr_t90_s": float(t90),
                "cstr_expected_t90_s": float(expected_t90),
                "gamma_kernel_integral": float(np.sum(kernel) * dt),
                "cascade_metrics": metrics.__dict__,
                "event_reset_residual": float(reset),
                "event_reset_step_value": float(reset_trace[step_index]),
                "chrono_second_window_value": float(chrono[close_40]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
