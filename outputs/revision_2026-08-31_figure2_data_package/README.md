# Figure 2 quantitative data package

## Recommended use

- **Fig. 2c:** use `02_canonical_extended_grid_1620.csv`. The revised manuscript reports the sub-0.5-min boundary check; using only the original 1,080 rows would incorrectly leave the 30-min sinusoid unresolved. `01_canonical_primary_grid_1080.csv` is retained for exact reproduction of the original six-point analysis.
- **Fig. 2b:** use `04_representative_model_traces.csv`. These are calculated outputs, not hand-drawn curves. The fixed condition is T=30 min, tau_transport=5 min, tau_sensor=1 min, and gamma shape k=3.
- **Fig. 2d:** use `05_empirical_waveforms_model_inputs.csv` for the seven plotted waveforms and `08_empirical_continuous_only_1155.csv` for the continuous stress-test metrics. The full 1,295-row state-model grid is retained in file 07.

## Exact definition of T

- Pulse: Gaussian FWHM.
- Ramp: duration of the linear rise from 0 to 1.
- Sinusoid: oscillation period.
- Tested values: 2, 5, 10, 30, and 60 min.

## Model definitions

- Plug flow: pure delay by tau_transport.
- Well mixed: exponential RTD with mean tau_transport.
- Distributed RTD: gamma distribution with mean tau_transport and shape k=3 in the primary and extended grids.
- Sensor: independent first-order response with tau_sensor.

## Strict criterion

Canonical: amplitude retention >=0.90, NRMSE <=0.10, Pearson r >=0.95, and absolute feature delay <=max(1 min,T/4).

Empirical: the same amplitude, NRMSE, and correlation limits, with source-specific delay limits of 5 min for Saha and 2 min for Choi. Relaxed limits are supplied in the empirical files.

## Interpretation boundary

The Saha and Choi curves are digitized outputs of published wearable devices. They are empirical waveform proxies for incremental-distortion tests, not device-free physiological inputs and not evidence for a blood-to-sweat transfer function.
