# Causal EMA before the learned predictor

## Completed sweep (2026-10-01)

All 24 runs and 193 batch artifacts verified. Zero-filter display arrays match the
previous six learned runs exactly, including times and validity. 424 tests pass.

| EMA tau | Clean error, mm | Noisy error, mm | Clean jitter, m/s² | Noisy jitter, m/s² |
| --- | ---: | ---: | ---: | ---: |
| 0 ms | 20.27 | 50.09 | 11.07 | 174.08 |
| 10 ms | 21.35 | 38.17 | 9.40 | 112.86 |
| 25 ms | 23.89 | 32.64 | 7.34 | 63.51 |
| 50 ms | 28.28 | 32.74 | 5.88 | 36.42 |

Under this synthetic noise, 50-ms EMA reduces learned jitter by about 4.8x and
error by about 35%, but clean-data error grows. Kalman extrapolation from the
preceding matched protocol has noisy error 31.90 mm and jitter 36.71 m/s²: the
prefiltered network has not demonstrated a clear advantage over it.

Clean effective-latency means increase from 76.40 to 96.44 ms as tau grows from
0 to 50 ms (coverage 71.7% to 80.0%). Noisy latency coverage remains poor: 1.0%,
3.0%, 12.9%, 18.3% for the four settings; the corresponding means do not use matched
eligible subsets. Do not select a winner using those means alone.

No setting is enabled by default or promoted to live use. Next: train on causally
corrupted inputs with clean future targets, then evaluate the same fixed corruption
conditions and assess more realistic detector errors. Gaussian augmentation must
not independently resample shared observations inside each overlapping window.

Metadata erratum: this retained run's protocol.json has the earlier generic
`tuning` text about existing defaults; its explicit prefilter_sweep_ms list and
resolved per-run configs correctly record this validation sweep. The script now
writes the specific sweep description for future runs. Batch artifacts were not edited.

Optional `history_prefilter_ms` was added to LearnedProcessor. Zero (default)
preserves the previous path. Positive values apply timestamp-based EMA to observed
joint positions before history interpolation and prediction. The same trained
checkpoint is used; this is not noise-aware retraining.

Only model history positions are filtered. Raw recordings, current root position,
event/availability timestamps, and raw fallback outputs are unchanged. Filter and
history reset on invalid observations, tracking gaps and explicit reset. Filter
lag is not subtracted from timestamps or hidden by relabelling target time. The
parameter is pinned in processor metadata and resolved experiment configs.

The validation sweep predeclares tau = 0/10/25/50 ms, with both clean inputs and
independent 10-mm Gaussian XYZ noise, across the same three clips and 100-ms delay.
It retains source 120 Hz, display 40 Hz, seed 42, and the 500-ms warm-up exclusion.
This is 24 runs. Tau zero is an internal reproduction control. Test data and live
defaults remain unchanged. Data and limitations match [the preceding study](cmu_delay_noise.md).

```powershell
.venv/Scripts/python.exe scripts/cmu_delay_noise.py outputs/cmu_expanded_20260924/index.json outputs/cmu_stability_20260927/h100_seed42 outputs/cmu_prefilter_NEW --prefilter-sweep
```

Retained directory: `outputs/cmu_prefilter_20261001/`. Batch completion and
verified.json are required before treating it as complete. Inspect error, jitter
and latency coverage together: a stronger filter may suppress noise while adding
lag. A low latency number with almost no eligible windows is not a success.

These are validation-tuned candidates after observing a noise failure, not a fresh
held-out study. Independent Gaussian noise is not calibrated webcam noise. A model
trained on unsmoothed mocap sees a changed input distribution under this filter;
the experiment measures that tradeoff rather than assuming it harmless.
