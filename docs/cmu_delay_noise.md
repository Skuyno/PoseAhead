# CMU display-time delay/noise pilot

## Completed results (2026-10-01)

All 36 runs completed and 289 batch artifacts passed integrity verification.
All display-time joint coverages are checked in the per-run metrics. The table
reports equal-weight sequence means, not the earlier window-weighted target error.

| Method | Clean MPJPE, mm | Noisy MPJPE, mm | Clean jitter, m/s² | Noisy jitter, m/s² |
| --- | ---: | ---: | ---: | ---: |
| Raw / hold | 29.28 | 36.08 | 4.80 | 58.95 |
| EMA | 40.27 | 41.04 | 4.10 | 11.17 |
| One-Euro | 57.68 | 57.85 | 3.89 | 5.40 |
| Kalman | 31.98 | 33.67 | 4.43 | 15.40 |
| Kalman extrapolation | 25.34 | 31.90 | 6.20 | 36.71 |
| Learned relative-last | 20.27 | 50.09 | 11.07 | 174.08 |

The neural model reduces clean display error but increases acceleration-residual
jitter. Independent 10-mm input noise makes its error worse than Raw, with much
larger jitter. This exposes sensitivity to this synthetic corruption; it does not
establish the error distribution of the real camera. The smoother filters trade
lower jitter for greater lag/error with their untuned defaults.

Latency needs particular caution. Clean Raw has 99.93-ms mean sequence latency
with 94.1% mean joint-window coverage; clean learned has 76.40 ms with 71.7% coverage.
These are not matched eligible subsets, so their difference is not a final paired
latency benefit claim. With noise the learned 59.73-ms aggregate has only one of
three sequences eligible and about 1.0% mean joint-window coverage. Do not report
that number alone as successful compensation. Raw noisy coverage is 14.1%.

No overall winner or acceptance claim is made. The last-two-sample extrapolator
from the previous target-window study is not included here. Next experiments
should address noise robustness (training-only corruption and/or causal filtering
before prediction) with clean-reference supervision, before live promotion.
Regression suite: 416 tests pass; Ruff passes. No test or live configuration changed.

## Protocol

This uses the shared experiment/batch runner, not the target-window evaluator.
The predeclared grid contains three unchanged validation recordings, six methods,
and two noise conditions: 36 runs. Test recordings are not scored.

- Source recordings: original 120-Hz prepared CMU poses.
- Artificial observation delay: 100 ms.
- Display/evaluation grid: 40 Hz, starting at 500 ms to exclude warm-up.
  Every 25-ms timestamp is present in the original 120-Hz reference (every third
  source frame). No pose is assigned a fabricated reference timestamp.
- Noise: independent Gaussian XYZ noise with sigma 0 or 10 mm per coordinate
  on non-pelvis joints; root unchanged, no dropout. Seed 42 for every method.
- Methods: Raw, EMA, One-Euro, Kalman, Kalman constant-velocity prediction at
  100 ms, and relative-last/global-shuffle MLP at 100 ms (preselected seed 42).
- Filters retain existing defaults; no test or validation parameter search here.
- Model history uses linear-past interpolation. Model weights remain unchanged.

The Kalman extrapolator here is **not** the last-two-sample constant-velocity
formula in the previous window-level reports. Do not equate their scores or treat
the two reports as a matched model-versus-simple-velocity comparison.

```powershell
.venv/Scripts/python.exe scripts/cmu_delay_noise.py outputs/cmu_expanded_20260924/index.json outputs/cmu_stability_20260927/h100_seed42 outputs/cmu_delay_noise_NEW
```

Retained directory: `outputs/cmu_delay_noise_20261001/`. Configs and protocol.json
are written before batch execution. Each child run saves resolved configs, source
hashes, degraded observations, processed/display poses, metrics and completion
hashes. The complete batch is integrity-verified before verified.json is written.

Metrics are display-time canonical17 MPJPE, root error, acceleration-residual
jitter and velocity-correlation effective latency. Latency uses nonoverlapping
40-frame windows and maximum lag 200 ms; ambiguous/static windows are rejected.
Always report coverage: a small latency estimate on a small eligible subset is not
proof of globally low delay. Group summaries weight the three sequence means
equally, not all frames equally. Null latency values must not be treated as zero.

CPU processor duration is recorded separately and is **not** injected into the
simulated delay. No GPU/camera/Unity throughput or camera-to-display timing is
measured. White independent noise is a stress condition, not a calibrated detector
error model. This is not final method selection or statistical significance.
