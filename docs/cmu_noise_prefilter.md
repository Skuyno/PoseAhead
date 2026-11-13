# Noise-aware MLP with causal history filtering

## Completed validation sweep (2026-10-03)

Used the existing noise-trained 100-ms checkpoint, without further training or
checkpoint selection. The same three validation clips, delay 100 ms, source 120 Hz,
display 40 Hz, noise sigma 0/10 mm, evaluation seed 42 and 500-ms warm-up exclusion
as the earlier studies were retained. Only EMA tau changes: 0/10/25/50 ms.

All 24 runs / 193 batch artifacts verified. All display sample fields in the six
tau-zero controls are exactly equal to the earlier unfiltered noise-trained runs.
Artifact directory: `outputs/cmu_noise_prefilter_20261003/`.

Equal-sequence-weight display macro means:

| EMA tau | Clean error, mm | Noisy error, mm | Clean jitter, m/s² | Noisy jitter, m/s² |
| --- | ---: | ---: | ---: | ---: |
| 0 ms | 24.09 | 32.37 | 7.27 | 72.44 |
| 10 ms | 25.40 | 29.58 | 6.65 | 45.10 |
| 25 ms | 28.31 | 30.41 | 5.80 | 24.84 |
| 50 ms | 32.89 | 33.99 | 5.10 | 14.61 |
| Kalman extrapolation defaults, earlier matched pilot | 25.34 | 31.90 | 6.20 | 36.71 |

The 10-ms candidate has the lowest noisy error among these four settings, but
more noisy jitter than Kalman extrapolation. The 25-ms candidate is more accurate
and smoother than Kalman on this noisy condition, but less accurate on clean data.
The 50-ms candidate suppresses jitter most strongly while worsening accuracy.
This is a validation tradeoff, not an overall winner or a production default.

Effective latency and mean eligible joint-window coverage:

| EMA tau | Clean latency, ms | Clean coverage | Noisy latency, ms | Noisy coverage |
| --- | ---: | ---: | ---: | ---: |
| 0 ms | 87.87 | 76.4% | 71.76 | 13.4% |
| 10 ms | 90.96 | 76.5% | 76.77 | 17.1% |
| 25 ms | 97.82 | 75.8% | 91.67 | 24.7% |
| 50 ms | 107.69 | 75.4% | 105.60 | 36.2% |

Eligible subsets differ across settings. In particular, noisy coverage is limited,
so these means must not be used as paired evidence of a latency improvement.
The strongest smoothing has clean effective latency above the injected 100-ms
delay; reduced jitter is not evidence of successful compensation by itself.
Inference CPU time is measured separately, not injected into simulated delay.

Only observed model-history joints are filtered. Root, raw fallback and event/
availability times retain their original policies; target time is not relabelled
to conceal filter lag. Root error remains 38.22 mm for all these learned settings.

No test clips were scored and no live configuration was changed. Independent
Gaussian noise is a synthetic stress condition, not calibrated MeTRAbs noise.
These three clips belong to one repeatedly consulted validation source ID, so
generalization to new people or real-camera recordings has not been demonstrated.

```powershell
.venv/Scripts/python.exe scripts/cmu_delay_noise.py outputs/cmu_expanded_20260924/index.json outputs/cmu_noise_training_20261003_v2/model outputs/cmu_noise_prefilter_NEW --prefilter-sweep
```

For the checkpoint's training protocol, see [noise augmentation](cmu_noise_training.md).
For the earlier clean-trained-model sweep, see [prefilter pilot](cmu_prefilter.md).
