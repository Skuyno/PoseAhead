# Constant-velocity residual predictor pilot

## Completed result (2026-10-03)

Implemented an optional analytic output baseline for the existing fixed-horizon
MLP. Instead of adding its correction to the last observation, the model adds it
to `last + (last - previous) * horizon_ns / step_ns`. The time interval is the
trained 20-ms history grid, not the original 120-Hz source interval. The zero
output layer makes epoch zero exactly constant velocity, up to float32 arithmetic.
The model still predicts pelvis-relative joints only and holds root translation.

Version-3 checkpoints explicitly record baseline and input representation. Loading
old version-1/2 checkpoints preserves hold semantics; contradictory/downgraded
velocity declarations are rejected. Training and shared-runner tests cover physical
normalization, gradients, offset equivariance, round-trip, repeatability, caller RNG,
checkpoint pinning, invalid parameters and test-split exclusion. Full suite: 438
passing tests; Ruff clean.

## Matched training experiment

Two predeclared 100-ms runs use the unchanged clean/noisy training exports and clean
validation. Each has relative-last inputs, global window mixing, seed 42, eight
epochs, 64 hidden units, batch size 64 and Adam learning rate 0.001. Each uses its
own previously fitted training-only normalizer. All 19,359 clean/noisy window
targets, time grids and anchor IDs were checked for equality before training.
The output baseline is the only model change from the previous training variants.

| Training histories | Selected epoch | Validation target MPJPE, mm | Best trained epoch MPJPE, mm |
| --- | ---: | ---: | ---: |
| Clean | 0 | 12.68 | 14.90 |
| Sigma-10mm observations, clean targets | 0 | 12.68 | 16.86 |

Reloaded checkpoints match the selected scores. Neither learned correction improves
clean validation on this fixed pilot; both selected models are the analytic
baseline, not evidence of a neural-network improvement. Training noise cannot be
credited with robustness here because the noisy-trained correction was not selected.
This clean-only checkpoint-selection protocol is intentionally unchanged; it does
not optimize a combined clean/noisy deployment objective.

Artifacts: `outputs/cmu_velocity_residual_20261003/`.

```powershell
.venv/Scripts/python.exe scripts/cmu_velocity_residual.py outputs/cmu_expanded_20260924 outputs/cmu_noise_training_20261003_v2 outputs/cmu_velocity_residual_NEW
```

## Shared display-time delay/noise evaluation

Evaluated the clean-selected epoch-zero checkpoint with the same three validation
clips, injected delay 100 ms, display 40 Hz, source 120 Hz, noise sigma 0/10 mm,
evaluation seed 42 and EMA tau 0/10/25/50 ms. All 24 runs / 193 batch artifacts
verified. This measures the analytic baseline through the actual learned-history
adapter, including warm-up/fallback semantics; it is not Kalman extrapolation.

| EMA tau | Clean error, mm | Noisy error, mm | Clean jitter, m/s² | Noisy jitter, m/s² |
| --- | ---: | ---: | ---: | ---: |
| 0 ms | 14.41 | 107.57 | 19.46 | 469.32 |
| 10 ms | 14.80 | 67.77 | 14.69 | 295.79 |
| 25 ms | 16.58 | 41.19 | 9.38 | 149.93 |
| 50 ms | 20.72 | 31.65 | 6.42 | 79.45 |

Unfiltered constant velocity is accurate on clean motion, but strongly amplifies
independent observation noise. A 50-ms filter controls error, yet its noisy jitter
remains much worse than the earlier matched Kalman-CV result (31.90 mm / 36.71 m/s²)
and noise-trained hold-residual MLP with tau 25 ms (30.41 mm / 24.84 m/s²).
The 12.68-mm target-aligned training score and 14.41-mm display score use different
queries/weighting and must not be presented as the same metric.

| EMA tau | Clean latency, ms | Clean coverage | Noisy latency, ms | Noisy coverage |
| --- | ---: | ---: | ---: | ---: |
| 0 ms | 76.53 | 65.0% | Not estimable | 0.0% |
| 10 ms | 86.18 | 88.5% | 51.09 | 1.0% |
| 25 ms | 90.34 | 93.2% | 62.16 | 7.4% |
| 50 ms | 95.64 | 93.9% | 80.19 | 13.0% |

Latency means use different eligible subsets; low noisy coverage prevents a robust
latency-compensation claim. Root error remains 38.22 mm, since the analytic joint
baseline does not forecast root. CPU time is not added to simulated delay.

Artifacts: `outputs/cmu_velocity_delay_noise_20261003/`.

```powershell
.venv/Scripts/python.exe scripts/cmu_delay_noise.py outputs/cmu_expanded_20260924/index.json outputs/cmu_velocity_residual_20261003/clean outputs/cmu_velocity_delay_noise_NEW --prefilter-sweep
```

No test data, extra subjects or live defaults were used. This is one seed and one
repeatedly consulted validation source ID. Gaussian noise is not calibrated camera
error. A new architecture option and reproducible negative result are delivered,
not M7 acceptance or a final winner. Further model tuning needs a predeclared robust
selection objective and broader data, while Unity retargeting and real-camera
acceptance remain independent unfinished integration work.
