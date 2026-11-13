# Clip-consistent noise augmentation pilot

## Completed result (2026-10-03)

Exported 19,359 augmented training windows; their targets, target/history times and
anchor IDs match the previous clean export exactly, while input histories differ.
Eight epochs completed; clean validation selected epoch 5 (target MPJPE 24.13 mm).
The checkpoint reloaded through the shared runner, and all six evaluation runs / 49
batch artifacts verified. Full regression suite: 426 passing tests, Ruff clean.

Display-time macro means on the fixed delay/noise protocol:

| Model / method | Clean error, mm | Noisy error, mm | Clean jitter, m/s² | Noisy jitter, m/s² |
| --- | ---: | ---: | ---: | ---: |
| Previous clean-trained MLP | 20.27 | 50.09 | 11.07 | 174.08 |
| Noise-trained MLP, no prefilter | 24.09 | 32.37 | 7.27 | 72.44 |
| Previous MLP + 50-ms prefilter | 28.28 | 32.74 | 5.88 | 36.42 |
| Kalman extrapolation defaults | 25.34 | 31.90 | 6.20 | 36.71 |

Noise augmentation reduces noisy error by about 35.4% and noisy jitter by 2.4x
relative to the previous unfiltered MLP, but worsens clean accuracy. It has not
demonstrated superiority over Kalman extrapolation in the noisy condition, and is
roughly twice as jittery there. No overall winner or live promotion is claimed.

Clean effective latency is 87.87 ms at 76.4% mean joint-window coverage. Noisy
71.76 ms uses only two eligible sequences and 13.4% coverage; it is not a reliable
whole-sequence compensation claim. Test data remains untouched. Evaluation artifacts:
`outputs/cmu_noise_eval_20261003/`. The controlled combination with causal filtering
is now complete: see [noise-aware prefilter sweep](cmu_noise_prefilter.md).
Broader corruption profiles/seeds and camera validation remain pending.

One predeclared 100-ms experiment on the unchanged expanded CMU split:

- Non-pelvis observation XYZ Gaussian noise: sigma 10 mm; root unchanged, no dropout.
- Base augmentation seed 2026, SHA-256-derived separate stream per sequence/clock.
- Corruption is generated once per original 120-Hz source frame, before history
  interpolation/window extraction. Shared observations in overlapping windows agree.
- Future target poses always come from the unmodified reference. Target timestamps,
  validity and input availability checks keep their existing semantics.
- Relative-last MLP, global mixing, eight epochs, model seed 42, existing architecture
  and optimizer. The normalizer is refitted on augmented **training histories only**.
- Epoch selection uses the exact previous clean validation directory; no noisy
  validation score or test data is used to choose the checkpoint in this pilot.

Window export has optional training_noise_std_m/training_noise_seed arguments.
Nonzero corruption is train-only and writes version-3 manifests containing the
noise stream and clean-supervision contract. Version-1/2 clean windows remain valid.
No noise is independently regenerated for each window or epoch in this experiment.
This is white-noise augmentation, not a realistic detector-error simulator.

```powershell
.venv/Scripts/python.exe scripts/cmu_noise_training.py outputs/cmu_expanded_20260924 outputs/cmu_noise_training_NEW
.venv/Scripts/python.exe scripts/cmu_delay_noise.py outputs/cmu_expanded_20260924/index.json outputs/cmu_noise_training_NEW/model outputs/cmu_noise_eval_NEW --learned-only
```

Current training directory: `outputs/cmu_noise_training_20261003_v2/`. The earlier
directory without `_v2` is an incomplete export from an implementation error caught
and fixed in testing (missing explicit dropout=0 argument); it has no completed
training report/model and must not be used. Existing artifacts were not overwritten.

Evaluation uses the same three validation clips, delay 100 ms, display grid 40 Hz,
noise sigma 0/10 mm and evaluation seed 42 as the previous delay/noise pilot, with
history prefilter disabled. This separates training corruption from inference
prefiltering. Comparing against prior learned/default-filter results does not prove
generalization: these validation clips have been used repeatedly in development.
