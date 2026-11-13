# CMU horizon and initialization stability

## Completed results (2026-09-27)

All twelve runs completed; every reloaded model matched its selected validation
metric. Errors below are canonical17 MPJPE in millimetres, including zero pelvis.

| Horizon | Hold | Velocity | MLP mean, three seeds | MLP min–max |
| --- | ---: | ---: | ---: | ---: |
| 33 ms | 10.30 | 2.60 | 5.62 | 5.50–5.68 |
| 66 ms | 20.37 | 6.70 | 12.37 | 12.34–12.43 |
| 100 ms | 30.52 | 12.68 | 21.07 | 21.00–21.12 |
| 150 ms | 44.88 | 24.34 | 36.12 | 35.87–36.45 |

The neural predictor beats hold in 12/12 runs and beats constant velocity in 0/12.
Its improvement over hold repeats across these seeds and horizons, but the stronger
clean-mocap baseline still wins. Do not interpret seed ranges as confidence intervals
for new people. No test scoring, live promotion or claim of latency reduction was made.
Regression suite after this change: 413 passing tests; Ruff passes.

## Protocol

Fixed follow-up to the input/batch-order ablation. Before running any training,
the script writes protocol.json for all twelve cells:

- Horizons: 33, 66, 100, 150 ms.
- Seeds: 42, 43, 44 (no best-seed selection).
- Relative-last inputs, global-window shuffle, two 64-unit hidden layers.
- Eight epochs, Adam 0.001, batch size 64; validation-selected epoch, including zero.
- Exact existing expanded-CMU windows and train-only normalizers; no new data.

```powershell
.venv/Scripts/python.exe scripts/cmu_seed_stability.py outputs/cmu_expanded_20260924 outputs/cmu_stability_NEW
```

Retained run: `outputs/cmu_stability_20260927/`. Each completed cell saves a model,
training log and result JSON. The final report.json is written only after all twelve
cells complete and pinned input manifest/normalizer hashes are rechecked. Existing
directories are never overwritten. An interrupted directory is not a complete study.

Every checkpoint is reloaded and evaluated on the same checked validation windows.
The reloaded error must match the selected training-log metric. Hold and last-two-
sample constant velocity are computed alongside it. The summary requires all cells,
rejects duplicate/nonfinite results and rejects differing baseline scores between
seeds at the same horizon. It reports every seed, arithmetic mean, minimum/maximum
and counts of improvements over each baseline.

This measures random-initialization/shuffle sensitivity of one selected architecture
on validation, not participant-level confidence intervals or independent test
generalization. The same three validation clips/source ID have been consulted during
development; they are not a fresh test. No noise, dropout, effective latency or jitter
is evaluated here. Results do not authorize automatic live-model promotion.
