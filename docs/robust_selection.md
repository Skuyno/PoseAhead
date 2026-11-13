# Robust fixed-comparator selection

## Purpose and frozen rule (2026-10-07)

This chooses a fixed **validation comparator**, not a final neural winner or proof
of latency compensation. Existing CMU pilots have already informed this protocol;
it is prospective for the new panel, not retrospectively predeclared for old results.
Training/epoch selection and archived checkpoints remain unchanged. Test is not scored.

The policy in `poseahead.evaluation.selection.SELECTION_RULE` is versioned and saved
before experiments. It requires the complete candidate x profile x noise-seed x
sequence panel; missing or duplicate cells are errors. Every candidate uses identical
observations/reference hashes, clock, evaluation grid, delay and metric settings.
Its parameters/checkpoint and the corruption profiles must match the pinned declaration.

Selection:

1. Every cell needs finite pose metrics and at least 95% joint/jitter coverage;
   its pose-validity mask must match the fixed baseline. An invalid comparator
   cannot authorize selection of a candidate on fewer/easier observations.
2. On clean data, equal-sequence/equal-noise-seed macro MPJPE and primary jitter
   may each grow by at most 10% relative to the declared baseline.
3. For each corrupted profile, compute macro MPJPE and jitter ratios to that same
   baseline. Score is the **maximum** ratio over metrics and corrupted profiles;
   lower is better. Clean data is a constraint, not another corrupted profile.
4. Keep the baseline eligible. Exact ties prefer it, then declared candidate order.
   Never pick the best noise seed or pool frames to overweight long clips.
5. A zero baseline metric allows zero candidate error, not division by zero or
   unlimited growth. Undefined ratios exclude the candidate. If none is eligible,
   report `no_eligible_comparator`, not a fabricated winner.

This conservative objective is not the SPEC/TZ three-metric superiority rule.
Effective latency is reported descriptively only when all candidate/baseline
eligibility masks match and every metric exists. Different masks or missing values
give `not_comparable`, even if the aggregate coverage percentages are equal.
Low coverage remains visible. **Latency is not optimized by this selector**; a
latency claim still requires the existing frozen paired `20 ms / 10%` comparison
or the pointwise 95% bootstrap noninferiority rule on a matched held-out panel.
Root error is separate: these MLPs hold observed root translation and do not forecast it.

## Prepared CMU continuation

`outputs/cmu_robust_selection_20261007/` pins 252 validation cases:

- Seven fixed candidates: Raw, default Kalman-CV at 100 ms, clean-trained seed-42
  100-ms MLP, noise-trained seed-42 MLP with history EMA tau 0/10/25/50 ms.
- Comparator used for normalization: Kalman-CV. Delay is 100 ms, display 40 Hz,
  original mocap source 120 Hz, first 500 ms excluded from display evaluation.
- Gaussian joint noise sigma 0/10/25/50 mm, one family at a time; pelvis/root
  unchanged. Evaluation noise seeds 42/43/44 are **not** new model seeds/people.
- Unchanged validation clips: `cmu/13/01`, `cmu/13/11`, `cmu/13/29`. These have one
  repeatedly consulted CMU source ID; verified independent-actor generalization
  is not established. Missing joints, outliers and gaps require a later protocol.
- Existing model files are verified and run on CPU; no retraining, epoch/seed
  reselection, camera access, Unity launch or production configuration change.

The preparer saves configs, `selection_protocol.json`, `declaration.json` and
`study.json` before scoring. Three independent batch directories contain 84 runs
each because repeated noise seeds must not inflate a batch's sequence count.
The final `selection_manifest.json` pins protocol/declaration/batch completion
hashes; `selection.json` is written only after all batches verify and agree with
the declaration. An interrupted directory is not a complete study.

```powershell
.venv/Scripts/python.exe scripts/cmu_robust_selection.py outputs/cmu_expanded_20260924/index.json outputs/cmu_stability_20260927/h100_seed42 outputs/cmu_noise_training_20261003_v2/model outputs/cmu_robust_selection_NEW
# Optional: prepare only, then run the frozen configs without overwriting results.
# Use the same arguments plus --prepare-only, then --run-prepared.

.venv/Scripts/python.exe main.py select-fixed-comparator outputs/cmu_robust_selection_20261007/selection_protocol.json outputs/cmu_robust_selection_20261007/selection_manifest.json --output outputs/robust_selection_recheck_NEW.json
```

The selector reads verified artifacts without importing Torch or loading models.
An existing output is never overwritten. It rejects test/train/unlabelled batches,
input/settings changes, incomplete mappings and declaration/hash inconsistencies.
The hashes alone cannot prove an unforgeable temporal preregistration; retain all
protocol/config files with the results. Selection does not automatically change
live defaults or export a final test comparator for every operating point.

## Status

All 252 cases completed; each of the three batches verified 84 runs / 673 files,
for 2,019 verified batch artifacts. `selection.json` selects **Kalman-CV** for this
protocol; it does not claim a three-metric superiority result or a test/live win.
Four `fixed_comparator_sigma*.json` files freeze the selected processor for the
four Gaussian operating points using the existing validation-comparator format.
The complete selection panel/provenance remains in `selection.json` and its manifest.

The only other eligible candidate, noise-trained MLP with tau 10 ms, has a worst
corrupted-profile ratio of 1.22375, versus baseline 1.0. Its noisy accuracy improves,
but the jitter tradeoff prevents selection. Tau 25 ms performs better on noisy
error/jitter while clean error rises about 11.7%, exceeding the frozen 10% constraint.

Macro means over three clips and three noise seeds, without participant-level CIs:

| Gaussian sigma | Kalman-CV: error mm / jitter m/s² | Noise MLP tau10 | Noise MLP tau25 |
| --- | ---: | ---: | ---: |
| 0 mm | 25.34 / 6.20 | 25.40 / 6.65 | 28.31 / 5.80 |
| 10 mm | 31.80 / 36.63 | 29.54 / 44.83 | 30.38 / 24.66 |
| 25 mm | 47.43 / 88.44 | 41.76 / 105.48 | 37.21 / 56.75 |
| 50 mm | 77.02 / 175.76 | 66.34 / 205.93 | 51.96 / 109.99 |

No candidate MLP has mask-matched latency on this whole panel. Even Kalman-CV lacks
a metric in some high-noise cells, so latency is not comparable there. Reduced
error/jitter is not called a latency victory. This establishes a conservative
fixed baseline for further work, not that an MLP is worse at every operating point.

Final regression verification: 632 Python tests and Ruff pass; the selection CLI
reproduced the archived selection report byte-for-byte. No old checkpoint or
production config was changed. The 2026-10-10 M8 continuation implements an
experimental four-checkpoint processor/policy with causal logs and two ablation
variants; its threshold calibration and final conclusions remain open. The original
252-run panel above is unchanged. See [adaptive work](adaptive_compensation.md).
