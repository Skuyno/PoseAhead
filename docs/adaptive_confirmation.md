# Frozen adaptive confirmation (2026-10-11)

Latest total: **360 new validation experiments / 2,892 verified batch artifacts**
across Gaussian (72), broader development stress (108), recovery (54) and SPEC
non-Gaussian stress (126). **862 Python tests and Ruff pass.** No new model training,
held-out scoring, live-default promotion, camera/Unity launch or network changes.

## Completed SPEC E8 non-Gaussian continuation

`scripts/cmu_spec_stress.py` froze and completed 126 cells / 1,011 verified artifacts
in `outputs/cmu_spec_stress_20261011`. It compares Kalman-CV and the unchanged
partial-recovery candidate, three seeds and three validation clips. Families are
applied separately, without added Gaussian noise: outlier probabilities .01/.05
with **fixed 150-mm displacement in uniformly random directions**, missing-joint
probabilities .05/.15, and delivered full-tracking loss for 100/250 ms every 3 s.
The clean condition is included. Missing-joint confidence is zero; tracking-loss
frames keep arrivals/IDs/times but invalidate every joint/confidence and remove
root/rotations. This is distinct from the earlier missing-arrival bursts.

| Profile | Kalman-CV error mm / jitter m/s2 | Partial recovery |
| --- | ---: | ---: |
| Fixed150 outlier 1% | 27.71 / 14.40 | 27.00 / 14.23 |
| Fixed150 outlier 5% | 36.27 / 44.37 | 34.30 / 42.32 |
| Missing joints 5% | 23.20 / 15.34 | 23.20 / 15.34 |
| Missing joints 15% | 23.42 / 24.21 | 23.42 / 24.21 |
| Full tracking loss 100 ms | 25.60 / 6.48 | 25.05 / 6.45 |
| Full tracking loss 250 ms | 26.23 / 6.64 | 25.65 / 6.61 |

Outlier error/jitter improve in all nine cells per level. At frequent missing
joints, the neural history never gains enough uninterrupted support and the
output is **exactly Kalman-CV** in this panel, not an invented neural improvement.
Clean metrics again exactly match the earlier recovery/Gaussian runs.

The complete-panel selector returns **`no_eligible_comparator` for both methods**,
not a fabricated winner: missing/gap profiles violate the unchanged minimum .95
pose/jitter coverage in at least one cell. Mean pose/jitter coverage is
94.79%/85.16% at missing5 and 84.66%/60.66% at missing15. Gap100 has mean jitter
coverage 95.10%, but its worst cell is 94.87%; gap250 means 91.89%/90.20%.
Jitter needs three valid successive samples, explaining stronger coverage loss
than pose error. Primary means exclude missing pairs, with coverage kept visible;
lower error on missing input must not be interpreted as improvement over clean
input or acceptance. Criteria/profiles were not relaxed or dropped.

The frozen M7 baseline stays Kalman-CV. No held-out/participant-level conclusion
or latency superiority follows. Gaussian recovery-vs-original equality was also
checked with real weights for 160 frames per sigma 0/10/25/50: positions, root,
alpha and target timestamps are exact matches (640 paired anchors).

## Offline detector-recording compatibility

`scripts/smoke_processor_replay.py` processes a saved recording without camera,
Unity or sockets; it checks UDP encode/decode in memory, preserves event clocks
and writes a separately derived NPZ plus a processor-only timing report. It does
**not** score camera observations against themselves as ground truth.

The retained M1 MeTRAbs recording (`outputs/fixtures/m1_raw_recording.npz`,
`m1-power-jump`, 60 valid detector frames, 2.529 s source span) runs successfully
through the partial-recovery config. Output is 60 valid **Kalman-only** packets;
neural probes are available but the paired observation-consistency guard rejects
their contribution. At the last anchor, h100 neural consistency error is 10.24 mm
versus baseline 7.32 mm, **not verified ground-truth accuracy**. Four anchors are
warmup, 56 request no supported forecast. CPU process-call p95 is 5.22 ms, excluding
MeTRAbs/capture/render. This tiny archived video-detector smoke is not a fresh
live-person run or proof of real-camera neural benefit. Output:
`outputs/m1_adaptive_recovery_smoke_20261011`.

Source ZIP snapshots retain Python package/scripts/processing configs and dependency
declarations, not recordings/checkpoints/environments. Their manifests verify every
file and the exact same Python-package digest used by experiment environments.
`scripts/source_snapshot.py OUTPUT.zip` creates a new archive; `--verify` checks it
without extraction. Earlier-source and SPEC-era snapshots remain separately retained;
they must not be relabelled as the source of a study with a different digest.

## Completed confirmation results

All **180 cells / 1,446 batch artifacts** completed and verified (Gaussian
579 files; broader stress 867 files). The original selected configuration was
unchanged throughout these panels. Gaussian confirmation retains `frozen_adaptive`
with worst corrupted error/jitter ratio **0.956031**. This averages three noise
seeds and three clips, but is not an independent-subject or held-out result.

| Profile | Kalman-CV error mm / jitter m/s2 | Frozen adaptive |
| --- | ---: | ---: |
| sigma0 | 25.34 / 6.20 | 24.79 / 6.17 |
| sigma10 | 31.80 / 36.63 | 30.40 / 34.39 |
| sigma25 | 47.43 / 88.44 | 43.42 / 80.04 |
| sigma50 | 77.02 / 175.76 | 68.73 / 156.99 |
| sigma25 + .002 dropout | 48.16 / 91.57 | 56.37 / 138.02 |
| sigma25 + .01 outlier, sigma200 | 51.74 / 100.50 | 47.09 / 90.83 |
| sigma25 + 150-ms missing arrivals | 48.10 / 86.86 | 45.58 / 83.37 |
| sigma25 + 300-ms missing arrivals | 50.31 / 85.98 | 46.79 / 79.10 |
| sigma25 + source20Hz | 63.83 / 101.94 | 59.16 / 97.73 |

Gaussian error and jitter improve in all 27 noisy paired cells; clean improvement
in both occurs in two of three clips (six cells due repeated seeds). The broader
panel selects **Kalman-CV**, not the adaptive candidate: dropout increases error
by 17.04% and jitter by 50.73%, making its worst ratio **1.507324**. All nine
dropout cells are worse in both metrics. Other corrupted families improve both
in all nine cells. Overall broad robustness must therefore not be claimed.

No whole-panel latency comparison passes the full-mask rule. Posthoc common
window diagnostics find clean delta -0.25 ms and sigma10 delta +0.77 ms. At sigma25
only eight of nine cells have any common eligible window; at sigma50 six of nine.
Incomplete macro delay stays null. These data do not establish reduced latency.
The CPU-only neural processing p95 macro is about 5.0-5.6 ms in this run; the
maximum cell p95 is 6.85 ms. This excludes detector/capture/render and is not live
timing acceptance. `advantage_claim` stays `not_assessed` throughout.

### Partial-tracking correction

The dropout regression exposed a mechanism problem: whole-body neural-support
reset also reset **every Kalman joint/root**, even when just one joint was missing.
The original variant is retained and its negative results remain archived.
New optional `preserve_partial_baseline: true` requires Kalman fallback. A partial
frame clears all neural/error/transition support but allows Kalman to process
usable joints independently. Invalid/low-confidence joints remain invalid and
are not filled. All-invalid/all-low-confidence observations and arrival gaps
over the configured limit still clear the baseline and immediately return Raw.
A separate arrival clock detects a gap even after a partial frame cleared the
neural tracker's history. This correction is not a new live default.

`scripts/cmu_adaptive_recovery.py` predeclares **54 corrective-validation cells**:
Kalman-CV, the unchanged original adaptive candidate and the partial-recovery
variant; clean/dropout profiles, three seeds and three clips. Only the baseline
recovery flag differs; model weights/history/gates/alpha limits are not tuned.
Output is `outputs/cmu_adaptive_recovery_20261011`. Independent/larger data and
camera-calibrated confidence/tracking still need additional study; the separate
completed official E8 continuation is reported above.

The correction completed **54 cells / 435 verified batch artifacts** and selects
`partial_recovery`, worst dropout ratio **0.980784** against Kalman's 1.0 and the
original adaptive 1.507324. The original variant's means exactly reproduce the
180-cell study, confirming that zeroing missing confidence did not manufacture
the improvement. Clean metrics are exactly unchanged for the corrected branch.

| sigma25 + .002 dropout | Error mm | Jitter m/s2 |
| --- | ---: | ---: |
| Kalman-CV | 48.16 | 91.57 |
| Original adaptive | 56.37 | 138.02 |
| Partial recovery | 47.24 | 89.64 |

Corrected error/jitter improve by **1.92%/2.11%** against Kalman-CV. This narrow
corrective validation is not a repeat of the full broad panel and cannot establish
robustness at severe 5%/15% dropout. The local example
`configs/processing/adaptive_kalman_recovery_pilot.json` exactly matches the selected
parameters and is experimental/custom-JSON only, not a live/default promotion.
The old example remains unchanged for reproducing earlier failures. Neural
forecast support still restarts after partial frames; only the independent
baseline state is preserved. No new model weights were trained.

The generator now zeros missing-joint confidence, as SPEC requires. Its RNG,
positions and validity are unchanged. Replaying the original dropout adaptive
case reproduces archived pose/target/jitter/latency/output-decision metrics exactly
in `outputs/cmu_missing_confidence_compatibility_20261011`; per-frame confidence
and invalid-frame feature logs correctly differ. Earlier artifacts are not edited.

## Train-only motion classification foundation

`scripts/cmu_motion_classes.py` freezes the SPEC E2 30th/70th training quantiles
using **160 valid training windows**. The frozen thresholds are **4.775371 / 12.540987
m/s2** in `outputs/cmu_motion_classes_20261011.json`. The fixed filter is a centered
Gaussian sigma25ms, support +/-3 samples, edge-value padding on an exact 40-Hz GT
grid. Two `numpy.gradient(edge_order=2)` operations yield acceleration. Window score
is the 95th percentile of all 40 x 16 non-pelvis acceleration magnitudes. Training
quantiles pool nonoverlapping windows, not independent participants. Invalid
filter/differentiation support makes a class unknown rather than interpolating it.

Filtering occurs on the whole reference before choosing the display-window phase.
Validation starts at 500 ms, matching the frozen experiments. Labels use **clean
reference only**, never model errors or corrupted observations. Scores below/equal
the lower threshold are smooth, scores above/equal the upper are high-dynamic/abrupt,
and the interior is medium. Equal-threshold ties prefer smooth.

| Validation clip | Smooth | Medium | High-dynamic/abrupt |
| --- | ---: | ---: | ---: |
| CMU13/01 | 18 | 0 | 0 |
| CMU13/11 | 1 | 0 | 1 |
| CMU13/29 | 6 | 27 | 4 |

`outputs/cmu_motion_labels_validation_20261011.json` retains all window identities.
Labels/calibration are reconstructed from pinned reference/train recordings before
stratification. `scripts/analyze_adaptive_motion.py` emits a descriptive class report
in the Gaussian study's `motion_classes.json`. Noisy profile means improve in
error/jitter in all three strata. On clean medium motion, jitter slightly rises
4.31 -> 4.35 m/s2; high-dynamic error drops 69.70 -> 67.40 mm. These are exploratory
strata with different clip composition: medium has one clip, high-dynamic two,
smooth three. Five high-dynamic windows are not independent people. Window-level
jitter omits cross-window derivative triplets and is explicitly distinguished
from whole-sequence jitter. No class-specific winner or held-out CI is claimed.

```powershell
.venv/Scripts/python.exe scripts/cmu_motion_classes.py freeze `
  outputs/cmu_expanded_20260924/index.json outputs/cmu_motion_classes_NEW.json
.venv/Scripts/python.exe scripts/cmu_motion_classes.py label-validation `
  outputs/cmu_motion_classes_NEW.json outputs/cmu_motion_labels_validation_NEW.json
.venv/Scripts/python.exe scripts/analyze_adaptive_motion.py `
  outputs/cmu_adaptive_confirmation_gaussian_20261011 `
  outputs/cmu_motion_classes_20261011.json outputs/cmu_motion_labels_validation_20261011.json `
  outputs/motion_report_NEW.json
```

## Scope fixed before scores

The previous 72-cell development pilot selected
`kalman_noisy_h100_tau25_cap25`. This continuation compares **only that frozen
configuration and Kalman-CV**, not another hyperparameter sweep. The script
verifies the archived pilot's complete artifact panel, reproduces its selection
report, normalizes the local processor config and requires exact agreement with
the previously selected parameters. Model manifests remain SHA-256 pinned.

No training, checkpoint changes, test scoring, camera/Unity operation or live
default promotion is performed. The fixed M7 comparison baseline remains
Kalman-CV. The config is still experimental. Seeds 42/43/44 are noise realizations,
not independent people; seed 42 and the clean clips were already consulted.

Both protocols and complete processor/profile/grid declarations are written
before their experiments run. Outputs are new, never overwritten:

- `outputs/cmu_adaptive_confirmation_gaussian_20261011`: 72 cells,
  2 methods x 4 profiles x 3 noise seeds x 3 validation clips.
- `outputs/cmu_adaptive_confirmation_stress_20261011`: 108 cells,
  2 methods x 6 profiles x 3 noise seeds x 3 validation clips.

The unchanged clips are CMU 13/01, 13/11 and 13/29. Source is 120 Hz,
injected arrival delay 100 ms, display scoring 40 Hz, nonoverlapping latency
windows 40 display frames, and evaluation starts at 500 ms. References are clean,
fully retained recordings; no future pose is supplied to a processor.

## Profiles

The Gaussian panel keeps sigma 0/10/25/50 mm. The broader panel uses:

| ID | Observation corruption |
| --- | --- |
| `sigma0` | Unchanged input |
| `sigma25_dropout002` | 25-mm Gaussian; independent per-joint dropout probability .002 |
| `sigma25_outlier01_200` | 25-mm Gaussian; per-joint outlier probability .01, extra Gaussian sigma 200 mm |
| `sigma25_gap150` | 25-mm Gaussian; 150-ms missing-arrival burst every 3000 ms |
| `sigma25_gap300` | 25-mm Gaussian; 300-ms missing-arrival burst every 3000 ms |
| `sigma25_stride6` | 25-mm Gaussian; retain every sixth observation (120 -> 20 Hz) |

These are synthetic stress tests, not an estimated DroidCam/MeTRAbs error model.
They are a development extension, not replacements for the fixed SPEC E8 levels.
The separate completed SPEC continuation above tests fixed 150-mm outliers,
5%/15% missing-joint levels and 100/250-ms delivered tracking loss.
Outliers perturb XYZ of valid non-pelvis joints and deliberately do not lower
confidence. Pelvis/root translation are not noise-corrupted. Dropout may affect
any joint; its validity is cleared and position zeroed. Existing invalid joints
are never repaired, and confidence values are not fabricated.
These archived 180 cells kept legacy missing-joint confidence values; the subsequent
compatibility correction now zeros them. Missing arrivals are not the same stress
family as delivered frames with all joints invalid and zero confidence.

Missing-arrival intervals are half-open, anchored to the first source timestamp;
the first begins only after one full period. Stride retains the first and every
Nth original observation. IDs and source/produced timestamps are not compacted.
The simulator holds the latest available output during a gap; it does not invoke
the processor on fake frames or continuously extrapolate to display time.

The original base Gaussian/dropout RNG is exactly preserved. Draws are consumed
for every original frame, even omitted arrivals, so retained anchors have the
same base corruption as the full-rate input. Outliers use a separate seeded RNG.
Every actual observation is saved in `observations_used.npz` for audit/replay.
The extended fields participate in batch comparison keys, selector profile
identity, input fingerprints and the predeclared artifact check. Disabled
extensions normalize away, preserving legacy experiment/config semantics.

## Decisions and delay diagnostics

The original complete-panel rule is unchanged: at least .95 pose/jitter coverage
and identical pose masks in every paired cell; clean mean pose error and jitter
may grow by no more than 10%; rank by the worst corrupted profile's maximum
macro error/jitter ratio to Kalman-CV, with baseline preferred on an exact tie.
Clip/noise-seed means have equal weight. Repeated seeds do not inflate participant
sample size. A failure on broader conditions must be reported, not hidden by
dropping a profile or retuning the frozen candidate.

The primary selector reports latency only when every actual eligibility mask
matches and metrics exist. `scripts/analyze_adaptive_confirmation.py` additionally
provides **posthoc descriptive common-window diagnostics**. It pairs exact
(time-window, joint) identities that are eligible in both outputs, reports
baseline-only/candidate-only/neither/common counts and the two medians on that
common subset. Grid and latency thresholds must match. Cells with no shared
window stay missing; an incomplete nine-cell macro is not silently averaged.

The common subset changes the estimand and can be selection-biased. It cannot
override the full-mask gate, establish a 20-ms advantage or provide a held-out
three-metric confidence-interval claim. CPU processing-call timing is reported
separately: it is not added to injected delay and is not capture-to-screen latency.

## Reproduction

```powershell
.venv/Scripts/python.exe scripts/cmu_adaptive_confirmation.py `
  outputs/cmu_expanded_20260924/index.json `
  configs/processing/adaptive_kalman_noisy_pilot.json `
  outputs/cmu_adaptive_baseline_20261011/selection.json `
  outputs/cmu_adaptive_confirmation_gaussian_20261011 --prepare-only

# On a previously prepared NEW directory, use --run-prepared instead.
# For the 108-cell broader panel, add --panel stress and use its own output path.

.venv/Scripts/python.exe scripts/analyze_adaptive_confirmation.py `
  outputs/cmu_adaptive_confirmation_gaussian_20261011 `
  outputs/cmu_adaptive_confirmation_gaussian_20261011/diagnostics.json
```

Experiment references/configuration resolve locally. Generated checkpoints and
CMU recordings must be retained; this is not a clean-machine download command.
Artifacts from previous studies are not rewritten to match newer implementation.

Next gates remain independent/larger validation data, latency-focused comparison,
a frozen held-out study and real-camera performance/accuracy. The earlier ten-minute camera run
did not meet
the 25-valid-FPS / p95-processing-50-ms targets; this offline confirmation does
not close those live gates.
