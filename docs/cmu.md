# CMU Motion Capture: initial integration

## Training-only expansion (2026-09-24)

The `--expanded-training` option adds nine explicitly selected clips from the
existing training IDs (02/05/06), yielding 16 training, three validation and three
test clips. The additional motions are strikes, bending/lifting, washing, two
dance sequences and four basketball sequences. Selection uses source motion
descriptions, not test scores. This broadens motion coverage but is **not a balanced
cohort** and adds no new participants. The actor-alias limitation still applies.

The expanded corpus contains 40,326 original frames: 19,903 training, 7,320
validation and 13,103 test. The earlier training portion had 3,935 frames. At a
100-ms horizon this yields 19,359 overlapping training windows, versus 3,697 in
the previous native-history run; these are not independent statistical samples.

```powershell
.venv/Scripts/python.exe scripts/cmu_pilot_training.py outputs/cmu_expanded_NEW --native-history --expanded-training
.venv/Scripts/python.exe scripts/compare_cmu_validation.py outputs/cmu_expanded_NEW --reference-run outputs/cmu_native_history_20260923
```

The retained run directory is `outputs/cmu_expanded_20260924/`. `selection.json`
records the exact clip list and settings before training. Architecture, eight
epochs, seed 42 and native-history protocol remain unchanged. A changed training
corpus necessarily refits train-only normalization; validation never fits it.

The comparison now checks exact numerical equality of validation history, targets,
timestamps and anchor IDs against the earlier native-history run. Its per-sequence
errors and equally weighted sequence means complement the window-weighted mean,
so the longest exercise recording does not silently define the entire conclusion.
The comparison does not select on or score test data, and does not measure
effective latency or jitter. Final results require a completed pilot_report.json
and validation_comparison.json; incomplete directories are not completed runs.

All four expanded-data trainings completed and again selected epoch zero. The
selected models therefore retain hold-pose errors (10.30/20.37/30.52/44.88 mm).
This is evidence that this particular data expansion/settings combination did not
help this MLP on this validation set, not that neural prediction is impossible.
Motion-relative inputs, a velocity-based residual baseline and training shard order
are candidates for subsequent controlled experiments, not established causes.

## Native-rate history update (2026-09-23)

`scripts/cmu_pilot_training.py --native-history OUTPUT` preserves the source 120-Hz
recordings and creates 10-point histories at 20-ms spacing with causal linear-past
interpolation. The source clip is no longer decimated. Prediction anchors remain
at source-frame times (120 Hz), while each history covers 180 ms, not the old 450 ms.
The same fixed 13 clips, subject-ID split, seed and eight epochs were retained.

Artifacts: `outputs/cmu_native_history_20260923/`. The four models again selected
epoch zero (hold pose). Validation MPJPE was 10.30, 20.37, 30.52 and 44.88 mm for
33/66/100/150 ms. Training window counts are 3753/3725/3697/3655 respectively.
More overlapping windows do not mean more independent motions or participants.
Because history duration, anchors and target interpolation changed, this is not a
matched ablation against the earlier 20-Hz run. Neither run is a final benchmark.
Test data was not scored and live defaults were not changed. The next substantial
research need remains a larger, balanced training set and shared delay/noise tests.

Matched native-history validation diagnostics (33/66/100/150 ms): simple last-two-
sample constant velocity achieved 2.60/6.70/12.68/24.34 mm MPJPE. This only measures
future-pose error on clean mocap, not effective latency, jitter or webcam robustness.
The complete comparison is retained in `validation_comparison.json` in the run.

The sections below retain the earlier 20-Hz pilot and its reproducible results.

CMU is the immediately available motion source for predictor experiments;
Human3.6M access can remain pending. Integration now includes acquisition,
conversion and exploratory training, **not a validated production predictor**.

Official sources:

- [Database](https://mocap.cs.cmu.edu/)
- [Units, reuse permission, and bulk-download guidance](https://mocap.cs.cmu.edu/faqs.php)
- [ASF/AMC tools](https://mocap.cs.cmu.edu/tools.php)

## Reproduce a pilot import

From the repository root, using the project's Python environment:

```powershell
.venv/Scripts/python.exe main.py download-cmu 07 01 data/cmu/pilot/07_01
.venv/Scripts/python.exe main.py import-cmu data/cmu/pilot/07_01 data/cmu/pilot/07_01/recording.npz
```

Both commands refuse to overwrite their output. Download failures may leave an
incomplete directory without download.json; inspect it and use a fresh directory
for a retry. No automated whole-site crawling is implemented. For larger-scale
acquisition CMU explicitly provides a bulk ASF/AMC archive in its FAQ.

The downloader saves ASF, AMC, the subject page, URLs, SHA-256 hashes, retrieval
time and the sample rate extracted from the actual trial row. It does not assume
every trial is 120 Hz. The importer checks the source hashes before conversion.
Data stays in the ignored data/ directory, not in Git.

## Coordinate and landmark contract

- Supported input: degree-based fully specified AMC, contiguous frames from 1;
  CMU ASF XYZ axes, default zero root rest transform, rotational bone channels.
  Other conventions are rejected rather than guessed.
- Lengths and root translations multiply by `0.0254 / ASF.units.length` metres.
- Column-vector forward kinematics uses `Rparent @ C @ Rmotion @ C.T` and the
  bone direction/length to reconstruct each endpoint.
- ASF Y-up coordinates map through `diag(1,-1,-1)` into a **fixed virtual camera**.
  This is not a measured camera calibration, and there is no real-video alignment.
- Pelvis is the ASF root. Spine is the lowerback endpoint, neck the thorax endpoint,
  head the upperneck endpoint; chest is the spine/neck midpoint. Shoulders are
  clavicle endpoints, elbows humerus endpoints, wrists radius endpoints. Hips are
  hipjoint endpoints, knees femur endpoints, ankles tibia endpoints.
- Store pelvis-relative joint positions and a separate absolute root translation.
  Frame 1 maps to timestamp zero; subsequent timestamps use the documented rate.
- Confidence 1 means numerically valid reconstructed source data, not detector
  confidence and not a guarantee of perfect anatomical ground truth.

These landmarks are a canonical approximation, not an official CMU evaluation
skeleton or an exact anatomical match to MeTRAbs. No rotations are exported.

## Pilot status and next gates

The initial three clips remain importer smoke data. A separate 13-clip pilot now
contains 24,358 source frames from IDs 02, 05, 06, 13 and 14. The fixed exploratory
policy uses 02/05/06 for training, 13 for validation, 14 for test. Namespaced IDs
(`CMU:02`, etc.) cannot be mistaken for Human3.6M subjects. Training indexes and
window stores now support this CMU policy without changing existing H36M indexes.

**This is a source-ID split, not independently verified actor identity.** Numeric
rest-skeleton fingerprints reject duplicated calibrations across partitions, but
different calibrations do not prove different people. Actor alias investigation
and a balanced, larger cohort remain gates before final research claims.

Ordinary import-cmu output remains unlabelled. Explicit pilot preparation selects
every sixth 120-Hz frame, yielding exactly 20 Hz (50-ms steps) without interpolating
input poses or changing their event/availability timestamps. Histories have ten
frames, covering 450 ms. Prediction targets may interpolate reference poses at
33/66/100/150 ms. Decimation has no anti-alias filter, so this is a pipeline pilot,
not the final motion-frequency or live-camera training configuration.

Rotation order and hierarchy recurrence were checked against CMU's official
amc_viewer.r1806 (`Library/Skeleton.cpp`, `Character/pose_utils.cpp`). A separate
SciPy quaternion implementation agrees with the matrix implementation within
1.4e-15 m over all 2777 frames of the three original clips. This comparison shares
the ASF parser and does not independently verify decoding or anatomical landmarks.
Visual validation remains open. The report is at
`data/cmu/reference/kinematics_check.json`.

## Reproduce exploratory training

```powershell
.venv/Scripts/python.exe scripts/cmu_pilot_training.py outputs/cmu_pilot_NEW
.venv/Scripts/python.exe scripts/compare_cmu_validation.py outputs/cmu_pilot_NEW
```

The script downloads only its explicit 13-trial list, verifies existing cached
files, writes a new index and trains four CPU models for eight epochs (seed 42).
It checks source skeleton partitions before training. Test recordings are indexed
but never exported as training windows, scored, or used for model selection.

Retained current run: `outputs/cmu_pilot_20260923_v2/`. The earlier directory without
`_v2` is a development run before index fingerprint hardening; do not use it as the
current reproducible artifact. Neither run overwrote the other.

Validation target error, canonical17 MPJPE including zero pelvis, millimetres:

| Horizon | Hold last pose | Last-two-frame constant velocity | Selected MLP |
| --- | ---: | ---: | ---: |
| 33 ms | 10.28 | 3.58 | 10.28 |
| 66 ms | 20.33 | 8.25 | 20.33 |
| 100 ms | 30.63 | 14.52 | 30.63 |
| 150 ms | 45.06 | 26.94 | 45.06 |

Every MLP selected epoch zero: training did not beat holding the last pose on this
validation sample. Constant velocity is a promising diagnostic baseline, not a
declared final winner: this run does not evaluate jitter, effective latency, noise,
confidence intervals, camera-domain errors or multiple seeds. No model was promoted
to the live configuration. Next: broaden balanced train data, improve time-grid
handling, validate actor aliases and landmarks, then run shared delay/noise suites.
