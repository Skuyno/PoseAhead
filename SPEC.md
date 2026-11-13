# Codex Implementation Specification

## Project: Predictive Latency Compensation for a Video-Driven Full-Body Avatar

This repository supports a university NIR that will continue into a bachelor VKR.

The goal is NOT to build a production avatar SDK.

The goal is to build a reproducible research platform and a sufficiently stable Unity demo for studying predictive latency compensation.

---

# 1. Primary Research Goal

Implement and evaluate:

```text
noisy monocular RGB pose input
    ->
full-body pose
    ->
selected temporal method
(raw / filter / fixed prediction / adaptive compensation)
    ->
Unity humanoid avatar
```

The final VKR must answer:

```text
Can prediction reduce effective avatar latency
without unacceptable growth of pose error and jitter?

Does adaptive compensation outperform fixed prediction
for some motion classes / operating points?
```

---

# 2. Scope Policy

Treat the project as two layers.

## Research core — highest priority

Implemented mainly in Python:

- pose representation;
- recording/replay;
- filtering;
- latency simulation;
- prediction;
- adaptive logic;
- experiments;
- metrics;
- plots;
- reproducibility.

## Demo/runtime layer — supporting priority

Implemented in Unity:

- receiving pose frames;
- retargeting;
- humanoid animation;
- debug telemetry;
- A/B visualization.

Do not move research logic into Unity.

Do not turn this into a large game-development project.

---

# 3. Mandatory vs Optional Scope

Always prioritize `MUST` items.

Do not implement `COULD` items while any important `MUST` item is incomplete.

## MUST

- full-body webcam pipeline;
- canonical PoseFrame;
- 3D pose representation;
- Unity humanoid runtime;
- versioned binary UDP transport;
- recording and replay;
- artificial latency;
- Raw baseline;
- EMA;
- One-Euro Filter;
- Kalman filtering;
- Kalman extrapolation;
- fixed-horizon learned predictor;
- adaptive compensation;
- MPJPE;
- jitter metric;
- effective latency metric;
- at least one public benchmark dataset with ground truth;
- offline experiment runner;
- config-driven experiments;
- adaptive-vs-fixed experiment;
- motion-class analysis with a predefined classification rule;
- at least one ablation;
- reproducible outputs;
- live Unity demo.

## SHOULD

- detector-residual-based pose-observation noise model;
- AMASS as a second public dataset;
- latency-aware learned comparison baseline;
- external high-speed-camera latency measurement;
- two-avatar A/B demo;
- uncertainty-based predictability signal.

## COULD

- user perception study;
- custom neural architecture;
- complex IK;
- foot locking;
- finger tracking;
- facial tracking;
- multi-camera fusion;
- multiple pose-estimation backends;
- VR support;
- advanced rendering.

---

# 4. Scientific Claim Constraints

Do not write documentation claiming novelty merely because the project uses:

- motion prediction for latency compensation;
- full-body prediction;
- arbitrary latency;
- latency-aware prediction;
- predictability-aware prediction in general.

Those directions already have prior work.

Use cautious wording.

The intended experimental delta is the combination of:

```text
noisy monocular RGB input
+
full body
+
real-time avatar control
+
predictability-aware adaptation
+
joint latency/jitter/accuracy evaluation
```

Do not automatically claim this combination is globally novel.

Phrase it as the target research setting unless the literature review explicitly supports a stronger claim.

---

# 5. Development Order

Follow this order unless repository state gives a strong reason not to.

```text
M1  inspect repository
M2  define PoseFrame + canonical skeleton
M3  camera + pose detector
M4  3D pose stage
M5  Python -> Unity transport
M6  Unity humanoid retargeting
M7  recording + replay
M8  Raw / EMA / One-Euro / Kalman
M9  artificial latency
M10 metrics + offline experiment runner
M11 fixed learned predictor
M12 adaptive method
M13 VKR experiments
M14 live demo and final documentation (A/B view optional)
```

Do not start M11 before M7-M10 are functional.

Do not start optional VKR extensions before the mandatory adaptive-vs-fixed benchmark works.

---

# 6. First Action

Before editing code:

1. inspect the repository;
2. identify reusable existing code;
3. identify current Python/package-management setup;
4. identify whether Unity project already exists;
5. inspect previous camera/avatar code if present;
6. document actual architecture in `docs/architecture.md`.

Do not perform a broad rewrite of existing working code without a concrete reason.

---

# 7. Recommended Architecture

```text
project/
|
+-- src/
|   +-- pose/
|   |   +-- detector/
|   |   +-- lifting/
|   |   +-- representation/
|   |
|   +-- temporal/
|   |   +-- raw/
|   |   +-- ema/
|   |   +-- one_euro/
|   |   +-- kalman/
|   |
|   +-- prediction/
|   |   +-- fixed/
|   |   +-- adaptive/
|   |   +-- models/
|   |
|   +-- runtime/
|   |   +-- camera/
|   |   +-- pipeline/
|   |   +-- transport/
|   |
|   +-- experiments/
|   |   +-- runners/
|   |   +-- metrics/
|   |   +-- plots/
|   |
|   +-- data/
|   |   +-- loaders/
|   |   +-- noise/
|   |
|   +-- recording/
|
+-- unity/
|   +-- AvatarRuntime/
|
+-- configs/
+-- tests/
+-- scripts/
+-- docs/
+-- outputs/
+-- README.md
```

Adapt to the repository if it already has a sensible layout.

---

# 8. Canonical PoseFrame

Implement an engine-independent structure.

Conceptual fields:

```python
PoseFrame:
    frame_id
    source_timestamp_ns
    produced_timestamp_ns

    root_position
    root_rotation

    joint_positions
    joint_rotations
    joint_confidence
    joint_validity

    metadata
```

`source_timestamp_ns` is the capture/event time and `produced_timestamp_ns` is
the time at which this version of the frame was produced. Both are monotonic
timestamps in one documented clock domain. Wall-clock time, if recorded, is
metadata only. Live and offline code must use event time rather than frame count.

The project default is a right-handed camera coordinate system in meters:
`+X` points right in the image, `+Y` points down, and `+Z` points away from the
camera. `joint_positions` are pelvis-relative and `pelvis == (0, 0, 0)`.
`root_position`, when valid, is the pelvis translation in the camera frame.
The Unity conversion is `(x, y, z) -> (x, -y, z)` followed by avatar-scale and
origin calibration.

Positions are mandatory. `root_position`, `root_rotation`, and joint rotations
may be absent when the monocular estimator cannot recover them reliably; missing
values are marked invalid rather than filled with zeros or identity rotations.
The pinned MeTRAbs PyTorch live implementation predicts absolute metric-scale
H36M-17 joints from the current RGB frame. Its pelvis becomes `root_position` and
all joint positions are made pelvis-relative. Label its root as
`estimated_monocular`, transmit the estimate flag, and never present inferred
depth/scale as a physical measurement or ground truth. Measured intrinsics may be
supplied; otherwise the configured field of view is an explicit assumption.
MeTRAbs has no per-joint confidence output, so detector confidence is carried only
as a labelled proxy. The removed floor/contact and body-length estimators are not
fallbacks. Running and airborne poses may be emitted but require real-motion and
ground-truth validation. See `docs/metrabs.md` for setup and limitations.
The initial pipeline predicts and transmits joint positions. Unity derives bone
rotations from parent-child directions and the avatar's calibrated T-pose.
Unobservable twist around a single bone axis is resolved from the rest pose or a
documented pole-vector rule and is not treated as measured ground truth.

Minimum canonical skeleton:

```text
pelvis
spine
chest
neck
head

left_shoulder
left_elbow
left_wrist

right_shoulder
right_elbow
right_wrist

left_hip
left_knee
left_ankle

right_hip
right_knee
right_ankle
```

Document in `docs/pose_format.md`:

- joint order;
- coordinate system;
- units;
- handedness;
- root;
- skeleton parent hierarchy and rest-pose convention;
- local/global transforms;
- quaternion ordering.

Add tests for:

- serialization;
- joint ordering;
- coordinate conversion.

---

# 9. Pose Estimation

Default live 3D backend:

```text
MeTRAbs EfficientNetV2-L PyTorch, pinned source and pretrained checkpoints
```

MeTRAbs predicts absolute 3D keypoints directly from the current RGB image and is
used only as the replaceable camera-observation backend. Keep the canonical
17-joint mapping and explicitly document its detector-confidence proxy. Do not
enable temporal refinement that uses future frames: hidden look-ahead would
invalidate latency experiments. MotionBERT and VideoPose3D may be used for offline
comparisons only after their temporal context is audited for causality.

Hide backend-specific output behind an interface.

The rest of the project must not import MeTRAbs-specific structures.

Expected detector output:

```text
frame_id
timestamp
2D joints
confidence
```

---

# 10. 3D Pose Stage

Make the 3D stage replaceable.

Research priority is NOT inventing a new pose-lifting model.

First implementation only needs to be sufficiently stable for:

- canonical PoseFrame output;
- offline evaluation;
- Unity control.

Keep all downstream algorithms independent of the chosen lifting backend.

---

# 11. Python -> Unity Transport

Use one versioned custom binary protocol over:

```text
UDP (little-endian, one PoseFrame per datagram)
```

OSC is not part of the initial implementation. Do not use TCP for the main
real-time stream unless proven necessary.

Packet metadata:

```text
frame_id
source_timestamp_ns
send_timestamp_ns
root transform
joint transforms
confidence
active algorithm
target_offset_ms
predictability score if available
```

Document the magic bytes, protocol version, flags, joint count/order, numeric
types, byte order, maximum packet size, and compatibility policy in
`docs/unity_protocol.md`. Keep each datagram at or below 1200 bytes; do not rely
on IP fragmentation. Python's `struct` and C# span/binary primitives are enough;
the protocol must not require a third-party OSC package.

Sender timestamps remain in the sender's monotonic clock domain. Unity records
receive/apply times in its local monotonic domain. One-way transport latency may
be obtained by subtraction only after the clock domains have been synchronized
or calibrated; otherwise report sender-side processing, receiver-side handling,
round-trip time, or an external end-to-end measurement separately.

Unity receiver must:

- ignore stale frames;
- tolerate packet loss;
- avoid unbounded queues;
- reject older out-of-order frames where appropriate.

---

# 12. Unity Responsibilities

Unity should only handle:

- packet receiving;
- coordinate conversion;
- humanoid bone mapping;
- retargeting;
- optional simple IK;
- rendering;
- telemetry;
- A/B display.

Unity must NOT contain:

- model training;
- dataset processing;
- experiment logic;
- prediction logic;
- adaptive logic;
- primary metrics.

Runtime method switching is therefore initiated in Python (for example through
a control channel) or selects among pose streams already computed by Python;
Unity must not reimplement the processors merely to support the demo UI.

---

# 13. Recording and Replay

Mandatory before neural prediction.

Record enough data to reproduce the pipeline offline.

Per-frame data should include:

```text
frame_id
source timestamp
raw 2D pose
confidence
3D pose
filtered pose
predicted pose
final output pose
algorithm metadata
processing timings
```

Store the timestamp/clock semantics and, when available, the ground-truth pose
and its timestamp. Raw detector/lifter observations must remain immutable;
derived filtered/predicted outputs belong to versioned run results so that a
recording can be reprocessed with different methods.

Prefer efficient structured storage:

- NPZ;
- Parquet;
- another justified format.

Avoid large JSON files unless data volume remains small.

Offline replay must use the same core temporal-processing implementations as live mode.

---

# 14. Temporal Processor Interface

All temporal methods should share a common interface.

Conceptual example:

```python
class PoseProcessor:
    def reset(self) -> None:
        ...

    def process(
        self,
        frame: PoseFrame,
        *,
        target_offset_ms: float = 0.0,
    ) -> PoseFrame:
        ...
```

Change the exact API if needed, but preserve:

- resettable state;
- explicit temporal target;
- compatibility with live and offline modes.

---

# 15. Baselines

Implement in this order:

1. Raw.
2. EMA.
3. One-Euro.
4. Kalman filter.
5. Kalman constant-velocity extrapolation.

Do not begin neural prediction until all are benchmarkable offline.

---

# 16. Artificial Latency

Required values:

```text
33
66
100
150 ms
```

`artificial_latency_ms` is injected after creation of the pose observation and
before the temporal processor. It models observation availability, not detector
runtime or display latency.

For latency `L` and evaluation/event time `t_eval`, the processor must not
receive observations with source time after:

```text
t_eval - L
```

This rule must be implemented using timestamps, including for variable frame
rates and dropped frames. Ground truth at an exact target time may be
interpolated only by the evaluator and must never enter the processor.

Do not conflate `artificial_latency_ms` with `target_offset_ms`: the former
controls when an observation becomes available; the latter is the prediction
target relative to the newest available observation. A method may choose an
offset that compensates all, part, or none of the delay.

Write an explicit unit/integration test proving that future-frame leakage does not occur.

This is a research-critical correctness requirement.

---

# 17. Fixed Learned Predictor

After the offline benchmark exists, implement one compact predictor architecture
with four separately trained checkpoints, one for each required horizon.

Requirements:

- lightweight enough for real-time use;
- no unnecessary architectural novelty;
- selects the checkpoint matching `target_offset_ms`;
- works in live and offline modes;
- can be benchmarked against Kalman extrapolation.

Use identical architecture, input history, preprocessing, training protocol,
and seeds for all horizons. The supported learned offsets are exactly
`33/66/100/150 ms`; `0 ms` means no predictor. An unsupported offset must be
rejected explicitly. The adaptive method selects among these checkpoints;
continuous interpolation is optional and must be evaluated separately.

Required evaluation offsets:

```text
33
66
100
150 ms
```

A compact MLP-style predictor inspired by established human-motion baselines is acceptable.

Do not optimize for SOTA before correctness and experiment reproducibility are established.

---

# 18. Adaptive Method

This is the central VKR contribution.

Implement an adaptive strategy using one or more predictability signals.

Possible signals:

```text
recent prediction error
velocity
acceleration
jerk
direction changes
detector confidence
model uncertainty
ensemble disagreement
```

Every signal must be causally available at decision time. In particular,
"recent prediction error" may use only predictions whose target observations
have already arrived; it must not use current or future ground truth.

Possible decisions:

```text
select prediction horizon
adjust prediction strength
blend prediction with measurement
```

Conceptual blending:

```text
P_out = alpha * P_predicted + (1 - alpha) * P_measured
```

Both terms must have explicit pose timestamps. If they represent different
times, the method must describe this as deliberate partial compensation; it
must not present the blend as two estimates of the same state. Blend positions
linearly, but use normalized quaternion interpolation (for example SLERP) for
rotations rather than component-wise averaging.

Do not hard-code this as the only supported strategy.

Create an interface that allows at least two adaptive variants for ablation if practical.

---

# 19. Offline Experiment Runner

Required conceptual flow:

```text
sequence
    ->
optional noise
    ->
artificial latency
    ->
processor
    ->
output
    ->
metrics
    ->
plots
```

All processors in a comparison must receive exactly the same causally available
observation stream and be evaluated against ground truth at the same target
times. Hyperparameters are selected on training/validation data, never on the
reported test sequences.

Config must specify:

```text
dataset
sequence
processor
processor parameters
prediction model
target_offset_ms
artificial_latency_ms
noise config
random seed
output directory
```

Each run must save the resolved config.

---

# 20. Metrics

Metrics must use physical time from timestamps. If a metric requires uniform
sampling, resample explicitly and record the rate and interpolation method.

Implement independent of Unity.

## MPJPE

Primary pose error.

Report the evaluation protocol explicitly. At minimum distinguish root-relative
MPJPE from global/root translation error. Do not mix raw MPJPE, scale-aligned
MPJPE, and Procrustes-aligned MPJPE under one label. Map every dataset to the
canonical skeleton before scoring and evaluate the prediction against ground
truth at its declared target timestamp.

## Jitter

Implement and document at least one:

```text
mean acceleration
or
jerk
or
high-frequency energy
```

Plain output acceleration/jerk also measures intentional fast motion and can
reward an over-smoothed but inaccurate method. The primary jitter score should
therefore be an error relative to ground-truth derivatives or a documented
high-frequency residual, with plain acceleration/jerk reported only as a
secondary smoothness measure.

## Effective Latency

Initial method:

```text
cross-correlation lag
```

Return milliseconds.

Define signal preprocessing, eligible moving windows, lag search range, sign
convention, sub-frame interpolation, and aggregation across joints/sequences.
Cross-correlation is unreliable for static or strongly periodic windows; mark
invalid windows and report coverage. Validate the sign and magnitude on
synthetic sequences with known delay.

## Runtime Timing

Track:

```text
capture
detector
3D stage
temporal processor
transport
Unity receive/apply if available
total
```

Use monotonic high-resolution timing.

For this project, the quantitative live acceptance criteria on the reference
machine are:

- **real-time:** at least 25 valid output PoseFrames/s over a 10-minute run,
  with 30 FPS as the target;
- **processing budget:** p95 from captured frame availability to UDP send no
  greater than 50 ms, excluding configured artificial latency;
- **stable:** no crash, NaN/Inf pose, or ever-growing queue; latest-frame queue
  depth at most one; at least 95% of captured frames either produce a valid pose
  or are explicitly counted as dropped/invalid.

Reference machine for the initial implementation: Windows 11 x64, AMD Ryzen 7
5700X, NVIDIA GeForce RTX 3070 Ti 8 GB, and 16 GB RAM. Every published run still
records actual CPU, GPU, RAM, OS, driver, and package versions.

---

# 21. Required VKR Experiments

Do not consider the VKR evaluation complete until these are possible.

## E1 — Horizon Sweep

For fixed predictor:

```text
33 / 66 / 100 / 150 ms
```

Measure:

```text
MPJPE
jitter
effective latency
```

State which artificial latency is held fixed for each sweep. At minimum report
the no-prediction offset (`0 ms`) and the matched condition
`target_offset_ms = artificial_latency_ms` for every required latency. A larger
offset/latency grid is preferable when compute permits.

## E2 — Motion Class Analysis

Use 1-second windows and a motion score equal to the 95th percentile, over
non-root joints, of ground-truth root-relative acceleration magnitude. Smooth
the ground-truth trajectory with one fixed documented low-pass filter before
differentiation. Compute score thresholds on the training split only:

```text
smooth: score <= training 30th percentile
medium: training 30th < score < training 70th percentile
high-dynamic/abrupt: score >= training 70th percentile
```

Freeze the thresholds before evaluation. Call the class "high-dynamic/abrupt",
not "unpredictable": abruptness is observable, while unpredictability depends on
a model. Do not classify a test segment using the error of the method being
compared.

## E3 — Baseline Comparison

Compare:

```text
Raw
EMA
One-Euro
Kalman
Kalman extrapolation
Fixed learned prediction
Adaptive method
```

## E4 — Adaptive vs Fixed

Main research experiment.

Choose the fixed comparator and all thresholds on validation data. Compare on
identical held-out sequences and report per-sequence results plus aggregate
uncertainty (for example a subject/sequence-level confidence interval), not only
one pooled mean.

There is no single unqualified winner across three metrics. Report the Pareto
front. A method may be called better at an operating point only if either:

1. it is no worse in effective latency, root-relative MPJPE, and primary jitter
   within the 95% sequence-bootstrap confidence intervals, and is strictly
   better in at least one; or
2. it reduces effective latency by at least 20 ms while MPJPE and primary jitter
   each increase by no more than 10% relative to the validation-selected fixed
   comparator.

For this project, growth above 10% in either MPJPE or primary jitter is the
provisional meaning of "unacceptable". These thresholds may be changed after a
pilot on training/validation data, but must be frozen before opening test results.

## E5 — Ablation

At least one meaningful ablation.

Example:

```text
fixed alpha
vs
adaptive alpha
```

or:

```text
adaptive horizon only
vs
adaptive blending only
vs
combined
```

## E6 — Real Webcam Validation

Run the final method in the real camera-to-Unity pipeline.

Without synchronized ground truth this experiment demonstrates runtime
feasibility and qualitative behavior, but it must not be used as quantitative
evidence that one method is more accurate than another.

---

# 22. Datasets and Noise

Do not block initial development on all datasets.

Mandatory primary benchmark:

```text
Human3.6M
+
custom webcam recordings
```

Use the following predictor split before window generation:

```text
train: S1, S5, S6, S7
validation: S8
test: S9, S11
```

AMASS is a `SHOULD` second dataset, not a blocker for the first complete
benchmark.

Use two explicitly separated evaluation tracks:

- **motion-prediction track:** mocap/3D ground truth (for example AMASS), with
  synthetic *pose-observation* corruption if required;
- **camera-pipeline track:** synchronized RGB or detector output with matched 3D
  ground truth where available (for example Human3.6M), plus custom webcam data
  for live validation.

AMASS contains mocap-derived body motion rather than monocular RGB frames, so
joint corruption applied to AMASS must not be called "camera-image noise" and
cannot alone support a claim about an RGB detector. Record dataset licenses and
access instructions; do not commit restricted data to the repository.

For VKR, support optional deterministic noise injection:

```text
Gaussian noise
outliers
missing joints
short tracking gaps
confidence degradation
```

Noise must be deterministic from a seed.

The initial stress-test levels, applied one corruption family at a time in
addition to the clean condition, are:

```text
Gaussian joint sigma: 10 / 25 / 50 mm
outlier probability: 1% / 5%, displacement magnitude: 150 mm
missing-joint probability: 5% / 15% per frame
tracking-gap duration: 100 / 250 ms
```

Set confidence to zero for missing joints and gaps. Noise values may be adjusted
after measuring MeTRAbs residuals on training/validation data, but the final
levels must be frozen before test evaluation. Do not run the full Cartesian
product unless an experiment explicitly requires it.

Split learned-model data by subject and source sequence (and, for AMASS, by
source dataset where practical) before window generation. Save split manifests
and ensure overlapping windows from one sequence never cross train/validation/
test boundaries.

---

# 23. Live Demo

Target environment: Unity 6.3 LTS `6000.3.23f1` on Windows x64. Keep this patch
in `ProjectVersion.txt`; upgrades require an explicit compatibility run and must
not happen silently during the experiment phase.

Unity should support switching methods at runtime.

Suggested keys:

```text
1 Raw
2 EMA
3 One-Euro
4 Kalman
5 Fixed Prediction
6 Adaptive Prediction
```

Display:

```text
FPS
active method
artificial/measured latency
target_offset_ms
predictability score
alpha
processing time
```

Display injected observation latency, prediction offset, measured processing
latency, and estimated effective latency as separate values; a generic
"latency" label is not sufficient.

Two-avatar A/B mode is `SHOULD`, not `MUST`.

Do not delay core experiments to perfect this UI.

---

# 24. Reproducibility

Every experiment output should contain:

```text
resolved config
run ID
timestamp
git commit hash if available
environment/package versions
metrics
plots
logs
```

Use explicit random seeds.

Do not encode experiment settings only as source-code constants.

---

# 25. Tests

Mandatory high-value tests:

```text
PoseFrame serialization
joint ordering
coordinate transforms
packet serialization/deserialization
latency simulation
future-frame leakage prevention
processor reset
deterministic noise
offline replay consistency
metric sanity checks
```

Do not create superficial tests only to increase coverage.

---

# 26. Documentation

Maintain:

```text
README.md
docs/architecture.md
docs/pose_format.md
docs/experiments.md
docs/unity_protocol.md
```

README must explain:

```text
environment setup
camera pipeline
Unity demo
recording
replay
offline experiment execution
metric calculation
```

---

# 27. NIR Definition of Done

NIR is technically complete when:

- webcam pipeline works;
- full-body pose is extracted;
- PoseFrame exists;
- Unity avatar follows the user;
- recording/replay works;
- artificial latency works;
- baseline processors work;
- metrics can be computed;
- fixed predictor exists or has a functional prototype.

Adaptive prediction may still be incomplete at the NIR boundary.

---

# 28. VKR Definition of Done

The project is ready for VKR write-up when:

- [ ] stable live pipeline exists;
- [ ] offline benchmark exists;
- [ ] baselines run on the same sequences;
- [ ] fixed learned predictor works;
- [ ] adaptive method works;
- [ ] MPJPE is computed;
- [ ] jitter is computed;
- [ ] effective latency is computed;
- [ ] at least one public ground-truth benchmark dataset is evaluated;
- [ ] adaptive-vs-fixed experiment is complete;
- [ ] at least one ablation is complete;
- [ ] real webcam validation is complete;
- [ ] experiments are config-driven;
- [ ] outputs are reproducible;
- [ ] final plots/tables are generated;
- [ ] Unity live demo works.

---

# 29. Stop Conditions

Do NOT expand scope when the mandatory VKR benchmark is incomplete.

Do not implement:

```text
new pose-estimation architecture
new rendering stack
complex IK
foot locking
finger tracking
facial animation
multi-camera fusion
VR support
user study
custom SOTA predictor
```

unless:

1. all `MUST` items are working;
2. main adaptive-vs-fixed experiment has been run;
3. remaining time clearly permits the extension.

---

# 30. Engineering Priority

When trade-offs arise, follow:

```text
1 correctness
2 reproducibility
3 experiment validity
4 measurable latency
5 baseline quality
6 adaptive method
7 live demo
8 visual polish
```

If a beautiful Unity demo conflicts with finishing the benchmark, finish the benchmark.

If a sophisticated neural model conflicts with obtaining reliable baselines, finish the baselines.

If a refactor does not directly support a current milestone, postpone it.

---

# 31. Recommended Immediate Milestone

The next milestone should normally be:

```text
Webcam
    ->
Pose Detector
    ->
3D Pose
    ->
Canonical PoseFrame
    ->
UDP
    ->
Unity Humanoid
```

Acceptance criteria:

- documented startup commands;
- stable webcam input;
- continuous pose output;
- timestamp propagation;
- Unity receives live pose;
- avatar follows major body movement;
- basic latency telemetry exists;
- temporal processors can later be inserted before transport.

After this works, implement recording/replay and offline benchmarking before neural prediction.

---

# 32. Final Principle

This repository must remain both:

```text
a real-time avatar demo
```

and:

```text
a reproducible research benchmark
```

The VKR is successful because it experimentally answers a latency-compensation question, not merely because the avatar animates.
