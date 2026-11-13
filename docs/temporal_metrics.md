# Temporal metrics: initial M6 implementation

The API lives in `poseahead.evaluation.temporal`. `MotionSeries` contains immutable
positions in metres, boolean validity, strictly increasing integer nanosecond
timestamps and an explicit clock domain. Output and reference must share the same
time grid and joint layout. No implicit interpolation or clock conversion occurs.

These metrics use **evaluation/display time**, not the timestamps of the original
camera observation or the prediction target. Otherwise a delayed observation
could be relabelled with its old source time and incorrectly appear delay-free.
`evaluation.snapshots.snapshot_motion` preserves simulation query times and marks
not-yet-available outputs invalid. It uses pelvis-relative positions, not root
translation. Ground-truth reference data never enters the causal processor.

## Primary jitter

`score_jitter(output, reference)` computes the mean Euclidean acceleration error
in m/s². For consecutive intervals h0 and h1 in seconds:

```text
v0 = (p1 - p0) / h0
v1 = (p2 - p1) / h1
a1 = 2 * (v1 - v0) / (h0 + h1)
primary = mean(||a_output - a_reference||)
```

This three-point estimate supports nonuniform sampling and is exact for quadratic
motion. All three samples must be valid in both streams. Eligible joint triples
are equally weighted. Plain output acceleration is reported as a secondary measure
on the same eligible triples; freezing a moving avatar cannot obtain zero primary
error merely by eliminating its motion. Empty coverage returns null. Sampling
rate and any upstream interpolation must accompany experiment reports, since
derivative errors depend on sampling. This is an acceleration-residual definition,
not a frequency-filtered perceptual jitter measure.

## Effective latency

`score_effective_latency(output, reference)` scores one preselected window on a
uniform grid. The [experiment runner](experiments.md) divides a configured grid into
fixed non-overlapping windows and aggregates eligible joint-window lags. Automatic
motion-based window selection and cross-sequence aggregation are not implemented.

1. Differentiate positions using the physical sample interval.
2. For each joint, subtract per-axis velocity means within each compared segment.
3. Compute normalized vector-velocity cross-correlation for integer lags within
   the configured range. Use a fixed central output segment and equally sized
   shifted reference segments, so overlap length does not favour large lags.
4. Refine an eligible interior peak by parabolic interpolation, clipped to half
   a sample. Positive lag means output trails reference; negative lag means lead.
5. Report median lag across eligible joints and each joint's result/rejection reason.

Defaults are ±200 ms search, minimum RMS demeaned velocity 0.0001 m/s, minimum
correlation 0.6, and competing-peak margin 0.02. Actual search bounds are rounded
down to the grid and reported. These are provisional, configurable validation
parameters, **not thresholds tuned on Human3.6M**.

Full-window missing data, stationary/constant-velocity motion, weak correlation,
boundary maxima, and separated competing peaks are excluded. Competing local
peaks at least three samples apart (including search boundaries) detect ambiguous
repetitions within the search interval. This heuristic does not prove that all
periodic or nearly linear motion is identifiable: periods outside the search
range and broad peaks require careful window selection and validation. A short
window with fewer than five central velocity samples is rejected outright.

Every result includes eligible/total joint windows and coverage. No eligible
windows means null latency, not 0 ms. The fixed pelvis of a root-relative skeleton
is naturally ineligible. Do not compare methods solely by their median while
ignoring different coverage or rejected windows.

## Simulation example

```python
from poseahead.evaluation import score_effective_latency, score_jitter
from poseahead.evaluation.snapshots import snapshot_motion
from poseahead.processing.config import create_processor
from poseahead.processing.simulation import simulate_processing

# `observations` and `ground_truth` must share a clock and camera geometry.
# Choose a uniform evaluation grid and prepare the reference on that exact grid.
snapshots = list(simulate_processing(
    observations, create_processor("kalman_cv"), evaluation_times_ns,
    delay_ns=100_000_000,
))
output = snapshot_motion(snapshots)
print(score_jitter(output, reference_motion))
print(score_effective_latency(output, reference_motion))
```

For latency, explicitly select a valid moving window after initial unavailability;
do not silently remove isolated missing samples or compress time. A replay with
injected 100 ms delay is tested end-to-end against the undelayed synthetic motion.
Analytic tests additionally cover −66/0/33/100/150 ms shifts, periodic/static
rejection, invalid samples and quadratic motion on irregular timestamps.

The existing `score-recording` CLI still computes pose/root error at declared
targets. Temporal APIs are separate intentionally: a derived recording alone does
not specify when each prediction was displayed. The `run-experiment` command now
supplies the display-time grid and combines these metrics. Dataset-wide orchestration
and real-data validation remain unfinished.
