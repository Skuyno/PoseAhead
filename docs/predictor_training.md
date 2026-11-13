# Fixed-horizon predictor: initial CPU implementation

Optional output baseline added 2026-10-03: `output_baseline` accepts `hold` (unchanged
default) or `constant_velocity`. The latter needs at least two history frames and
adds the network correction to `last + (last - previous) * horizon_ns / step_ns`.
The calculation uses the checkpoint's trained time grid; affine normalization
preserves this physical extrapolation. Epoch zero is then constant velocity, not
hold, and remains eligible for validation selection. Manifest version 3 explicitly
pins `output_baseline` and `input_features`; versions 1/2 retain hold semantics and
reject a contradictory velocity baseline. Root translation is still held separately.
This is an experimental architecture option, not an established improvement.

Optional training settings added 2026-09-27: `input_features` accepts `positions`
(legacy default) or `relative_last`; `batch_order` accepts `within_shards` (legacy)
or `global_windows` (bounded pilot cache). Relative-last checkpoints explicitly use
manifest version 2; legacy version 1 retains its original meaning. See
[CMU controlled experiment](cmu_model_ablation.md) for protocol, results and limits.

The first learned model is a small residual MLP. This is an executable training
prototype, not a validated architecture choice or completed M7 result. There are
no licensed Human3.6M-trained checkpoints yet. Learned offline processing and an
experimental causal live/UDP path are available; real-camera acceptance is pending.

To generate retained, explicitly synthetic data and a trained smoke checkpoint:

```powershell
poetry run python scripts/smoke_predictor_training.py outputs/predictor_smoke --horizon-ms 33
```

Use a new directory per run; `--horizon-ms` supports all four required offsets.
This test deliberately uses the same analytic linear law across synthetic split
fixtures. It validates software plumbing, not generalization across human subjects.

## Verified training data and normalization

```powershell
poetry run poseahead verify-windows outputs/windows/train_33
poetry run poseahead verify-windows outputs/windows/validation_33
poetry run poseahead fit-normalizer outputs/windows/train_33 outputs/normalizer_33.json
```

`WindowStore` streams one shard at a time, checks filename safety, hashes, numeric
schema, finite coordinates, zero pelvis, ordered unique anchors, causal uniform
history, target horizon and source clip bounds. Labels must match the fixed subject
split. Original recordings need not be present to read an exported shard set;
retain the dataset index and raw recordings separately for provenance.

Normalization uses train histories only, never their targets or validation/test
data. Streaming population mean/variance are computed per joint and axis in float64.
Repeated appearances in overlapping history windows receive equal weight; this is
window-weighted, not unique-frame or subject-weighted. The standard-deviation floor
defaults to 0.001 m. Pelvis mean/scale are forced to 0/1. The same frozen transform
is applied to input histories and targets and inverted after prediction.

The normalizer records its train-window manifest hash, dataset index hash, history,
sample interval and horizon. Training rejects a normalizer fitted on any other
train export. Validation must use the same dataset index and feature/time contract.
If any split is reindexed, regenerate compatible window exports; paths/hashes are
deliberately part of the provenance. Files are never overwritten.

## Training

Torch is optional for the base package. This path was checked with the locally
installed pinned MeTRAbs environment (`torch 2.1.0+cu121`), using CPU only; no new
package installation is performed by the training command.

Create a JSON config (relative paths resolve against its directory):

```json
{
  "version": 1,
  "train_windows": "windows/train_33",
  "validation_windows": "windows/validation_33",
  "normalizer": "normalizer_33.json",
  "seed": 42,
  "epochs": 50,
  "batch_size": 64,
  "hidden_size": 64,
  "learning_rate": 0.001
}
```

```powershell
poetry run poseahead train-predictor outputs/training_33.json outputs/model_33
```

Architecture: flatten the historical 16 non-pelvis joints, then two hidden linear
layers with ReLU and an output linear layer predicting a normalized displacement
from the last observed pose. The pelvis is always zero. Zero-initialized output
weights make epoch zero exactly the hold-last-pose baseline. Separate runs/checkpoints
are required for 33/66/100/150 ms; architecture/history/preprocessing choices must
remain identical for the planned controlled horizon comparison.

The above describes the default hold baseline; with `output_baseline:
constant_velocity`, the same network predicts a correction to extrapolation instead.

The CPU loop uses Adam, normalized non-pelvis MSE and gradient-norm clipping at 1.
By default it shuffles windows within each shard with a seeded CPU RNG, while shard
order stays fixed. Optional `global_windows` mixes all training windows each epoch
under a conservative 256-MiB pilot cache limit. Torch's caller CPU RNG state is restored and
the training path does not initialize CUDA or modify thread settings. CPU thread
configuration can still affect runtime and cross-environment numerical behaviour.

Validation reports physical unaligned canonical-17 MPJPE in mm, including the zero
pelvis. The best strictly improved validation score selects the checkpoint, with
epoch zero eligible; an unsuccessful training run may correctly retain the declared
hold or constant-velocity baseline.
Test data is neither an optimizer input nor a checkpoint-selection input. This
does not prevent a person from manually tuning hyperparameters using test results;
maintain the validation-only research protocol outside the tool as well.

Output consists of numeric-only `model.npz`, copied `normalizer.json`, and a final
`manifest.json` with resolved config, all epoch metrics, selected epoch, train/val
manifest hashes, weight/normalizer hashes, runtime library versions and hashes of
the relevant source files. No pickle model deserialization is used. Directories
are not reused; checkpoint/resume and optimizer-state persistence are not implemented.
Results are written after training, so interrupted training does not retain epochs.

## Checked inference

```python
from poseahead.prediction.checkpoint import FixedHorizonPredictor

predictor = FixedHorizonPredictor("outputs/model_33")
predicted_positions_m = predictor.predict(history_positions_m, history_timestamps_ns)
```

The history shape must be `(history_frames, 17, 3)`, with finite pelvis-relative
positions and a tuple of Python integer nanosecond timestamps on the trained grid.
Result shape is `(17, 3)` and its target time is the last history timestamp plus
`predictor.horizon_ms`. The loader verifies weight hashes, parameter names/shapes,
normalizer provenance and horizon compatibility. It preserves caller CPU RNG state.

This API predicts root-relative positions only. It does not predict root translation,
bone rotations, confidence or validity; missing observations must be handled before
invocation. It rejects irregular history rather than silently resampling or padding.
Do not connect a variable-rate webcam directly to this fixed-rate model without a
causal history adapter and explicit fallback policy.

Tests exercise synthetic labelled linear motion, improvement over hold, deterministic
repeat training/weights, checkpoint round-trip, tampering detection and test-split
rejection. Synthetic labels are test fixtures, not evidence that the actual dataset
has been imported or that the predictor generalizes. CMU validation training,
four-horizon/three-seed runs and shared delay/noise evaluation are now available;
see the dated CMU study documents. Final held-out test evaluation, camera acceptance,
training runtime budgets and cross-machine reproducibility remain pending.

## Shared processor and experiment runner

Use a processor config rather than a baseline-only `--method` flag:

```json
{
  "method": "learned",
  "parameters": {"checkpoint": "model_33"}
}
```

`checkpoint` resolves relative to the processor config or enclosing experiment config.
The runner expands it to an absolute path and pins the checkpoint manifest SHA-256.
The loader also checks the weight and normalizer hashes. Re-running a resolved
experiment rejects a changed checkpoint instead of silently using different weights.
Checkpoints remain external artifacts: archive them along with run outputs, because
an experiment directory does not copy the entire model.

`LearnedProcessor` defaults to strict `history_sampling: exact`: it keeps only the
required number of raw observations at the exact trained interval with fully valid,
positive-confidence joints. Predictions never enter its history. An incomplete
history produces raw `warmup` fallback. Invalid observations clear history;
irregular intervals restart it from the current valid observation in strict mode.
Fallbacks explicitly carry target offset zero, not the configured prediction horizon.
Root translation is held from the current observation; only root-relative joint
positions are predicted. Predicted poses clear rotations. Confidence is inherited
input confidence, not calibrated model uncertainty.

Each output logs `prediction_status`, actual target time/offset, configured horizon,
root policy and checkpoint parameters. Reports include per-observation status counts
and prediction fraction; metrics include fallbacks as actual outputs. This fraction
counts processed observations, not display queries holding the latest output.
Inference failures are raised without advancing history, not silently credited as
successful predictions.

After creating the smoke fixture at `outputs/experiments/predictor_smoke_20260922`:

```powershell
poetry run poseahead run-experiment configs/experiments/learned_synthetic33.json outputs/experiments/learned_replay
poetry run poseahead verify-experiment outputs/experiments/learned_replay
```

This example has initial missing display observations due to injected delay and two
history warm-up outputs. Its linear motion is unsuitable for correlation lag
estimation; null effective latency is expected, not a zero-latency success.
The same processor config works with `process-recording`, and experiment configs can
join split-checked batches. No real-data performance or predictive quality is
established here.

## Experimental variable-rate webcam path

Live learned processing requires explicit `history_sampling: linear_past` in its
processor config. Strict mode is rejected before camera/backend startup so a
variable-rate webcam cannot silently run a permanently warming predictor.

At every new valid observation, the adapter anchors the uniform history grid to
that observation's event time. Older grid points are interpolated between already
received observations only; the upper interpolation endpoint may be later than an
older grid point but never later than the current anchor. This reconstruction is
causal at the prediction time, not a claim that interpolated samples were available
at their historical timestamps. Nothing is extrapolated to fill history.

`max_observation_gap_ms` (default 100, configurable integer 1–1000) limits interpolation
across tracking gaps. Larger gaps reset the history. Invalid observations clear it.
The adapter retains at most one bracketing observation before the earliest required
grid point and has a hard `max_history_observations` capacity (default 4096). If the
cap prevents reconstruction it emits raw `history_capacity` fallback; it does not
grow an unbounded queue. These are explicit initial settings, not validation-tuned
tracking thresholds. Reconstructing 50 Hz input from a slower camera cannot recover
unobserved motion and requires real-data accuracy evaluation.

```powershell
# Requires the previously generated SYNTHETIC smoke checkpoint; diagnostic use only.
poetry run python scripts/benchmark_webcam.py --processor-config configs/processing/learned_synthetic33.json --send-udp --duration 30
```

UDP identifies actual predicted frames with the existing 33/66/100/150 ms algorithm
IDs and actual target offsets. Warm-up, invalid-history and tracking-gap fallbacks
use Raw with offset zero. The configured horizon is not falsely credited to fallback
frames. Live JSON reports include pinned checkpoint parameters and prediction-status
counts, separate from missing-detection counts. Raw recording remains before processing.

Tests use fake cameras/predictors plus a real localhost socket to check warm-up→prediction
packet transitions. Resampling tests cover known linear motion, irregular cadence,
tracking loss, bounded memory, exact-grid equivalence and no future-anchor use.
The camera/GPU/live FPS acceptance has **not** been rerun with learned inference, and
synthetic checkpoint quality is not suitable evidence for realistic avatar motion.
