# Recording, causal processing and initial evaluation

## Commands

Create the output directory first; use a new filename for each recording/run.

```powershell
poetry run python scripts/benchmark_webcam.py --duration 30 --record-output outputs/raw.npz --processor-config configs/processing/one_euro.json
poetry run poseahead inspect-recording outputs/raw.npz
poetry run poseahead process-recording outputs/raw.npz outputs/filtered.npz --processor-config configs/processing/one_euro.json --run-id one-euro-01
poetry run poseahead pack-sample outputs/fixtures/m1_pose_sample outputs/sample.npz --sequence-id m1-sample --clock-domain recorded_producer_monotonic
```

Webcam recording stores **raw observations before filtering**, including partial
poses and explicit all-invalid frames for detection failures. It does not store
camera images. Capture is in memory, capped by `--record-max-frames` (100,000 by
default), then saved on successful completion; it is not crash-recoverable streaming
storage. An empty capture cannot form a recording. Original files are never
overwritten; derived outputs require a separate path and run ID.

NPZ v1 contains typed numeric arrays and a UTF-8 JSON manifest, loaded with
`allow_pickle=False`. Optional roots/rotations have explicit presence masks.
IDs, source/produced times, validity, confidence and metadata are preserved.
The writer publishes a fully flushed temporary file with atomic no-replace hard
link creation, requiring filesystem hard-link support (tested on local NTFS).

## Baselines

JSON configs under `configs/processing/` select `raw`, `ema`, `one_euro`, `kalman`
or `kalman_cv`. They are initial parameters, not validation-tuned winners.

- EMA uses physical event-time intervals and a time constant in milliseconds.
- One-Euro uses per-axis velocity-adaptive cutoffs; its derivative uses successive
  raw observations. The algorithm reference is the [authors' page](https://gery.casiez.net/1euro/).
- Kalman estimates independent position/velocity states with continuous
  white-acceleration process noise, using metres and seconds.
- Kalman CV extrapolates a fixed 0–150 ms horizon (100 ms in the supplied config)
  without feeding predictions back into filter state.

Invalid joints, missing roots and long tracking gaps reset affected state.
Filters do not invent confidence or rotations. They are not bone-length
constraints or Humanoid retargeting. The same classes serve live and replay paths.

Source and produced timestamps remain those of the observation. Prediction target
time is separate metadata (`target_timestamp_ns`, `target_offset_ns`), not a new
measurement timestamp. Offline processing compute cost is not simulated; webcam
capture-to-send telemetry includes actual filtering cost. UDP sends the correct
algorithm ID and target offset; its v2 offset requires whole milliseconds, so
fractional horizons are rejected for live UDP instead of silently rounded.

## Replay and delay

`replay(sequence)` is deterministic accelerated replay. `replay(sequence, speed=1)`
paces original production-time availability with a monotonic host clock while
preserving recorded timestamps. Replay does not drop frames.

```python
from poseahead.processing.config import load_processor
from poseahead.processing.simulation import simulate_processing
from poseahead.recording import load_recording

sequence = load_recording("outputs/raw.npz")
processor = load_processor("configs/processing/kalman_cv.json")
times = [frame.produced_timestamp_ns + 66_000_000 for frame in sequence.frames]
for snapshot in simulate_processing(sequence, processor, times, delay_ns=66_000_000):
    print(snapshot.evaluation_timestamp_ns, snapshot.observation_age_ns)
```

Queries use the recording's clock. An observation becomes available only at
`produced_timestamp_ns + delay_ns`; no future observations enter the processor.
Each observation is processed once. Queries between arrivals hold the last output,
including invalid output; they do not secretly advance the predictor or hide lost
tracking. Snapshots expose the represented target separately from evaluation time.
The configured prediction horizon and injected delay are independent parameters.

## Initial error metrics

```powershell
poetry run poseahead score-recording outputs/predictions.npz outputs/ground_truth.npz --reference-interpolation linear --max-gap-ms 50
```

The score compares each estimate with the reference at its declared prediction
target, falling back to source time for non-predictive outputs. Exact timestamp
matching is the default. Explicit linear interpolation uses both reference
endpoints, never extrapolates and never spans a gap above the configured limit.
Reference look-ahead is confined to scoring, not available to processors.

Reported metrics are unaligned canonical-17 pelvis-relative MPJPE in mm, including
the zero pelvis, and separate absolute root error in mm. This is **not** standard
H36M-17 protocol MPJPE: our chest is synthesized and joint mapping differs.
Pairs require positive confidence and validity; unavailable targets/joints are
excluded with coverage counts. No eligible pairs means JSON null, not zero error.
Aggregation weights eligible joint pairs equally. No scale/Procrustes alignment.

Clock labels must match; the caller must also ensure identical camera geometry
and genuine reference data. Comparing a processed MeTRAbs recording against raw
MeTRAbs measures deviation from the backend, **not ground-truth accuracy**.
These metrics alone do not establish latency-compensation benefit. Initial jitter
and effective-latency APIs are described in [temporal_metrics.md](temporal_metrics.md).
Noise profiles, controlled experiments and licensed Human3.6M evaluation remain pending.
