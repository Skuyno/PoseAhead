# Canonical Pose Format

`PoseFrame` is the only pose representation allowed to cross from perception
backends into processing, prediction, recording, evaluation, or transport.

## Coordinates and units

- right-handed camera coordinates in metres;
- `+X` points right in the image, `+Y` down, and `+Z` away from the camera;
- joint positions are relative to the pelvis, which is exactly `(0, 0, 0)`;
- optional `root_position` is the pelvis translation in camera coordinates;
- Unity conversion is `(x, y, z) -> (x, -y, z)`, then avatar calibration.

Both timestamps are nanoseconds in the producer's monotonic clock domain.
`source_timestamp_ns` is capture/event time and `produced_timestamp_ns` is when
that version of the pose became available. They are not wall-clock timestamps.

`PoseSequence` groups immutable frames under a named `sequence_id` and
`clock_domain`. IDs and event timestamps must strictly increase; produced times
may be equal but must not decrease. Frame gaps and irregular sampling are allowed.
Declared frame clock metadata must agree with the sequence. Producers must name
clock domains correctly; numeric values alone cannot reveal mixed clocks.
Empty sequences are rejected. Duration is last event time minus first event time.

Dataset timestamps may use `frame_timestamp_ns` with an exact rational frame
rate. Ground-truth conversion is described in [human36m.md](human36m.md).

## Joint order and hierarchy

The numeric order is stable and shared by storage and UDP.

| ID | Joint | Parent |
|---:|---|---|
| 0 | pelvis | none |
| 1 | spine | pelvis |
| 2 | chest | spine |
| 3 | neck | chest |
| 4 | head | neck |
| 5 | left_shoulder | chest |
| 6 | left_elbow | left_shoulder |
| 7 | left_wrist | left_elbow |
| 8 | right_shoulder | chest |
| 9 | right_elbow | right_shoulder |
| 10 | right_wrist | right_elbow |
| 11 | left_hip | pelvis |
| 12 | left_knee | left_hip |
| 13 | left_ankle | left_knee |
| 14 | right_hip | pelvis |
| 15 | right_knee | right_hip |
| 16 | right_ankle | right_knee |

## Rotations and validity

Positions are mandatory. Root and joint rotations are optional because a
monocular position estimator cannot observe bone twist reliably. When present,
quaternions use `(x, y, z, w)` ordering. Joint rotations are local to the parent;
the root rotation is in camera coordinates. The Unity runtime initially derives
bone directions from positions and its calibrated T-pose.

Present rotations must be unit quaternions (absolute norm tolerance `1e-4`).
Zero or scaled quaternions are rejected, not implicitly normalized. This applies
equally to Python construction/storage and the C# UDP decoder.

`joint_validity` is authoritative. Invalid joints still contain finite numeric
values so array processing remains safe, but consumers must not interpret those
values as observations. Confidence is always in `[0, 1]`.
Input validity masks accept booleans or numeric 0/1 only: NaN, other numbers,
strings and nulls are rejected before boolean conversion. Unused high bits in
serialized validity masks must be zero.

An invalid hip/root invalidates every pelvis-relative joint. The adapter clears
confidence for invalid outputs. `PoseFrame.is_valid` uses a strict benchmark
policy: all 17 joints valid with positive confidence. Partial frames remain
available to future processors, but do not count toward full-body valid FPS.

Arrays are backed by immutable bytes. Metadata is recursively copied into
immutable mappings/tuples and accepts finite JSON values only. Storage converts
these containers back to JSON without changing the immutable observation.

## MeTRAbs/H36M mapping

MeTRAbs emits absolute H36M-17 joints in camera coordinates and millimetres. The
adapter converts them to metres, stores source `pelv` as `root_position`, and
subtracts it from every canonical joint. Limbs, pelvis, spine, neck and head map
by audited H36M names. Canonical chest is synthesized halfway between source
spine and neck; the H36M head-top point is unused.

Successful roots carry `root_translation_status=estimated_monocular` and
`scale_status=metrabs_metric_estimate`. This is an inferred scale and depth, not a
physical measurement or trusted ground truth. Measured intrinsics improve camera
geometry; otherwise a centered 55-degree field of view is assumed.

The model exposes person-box confidence but no per-joint confidences. The adapter
therefore repeats detector confidence as a labelled proxy and invalidates nonfinite
joints. It must not be interpreted as calibrated per-joint uncertainty. See
`metrabs.md` for setup, causality, licensing and measured smoke-test limitations.
