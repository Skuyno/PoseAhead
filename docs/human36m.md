# Human3.6M canonical adapter

## Data access status (2026-09-22)

No Human3.6M archives or prepared pose exports were found in PoseAhead or the checked
`../MyNirV2/data` inventory. MyNirV2 contains face/CelebA and synthetic expression
assets, not a body-motion ground-truth replacement. Other locations have not been
searched; a user-provided dataset path is still needed.

The official [download browser](https://vision.imar.ro/human3.6m/filebrowser.php)
requires login. The [registration page](https://vision.imar.ro/human3.6m/register.php)
requires an academic email address, institution and supervisor information.
Registration and acceptance of the [dataset terms](https://vision.imar.ro/human3.6m/eula.php)
must be handled by the user; no account was created and no terms were accepted.
Keep downloaded archives outside Git and retain the original sequence, frame-rate,
joint-layout and camera-calibration metadata for conversion.

The current importer consumes explicit NPY/JSON prepared exports, not arbitrary
official archives. Official-file conversion must be implemented and validated
against the actual supplied layout before training; renaming an archive does not
make it compatible. Temporal prediction requires ordered sequences, not a shuffled
collection of independently sampled pose images.

`poseahead.datasets.human36m.camera_poses_to_sequence` accepts an array of shape
`(N, 32, 3)` containing absolute camera-space ground-truth positions in millimetres.
Callers must first decode the source files and apply the correct camera extrinsics
if positions are in world space. Axes, units and layouts are not inferred.

The 32-joint indexing follows the skeleton in the primary
[VideoPose3D implementation](https://github.com/facebookresearch/VideoPose3D/blob/main/common/h36m_dataset.py).
The mapping matches the existing live MeTRAbs convention:

| Canonical joint | Source index |
|---|---:|
| pelvis | 0 |
| spine | 12 |
| chest | midpoint of 12 and 13 |
| neck | 13 |
| head | 14 |
| left shoulder / elbow / wrist | 17 / 18 / 19 |
| right shoulder / elbow / wrist | 25 / 26 / 27 |
| left hip / knee / ankle | 6 / 7 / 8 |
| right hip / knee / ankle | 1 / 2 / 3 |

Source index 15 is unused. Canonical labels are a project convention; evaluation
must report the mapping and synthesized chest explicitly instead of presenting
canonical MPJPE as standard H36M-17 MPJPE.

```python
from poseahead.datasets.human36m import camera_poses_to_sequence

sequence = camera_poses_to_sequence(
    camera_positions_mm,
    sequence_id="S1/Walking/1",
    camera_id="54138969",
    fps_numerator=50,
    first_frame_index=0,
)
```

Output positions are pelvis-relative metres; `root_position` retains absolute
camera translation. Confidence 1 means valid ground truth. Nonfinite data is
rejected; rotations remain absent. Ground truth is labelled explicitly in metadata.

Event timestamps use exact rational frame-rate arithmetic; at 50 Hz the interval
is 20,000,000 ns. `first_frame_index` preserves the position of an extracted clip
in the source timeline. Produced time equals event time for these ground-truth
frames; it is not a measured processing duration. RGB inference observations and
simulated delayed observations must be created separately.

## Prepared-file importer

`poseahead import-h36m manifest.json output.npz` loads numeric, pickle-free NPY
positions using this JSON manifest (paths are relative to the manifest):

```json
{
  "format": "poseahead-human36m",
  "version": 1,
  "subject": "S1",
  "action": "Walking",
  "take": "1",
  "camera_id": "54138969",
  "positions_file": "positions.npy",
  "coordinate_space": "camera",
  "units": "mm",
  "fps_numerator": 50,
  "fps_denominator": 1,
  "first_frame_index": 0
}
```

For `coordinate_space: world`, additionally provide `rotation_world_to_camera`
(3x3 proper orthonormal matrix) and `translation_world_to_camera_mm` (three values).
The exact row-vector convention is `X_camera = X_world @ R.T + t`.
**The translation is not the camera centre**: for a world-space camera centre C,
the column-vector relation is `t = -R @ C`. Camera-space input rejects extrinsics
to prevent accidental double transformation. Axes and camera IDs are never guessed.

Every output frame receives the fixed subject split label:
`train=S1,S5,S6,S7`, `validation=S8`, `test=S9,S11`. Unsupported subjects are rejected.
This labels clips; split-safe training-window generation is documented in
[training_windows.md](training_windows.md).
Dataset download, official raw CDF decoding and
real-data evaluation remain unfinished. Current importer tests use synthetic data.

## Freeze the prepared dataset inventory

```powershell
poetry run poseahead freeze-dataset data/h36m_index.json data/S1_Walking.npz data/S8_Walking.npz data/S9_Walking.npz
poetry run poseahead verify-dataset data/h36m_index.json
```

The index records absolute paths, SHA-256 hashes, sequence and clock IDs, subject,
split, frame count and time bounds. It validates every frame's labels and rejects
duplicate files or sequence/clock identities. The split policy is centralized and
read-only in code. Existing index files are never overwritten; create a new version
when the dataset inventory changes. Parent directories must exist.

Verification re-reads the recordings and checks hashes, metadata and the fixed
partition policy. It is an explicit command; batch runs independently validate
their referenced recordings and do not automatically consume this index yet.
The index does not download data, prove completeness of the official dataset,
authenticate manually supplied labels, or itself generate training windows. Paths are
absolute, so relocation requires a new index. The JSON is not filesystem-immutable
or signed; preserve the frozen index with the research artifacts.
