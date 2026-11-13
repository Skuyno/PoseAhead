# Predictor training-window export

```powershell
poetry run poseahead verify-dataset data/h36m_index.json
poetry run poseahead export-windows data/h36m_index.json outputs/windows/train_33 --split train --horizon-ms 33
```

Run separately for train/validation/test and each supported horizon: 33, 66, 100,
150 ms. Existing output directories are never reused. The exporter verifies the
entire pinned dataset index, selects the requested split, then generates windows
independently within each clip. No concatenation across clips or subject partitions.
The index and input recording hashes are retained in the output manifest.

Defaults: ten historical frames, a 20,000,000 ns interval (50 Hz), stride one and
256 windows per shard. These are initial configurable input-history parameters,
not validation-selected predictor settings. `--history-frames`, `--step-ns`,
`--stride` and `--shard-size` make the choices explicit. Shard size is capped at 4096.
Only one source recording and one output chunk are held at a time, although the
recording loader still loads each complete clip into memory.

Input features are canonical-17 root-relative positions in metres, shape
`(window, history, 17, 3)`. Pelvis stays zero. Root translation, rotations and
confidence are not model features in this initial export. No normalization or
statistics are fitted, avoiding accidental use of validation/test data. Future
normalization must be fitted exclusively on training data and applied unchanged.

An anchor is the last history frame's event time. In default `exact` mode every sample must lie
exactly on the configured grid and must have been produced by the anchor time.
All joints must have positive confidence and validity. Input history is not
interpolated, padded or filled from future observations. This contract targets
prepared ground truth; inference observations with nonzero availability delays
need an explicitly different training-time protocol rather than timestamp relabelling.

Optional `--history-sampling linear_past` reconstructs the regular history grid
from original irregular/high-rate observations. For each anchor, both interpolation
endpoints must be at or before the anchor and produced by that anchor; no future
observation is admitted. Every source observation in the bracketing span must be
valid, including samples not selected as endpoints. Gaps larger than
`--max-history-gap-ns` (default 100 ms) reject the window. This is interpolation of
past history at prediction time, not an assertion that the interpolated sample was
already available at its nominal grid timestamp. No extrapolation or padding is used.

This mode shares its sampler with the live LearnedProcessor and writes version-2
window manifests containing history_sampling and max_history_gap_ns. Exact exports
remain version 1; readers support both, and training rejects mismatched sampling
settings between train and validation. Source timestamps/frame IDs remain original;
the stored history timestamps describe the reconstructed grid. Tests compare actual
live-processor inputs with exported inputs and verify future-prefix independence.

Target time is `anchor + horizon`, independent of frame index. For horizons not
aligned to 50 Hz (e.g. 33 ms), targets are linearly interpolated between surrounding
ground-truth poses. Both endpoints must be fully valid and their separation must
not exceed the configured sample interval. This is offline supervision only: target
endpoints never enter the input history. Out-of-range targets and gaps are excluded.

Each pickle-free NPZ shard contains:

- `history_positions_m`: float32 input poses;
- `history_timestamps_ns`: uint64 event times;
- `anchor_frame_id`: uint64 source frame IDs;
- `target_timestamp_ns`: uint64 declared target time;
- `target_positions_m`: float32 supervision.

`manifest.json` is written last and records the split, horizon, feature definition,
sampling settings, source identities/hashes, each shard's hash and per-clip counts.
Rejection categories cover irregular history, invalid history, unavailable history,
out-of-range target, target gap and invalid target. Each candidate has one outcome,
using the first failing check. A short clip may produce no candidates, and an export
may legitimately contain zero windows; inspect counts before training.

On failure, an incomplete directory may remain without a manifest. It must not be
treated as a complete export. Archives should be loaded with `allow_pickle=False`.
These files do not constitute trained checkpoints or evidence of predictor quality.
`verify-windows` and the checked `WindowStore` now validate these shards; train-only
normalization and a CPU MLP training loop are described in
[predictor_training.md](predictor_training.md). Synthetic regression tests and small
real CMU pilots are available; see [cmu.md](cmu.md) for experimental limitations.
