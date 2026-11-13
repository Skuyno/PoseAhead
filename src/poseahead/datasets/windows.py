"""Causal root-relative training windows from split-verified prepared recordings."""

import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from poseahead.artifacts import sha256_file
from poseahead.core import PoseSequence
from poseahead.core.history import sample_history
from poseahead.datasets.augmentation import noisy_training_observations, validate_noise
from poseahead.datasets.index import verify_dataset
from poseahead.datasets.splits import sequence_split
from poseahead.recording import load_recording


@dataclass(frozen=True, slots=True)
class TrainingWindow:
    history_positions_m: np.ndarray
    history_timestamps_ns: tuple[int, ...]
    anchor_frame_id: int
    target_timestamp_ns: int
    target_positions_m: np.ndarray


def _settings(history_frames, horizon_ms, step_ns, stride):
    for name, value in (("history_frames", history_frames), ("step_ns", step_ns),
                        ("stride", stride)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(horizon_ms) is not int or horizon_ms not in (33, 66, 100, 150):
        raise ValueError("training horizon must be 33, 66, 100 or 150 ms")


def _sampling_settings(history_sampling, max_history_gap_ns):
    if history_sampling not in ("exact", "linear_past"):
        raise ValueError("history_sampling must be exact or linear_past")
    if type(max_history_gap_ns) is not int or not 1 <= max_history_gap_ns <= 1_000_000_000:
        raise ValueError("max_history_gap_ns must be in [1, 1000000000]")


def iter_training_windows(
    sequence: PoseSequence, *, history_frames=10, horizon_ms=100, step_ns=20_000_000,
    stride=1, stats=None, history_sampling="exact", max_history_gap_ns=100_000_000,
    training_noise_std_m=0., training_noise_seed=2026,
):
    """Yield valid windows within one labelled clip using only available history.

    Histories require the configured uniform event-time grid, fully valid joints
    with positive confidence and availability by anchor event time. Targets use
    exact or gap-limited linear ground-truth interpolation. Root is not a feature.
    """
    _settings(history_frames, horizon_ms, step_ns, stride)
    _sampling_settings(history_sampling, max_history_gap_ns)
    validate_noise(training_noise_std_m, training_noise_seed)
    split, _ = sequence_split(sequence)
    if split == "unlabelled":
        raise ValueError("training windows require labelled supported data")
    if stats is None:
        stats = {}
    for key in ("candidates", "accepted", "history_gap", "invalid_history",
                "unavailable_history", "target_outside", "target_gap", "invalid_target"):
        stats.setdefault(key, 0)
    frames = sequence.frames
    observed = (noisy_training_observations(sequence, training_noise_std_m, training_noise_seed)
                .frames if training_noise_std_m else frames)
    times = [f.source_timestamp_ns for f in frames]
    for anchor in range(history_frames - 1, len(frames), stride):
        stats["candidates"] += 1
        history = observed[anchor - history_frames + 1:anchor + 1]
        anchor_ns = times[anchor]
        grid = tuple(anchor_ns - (history_frames - 1 - i) * step_ns
                     for i in range(history_frames))
        if history_sampling == "linear_past":
            start = bisect_left(times, grid[0], 0, anchor + 1)
            if start <= anchor and times[start] != grid[0]:
                start -= 1
            if start < 0:
                stats["history_gap"] += 1
                continue
            history = observed[start:anchor + 1]
            gap = any(b.source_timestamp_ns - a.source_timestamp_ns > max_history_gap_ns
                      for a, b in zip(history, history[1:], strict=False))
        else:
            gap = any(f.source_timestamp_ns != grid[i] for i, f in enumerate(history))
        if gap:
            stats["history_gap"] += 1
            continue
        if any(not (f.joint_validity & (f.joint_confidence > 0)).all() for f in history):
            stats["invalid_history"] += 1
            continue
        if any(f.produced_timestamp_ns > anchor_ns for f in history):
            stats["unavailable_history"] += 1
            continue
        target_ns = anchor_ns + horizon_ms * 1_000_000
        right = bisect_left(times, target_ns)
        if right == len(times):
            stats["target_outside"] += 1
            continue
        left = right if times[right] == target_ns else right - 1
        if times[right] - times[left] > step_ns:
            stats["target_gap"] += 1
            continue
        if any(not (frames[i].joint_validity & (frames[i].joint_confidence > 0)).all()
               for i in (left, right)):
            stats["invalid_target"] += 1
            continue
        alpha = 0. if left == right else (target_ns - times[left]) / (times[right] - times[left])
        target = ((1 - alpha) * frames[left].joint_positions.astype(np.float64)
                  + alpha * frames[right].joint_positions.astype(np.float64)).astype(np.float32)
        positions = (sample_history(history, grid, max_history_gap_ns)
                     if history_sampling == "linear_past"
                     else np.stack([f.joint_positions for f in history]))
        if positions is None:
            stats["history_gap"] += 1
            continue
        positions = positions.astype(np.float32)
        positions = np.frombuffer(positions.tobytes(), dtype=np.float32).reshape(positions.shape)
        target = np.frombuffer(target.tobytes(), dtype=np.float32).reshape(17, 3)
        stats["accepted"] += 1
        yield TrainingWindow(positions, grid,
                             frames[anchor].frame_id, target_ns, target)


def export_training_windows(
    index_path, output_directory, *, split, history_frames=10, horizon_ms=100,
    step_ns=20_000_000, stride=1, shard_size=256,
    history_sampling="exact", max_history_gap_ns=100_000_000,
    training_noise_std_m=0., training_noise_seed=2026,
):
    """One split and one horizon per artifact; stream bounded shards per sequence.

    The dataset index is verified before partition selection or window generation.
    Completion manifest is published last; partial directories are never reused.
    """
    _settings(history_frames, horizon_ms, step_ns, stride)
    _sampling_settings(history_sampling, max_history_gap_ns)
    validate_noise(training_noise_std_m, training_noise_seed)
    if training_noise_std_m and split != "train":
        raise ValueError("training augmentation requires train split")
    if split not in ("train", "validation", "test"):
        raise ValueError("unsupported training-window split")
    if type(shard_size) is not int or not 1 <= shard_size <= 4096:
        raise ValueError("shard_size must be between 1 and 4096")
    index_path, output_directory = Path(index_path), Path(output_directory)
    if output_directory.exists():
        raise FileExistsError("training-window output directory already exists")
    index_hash = sha256_file(index_path)
    verify_dataset(index_path)
    manifest = json.loads(index_path.read_text(encoding="utf-8"))
    if sha256_file(index_path) != index_hash:
        raise ValueError("dataset index changed during verification")
    entries = [e for e in manifest["recordings"] if e["split"] == split]
    if not entries:
        raise ValueError("dataset index contains no recordings for selected split")
    output_directory.mkdir(parents=True, exist_ok=False)
    shards, sequences = [], []

    def save_chunk(chunk, entry_index):
        name = f"shard-{len(shards):06d}.npz"
        with (output_directory / name).open("xb") as stream:
            np.savez_compressed(
                stream, history_positions_m=np.stack([w.history_positions_m for w in chunk]),
                history_timestamps_ns=np.array([w.history_timestamps_ns for w in chunk], np.uint64),
                target_positions_m=np.stack([w.target_positions_m for w in chunk]),
                target_timestamp_ns=np.array([w.target_timestamp_ns for w in chunk], np.uint64),
                anchor_frame_id=np.array([w.anchor_frame_id for w in chunk], np.uint64),
            )
        shards.append({"file": name, "sha256": sha256_file(output_directory / name),
                       "windows": len(chunk), "sequence_index": entry_index})

    for i, entry in enumerate(entries):
        path = Path(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("dataset recording changed after verification")
        sequence = load_recording(path)
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("dataset recording changed during loading")
        stats, chunk = {}, []
        for window in iter_training_windows(sequence, history_frames=history_frames,
                                             horizon_ms=horizon_ms, step_ns=step_ns,
                                             stride=stride, stats=stats,
                                             history_sampling=history_sampling,
                                             max_history_gap_ns=max_history_gap_ns,
                                             training_noise_std_m=training_noise_std_m,
                                             training_noise_seed=training_noise_seed):
            chunk.append(window)
            if len(chunk) == shard_size:
                save_chunk(chunk, i)
                chunk.clear()
        if chunk:
            save_chunk(chunk, i)
        sequences.append({**entry, "window_counts": stats})
    report = {
        "format": "poseahead-training-windows", "version": 1,
        "split": split, "horizon_ms": horizon_ms, "history_frames": history_frames,
        "step_ns": step_ns, "stride": stride, "shard_size": shard_size,
        "dataset_index": str(index_path.resolve()), "dataset_index_sha256": index_hash,
        "features": "canonical17_root_relative_positions_m_no_normalization",
        "target_interpolation": "linear_reference_only_max_gap_step_ns",
        "sequences": sequences, "shards": shards,
        "windows": sum(s["windows"] for s in shards),
    }
    if history_sampling != "exact":
        report.update(version=2, history_sampling=history_sampling,
                      max_history_gap_ns=max_history_gap_ns)
    if training_noise_std_m:
        report.update(version=3, history_sampling=history_sampling,
                      max_history_gap_ns=max_history_gap_ns, training_noise={
                          "std_m": training_noise_std_m, "seed": training_noise_seed,
                          "stream": "sha256_seed_sequence_clock_first8_le_PCG64_per_source_frame",
                          "targets": "clean_reference"})
    with (output_directory / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return {"split": split, "horizon_ms": horizon_ms, "windows": report["windows"],
            "shards": len(shards), "sequences": len(sequences)}
