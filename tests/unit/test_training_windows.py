import json
from dataclasses import replace

import numpy as np
import pytest

from poseahead.core import PoseSequence
from poseahead.datasets.human36m import camera_poses_to_sequence
from poseahead.datasets.index import freeze_dataset
from poseahead.datasets.windows import export_training_windows, iter_training_windows
from poseahead.recording import save_recording


def sequence(subject="S1", split="train", count=30):
    positions = np.zeros((count, 32, 3))
    positions[:, 1:, 0] = np.arange(count)[:, None] * 20  # 1 m/s at 50 Hz
    result = camera_poses_to_sequence(
        positions, sequence_id=f"{subject}/walking/1", camera_id="cam",
    )
    return PoseSequence(result.sequence_id, result.clock_domain, tuple(
        replace(f, metadata={**f.metadata, "subject": subject, "split": split})
        for f in result.frames
    ))


@pytest.mark.parametrize("horizon", [33, 66, 100, 150])
def test_target_interpolation_and_causal_history(horizon):
    data = sequence()
    windows = list(iter_training_windows(data, history_frames=3, horizon_ms=horizon))
    first = windows[0]
    assert first.history_timestamps_ns == (0, 20_000_000, 40_000_000)
    assert first.target_timestamp_ns == 40_000_000 + horizon * 1_000_000
    assert first.target_positions_m[1, 0] == pytest.approx(.04 + horizon / 1000)
    assert all(w.history_timestamps_ns[-1] < w.target_timestamp_ns for w in windows)
    with pytest.raises(ValueError):
        first.history_positions_m.setflags(write=True)


def test_future_change_does_not_modify_input_history():
    original = sequence()
    frames = list(original.frames)
    frames[4] = replace(frames[4], joint_positions=frames[4].joint_positions * 10)
    modified = PoseSequence(original.sequence_id, original.clock_domain, tuple(frames))
    first = next(iter_training_windows(original, history_frames=3, horizon_ms=33))
    second = next(iter_training_windows(modified, history_frames=3, horizon_ms=33))
    np.testing.assert_array_equal(first.history_positions_m, second.history_positions_m)
    assert not np.array_equal(first.target_positions_m, second.target_positions_m)


def test_gaps_invalid_joints_and_unavailable_observations_are_excluded():
    original = sequence()
    frames = list(original.frames)
    frames[2] = replace(frames[2], joint_validity=np.zeros(17, bool))
    frames[10] = replace(frames[10], produced_timestamp_ns=frames[10].source_timestamp_ns + 1)
    del frames[20]
    data = PoseSequence(original.sequence_id, original.clock_domain, tuple(frames))
    stats = {}
    windows = list(iter_training_windows(data, history_frames=3, horizon_ms=33, stats=stats))
    assert stats["invalid_history"] == 3
    assert stats["unavailable_history"] == 1
    assert stats["history_gap"] == 2
    assert stats["target_gap"] > 0
    assert stats["target_outside"] > 0
    assert stats["candidates"] == sum(v for k, v in stats.items() if k != "candidates")
    assert stats["accepted"] == len(windows)


def test_export_selects_split_before_windows_and_bounds_shards(tmp_path):
    paths = []
    for subject, split in (("S1", "train"), ("S8", "validation"), ("S9", "test")):
        path = tmp_path / f"{subject}.npz"
        save_recording(path, sequence(subject, split))
        paths.append(path)
    index = tmp_path / "index.json"
    freeze_dataset(paths, index)
    output = tmp_path / "windows"
    report = export_training_windows(index, output, split="train", history_frames=3,
                                     horizon_ms=33, shard_size=4)
    assert report["sequences"] == 1
    assert report["windows"] == 26
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["sequences"][0]["subject"] == "S1"
    assert max(s["windows"] for s in manifest["shards"]) <= 4
    for shard in manifest["shards"]:
        with np.load(output / shard["file"], allow_pickle=False) as arrays:
            assert arrays["history_positions_m"].shape[1:] == (3, 17, 3)
            assert arrays["history_timestamps_ns"].dtype == np.uint64
    with pytest.raises(FileExistsError):
        export_training_windows(index, output, split="train")


def test_no_cross_clip_windows_or_invalid_horizon():
    assert list(iter_training_windows(sequence(count=2), history_frames=3)) == []
    with pytest.raises(ValueError, match="horizon"):
        list(iter_training_windows(sequence(), horizon_ms=50))


def test_noise_changes_histories_only_and_is_consistent_across_windows():
    data = sequence()
    clean = list(iter_training_windows(data, history_frames=3))
    noisy = list(iter_training_windows(data, history_frames=3, training_noise_std_m=.01))
    repeated = list(iter_training_windows(data, history_frames=3, training_noise_std_m=.01))
    for original, a, b in zip(clean, noisy, repeated, strict=True):
        np.testing.assert_array_equal(original.target_positions_m, a.target_positions_m)
        np.testing.assert_array_equal(a.history_positions_m, b.history_positions_m)
        assert not np.array_equal(original.history_positions_m, a.history_positions_m)
        np.testing.assert_array_equal(a.history_positions_m[:, 0], 0)
    np.testing.assert_array_equal(noisy[0].history_positions_m[1:],
                                  noisy[1].history_positions_m[:-1])
    with pytest.raises(ValueError, match="train split"):
        list(iter_training_windows(sequence("S8", "validation"), training_noise_std_m=.01))


def test_noise_export_manifest_and_clean_target_contract(tmp_path):
    from poseahead.datasets.window_store import WindowStore
    path = tmp_path / "data.npz"
    save_recording(path, sequence())
    index = tmp_path / "index.json"
    freeze_dataset([path], index)
    output = tmp_path / "noise"
    export_training_windows(index, output, split="train", training_noise_std_m=.01)
    store = WindowStore(output)
    store.verify()
    assert store.manifest["version"] == 3
    store.manifest["training_noise"]["targets"] = "noisy"
    (output / "manifest.json").write_text(json.dumps(store.manifest))
    with pytest.raises(ValueError, match="corruption"):
        WindowStore(output)


def high_rate_sequence():
    original = sequence(count=100)
    frames = tuple(replace(f, source_timestamp_ns=i * 1_000_000_000 // 120,
                           produced_timestamp_ns=i * 1_000_000_000 // 120)
                   for i, f in enumerate(original.frames))
    return PoseSequence(original.sequence_id, original.clock_domain, frames)


def test_native_rate_causal_interpolation_and_future_independence():
    data = high_rate_sequence()
    windows = list(iter_training_windows(data, history_sampling="linear_past"))
    first = windows[0]
    assert first.anchor_frame_id == 22
    assert np.diff(first.history_timestamps_ns).tolist() == [20_000_000] * 9
    # Shared live sampler must produce the same input given the same prefix.
    from poseahead.core.history import sample_history
    expected = sample_history(data.frames[:23], first.history_timestamps_ns, 100_000_000)
    np.testing.assert_array_equal(first.history_positions_m, expected.astype(np.float32))
    frames = tuple(replace(f, joint_positions=f.joint_positions * 100) if i > 22 else f
                   for i, f in enumerate(data.frames))
    changed = PoseSequence(data.sequence_id, data.clock_domain, frames)
    second = next(iter_training_windows(changed, history_sampling="linear_past"))
    np.testing.assert_array_equal(first.history_positions_m, second.history_positions_m)
    assert not np.array_equal(first.target_positions_m, second.target_positions_m)


def test_linear_history_rejects_invalid_unavailable_and_gaps():
    data = high_rate_sequence()
    for field, value in (("joint_validity", np.zeros(17, bool)),
                         ("produced_timestamp_ns", 20 * 1_000_000_000 // 120 + 1)):
        # Availability delayed by 1 ns only rejects an anchor at the delayed frame.
        frames = list(data.frames)
        frames[20] = replace(frames[20], **{field: value})
        modified = PoseSequence(data.sequence_id, data.clock_domain, tuple(frames))
        stats = {}
        list(iter_training_windows(modified, history_frames=3,
                                   history_sampling="linear_past", stats=stats))
        assert stats["invalid_history" if field == "joint_validity" else
                     "unavailable_history"] > 0
        assert stats["candidates"] == sum(v for k, v in stats.items() if k != "candidates")
    stats = {}
    list(iter_training_windows(data, history_sampling="linear_past",
                               max_history_gap_ns=1_000_000, stats=stats))
    assert stats["accepted"] == 0


def test_linear_history_manifest_roundtrip(tmp_path):
    from poseahead.datasets.window_store import WindowStore
    path = tmp_path / "native.npz"
    save_recording(path, high_rate_sequence())
    index = tmp_path / "index.json"
    freeze_dataset([path], index)
    output = tmp_path / "windows"
    export_training_windows(index, output, split="train", history_sampling="linear_past")
    store = WindowStore(output)
    assert store.manifest["version"] == 2
    assert store.verify()["windows"] > 0
    manifest = store.manifest
    manifest["history_sampling"] = "future"
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="history_sampling"):
        WindowStore(output)


def test_native_export_matches_live_processor_inputs(tmp_path, monkeypatch):
    from poseahead.processing.learned import LearnedProcessor

    captured = {}

    class CapturePredictor:
        history_frames = 10
        step_ns = 20_000_000
        horizon_ms = 100

        def predict(self, positions, timestamps_ns):
            captured[timestamps_ns[-1]] = positions.copy()
            return positions[-1]

    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr("poseahead.processing.learned.FixedHorizonPredictor",
                        lambda _: CapturePredictor())
    live = LearnedProcessor(checkpoint=tmp_path, history_sampling="linear_past")
    data = high_rate_sequence()
    for frame in data.frames:
        live.process(frame)
    for window in iter_training_windows(data, history_sampling="linear_past"):
        expected = captured[window.history_timestamps_ns[-1]].astype(np.float32)
        np.testing.assert_array_equal(window.history_positions_m, expected)
