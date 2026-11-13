import json
from dataclasses import replace

import numpy as np
import pytest

from poseahead.core import PoseFrame, PoseSequence
from poseahead.core.serialization import encode_pose_frame
from poseahead.recording import load_recording, replay, save_recording


def sample():
    first = PoseFrame(0, 100, 200, np.zeros((17, 3)), np.ones(17),
                      np.ones(17, dtype=bool), metadata={"text": "поза", "nested": [1, None]})
    second = replace(first, frame_id=4, source_timestamp_ns=20_000_100,
                     produced_timestamp_ns=40_000_200, root_position=[1, 2, 3],
                     root_rotation=[0, 0, 0, 1],
                     joint_rotations=np.tile([0, 0, 0, 1], (17, 1)))
    return PoseSequence("clip", "test", (first, second))


def test_recording_roundtrip_is_exact_and_does_not_overwrite(tmp_path):
    path = tmp_path / "poses.npz"
    sequence = sample()
    save_recording(path, sequence)
    restored = load_recording(path)
    assert restored.sequence_id == sequence.sequence_id
    assert restored.clock_domain == sequence.clock_domain
    assert [encode_pose_frame(f) for f in restored.frames] == [
        encode_pose_frame(f) for f in sequence.frames
    ]
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_recording(path, sequence)
    assert path.read_bytes() == before


@pytest.mark.parametrize("corruption", ["version", "dtype", "time", "optional", "missing"])
def test_recording_rejects_bad_schema_and_invalid_poses(tmp_path, corruption):
    path = tmp_path / "good.npz"
    save_recording(path, sample())
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if corruption == "version":
        manifest = json.loads(arrays["manifest"].tobytes())
        manifest["version"] = True
        arrays["manifest"] = np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8)
    elif corruption == "dtype":
        arrays["ids"] = arrays["ids"].astype(float)
    elif corruption == "time":
        arrays["source_ns"][1] = arrays["source_ns"][0]
    elif corruption == "optional":
        arrays["root_rotations"][1] = 0
    else:
        del arrays["positions"]
    bad = tmp_path / "bad.npz"
    np.savez(bad, **arrays)
    with pytest.raises(ValueError):
        load_recording(bad)


def test_accelerated_replay_preserves_frames_and_never_uses_wall_time():
    def forbidden(*args):
        raise AssertionError("accelerated replay must not use wall time")
    sequence = sample()
    frames = list(replay(sequence, clock_ns=forbidden, sleep=forbidden))
    assert all(a is b for a, b in zip(frames, sequence.frames, strict=True))


def test_paced_replay_uses_availability_time_and_does_not_accumulate_consumer_delay():
    now = [1_000_000_000]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += round(seconds * 1e9)

    iterator = replay(sample(), speed=2, clock_ns=lambda: now[0], sleep=sleep)
    first = next(iterator)
    now[0] += 5_000_000  # consumer spends 5 ms processing the first observation
    second = next(iterator)
    assert sum(slept) == pytest.approx(.015)
    assert second.produced_timestamp_ns - first.produced_timestamp_ns == 40_000_000


@pytest.mark.parametrize("speed", [0, -1, True, float("nan"), float("inf")])
def test_invalid_replay_speed(speed):
    with pytest.raises(ValueError):
        list(replay(sample(), speed=speed))


def test_interrupted_save_does_not_publish_a_partial_recording(tmp_path, monkeypatch):
    def fail(stream, **arrays):
        stream.write(b"partial archive")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="disk failure"):
        save_recording(tmp_path / "recording.npz", sample())
    assert list(tmp_path.iterdir()) == []


def test_truncated_archive_reports_validation_error(tmp_path):
    path = tmp_path / "truncated.npz"
    save_recording(path, sample())
    path.write_bytes(path.read_bytes()[:40])
    with pytest.raises(ValueError, match="damaged recording"):
        load_recording(path)


def test_atomic_publication_cannot_overwrite_another_writer(tmp_path, monkeypatch):
    import os

    real_link = os.link
    path = tmp_path / "recording.npz"

    def race(source, destination):
        path.write_bytes(b"other writer owns this file")
        real_link(source, destination)

    monkeypatch.setattr(os, "link", race)
    with pytest.raises(FileExistsError):
        save_recording(path, sample())
    assert path.read_bytes() == b"other writer owns this file"
    assert list(tmp_path.iterdir()) == [path]
