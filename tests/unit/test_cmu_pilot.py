import json
from dataclasses import replace

import numpy as np
import pytest

from poseahead.core import PoseSequence
from poseahead.datasets.cmu_pilot import check_skeleton_partitions, prepare_pilot
from poseahead.datasets.human36m import camera_poses_to_sequence
from poseahead.datasets.index import freeze_dataset, verify_dataset
from poseahead.datasets.splits import sequence_split
from poseahead.datasets.window_store import WindowStore
from poseahead.datasets.windows import export_training_windows
from poseahead.recording import save_recording


def source(subject="02"):
    sequence = camera_poses_to_sequence(np.zeros((150, 32, 3)), sequence_id=f"cmu/{subject}",
                                        camera_id="fixture", fps_numerator=120)
    return PoseSequence(sequence.sequence_id, sequence.clock_domain, tuple(
        replace(f, metadata={"dataset": "CMU", "cmu_subject": subject, "fps_numerator": 120})
        for f in sequence.frames))


def test_pilot_decimation_preserves_source_time_and_poses():
    original = source()
    prepared = prepare_pilot(original, "a" * 64)
    assert sequence_split(prepared) == ("train", "CMU:02")
    assert len(prepared.frames) == 25
    for i, frame in enumerate(prepared.frames):
        assert frame.source_timestamp_ns == i * 50_000_000
        assert frame.frame_id == i * 6
        np.testing.assert_array_equal(frame.joint_positions, original.frames[i * 6].joint_positions)


def test_reject_unsupported_subject_and_missing_frame():
    with pytest.raises(ValueError, match="policy"):
        prepare_pilot(source("07"), "a" * 64)
    original = source()
    broken = PoseSequence(original.sequence_id, original.clock_domain, original.frames[1:])
    with pytest.raises(ValueError, match="timing"):
        prepare_pilot(broken, "a" * 64)


def test_same_skeleton_must_not_cross_splits(tmp_path):
    sequences = [prepare_pilot(source(subject), "a" * 64) for subject in ("02", "13")]
    with pytest.raises(ValueError, match="calibration"):
        check_skeleton_partitions(sequences)
    paths = []
    for i, sequence in enumerate(sequences):
        path = tmp_path / f"{i}.npz"
        save_recording(path, sequence)
        paths.append(path)
    with pytest.raises(ValueError, match="calibration"):
        freeze_dataset(paths, tmp_path / "index.json")
    assert not (tmp_path / "index.json").exists()


def test_cmu_index_windows_and_policy_tamper(tmp_path):
    paths = []
    for i, subject in enumerate(("02", "13", "14")):
        sequence = prepare_pilot(source(subject), str(i) * 64)
        path = tmp_path / f"{subject}.npz"
        save_recording(path, sequence)
        paths.append(path)
    index = tmp_path / "index.json"
    freeze_dataset(paths, index)
    assert verify_dataset(index)["split_counts"] == {"train": 1, "validation": 1, "test": 1}
    for split in ("train", "validation", "test"):
        export_training_windows(index, tmp_path / split, split=split, step_ns=50_000_000)
        store = WindowStore(tmp_path / split)
        store.verify()
        assert store.manifest["windows"] > 0
    manifest = json.loads(index.read_text())
    assert manifest["format"] == "poseahead-cmu-index"
    manifest["subject_splits"]["CMU:14"] = "train"
    index.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="policy"):
        verify_dataset(index)


def test_dataset_label_must_match_subject_namespace():
    sequence = prepare_pilot(source(), "a" * 64)
    frames = tuple(replace(f, metadata={**f.metadata, "dataset": "Human3.6M"})
                   for f in sequence.frames)
    with pytest.raises(ValueError, match="split"):
        sequence_split(PoseSequence(sequence.sequence_id, sequence.clock_domain, frames))
