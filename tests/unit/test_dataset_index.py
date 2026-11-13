import json
from dataclasses import replace

import numpy as np
import pytest

from poseahead.cli import main
from poseahead.core import PoseSequence
from poseahead.datasets.human36m import camera_poses_to_sequence
from poseahead.datasets.index import freeze_dataset, verify_dataset
from poseahead.datasets.splits import SUBJECT_SPLITS
from poseahead.recording import save_recording


def recording(tmp_path, subject, split=None):
    sequence = camera_poses_to_sequence(np.zeros((3, 32, 3)), sequence_id=f"{subject}/walk/1",
                                        camera_id="camera")
    sequence = PoseSequence(sequence.sequence_id, sequence.clock_domain, tuple(
        replace(f, metadata={**f.metadata, "subject": subject,
                              "split": split or SUBJECT_SPLITS[subject]})
        for f in sequence.frames
    ))
    path = tmp_path / f"{subject}.npz"
    save_recording(path, sequence)
    return path


def test_index_pins_all_partitions_and_preserves_existing_file(tmp_path, capsys):
    paths = [recording(tmp_path, s) for s in ("S1", "S8", "S9")]
    output = tmp_path / "dataset.json"
    assert main(["freeze-dataset", str(output), *(str(p) for p in paths)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["split_counts"] == {"train": 1, "validation": 1, "test": 1}
    before = output.read_bytes()
    assert main(["verify-dataset", str(output)]) == 0
    capsys.readouterr()
    assert main(["freeze-dataset", str(output), str(paths[0])]) == 2
    assert output.read_bytes() == before
    assert "Dataset error" in capsys.readouterr().err


def test_index_detects_replaced_data_and_edited_policy(tmp_path):
    path = recording(tmp_path, "S8")
    index = tmp_path / "index.json"
    freeze_dataset([path], index)
    original = path.read_bytes()
    path.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_dataset(index)
    path.write_bytes(original)
    manifest = json.loads(index.read_text())
    manifest["subject_splits"]["S8"] = "test"
    index.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="policy"):
        verify_dataset(index)


def test_wrong_split_and_duplicate_recordings_rejected_before_index_write(tmp_path):
    wrong = recording(tmp_path, "S9", split="train")
    index = tmp_path / "index.json"
    with pytest.raises(ValueError, match="split"):
        freeze_dataset([wrong], index)
    good = recording(tmp_path, "S8")
    with pytest.raises(ValueError, match="duplicate"):
        freeze_dataset([good, good], index)
    assert not index.exists()


def test_split_policy_cannot_be_mutated():
    with pytest.raises(TypeError):
        SUBJECT_SPLITS["S8"] = "test"
