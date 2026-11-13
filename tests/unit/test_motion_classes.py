import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from poseahead.core import PoseFrame, PoseSequence
from poseahead.datasets.index import freeze_dataset
from poseahead.recording import save_recording


@pytest.fixture
def classifier():
    path = Path(__file__).resolve().parents[2] / "scripts/cmu_motion_classes.py"
    spec = importlib.util.spec_from_file_location("motion_classes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ground_truth(subject, *, acceleration=1., count=240):
    frames = []
    for i in range(count):
        positions = np.zeros((17, 3))
        positions[1:, 0] = .5 * acceleration * (i * .025)**2
        frames.append(PoseFrame(i, i * 25_000_000, i * 25_000_000, positions,
                                np.ones(17), np.ones(17, bool), metadata={
                                    "dataset": "CMU", "subject": subject,
                                    "split": "train" if subject == "CMU:02" else "validation",
                                    "skeleton_fingerprint": subject[-2:] * 32,
                                }))
    return PoseSequence(f"motion/{subject}", "test", tuple(frames))


def arrays(sequence):
    return (np.stack([f.joint_positions for f in sequence.frames]),
            np.stack([f.joint_validity for f in sequence.frames]),
            tuple(f.source_timestamp_ns for f in sequence.frames))


def test_fixed_filter_derivative_and_one_second_windows(classifier):
    sequence = ground_truth("CMU:02", acceleration=2.)
    scores = classifier.window_scores(*arrays(sequence))
    assert len(scores) == 6
    assert scores[1]["score_m_s2"] == pytest.approx(2., abs=2e-4)
    assert scores[1]["start_ns"] == 1_000_000_000
    assert scores[1]["end_ns"] == 1_975_000_000
    stationary = ground_truth("CMU:02", acceleration=0.)
    assert all(s["score_m_s2"] == 0 for s in classifier.window_scores(*arrays(stationary)))


def test_invalid_support_does_not_smear_missing_ground_truth_into_valid_class(classifier):
    sequence = ground_truth("CMU:02")
    positions, valid, times = arrays(sequence)
    valid[39, 2] = False
    scores = classifier.window_scores(positions, valid, times)
    assert scores[0]["score_m_s2"] is None
    assert scores[1]["score_m_s2"] is None
    assert scores[2]["score_m_s2"] is not None


def test_warmup_window_phase_filters_whole_gt_not_the_cropped_edge(classifier):
    sequence = ground_truth("CMU:02", acceleration=2.)
    assert classifier.sequence_scores(sequence, 500_000_000) == classifier.window_scores(
        *arrays(sequence), window_offset=20,
    )


@pytest.mark.parametrize("score, expected", [
    (None, "unknown"), (1., "smooth"), (1.5, "medium"), (2., "high-dynamic/abrupt"),
])
def test_class_boundaries_do_not_use_method_errors(classifier, score, expected):
    assert classifier.motion_class(score, 1., 2.) == expected
    assert classifier.motion_class(0., 0., 0.) == "smooth"


@pytest.mark.parametrize("score, low, high", [
    (float("nan"), 1, 2), (-1, 1, 2), (1, 2, 1), (1, 1, float("inf")),
])
def test_invalid_scores_thresholds_fail(classifier, score, low, high):
    with pytest.raises(ValueError):
        classifier.motion_class(score, low, high)


def test_train_only_freeze_validation_labels_and_no_overwrite(classifier, tmp_path):
    paths = []
    for i, acceleration in enumerate((0., 2.)):
        sequence = ground_truth("CMU:02", acceleration=acceleration)
        sequence = PoseSequence(f"train{i}", sequence.clock_domain, sequence.frames)
        path = tmp_path / f"train{i}.npz"
        save_recording(path, sequence)
        paths.append(path)
    validation = tmp_path / "validation.npz"
    save_recording(validation, ground_truth("CMU:13", acceleration=10.))
    paths.append(validation)
    # A test sequence is present but its motion is never classified/scored.
    source = ground_truth("CMU:13", acceleration=1e3)
    test = PoseSequence("untouched-test", source.clock_domain, tuple(
        replace(f, metadata={**f.metadata, "subject": "CMU:14", "split": "test",
                              "skeleton_fingerprint": "14" * 32}) for f in source.frames))
    test_path = tmp_path / "test.npz"
    save_recording(test_path, test)
    paths.append(test_path)
    index = tmp_path / "index.json"
    freeze_dataset(paths, index)
    calibration = tmp_path / "classes.json"
    frozen = classifier.freeze_classes(index, calibration)
    assert frozen["training_valid_windows"] == 12
    assert {s["sequence_id"] for s in frozen["training_sequences"]} == {"train0", "train1"}
    assert frozen["abrupt_lower_m_s2"] < 3
    assert frozen["test_scored"] is False
    before = calibration.read_bytes()
    with pytest.raises(FileExistsError):
        classifier.freeze_classes(index, calibration)
    assert calibration.read_bytes() == before
    labels = classifier.label_validation(calibration, tmp_path / "labels.json")
    assert len(labels["sequences"]) == 1
    assert labels["sequences"][0]["sequence_id"] == "motion/CMU:13"
    assert labels["sequences"][0]["counts"]["high-dynamic/abrupt"] > 0
    assert labels["test_scored"] is False
    with pytest.raises(FileExistsError):
        classifier.label_validation(calibration, tmp_path / "labels.json")
    changed = dict(frozen, smooth_upper_m_s2=123.)
    calibration.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="thresholds"):
        classifier.label_validation(calibration, tmp_path / "wrong.json")
    assert not (tmp_path / "wrong.json").exists()
