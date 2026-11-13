from dataclasses import replace

import numpy as np
import pytest

from poseahead.core import PoseFrame, PoseSequence
from poseahead.evaluation import score_pose_error


def frame(i, milliseconds, value=0):
    p = np.zeros((17, 3))
    p[1:, 0] = value
    t = int(milliseconds * 1e6)
    return PoseFrame(i, t, t, p, np.ones(17), np.ones(17, dtype=bool),
                     root_position=[value, 0, 3])


def seq(*frames, clock="test"):
    return PoseSequence("test", clock, frames)


def test_analytic_pose_and_root_error_are_separate():
    report = score_pose_error(seq(frame(0, 0, 1)), seq(frame(0, 0)))
    assert report["mpjpe_mm"] == pytest.approx(1000 * 16 / 17)
    assert report["root_error_mm"] == 1000
    assert report["joint_coverage"] == report["root_coverage"] == 1


def test_declared_prediction_target_not_source_or_production_time_is_scored():
    prediction = replace(frame(0, 0, .033), produced_timestamp_ns=50_000_000,
                         metadata={"target_timestamp_ns": 33_000_000,
                                   "target_offset_ns": 33_000_000})
    reference = seq(frame(0, 0), frame(1, 20, .020), frame(2, 40, .040))
    exact = score_pose_error(seq(prediction), reference)
    assert exact["mpjpe_mm"] is None
    assert exact["joint_coverage"] == 0
    linear = score_pose_error(seq(prediction), reference, interpolation="linear")
    assert linear["mpjpe_mm"] == pytest.approx(0, abs=1e-5)
    assert linear["root_error_mm"] == pytest.approx(0, abs=1e-5)


def test_missing_joints_and_roots_do_not_create_zero_error_credit():
    missing = replace(frame(0, 0), joint_validity=np.zeros(17, bool), root_position=None)
    report = score_pose_error(seq(missing), seq(frame(0, 0)))
    assert report["mpjpe_mm"] is report["root_error_mm"] is None
    assert report["valid_joint_pairs"] == report["root_pairs"] == 0
    assert report["matched_reference_frames"] == 1


@pytest.mark.parametrize("target", [10, 110])
def test_no_interpolation_across_large_gaps_or_outside_reference(target):
    report = score_pose_error(seq(frame(0, target)), seq(frame(0, 0), frame(1, 100)),
                              interpolation="linear")
    assert report["matched_reference_frames"] == 0


def test_interpolation_requires_joint_valid_at_both_endpoints():
    left, right = frame(0, 0), frame(1, 20)
    validity = np.ones(17, bool)
    validity[1] = False
    right = replace(right, joint_validity=validity)
    report = score_pose_error(seq(frame(0, 10)), seq(left, right), interpolation="linear")
    assert report["valid_joint_pairs"] == 16


def test_clock_and_target_metadata_mismatch_rejected():
    with pytest.raises(ValueError, match="clock"):
        score_pose_error(seq(frame(0, 0)), seq(frame(0, 0), clock="another"))
    bad = replace(frame(0, 0), metadata={"target_timestamp_ns": 10, "target_offset_ns": 20})
    with pytest.raises(ValueError, match="disagree"):
        score_pose_error(seq(bad), seq(frame(0, 0)))
