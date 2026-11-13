import numpy as np
import pytest

from poseahead.core import Joint
from poseahead.perception.metrabs_mapping import metrabs_to_pose_frame


def _pose() -> np.ndarray:
    pose = np.arange(17 * 3, dtype=np.float32).reshape(17, 3) * 10
    pose[:, 2] += 3000
    return pose


def test_absolute_millimetres_map_to_relative_canonical_metres():
    source = _pose()
    frame = metrabs_to_pose_frame(
        source,
        detection_confidence=0.8,
        confidence_threshold=0.3,
        frame_id=7,
        source_timestamp_ns=10,
        produced_timestamp_ns=20,
    )
    root = source[0] / 1000
    np.testing.assert_allclose(frame.root_position, root)
    np.testing.assert_allclose(frame.joint_positions[Joint.PELVIS], 0)
    np.testing.assert_allclose(frame.joint_positions[Joint.LEFT_KNEE], source[5] / 1000 - root)
    expected_chest = (source[7] + source[8]) / 2000 - root
    np.testing.assert_allclose(frame.joint_positions[Joint.CHEST], expected_chest)
    assert frame.is_valid
    assert np.all(frame.joint_confidence == np.float32(0.8))
    assert frame.metadata["confidence_semantics"].startswith("person_detection_proxy")


def test_low_person_confidence_withholds_pose_and_root():
    frame = metrabs_to_pose_frame(
        _pose(),
        detection_confidence=0.2,
        confidence_threshold=0.3,
        frame_id=0,
        source_timestamp_ns=0,
        produced_timestamp_ns=1,
    )
    assert frame.root_position is None
    assert not frame.joint_validity.any()
    assert not frame.joint_confidence.any()


def test_nonfinite_joint_is_marked_invalid_without_leaking_nan():
    source = _pose()
    source[13, 0] = np.nan
    frame = metrabs_to_pose_frame(
        source,
        detection_confidence=0.9,
        confidence_threshold=0.3,
        frame_id=0,
        source_timestamp_ns=0,
        produced_timestamp_ns=1,
    )
    assert not frame.joint_validity[Joint.LEFT_WRIST]
    assert np.isfinite(frame.joint_positions).all()


@pytest.mark.parametrize(
    "pose,confidence",
    [
        (np.zeros((16, 3)), 0.5),
        (np.zeros((17, 2)), 0.5),
        (np.zeros((17, 3)), float("nan")),
        (np.zeros((17, 3)), 2),
    ],
)
def test_bad_inputs_are_rejected(pose, confidence):
    with pytest.raises(ValueError):
        metrabs_to_pose_frame(
            pose,
            detection_confidence=confidence,
            confidence_threshold=0.3,
            frame_id=0,
            source_timestamp_ns=0,
            produced_timestamp_ns=1,
        )
