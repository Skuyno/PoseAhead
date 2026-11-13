"""Tests for the canonical pose representation."""

from __future__ import annotations

import numpy as np
import pytest

from poseahead.core import JOINT_COUNT, JOINT_PARENTS, Joint, PoseFrame
from poseahead.core.pose_frame import camera_to_unity, unity_to_camera


def make_frame(**overrides: object) -> PoseFrame:
    values: dict[str, object] = {
        "frame_id": 7,
        "source_timestamp_ns": 1_000,
        "produced_timestamp_ns": 1_500,
        "joint_positions": np.zeros((JOINT_COUNT, 3), dtype=np.float32),
        "joint_confidence": np.ones(JOINT_COUNT, dtype=np.float32),
        "joint_validity": np.ones(JOINT_COUNT, dtype=np.bool_),
    }
    values.update(overrides)
    return PoseFrame(**values)  # type: ignore[arg-type]


def test_joint_order_and_hierarchy_are_stable() -> None:
    assert len(Joint) == JOINT_COUNT == 17
    assert [joint.value for joint in Joint] == list(range(JOINT_COUNT))
    assert JOINT_PARENTS[Joint.PELVIS] == -1
    assert JOINT_PARENTS[Joint.LEFT_WRIST] == Joint.LEFT_ELBOW
    assert JOINT_PARENTS[Joint.RIGHT_ANKLE] == Joint.RIGHT_KNEE


def test_frame_copies_and_freezes_arrays() -> None:
    positions = np.zeros((JOINT_COUNT, 3), dtype=np.float64)
    frame = make_frame(joint_positions=positions)
    positions[Joint.HEAD, 0] = 99.0

    assert frame.joint_positions.dtype == np.float32
    assert frame.joint_positions[Joint.HEAD, 0] == 0.0
    assert not frame.joint_positions.flags.writeable
    with pytest.raises(ValueError):
        frame.joint_positions.setflags(write=True)


def test_nested_metadata_is_immutable_and_preserved_by_serialization() -> None:
    from poseahead.core.serialization import decode_pose_frame, encode_pose_frame

    source = {"nested": {"method": "raw", "thresholds": [0.1, 0.2]}}
    frame = make_frame(metadata=source)
    encoded = encode_pose_frame(frame)
    source["nested"]["thresholds"][0] = 99
    assert encode_pose_frame(frame) == encoded
    with pytest.raises(TypeError):
        frame.metadata["nested"]["method"] = "mutated"
    with pytest.raises(TypeError):
        frame.metadata["nested"]["thresholds"][0] = 99
    assert decode_pose_frame(encoded).metadata == frame.metadata


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("joint_positions", np.zeros((16, 3)), "shape"),
        ("joint_confidence", np.full(JOINT_COUNT, 1.1), r"\[0, 1\]"),
        ("produced_timestamp_ns", 999, "must not precede"),
    ],
)
def test_frame_rejects_invalid_values(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_frame(**{field: value})


def test_pelvis_must_be_origin() -> None:
    positions = np.zeros((JOINT_COUNT, 3), dtype=np.float32)
    positions[Joint.PELVIS] = [0.1, 0.0, 0.0]
    with pytest.raises(ValueError, match="pelvis"):
        make_frame(joint_positions=positions)


def test_camera_unity_conversion_round_trip() -> None:
    camera = np.array([[1.0, 2.0, 3.0], [-4.0, -5.0, 6.0]], dtype=np.float32)
    unity = camera_to_unity(camera)

    np.testing.assert_array_equal(unity, [[1.0, -2.0, 3.0], [-4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(unity_to_camera(unity), camera)


@pytest.mark.parametrize("value", [np.nan, np.inf, -1, 2, 0.5, "False", None])
def test_invalid_validity_mask_is_not_coerced_to_true(value):
    with pytest.raises(ValueError, match="joint_validity"):
        make_frame(joint_validity=[value] * JOINT_COUNT)


def test_numeric_binary_validity_is_supported():
    mask = np.arange(JOINT_COUNT) % 2
    np.testing.assert_array_equal(make_frame(joint_validity=mask).joint_validity, mask == 1)


@pytest.mark.parametrize("rotation", [[0, 0, 0, 0], [0, 0, 0, 2], [1e30, 0, 0, 0]])
@pytest.mark.parametrize("field", ["root_rotation", "joint_rotations"])
def test_non_unit_quaternions_are_rejected(rotation, field):
    value = np.tile(rotation, (JOINT_COUNT, 1)) if field == "joint_rotations" else rotation
    with pytest.raises(ValueError, match="unit quaternions"):
        make_frame(**{field: value})


def test_quaternion_rounding_is_accepted_without_silent_normalization():
    frame = make_frame(root_rotation=[0, 0, 0, 1.00001])
    assert frame.root_rotation[3] > 1
