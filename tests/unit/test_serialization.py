"""Tests for versioned PoseFrame serialization."""

from __future__ import annotations

import numpy as np
import pytest

from poseahead.core import JOINT_COUNT, PoseFrame
from poseahead.core.serialization import decode_pose_frame, encode_pose_frame


def test_serialization_round_trip_with_optional_fields() -> None:
    positions = np.arange(JOINT_COUNT * 3, dtype=np.float32).reshape(JOINT_COUNT, 3)
    positions -= positions[0]
    frame = PoseFrame(
        frame_id=42,
        source_timestamp_ns=10_000,
        produced_timestamp_ns=12_000,
        root_position=np.array([0.1, -0.2, 3.0]),
        root_rotation=np.array([0.0, 0.0, 0.0, 1.0]),
        joint_positions=positions,
        joint_rotations=np.tile([0.0, 0.0, 0.0, 1.0], (JOINT_COUNT, 1)),
        joint_confidence=np.linspace(0.1, 0.9, JOINT_COUNT),
        joint_validity=np.arange(JOINT_COUNT) % 2 == 0,
        metadata={"method": "raw", "unicode": "поза"},
    )

    encoded = encode_pose_frame(frame)
    decoded = decode_pose_frame(encoded)

    assert decoded.frame_id == frame.frame_id
    assert decoded.source_timestamp_ns == frame.source_timestamp_ns
    assert decoded.produced_timestamp_ns == frame.produced_timestamp_ns
    assert decoded.metadata == frame.metadata
    np.testing.assert_array_equal(decoded.joint_positions, frame.joint_positions)
    np.testing.assert_array_equal(decoded.joint_confidence, frame.joint_confidence)
    np.testing.assert_array_equal(decoded.joint_validity, frame.joint_validity)
    np.testing.assert_array_equal(decoded.root_position, frame.root_position)
    np.testing.assert_array_equal(decoded.root_rotation, frame.root_rotation)
    np.testing.assert_array_equal(decoded.joint_rotations, frame.joint_rotations)
    assert len(encoded) < 1_200


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data[:10],
        lambda data: b"NOPE" + data[4:],
        lambda data: data + b"trailing",
    ],
)
def test_decoder_rejects_malformed_data(mutation: object) -> None:
    frame = PoseFrame(
        frame_id=0,
        source_timestamp_ns=1,
        produced_timestamp_ns=1,
        joint_positions=np.zeros((JOINT_COUNT, 3)),
        joint_confidence=np.ones(JOINT_COUNT),
        joint_validity=np.ones(JOINT_COUNT, dtype=np.bool_),
    )
    data = encode_pose_frame(frame)
    with pytest.raises(ValueError):
        decode_pose_frame(mutation(data))  # type: ignore[operator]


def test_storage_rejects_nonzero_validity_padding():
    frame = PoseFrame(0, 1, 1, np.zeros((17, 3)), np.ones(17), np.ones(17, dtype=bool))
    data = bytearray(encode_pose_frame(frame))
    data[36 + 204 + 68 + 2] |= 2  # header + positions + confidence + final mask byte
    with pytest.raises(ValueError, match="unused validity bits"):
        decode_pose_frame(bytes(data))
