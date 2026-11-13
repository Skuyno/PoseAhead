"""Golden behavior tests for UDP protocol version 2."""

from __future__ import annotations

import base64
import itertools
import struct
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from poseahead.core import JOINT_COUNT, PoseFrame
from poseahead.transport import (
    MAX_DATAGRAM_SIZE,
    Algorithm,
    decode_udp_packet,
    encode_udp_packet,
)


def make_frame(*, rotations: bool = False) -> PoseFrame:
    positions = np.arange(JOINT_COUNT * 3, dtype=np.float32).reshape(JOINT_COUNT, 3)
    positions = (positions - positions[0]) / 100.0
    return PoseFrame(
        frame_id=123,
        source_timestamp_ns=1_000,
        produced_timestamp_ns=1_500,
        root_position=[0.1, 0.2, 2.5],
        root_rotation=[0.0, 0.0, 0.0, 1.0] if rotations else None,
        joint_positions=positions,
        joint_rotations=(np.tile([0.0, 0.0, 0.0, 1.0], (JOINT_COUNT, 1)) if rotations else None),
        joint_confidence=np.linspace(0.0, 1.0, JOINT_COUNT),
        joint_validity=np.arange(JOINT_COUNT) % 3 != 0,
    )


@pytest.mark.parametrize("rotations", [False, True])
def test_udp_packet_round_trip_and_size(rotations: bool) -> None:
    source = make_frame(rotations=rotations)
    encoded = encode_udp_packet(
        source,
        algorithm=Algorithm.PREDICTOR_100_MS,
        target_offset_ms=100,
        predictability=0.75,
        send_timestamp_ns=2_000,
        session_id=500,
    )
    decoded = decode_udp_packet(encoded)

    assert len(encoded) <= MAX_DATAGRAM_SIZE
    assert decoded.send_timestamp_ns == 2_000
    assert decoded.session_id == 500
    assert decoded.algorithm == Algorithm.PREDICTOR_100_MS
    assert decoded.target_offset_ms == 100
    assert decoded.predictability == pytest.approx(0.75)
    np.testing.assert_array_equal(decoded.frame.joint_positions, source.joint_positions)
    np.testing.assert_array_equal(decoded.frame.joint_confidence, source.joint_confidence)
    np.testing.assert_array_equal(decoded.frame.joint_validity, source.joint_validity)
    np.testing.assert_array_equal(decoded.frame.root_position, source.root_position)


def test_current_position_only_packet_is_small() -> None:
    packet = encode_udp_packet(make_frame(), send_timestamp_ns=2_000)
    assert len(packet) == 343


def test_decoder_rejects_corruption() -> None:
    packet = bytearray(encode_udp_packet(make_frame(), send_timestamp_ns=2_000))
    packet[50] ^= 0xFF
    with pytest.raises(ValueError, match="CRC"):
        decode_udp_packet(bytes(packet))


def test_decoder_rejects_unknown_version_even_with_valid_crc() -> None:
    packet = bytearray(encode_udp_packet(make_frame(), send_timestamp_ns=2_000))
    packet[4] = 99
    packet[-4:] = struct.pack("<I", __import__("zlib").crc32(packet[:-4]))
    with pytest.raises(ValueError, match="version"):
        decode_udp_packet(bytes(packet))


def test_python_encoder_matches_cross_language_golden_packet() -> None:
    golden_path = Path(__file__).parents[1] / "fixtures" / "udp_v2_golden.base64"
    golden = base64.b64decode(golden_path.read_text(encoding="ascii"))
    frame = PoseFrame(
        frame_id=1,
        source_timestamp_ns=100,
        produced_timestamp_ns=200,
        root_position=[1.0, 2.0, 3.0],
        joint_positions=np.zeros((JOINT_COUNT, 3)),
        joint_confidence=np.ones(JOINT_COUNT),
        joint_validity=np.ones(JOINT_COUNT, dtype=np.bool_),
    )
    assert encode_udp_packet(frame, send_timestamp_ns=300, session_id=42) == golden


def test_v1_packets_are_explicitly_rejected() -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "udp_v1_golden.base64"
    with pytest.raises(ValueError, match="version"):
        decode_udp_packet(base64.b64decode(fixture.read_text(encoding="ascii")))


def test_full_cross_language_fixture_matches_python_encoder():
    positions = np.arange(51, dtype=np.float32).reshape(17, 3) * 0.01
    positions -= np.array([0, 0.01, 0.02], dtype=np.float32)
    frame = PoseFrame(
        5, 100, 200, positions, np.ones(17), np.arange(17) % 2 == 0,
        root_position=[1, 2, 3], root_rotation=[0, 0, 0, 1],
        joint_rotations=np.tile([0, 0, 0, 1], (17, 1)),
    )
    fixture = Path(__file__).parents[1] / "fixtures" / "udp_v2_full_golden.base64"
    data = encode_udp_packet(
        frame, session_id=42, send_timestamp_ns=300, algorithm=Algorithm.KALMAN_EXTRAPOLATION,
        target_offset_ms=100, predictability=0.75,
    )
    assert len(data) == 635
    assert data == base64.b64decode(fixture.read_text(encoding="ascii"))


@pytest.mark.parametrize("flags", itertools.product((False, True), repeat=4))
def test_all_optional_field_combinations(flags):
    root, rotation, joints, predictability = flags
    frame = make_frame(rotations=True)
    frame = replace(frame, root_position=frame.root_position if root else None,
                    root_rotation=frame.root_rotation if rotation else None,
                    joint_rotations=frame.joint_rotations if joints else None)
    data = encode_udp_packet(frame, send_timestamp_ns=2000,
                             predictability=0.5 if predictability else None)
    assert len(data) == 331 + 12 * root + 16 * rotation + 272 * joints + 4 * predictability
    decoded = decode_udp_packet(data)
    for field in ("root_position", "root_rotation", "joint_rotations", "joint_positions",
                  "joint_confidence", "joint_validity"):
        np.testing.assert_equal(getattr(decoded.frame, field), getattr(frame, field))


@pytest.mark.parametrize("algorithm", [Algorithm.EMA, Algorithm.ONE_EURO,
                                       Algorithm.KALMAN_EXTRAPOLATION])
def test_required_baselines_have_distinct_round_trip_ids(algorithm):
    packet = encode_udp_packet(make_frame(), algorithm=algorithm, send_timestamp_ns=2000)
    assert decode_udp_packet(packet).algorithm == algorithm


@pytest.mark.parametrize("fields", [
    {"algorithm": 255}, {"algorithm": True}, {"algorithm": 1.5},
    {"send_timestamp_ns": -1}, {"send_timestamp_ns": 2**64},
    {"send_timestamp_ns": float("nan")}, {"send_timestamp_ns": True},
    {"target_offset_ms": 1.5}, {"target_offset_ms": True}, {"target_offset_ms": 32768},
    {"predictability": float("nan")}, {"predictability": True},
])
def test_encoder_rejects_invalid_header_fields_with_value_error(fields):
    with pytest.raises(ValueError):
        encode_udp_packet(make_frame(), **fields)


@pytest.mark.parametrize("offset, value", [(80, 0.0), (300, 2.0)])
def test_decoder_rejects_invalid_quaternions_even_with_valid_crc(offset, value):
    fixture = Path(__file__).parents[1] / "fixtures" / "udp_v2_full_golden.base64"
    data = bytearray(base64.b64decode(fixture.read_text(encoding="ascii")))
    struct.pack_into("<f", data, offset, value)
    struct.pack_into("<I", data, len(data) - 4, zlib.crc32(data[:-4]))
    with pytest.raises(ValueError, match="unit quaternions"):
        decode_udp_packet(bytes(data))


def test_decoder_rejects_nonzero_validity_padding():
    data = bytearray(encode_udp_packet(make_frame(), send_timestamp_ns=2000))
    data[-5] |= 2
    struct.pack_into("<I", data, len(data) - 4, zlib.crc32(data[:-4]))
    with pytest.raises(ValueError, match="unused validity bits"):
        decode_udp_packet(bytes(data))


def test_estimated_root_flag_requires_root_payload():
    frame = replace(make_frame(), root_position=None)
    data = bytearray(encode_udp_packet(frame, send_timestamp_ns=2000))
    data[5] |= 16
    struct.pack_into("<I", data, len(data) - 4, zlib.crc32(data[:-4]))
    with pytest.raises(ValueError, match="estimated root flag requires"):
        decode_udp_packet(bytes(data))
