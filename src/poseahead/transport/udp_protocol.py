"""PoseAhead binary UDP protocol version 2, with ordered sender sessions."""

from __future__ import annotations

import socket
import struct
import time
import zlib
from dataclasses import dataclass
from enum import IntEnum
from typing import Final

import numpy as np

from poseahead.core import JOINT_COUNT, PoseFrame

MAGIC: Final = b"PAUD"
VERSION: Final = 2
_PROCESS_SESSION_ID = time.perf_counter_ns()
MAX_DATAGRAM_SIZE: Final = 1_200
_ROOT_POSITION = 1 << 0
_ROOT_ROTATION = 1 << 1
_JOINT_ROTATIONS = 1 << 2
_PREDICTABILITY = 1 << 3
_ESTIMATED_ROOT = 1 << 4
_KNOWN_FLAGS = (
    _ROOT_POSITION | _ROOT_ROTATION | _JOINT_ROTATIONS | _PREDICTABILITY | _ESTIMATED_ROOT
)
_HEADER = struct.Struct("<4sBBBBQQQQQhH")
_CRC = struct.Struct("<I")
_VALIDITY_BYTES = (JOINT_COUNT + 7) // 8


class Algorithm(IntEnum):
    """Stable IDs for methods exposed to the Unity overlay."""

    RAW = 0
    HOLD = 1
    LINEAR = 2
    DOUBLE_EXPONENTIAL = 3
    KALMAN = 4
    EMA = 5
    ONE_EURO = 6
    KALMAN_EXTRAPOLATION = 7
    PREDICTOR_33_MS = 10
    PREDICTOR_66_MS = 11
    PREDICTOR_100_MS = 12
    PREDICTOR_150_MS = 13
    ADAPTIVE = 20


@dataclass(frozen=True, slots=True)
class UdpPosePacket:
    frame: PoseFrame
    session_id: int
    send_timestamp_ns: int
    algorithm: Algorithm
    target_offset_ms: int
    predictability: float | None


def encode_udp_packet(
    frame: PoseFrame,
    *,
    algorithm: Algorithm = Algorithm.RAW,
    target_offset_ms: int = 0,
    predictability: float | None = None,
    send_timestamp_ns: int | None = None,
    session_id: int | None = None,
) -> bytes:
    """Encode one PoseFrame into one checksummed little-endian datagram."""
    send_ns = time.perf_counter_ns() if send_timestamp_ns is None else send_timestamp_ns
    session = _PROCESS_SESSION_ID if session_id is None else session_id
    if isinstance(algorithm, bool) or not isinstance(algorithm, int):
        raise ValueError("algorithm must be a supported integer ID")
    try:
        algorithm = Algorithm(algorithm)
    except ValueError as error:
        raise ValueError(f"unsupported algorithm ID: {algorithm}") from error
    if isinstance(session, bool) or not isinstance(session, int) or not 0 < session < 2**64:
        raise ValueError("session_id must be a positive uint64")
    if isinstance(send_ns, bool) or not isinstance(send_ns, int) or not 0 <= send_ns < 2**64:
        raise ValueError("send_timestamp_ns must be a non-negative uint64 integer")
    if send_ns < frame.produced_timestamp_ns:
        raise ValueError("send_timestamp_ns must not precede produced_timestamp_ns")
    if type(target_offset_ms) is not int or not -(2**15) <= target_offset_ms < 2**15:
        raise ValueError("target_offset_ms does not fit int16")
    if predictability is not None and (
        type(predictability) not in (float, int) or not 0.0 <= predictability <= 1.0
    ):
        raise ValueError("predictability must be in [0, 1]")

    flags = 0
    flags |= _ROOT_POSITION if frame.root_position is not None else 0
    flags |= _ROOT_ROTATION if frame.root_rotation is not None else 0
    flags |= _JOINT_ROTATIONS if frame.joint_rotations is not None else 0
    flags |= _PREDICTABILITY if predictability is not None else 0
    flags |= (_ESTIMATED_ROOT if frame.root_position is not None
              and frame.metadata.get("root_translation_status") == "estimated_monocular" else 0)
    parts = [
        _HEADER.pack(
            MAGIC,
            VERSION,
            flags,
            JOINT_COUNT,
            int(algorithm),
            session,
            frame.frame_id,
            frame.source_timestamp_ns,
            frame.produced_timestamp_ns,
            send_ns,
            target_offset_ms,
            0,
        )
    ]
    if predictability is not None:
        parts.append(struct.pack("<f", predictability))
    if frame.root_position is not None:
        parts.append(np.asarray(frame.root_position, dtype="<f4").tobytes())
    if frame.root_rotation is not None:
        parts.append(np.asarray(frame.root_rotation, dtype="<f4").tobytes())
    parts.append(np.asarray(frame.joint_positions, dtype="<f4").tobytes())
    if frame.joint_rotations is not None:
        parts.append(np.asarray(frame.joint_rotations, dtype="<f4").tobytes())
    parts.append(np.asarray(frame.joint_confidence, dtype="<f4").tobytes())
    parts.append(np.packbits(frame.joint_validity, bitorder="little").tobytes())
    payload = b"".join(parts)
    packet = payload + _CRC.pack(zlib.crc32(payload))
    if len(packet) > MAX_DATAGRAM_SIZE:
        raise ValueError(f"UDP packet exceeds {MAX_DATAGRAM_SIZE} bytes")
    return packet


def decode_udp_packet(data: bytes) -> UdpPosePacket:
    """Decode a v2 packet, rejecting corruption and incompatible fields."""
    if len(data) > MAX_DATAGRAM_SIZE:
        raise ValueError("UDP pose packet exceeds the datagram limit")
    if len(data) < _HEADER.size + _CRC.size:
        raise ValueError("truncated UDP pose packet")
    payload = data[: -_CRC.size]
    expected_crc = _CRC.unpack_from(data, len(payload))[0]
    if zlib.crc32(payload) != expected_crc:
        raise ValueError("UDP pose packet CRC mismatch")
    (
        magic,
        version,
        flags,
        joint_count,
        algorithm_id,
        session_id,
        frame_id,
        source_ns,
        produced_ns,
        send_ns,
        target_offset_ms,
        reserved,
    ) = _HEADER.unpack_from(payload)
    if magic != MAGIC:
        raise ValueError("invalid UDP pose magic")
    if version != VERSION:
        raise ValueError(f"unsupported UDP pose version: {version}")
    if flags & ~_KNOWN_FLAGS:
        raise ValueError("unknown UDP pose flags")
    if flags & _ESTIMATED_ROOT and not flags & _ROOT_POSITION:
        raise ValueError("estimated root flag requires a root position")
    if joint_count != JOINT_COUNT:
        raise ValueError(f"unsupported joint count: {joint_count}")
    if reserved != 0:
        raise ValueError("reserved UDP pose header field must be zero")
    if session_id == 0:
        raise ValueError("session_id must be positive")
    try:
        algorithm = Algorithm(algorithm_id)
    except ValueError as error:
        raise ValueError(f"unsupported algorithm ID: {algorithm_id}") from error
    if produced_ns < source_ns or send_ns < produced_ns:
        raise ValueError("UDP pose timestamps are not monotonic")

    offset = _HEADER.size

    def take_float32(count: int) -> np.ndarray:
        nonlocal offset
        byte_count = count * 4
        if offset + byte_count > len(payload):
            raise ValueError("truncated UDP pose payload")
        result = np.frombuffer(payload, dtype="<f4", count=count, offset=offset).copy()
        offset += byte_count
        return result

    predictability = float(take_float32(1)[0]) if flags & _PREDICTABILITY else None
    if predictability is not None and not 0.0 <= predictability <= 1.0:
        raise ValueError("UDP pose predictability is outside [0, 1]")
    root_position = take_float32(3) if flags & _ROOT_POSITION else None
    root_rotation = take_float32(4) if flags & _ROOT_ROTATION else None
    positions = take_float32(JOINT_COUNT * 3).reshape(JOINT_COUNT, 3)
    rotations = (
        take_float32(JOINT_COUNT * 4).reshape(JOINT_COUNT, 4) if flags & _JOINT_ROTATIONS else None
    )
    confidence = take_float32(JOINT_COUNT)
    if offset + _VALIDITY_BYTES != len(payload):
        raise ValueError("UDP pose packet length mismatch")
    if payload[-1] & 0xFE:
        raise ValueError("unused validity bits must be zero")
    validity = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8, count=_VALIDITY_BYTES, offset=offset),
        bitorder="little",
    )[:JOINT_COUNT].astype(np.bool_)

    frame = PoseFrame(
        frame_id=frame_id,
        source_timestamp_ns=source_ns,
        produced_timestamp_ns=produced_ns,
        root_position=root_position,
        root_rotation=root_rotation,
        joint_positions=positions,
        joint_rotations=rotations,
        joint_confidence=confidence,
        joint_validity=validity,
        metadata={"transport": "udp-v2", **(
            {"root_translation_status": "estimated_monocular"} if flags & _ESTIMATED_ROOT else {}
        )},
    )
    return UdpPosePacket(frame, session_id, send_ns, algorithm, target_offset_ms, predictability)


class UdpPoseSender:
    """Small stateless sender; one call always maps to one datagram."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5005) -> None:
        self._target = (host, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.session_id = time.perf_counter_ns()

    def send(self, frame: PoseFrame, **packet_fields: object) -> int:
        packet = encode_udp_packet(
            frame,
            session_id=self.session_id,
            **packet_fields,  # type: ignore[arg-type]
        )
        return self._socket.sendto(packet, self._target)

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> UdpPoseSender:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
