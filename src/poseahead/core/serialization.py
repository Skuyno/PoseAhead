"""Versioned serialization for canonical PoseFrames.

This storage representation is independent of the later UDP transport policy.
"""

from __future__ import annotations

import json
import struct
from typing import Final

import numpy as np

from poseahead.core.joints import JOINT_COUNT
from poseahead.core.metadata import metadata_to_json
from poseahead.core.pose_frame import PoseFrame

MAGIC: Final = b"PAFR"
VERSION: Final = 1
_ROOT_POSITION = 1 << 0
_ROOT_ROTATION = 1 << 1
_JOINT_ROTATIONS = 1 << 2
_HEADER = struct.Struct("<4sBBHQQQI")


def encode_pose_frame(frame: PoseFrame) -> bytes:
    """Encode a frame into deterministic little-endian bytes."""
    flags = 0
    flags |= _ROOT_POSITION if frame.root_position is not None else 0
    flags |= _ROOT_ROTATION if frame.root_rotation is not None else 0
    flags |= _JOINT_ROTATIONS if frame.joint_rotations is not None else 0
    metadata = json.dumps(
        metadata_to_json(frame.metadata),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    parts = [
        _HEADER.pack(
            MAGIC,
            VERSION,
            flags,
            JOINT_COUNT,
            frame.frame_id,
            frame.source_timestamp_ns,
            frame.produced_timestamp_ns,
            len(metadata),
        ),
        np.asarray(frame.joint_positions, dtype="<f4").tobytes(),
        np.asarray(frame.joint_confidence, dtype="<f4").tobytes(),
        np.packbits(frame.joint_validity, bitorder="little").tobytes(),
    ]
    if frame.root_position is not None:
        parts.append(np.asarray(frame.root_position, dtype="<f4").tobytes())
    if frame.root_rotation is not None:
        parts.append(np.asarray(frame.root_rotation, dtype="<f4").tobytes())
    if frame.joint_rotations is not None:
        parts.append(np.asarray(frame.joint_rotations, dtype="<f4").tobytes())
    parts.append(metadata)
    return b"".join(parts)


def decode_pose_frame(data: bytes) -> PoseFrame:
    """Decode bytes and reject unknown, truncated, or trailing data."""
    if len(data) < _HEADER.size:
        raise ValueError("truncated PoseFrame header")
    magic, version, flags, joint_count, frame_id, source_ns, produced_ns, metadata_size = (
        _HEADER.unpack_from(data)
    )
    if magic != MAGIC:
        raise ValueError("invalid PoseFrame magic")
    if version != VERSION:
        raise ValueError(f"unsupported PoseFrame version: {version}")
    if joint_count != JOINT_COUNT:
        raise ValueError(f"unsupported joint count: {joint_count}")
    if flags & ~(_ROOT_POSITION | _ROOT_ROTATION | _JOINT_ROTATIONS):
        raise ValueError("unknown PoseFrame flags")

    offset = _HEADER.size

    def take_float32(count: int) -> np.ndarray:
        nonlocal offset
        byte_count = count * 4
        if offset + byte_count > len(data):
            raise ValueError("truncated PoseFrame payload")
        result = np.frombuffer(data, dtype="<f4", count=count, offset=offset).copy()
        offset += byte_count
        return result

    positions = take_float32(JOINT_COUNT * 3).reshape(JOINT_COUNT, 3)
    confidence = take_float32(JOINT_COUNT)
    validity_size = (JOINT_COUNT + 7) // 8
    if offset + validity_size > len(data):
        raise ValueError("truncated PoseFrame validity mask")
    if data[offset + validity_size - 1] & 0xFE:
        raise ValueError("unused validity bits must be zero")
    validity = np.unpackbits(
        np.frombuffer(data, dtype=np.uint8, count=validity_size, offset=offset),
        bitorder="little",
    )[:JOINT_COUNT].astype(np.bool_)
    offset += validity_size

    root_position = take_float32(3) if flags & _ROOT_POSITION else None
    root_rotation = take_float32(4) if flags & _ROOT_ROTATION else None
    rotations = (
        take_float32(JOINT_COUNT * 4).reshape(JOINT_COUNT, 4) if flags & _JOINT_ROTATIONS else None
    )
    if offset + metadata_size != len(data):
        raise ValueError("PoseFrame metadata length mismatch or trailing bytes")
    try:
        metadata = json.loads(data[offset:].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid PoseFrame metadata") from error
    if not isinstance(metadata, dict):
        raise ValueError("PoseFrame metadata must be a JSON object")

    return PoseFrame(
        frame_id=frame_id,
        source_timestamp_ns=source_ns,
        produced_timestamp_ns=produced_ns,
        joint_positions=positions,
        joint_confidence=confidence,
        joint_validity=validity,
        root_position=root_position,
        root_rotation=root_rotation,
        joint_rotations=rotations,
        metadata=metadata,
    )
