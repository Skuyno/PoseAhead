"""Validated canonical pose frame."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from poseahead.core.joints import JOINT_COUNT, Joint
from poseahead.core.metadata import freeze_metadata

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
QUATERNION_NORM_TOLERANCE = 1e-4


def _validate_quaternions(value: FloatArray) -> None:
    norms = np.linalg.norm(value.astype(np.float64), axis=-1)
    if not np.all(np.abs(norms - 1.0) <= QUATERNION_NORM_TOLERANCE):
        raise ValueError("rotations must be unit quaternions (norm tolerance 1e-4)")


def _readonly_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> NDArray[Any]:
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    # Immutable bytes also prevent callers from re-enabling the writable flag.
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=dtype).reshape(shape)


@dataclass(frozen=True, slots=True)
class PoseFrame:
    """One pelvis-relative pose in the PoseAhead canonical coordinate system.

    Timestamps are nanoseconds from the producer's monotonic clock. Positions
    use right-handed camera coordinates in metres: +X right, +Y down, +Z away.
    """

    frame_id: int
    source_timestamp_ns: int
    produced_timestamp_ns: int
    joint_positions: FloatArray
    joint_confidence: FloatArray
    joint_validity: BoolArray
    root_position: FloatArray | None = None
    root_rotation: FloatArray | None = None
    joint_rotations: FloatArray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("frame_id", "source_timestamp_ns", "produced_timestamp_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**64:
                raise ValueError(f"{name} must be a non-negative uint64 integer")
        if self.produced_timestamp_ns < self.source_timestamp_ns:
            raise ValueError("produced_timestamp_ns must not precede source_timestamp_ns")

        positions = _readonly_array(
            self.joint_positions,
            name="joint_positions",
            shape=(JOINT_COUNT, 3),
            dtype=np.dtype(np.float32),
        )
        confidence = _readonly_array(
            self.joint_confidence,
            name="joint_confidence",
            shape=(JOINT_COUNT,),
            dtype=np.dtype(np.float32),
        )
        raw_validity = np.asarray(self.joint_validity)
        if raw_validity.dtype.kind not in "biuf" or not np.all(
            (raw_validity == 0) | (raw_validity == 1)
        ):
            raise ValueError("joint_validity must contain only booleans or numeric 0/1")
        validity = _readonly_array(
            raw_validity,
            name="joint_validity",
            shape=(JOINT_COUNT,),
            dtype=np.dtype(np.bool_),
        )

        if not np.isfinite(positions).all():
            raise ValueError("joint_positions must contain only finite values")
        if not np.isfinite(confidence).all():
            raise ValueError("joint_confidence must contain only finite values")
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("joint_confidence values must be in [0, 1]")
        if not np.allclose(positions[Joint.PELVIS], 0.0, atol=1e-6):
            raise ValueError("pelvis joint position must be the origin")

        object.__setattr__(self, "joint_positions", positions)
        object.__setattr__(self, "joint_confidence", confidence)
        object.__setattr__(self, "joint_validity", validity)
        object.__setattr__(self, "root_position", self._optional_vector(self.root_position, 3))
        object.__setattr__(self, "root_rotation", self._optional_vector(self.root_rotation, 4))
        if self.root_rotation is not None:
            _validate_quaternions(self.root_rotation)

        if self.joint_rotations is not None:
            rotations = _readonly_array(
                self.joint_rotations,
                name="joint_rotations",
                shape=(JOINT_COUNT, 4),
                dtype=np.dtype(np.float32),
            )
            if not np.isfinite(rotations).all():
                raise ValueError("joint_rotations must contain only finite values")
            _validate_quaternions(rotations)
            object.__setattr__(self, "joint_rotations", rotations)

        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))

    @property
    def is_valid(self) -> bool:
        """Strict full-body benchmark policy: all canonical joints must be valid."""
        return bool(np.all(self.joint_validity) and np.all(self.joint_confidence > 0))

    @staticmethod
    def _optional_vector(value: object | None, length: int) -> FloatArray | None:
        if value is None:
            return None
        vector = _readonly_array(
            value,
            name="optional transform",
            shape=(length,),
            dtype=np.dtype(np.float32),
        )
        if not np.isfinite(vector).all():
            raise ValueError("optional transforms must contain only finite values")
        return vector


def camera_to_unity(positions: object) -> FloatArray:
    """Convert camera axes (+Y down) to Unity axes (+Y up)."""
    converted = np.asarray(positions, dtype=np.float32).copy()
    if converted.ndim == 0 or converted.shape[-1] != 3:
        raise ValueError("positions must end with an XYZ dimension")
    converted[..., 1] *= -1.0
    return converted


def unity_to_camera(positions: object) -> FloatArray:
    """Convert Unity axes to canonical camera axes (the transform is involutive)."""
    return camera_to_unity(positions)
