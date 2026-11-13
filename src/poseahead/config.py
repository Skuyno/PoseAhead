"""Strict JSON live configuration; paths are interpreted by the invoking command."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from poseahead.perception.camera_geometry import CameraIntrinsics
from poseahead.perception.precision import CROP_PRECISIONS


@dataclass(frozen=True)
class LiveConfig:
    camera_index: int = 0
    confidence_threshold: float = 0.3
    num_aug: int = 1
    internal_batch_size: int = 8
    detector_interval: int = 10
    max_input_side: int | None = 640
    default_fov_degrees: float = 55.0
    udp_host: str = "127.0.0.1"
    udp_port: int = 5005
    camera_intrinsics: CameraIntrinsics | None = None
    crop_precision: str = "autocast"

    def __post_init__(self) -> None:
        if not isinstance(self.crop_precision, str) or self.crop_precision not in CROP_PRECISIONS:
            raise ValueError("crop_precision must be autocast or fp16_conv")
        if isinstance(self.camera_intrinsics, dict):
            try:
                object.__setattr__(
                    self, "camera_intrinsics", CameraIntrinsics(**self.camera_intrinsics)
                )
            except TypeError as error:
                raise ValueError(f"invalid camera_intrinsics config: {error}") from error
        if self.camera_intrinsics is not None and not isinstance(
            self.camera_intrinsics, CameraIntrinsics
        ):
            raise ValueError("camera_intrinsics must be an object or null")
        for name in (
            "camera_index",
            "num_aug",
            "internal_batch_size",
            "detector_interval",
            "udp_port",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
        if (
            self.camera_index < 0
            or self.num_aug < 1
            or self.internal_batch_size < self.num_aug
            or self.detector_interval < 1
            or not 1 <= self.udp_port <= 65535
        ):
            raise ValueError("invalid camera, augmentation, batch, or UDP setting")
        if self.max_input_side is not None and (
            type(self.max_input_side) is not int or self.max_input_side < 64
        ):
            raise ValueError("max_input_side must be null or an integer of at least 64")
        if (
            type(self.confidence_threshold) not in (float, int)
            or not math.isfinite(self.confidence_threshold)
            or not 0 < self.confidence_threshold <= 1
        ):
            raise ValueError("confidence_threshold must be in (0, 1]")
        if (
            type(self.default_fov_degrees) not in (float, int)
            or not math.isfinite(self.default_fov_degrees)
            or not 1 < self.default_fov_degrees < 179
        ):
            raise ValueError("default_fov_degrees must be in (1, 179)")
        if not isinstance(self.udp_host, str) or not self.udp_host.strip():
            raise ValueError("udp_host must be a nonempty string")

    def to_dict(self) -> dict:
        return asdict(self)


def load_live_config(path: str | Path) -> LiveConfig:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("live configuration must be a JSON object")
    try:
        return LiveConfig(**values)
    except TypeError as error:
        raise ValueError(f"invalid live configuration fields: {error}") from error
