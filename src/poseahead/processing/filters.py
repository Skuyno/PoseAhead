"""Timestamp-based position filters; orientation estimation remains separate."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Protocol

import numpy as np

from poseahead.core import PoseFrame


class PoseProcessor(Protocol):
    method: str

    def reset(self) -> None: ...
    def process(self, frame: PoseFrame) -> PoseFrame: ...


class RawProcessor:
    method = "raw"

    def reset(self) -> None:
        self._previous: PoseFrame | None = None

    def __init__(self) -> None:
        self.reset()

    def _validate(self, frame: PoseFrame) -> None:
        previous = self._previous
        if previous is not None:
            if frame.frame_id <= previous.frame_id:
                raise ValueError("processor requires increasing frame IDs; reset for a new stream")
            if frame.source_timestamp_ns <= previous.source_timestamp_ns:
                raise ValueError("processor requires increasing source timestamps")
            if frame.produced_timestamp_ns < previous.produced_timestamp_ns:
                raise ValueError("processor requires nondecreasing produced timestamps")
            if frame.metadata.get("clock_domain") != previous.metadata.get("clock_domain"):
                raise ValueError("processor clock changed; reset for a new stream")

    def process(self, frame: PoseFrame) -> PoseFrame:
        self._validate(frame)
        self._previous = frame
        return frame


class EmaProcessor(RawProcessor):
    method = "ema"

    def __init__(self, *, time_constant_ms: float = 50.0, reset_gap_ms: float = 250.0):
        for name, value in (("time_constant_ms", time_constant_ms), ("reset_gap_ms", reset_gap_ms)):
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.time_constant_ms = float(time_constant_ms)
        self.reset_gap_ms = float(reset_gap_ms)
        super().__init__()

    def process(self, frame: PoseFrame) -> PoseFrame:
        self._validate(frame)
        previous = self._previous
        positions = frame.joint_positions.astype(np.float64)
        root = frame.root_position
        current_valid = frame.joint_validity & (frame.joint_confidence > 0)
        positions[~current_valid] = 0
        if previous is not None:
            elapsed_ms = (frame.source_timestamp_ns - previous.source_timestamp_ns) / 1e6
            if elapsed_ms <= self.reset_gap_ms:
                alpha = -math.expm1(-elapsed_ms / self.time_constant_ms)
                both_valid = current_valid & previous.joint_validity & (
                    previous.joint_confidence > 0
                )
                positions[both_valid] = (
                    alpha * positions[both_valid]
                    + (1 - alpha) * previous.joint_positions[both_valid]
                )
                if root is not None and previous.root_position is not None:
                    root = alpha * root.astype(np.float64) + (1 - alpha) * previous.root_position
        result = replace(
            frame, joint_positions=positions, root_position=root,
            root_rotation=None, joint_rotations=None,
            metadata={**frame.metadata, "processor": "ema",
                      "time_constant_ms": self.time_constant_ms, "reset_gap_ms": self.reset_gap_ms,
                      "processing_time_semantics": "input_availability_compute_cost_unmodelled"},
        )
        self._previous = result
        return result
