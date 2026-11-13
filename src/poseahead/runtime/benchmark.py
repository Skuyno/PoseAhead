"""Shared benchmark accounting; numerical results never imply a live acceptance pass."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from poseahead.core import PoseFrame


def latency_summary(values: list[float]) -> dict:
    if not values:
        return {"samples": 0, "median": None, "p95": None, "max": None}
    if not np.isfinite(values).all() or np.any(np.asarray(values) < 0):
        raise ValueError("latency samples must be finite and non-negative")
    return {
        "samples": len(values),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
    }


@dataclass
class BenchmarkStats:
    valid_frames: int = 0
    missing_frames: int = 0
    partial_frames: int = 0
    valid_pose_with_root_frames: int = 0
    estimated_root_frames: int = 0
    latency_ms: list[float] = field(default_factory=list)
    valid_latency_ms: list[float] = field(default_factory=list)
    send_latency_ms: list[float] = field(default_factory=list)

    def observe(self, pose: PoseFrame | None, elapsed_ms: float) -> bool:
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise ValueError("observation latency must be finite and non-negative")
        self.latency_ms.append(elapsed_ms)
        if pose is None:
            self.missing_frames += 1
            return False
        if not pose.is_valid:
            self.partial_frames += 1
            return False
        self.valid_frames += 1
        if pose.root_position is not None:
            self.valid_pose_with_root_frames += 1
            if pose.metadata.get("root_translation_status") == "estimated_monocular":
                self.estimated_root_frames += 1
        self.valid_latency_ms.append(elapsed_ms)
        return True

    def report(self, elapsed_s: float) -> dict:
        if not math.isfinite(elapsed_s) or elapsed_s <= 0:
            raise ValueError("benchmark duration must be finite and positive")
        return {
            "duration_s": elapsed_s,
            "processed_frames": len(self.latency_ms),
            "valid_pose_frames": self.valid_frames,
            "valid_pose_with_root_frames": self.valid_pose_with_root_frames,
            "estimated_root_frames": self.estimated_root_frames,
            "missing_pose_frames": self.missing_frames,
            "partial_or_invalid_pose_frames": self.partial_frames,
            "valid_pose_fps": self.valid_frames / elapsed_s,
            "processed_fps": len(self.latency_ms) / elapsed_s,
            "validity_policy": "all_17_joints_valid_and_positive_confidence",
            "all_results_latency_ms": latency_summary(self.latency_ms),
            "valid_pose_latency_ms": latency_summary(self.valid_latency_ms),
            "capture_available_to_send_complete_ms": latency_summary(self.send_latency_ms),
            "acceptance_status": "not_assessed",
        }
