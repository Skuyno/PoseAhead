"""Fixed-grid learned prediction with explicit causal history and raw fallback."""

from collections import deque
from dataclasses import replace
from math import isfinite
from pathlib import Path

import numpy as np

from poseahead.artifacts import sha256_file
from poseahead.core.history import sample_history
from poseahead.prediction.checkpoint import FixedHorizonPredictor
from poseahead.processing.filters import EmaProcessor, RawProcessor


class LearnedProcessor(RawProcessor):
    method = "learned"

    def __init__(self, *, checkpoint, checkpoint_sha256=None, history_sampling="exact",
                 max_observation_gap_ms=100, max_history_observations=4096,
                 history_prefilter_ms=0):
        if (type(history_prefilter_ms) not in (int, float)
                or not isfinite(history_prefilter_ms) or not 0 <= history_prefilter_ms <= 1000):
            raise ValueError("history_prefilter_ms must be finite in [0, 1000]")
        if history_sampling not in ("exact", "linear_past"):
            raise ValueError("history_sampling must be exact or linear_past")
        if type(max_observation_gap_ms) is not int or not 1 <= max_observation_gap_ms <= 1000:
            raise ValueError("max_observation_gap_ms must be an integer in [1, 1000]")
        if type(max_history_observations) is not int or not 2 <= max_history_observations <= 65536:
            raise ValueError("max_history_observations must be in [2, 65536]")
        if not isinstance(checkpoint, (str, Path)) or not str(checkpoint).strip():
            raise ValueError("checkpoint must be a directory path")
        path = Path(checkpoint).resolve()
        digest = sha256_file(path / "manifest.json")
        if checkpoint_sha256 is not None and checkpoint_sha256 != digest:
            raise ValueError("checkpoint manifest checksum mismatch")
        self.predictor = FixedHorizonPredictor(path)
        if sha256_file(path / "manifest.json") != digest:
            raise ValueError("checkpoint manifest changed during loading")
        self.history_sampling = history_sampling
        self.max_gap_ns = max_observation_gap_ms * 1_000_000
        self.capacity = max_history_observations
        if self.capacity < self.predictor.history_frames:
            raise ValueError("history capacity is smaller than trained history")
        self.parameters = {"checkpoint": str(path), "checkpoint_sha256": digest,
                           "history_sampling": history_sampling,
                           "max_observation_gap_ms": max_observation_gap_ms,
                           "max_history_observations": max_history_observations}
        self.target_offset_ns = self.predictor.horizon_ms * 1_000_000
        self._prefilter = (EmaProcessor(time_constant_ms=history_prefilter_ms,
                                       reset_gap_ms=max_observation_gap_ms)
                           if history_prefilter_ms else None)
        if self._prefilter is not None:
            self.parameters["history_prefilter_ms"] = history_prefilter_ms
        super().__init__()

    def reset(self):
        super().reset()
        self._history = deque(maxlen=self.capacity)
        if self._prefilter is not None:
            self._prefilter.reset()

    def _sample(self, history, grid):
        return sample_history(history, grid, self.max_gap_ns)

    def process(self, frame):
        self._validate(frame)
        if ("target_timestamp_ns" in frame.metadata
                and frame.metadata["target_timestamp_ns"] != frame.source_timestamp_ns):
            raise ValueError("learned processor requires observations, not prior predictions")
        history = list(self._history)
        status = "warmup"
        sampled = None
        grid = tuple(frame.source_timestamp_ns - (self.predictor.history_frames - 1 - i)
                     * self.predictor.step_ns for i in range(self.predictor.history_frames))
        if not (frame.joint_validity & (frame.joint_confidence > 0)).all():
            history = []
            status = "invalid_observation"
            if self._prefilter is not None:
                self._prefilter.reset()
        else:
            interval = (frame.source_timestamp_ns - history[-1].source_timestamp_ns
                        if history else None)
            if interval is not None and (
                (self.history_sampling == "exact" and interval != self.predictor.step_ns)
                or (self.history_sampling == "linear_past" and interval > self.max_gap_ns)
            ):
                history = []
                status = ("irregular_interval" if self.history_sampling == "exact"
                          else "tracking_gap")
            history_frame = frame
            if self._prefilter is not None:
                if not history:
                    self._prefilter.reset()
                filtered = self._prefilter.process(frame)
                # Only model features are filtered; root and raw fallback remain original.
                history_frame = replace(frame, joint_positions=filtered.joint_positions)
            history.append(history_frame)
            if self.history_sampling == "exact":
                history = history[-self.predictor.history_frames:]
                if len(history) == self.predictor.history_frames:
                    sampled = np.stack([f.joint_positions for f in history])
            else:
                # Retain one bracketing sample before the oldest grid point.
                while len(history) > 1 and history[1].source_timestamp_ns <= grid[0]:
                    history.pop(0)
                if len(history) > self.capacity:
                    history = history[-self.capacity:]
                    status = "history_capacity"
                sampled = self._sample(history, grid)
        metadata = {**frame.metadata, "processor": self.method,
                    "processor_parameters": self.parameters,
                    "configured_target_offset_ns": self.target_offset_ns,
                    "history_sampling": self.history_sampling,
                    "history_buffer_observations": len(history),
                    "root_prediction_policy": "hold_latest_observation",
                    "processing_time_semantics": "input_availability_compute_cost_unmodelled"}
        if sampled is not None:
            positions = self.predictor.predict(sampled, grid)
            metadata.update(prediction_status="predicted", target_offset_ns=self.target_offset_ns,
                            target_timestamp_ns=frame.source_timestamp_ns + self.target_offset_ns)
            result = replace(frame, joint_positions=positions, joint_rotations=None,
                             root_rotation=None, metadata=metadata)
        else:
            # Explicitly reset target semantics: a fallback is the observed pose,
            # not a future prediction credited to the model's configured horizon.
            metadata.update(prediction_status=status, target_offset_ns=0,
                            target_timestamp_ns=frame.source_timestamp_ns)
            result = replace(frame, metadata=metadata)
        self._history = deque(history, maxlen=self.capacity)
        self._previous = frame  # original stream used for order/clock validation
        return result
