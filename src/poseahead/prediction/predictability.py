"""Causal adaptation signals from observations and already-matured predictions."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from poseahead.processing.filters import RawProcessor

HORIZONS_MS = (33, 66, 100, 150)


@dataclass(frozen=True, slots=True)
class PredictabilityFeatures:
    source_timestamp_ns: int
    status: str
    minimum_confidence: float
    speed_m_s: float | None
    acceleration_m_s2: float | None
    recent_error_m: tuple[float | None, ...]
    matured_counts: tuple[int, ...]
    pending_predictions: int
    discarded_predictions: int

    def to_dict(self):
        return {
            "source_timestamp_ns": self.source_timestamp_ns, "status": self.status,
            "minimum_confidence": self.minimum_confidence, "speed_m_s": self.speed_m_s,
            "acceleration_m_s2": self.acceleration_m_s2,
            "recent_observation_error_m": dict(zip(
                map(str, HORIZONS_MS), self.recent_error_m, strict=True)),
            "matured_counts": dict(zip(map(str, HORIZONS_MS), self.matured_counts, strict=True)),
            "pending_predictions": self.pending_predictions,
            "discarded_predictions": self.discarded_predictions,
            "scope": "nonpelvis_relative_joints_observations_not_ground_truth_accuracy",
        }


class CausalPredictabilityTracker:
    """Observe first, then record current-anchor predictions for later scoring.

    No policy thresholds, model choice, probability or live promotion are implied.
    A prediction is scored against interpolation of two *arrived* observations.
    Missing/low-confidence observations and long gaps invalidate that support.
    """

    def __init__(self, *, max_gap_ms=100, minimum_confidence=0.,
                 error_history=8, pending_per_horizon=64, dynamics_window_ms=0):
        if (type(dynamics_window_ms) is not int
                or dynamics_window_ms != 0 and not 20 <= dynamics_window_ms <= 1000):
            raise ValueError("dynamics_window_ms must be zero or an integer in [20, 1000]")
        self.dynamics_window_ns = dynamics_window_ms * 1_000_000
        if type(max_gap_ms) is not int or not 1 <= max_gap_ms <= 1000:
            raise ValueError("max_gap_ms must be an integer in [1, 1000]")
        if (type(minimum_confidence) not in (int, float)
                or not np.isfinite(minimum_confidence) or not 0 <= minimum_confidence <= 1):
            raise ValueError("minimum_confidence must be finite in [0, 1]")
        if type(error_history) is not int or not 1 <= error_history <= 256:
            raise ValueError("error_history must be in [1, 256]")
        if type(pending_per_horizon) is not int or not 1 <= pending_per_horizon <= 4096:
            raise ValueError("pending_per_horizon must be in [1, 4096]")
        self.max_gap_ns = max_gap_ms * 1_000_000
        self.minimum_confidence = minimum_confidence
        self.error_history, self.capacity = error_history, pending_per_horizon
        self._order = RawProcessor()
        self.reset()

    def reset(self):
        self._order.reset()
        self._clear_support()
        self._discarded = 0

    def _clear_support(self):
        if hasattr(self, "_pending"):
            self._discarded += sum(len(q) for q in self._pending.values())
        self._pending = {h: deque() for h in HORIZONS_MS}
        self._errors = {h: deque(maxlen=self.error_history) for h in HORIZONS_MS}
        self._counts = {h: 0 for h in HORIZONS_MS}
        self._recorded_anchor = {h: None for h in HORIZONS_MS}
        self._previous = self._velocity = self._interval_ns = None
        self._dynamics_history = deque(maxlen=256)

    def _window_dynamics(self, frame):
        history = self._dynamics_history
        history.append((frame.source_timestamp_ns, frame.joint_positions[1:]))
        cutoff = frame.source_timestamp_ns - self.dynamics_window_ns
        # Retain one arrived sample bracketing the trailing cutoff, never future data.
        while len(history) > 1 and history[1][0] <= cutoff:
            history.popleft()
        if len(history) < 3 or history[0][0] > cutoff:
            return None, None
        times = np.array([(t - frame.source_timestamp_ns) / self.dynamics_window_ns
                          for t, _ in history])
        design = np.stack((times**2, times, np.ones_like(times)), axis=1)
        positions = np.stack([p for _, p in history]).astype(np.float64).reshape(len(history), -1)
        coefficients, _, rank, _ = np.linalg.lstsq(
            design, positions - positions[-1], rcond=None,
        )
        if rank != 3:
            return None, None
        seconds = self.dynamics_window_ns / 1e9
        velocity = coefficients[1].reshape(16, 3) / seconds
        acceleration = 2 * coefficients[0].reshape(16, 3) / seconds**2
        return (float(np.linalg.norm(velocity, axis=1).mean()),
                float(np.linalg.norm(acceleration, axis=1).mean()))

    def observe(self, frame):
        target = frame.metadata.get("target_timestamp_ns", frame.source_timestamp_ns)
        offset = frame.metadata.get("target_offset_ns", 0)
        if (type(target) is not int or target != frame.source_timestamp_ns
                or type(offset) is not int or offset != 0):
            raise ValueError("predictability features require observations, not future predictions")
        self._order.process(frame)  # Shared public clock/order validator.
        confidence = float(frame.joint_confidence.min())
        usable = bool((frame.joint_validity & (frame.joint_confidence > 0)
                       & (frame.joint_confidence >= self.minimum_confidence)).all())
        status, speed, acceleration = "warmup", None, None
        if not usable:
            self._clear_support()
            status = "invalid_or_low_confidence"
        else:
            previous = self._previous
            if previous is not None:
                interval = frame.source_timestamp_ns - previous.source_timestamp_ns
                if interval > self.max_gap_ns:
                    self._clear_support()
                    previous = None
                    status = "tracking_gap"
            if previous is not None:
                interval = frame.source_timestamp_ns - previous.source_timestamp_ns
                velocity = (frame.joint_positions[1:].astype(np.float64)
                            - previous.joint_positions[1:]) / (interval / 1e9)
                speed = float(np.linalg.norm(velocity, axis=1).mean())
                if self._velocity is not None:
                    dt = (interval + self._interval_ns) / 2e9
                    acceleration = float(np.linalg.norm(
                        (velocity - self._velocity) / dt, axis=1,
                    ).mean())
                    status = "ready"
                self._velocity, self._interval_ns = velocity, interval
                for horizon, pending in self._pending.items():
                    while pending and pending[0][0] <= frame.source_timestamp_ns:
                        target, positions = pending.popleft()
                        if target < previous.source_timestamp_ns:
                            self._discarded += 1
                            continue
                        alpha = (target - previous.source_timestamp_ns) / interval
                        observed = ((1 - alpha) * previous.joint_positions[1:].astype(np.float64)
                                    + alpha * frame.joint_positions[1:])
                        error = float(np.linalg.norm(positions[1:] - observed, axis=1).mean())
                        self._errors[horizon].append(error)
                        self._counts[horizon] += 1
            if self.dynamics_window_ns:
                speed, acceleration = self._window_dynamics(frame)
                if acceleration is not None:
                    status = "ready"
                elif status != "tracking_gap":
                    status = "warmup"
            self._previous = frame
        return PredictabilityFeatures(
            frame.source_timestamp_ns, status, confidence, speed, acceleration,
            tuple(float(np.mean(self._errors[h])) if self._errors[h] else None
                  for h in HORIZONS_MS),
            tuple(self._counts[h] for h in HORIZONS_MS),
            sum(len(q) for q in self._pending.values()), self._discarded,
        )

    def record_prediction(self, frame):
        previous = self._previous
        if (previous is None or frame.frame_id != previous.frame_id
                or frame.source_timestamp_ns != previous.source_timestamp_ns
                or frame.metadata.get("clock_domain") != previous.metadata.get("clock_domain")):
            raise ValueError("prediction must use the latest usable observation anchor/clock")
        offset = frame.metadata.get("target_offset_ns")
        target = frame.metadata.get("target_timestamp_ns")
        if (type(offset) is not int or offset not in [h * 1_000_000 for h in HORIZONS_MS]
                or type(target) is not int or target != frame.source_timestamp_ns + offset
                or target >= 2**64):
            raise ValueError("prediction needs an explicit supported future target timestamp")
        if not (frame.joint_validity & (frame.joint_confidence > 0)).all():
            raise ValueError("cannot register an invalid prediction")
        horizon = offset // 1_000_000
        if self._recorded_anchor[horizon] == frame.frame_id:
            raise ValueError("prediction already recorded for this anchor/horizon")
        pending = self._pending[horizon]
        if len(pending) == self.capacity:
            pending.popleft()
            self._discarded += 1
        pending.append((target, frame.joint_positions.astype(np.float64, copy=True)))
        self._recorded_anchor[horizon] = frame.frame_id
